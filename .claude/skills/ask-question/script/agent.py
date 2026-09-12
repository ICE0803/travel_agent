"""
RAG知识库智能体 RAGKnowledgeAgent
职责：基于向量数据库的知识检索与问答

核心功能：
1. 知识库构建：将商旅相关文档向量化并存储到Milvus Lite
2. 语义检索：根据用户查询检索最相关的知识片段
3. 知识问答：结合检索到的知识和LLM生成准确答案
4. 知识管理：支持添加、更新、删除知识库内容

技术栈：
- Milvus Lite: 轻量级向量数据库（本地存储）
- sentence-transformers: 文本向量化模型
- LLM: 用户配置的豆包模型用于生成答案

安装：
pip install milvus sentence-transformers
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict
import json
import logging
import os
from pathlib import Path

# Add project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

_GRPC_MAX_MS = '2147483647'  # gRPC 使用的 int32 上限，约 24.8 天
os.environ['GRPC_KEEPALIVE_TIME_MS'] = _GRPC_MAX_MS
os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '20000'
os.environ['GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = '0'
os.environ['GRPC_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS'] = _GRPC_MAX_MS
os.environ['GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS'] = _GRPC_MAX_MS

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient, DataType
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG dependencies not available: {e}")
    logger.warning("Install with: pip install pymilvus sentence-transformers")
    DEPENDENCIES_AVAILABLE = False


class RAGKnowledgeAgent(AgentBase):
    """RAG知识库智能体"""

    def __init__(
        self,
        name: str = "RAGKnowledgeAgent",
        model=None,
        knowledge_base_path: str = None,
        collection_name: str = "business_travel_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 3,
        similarity_threshold: float = None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model = model
        
        if knowledge_base_path is None:
            # Default to local data directory in skill folder
            current_dir = Path(__file__).parent.parent
            knowledge_base_path = str(current_dir / "data" / "rag_knowledge")

        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        self.top_k = top_k
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

        if similarity_threshold is None:
            try:
                from config import RAG_CONFIG
                similarity_threshold = RAG_CONFIG.get("similarity_threshold", 0.5)
            except Exception:
                similarity_threshold = 0.5
        self.similarity_threshold = similarity_threshold
        logger.info(f"相似度阈值: {self.similarity_threshold}")

        if not DEPENDENCIES_AVAILABLE:
            logger.error("RAG dependencies not installed. Install with: pip install pymilvus sentence-transformers")
            self.initialized = False
            return

        # 优先使用 config 中的配置（支持本地路径，避免连 HuggingFace）
        try:
            from config import RAG_CONFIG
            embedding_model = RAG_CONFIG.get("embedding_model", embedding_model)
        except Exception:
            pass

        # 若配置的是本地路径且存在，则从本地加载，否则按模型 ID 使用（会联网）
        model_path_or_id = embedding_model
        path_obj = Path(embedding_model).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        if path_obj.exists():
            model_path_or_id = str(path_obj.resolve())
            logger.info(f"Using local embedding model: {model_path_or_id}")
        else:
            if "/" in embedding_model or "\\" in embedding_model or embedding_model.startswith("."):
                logger.warning(
                    f"Configured embedding path does not exist: {embedding_model}，将使用 BAAI/bge-small-zh-v1.5 并尝试联网下载。"
                )
                model_path_or_id = "BAAI/bge-small-zh-v1.5"
        logger.info(f"Loading embedding model: {model_path_or_id}")
        self.embedding_model = SentenceTransformer(model_path_or_id)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 初始化 Milvus Lite（本地文件存储）
        milvus_db_path = str(self.knowledge_base_path / "milvus_lite.db")
        logger.info(f"Initializing Milvus Lite at: {milvus_db_path}")

        self.milvus_client = MilvusClient(milvus_db_path, grpc_options={"keepalive_time": _GRPC_MAX_MS, "keepalive_timeout": "20000", "keepalive_permit_without_calls": "0", "http2_min_recv_ping_interval_without_data": _GRPC_MAX_MS, "http2_min_ping_interval_without_data": _GRPC_MAX_MS})
        self._client_created_at = None  # 用于追踪客户端创建时间

        # 检查collection是否存在
        if self.milvus_client.has_collection(collection_name):
            logger.info(f"Loaded existing collection: {collection_name}")
        else:
            # 创建新collection
            logger.info(f"Creating new collection: {collection_name}")
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",  # 余弦相似度
                auto_id=False,
            )
            logger.info(f"Created new collection: {collection_name}")

        self.initialized = True
        self._milvus_db_path = milvus_db_path  # 保存路径用于重连
        logger.info("RAG Knowledge Agent (Milvus Lite) initialized successfully")

    def _ensure_connection(self):
        """确保 Milvus 连接正常，如果需要则重新创建客户端"""
        try:
            # 尝试一个轻量级操作来检查连接
            self.milvus_client.has_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Milvus connection issue detected: {e}, reconnecting...")
            try:
                # 关闭旧连接
                if hasattr(self.milvus_client, 'close'):
                    try:
                        self.milvus_client.close()
                    except:
                        pass

                # 重新创建客户端
                self.milvus_client = MilvusClient(self._milvus_db_path)
                logger.info("Milvus client reconnected successfully")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect Milvus: {reconnect_error}")
                raise

    def add_documents(self, documents: List[Dict[str, str]]) -> Dict:
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 {'content': '内容', 'metadata': {...}}

        Returns:
            添加结果统计
        """
        if not self.initialized:
            return {"status": "error", "message": "RAG Agent not initialized"}

        try:
            # 确保连接正常
            self._ensure_connection()
            # 获取当前文档总数，用于生成连续的ID
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            current_count = stats.get("row_count", 0)

            # 准备数据
            data_to_insert = []

            for i, doc in enumerate(documents):
                # Milvus 要求 id 必须是 int64
                doc_id = current_count + i + 1
                content = doc['content']
                metadata = doc.get('metadata', {})

                # 生成向量
                embedding = self.embedding_model.encode(content).tolist()

                # Milvus 数据格式
                data_to_insert.append({
                    "id": doc_id,
                    "vector": embedding,
                    "content": content,
                    "metadata": json.dumps(metadata, ensure_ascii=False)  # 将metadata转为JSON字符串
                })

            # 批量插入到 Milvus
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=data_to_insert
            )

            # 获取总数
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            total_count = stats.get("row_count", len(documents))

            logger.info(f"Successfully added {len(documents)} documents to knowledge base")
            return {
                "status": "success",
                "added_count": len(documents),
                "total_count": total_count
            }

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索知识库

        Args:
            query: 查询文本
            top_k: 返回top k个结果

        Returns:
            检索结果列表
        """
        if not self.initialized:
            return []

        try:
            # 确保连接正常
            self._ensure_connection()
            # pymilvus 3.x 需先加载 collection 才能检索，否则报 "in state released"
            self.milvus_client.load_collection(self.collection_name)
            k = top_k or self.top_k

            # 生成查询向量
            query_embedding = self.embedding_model.encode(query).tolist()

            # 在 Milvus 中检索
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["id", "content", "metadata"]
            )

            # 格式化结果
            retrieved_docs = []
            if results and len(results) > 0:
                for hit in results[0]:
                    # 解析metadata
                    metadata_str = hit.get("entity", {}).get("metadata", "{}")
                    try:
                        metadata = json.loads(metadata_str)
                    except:
                        metadata = {}

                    retrieved_docs.append({
                        'id': hit.get("entity", {}).get("id", ""),
                        'content': hit.get("entity", {}).get("content", ""),
                        'metadata': metadata,
                        'distance': hit.get("distance", 0.0)
                    })


            if self.similarity_threshold is not None and retrieved_docs:
                raw_count = len(retrieved_docs)
                top_score = max(float(d.get("distance", 0.0)) for d in retrieved_docs)
                kept = [
                    d for d in retrieved_docs
                    if float(d.get("distance", 0.0)) >= self.similarity_threshold
                ]
                if len(kept) < raw_count:
                    logger.info(
                        "相似度过滤：丢弃 %d/%d 条（阈值 %.2f，本次最高分 %.4f）",
                        raw_count - len(kept), raw_count,
                        self.similarity_threshold, top_score,
                    )
                retrieved_docs = kept

            if not retrieved_docs:
                logger.info(
                    "无满足阈值的知识片段（query=%s，阈值=%.2f）",
                    query[:50], self.similarity_threshold,
                )
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        RAG问答主流程
        1. 接收用户查询
        2. 检索相关知识
        3. 结合知识生成答案
        """
        if not self.initialized:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "error",
                    "message": "RAG Agent not initialized. Please install dependencies: pip install pymilvus sentence-transformers"
                }),
                role="assistant"
            )

        if x is None:
            return Msg(name=self.name, content=json.dumps({}), role="assistant")

        # 获取用户查询
        if isinstance(x, list):
            content = x[-1].content if x else ""
        else:
            content = x.content

        # 尝试解析 JSON 输入 (来自 Orchestrator)
        user_query = content
        if isinstance(content, str) and content.strip().startswith('{'):
            try:
                import json
                data = json.loads(content)
                # 只要解析成功，就认为 content 是结构化数据，尝试提取 query
                extracted_query = ""
                if "context" in data and isinstance(data["context"], dict):
                    extracted_query = data["context"].get("rewritten_query", "")
                elif "rewritten_query" in data:
                    extracted_query = data.get("rewritten_query", "")
                
                # 使用提取到的 query（即使为空，也比 JSON 字符串好）
                user_query = extracted_query
            except:
                pass  # 解析失败则保留原字符串

        # 检索相关知识
        retrieved_docs = self.search_knowledge(user_query)

        if not retrieved_docs:
            result = {
                "status": "no_knowledge",
                "query": user_query,
                "answer": "抱歉，我在知识库中没有找到相关信息。",
                "retrieved_documents": []
            }
            return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

        # 构建知识上下文
        knowledge_context = "\n\n".join([
            f"【知识片段{i+1}】\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # 如果有LLM，使用LLM生成答案
        if self.model:
            # 动态读取 Prompt 指令 (Progressive Disclosure)
            skill_instruction = self.skill_loader.get_skill_content("ask-question")
            if not skill_instruction:
                skill_instruction = "请基于知识库中的信息回答用户的问题。"

            prompt = f"""你是一个商旅知识专家。请严格基于以下知识库中的信息回答用户的问题。

【用户问题】
{user_query}

【知识库信息】
{knowledge_context}

【任务说明】
{skill_instruction}

【重要约束】
1. 如果【知识库信息】中没有包含回答用户问题所需的信息，请直接回答“抱歉，知识库中没有找到相关信息”，不要尝试根据你自己的知识编造答案。
2. 即使问题很基础，如果知识库里没写，就说不知道。
3. 请以专业、客观的语气回答。
"""

            try:
                # 调用LLM生成答案
                messages = [
                    {"role": "system", "content": "你是一个商旅知识专家。"},
                    {"role": "user", "content": prompt}
                ]
                response = await self.model(messages)

                # 获取响应内容 - 处理异步生成器
                answer = ""
                if hasattr(response, '__aiter__'):
                    # 异步生成器，需要迭代获取内容
                    async for chunk in response:
                        if isinstance(chunk, str):
                            answer = chunk
                        elif hasattr(chunk, 'content'):
                            if isinstance(chunk.content, str):
                                answer = chunk.content
                            elif isinstance(chunk.content, list):
                                for item in chunk.content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        answer = item.get('text', '')
                elif hasattr(response, 'text'):
                    answer = response.text
                elif hasattr(response, 'content'):
                    answer = response.content
                elif isinstance(response, dict) and 'content' in response:
                    answer = response['content']
                else:
                    answer = str(response) if response else "无法生成答案"

                if not answer:
                    answer = "无法生成答案"
                
                # 清理 LLM 可能输出的 JSON 格式
                answer_str = answer.strip()
                if answer_str.startswith("{") and answer_str.endswith("}"):
                    try:
                        import json
                        json_obj = json.loads(answer_str)
                        # 如果 LLM 输出了 {"answer": "..."} 或 {"content": "..."}
                        if isinstance(json_obj, dict):
                            answer = json_obj.get("answer") or json_obj.get("content") or answer
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error generating answer with LLM: {e}")
                answer = f"知识库中找到相关信息，但生成答案时出错：{str(e)}"
        else:
            # 如果没有LLM，直接返回检索到的知识
            answer = "以下是知识库中的相关信息：\n\n" + knowledge_context

        result = {
            "status": "success",
            "query": user_query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc['metadata']
                }
                for doc in retrieved_docs
            ]
        }

        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        if not self.initialized:
            return {"status": "error", "message": "Not initialized"}

        try:
            # 确保连接正常
            self._ensure_connection()
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            return {
                "status": "success",
                "collection_name": self.collection_name,
                "total_documents": stats.get("row_count", 0),
                "knowledge_base_path": str(self.knowledge_base_path)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        """关闭 Milvus 连接"""
        if hasattr(self, 'milvus_client'):
            try:
                if hasattr(self.milvus_client, 'close'):
                    self.milvus_client.close()
                    logger.info("Milvus client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Milvus client: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()
