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

# 混合检索（BM25 + RRF）。导入失败时自动退回纯向量检索，不影响主流程。
try:
    from utils.hybrid_retriever import BM25Index, reciprocal_rank_fusion
    HYBRID_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    BM25Index = None
    reciprocal_rank_fusion = None
    HYBRID_AVAILABLE = False
    logger.warning(f"混合检索模块不可用，将退回纯向量检索: {e}")


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

        # ---- 混合检索：把全部 chunk 载入内存构建 BM25 索引 ----
        self._bm25 = None
        self._corpus: Dict[object, Dict] = {}      # id -> {content, metadata}
        self._hybrid_cfg: Dict = {}
        try:
            from config import RAG_CONFIG as _RAG_CFG
            self._hybrid_cfg = (_RAG_CFG.get("hybrid") or {})
        except Exception as e:
            logger.warning(f"读取 RAG_CONFIG['hybrid'] 失败，使用默认值: {e}")
            self._hybrid_cfg = {}

        if self._hybrid_cfg.get("enabled") and HYBRID_AVAILABLE:
            try:
                self._build_bm25_index()
            except Exception as e:
                # 索引构建失败不应影响 Agent 可用性，退回纯向量检索
                logger.warning(f"BM25 索引构建失败，本次退回纯向量检索: {e}")
                self._bm25 = None
        elif self._hybrid_cfg.get("enabled") and not HYBRID_AVAILABLE:
            logger.warning("配置要求混合检索，但 hybrid_retriever 不可用")
        else:
            logger.info("混合检索未启用（hybrid.enabled=False），使用纯向量检索")

        self.initialized = True
        self._milvus_db_path = milvus_db_path  # 保存路径用于重连
        logger.info("RAG Knowledge Agent (Milvus Lite) initialized successfully")

    def _build_bm25_index(self):
        """
        从 Milvus 取出全部 chunk，构建内存 BM25 索引。

        注意：知识库重新灌数据后需重建（重启进程即可）。
        """
        self._ensure_connection()
        # pymilvus 3.x：query/get 前必须先 load，否则报 "in state released"
        self.milvus_client.load_collection(self.collection_name)
        rows = self.milvus_client.query(
            collection_name=self.collection_name,
            filter="id >= 0",          # milvus-lite 需要一个表达式才能全量取
            output_fields=["id", "content", "metadata"],
            limit=16384,
        )

        docs = []
        self._corpus = {}
        for r in rows:
            cid = r.get("id")
            content = r.get("content") or ""
            raw_meta = r.get("metadata") or "{}"
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except Exception:
                meta = {}
            self._corpus[cid] = {"content": content, "metadata": meta}
            docs.append((cid, content))

        if not docs:
            raise RuntimeError("知识库为空，无法构建 BM25 索引")

        cfg = self._hybrid_cfg or {}
        self._bm25 = BM25Index(
            docs,
            k1=float(cfg.get("bm25_k1", 1.5)),
            b=float(cfg.get("bm25_b", 0.75)),
        )
        st = self._bm25.stats()
        logger.info(
            "BM25 索引构建完成：%d 个 chunk，平均长度 %.1f token，词表 %d，"
            "分词=%s，停用词过滤=%s",
            st["docs"], st["avgdl"], st["vocab"],
            "jieba" if st.get("jieba") else "bigram(降级)",
            st.get("drop_stopwords"),
        )

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

    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """
        纯向量检索（不含阈值过滤），返回带 distance（余弦相似度）的原始结果。
        """
        self._ensure_connection()
        # pymilvus 3.x 需先加载 collection 才能检索，否则报 "in state released"
        self.milvus_client.load_collection(self.collection_name)

        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["id", "content", "metadata"],
        )

        out: List[Dict] = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                metadata_str = entity.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except Exception:
                    metadata = {}
                out.append({
                    "id": entity.get("id", ""),
                    "content": entity.get("content", ""),
                    "metadata": metadata,
                    "distance": float(hit.get("distance", 0.0)),
                })
        return out

    def search_knowledge(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        检索知识库。

        - hybrid.enabled=True 且 BM25 索引可用时：向量路 + BM25 路 → RRF 融合
        - 否则：纯向量检索（行为与改造前完全一致）

        Args:
            query: 查询文本
            top_k: 返回top k个结果

        Returns:
            检索结果列表
        """
        if not self.initialized:
            return []

        cfg = self._hybrid_cfg or {}
        use_hybrid = bool(cfg.get("enabled")) and self._bm25 is not None

        try:
            if not use_hybrid:
                return self._search_vector_only(query, top_k)
            return self._search_hybrid(query, top_k)
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    def _search_vector_only(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """纯向量检索 + 相似度阈值过滤（改造前的原始行为）。"""
        k = top_k or self.top_k
        retrieved_docs = self._vector_search(query, k)

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

    def _search_hybrid(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        混合检索：向量路 + BM25 路 → RRF 融合。

        关键设计（容易踩坑）：
          相似度阈值 similarity_threshold 是给**余弦分数**用的，而 RRF 分数
          量级只有 1/(k+rank) ≈ 0.008~0.03，两者量纲完全不同。
          因此阈值**只作用于向量路**，BM25 单路命中走独立的 min_bm25_score 准入。
          否则把 0.5 套到 RRF 分数上会把所有结果过滤光。
        """
        cfg = self._hybrid_cfg or {}
        final_k = int(top_k or cfg.get("final_top_k") or self.top_k)
        kd = int(cfg.get("top_k_dense", 10))
        ks = int(cfg.get("top_k_sparse", 10))
        rrf_k = int(cfg.get("rrf_k", 60))
        min_bm25 = float(cfg.get("min_bm25_score", 0.0) or 0.0)
        thr = self.similarity_threshold

        # ---- 向量路（阈值只在这里生效）----
        dense = self._vector_search(query, kd)
        if thr is not None:
            dense = [d for d in dense if d["distance"] >= thr]

        # ---- BM25 路 ----
        sparse = [(cid, s) for cid, s in self._bm25.search(query, ks) if s >= min_bm25]

        logger.info(
            "混合检索：向量路 %d 条，BM25 路 %d 条（min_bm25=%.2f）",
            len(dense), len(sparse), min_bm25,
        )

        if not dense and not sparse:
            logger.info(
                "混合检索无满足阈值的知识片段（query=%s，余弦阈值=%.2f，bm25阈值=%.2f）",
                query[:50], thr if thr is not None else -1, min_bm25,
            )
            return []

        # ---- RRF 融合 ----
        dense_ranked = [(d["id"], d["distance"]) for d in dense]
        fused = reciprocal_rank_fusion([dense_ranked, sparse], k=rrf_k)

        by_id = {d["id"]: d for d in dense}
        sparse_ids = {cid for cid, _ in sparse}

        out: List[Dict] = []
        for doc_id, rrf_score in fused:
            if doc_id in by_id:
                item = dict(by_id[doc_id])
                item["matched_by"] = "vector+bm25" if doc_id in sparse_ids else "vector"
            else:
                # 仅 BM25 命中：语料里有但向量路没召回
                rec = self._corpus.get(doc_id)
                if rec is None:
                    continue
                item = {
                    "id": doc_id,
                    "content": rec["content"],
                    "metadata": rec["metadata"],
                    "distance": None,          # 无余弦分数（reply() 不使用该字段）
                    "matched_by": "bm25",
                }
            item["rrf_score"] = round(rrf_score, 6)
            out.append(item)
            if len(out) >= final_k:
                break

        logger.info(
            "混合检索返回 %d 条（来源：%s）",
            len(out), [d.get("matched_by") for d in out],
        )
        return out

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
