"""
初始化RAG知识库 (Plugin Version)
从 .claude/skills/ask-question/data/documents 目录加载商旅相关文档并导入到向量数据库中
"""
import sys
import os
import importlib.util
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径 (假设脚本在 .claude/skills/ask-question/script/)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import LLM_CONFIG
from agentscope.model import OpenAIChatModel

# 动态加载同目录下的 agent.py
def load_rag_agent_class():
    agent_script = current_dir / "agent.py"
    spec = importlib.util.spec_from_file_location("RAGKnowledgeAgentModule", agent_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["RAGKnowledgeAgentModule"] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent

RAGKnowledgeAgent = load_rag_agent_class()

def split_text(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    """
    简单的文本切分：优先按段落切分，控制每块大小
    """
    chunks = []
    
    # 预处理：按空行分割成段落
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        if line.strip() == "":
            if current_para:
                paragraphs.append("\n".join(current_para))
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append("\n".join(current_para))
    
    # 组合段落
    current_chunk = ""
    
    for para in paragraphs:
        # 如果加上当前段落还未超限
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += "\n\n" + para
        else:
            # 已经超限，先保存当前 chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果单个段落非常长，强制切分
            if len(para) > max_chars:
                # 这里的逻辑简单处理：直接把长段落作为新起点（可能会再次被切分，如果这里加递归太复杂，
                # 简单起见，如果段落超长，就按长度硬切）
                remaining = para
                while len(remaining) > max_chars:
                    chunks.append(remaining[:max_chars])
                    remaining = remaining[max_chars - overlap:]
                current_chunk = remaining
            else:
                # 开启新 chunk，并带上前一个 chunk 的尾部作为 overlap（如果需要）
                # 这里简单起见，不搞 overlap 了，因为是按自然段落切的
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def load_documents_from_directory(directory_path: str) -> List[Dict]:
    """
    从指定目录加载所有文档
    """
    documents = []
    doc_dir = Path(directory_path)

    if not doc_dir.exists():
        print(f"❌ 文档目录不存在: {directory_path}")
        return documents

    # 获取所有.txt文件并排序
    doc_files = sorted(doc_dir.glob("*.txt"))

    if not doc_files:
        print(f"❌ 未找到任何文档文件 (.txt)")
        return documents

    # 定义类别映射（根据文件名判断）
    category_mapping = {
        "travel_standards": "差旅规定",
        "reimbursement_policy": "报销规定",
        "booking_guide": "预订指南",
        "faq": "FAQ",
        "emergency_procedures": "应急指南",
        "platform_guide": "平台指南",
        "city_specific_tips": "城市指南",
        "environmental_initiatives": "环保倡议"
    }

    total_chunks = 0

    for file_path in doc_files:
        try:
            # 从文件名提取编号作为doc_id (如: 01_travel_standards.txt -> doc_001)
            filename_parts = file_path.stem.split('_', 1)
            if len(filename_parts) >= 2:
                doc_num = filename_parts[0]
                doc_key = filename_parts[1] if len(filename_parts) > 1 else ""
            else:
                doc_num = file_path.stem
                doc_key = ""

            base_doc_id = f"doc_{doc_num}"

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                print(f"   ⚠️  跳过空文件: {file_path.name}")
                continue

            # 提取标题（第一行）
            lines = content.split('\n')
            title = lines[0].strip() if lines else file_path.stem

            # 根据文件名确定类别
            category = "商旅知识"
            for key, cat in category_mapping.items():
                if key in doc_key:
                    category = cat
                    break

            # --- 文档切分逻辑 ---
            chunks = split_text(content, max_chars=600, overlap=100)
            
            for i, chunk_content in enumerate(chunks):
                doc_id = f"{base_doc_id}_{i+1}"
                
                # 构建文档对象
                document = {
                    "id": doc_id,
                    "content": chunk_content,
                    "metadata": {
                        "category": category,
                        "title": f"{title} (Part {i+1})",
                        "source": "商旅知识库文档",
                        "file_path": str(file_path),
                        "version": "2024版",
                        "parent_doc": file_path.name
                    }
                }
                documents.append(document)
            
            total_chunks += len(chunks)
            print(f"   ✓ 加载文档: {file_path.name} -> {len(chunks)} chunks")

        except Exception as e:
            print(f"   ❌ 加载文件失败 {file_path.name}: {e}")
            continue

    return documents


def main():
    print("="*70)
    print("初始化RAG知识库 (Plugin Version) - With Chunking")
    print("="*70)
    print()

    rag_agent = None
    try:
        # 创建模型
        print("1. 创建模型...")
        model = OpenAIChatModel(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            client_kwargs={
                "base_url": LLM_CONFIG["base_url"],
            },
            temperature=LLM_CONFIG.get("temperature", 0.7),
            max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        )
        print("✓ 模型创建成功")
        print()

        # 定义路径
        skill_root = current_dir.parent
        knowledge_base_path = skill_root / "data" / "rag_knowledge"
        documents_dir = skill_root / "data" / "documents"

        # 确保目录存在
        knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # 创建RAG Agent
        print("2. 初始化RAG Agent...")
        print(f"   知识库路径: {knowledge_base_path}")
        rag_agent = RAGKnowledgeAgent(
            name="RAGKnowledgeAgent",
            model=model,
            knowledge_base_path=str(knowledge_base_path),
            collection_name="business_travel_knowledge",
            top_k=3
        )

        if not rag_agent.initialized:
            print("❌ RAG Agent初始化失败")
            return

        print("✓ RAG Agent初始化成功")
        print()

        # 从文件加载文档
        print(f"3. 从 {documents_dir} 加载文档...")
        documents = load_documents_from_directory(str(documents_dir))

        if not documents:
            print("❌ 未加载到任何文档")
            return

        print(f"✓ 成功切分并加载 {len(documents)} 个片段")
        print()

        # 添加文档到RAG知识库
        print("4. 将文档添加到RAG知识库...")
        
        # 在添加之前，先清空旧的 collection（如果只是追加的话，ID会冲突，这里我们假设是从头开始）
        # RAGKnowledgeAgent 目前的实现是直接 insert。
        # 由于我们之前已经运行过一次，且 ID 策略变了（doc_001 -> doc_001_1），这可能会导致混合。
        # 最好的办法是 drop collection。
        if rag_agent.milvus_client.has_collection(rag_agent.collection_name):
            print("   ⚠️  检测到已存在 Collection，正在删除重建以避免数据污染...")
            rag_agent.milvus_client.drop_collection(rag_agent.collection_name)
            # 重新创建
            rag_agent.milvus_client.create_collection(
                collection_name=rag_agent.collection_name,
                dimension=rag_agent.embedding_dim,
                metric_type="COSINE",
                auto_id=False,
            )
            print("   ✓ Collection 重建完成")

        result = rag_agent.add_documents(documents)

        if result["status"] == "success":
            print(f"✓ 成功添加 {result['added_count']} 个片段")
            print(f"✓ 知识库总文档数: {result['total_count']}")
        else:
            print(f"❌ 添加文档失败: {result.get('message', 'Unknown error')}")
            return

        print()

        # 获取统计信息
        print("5. 知识库统计信息:")
        stats = rag_agent.get_stats()
        if stats["status"] == "success":
            print(f"   - Collection: {stats.get('collection_name')}")
            print(f"   - 文档数量: {stats.get('total_documents')}")
            print(f"   - 存储路径: {stats.get('knowledge_base_path')}")
        print()

        # 测试检索
        print("6. 测试知识检索...")
        test_queries = [
            "出差住宿标准是多少？",
            "航班延误了怎么办？",
            "机票应该提前多久预订？"
        ]

        for query in test_queries:
            print(f"\n   查询: {query}")
            results = rag_agent.search_knowledge(query, top_k=2)
            if results:
                print(f"   ✓ 找到 {len(results)} 个相关文档")
                for i, doc in enumerate(results, 1):
                    # 安全获取 metadata
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    title = metadata.get('title', 'Unknown')
                    distance = doc.get('distance', 0.0)
                    print(f"      [{i}] {title} (相似度: {1-distance:.3f})")
            else:
                print("   ❌ 未找到相关文档")

        print()
        print("="*70)
        print("知识库初始化完成！")
        print("="*70)

    finally:
        # 确保资源被正确清理
        if rag_agent:
            print("\n正在清理资源...")
            try:
                rag_agent.close()
            except:
                pass
            print("✓ 资源清理完成")


if __name__ == "__main__":
    main()
