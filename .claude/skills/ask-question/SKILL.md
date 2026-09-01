---
name: ask-question
description: Use this skill when the user asks questions about travel policies, reimbursement, booking guides, city information, or any travel-related questions. Triggers when user asks "XX标准是多少", "如何XX", "XX怎么办", or any question format. This skill uses RAGKnowledgeAgent to retrieve answers from the knowledge base.
---

# Ask Travel Question (RAG 知识库问答)

回答用户关于差旅政策、报销、预订、城市指南等的问题，使用 **RAGKnowledgeAgent** 从本地知识库检索并生成答案。

## When to Use

- 用户问「XX标准是多少」「如何报销」「航班延误怎么办」等
- 需要基于企业/项目知识文档回答时

## Agent

- **RAGKnowledgeAgent** (`agents/rag_knowledge_agent.py`)
- 所有子 Agent 均使用 **model 对象**（非 model_config_name），需先创建 `OpenAIChatModel`
- **异步**：`reply()` 为 `async`，需 `await`

## 初始化与调用

```python
import asyncio
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from config import LLM_CONFIG
from agents.rag_knowledge_agent import RAGKnowledgeAgent
import json

async def ask_question(user_query: str):
    init_agentscope()
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": 60},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    # 嵌入模型路径从 config.RAG_CONFIG 读取，默认 data/models/bge-small-zh-v1.5
    rag_agent = RAGKnowledgeAgent(
        name="RAGKnowledgeAgent",
        model=model,
        knowledge_base_path="./data/rag_knowledge",
        collection_name="business_travel_knowledge",
        top_k=3,
    )
    if not getattr(rag_agent, "initialized", True):
        return {"error": "RAG 未初始化，请先运行 python scripts/init_knowledge_base.py"}
    user_msg = Msg(name="user", content=user_query, role="user")
    result = await rag_agent.reply(user_msg)
    return json.loads(result.content) if isinstance(result.content, str) else result.content

# 使用
data = asyncio.run(ask_question("北京的住宿标准是多少？"))
# data: {"status": "success"|"no_knowledge", "answer": "...", "retrieved_documents": [...], "query": "..."}
```

## 返回格式

- `status`: `"success"` 或 `"no_knowledge"`
- `answer`: 自然语言答案
- `retrieved_documents`: 列表，每项含 `content`, `metadata`
- `query`: 用户问题

## 知识库

- 路径：`data/rag_knowledge/`（Milvus Lite）
- 源文档：`data/documents/`，共 8 类（差旅标准、报销、预订、FAQ、紧急处理、平台指南、城市指南、环保）
- 首次使用前需执行：`python scripts/init_knowledge_base.py`


## 回答生成指南

【回答要求】
1. 必须严格基于知识库中的信息进行回答，严禁编造。
2. 如果检索到的知识库信息与问题无关，或者信息不足以回答问题，请直接回答“知识库中没有相关信息”。
3. 回答要准确、简洁、有条理。
4. 如果有多个相关信息，可以分点说明。

请直接给出答案。
