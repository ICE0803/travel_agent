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

- **RAGKnowledgeAgent**（`.claude/skills/ask-question/script/agent.py`）
- 所有子 Agent 均使用 **model 对象**（非 model_config_name），需先创建 `OpenAIChatModel`
- **异步**：`reply()` 为 `async`，需 `await`

## 检索方式：混合检索（BM25 + 向量 + RRF）

知识库检索**不是**单一向量检索，而是双路召回后融合：

```
用户 query
   ├─ 向量路：bge-small-zh-v1.5 编码 → Milvus COSINE 检索 → top_k_dense 条
   │            └─ similarity_threshold（默认 0.5）过滤，只作用于这一路
   └─ BM25 路：jieba 分词 → Okapi BM25 打分 → top_k_sparse 条
                └─ min_bm25_score 准入过滤
                        ↓
              RRF 融合（k=60）→ 取 final_top_k 条
```

**为什么两路**：向量路擅长语义匹配（换个说法也能召回），但对关键词精确匹配不敏感；BM25 路正好相反。两者互补。

**融合用 RRF 而非加权求和**：BM25 分数无上界、余弦相似度在 -1~1，量纲不同，直接加权需要归一化且归一化方式本身要调参。RRF 只看排名：`RRF(d) = Σ 1/(k + rank_i(d))`，天然规避量纲问题。

### ⚠️ 阈值不能混用（本技能最易踩的坑）

`similarity_threshold`（默认 0.5）是给**余弦分数**用的，而 RRF 分数只有 `1/(60+rank) ≈ 0.008~0.03` 量级。

**如果把 0.5 套到 RRF 分数上，所有结果都会被过滤光**，Agent 会永远回答「知识库中没有相关信息」。

代码里的处理方式：阈值**只作用于向量路**，BM25 单路命中走独立的 `min_bm25_score`。

### 配置

见 `config.py` → `RAG_CONFIG["hybrid"]`：

| 参数 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 设为 `false` 退回纯向量检索（A/B 对比、回归排查用） |
| `top_k_dense` / `top_k_sparse` | 10 / 10 | 两路各自的召回条数 |
| `rrf_k` | 60 | RRF 平滑常数 |
| `final_top_k` | 3 | 融合后最终返回条数 |
| `bm25_k1` / `bm25_b` | 1.5 / 0.75 | BM25 的词频饱和与长度归一化参数 |
| `min_bm25_score` | 0.5 | BM25 单路准入阈值，**需按语料调优** |

**分词**：优先 jieba（`lcut_for_search`），未安装时自动降级为字符二元组；另带停用词过滤（BM25 的 IDF 恒为正，不过滤会让「的/是/在/如何」这类虚词贡献噪声分数）。

**实现模块**：`utils/hybrid_retriever.py`（`tokenize` / `BM25Index` / `reciprocal_rank_fusion`）。

> BM25 索引在 Agent 初始化时从 Milvus 全量载入内存构建。
> **知识库重新灌数据后需重启进程**，否则索引是旧的。

## 初始化与调用

该 Agent 是插件式加载的（不在 `agents/` 包内），独立调用需用 importlib：

```python
import asyncio
import importlib.util
import json
from pathlib import Path

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from config import LLM_CONFIG

# 动态加载技能插件里的 Agent 类
_spec = importlib.util.spec_from_file_location(
    "ask_question_agent",
    Path(".claude/skills/ask-question/script/agent.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
RAGKnowledgeAgent = _module.RAGKnowledgeAgent


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
    # 检索参数（含 hybrid 混合检索）同样从 RAG_CONFIG 读取
    rag_agent = RAGKnowledgeAgent(
        name="RAGKnowledgeAgent",
        model=model,
        collection_name="business_travel_knowledge",
        top_k=3,
    )
    if not getattr(rag_agent, "initialized", True):
        return {"error": "RAG 未初始化，请先运行 "
                         "python .claude/skills/ask-question/script/init_knowledge_base.py"}
    user_msg = Msg(name="user", content=user_query, role="user")
    result = await rag_agent.reply(user_msg)
    return json.loads(result.content) if isinstance(result.content, str) else result.content


# 使用
data = asyncio.run(ask_question("北京的住宿标准是多少？"))
# data: {"status": "success"|"no_knowledge", "answer": "...", "retrieved_documents": [...], "query": "..."}
```

> 日常使用直接跑 `python cli.py`，由 `LazyAgentRegistry` 自动加载本技能，无需手动 import。

## 返回格式

`reply()` 返回的 JSON：

- `status`: `"success"` 或 `"no_knowledge"`
- `answer`: 自然语言答案
- `retrieved_documents`: 列表，每项含 **`content`**（截断到 200 字符）与 **`metadata`**（`category` / `title` / `file_path` / `parent_doc`，用于文档溯源）
- `query`: 用户问题

> `search_knowledge()`（更底层的方法）返回的每项还带两个**调试/评测用字段**：
> - `matched_by`：`"vector"` / `"bm25"` / `"vector+bm25"`，标明该条由哪一路召回
> - `rrf_score`：RRF 融合得分
> - 仅 BM25 命中的条目 `distance` 为 `null`（该类条目没有余弦分数）
>
> 这两个字段**不会**出现在 `reply()` 的输出里。

## 知识库

- 路径：`.claude/skills/ask-question/data/rag_knowledge/`（Milvus Lite，本地 `.db`）
- 源文档：`.claude/skills/ask-question/data/documents/`，共 12 类（差旅标准、报销、预订、FAQ、紧急处理、平台指南、城市指南、环保、会员权益、国际差旅、景点指南、特殊时期政策），合计 90 个 chunk
- 首次使用前需执行：
  ```bash
  python .claude/skills/ask-question/script/init_knowledge_base.py
  ```
- 分块策略：按段落切分，每块 ≤600 字符、超长段落按 100 字符重叠硬切

## 自检

```powershell
# 混合检索算法单测（离线，不依赖网络/Milvus/jieba）
venv\Scripts\python.exe tests\test_hybrid_retriever.py
```


## 回答生成指南

【回答要求】
1. 必须严格基于知识库中的信息进行回答，严禁编造。
2. 如果检索到的知识库信息与问题无关，或者信息不足以回答问题，请直接回答“知识库中没有相关信息”。
3. 回答要准确、简洁、有条理。
4. 如果有多个相关信息，可以分点说明。

请直接给出答案。
