# Aligo 商旅 Skills（Claude / Cursor 可用）

基于本项目 **agents/** 实际实现的业务 Skills，便于在 Claude 或 Cursor 中按意图调用对应 Agent。

---

## 怎么用、怎么问 Claude

在对话里**用自然语言说出你的需求**，Claude 会根据描述自动选用对应的 Skill（或组合多个）。不需要记命令，直接像和同事说话一样问即可。

下面按 Skill 列出**典型问法**和**会得到什么**。

---

### 1. ask-question（知识库问答）

**怎么问：**
- 「北京的住宿标准是多少？」
- 「差旅费用什么时候报销？流程是啥？」
- 「航班延误了怎么办？」
- 「如何改签/退票？」
- 「上海差旅有什么注意事项？」

**会干啥：** 用 RAG 从项目知识库（`data/documents/`）检索文档，用 LLM 生成答案，并给出参考来源。  
**前置条件：** 已运行过 `python scripts/init_knowledge_base.py`。

---

### 2. query-info（天气 / 网络搜索）

**怎么问：**
- 「北京明天天气怎么样？」
- 「上海下周天气」
- 「查一下 XX 景点开放时间」
- 「搜一下最近 XX 新闻」

**会干啥：** 天气走 wttr.in；其他用 DDGS 做网络搜索并给摘要。  
**注意：** 差旅标准、报销政策这类请用「知识库问答」问，不要用「查一下差旅标准」当搜索。

---

### 3. plan-trip（行程规划）

**怎么问：**
- 「帮我规划一下从上海到北京的行程」
- 「2 月 27 日去北京出差，帮我安排一下」
- 「我想去杭州玩两天，怎么安排？」
- 「从广州到成都，3 天，规划路线」

**会干啥：** 先做意图识别，再收集出发地、目的地、日期等，最后生成行程（每日安排、交通、住宿建议等）。  
**可能被追问：** 缺出发地、日期等信息时，Claude 会问你补全。

---

### 4. memory-query（查我的历史）

**怎么问：**
- 「我去过哪些地方？」
- 「我上次去北京是什么时候？」
- 「我之前说过什么偏好？」
- 「我的旅行记录有哪些？」

**会干啥：** 从长期记忆（`data/memory/{user_id}.json`）里查你的行程、偏好、对话摘要，用自然语言回答。  
**前置条件：** 需要有 MemoryManager（user_id/session_id），且之前有用过 CLI 或其它方式写入过记忆。

---

### 5. preference（保存 / 改我的偏好）

**怎么问：**
- 「我喜欢住汉庭」
- 「我还喜欢如家」（追加酒店偏好）
- 「我常坐东航」（追加航空公司）
- 「我搬家到上海浦东了」（覆盖常住地）
- 「改成靠窗座位」（覆盖座位偏好）

**会干啥：** 识别是「追加」还是「覆盖」，把偏好写入长期记忆；后续规划、记忆查询都会用到。  
**前置条件：** 同 memory-query，需要 MemoryManager。

---

### 组合问法（一次多件事）

也可以一句话里带多个意图，例如：
- 「我想去北京出差，先帮我查一下北京天气，再按差标说说住宿标准。」  
Claude 会依次用 **query-info**（天气）和 **ask-question**（住宿标准）。

---

## 可用 Skills 一览

| Skill | 用途 | 触发示例 | 主要 Agent |
|-------|------|----------|------------|
| **ask-question** | 差旅政策/报销/预订/城市指南等知识问答 | 「XX标准是多少」「如何报销」「航班延误怎么办」 | RAGKnowledgeAgent |
| **query-info** | 天气、网络搜索等实时信息 | 「天气怎么样」「查一下XX」 | InformationQueryAgent |
| **plan-trip** | 行程规划 | 「规划行程」「从XX到XX」「X月X日去北京」 | IntentionAgent → EventCollectionAgent → ItineraryPlanningAgent |
| **memory-query** | 查询用户自己的历史行程与偏好 | 「我去过哪些地方」「我上次去北京是什么时候」 | MemoryQueryAgent |
| **preference** | 保存/追加/覆盖用户偏好 | 「我喜欢住汉庭」「我还喜欢如家」「我搬家到上海了」 | PreferenceAgent |

---

## 统一约定（与代码一致）

1. **模型传入方式**  
   所有 Agent 使用 **`model=model`**（传入已创建的 `OpenAIChatModel` 实例）。  
   本项目**没有** `model_config_name` 参数。

2. **异步调用**  
   所有子 Agent 的 `reply()` 均为 **async**，调用时需 **await**。

3. **模型创建**  
   ```python
   from agentscope.model import OpenAIChatModel
   from config import LLM_CONFIG
   model = OpenAIChatModel(
       model_name=LLM_CONFIG["model_name"],
       api_key=LLM_CONFIG["api_key"],
       client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": 60},
       temperature=LLM_CONFIG.get("temperature", 0.7),
       max_tokens=LLM_CONFIG.get("max_tokens", 2000),
   )
   ```

4. **依赖 main.py / cli.py**  
   Skills 直接导入 **agents/** 与 **context/**，不依赖 `main.py` 或 `cli.py`。完整交互流程见 `cli.py`。

---

## Agent 与文件对应

| Agent | 文件 | 职责 |
|-------|------|------|
| IntentionAgent | intention_agent.py | 意图识别与智能体调度 |
| EventCollectionAgent | event_collection_agent.py | 出发地、目的地、日期等事项收集 |
| PreferenceAgent | preference_agent.py | 用户偏好识别（追加/覆盖） |
| InformationQueryAgent | information_query_agent.py | 天气、网络搜索 |
| RAGKnowledgeAgent | rag_knowledge_agent.py | 知识库检索与问答 |
| MemoryQueryAgent | memory_query_agent.py | 基于长期记忆回答历史问题 |
| ItineraryPlanningAgent | itinerary_planning_agent.py | 生成行程计划 |
| OrchestrationAgent | orchestration_agent.py | 协调多 Agent（CLI 主流程使用） |

---

## 目录结构

```
.claude/skills/
├── README.md
├── ask-question/SKILL.md
├── query-info/SKILL.md
├── plan-trip/SKILL.md
├── memory-query/SKILL.md
└── preference/SKILL.md
```

---
