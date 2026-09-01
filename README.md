# ICE 智能旅行助手

基于 **AgentScope** 多智能体框架 + **大语言模型（OpenAI 兼容接口，可通过 `config.py` 配置，如豆包 / DeepSeek）** 的多智能体旅行规划系统。采用 **Plan-and-Execute** 架构，实现语义意图识别、两层记忆系统、RAG 知识库、联网搜索和优先级并行调度。

## ✨ 核心亮点

### 🎯 智能意图识别
- 基于 LLM 语义理解的多意图识别（6 大类：行程规划、记忆查询、偏好管理、知识问答、信息查询、事项收集）
- 输出结构化决策：**推理过程 + 多意图（含置信度）+ 关键实体 + Query 改写 + Agent 调度计划（agent_schedule）**
- 自然语言理解，不依赖关键词匹配

### 🧠 两层记忆架构
- **短期记忆**：会话级**滑动窗口**，保存最近 `10` 轮对话，用于上下文理解与消歧
- **长期记忆**：**JSON 文件持久化**（`data/memory/{user_id}.json`），保存用户偏好、历史行程、全量聊天记录，支持**跨会话**访问
- **LLM 异步总结**：定期对长期记忆生成摘要，随上下文注入意图识别
- **偏好智能识别**：自动判断"追加"（"我还喜欢如家"）还是"覆盖"（"我搬家到上海了"）
- > 架构上预留了生产环境演进：短期记忆可替换为 **Redis**（TTL 会话共享），长期记忆可替换为 **PostgreSQL**；当前 Demo 以内存 + JSON 实现，接口已封装在 `context/` 下，方便切换。

### 📚 RAG 知识库
- **Milvus Lite** 向量数据库（本地 `.db`）+ **bge-small-zh-v1.5** 中文 Embedding 模型（本地部署 `data/models/`）
- **文档分块（Chunking）**：长文档按段落切分（默认每块 ≤600 字符、重叠 100 字符）
- **余弦相似度检索**（Top-K=3）+ **文档溯源**：返回 `metadata`（类别、标题、来源、原文档路径），保证可追溯、可验证
- 知识来源：差旅规定、报销、预订指南、FAQ、应急处理、平台指南、城市指南、环保倡议 8 类文档

### ⚡ 优先级并行调度
- **Plan-and-Execute**：`IntentionAgent`（规划）→ `OrchestrationAgent`（调度）→ 子 Agent（执行）
- **同优先级 Agent 并行执行**（`asyncio.gather`），不同优先级按依赖串行
- 例：`Priority 1`（记忆查询、事项收集、偏好、信息查询、RAG）并行 → `Priority 2`（行程规划）整合

### 🏗️ 插件化架构
- **Skill Plugins**：所有子 Agent 重构为独立插件（`.claude/skills/`），支持独立开发、测试、部署
- **LazyAgentRegistry**：启动时自动扫描 skills 目录、动态发现并注册，无需手动配置
- **懒加载**：未被调用的 Skill 不加载，系统启动更快
- **Progressive Disclosure（渐进式披露）**：意图识别阶段仅加载 Skill 元数据，执行阶段按需加载详细指令，控制上下文与 Token 开销

### 🛡️ 稳定性保障（连接可用性）
- **熔断器**：连续失败若干次后打开，暂停调用 LLM，一段时间后**半开试探**恢复
- **指数退避重试**：仅对超时、429、5xx 等可重试错误自动重试（最多 3 次）
- **健康检查**：会话内 `health` 查看熔断状态并探测 LLM；命令行 `python cli.py health` 可做独立探测（退出码 0/1，便于监控）

---

## 系统架构

```
用户输入
   ↓
┌──────────────────────────────────────────────────────────┐
│  IntentionAgent (意图识别)                                 │
│  - 语义理解意图（非关键词匹配）                            │
│  - 识别关键实体 / 生成调度计划 / 确定优先级               │
│  - 动态加载 Skills 元数据 (Progressive Disclosure)        │
└──────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  OrchestrationAgent (协调器)                              │
│  - 按优先级调度 / 同优先级并行                            │
│  - 管理 Agent 间消息传递 / 集成两层记忆                    │
│  - 动态实例化 Skills (LazyAgentRegistry)                  │
└──────────────────────────────────────────────────────────┘
   ↓
┌───────────── Priority 1 (并行执行，信息收集) ─────────────┐
│  MemoryQuery  记忆查询      .claude/skills/memory-query   │
│  EventCollection 事项收集   .claude/skills/event-collection│
│  Preference   偏好管理      .claude/skills/preference      │
│  InformationQuery 信息查询  .claude/skills/query-info      │
│  RAGKnowledgeAgent 知识问答 .claude/skills/ask-question    │
└──────────────────────────────┬────────────────────────────┘
   ↓
┌───────────── Priority 2 (依赖 P1 结果，串行) ─────────────┐
│  ItineraryPlanningAgent 行程规划  .claude/skills/plan-trip │
└──────────────────────────────┬────────────────────────────┘
   ↓
[结果聚合 + 记忆回写 + 生成人性化回复]
   ↓
用户看到结果
```

### 连接与可用性

为保证 LLM 服务不稳定时的可用性，在调用链外部增加了以下机制（不改变原有业务逻辑）：

| 机制 | 说明 |
|------|------|
| **熔断器** | 连续失败若干次后暂停调用 LLM，直接提示「服务暂时不可用」；一段时间后自动半开试探恢复。 |
| **重试与退避** | 对意图识别、编排两次 LLM 调用做有限次重试，仅对超时、429、5xx 等可重试错误生效，采用指数退避。 |
| **健康检查** | 会话内输入 `health` 查看熔断状态并探测 LLM 是否可达；命令行 `python cli.py health` 可单独探测（退出码 0/1）。 |

配置见 `config.py` 中的 `RESILIENCE_CONFIG`。

---

## 📊 关键指标（优化效果）

> 说明：以下为项目优化与评测结果；具体数值会随测试集、运行环境与模型不同而有差异，核心模块可通过 `tests/` 下的脚本复现验证。

| 指标 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| 意图识别准确率 | 65%（关键词匹配） | 90%+（LLM 语义理解） | 多意图、可消歧、支持语境 |
| 知识库问答准确率 | - | 95% | 基于 RAG 检索 + 文档溯源 |
| 用户偏好记忆准确率 | - | 95% | 智能识别追加/覆盖 |
| 系统响应时间 | 30 秒（串行） | 15 秒（优先级并行） | 同优先级 Agent 并行执行 |
| 系统启动速度 | 未优化 | 快 | 懒加载 + 渐进式披露 |

**优化路径**：
1. **V1.0**：关键词匹配意图识别 + 串行调度
2. **V2.0**：两层记忆系统 + RAG 知识库 + 联网搜索
3. **V3.0**：LLM 语义理解意图识别 + 优先级并行调度
4. **V4.0**：Skill Plugins 插件化架构 + LazyAgentRegistry + 懒加载

---

## 核心功能

### 1. 意图识别（基于 LLM 语义理解）

支持 **6 大类意图**自动识别（含多意图与置信度）：

- ✅ **itinerary_planning**：规划未来行程，如「我想3月11日从北京去杭州出差一周」
- ✅ **memory_query**：查询历史记忆，如「我去过哪里？我之前说过什么偏好？」
- ✅ **preference**：管理用户偏好（支持追加/覆盖），如「我还喜欢如家」「我搬家到上海了」
- ✅ **rag_knowledge**：查询企业差旅知识库，如「差旅标准是什么？」
- ✅ **information_query**：联网查询实时信息，如「杭州明天天气怎么样？」
- ✅ **event_collection**：收集行程要素（出发地、目的地、日期、时长、目的）

### 2. 两层记忆系统

**短期记忆（会话级）**
- 内存中的滑动窗口，保存最近 10 轮对话（每轮 = 用户 + 助手）

**长期记忆（持久化）**
- **JSON 文件**：`data/memory/{user_id}.json`，包含用户偏好、历史行程、完整聊天历史和统计
- **偏好管理**：支持动态任意偏好类型，智能识别追加/覆盖动作
- **历史行程**：出发地、目的地、时间、目的，支持跨会话查询
- **统计**：常去目的地、总行程数
- **LLM 异步总结**：自动生成历史摘要，注入上下文

### 3. RAG 知识库

- **向量数据库**：Milvus Lite（本地 `.db`）
- **Embedding**：`bge-small-zh-v1.5`（本地部署，`data/models/bge-small-zh-v1.5/`）
- **文档处理**：分块 + 余弦相似度检索（Top-K=3）
- **可追溯性**：返回文档来源（类别、标题、源文档路径），支持知识溯源
- **知识内容（8 类）**：差旅规定、报销、预订指南、FAQ、应急处理、平台指南、城市指南、环保倡议

### 4. 信息查询（联网搜索）

- **天气**：`wttr.in` 免费接口（结果可靠、无需 Key）
- **网络搜索**：`ddgs`（DuckDuckGo 多后端），开启 safesearch + 过滤可疑域名
- **LLM 自动摘要**：对搜索结果智能提取，返回来源
- **异步查询**：提升响应速度

### 5. 优先级并行调度

- **多意图识别**：支持 6 类意图
- **优先级 + 并行混合模式**：同优先级并行，不同优先级按依赖串行
- **动态调度**：根据意图识别结果实时分配优先级
- **性能提升**：响应时间从 30 秒降至 15 秒（-50%）

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

复制模板并填入你的 API Key（OpenAI 兼容接口）：

```bash
cp config.example.py config.py   # macOS/Linux
# copy config.example.py config.py   # Windows
```

编辑 `config.py`：

```python
LLM_CONFIG = {
    "api_key": "your-api-key-here",   # 替换为你的 Key
    "model_name": "deepseek-v4-flash-vision-exp",  # 或豆包等 OpenAI 兼容模型
    "base_url": "https://api.deepseek.com/v1",     # 对应模型的 base_url
    "temperature": 0.7,
    "max_tokens": 8192,
}
```

### 3. 初始化知识库

```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

### 4. 启动系统

```bash
python cli.py
```

---

## 子智能体详解 (Skills)

所有子智能体均位于 `.claude/skills/{skill}/script/agent.py`，实现自包含，支持动态发现与加载。

| Skill | 职责 |
|------|------|
| **memory-query**（记忆查询） | 查询旅行历史、用户偏好、历史对话摘要，用 LLM 生成自然语言回答 |
| **event-collection**（事项收集） | 提取出发地、目的地、出发时间、返程时间、出行目的，主动推断缺失信息 |
| **preference**（偏好管理） | 管理酒店、航空、座位、房型、机型、餐饮、交通、预算等偏好；支持任意自定义类型；智能识别追加/覆盖 |
| **query-info**（信息查询） | `wttr.in`（天气）+ DDGS（联网搜索）+ LLM 摘要 |
| **ask-question**（知识问答） | Milvus Lite + bge-small-zh-v1.5 检索知识库并生成答案，返回文档溯源 |
| **plan-trip**（行程规划） | 整合各 Skill 结果与用户偏好，生成完整行程（每日安排、住宿、餐饮、交通、注意事项） |

---

## CLI 使用指南

### 启动

```bash
python cli.py
```

### 内置命令

| 命令 | 说明 |
|------|------|
| `help` | 显示帮助 |
| `status` | 查看当前状态和记忆 |
| `health` | 检查 LLM 服务是否可用并显示熔断器状态 |
| `clear` | 清空当前任务（保留长期记忆） |
| `history` | 查看历史行程 |
| `preferences` | 查看用户偏好 |
| `exit` | 退出程序 |

单独做健康检查（不进入交互）：`python cli.py health`，返回 `OK` / `FAIL: ...`，退出码 0/1。

---

## 测试

### 集成测试（QA）
完整跑通所有意图和子智能体的端到端测试，并生成 QA 报告：
```bash
python tests/test_cli_qa.py
```

### 单元 / 模块测试
```bash
python tests/test_memory_system.py      # 记忆系统
python tests/test_intention_agent.py    # 意图识别
python tests/test_information_query_agent.py  # 信息查询（天气/搜索，需联网）
```

---

## 项目结构

```
travel_agent/
├── agents/                          # 核心编排层
│   ├── intention_agent.py           # 意图识别（语义理解）
│   ├── orchestration_agent.py       # 协调器（优先级并行调度）
│   └── lazy_agent_registry.py       # 智能体插件注册器（懒加载）
├── .claude/skills/                  # Skill Plugins (子智能体)
│   ├── ask-question/                # 知识库问答 (RAG)
│   │   ├── script/agent.py
│   │   ├── script/init_knowledge_base.py
│   │   ├── data/documents/          # 8 类知识源文档
│   │   └── SKILL.md
│   ├── event-collection/            # 事项收集
│   ├── plan-trip/                   # 行程规划
│   ├── preference/                  # 偏好管理
│   ├── query-info/                  # 信息查询
│   └── memory-query/                # 记忆查询
├── context/                         # 记忆系统
│   ├── memory_manager.py            # 记忆管理器
│   ├── short_term_memory.py         # 短期记忆（滑动窗口）
│   └── long_term_memory.py          # 长期记忆（JSON 持久化）
├── data/
│   ├── memory/                      # 长期记忆 JSON（user_id.json）
│   └── models/bge-small-zh-v1.5/    # 本地 Embedding 模型
├── tests/                           # 测试脚本
├── utils/                           # 工具与连接可用性
│   ├── circuit_breaker.py           # 熔断器
│   ├── llm_resilience.py            # 重试退避、健康检查
│   ├── json_parser.py               # 鲁棒 JSON 解析
│   └── skill_loader.py              # Skill 加载器（渐进式披露）
├── cli.py                           # CLI 主程序
├── config.py                        # 配置（本地，含 API Key，不入库）
├── config.example.py                # 配置模板
├── config_agentscope.py             # AgentScope 初始化与模型配置
└── README.md
```

---

## 技术栈总览

### 核心框架
- 📦 **AgentScope 1.0.16** - 多智能体框架
- 🤖 **大语言模型（OpenAI 兼容）** - 通过 `OpenAIChatModel` 配置，如 DeepSeek / 豆包

### 数据存储
- 🗄️ **JSON 文件** - 长期记忆持久化（`data/memory/{user_id}.json`）
- 💾 **内存滑动窗口** - 短期记忆（会话级）
- 🔍 **Milvus Lite** - 向量数据库（本地 `.db`，RAG 知识库）

### 向量化与检索
- 🧠 **bge-small-zh-v1.5** - 中文 Embedding 模型（本地部署）
- 📚 **Sentence-Transformers** - 向量化工具库
- 🎯 **余弦相似度检索** - Top-K 检索算法

### 联网与搜索
- 🌐 **wttr.in** - 天气查询（免费）
- 🔎 **ddgs** - 网络搜索（多后端）
- 📝 **LLM 自动摘要** - 搜索结果智能提取

### 架构设计
- 🏗️ **Skill Plugins 插件化架构** - 独立开发、测试、部署
- 🔄 **LazyAgentRegistry 动态发现** - 自动扫描注册 Agent 插件
- ⚡ **懒加载** - 未使用的 Skill 不加载（启动快）
- 🔀 **Progressive Disclosure 渐进式披露** - 意图阶段仅元数据、执行阶段按需加载指令
- 🎯 **优先级 + 并行混合调度** - `asyncio.gather` 并发执行

### 稳定性保障
- 🔁 **指数退避重试** - 自动重试失败请求（最大 3 次）
- 🩺 **熔断器机制** - 连续失败后暂停调用
- 💊 **健康检查** - 实时监控 LLM 服务可用性

### 用户界面
- 🖥️ **Rich** - CLI 终端界面

---

## ⚠️ 注意事项

### 模型配置
- 必须配置 API Key（在 `config.py`，或复制 `config.example.py` 后填写）
- 支持任意 OpenAI 兼容接口（DeepSeek、豆包等），需同时设置 `model_name` 与 `base_url` 保持一致
- Embedding 模型为本地 `data/models/bge-small-zh-v1.5/`

### 数据存储
- **短期记忆**：内存滑动窗口（会话级）
- **长期记忆**：JSON 文件（`data/memory/{user_id}.json`）
- README 中提到的 **Redis / PostgreSQL** 是**面向生产环境的架构演进方案**，当前 Demo 以 JSON + 内存落地，接口已封装在 `context/` 便于切换

### 知识库初始化
- 首次运行前必须初始化 RAG 知识库：
  ```bash
  python .claude/skills/ask-question/script/init_knowledge_base.py
  ```
- 源文档：`.claude/skills/ask-question/data/documents/`
- 向量库文件：`.claude/skills/ask-question/data/rag_knowledge/milvus_lite.db`
- 若使用 `pymilvus 3.x`，需在 `search_knowledge()` 中先调用 `load_collection()` 再检索（代码已处理）

---

## 🚀 未来规划

- [ ] 完整实现 PostgreSQL 持久化 / Redis 缓存层（当前为 JSON + 内存 Demo）
- [ ] 更完整的多路召回（向量 + BM25 混合检索、Rerank）
- [ ] 支持更多 LLM / 切换模型
- [ ] Web 界面（FastAPI + React）
- [ ] 更多 Skill 插件（酒店预订、机票查询等）
- [ ] 监控与日志系统（链路追踪、Token 统计）

---

## 许可证

MIT License
