# ICE 智能旅行助手

基于 **AgentScope** 多智能体框架 + **大语言模型（OpenAI 兼容接口，可通过 `config.py` 配置，如豆包 / DeepSeek）** 的多智能体旅行规划系统。采用 **Plan-and-Execute** 架构，实现语义意图识别、两层记忆系统（**Redis + PostgreSQL**）、RAG 知识库、联网搜索和优先级并行调度。

## ✨ 核心亮点

### 🎯 智能意图识别
- 基于 LLM 语义理解的多意图识别（6 大类：行程规划、记忆查询、偏好管理、知识问答、信息查询、事项收集）
- 输出结构化决策：**推理过程 + 多意图（含置信度）+ 关键实体 + Query 改写 + Agent 调度计划（agent_schedule）**
- **置信度参与调度**：`agent_schedule` 每项都带 `confidence`，低于阈值（默认 0.5）的任务不予执行；全部被过滤时引导用户补充信息，而不是拿猜测乱调 Agent
- **闲聊兜底**：与差旅无关的输入识别为 `chitchat`，不调用任何 Skill，礼貌告知服务范围并给出示例
- 自然语言理解，不依赖关键词匹配

### 🧠 两层记忆架构
- **短期记忆**：**Redis LIST** 存储，`RPUSH` + `LTRIM` 实现固定长度滑动窗口，保存最近 `10` 轮对话；每次写入续期 `EXPIRE`（**TTL 1 小时**），做到会话级滑动过期
- **长期记忆**：**PostgreSQL** 持久化（`users` / `user_preferences` / `chat_history` / `trip_history`），保存用户偏好、历史行程、全量聊天记录，支持**跨会话**访问
- **Redis 缓存层**：偏好热数据（**Write-Through** 写库即刷新，TTL **10 分钟**）+ LLM 总结结果（TTL **30 分钟**）；`status` 命令可查看实时命中率
- **LLM 异步总结**：定期对长期记忆生成摘要，随上下文注入意图识别；用 `chat_history.MAX(id)` 作为**版本号 watermark**，避免「写入新消息后仍返回旧总结」的脏读
- **偏好智能识别**：自动判断"追加"（"我还喜欢如家"）还是"覆盖"（"我搬家到上海了"）
- **自动降级**：`STORAGE_CONFIG["backend"]="auto"`（默认）时若连不上 PostgreSQL / Redis，自动退回 JSON 文件 + 内存实现，功能不中断；连接池与降级策略集中在 `context/backends.py`，表结构见 `context/schema.sql`

### 📚 RAG 知识库
- **Milvus Lite** 向量数据库（本地 `.db`）+ **bge-small-zh-v1.5** 中文 Embedding 模型（本地部署 `data/models/`）
- **文档分块（Chunking）**：长文档按段落切分（默认每块 ≤600 字符、重叠 100 字符）
- **余弦相似度检索**（Top-K=3）+ **相似度阈值过滤** + **文档溯源**：低于阈值（默认 0.5）的召回片段直接丢弃，知识库无相关内容时明确回答「找不到」而不是让模型硬编；返回 `metadata`（类别、标题、来源、原文档路径），保证可追溯、可验证
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
| 缓存命中率 | - | `status` 实时查看 | Redis 偏好 / 总结命中计数（会话内累计值） |

**优化路径**：
1. **V1.0**：关键词匹配意图识别 + 串行调度
2. **V2.0**：两层记忆系统 + RAG 知识库 + 联网搜索
3. **V3.0**：LLM 语义理解意图识别 + 优先级并行调度
4. **V4.0**：Skill Plugins 插件化架构 + LazyAgentRegistry + 懒加载
5. **V5.0**：PostgreSQL 长期记忆 + Redis 缓存层（Write-Through / Lazy Loading / watermark 防脏读）

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

**置信度阈值**：`agent_schedule` 中每一项都带 `confidence`（0~1）。低于 `INTENT_CONFIG.confidence_threshold`（默认 0.5）的任务不予执行；若全部被过滤，返回 `low_confidence`，由 CLI 引导用户补充信息。模型未输出该字段时按 `1.0` 处理，避免因缺字段误伤正常调度。

**闲聊兜底**：与差旅出行完全无关的输入（打招呼、闲聊等）识别为 `chitchat`，要求 `agent_schedule` 输出空数组、不调用任何 Skill；CLI 礼貌告知仅提供差旅服务并给出示例，而不是把无关问题硬套成差旅需求。

### 2. 两层记忆系统

**短期记忆（会话级，Redis）**
- Redis **LIST** 结构，key = `session:{session_id}:messages`
- 保存最近 10 轮对话（每轮 = 用户 + 助手），`LTRIM` 自动淘汰旧消息
- 每次写入 `EXPIRE` 续期，TTL 1 小时（滑动过期）
- 按 `session_id` 隔离，多会话互不干扰；`clear` / `end_session` 时 `DEL` 清理

**长期记忆（跨会话，PostgreSQL）**
- 4 张表：`users` / `user_preferences` / `chat_history` / `trip_history`
- **偏好管理**：`value` 用 **JSONB** 存储，标量或列表均可，新增偏好类型无需改表；`ON CONFLICT DO UPDATE` 保证幂等写入
- **历史行程**：出发地、目的地、时间、目的，按 `(user_id, created_at DESC)` 索引查询
- **统计**：总行程数、总消息数、常去目的地全部改为 **SQL 聚合查询**（`GROUP BY`），不再全量加载到内存
- **LLM 异步总结**：自动生成历史摘要并缓存到 Redis

**缓存层（Redis）**
- 偏好热数据：**Write-Through**（写库后立即刷新）+ **Lazy Loading**（未命中回源并回填），TTL 10 分钟
- LLM 总结：TTL 30 分钟，用 `chat_history.MAX(id)` 做版本号，有新消息即自动失效
- 命中 / 未命中计数写入 Redis，`status` 命令展示实时命中率

### 3. RAG 知识库

- **向量数据库**：Milvus Lite（本地 `.db`）
- **Embedding**：`bge-small-zh-v1.5`（本地部署，`data/models/bge-small-zh-v1.5/`）
- **文档处理**：分块 + 余弦相似度检索（Top-K=3）
- **相似度阈值过滤**：低于 `RAG_CONFIG.similarity_threshold`（默认 0.5）的召回片段直接丢弃，数量归零时明确回答「知识库中没有找到相关信息」
- **四层防幻觉**：Prompt 强约束 → RAG 知识增强 → 相似度阈值过滤 → 文档溯源
- **可追溯性**：返回文档来源（类别、标题、源文档路径），支持知识溯源
- **知识内容（8 类）**：差旅规定、报销、预订指南、FAQ、应急处理、平台指南、城市指南、环保倡议

### 4. 信息查询（联网搜索）

- **天气**：`wttr.in` 免费接口（结果可靠、无需 Key）
- **网络搜索**：**多后端可插拔** —— Tavily（主）/ DDGS（兜底），按 `auto_order` 依次尝试，前一个失败自动换下一个
- **通道可视**：返回结果的 `results.engine` 标注本次实际走哪个后端；全失败时 `results.attempts` 列出各自原因
- **统一归一化**：无论哪个后端，结果统一为 `{title, snippet, url}`，并过滤可疑域名（safesearch / 可疑 TLD）
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

### 3. 启动 Redis 与 PostgreSQL

系统默认（`STORAGE_CONFIG["backend"] = "auto"`）会优先连接 Redis + PostgreSQL；两者都连不上时自动降级为 JSON + 内存，功能仍可用。

```bash
# 方式一：本地 / WSL 安装
sudo apt install -y postgresql redis-server
sudo service postgresql start && sudo service redis-server start

# 方式二：Docker
docker run -d --name travel-redis -p 6379:6379 redis:7-alpine
docker run -d --name travel-postgres -p 5432:5432 \
  -e POSTGRES_DB=travel_agent -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=你的密码 postgres:16-alpine
```

确认两个服务可连：

```bash
psql -h localhost -U postgres -c "SELECT 1"
redis-cli ping          # 期望 PONG
```

连接参数在 `config.py` 的 `STORAGE_CONFIG` 中配置。

### 4. 建表

```bash
python scripts/init_db.py
```

幂等，可重复执行。

### 5. （可选）迁移历史 JSON 记忆

如果你之前用 JSON 模式跑过，把旧数据迁到 PostgreSQL：

```bash
python scripts/migrate_json_to_pg.py
```

> ⚠️ **只跑一次**。`chat_history` / `trip_history` 没有唯一约束，重复执行会产生重复行。

### 6. 初始化知识库

```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

### 7. 启动系统

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
| **query-info**（信息查询） | `wttr.in`（天气）+ Tavily / DDGS（联网搜索）+ LLM 摘要 |
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
| `status` | 查看当前状态、记忆和**缓存命中率** |
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
│   ├── backends.py                  # 连接池单例（PG ConnectionPool + Redis）+ 自动降级
│   ├── schema.sql                   # PostgreSQL 表结构（幂等）
│   ├── memory_manager.py            # 记忆管理器（LLM 总结缓存 + 命中率统计）
│   ├── short_term_memory.py         # 短期记忆（Redis LIST + TTL 滑动窗口）
│   └── long_term_memory.py          # 长期记忆（PostgreSQL + 偏好 Write-Through 缓存）
├── scripts/
│   ├── init_db.py                   # 建表（幂等）
│   └── migrate_json_to_pg.py        # 历史 JSON 记忆迁移到 PostgreSQL
├── data/
│   ├── memory/                      # 历史 JSON 记忆（降级模式使用）
│   ├── memory_bak/                  # 迁移前备份
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

### 数据存储与缓存
- 🐘 **PostgreSQL** - 长期记忆持久化（`users` / `user_preferences` / `chat_history` / `trip_history`）
- ⚡ **Redis** - 短期记忆会话窗口 + 偏好热数据 + LLM 总结缓存
- 🔌 **psycopg3 + psycopg_pool** - PostgreSQL 连接池（进程级单例）
- 🔍 **Milvus Lite** - 向量数据库（本地 `.db`，RAG 知识库）
- 🔁 **自动降级** - 连不上 PG / Redis 时退回 JSON + 内存（`context/backends.py`）

### 向量化与检索
- 🧠 **bge-small-zh-v1.5** - 中文 Embedding 模型（本地部署）
- 📚 **Sentence-Transformers** - 向量化工具库
- 🎯 **余弦相似度检索** - Top-K 检索算法

### 联网与搜索
- 🌐 **wttr.in** - 天气查询（免费）
- 🔎 **Tavily** - 网络搜索主通道（RAG 友好，免费 1000 credits/月）
- 🔁 **ddgs** - 零配置兜底通道（多后端）
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
- **短期记忆**：Redis LIST（会话级，TTL 1 小时）
- **长期记忆**：PostgreSQL（`travel_agent` 库，4 张表）
- **缓存**：Redis 存偏好热数据（TTL 10 分钟）与 LLM 总结（TTL 30 分钟）
- **首次使用**必须先跑 `python scripts/init_db.py` 建表
- **降级开关**：`STORAGE_CONFIG["backend"]` 设为 `"local"` 可强制走 JSON + 内存（离线演示 / CI）；设为 `"postgres"` 则连接失败时直接抛错（便于暴露配置问题）；默认 `"auto"` 自动降级

### 知识库初始化
- 首次运行前必须初始化 RAG 知识库：
  ```bash
  python .claude/skills/ask-question/script/init_knowledge_base.py
  ```
- 源文档：`.claude/skills/ask-question/data/documents/`
- 向量库文件：`.claude/skills/ask-question/data/rag_knowledge/milvus_lite.db`
- 若使用 `pymilvus 3.x`，需在 `search_knowledge()` 中先调用 `load_collection()` 再检索（代码已处理）

### 网络搜索配置（多后端可插拔）
- 后端由 `config.py` 的 `SEARCH_CONFIG` 控制：
  ```python
  "backend": "auto",                    # auto | tavily | ddgs
  "auto_order": ["tavily", "ddgs"],     # auto 模式下的尝试顺序
  ```
  `auto` 会依次尝试，任一成功即用；某个后端未配置 / 超时 / 配额用尽 / 返回 0 条，都会自动换下一个。
- **后端对比**：

  | 后端 | 凭据 | 免费额度 | 说明 |
  |------|------|----------|------|
  | **Tavily** ⭐ | `TAVILY_API_KEY` | 1000 credits/月 | **主通道**。返回抽取好的正文，适合 RAG；控制台 <https://app.tavily.com> |
  | DDGS | 无需 Key | 无限制 | 兜底通道。抓取公开页面，零配置；实测部分后端已失效，较脆弱 |

- **填 Key 推荐直接写进 `config.py`**（`config.py` 已被 `.gitignore` 忽略，Key 不会进仓库）：
  ```python
  SEARCH_CONFIG = {
      "tavily": {"api_key": "tvly-你的Key", ...},
      ...
  }
  ```
  > ⚠️ 也可以用环境变量 `setx TAVILY_API_KEY "tvly-你的Key"`，但要注意 **Windows 的坑**：
  > `setx` 不会注入**已运行**的进程。若终端继承自已运行的父进程（Windows Terminal / VS Code /
  > 资源管理器未刷新），**重开标签页甚至重开窗口都可能仍读不到**，表现为「明明设置了，程序却说未配置」。
  > 遇到这种情况，最省事的办法就是回到上面直接写进 `config.py`。
- **自检与排错**：
  ```powershell
  venv\Scripts\python.exe scripts\check_search_api.py          # 逐后端实测（Tavily/DDGS）
  venv\Scripts\python.exe scripts\check_search_api.py tavily   # 只测 Tavily
  venv\Scripts\python.exe tests\test_search_backend.py         # 离线验证回退编排（32 项）
  ```
  诊断脚本会自动识别「环境变量已在注册表但当前进程读不到」这种情况，并打印注册表里的值和解法。
- 返回结果的 `results.engine` 标明本次实际使用的后端；若前面有后端失败但最终成功，`results.fallback` 记录失败原因（避免静默降级无人察觉）；全部失败时 `results.attempts` 列出每个后端的原因。

---

## 🚀 未来规划

- [x] ~~PostgreSQL 持久化 / Redis 缓存层~~（已完成：`context/backends.py` + `context/schema.sql`）
- [x] ~~网络搜索主通道接入 Tavily~~（已完成：`query-info` 技能，DDGS 自动兜底）
- [ ] 缓存命中率的独立基准测试（冷启动 / 多会话场景，目前 `status` 显示的是会话内累计值）
- [ ] 更完整的多路召回（向量 + BM25 混合检索、Rerank）
- [ ] 支持更多 LLM / 切换模型
- [ ] Web 界面（FastAPI + React）
- [ ] 更多 Skill 插件（酒店预订、机票查询等）
- [ ] 监控与日志系统（链路追踪、Token 统计）

---

## 许可证

MIT License
