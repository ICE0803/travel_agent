LLM_CONFIG = {
    "api_key": "your-api-key-here",
    "model_name": "deepseek-v4-flash-vision-exp",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}
SYSTEM_CONFIG = {"enable_llm": True, "log_level": "INFO", "max_retries": 3, "timeout": 60}
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
    # 向量路余弦阈值（防幻觉）。用 scripts/eval_retrieval.py 实测可分区间后取中点；
    # 区间内任意值对 Hit@k / MRR 完全等价，只影响鲁棒性余量。
    # 当前实测可分区间 (0.482, 0.545)，中点即 0.514。
    "similarity_threshold": 0.514,
    "hybrid": {
        "enabled": True,        # False 则退回纯向量检索（用于 A/B 对比）
        "top_k_dense": 10,      # 向量路召回条数
        "top_k_sparse": 10,     # BM25 路召回条数
        "rrf_k": 60,            # RRF 公式里的 k（原论文经验值）
        "dense_weight": 1.0,    # RRF 向量路权重（只有 dense:sparse 比值有意义）
        "sparse_weight": 1.5,   # RRF BM25 路权重；调大表示更信任关键词精确命中
                                # 用 scripts/tune_rrf_weights.py 扫描确定（实测平台 1.05~1.95）
        "rrf_candidates": 10,   # RRF 融合后送入精排的候选条数（需 >= final_top_k）
                                # 精排未启用时该值不生效，行为与改造前一致
        "max_per_doc": 0,       # 每篇文档最多保留几个 chunk；0 = 不限制（默认）
                                # 设 1 会让"取 3 条喂给 LLM"来自 3 篇不同文档，但实测对
                                # Hit@k / MRR 中性、且会拉低"精排重排"，故默认关闭
        "final_top_k": 3,       # 融合后最终返回条数
        "bm25_k1": 1.5,         # BM25 词频饱和参数
        "bm25_b": 0.75,         # BM25 文档长度归一化参数
        "min_bm25_score": 0.5,  # BM25 单词面命中的准入阈值（防幻觉用，需用评测脚本调）
    },
    "rerank": {
        "enabled": False,                          # True 开启；关闭时行为与改造前逐字节一致

        # filter : 精排分数只做**准入判断**，排序仍用 RRF（实测最优）
        # rerank : 用精排分数**重排**（实测在本语料上为负收益，保留用于 A/B 对比）
        "mode": "filter",

        "model": "data/models/bge-reranker-base",  # 本地目录优先
        "device": None,                            # None=自动（本机为 CPU）
        "batch_size": 16,
        "max_length": 512,                         # 延迟的第一大旋钮（模型上限 514）

        # mode=filter 时**必填**：为空则精排会被直接跳过（零开销）
        # 取值用 scripts/eval_retrieval.py 的【精排路】标定结果
        "score_threshold": None,

        "allow_download": False,                   # 本地模型缺失时是否允许联网拉取
    }
}
RESILIENCE_CONFIG = {
    "max_retries": 3, "retry_base_delay_sec": 1.0, "retry_max_delay_sec": 30.0,
    "circuit_failure_threshold": 5, "circuit_recovery_timeout_sec": 60.0,
    "circuit_half_open_successes": 2, "health_check_timeout_sec": 10.0,
}
INTENT_CONFIG = {
    # agent_schedule 中 confidence 低于此值的任务不予调度
    "confidence_threshold": 0.5,
}
# 网络搜索：多后端可插拔（Tavily / DDGS）
SEARCH_CONFIG = {
    # auto: 按 auto_order 依次尝试；也可强制 "tavily" / "ddgs"
    "backend": "auto",
    "auto_order": ["tavily", "ddgs"],
    "tavily": {
        # 控制台 https://app.tavily.com ；免费 1000 credits/月
        # 建议改用环境变量 TAVILY_API_KEY
        "api_key": "",
        "endpoint": "https://api.tavily.com/search",
        "search_depth": "basic",   # basic(1 credit) | advanced(2 credits)
        "max_results": 10,
        "topic": "general",
        "timeout": 15.0,
        "max_content_chars": 500,  # Tavily 返回正文较长，截断后再摘要
    },
    "ddgs": {
        # 用 scripts/check_search_api.py 实测本机可用的后端再填这里
        # （实测 duckduckgo 已失效，yandex 可用）
        "backends": ["bing", "auto", "yandex"],
        "max_results": 10,
        "region": "cn-zh",
        "safesearch": "on",
    },
    "max_results": 5,        # 过滤可疑域名后最多保留几条给 LLM 摘要
}
# 存储后端：Redis 缓存 + PostgreSQL 长期记忆
STORAGE_CONFIG = {
    # auto     : 优先 PG/Redis，连不上自动降级为 JSON + 内存
    # postgres : 强制使用，连不上直接抛错（便于发现配置问题）
    # local    : 强制走旧实现（离线演示 / CI）
    "backend": "auto",

    "postgres": {
        "host": "localhost",
        "port": 5432,
        "dbname": "travel_agent",
        "user": "postgres",
        "password": "your_db_password",
        "min_pool": 1,
        "max_pool": 10,
        "connect_timeout": 5,
    },

    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "socket_timeout": 2.0,
        "max_connections": 16,
    },

    # TTL 单位秒
    "ttl": {
        "session": 3600,      # 短期记忆（会话）1 小时
        "preferences": 600,   # 偏好热数据 10 分钟
        "summary": 1800,      # LLM 总结 30 分钟
    },
}