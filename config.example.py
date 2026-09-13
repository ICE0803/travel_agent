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
    "similarity_threshold": 0.5
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
        "backends": ["bing", "duckduckgo", "auto"],
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