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