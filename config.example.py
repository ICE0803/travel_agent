LLM_CONFIG = {
    "api_key": "your-api-key-here",
    "model_name": "deepseek-v4-flash-vision-exp",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}
SYSTEM_CONFIG = {"enable_llm": True, "log_level": "INFO", "max_retries": 3, "timeout": 60}
RAG_CONFIG = {"embedding_model": "data/models/bge-small-zh-v1.5"}
RESILIENCE_CONFIG = {
    "max_retries": 3, "retry_base_delay_sec": 1.0, "retry_max_delay_sec": 30.0,
    "circuit_failure_threshold": 5, "circuit_recovery_timeout_sec": 60.0,
    "circuit_half_open_successes": 2, "health_check_timeout_sec": 10.0,
}