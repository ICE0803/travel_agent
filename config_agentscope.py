"""
AgentScope Configuration for ICE Multi-Agent Travel Planning System
适配 AgentScope 1.0.16+
"""
import agentscope
from config import LLM_CONFIG

def init_agentscope():
    """
    初始化AgentScope

    注意：AgentScope 1.0.16+ 版本的API已改变：
    - init()函数不再接受model_configs参数
    - 模型配置改为直接在Agent初始化时指定
    """
    agentscope.init(
        project="ICE-Travel-Planning",
        name="multi_agent_system",
        logging_level="INFO"
    )

    print(f"✓ AgentScope initialized (version: {agentscope.__version__})")


def get_model_config():
    """
    获取模型配置（用于Agent初始化）

    Returns:
        dict: 模型配置字典
    """
    return {
        "model_type": "openai_chat",  # 使用OpenAI兼容接口
        "config_name": "deepseek_api",
        "model_name": LLM_CONFIG["model_name"],
        "api_key": LLM_CONFIG["api_key"],
        "base_url": LLM_CONFIG["base_url"],
        "temperature": LLM_CONFIG.get("temperature", 0.7),
        "max_tokens": LLM_CONFIG.get("max_tokens", 2000),
    }
