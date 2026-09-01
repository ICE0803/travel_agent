#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试意图识别智能体

使用方法：
  conda activate base  # 先激活conda环境
  python tests/test_intention_agent.py
"""
import sys
import os
import asyncio
import json

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import agentscope
    print(f"AgentScope: {agentscope.__version__}")
    print("=" * 70)
    print()
except ImportError as e:
    print("❌ AgentScope 未安装或路径不正确")
    print(f"错误: {e}")
    print()
    print("解决方案：")
    print("1. 打开新终端")
    print("2. 运行: conda activate base")
    print("3. 运行: python tests/test_intention_agent.py")
    print()
    print("或者使用绝对路径:")
    print("  /opt/miniconda3/bin/python tests/test_intention_agent.py")
    print("=" * 70)
    sys.exit(1)

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from agents.intention_agent import IntentionAgent
from config import LLM_CONFIG


async def test_intention_agent():
    """测试意图识别智能体的各种场景"""

    # 初始化AgentScope
    print("初始化 AgentScope...")
    try:
        init_agentscope()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建模型配置
    print("创建模型...")
    try:
        model = OpenAIChatModel(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            client_kwargs={
                "base_url": LLM_CONFIG["base_url"],
            },
            temperature=LLM_CONFIG.get("temperature", 0.7),
            max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        )
        print(f"✓ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建意图识别智能体
    print("创建意图识别智能体...")
    try:
        agent = IntentionAgent(
            name="IntentionAgent",
            model=model
        )
        print(f"✓ Agent 创建成功")
    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试用例
    test_cases = [
        {
            "name": "行程规划",
            "query": "我要从北京去上海出差3天"
        },
        {
            "name": "偏好收集",
            "query": "我的家在杭州，我喜欢住汉庭酒店"
        },
        {
            "name": "个性化需求",
            "query": "我要大机型，靠窗座位"
        },
        {
            "name": "信息查询",
            "query": "上海的天气怎么样？"
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "="*70)
        print(f"测试 {i}: {test_case['name']}")
        print("="*70)
        print(f"用户查询: {test_case['query']}")
        print()

        try:
            # 创建消息
            user_msg = Msg(name="User", content=test_case['query'], role="user")

            # 调用意图识别 (async)
            result = await agent(user_msg)

            # 解析JSON结果
            result_data = json.loads(result.content)

            # 显示结果
            display_result(result_data)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

        print()


def display_result(result):
    """格式化显示结果"""

    # 推理过程
    if result.get("reasoning"):
        print("【推理过程】")
        print("-" * 60)
        print(result["reasoning"])
        print("-" * 60)
        print()

    # 识别的意图
    intents = result.get("intents", [])
    if intents:
        print("【识别的意图】")
        for intent in intents:
            intent_type = intent.get("type", "未知")
            confidence = intent.get("confidence", 0)
            description = intent.get("description", "")
            reason = intent.get("reason", "")
            print(f"  • {intent_type} (置信度: {confidence:.2f})")
            if description:
                print(f"    说明: {description}")
            if reason:
                print(f"    原因: {reason}")
        print()

    # 关键实体
    entities = result.get("key_entities", {})
    if entities and any(entities.values()):
        print("【提取的关键实体】")
        for key, value in entities.items():
            if value and value != f"{key}（如果有）":  # 过滤掉模板文本
                print(f"  • {key}: {value}")
        print()

    # 标准化查询
    rewritten = result.get("rewritten_query")
    if rewritten:
        print("【智能Query改写】")
        print(f"  {rewritten}")
        print()

    # 智能体调度计划
    schedule = result.get("agent_schedule", [])
    if schedule:
        print("【智能体调度计划】")
        for agent in schedule:
            agent_name = agent.get("agent_name") or agent.get("agent_type")  # 兼容两种命名
            priority = agent.get("priority", 0)
            reason = agent.get("reason", "")
            expected_output = agent.get("expected_output", "")
            print(f"  {priority}. {agent_name}")
            if reason:
                print(f"     原因: {reason}")
            if expected_output:
                print(f"     期望输出: {expected_output}")
        print()


if __name__ == "__main__":
    print("="*70)
    print("意图识别智能体测试")
    print("="*70)
    print()

    asyncio.run(test_intention_agent())

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
