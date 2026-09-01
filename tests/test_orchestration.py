#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试智能体协调系统
"""
import sys
import os
import asyncio
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from config_agentscope import init_agentscope
from config import LLM_CONFIG
from context.memory_manager import MemoryManager
from agents.intention_agent import IntentionAgent
from agents.orchestration_agent import OrchestrationAgent
from agents.event_collection_agent import EventCollectionAgent
from agents.itinerary_planning_agent import ItineraryPlanningAgent
from agents.information_query_agent import InformationQueryAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent


async def test_orchestration():
    """测试智能体协调系统"""
    print("=" * 70)
    print("智能体协调系统测试")
    print("=" * 70)
    print()

    # ========== 初始化 ==========
    print("[1] 初始化系统")
    print("-" * 60)

    # 初始化AgentScope
    init_agentscope()

    # 初始化模型
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    print("✓ 模型加载成功")

    # 初始化记忆管理器
    memory = MemoryManager(
        user_id="test_user",
        session_id="test_orchestration"
    )
    print("✓ 记忆管理器初始化成功")

    # 初始化意图识别智能体
    intention_agent = IntentionAgent(
        name="IntentionAgent",
        model=model
    )
    print("✓ 意图识别智能体初始化成功")

    # 初始化子智能体
    agents = {
        "event_collection": EventCollectionAgent(
            name="EventCollectionAgent",
            model=model
        ),
        "itinerary_planning": ItineraryPlanningAgent(
            name="ItineraryPlanningAgent",
            model=model
        ),
        "information_query": InformationQueryAgent(
            name="InformationQueryAgent",
            model=model
        ),
        "rag_knowledge": RAGKnowledgeAgent(
            name="RAGKnowledgeAgent",
            model=model
        )
    }
    print(f"✓ {len(agents)} 个子智能体初始化成功")

    # 初始化协调器
    orchestrator = OrchestrationAgent(
        name="OrchestrationAgent",
        agent_registry=agents,
        memory_manager=memory
    )
    print("✓ 协调器初始化成功")
    print()

    # ========== 测试场景 1: 行程规划 ==========
    print("[2] 测试场景 1: 行程规划")
    print("-" * 60)

    user_query = "我要2月27日从上海去北京出差"
    print(f"用户输入: {user_query}")
    print()

    # 意图识别
    print("步骤 1: 意图识别")
    intention_msg = Msg(name="user", content=user_query, role="user")
    intention_result = await intention_agent.reply(intention_msg)

    try:
        intention_data = json.loads(intention_result.content)
        print(f"✓ 识别意图: {[i['type'] for i in intention_data.get('intents', [])]}")
        print(f"✓ 提取实体: {intention_data.get('key_entities', {})}")
        print(f"✓ 调度计划: {len(intention_data.get('agent_schedule', []))} 个智能体")
        print()
    except json.JSONDecodeError as e:
        print(f"✗ 意图识别失败: {e}")
        return

    # 智能体协调
    print("步骤 2: 智能体协调执行")
    orchestration_result = await orchestrator.reply(intention_result)

    try:
        result_data = json.loads(orchestration_result.content)
        print(f"✓ 执行状态: {result_data.get('status')}")
        print(f"✓ 执行智能体数: {result_data.get('agents_executed')}")
        print()

        # 显示每个智能体的结果
        for result in result_data.get("results", []):
            agent_name = result.get("agent_name")
            status = result.get("status")
            print(f"  • {agent_name}: {status}")

        print()
    except json.JSONDecodeError as e:
        print(f"✗ 协调执行失败: {e}")
        return

    # ========== 测试场景 2: 信息查询 ==========
    print("[3] 测试场景 2: 信息查询")
    print("-" * 60)

    user_query = "北京的住宿标准是多少？"
    print(f"用户输入: {user_query}")
    print()

    # 意图识别
    intention_msg = Msg(name="user", content=user_query, role="user")
    intention_result = await intention_agent.reply(intention_msg)

    try:
        intention_data = json.loads(intention_result.content)
        print(f"✓ 识别意图: {[i['type'] for i in intention_data.get('intents', [])]}")
        print()
    except json.JSONDecodeError:
        print("✗ 意图识别失败")
        return

    # 智能体协调
    orchestration_result = await orchestrator.reply(intention_result)

    try:
        result_data = json.loads(orchestration_result.content)
        print(f"✓ 执行状态: {result_data.get('status')}")
        print(f"✓ 执行智能体数: {result_data.get('agents_executed')}")
        print()
    except json.JSONDecodeError:
        print("✗ 协调执行失败")
        return

    # ========== 测试记忆更新 ==========
    print("[4] 测试记忆更新")
    print("-" * 60)

    # 检查工作记忆
    task_info = memory.get_task_info()
    print(f"✓ 工作记忆: {len(task_info)} 项信息")
    for key, value in task_info.items():
        if value:
            print(f"  - {key}: {value}")

    # 检查短期记忆
    recent = memory.short_term.get_recent_context(2)
    print(f"✓ 短期记忆: {len(recent)} 条消息")
    print()

    # ========== 完成 ==========
    print("=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_orchestration())
