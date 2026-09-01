"""
执行行程规划任务
处理用户的出差行程规划需求
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add parent directory to ensure module imports work
parent_dir = Path(__file__).parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

# Import from absolute paths
import config_agentscope
import config
from agents.intention_agent import IntentionAgent
from agents.orchestration_agent import OrchestrationAgent

LLM_CONFIG = config.LLM_CONFIG
init_agentscope = config_agentscope.init_agentscope

# Import agents
import importlib.util

def load_agent_module(skill_name, agent_class_name):
    """Dynamically load agent module from skill directory"""
    # Navigate from current file's location to the skills directory
    # Current: /path/to/shanglv/.claude/skills/plan-trip/script/plan_trip_execution.py
    # Target: /path/to/shanglv/.claude/skills/<skill_name>/script/agent.py
    current_file = Path(__file__).resolve()
    skills_dir = current_file.parent.parent.parent  # Go up to .claude/skills/
    agent_path = skills_dir / skill_name / "script" / "agent.py"

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    spec = importlib.util.spec_from_file_location(f"{skill_name}_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, agent_class_name)


async def plan_trip(user_query: str):
    """
    行程规划主流程

    Args:
        user_query: 用户输入的行程需求

    Returns:
        dict: 行程规划结果
    """
    # 初始化 AgentScope
    init_agentscope()

    # 创建模型
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": 60},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 8192),
    )

    print("✓ 模型初始化完成")

    # 创建用户消息
    user_msg = Msg(name="user", content=user_query, role="user")

    # 1. 意图识别
    print("\n[步骤 1/3] 正在识别意图...")
    intention_agent = IntentionAgent(name="IntentionAgent", model=model)
    intention_result = await intention_agent.reply(user_msg)
    intention_data = json.loads(intention_result.content)

    print(f"✓ 识别到意图: {', '.join([i['type'] for i in intention_data.get('intents', [])])}")
    print(f"✓ 改写后的查询: {intention_data.get('rewritten_query', user_query)}")

    # 2. 加载子智能体
    print("\n[步骤 2/3] 正在加载智能体...")
    EventCollectionAgent = load_agent_module("event-collection", "EventCollectionAgent")
    ItineraryPlanningAgent = load_agent_module("plan-trip", "ItineraryPlanningAgent")

    # 创建智能体实例
    event_agent = EventCollectionAgent(name="EventCollectionAgent", model=model)
    itinerary_agent = ItineraryPlanningAgent(name="ItineraryPlanningAgent", model=model)

    # 创建协调器并注册智能体
    orchestrator = OrchestrationAgent(
        name="OrchestrationAgent",
        agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": itinerary_agent
        }
    )

    print("✓ 智能体加载完成")

    # 3. 执行协调流程
    print("\n[步骤 3/3] 正在生成行程规划...")
    orchestration_input = Msg(
        name="user",
        content=json.dumps(intention_data, ensure_ascii=False),
        role="user"
    )

    orchestration_result = await orchestrator.reply(orchestration_input)
    result_data = json.loads(orchestration_result.content)

    print("✓ 行程规划完成")

    # 4. 提取并返回行程规划
    if result_data.get("status") == "completed":
        # 从结果中提取行程规划
        for agent_result in result_data.get("results", []):
            if agent_result.get("agent_name") == "itinerary_planning":
                itinerary_data = agent_result.get("data", {})
                return itinerary_data

    return result_data


async def main():
    """主函数"""
    user_query = "我从3月11日从北京出发，在杭州出差一周，3月18日返回北京，帮我安排行程"

    print("=" * 60)
    print("Aligo 行程规划系统")
    print("=" * 60)
    print(f"\n用户需求: {user_query}\n")

    try:
        result = await plan_trip(user_query)

        print("\n" + "=" * 60)
        print("行程规划结果")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 格式化输出行程
        if "itinerary" in result:
            itinerary = result["itinerary"]
            print("\n" + "=" * 60)
            print(f"【{itinerary.get('title', '行程规划')}】")
            print("=" * 60)
            print(f"总行程: {itinerary.get('duration', 'N/A')}")
            print(f"路线: {itinerary.get('route', 'N/A')}")
            print()

            # 每日行程
            for day_plan in itinerary.get("daily_plans", []):
                print(f"Day {day_plan['day']} - {day_plan.get('date', 'N/A')} ({day_plan.get('city', 'N/A')})")
                print(f"主题: {day_plan.get('theme', 'N/A')}")
                print()

                for activity in day_plan.get("activities", []):
                    print(f"  [{activity.get('time', 'N/A')}] {activity.get('location', 'N/A')}")
                    print(f"  {activity.get('description', 'N/A')}")
                    if activity.get("transport"):
                        print(f"  交通: {activity['transport']}")
                    print()

                # 用餐信息
                meals = day_plan.get("meals", {})
                if meals:
                    print(f"  用餐:")
                    if meals.get("lunch"):
                        print(f"    午餐: {meals['lunch']}")
                    if meals.get("dinner"):
                        print(f"    晚餐: {meals['dinner']}")
                    print()

            # 注意事项
            notes = itinerary.get("notes", [])
            if notes:
                print("注意事项:")
                for note in notes:
                    print(f"  • {note}")
                print()

            # 预算
            if itinerary.get("estimated_budget"):
                print(f"预计费用: {itinerary['estimated_budget']}")

    except Exception as e:
        print(f"\n❌ 规划失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
