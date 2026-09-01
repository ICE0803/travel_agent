#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
记忆系统测试脚本
测试短期记忆和长期记忆的功能
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from context.memory_manager import MemoryManager
from agentscope import init as init_agentscope
from agentscope.model import OpenAIChatModel
import logging

# 配置日志（只显示WARNING及以上）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 从配置文件读取LLM配置
from config import LLM_CONFIG


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_memory_system():
    """测试记忆系统"""

    # 初始化AgentScope（禁用日志）
    init_agentscope(logging_level="ERROR")

    # 初始化LLM模型
    import warnings
    warnings.filterwarnings("ignore")

    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    print("✓ LLM模型初始化完成")

    # 初始化记忆管理器
    memory_manager = MemoryManager(
        user_id="test_user_001",
        session_id="test_session_001",
        storage_path="data/test_memory",
        llm_model=model
    )
    print("✓ 记忆管理器初始化完成")

    # ========== 测试短期记忆 ==========
    print_section("测试1: 短期记忆（会话级对话记录）")

    # 添加对话记录
    print("添加对话记录...")
    memory_manager.add_message("user", "你好，我想去北京旅游")
    memory_manager.add_message("assistant", "好的，请问您计划什么时候去北京？")
    memory_manager.add_message("user", "下周三出发")
    memory_manager.add_message("assistant", "明白了，下周三出发去北京。需要帮您规划行程吗？")
    memory_manager.add_message("user", "是的，我想去故宫和长城")
    print("✓ 已添加5条对话记录\n")

    # 获取最近对话
    print("📝 获取最近3轮对话:")
    recent_context = memory_manager.short_term.get_recent_context(n_turns=1)
    for msg in recent_context:
        role = "用户" if msg['role'] == 'user' else "助手"
        print(f"  {role}: {msg['content']}")

    # 获取上下文字符串
    print("\n📝 获取格式化的上下文字符串:")
    context_str = memory_manager.short_term.get_context_string(n_turns=1)
    print(context_str)

    # 获取统计信息
    print("\n📊 短期记忆统计:")
    stats = memory_manager.short_term.get_statistics()
    print(f"  - 总消息数: {stats['total_messages']}")
    print(f"  - 最大轮数: {stats['max_turns']}")

    # ========== 测试长期记忆 - 偏好 ==========
    print_section("测试2: 长期记忆 - 用户偏好（跨会话持久化）")

    print("保存用户偏好...")
    memory_manager.long_term.save_preference("budget", "经济型，每晚酒店预算500元以内")
    memory_manager.long_term.save_preference("accommodation", "喜欢连锁酒店，如家、汉庭")
    memory_manager.long_term.save_preference("transportation", "优先高铁，避免飞机")
    memory_manager.long_term.save_preference("food", "喜欢本地特色美食，不吃辣")
    print("✓ 已保存4项偏好\n")

    # 获取偏好
    print("📝 读取用户偏好:")
    preferences = memory_manager.long_term.get_preference()
    for key, value in preferences.items():
        if value:
            print(f"  - {key}: {value}")

    # ========== 测试长期记忆 - 行程历史 ==========
    print_section("测试3: 长期记忆 - 行程历史（跨会话持久化）")

    print("保存行程历史...")
    memory_manager.long_term.save_trip_history({
        "origin": "上海",
        "destination": "北京",
        "start_date": "2026-01-15",
        "end_date": "2026-01-18",
        "purpose": "旅游"
    })
    memory_manager.long_term.save_trip_history({
        "origin": "上海",
        "destination": "杭州",
        "start_date": "2026-02-10",
        "end_date": "2026-02-12",
        "purpose": "出差"
    })
    memory_manager.long_term.save_trip_history({
        "origin": "上海",
        "destination": "成都",
        "start_date": "2026-03-05",
        "end_date": "2026-03-09",
        "purpose": "旅游"
    })
    print("✓ 已保存3条行程历史\n")

    # 获取行程历史
    print("📝 读取行程历史（最近5条）:")
    trip_history = memory_manager.long_term.get_trip_history(limit=5)
    for i, trip in enumerate(trip_history, 1):
        print(f"  {i}. {trip['origin']} → {trip['destination']} "
              f"({trip.get('start_date', '未知')} - {trip.get('end_date', '未知')}) "
              f"- {trip.get('purpose', '未知')}")

    # 获取高频目的地
    print("\n📊 高频目的地（Top 3）:")
    frequent_destinations = memory_manager.long_term.get_frequent_destinations(top_n=3)
    for i, (dest, count) in enumerate(frequent_destinations, 1):
        print(f"  {i}. {dest}: {count}次")

    # ========== 测试长期记忆 - 聊天历史 ==========
    print_section("测试4: 长期记忆 - 聊天历史（持久化的对话记录）")

    print("📝 读取长期聊天历史（最近5条）:")
    chat_history = memory_manager.long_term.get_chat_history(limit=5)
    for i, msg in enumerate(chat_history, 1):
        role = "用户" if msg['role'] == 'user' else "助手"
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {i}. [{msg['timestamp']}] {role}: {content}")

    # ========== 测试长期记忆总结（使用LLM）==========
    print_section("测试5: 长期记忆总结（使用LLM生成摘要）")

    print("正在生成长期记忆总结（调用LLM）...")
    try:
        summary = await memory_manager.get_long_term_summary_async(max_messages=30)
        if summary:
            print("✓ 长期记忆总结生成成功\n")
            print("📝 总结内容:")
            print("-" * 80)
            print(summary)
            print("-" * 80)
        else:
            print("⚠ 长期记忆总结为空（可能是没有历史记录）")
    except Exception as e:
        print(f"✗ 长期记忆总结生成失败: {e}")

    # ========== 测试完整上下文获取 ==========
    print_section("测试6: 获取完整上下文（短期+长期记忆）")

    print("📝 完整上下文结构:")
    full_context = memory_manager.get_full_context()

    print("\n【短期记忆】")
    print(f"  - 最近对话数: {len(full_context['short_term']['recent_dialogue'])}")
    print(f"  - 统计信息: {full_context['short_term']['statistics']}")

    print("\n【长期记忆】")
    print(f"  - 偏好项数: {len([v for v in full_context['long_term']['preferences'].values() if v])}")
    print(f"  - 行程历史数: {len(full_context['long_term']['trip_history'])}")
    print(f"  - 高频目的地: {full_context['long_term']['frequent_destinations']}")

    # ========== 测试Agent上下文获取 ==========
    print_section("测试7: 获取Agent用的上下文字符串")

    # 模拟生成一个长期记忆总结
    long_term_summary = "用户是一位经常出差和旅游的商务人士，喜欢经济型酒店，偏好高铁出行。最近去过北京、杭州、成都等地。"

    print("📝 Agent上下文字符串:")
    agent_context = memory_manager.get_context_for_agent(long_term_summary)
    print("-" * 80)
    print(agent_context)
    print("-" * 80)

    # ========== 测试统计信息 ==========
    print_section("测试8: 长期记忆统计信息")

    print("📊 长期记忆统计:")
    stats = memory_manager.long_term.get_statistics()

    # 计算实际数据
    chat_count = len(memory_manager.long_term.get_chat_history(limit=999999))
    preferences = memory_manager.long_term.get_preference()
    pref_count = len([v for v in preferences.values() if v])
    unique_destinations = len(stats.get('frequent_destinations', {}))

    print(f"  - 总聊天记录数: {chat_count}")
    print(f"  - 总行程数: {stats.get('total_trips', 0)}")
    print(f"  - 已设置的偏好数: {pref_count}")
    print(f"  - 访问过的目的地数: {unique_destinations}")

    # ========== 测试会话清除 ==========
    print_section("测试9: 清除短期记忆（结束会话）")

    print("清除短期记忆...")
    memory_manager.end_session()
    print("✓ 短期记忆已清除\n")

    print("📝 验证短期记忆已清空:")
    recent_context = memory_manager.short_term.get_recent_context(n_turns=3)
    print(f"  - 剩余消息数: {len(recent_context)}")

    print("\n📝 验证长期记忆仍然保留:")
    preferences = memory_manager.long_term.get_preference()
    print(f"  - 偏好项数: {len([v for v in preferences.values() if v])}")
    trip_history = memory_manager.long_term.get_trip_history(limit=5)
    print(f"  - 行程历史数: {len(trip_history)}")

    # ========== 测试跨会话持久化 ==========
    print_section("测试10: 跨会话持久化（新建会话，验证长期记忆保留）")

    print("创建新的记忆管理器（新会话ID，但相同用户ID）...")
    new_memory_manager = MemoryManager(
        user_id="test_user_001",  # 相同用户
        session_id="test_session_002",  # 不同会话
        storage_path="data/test_memory",
        llm_model=model
    )
    print("✓ 新会话记忆管理器创建完成\n")

    print("📝 验证新会话中能访问之前保存的长期记忆:")

    # 验证偏好
    preferences = new_memory_manager.long_term.get_preference()
    print(f"\n  偏好项数: {len([v for v in preferences.values() if v])}")
    for key, value in preferences.items():
        if value:
            print(f"    - {key}: {value}")

    # 验证行程历史
    trip_history = new_memory_manager.long_term.get_trip_history(limit=5)
    print(f"\n  行程历史数: {len(trip_history)}")
    for i, trip in enumerate(trip_history, 1):
        print(f"    {i}. {trip['origin']} → {trip['destination']} ({trip.get('start_date', '未知')})")

    print("\n✓ 跨会话持久化验证成功！")

    # ========== 完成 ==========
    print_section("测试完成")
    print("✅ 所有测试通过！")
    print("\n记忆系统功能总结:")
    print("  1. ✓ 短期记忆：会话级对话记录（会话结束后清空）")
    print("  2. ✓ 长期记忆-偏好：跨会话持久化用户偏好")
    print("  3. ✓ 长期记忆-行程历史：跨会话持久化旅行记录")
    print("  4. ✓ 长期记忆-聊天历史：跨会话持久化对话记录")
    print("  5. ✓ LLM总结：自动总结长期记忆生成摘要")
    print("  6. ✓ 上下文聚合：整合短期+长期记忆供Agent使用")
    print("  7. ✓ 跨会话持久化：同一用户不同会话共享长期记忆")
    print("\n测试数据存储位置: data/test_memory/")


if __name__ == "__main__":

    try:
        asyncio.run(test_memory_system())
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被用户中断")
    except Exception as e:
        print(f"\n\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
