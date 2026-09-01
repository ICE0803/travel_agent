#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试RAG知识库智能体

使用方法：
  # 先初始化知识库
  python scripts/init_knowledge_base.py

  # 然后运行测试
  python tests/test_rag_agent.py
"""
import sys
import os
import asyncio
import json
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from agents.rag_knowledge_agent import RAGKnowledgeAgent
from config import LLM_CONFIG


async def test_rag_agent():
    """测试RAG知识库智能体"""

    print("="*70)
    print("RAG知识库智能体测试")
    print("="*70)
    print()

    # 初始化
    print("初始化 AgentScope...")
    init_agentscope()
    print("✓ AgentScope initialized")
    print()

    print("创建模型...")
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    print("✓ 模型创建成功")
    print()

    print("创建RAG Knowledge Agent...")
    agent = RAGKnowledgeAgent(
        name="RAGKnowledgeAgent",
        model=model,
        knowledge_base_path="./data/rag_knowledge",
        collection_name="business_travel_knowledge",
        top_k=3
    )

    if not agent.initialized:
        print("❌ RAG Agent未正确初始化")
        print("   请先运行: python scripts/init_knowledge_base.py")
        return

    print("✓ RAG Agent创建成功")
    stats = agent.get_stats()
    if stats["status"] == "success":
        print(f"   - 知识库文档数: {stats['total_documents']}")
        print(f"   - 存储路径: {stats['knowledge_base_path']}")
    print()

    # 测试用例 - 涵盖8个文档类别
    test_cases = [
        {
            "name": "住宿标准查询",
            "query": "北京出差的住宿标准是多少？",
            "expected": ["500", "一线"]
        },
        {
            "name": "报销规定查询",
            "query": "差旅费用应该在什么时候报销？",
            "expected": ["30", "天"]
        },
        {
            "name": "预订建议查询",
            "query": "机票应该提前多久预订比较好？",
            "expected": ["7", "14"]
        },
        {
            "name": "FAQ查询",
            "query": "出差可以携带家属吗？",
            "expected": ["不可以"]
        },
        {
            "name": "紧急情况查询",
            "query": "航班延误了应该怎么办？",
            "expected": ["改签", "凭证"]
        },
        {
            "name": "城市指南查询",
            "query": "北京有哪些机场？",
            "expected": ["首都", "大兴"]
        },
        {
            "name": "平台功能查询",
            "query": "阿里商旅平台有哪些功能？",
            "expected": ["申请", "预订"]
        },
        {
            "name": "环保建议查询",
            "query": "出差怎么做到环保？",
            "expected": ["高铁", "公共交通"]
        },
    ]

    passed = 0
    failed = 0
    total_time = 0

    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "="*70)
        print(f"测试 {i}/{len(test_cases)}: {test_case['name']}")
        print("="*70)
        print(f"用户查询: {test_case['query']}")
        print()

        start_time = time.time()
        try:
            # 创建消息
            user_msg = Msg(name="User", content=test_case['query'], role="user")

            # 调用RAG Agent
            result = await agent(user_msg)

            # 解析结果
            result_data = json.loads(result.content)

            elapsed = time.time() - start_time
            total_time += elapsed

            # 显示结果
            status = result_data.get('status', 'unknown')
            print(f"【状态】{status}")
            print()

            if status == 'success':
                answer = result_data.get('answer', '')
                print("【AI回答】")
                print("-" * 60)
                print(answer)
                print("-" * 60)
                print()

                # 验证关键词
                found_keywords = [kw for kw in test_case.get('expected', []) if kw in answer]
                if found_keywords:
                    print(f"✓ 关键词验证通过: {', '.join(found_keywords)}")
                    success = True
                else:
                    print(f"⚠️  未找到预期关键词: {', '.join(test_case.get('expected', []))}")
                    success = False

                # 显示检索到的文档
                retrieved_docs = result_data.get('retrieved_documents', [])
                if retrieved_docs:
                    print(f"✓ 检索到 {len(retrieved_docs)} 篇参考文档")
                    for j, doc in enumerate(retrieved_docs, 1):
                        metadata = doc.get('metadata', {})
                        print(f"  [{j}] {metadata.get('title', 'Unknown')} ({metadata.get('category', 'N/A')})")

                print(f"\n⏱️  响应时间: {elapsed:.1f}秒")

                if success:
                    print("✅ 测试通过")
                    passed += 1
                else:
                    print("⚠️  测试未完全通过")
                    failed += 1

            else:
                error_msg = result_data.get('message', result_data.get('answer', ''))
                print(f"❌ 查询失败: {error_msg}")
                failed += 1

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        print()

    # 打印统计
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"总测试数: {len(test_cases)}")
    print(f"✓ 通过: {passed} ({passed/len(test_cases)*100:.0f}%)")
    print(f"✗ 失败: {failed} ({failed/len(test_cases)*100:.0f}%)")
    print(f"平均响应时间: {total_time/len(test_cases):.1f}秒")
    print(f"总耗时: {total_time:.1f}秒")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_rag_agent())
    print("\n测试完成！")
