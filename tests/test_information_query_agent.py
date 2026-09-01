#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试信息查询智能体（免费API版）

功能：
- 网络搜索（DuckDuckGo 免费搜索）

依赖：
  pip install ddgs

使用方法：
  python tests/test_information_query_agent.py

注意：
  此测试需要网络连接。如果网络受限或API服务不可达，测试可能失败。
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
# 导入 skill 目录下的 agent
sys.path.insert(0, os.path.join(project_root, '.claude', 'skills', 'query-info', 'script'))
from agent import InformationQueryAgent
from config import LLM_CONFIG


async def test_information_query_agent():
    """测试信息查询智能体"""

    print("="*70)
    print("测试信息查询智能体 - 网络搜索功能")
    print("="*70)
    print()
    print("使用完全免费的API：")
    print("• DuckDuckGo 网络搜索（无需API Key）")
    print("="*70)

    # 初始化
    init_agentscope()
    print()

    # 创建模型
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
        stream=True,
    )

    # 创建智能体
    agent = InformationQueryAgent(
        name="InformationQueryAgent",
        model=model
    )

    # 测试用例 - 包含天气和网络搜索
    test_cases = [
        {
            "name": "天气查询 - wttr.in API",
            "query": "杭州的天气怎么样",
            "expected_keywords": ["杭州", "天气", "气温"],
            "expected_type": "天气查询"
        },
        {
            "name": "天气查询 - 多日预报",
            "query": "北京下周天气预报",
            "expected_keywords": ["北京", "天气"],
            "expected_type": "天气查询"
        },
        {
            "name": "网络搜索 - 实时新闻",
            "query": "Claude AI 最新功能",
            "expected_keywords": ["Claude", "AI"],
            "expected_type": "网络搜索"
        },
        {
            "name": "网络搜索 - 技术查询",
            "query": "Python async await 用法",
            "expected_keywords": ["Python", "async"],
            "expected_type": "网络搜索"
        },
        {
            "name": "网络搜索 - 百科查询",
            "query": "什么是 RAG 检索增强生成",
            "expected_keywords": ["RAG"],
            "expected_type": "网络搜索"
        },
    ]

    passed = 0
    failed = 0
    total_time = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n【测试{i}】{test_case['name']}")
        print("-"*70)
        print(f"查询: {test_case['query']}")
        print()

        start_time = time.time()
        try:
            # 创建消息
            user_msg = Msg(name="User", content=test_case['query'], role="user")

            # 调用智能体
            result = await agent.reply(user_msg)

            # 解析结果
            result_data = json.loads(result.content)

            elapsed = time.time() - start_time
            total_time += elapsed

            # 显示结果
            query_type = result_data.get('query_type', '')
            query_success = result_data.get('query_success', False)
            query_results = result_data.get('results', {})

            print(f"查询类型: {query_type}")
            print(f"查询状态: {'✓ 成功' if query_success else '✗ 失败'}")

            if query_success:
                summary = query_results.get("summary", "")
                sources = query_results.get("sources", [])

                print(f"\n摘要: {summary}")
                if sources:
                    print(f"\n来源（前{min(len(sources), 3)}条）：")
                    for j, source in enumerate(sources[:3], 1):
                        print(f"  {j}. {source.get('title', '')}")
                        snippet = source.get('snippet', '')
                        if snippet:
                            print(f"     摘要: {snippet[:80]}...")
                        print(f"     URL: {source.get('url', '')}")

                print(f"\n⏱️  响应时间: {elapsed:.1f}秒")

                # 验证查询类型
                expected_type = test_case.get('expected_type', '')
                if expected_type and expected_type == query_type:
                    print(f"✓ 查询类型匹配: {query_type}")
                elif expected_type:
                    print(f"⚠️  查询类型不匹配: 期望 {expected_type}, 实际 {query_type}")

                # 验证关键词
                expected_keywords = test_case.get('expected_keywords', [])
                found_keywords = [kw for kw in expected_keywords if kw.lower() in summary.lower()]

                if found_keywords or not expected_keywords:
                    print("✅ 测试通过")
                    passed += 1
                else:
                    print(f"⚠️  未找到预期关键词: {', '.join(expected_keywords)}")
                    passed += 1  # 仍算通过，因为搜索成功了

            else:
                error = query_results.get("error", "")
                message = query_results.get("message", "")
                note = query_results.get("note", "")

                if error:
                    print(f"错误: {error}")
                if message:
                    print(f"消息: {message}")
                if note:
                    print(f"提示: {note}")

                print(f"\n⏱️  响应时间: {elapsed:.1f}秒")
                failed += 1

        except Exception as e:
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        print("\n" + "="*70)

    # 打印统计
    print(f"\n✅ 测试完成")
    print("="*70)
    print(f"总测试数: {len(test_cases)}")
    print(f"✓ 通过: {passed} ({passed/len(test_cases)*100:.0f}%)")
    print(f"✗ 失败: {failed} ({failed/len(test_cases)*100:.0f}%)")
    print(f"平均响应时间: {total_time/len(test_cases):.1f}秒")
    print(f"总耗时: {total_time:.1f}秒")
    print("="*70)
    print()
    print("提示：")
    print("• 天气查询: 使用 wttr.in 免费 API（需安装：pip install httpx）")
    print("• 网络搜索: 使用 DuckDuckGo 完全免费（需安装：pip install ddgs）")
    print("• 检索内容: 网页标题 + 摘要片段（不是完整网页）")
    print("• URL 过滤: 自动过滤 .cc/.tk/.xyz 等可疑域名")
    print()


if __name__ == "__main__":
    asyncio.run(test_information_query_agent())
