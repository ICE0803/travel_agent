"""
CLI 自动问答测试 - 直接运行生成 QA 对文档
Usage: python tests/test_cli_qa.py
使用 cli.AligoCLI._display_results 统一打印结果，避免重复逻辑。
"""
import sys
import asyncio
import json
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def capture_display_results(result_data: dict) -> str:
    """调用 cli 的 _display_results，将输出捕获为字符串（无 ANSI 颜色）。"""
    from rich.console import Console
    from cli import AligoCLI

    capture = StringIO()
    console = Console(file=capture, force_terminal=False, no_color=True)
    cli = AligoCLI()
    cli.console = console
    cli._display_results(result_data)
    return capture.getvalue().strip()


# 测试问题 - 覆盖所有功能
QUESTIONS = [
    "出差住宿标准是多少？",
    "如何报销差旅费用？需要哪些材料？",
    "我从3月11日从北京出发，在杭州出差一周，3月18日返回北京，帮我安排行程",
    "机票应该提前多久预订？有什么注意事项？",
    "我偏好住万豪酒店和希尔顿，喜欢坐国航和东航，座位要靠窗，家住北京朝阳区，请记住",
    "杭州下周的天气怎么样？",
    "从北京到深圳出差，住宿和交通标准分别是多少？",
    "查询我最近的差旅记录",
    "航班取消了怎么办？紧急情况联系谁？",
    "我要去上海出差5天，帮我规划详细行程",
]


async def main():
    """运行测试并生成文档"""
    print("="*70)
    print("CLI QA 测试 - 开始")
    print("="*70)

    # 初始化系统 - 完全按照 CLI 的方式
    print("\n[1/3] 初始化系统...")

    from config import LLM_CONFIG
    from config_agentscope import init_agentscope
    from agentscope.model import OpenAIChatModel
    from context.memory_manager import MemoryManager
    from agents.intention_agent import IntentionAgent
    from agents.orchestration_agent import OrchestrationAgent
    from agents.lazy_agent_registry import LazyAgentRegistry
    # Removed manual imports of sub-agents as they are now dynamically loaded
    from agentscope.message import Msg

    # 初始化 AgentScope
    init_agentscope()

    # 初始化模型
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"]},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )

    # 初始化记忆管理器
    memory_manager = MemoryManager(
        user_id="test_user",
        session_id="test_session",
        llm_model=model
    )

    # 初始化意图识别智能体
    intention_agent = IntentionAgent(
        name="IntentionAgent",
        model=model
    )

    # 使用懒加载注册器
    agent_cache = {}
    lazy_registry = LazyAgentRegistry(model, agent_cache, memory_manager)
    
    # 无需手动实例化子 Agent，OrchestrationAgent 会通过 LazyRegistry 自动加载
    # agent_cache["memory_query"] = ... (删除手动加载)
    # agent_cache["preference"] = ... (删除手动加载)

    # 初始化协调器 - 完全按照 CLI 的参数
    orchestrator = OrchestrationAgent(
        name="OrchestrationAgent",
        agent_registry=lazy_registry,
        memory_manager=memory_manager
    )

    print("✓ 系统初始化完成")

    # 运行测试
    print(f"\n[2/3] 运行 {len(QUESTIONS)} 个测试问题...")
    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n问题 {i}/{len(QUESTIONS)}: {question}")
        start = datetime.now()

        try:
            # 1. 意图识别
            context_messages = [Msg(name="user", content=question, role="user")]
            intention_result = await intention_agent.reply(context_messages)

            # 2. 编排执行
            orchestration_result = await orchestrator.reply(intention_result)

            # 3. 解析结果并用 CLI 同一套逻辑打印（捕获为字符串）
            result_data = json.loads(orchestration_result.content)
            duration = (datetime.now() - start).total_seconds()
            answer = capture_display_results(result_data)

            results.append({
                "num": i,
                "question": question,
                "answer": answer,
                "status": "success",
                "duration": round(duration, 2)
            })
            print(f"✓ 完成 ({duration:.1f}s)")

        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            results.append({
                "num": i,
                "question": question,
                "answer": f"错误: {str(e)}",
                "status": "error",
                "duration": round(duration, 2)
            })
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)

    # 保存结果
    print("\n[3/3] 保存结果...")
    save_results(results)
    print("✓ 完成")

    # 打印统计
    success = sum(1 for r in results if r["status"] == "success")
    total_time = sum(r["duration"] for r in results)
    print(f"\n{'='*70}")
    print(f"统计: {success}/{len(results)} 成功, 总耗时 {total_time:.1f}s")
    print(f"{'='*70}\n")


def save_results(results: List[Dict]):
    """保存结果为 Markdown"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = project_root / "tests" / "results" / f"qa_test_{timestamp}.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        # 标题
        f.write(f"# CLI QA 测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 统计
        success = sum(1 for r in results if r["status"] == "success")
        total_time = sum(r["duration"] for r in results)
        f.write(f"## 统计\n\n")
        f.write(f"- 总问题: {len(results)}\n")
        f.write(f"- 成功: {success} ({success/len(results)*100:.1f}%)\n")
        f.write(f"- 失败: {len(results)-success}\n")
        f.write(f"- 总耗时: {total_time:.1f}秒\n")
        f.write(f"- 平均: {total_time/len(results):.1f}秒/问题\n\n")

        # QA 对
        f.write(f"## QA 对\n\n")
        for r in results:
            icon = "✅" if r["status"] == "success" else "❌"
            f.write(f"### {icon} Q{r['num']}: {r['question']}\n\n")
            f.write(f"**耗时**: {r['duration']}秒\n\n")
            f.write(f"**回答**:\n\n```\n{r['answer']}\n```\n\n")
            f.write(f"---\n\n")

    print(f"结果已保存: {output}")


if __name__ == "__main__":
    asyncio.run(main())
