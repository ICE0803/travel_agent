"""
记忆查询智能体 MemoryQueryAgent
职责：基于用户的长期记忆回答历史相关问题

核心功能：
1. 查询用户旅行历史（trip_history）
2. 查询用户偏好（preferences）
3. 查询历史对话记录（chat_history）
4. 使用LLM基于记忆生成自然语言回答

适用场景：
- 用户询问："我去过哪些地方？"
- 用户询问："我之前说过什么偏好？"
- 用户询问："我上次去北京是什么时候？"
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict
import json
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

logger = logging.getLogger(__name__)


class MemoryQueryAgent(AgentBase):
    """记忆查询智能体 - 基于长期记忆回答用户问题"""

    def __init__(
        self,
        name: str = "MemoryQueryAgent",
        model=None,
        memory_manager=None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        处理记忆查询请求

        Args:
            x: 输入消息，包含用户查询和上下文

        Returns:
            Msg: 基于记忆的回答
        """
        if x is None:
            return Msg(name=self.name, content=json.dumps({}), role="assistant")

        # 解析输入
        if isinstance(x, list):
            input_content = x[-1].content if x else "{}"
        else:
            input_content = x.content

        try:
            input_data = json.loads(input_content) if isinstance(input_content, str) else input_content
        except json.JSONDecodeError:
            input_data = {"context": {"rewritten_query": str(input_content)}}

        # 获取用户查询
        context = input_data.get("context", {})
        user_query = context.get("rewritten_query", "")
        if not user_query:
            # 尝试从 recent_dialogue 获取最后一条用户消息
            recent_dialogue = context.get("recent_dialogue", [])
            if recent_dialogue:
                for msg in reversed(recent_dialogue):
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break

        if not user_query:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "error",
                    "message": "无法获取用户查询"
                }),
                role="assistant"
            )

        # 获取长期记忆
        trip_history = []
        preferences = {}
        chat_summary = ""

        if self.memory_manager:
            # 获取旅行历史（最近50条）
            trip_history = self.memory_manager.long_term.get_trip_history(limit=50)

            # 获取用户偏好
            preferences = self.memory_manager.long_term.get_preference()

            # 获取历史对话摘要（如果有LLM的话）
            try:
                chat_summary = await self.memory_manager.get_long_term_summary_async(max_messages=30)
            except Exception as e:
                logger.warning(f"Failed to get chat summary: {e}")
                chat_summary = ""

        # 格式化旅行历史
        trip_text = self._format_trip_history(trip_history)

        # 格式化偏好
        pref_text = self._format_preferences(preferences)

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("memory-query")
        if not skill_instruction:
            skill_instruction = "请基于用户的历史记忆回答问题，如无相关记录请诚实说明。"

        # 构建 prompt
        prompt = f"""你是一个个人记忆助手，请基于用户的历史记忆回答问题。

【用户问题】
{user_query}

【用户旅行历史】
{trip_text}

【用户偏好】
{pref_text}

【历史对话摘要】
{chat_summary if chat_summary else "（暂无历史对话摘要）"}

【任务说明】
{skill_instruction}
"""

        try:
            # 调用LLM生成回答
            response = await self.model([
                {"role": "system", "content": "你是一个个人记忆助手，帮助用户查询和理解他们的历史记录。"},
                {"role": "user", "content": prompt}
            ])

            # 获取响应文本 - 处理异步生成器
            answer = ""
            if hasattr(response, '__aiter__'):
                # 异步生成器，需要迭代获取内容
                async for chunk in response:
                    if isinstance(chunk, str):
                        answer = chunk
                    elif hasattr(chunk, 'content'):
                        if isinstance(chunk.content, str):
                            answer = chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    answer = item.get('text', '')
            elif hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, dict) and 'content' in response:
                answer = response['content']
            else:
                answer = str(response) if response else "无法生成回答"

            if not answer:
                answer = "无法基于记忆生成回答"

            logger.info(f"Memory query answered: {user_query[:50]}")

            result = {
                "status": "success",
                "query": user_query,
                "answer": answer,
                "memory_sources": {
                    "trip_count": len(trip_history),
                    "has_preferences": any(v for v in preferences.values() if v),
                    "has_chat_summary": bool(chat_summary)
                }
            }

            return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

        except Exception as e:
            logger.error(f"Memory query failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "error",
                    "message": f"记忆查询失败: {str(e)}",
                    "query": user_query
                }),
                role="assistant"
            )

    def _format_trip_history(self, trip_history: List[Dict]) -> str:
        """格式化旅行历史"""
        if not trip_history:
            return "（暂无旅行记录）"

        lines = []
        for i, trip in enumerate(trip_history, 1):
            origin = trip.get("origin", "未知")
            destination = trip.get("destination", "未知")
            start_date = trip.get("start_date", "")
            end_date = trip.get("end_date", "")
            purpose = trip.get("purpose", "旅游")
            timestamp = trip.get("timestamp", "")

            if start_date and end_date:
                lines.append(f"{i}. {origin} → {destination} ({start_date} 至 {end_date}) - {purpose}")
            elif start_date:
                lines.append(f"{i}. {origin} → {destination} ({start_date}) - {purpose}")
            else:
                lines.append(f"{i}. {origin} → {destination} - {purpose} (记录时间: {timestamp})")

        return "\n".join(lines)

    def _format_preferences(self, preferences: Dict) -> str:
        """格式化用户偏好"""
        if not preferences or not any(v for v in preferences.values() if v):
            return "（暂无偏好记录）"

        lines = []
        pref_names = {
            "budget": "预算偏好",
            "accommodation": "住宿偏好",
            "transportation": "交通偏好",
            "food": "餐饮偏好",
            "activity": "活动偏好",
            "other": "其他偏好"
        }

        for key, value in preferences.items():
            if value and key in pref_names:
                lines.append(f"- {pref_names[key]}: {value}")

        return "\n".join(lines) if lines else "（暂无偏好记录）"
