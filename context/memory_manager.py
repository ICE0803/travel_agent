"""
记忆管理器 (Memory Manager)
统一管理两层记忆，提供简单的API

架构：
- 短期记忆：Redis LIST（会话级滑动窗口，TTL 1 小时）
- 长期记忆：PostgreSQL（跨会话持久化，含偏好 Write-Through 缓存）
- LLM 总结缓存：Redis（TTL 30 分钟，带 watermark 版本号防止脏读）
"""
from typing import Dict, Any, List, Optional
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .backends import get_redis
from config import STORAGE_CONFIG
import logging
import json

logger = logging.getLogger(__name__)

SUMMARY_TTL = STORAGE_CONFIG.get("ttl", {}).get("summary", 1800)


class MemoryManager:
    """
    记忆管理器：统一管理两层记忆
    - 短期记忆：最近对话（会话级，Redis）
    - 长期记忆：用户偏好和历史（跨会话，PostgreSQL）
    """

    def __init__(self, user_id: str, session_id: str, storage_path: str = "data/memory", llm_model=None):
        """
        初始化记忆管理器

        Args:
            user_id: 用户ID
            session_id: 会话ID
            storage_path: 长期记忆存储路径（仅 JSON 降级模式使用）
            llm_model: LLM模型实例（用于总结长期记忆）
        """
        self.user_id = user_id
        self.session_id = session_id
        self.llm_model = llm_model

        # 初始化两层记忆
        # 注意：session_id 必须传进去，否则所有会话会共用同一个 Redis key
        self.short_term = ShortTermMemory(max_turns=10, session_id=session_id)
        self.long_term = LongTermMemory(user_id, storage_path)

        # LLM 总结缓存
        self._redis = get_redis()
        self._sum_key = f"user:{user_id}:summary"
        self.sum_hit_key = f"{self._sum_key}:hit"
        self.sum_miss_key = f"{self._sum_key}:miss"

        if self._redis is not None:
            logger.info(
                "Memory manager initialized for user %s, session %s (redis cache on)",
                user_id, session_id,
            )
        else:
            logger.info(
                "Memory manager initialized for user %s, session %s (redis cache off)",
                user_id, session_id,
            )

    # ========== 短期记忆操作 ==========

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到短期记忆和长期记忆

        Args:
            role: 角色 (user/assistant)
            content: 消息内容
            metadata: 元数据
        """
        # 添加到短期记忆（当前会话，Redis）
        self.short_term.add_message(role, content, metadata)

        # 同时添加到长期记忆（跨会话持久化，PostgreSQL）
        self.long_term.add_chat_message(role, content, self.session_id)

    # ========== 长期记忆操作 ==========
    # 注意：大部分方法直接使用 self.short_term 和 self.long_term 即可，无需封装

    # ========== 综合查询 ==========

    def get_full_context(self) -> Dict[str, Any]:
        """
        获取完整上下文（两层记忆）

        Returns:
            完整上下文字典
        """
        return {
            "short_term": {
                "recent_dialogue": self.short_term.get_recent_context(5),
                "context_string": self.short_term.get_context_string(5),
                "statistics": self.short_term.get_statistics()
            },
            "long_term": {
                "preferences": self.long_term.get_preference(),
                "chat_history": self.long_term.get_chat_history(10),
                "trip_history": self.long_term.get_trip_history(5),
                "frequent_destinations": self.long_term.get_frequent_destinations(3),
                "statistics": self.long_term.get_statistics()
            }
        }

    def get_context_for_agent(self, long_term_summary: str = None) -> str:
        """
        获取用于Agent的上下文字符串

        Args:
            long_term_summary: 长期记忆总结（可选，需提前调用 get_long_term_summary_async）

        Returns:
            格式化的上下文字符串
        """
        lines = []

        # 长期记忆总结（历史会话）
        if long_term_summary:
            lines.append("【历史会话总结】")
            lines.append(long_term_summary)
            lines.append("")

        # 用户偏好（走 Redis 缓存）
        prefs = self.long_term.get_preference()
        has_prefs = any(v for v in prefs.values() if v)
        if has_prefs:
            lines.append("【用户偏好】")
            for key, value in prefs.items():
                if value:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        # 短期记忆（当前会话）
        context_str = self.short_term.get_context_string(3)
        if context_str != "无历史对话":
            lines.append("【当前会话对话】")
            lines.append(context_str)
            lines.append("")

        return "\n".join(lines) if lines else "无上下文信息"

    # ========== 会话管理 ==========

    def end_session(self):
        """结束会话（只清短期记忆，长期记忆保留）"""
        self.short_term.clear()
        logger.info(f"Session ended: {self.session_id}")

    # ========== 缓存统计 ==========

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存命中率统计（用于 status 命令展示）

        Returns:
            {"enabled": bool, "preferences": {...}, "summary": {...}}
        """
        if self._redis is None:
            return {"enabled": False}

        def _pair(hit_key: str, miss_key: str) -> Dict[str, Any]:
            try:
                hit = int(self._redis.get(hit_key) or 0)
                miss = int(self._redis.get(miss_key) or 0)
            except Exception as e:
                logger.warning("读取缓存计数失败: %s", e)
                hit = miss = 0
            total = hit + miss
            return {
                "hit": hit,
                "miss": miss,
                "total": total,
                "rate": round(hit / total * 100, 1) if total else 0.0,
            }

        # 偏好缓存的计数键由 LongTermMemory 维护，这里按同一约定拼接
        pref_hit = getattr(self.long_term, "pref_hit_key", f"user:{self.user_id}:prefs:hit")
        pref_miss = getattr(self.long_term, "pref_miss_key", f"user:{self.user_id}:prefs:miss")

        return {
            "enabled": True,
            "preferences": _pair(pref_hit, pref_miss),
            "summary": _pair(self.sum_hit_key, self.sum_miss_key),
        }

    # ========== LLM 总结（带 Redis 缓存） ==========

    async def get_long_term_summary_async(self, max_messages: int = 50) -> str:
        """
        使用LLM总结长期聊天历史（异步版本，带 Redis 缓存）

        缓存策略：
        - key: user:{user_id}:summary
        - value: {"watermark": <chat_history 当前最大 id>, "summary": "..."}
        - 读取时比对 watermark，不一致说明有新消息 → 缓存失效，重新生成
          这样解决"用户刚说完新偏好，总结还是旧的"的脏读问题
        - TTL 由 STORAGE_CONFIG["ttl"]["summary"] 控制（默认 30 分钟）

        Args:
            max_messages: 最多总结的消息数量

        Returns:
            总结后的文本
        """
        if not self.llm_model:
            return ""

        watermark = self._get_history_watermark()

        # 1) 尝试读缓存
        if self._redis is not None and watermark is not None:
            try:
                raw = self._redis.get(self._sum_key)
                if raw is not None:
                    payload = json.loads(raw)
                    if payload.get("watermark") == watermark:
                        self._redis.incr(self.sum_hit_key)
                        logger.debug("长期记忆总结命中缓存 (watermark=%s)", watermark)
                        return payload.get("summary", "")
                self._redis.incr(self.sum_miss_key)
            except Exception as e:
                logger.warning("总结缓存读取失败，直接重新生成: %s", e)

        # 2) 未命中 → 调 LLM 生成
        summary = await self._generate_summary(max_messages)

        # 3) 回填缓存（只缓存成功且非空的结果，避免把 LLM 失败也缓存 30 分钟）
        if summary and self._redis is not None and watermark is not None:
            try:
                self._redis.setex(
                    self._sum_key,
                    SUMMARY_TTL,
                    json.dumps(
                        {"watermark": watermark, "summary": summary},
                        ensure_ascii=False,
                    ),
                )
                logger.debug("长期记忆总结已写入缓存 (TTL=%ss)", SUMMARY_TTL)
            except Exception as e:
                logger.warning("总结缓存写入失败: %s", e)

        return summary

    def _get_history_watermark(self) -> Optional[int]:
        """
        取当前用户聊天记录的版本号（chat_history 的最大 id）
        用于判断总结缓存是否已失效
        """
        getter = getattr(self.long_term, "get_history_watermark", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception as e:
            logger.warning("获取 history watermark 失败: %s", e)
            return None

    async def _generate_summary(self, max_messages: int = 50) -> str:
        """
        真正调用 LLM 生成长期记忆总结（无缓存）

        Args:
            max_messages: 最多总结的消息数量

        Returns:
            总结后的文本
        """
        if not self.llm_model:
            return ""

        # 获取长期聊天历史（排除当前会话）
        all_history = self.long_term.get_chat_history(limit=max_messages)
        history_from_other_sessions = [
            msg for msg in all_history
            if msg.get("session_id") != self.session_id
        ]

        # 获取行程历史
        trip_history = self.long_term.get_trip_history(limit=20)

        # 如果既没有聊天记录也没有行程记录，直接返回
        if not history_from_other_sessions and not trip_history:
            return ""

        # 构建聊天记录文本
        history_text = []
        for msg in history_from_other_sessions[-max_messages:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            history_text.append(f"[{timestamp}] {role}: {content}")

        history_str = "\n".join(history_text) if history_text else "（无聊天记录）"

        # 构建行程历史文本
        trip_text = []
        for trip in trip_history:
            origin = trip.get("origin", "未知")
            destination = trip.get("destination", "未知")
            start_date = trip.get("start_date", "")
            end_date = trip.get("end_date", "")
            purpose = trip.get("purpose", "旅游")
            timestamp = trip.get("timestamp", "")

            if start_date and end_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date} 至 {end_date}) - {purpose}")
            elif start_date:
                trip_text.append(f"[{timestamp}] {origin} → {destination} ({start_date}) - {purpose}")
            else:
                trip_text.append(f"[{timestamp}] {origin} → {destination} - {purpose}")

        trip_str = "\n".join(trip_text) if trip_text else "（无行程记录）"

        # 使用LLM总结
        summarization_prompt = f"""请总结以下历史信息中的关键内容，包括：
1. 用户的旅行偏好和习惯
2. 用户询问过的重要问题
3. 用户的出行历史和目的地
4. 其他重要的上下文信息

【历史聊天记录】
{history_str}

【历史行程记录】
{trip_str}

请用简洁的语言总结（不超过200字）："""

        try:
            # 调用模型（异步调用）
            response = await self.llm_model([{"role": "user", "content": summarization_prompt}])

            # 处理异步生成器响应
            summary = ""
            if hasattr(response, '__aiter__'):
                # 异步生成器，需要迭代获取内容
                async for chunk in response:
                    if isinstance(chunk, str):
                        summary = chunk
                    elif hasattr(chunk, 'content'):
                        if isinstance(chunk.content, str):
                            summary = chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    summary = item.get('text', '')
            elif hasattr(response, 'content'):
                summary = str(response.content)
            else:
                summary = str(response)

            logger.info(f"Generated long-term memory summary ({len(summary)} chars)")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate long-term summary: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def get_long_term_summary(self, max_messages: int = 50) -> str:
        """
        使用LLM总结长期聊天历史（同步版本）

        Args:
            max_messages: 最多总结的消息数量

        Returns:
            总结后的文本
        """
        import asyncio

        # 检查是否在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 已经在事件循环中，不能使用 asyncio.run
            logger.warning("get_long_term_summary called from async context, please use get_long_term_summary_async instead")
            return ""
        except RuntimeError:
            # 没有运行的事件循环，可以使用 asyncio.run
            return asyncio.run(self.get_long_term_summary_async(max_messages))
