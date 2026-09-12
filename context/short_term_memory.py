"""
短期记忆 (Short-term Memory)

- Redis 后端：LIST 存储，RPUSH + LTRIM 实现固定长度滑动窗口，每轮写入续期 TTL
- 本地后端：内存 list + 切片淘汰（Redis 不可用时自动降级）
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from config import STORAGE_CONFIG

from .backends import get_redis

logger = logging.getLogger(__name__)

SESSION_TTL = STORAGE_CONFIG.get("ttl", {}).get("session", 3600)


class ShortTermMemory:
    """
    短期记忆：存储最近的对话历史

    - Redis: key = session:{session_id}:messages，LIST 结构
    - 返回结构与旧实现完全一致：{role, content, timestamp, metadata}
    """

    def __init__(self, max_turns: int = 10, session_id: str = "default"):
        self.max_turns = max_turns
        self.session_id = session_id
        self._max_messages = max_turns * 2  # 一轮 = 用户 + 助手

        self._redis = get_redis()
        self._key = f"session:{session_id}:messages"

        # 本地降级用
        self.messages: List[Dict[str, Any]] = []

        if self._redis is None:
            logger.info("短期记忆使用本地内存（Redis 不可用）")
        else:
            logger.info("短期记忆使用 Redis: %s (TTL=%ss)", self._key, SESSION_TTL)

    # ---------- 写 ----------

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """添加消息到短期记忆，并自动淘汰旧消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.rpush(self._key, json.dumps(message, ensure_ascii=False))
                pipe.ltrim(self._key, -self._max_messages, -1)
                pipe.expire(self._key, SESSION_TTL)  # 滑动过期
                pipe.execute()
                logger.debug("Added message to short-term memory (redis): %s", role)
                return
            except Exception as e:
                # 永久降级，保证读写来源一致
                logger.warning("Redis 写入失败，本实例降级为本地内存: %s", e)
                self._redis = None

        self.messages.append(message)
        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages:]
        logger.debug("Added message to short-term memory (local): %s", role)

    # ---------- 读 ----------

    def get_recent_context(self, n_turns: int = None) -> List[Dict[str, Any]]:
        """获取最近 n 轮对话（按时间正序）"""
        n_messages = (n_turns * 2) if n_turns else self._max_messages

        if self._redis is not None:
            try:
                raw = self._redis.lrange(self._key, -n_messages, -1)
                return [json.loads(r) for r in raw]
            except Exception as e:
                logger.warning("Redis 读取失败，本实例降级为本地内存: %s", e)
                self._redis = None

        if len(self.messages) > n_messages:
            return self.messages[-n_messages:]
        return self.messages.copy()

    def get_context_string(self, n_turns: int = 5) -> str:
        """获取最近对话的字符串表示"""
        messages = self.get_recent_context(n_turns)
        if not messages:
            return "无历史对话"
        lines = []
        for msg in messages:
            role_name = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_name}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """清空短期记忆（会话级）"""
        if self._redis is not None:
            try:
                self._redis.delete(self._key)
            except Exception as e:
                logger.warning("Redis 删除失败: %s", e)
        self.messages = []
        logger.info("Short-term memory cleared (session=%s)", self.session_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        msgs: List[Dict[str, Any]] = []
        total = 0

        if self._redis is not None:
            try:
                total = self._redis.llen(self._key)
                raw = self._redis.lrange(self._key, -self._max_messages, -1)
                msgs = [json.loads(r) for r in raw]
            except Exception as e:
                logger.warning("Redis 统计读取失败: %s", e)
        else:
            total = len(self.messages)
            msgs = self.messages

        return {
            "total_messages": total,
            "max_turns": self.max_turns,
            "oldest_message_time": msgs[0]["timestamp"] if msgs else None,
            "newest_message_time": msgs[-1]["timestamp"] if msgs else None,
        }