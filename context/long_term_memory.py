"""
长期记忆 (Long-term Memory)
持久化存储用户信息，支持跨会话访问
"""
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from psycopg.types.json import Jsonb 

from config import STORAGE_CONFIG
from .backends import get_pg_pool, get_redis

PREF_TTL = STORAGE_CONFIG.get("ttl", {}).get("preferences", 600)

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆：持久化用户信息
    - 用户偏好（家庭地址、酒店品牌、航空公司等）
    - 历史行程记录
    - 统计信息
    """

    def __init__(self, user_id: str, storage_path: str = "data/memory"):
        self.user_id = user_id
        self.storage_path = storage_path
        self.db_path = os.path.join(storage_path, f"{user_id}.json")

        self._pg = get_pg_pool()
        self._redis = get_redis()

        # 偏好缓存 key（命中率统计用）
        self._pref_cache_key = f"user:{user_id}:prefs"
        self.pref_hit_key = f"{self._pref_cache_key}:hit"
        self.pref_miss_key = f"{self._pref_cache_key}:miss"

        if self._pg is not None:
            self._ensure_user()
            logger.info("Long-term memory initialized for user: %s (postgres)", user_id)
        else:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.data = self._load()          # 原 JSON 逻辑保留
            logger.info("Long-term memory initialized for user: %s (json)", user_id)


    def _ensure_user(self):
        self._execute(
            "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
            (self.user_id,),
        )

    def _execute(self, sql, params=None, fetch=True):
        with self._pg.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                if fetch and cur.description is not None:
                    return cur.fetchall()
                return None


    def _load(self) -> Dict[str, Any]:
        """从文件加载数据"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded long-term memory from {self.db_path}")

                    # 数据迁移：兼容旧格式
                    data = self._migrate_data(data)
                    return data
            except Exception as e:
                logger.error(f"Failed to load long-term memory: {e}")
                return self._init_data()
        else:
            logger.info("No existing long-term memory, creating new")
            return self._init_data()

    def _migrate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        迁移旧数据格式到新格式

        Args:
            data: 原始数据

        Returns:
            迁移后的数据
        """
        # 1. 确保必需字段存在
        if "chat_history" not in data:
            data["chat_history"] = []
        if "trip_history" not in data:
            data["trip_history"] = []
        if "statistics" not in data:
            data["statistics"] = {}
        if "total_messages" not in data.get("statistics", {}):
            data["statistics"]["total_messages"] = 0
        if "preferences" not in data:
            data["preferences"] = []

        # 2. 迁移旧格式：字典 → 列表
        if isinstance(data.get("preferences"), dict):
            old_prefs = data["preferences"]
            new_prefs = []
            for pref_type, pref_value in old_prefs.items():
                if pref_value is not None:
                    new_prefs.append({"type": pref_type, "value": pref_value})
            data["preferences"] = new_prefs
            logger.info(f"Migrated: Converted preferences from dict to list ({len(new_prefs)} items)")

        # 3. 修复嵌套 bug（旧代码产生的错误数据）
        if isinstance(data.get("preferences"), list):
            fixed_prefs = []
            for pref in data["preferences"]:
                if isinstance(pref, dict):
                    # 错误的嵌套：{"type": "preferences", "value": [...]}
                    if pref.get("type") == "preferences" and isinstance(pref.get("value"), list):
                        for nested_pref in pref["value"]:
                            if isinstance(nested_pref, dict) and "type" in nested_pref:
                                fixed_prefs.append({"type": nested_pref["type"], "value": nested_pref["value"]})
                        logger.info("Migrated: Fixed nested preferences bug")
                    else:
                        fixed_prefs.append(pref)

            if fixed_prefs != data["preferences"]:
                data["preferences"] = fixed_prefs

        # 保存迁移后的数据
        self.data = data
        self._save()

        return data

    def _init_data(self) -> Dict[str, Any]:
        """初始化数据结构"""
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": [],  # 偏好列表: [{"type": "home_location", "value": "天津"}, ...]
            "chat_history": [],  # 所有聊天记录（跨会话）
            "trip_history": [],  # 所有行程记录
            "statistics": {
                "total_trips": 0,
                "total_messages": 0,
                "frequent_destinations": {}
            }
        }

    def _save(self):
        """保存数据到文件"""
        try:
            self.data["updated_at"] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved long-term memory to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")

    def get_preference(self, pref_type: str = None) -> Any:
        prefs = self._get_prefs_cached()
        if pref_type is None:
            return prefs
        return prefs.get(pref_type)

    def _get_prefs_cached(self) -> Dict[str, Any]:
        """先查 Redis，未命中回源存储并回填（Lazy Loading）"""
        if self._redis is not None:
            try:
                raw = self._redis.get(self._pref_cache_key)
                if raw is not None:
                    self._redis.incr(self.pref_hit_key)
                    return json.loads(raw)
                self._redis.incr(self.pref_miss_key)
            except Exception as e:
                logger.warning("偏好缓存读取失败，回源: %s", e)

        prefs = self._load_prefs_from_store()
        self._write_pref_cache(prefs)
        return prefs

    def _load_prefs_from_store(self) -> Dict[str, Any]:
        if self._pg is not None:
            rows = self._execute(
                "SELECT pref_type, value FROM user_preferences WHERE user_id = %s",
                (self.user_id,),
            )
            return {r[0]: r[1] for r in rows}
        return {p.get("type"): p.get("value") for p in self.data["preferences"]}

    def _write_pref_cache(self, prefs: Dict[str, Any]):
        if self._redis is None:
            return
        try:
            self._redis.setex(
                self._pref_cache_key, PREF_TTL,
                json.dumps(prefs, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("偏好缓存写入失败: %s", e)

    def save_preference(self, pref_type: str, value: Any):
        if self._pg is not None:
            self._execute(
                """
                INSERT INTO user_preferences (user_id, pref_type, value, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (user_id, pref_type)
                DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (self.user_id, pref_type, Jsonb(value)),
            )
        else:
            # ↓↓↓ 原 JSON 逻辑，原样保留 ↓↓↓
            preferences = self.data["preferences"]
            found = False
            for pref in preferences:
                if pref.get("type") == pref_type:
                    pref["value"] = value
                    found = True
                    break
            if not found:
                preferences.append({"type": pref_type, "value": value})
            self._save()

        # Write-Through：写库后立即刷新缓存
        self._write_pref_cache(self._load_prefs_from_store())
        logger.info("Saved preference: %s = %s", pref_type, value)

    def _append_to_list_pref(self, pref_type: str, item: str):
        existing = self.get_preference(pref_type)
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        if item not in existing:
            existing.append(item)
        self.save_preference(pref_type, existing)

    def add_hotel_brand(self, brand: str):
        self._append_to_list_pref("hotel_brands", brand)

    def add_airline(self, airline: str):
        self._append_to_list_pref("airlines", airline)

    def add_chat_message(self, role: str, content: str, session_id: str = None):
        if self._pg is not None:
            self._execute(
                """INSERT INTO chat_history (user_id, session_id, role, content, created_at)
                   VALUES (%s, %s, %s, %s, now())""",
                (self.user_id, session_id, role, content),
                fetch=False,
            )
        else:
            message = {
                "role": role, "content": content,
                "timestamp": datetime.now().isoformat(), "session_id": session_id,
            }
            self.data["chat_history"].append(message)
            self.data["statistics"]["total_messages"] += 1
            self._save()
        logger.debug("Added chat message to long-term memory: %s", role)

    def get_chat_history(self, limit: int = None, session_id: str = None) -> List[Dict[str, Any]]:
        if self._pg is None:
            messages = self.data["chat_history"]
            if session_id:
                messages = [m for m in messages if m.get("session_id") == session_id]
            if limit:
                return messages[-limit:]
            return messages

        sql = "SELECT role, content, created_at, session_id FROM chat_history WHERE user_id = %s"
        params: List[Any] = [self.user_id]
        if session_id:
            sql += " AND session_id = %s"
            params.append(session_id)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(limit if limit else 1_000_000)

        rows = self._execute(sql, tuple(params))
        # 倒序取出后反转，恢复时间正序（与 JSON 版语义一致）
        return [
            {
                "role": r[0],
                "content": r[1],
                "timestamp": r[2].isoformat() if r[2] else "",
                "session_id": r[3],
            }
            for r in reversed(rows)
        ]

    def save_trip_history(self, trip_info: Dict[str, Any]):
        if self._pg is not None:
            self._execute(
                """INSERT INTO trip_history
                     (user_id, origin, destination, start_date, end_date, purpose, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, now())""",
                (self.user_id, trip_info.get("origin"), trip_info.get("destination"),
                 trip_info.get("start_date"), trip_info.get("end_date"),
                 trip_info.get("purpose")),
                fetch=False,
            )
        else:
            trip_record = {
                "trip_id": f"trip_{len(self.data['trip_history']) + 1}",
                "timestamp": datetime.now().isoformat(),
                **trip_info,
            }
            self.data["trip_history"].append(trip_record)
            self.data["statistics"]["total_trips"] += 1
            destination = trip_info.get("destination")
            if destination:
                freq = self.data["statistics"]["frequent_destinations"]
                freq[destination] = freq.get(destination, 0) + 1
            self._save()
        logger.info("Saved trip history: %s -> %s",
                    trip_info.get("origin"), trip_info.get("destination"))

    def get_trip_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._pg is None:
            return self.data["trip_history"][-limit:] if limit else self.data["trip_history"]

        rows = self._execute(
            """SELECT 'trip_' || id AS trip_id, created_at, origin, destination,
                      start_date, end_date, purpose
               FROM trip_history WHERE user_id = %s
               ORDER BY id DESC LIMIT %s""",
            (self.user_id, limit if limit else 1_000_000),
        )
        return [
            {
                "trip_id": r[0],
                "timestamp": r[1].isoformat() if r[1] else "",
                "origin": r[2], "destination": r[3],
                "start_date": r[4], "end_date": r[5], "purpose": r[6],
            }
            for r in reversed(rows)
        ]

    def get_frequent_destinations(self, top_n: int = 5) -> List[tuple]:
        if self._pg is None:
            freq = self.data["statistics"]["frequent_destinations"]
            return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

        rows = self._execute(
            """SELECT destination, COUNT(*) AS c FROM trip_history
               WHERE user_id = %s AND destination IS NOT NULL AND destination <> ''
               GROUP BY destination ORDER BY c DESC, destination ASC LIMIT %s""",
            (self.user_id, top_n),
        )
        return [(r[0], r[1]) for r in rows]

    def increment_query_count(self):
        """修复：旧实现读 statistics["total_queries"]，但该 key 从未初始化 → 必然 KeyError"""
        if self._pg is not None:
            self._execute(
                "UPDATE users SET query_count = query_count + 1, updated_at = now() "
                "WHERE user_id = %s",
                (self.user_id,), fetch=False,
            )
        else:
            self.data["statistics"]["total_queries"] = (
                self.data["statistics"].get("total_queries", 0) + 1
            )
            self._save()

    def get_history_watermark(self) -> int:
        """聊天记录版本号，供 LLM 总结缓存判断是否失效"""
        if self._pg is not None:
            return self._execute(
                "SELECT COALESCE(MAX(id), 0) FROM chat_history WHERE user_id = %s",
                (self.user_id,),
            )[0][0]
        return len(self.data.get("chat_history", []))

    def get_statistics(self) -> Dict[str, Any]:
        if self._pg is None:
            return self.data["statistics"].copy()

        total_trips = self._execute(
            "SELECT COUNT(*) FROM trip_history WHERE user_id = %s", (self.user_id,)
        )[0][0]
        total_messages = self._execute(
            "SELECT COUNT(*) FROM chat_history WHERE user_id = %s", (self.user_id,)
        )[0][0]
        freq_rows = self._execute(
            """SELECT destination, COUNT(*) FROM trip_history
               WHERE user_id = %s AND destination IS NOT NULL AND destination <> ''
               GROUP BY destination""",
            (self.user_id,),
        )
        return {
            "total_trips": total_trips,
            "total_messages": total_messages,
            "frequent_destinations": {r[0]: r[1] for r in freq_rows},
        }

    def clear_history(self):
        if self._pg is not None:
            self._execute("DELETE FROM chat_history WHERE user_id = %s", (self.user_id,), fetch=False)
            self._execute("DELETE FROM trip_history  WHERE user_id = %s", (self.user_id,), fetch=False)
        else:
            self.data["chat_history"] = []
            self.data["trip_history"] = []
            self.data["statistics"].update(
                {"total_trips": 0, "total_messages": 0, "frequent_destinations": {}}
            )
            self._save()


    def delete_all(self):
        if self._pg is not None:
            # users 上的 ON DELETE CASCADE 会连带清掉偏好/聊天/行程
            self._execute("DELETE FROM users WHERE user_id = %s", (self.user_id,), fetch=False)
        elif os.path.exists(self.db_path):
            os.remove(self.db_path)