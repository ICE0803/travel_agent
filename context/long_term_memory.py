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

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆：持久化用户信息
    - 用户偏好（家庭地址、酒店品牌、航空公司等）
    - 历史行程记录
    - 统计信息
    """

    def __init__(self, user_id: str, storage_path: str = "data/memory"):
        """
        初始化长期记忆

        Args:
            user_id: 用户ID
            storage_path: 存储路径
        """
        self.user_id = user_id
        self.storage_path = storage_path
        self.db_path = os.path.join(storage_path, f"{user_id}.json")

        # 确保存储目录存在
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        # 加载或初始化数据
        self.data = self._load()
        logger.info(f"Long-term memory initialized for user: {user_id}")

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

    def save_preference(self, pref_type: str, value: Any):
        """
        保存用户偏好（列表格式）

        Args:
            pref_type: 偏好类型
            value: 偏好值
        """
        # 查找是否已存在该类型的偏好
        preferences = self.data["preferences"]
        found = False

        for pref in preferences:
            if pref.get("type") == pref_type:
                pref["value"] = value
                found = True
                break

        # 如果不存在，添加新的偏好
        if not found:
            preferences.append({"type": pref_type, "value": value})

        self._save()
        logger.info(f"Saved preference: {pref_type} = {value}")

    def get_preference(self, pref_type: str = None) -> Any:
        """
        获取用户偏好

        Args:
            pref_type: 偏好类型，None返回字典格式的全部偏好

        Returns:
            偏好值或偏好字典
        """
        preferences = self.data["preferences"]

        if pref_type is None:
            # 返回字典格式，方便调用方使用
            result = {}
            for pref in preferences:
                result[pref.get("type")] = pref.get("value")
            return result
        else:
            # 查找特定类型的偏好
            for pref in preferences:
                if pref.get("type") == pref_type:
                    return pref.get("value")
            return None

    def add_hotel_brand(self, brand: str):
        """添加酒店品牌偏好（追加到列表）"""
        # 查找 hotel_brands 偏好
        preferences = self.data["preferences"]
        found = False

        for pref in preferences:
            if pref.get("type") == "hotel_brands":
                # 确保 value 是列表
                if not isinstance(pref["value"], list):
                    pref["value"] = [pref["value"]] if pref["value"] else []

                # 追加品牌
                if brand not in pref["value"]:
                    pref["value"].append(brand)
                found = True
                break

        # 如果不存在，创建新的
        if not found:
            preferences.append({"type": "hotel_brands", "value": [brand]})

        self._save()
        logger.info(f"Added hotel brand preference: {brand}")

    def add_airline(self, airline: str):
        """添加航空公司偏好（追加到列表）"""
        # 查找 airlines 偏好
        preferences = self.data["preferences"]
        found = False

        for pref in preferences:
            if pref.get("type") == "airlines":
                # 确保 value 是列表
                if not isinstance(pref["value"], list):
                    pref["value"] = [pref["value"]] if pref["value"] else []

                # 追加航空公司
                if airline not in pref["value"]:
                    pref["value"].append(airline)
                found = True
                break

        # 如果不存在，创建新的
        if not found:
            preferences.append({"type": "airlines", "value": [airline]})

        self._save()
        logger.info(f"Added airline preference: {airline}")

    def add_chat_message(self, role: str, content: str, session_id: str = None):
        """
        添加聊天消息到长期记忆

        Args:
            role: 角色 (user/assistant)
            content: 消息内容
            session_id: 会话ID（可选）
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }

        self.data["chat_history"].append(message)
        self.data["statistics"]["total_messages"] += 1
        self._save()
        logger.debug(f"Added chat message to long-term memory: {role}")

    def get_chat_history(self, limit: int = None, session_id: str = None) -> List[Dict[str, Any]]:
        """
        获取聊天历史

        Args:
            limit: 返回数量限制
            session_id: 会话ID（只返回特定会话的消息）

        Returns:
            消息列表
        """
        messages = self.data["chat_history"]

        if session_id:
            messages = [m for m in messages if m.get("session_id") == session_id]

        if limit:
            return messages[-limit:]
        return messages

    def save_trip_history(self, trip_info: Dict[str, Any]):
        """
        保存行程历史

        Args:
            trip_info: 行程信息
        """
        trip_record = {
            "trip_id": f"trip_{len(self.data['trip_history']) + 1}",
            "timestamp": datetime.now().isoformat(),
            **trip_info
        }

        self.data["trip_history"].append(trip_record)

        # 更新统计信息
        self.data["statistics"]["total_trips"] += 1

        # 更新常去目的地统计
        destination = trip_info.get("destination")
        if destination:
            freq = self.data["statistics"]["frequent_destinations"]
            freq[destination] = freq.get(destination, 0) + 1

        self._save()
        logger.info(f"Saved trip history: {trip_record['trip_id']}")

    def get_trip_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取历史行程

        Args:
            limit: 返回数量限制

        Returns:
            行程列表
        """
        return self.data["trip_history"][-limit:] if limit else self.data["trip_history"]

    def get_frequent_destinations(self, top_n: int = 5) -> List[tuple]:
        """
        获取常去目的地

        Args:
            top_n: 返回前N个

        Returns:
            [(destination, count), ...]
        """
        freq = self.data["statistics"]["frequent_destinations"]
        sorted_dest = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_dest[:top_n]

    def increment_query_count(self):
        """增加查询计数"""
        self.data["statistics"]["total_queries"] += 1
        self._save()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.data["statistics"].copy()

    def clear_history(self):
        """清空历史记录（保留偏好）"""
        self.data["chat_history"] = []
        self.data["trip_history"] = []
        self.data["statistics"]["total_trips"] = 0
        self.data["statistics"]["total_messages"] = 0
        self.data["statistics"]["frequent_destinations"] = {}
        self._save()
        logger.info("Cleared all history (chat + trips)")

    def delete_all(self):
        """删除所有数据（包括文件）"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            logger.warning(f"Deleted long-term memory file: {self.db_path}")
