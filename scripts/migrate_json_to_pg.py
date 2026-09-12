"""把 data/memory/*.json 的历史记忆迁移到 PostgreSQL"""
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from psycopg.types.json import Jsonb
from config import STORAGE_CONFIG

MEMORY_DIR = PROJECT_ROOT / "data" / "memory"


def parse_ts(value, fallback=None):
    if not value:
        return fallback or datetime.now()
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return fallback or datetime.now()


def main():
    cfg = STORAGE_CONFIG["postgres"]
    dsn = (f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
           f"user={cfg['user']} password={cfg['password']}")

    files = sorted(MEMORY_DIR.glob("*.json"))
    if not files:
        print(f"未找到记忆文件: {MEMORY_DIR}")
        return

    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            for path in files:
                user_id = path.stem
                raw = json.loads(path.read_text(encoding="utf-8"))

                cur.execute(
                    "INSERT INTO users (user_id) VALUES (%s) "
                    "ON CONFLICT (user_id) DO NOTHING", (user_id,))

                # 偏好（兼容旧的 dict 格式和嵌套脏数据）
                prefs = raw.get("preferences", [])
                if isinstance(prefs, dict):
                    prefs = [{"type": k, "value": v} for k, v in prefs.items() if v]
                pref_count = 0
                for p in prefs:
                    if not isinstance(p, dict):
                        continue
                    ptype, pvalue = p.get("type"), p.get("value")
                    if ptype == "preferences" and isinstance(pvalue, list):
                        for n in pvalue:
                            if isinstance(n, dict) and n.get("type"):
                                cur.execute(
                                    """INSERT INTO user_preferences (user_id, pref_type, value)
                                       VALUES (%s, %s, %s)
                                       ON CONFLICT (user_id, pref_type)
                                       DO UPDATE SET value = EXCLUDED.value""",
                                    (user_id, n["type"], Jsonb(n["value"])))
                                pref_count += 1
                        continue
                    if ptype and pvalue is not None:
                        cur.execute(
                            """INSERT INTO user_preferences (user_id, pref_type, value)
                               VALUES (%s, %s, %s)
                               ON CONFLICT (user_id, pref_type)
                               DO UPDATE SET value = EXCLUDED.value""",
                            (user_id, ptype, Jsonb(pvalue)))
                        pref_count += 1

                # 聊天记录：用原 timestamp 回填 created_at，保持时间线
                chats = raw.get("chat_history", [])
                for m in chats:
                    cur.execute(
                        """INSERT INTO chat_history
                             (user_id, session_id, role, content, created_at)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (user_id, m.get("session_id"), m.get("role", "user"),
                         m.get("content", ""), parse_ts(m.get("timestamp"))))

                trips = raw.get("trip_history", [])
                for t in trips:
                    cur.execute(
                        """INSERT INTO trip_history
                             (user_id, origin, destination, start_date, end_date,
                              purpose, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (user_id, t.get("origin"), t.get("destination"),
                         t.get("start_date"), t.get("end_date"), t.get("purpose"),
                         parse_ts(t.get("timestamp"))))

                print(f"✓ {path.name}: 偏好 {pref_count} | 聊天 {len(chats)} | 行程 {len(trips)}")

        conn.commit()
    print("\n✓ 迁移完成")


if __name__ == "__main__":
    main()