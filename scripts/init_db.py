import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from config import STORAGE_CONFIG


def main():
    cfg = STORAGE_CONFIG["postgres"]
    dsn = (
        f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
        f"user={cfg['user']} password={cfg['password']} "
        f"connect_timeout={cfg.get('connect_timeout', 5)}"
    )
    sql = (PROJECT_ROOT / "context" / "schema.sql").read_text(encoding="utf-8")

    print(f"连接 PostgreSQL: {cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            print("✓ schema.sql 执行完成")
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' ORDER BY table_name"
            )
            print("  已存在表:", ", ".join(r[0] for r in cur.fetchall()))
    print("✓ 数据库初始化完成")


if __name__ == "__main__":
    main()