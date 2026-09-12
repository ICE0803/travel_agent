"""
连接与后端选择（进程级单例）

- PostgreSQL: psycopg_pool.ConnectionPool
- Redis:      redis.Redis
- backend="auto" 时连不上自动降级为本地实现（JSON + 内存）
"""
import atexit
import logging
from typing import Any, Optional

from config import STORAGE_CONFIG

logger = logging.getLogger(__name__)

_pg_pool: Optional[Any] = None
_redis_client: Optional[Any] = None
_backend: str = "unknown"
_initialized: bool = False


def init_backends(force: bool = False) -> str:
    """初始化后端连接。重复调用幂等。"""
    global _pg_pool, _redis_client, _backend, _initialized
    if _initialized and not force:
        return _backend

    mode = STORAGE_CONFIG.get("backend", "auto")

    if mode == "local":
        _backend, _initialized = "local", True
        logger.info("存储后端: 本地实现（JSON + 内存）")
        return _backend

    pg_ok = _try_init_pg()
    redis_ok = _try_init_redis()

    if pg_ok and redis_ok:
        _backend = "postgres+redis"
    elif pg_ok:
        _backend = "postgres"
    elif redis_ok:
        _backend = "redis"
    else:
        _backend = "local"
        if mode == "postgres":
            raise RuntimeError("backend=postgres 但连接失败，请检查 WSL 里 postgresql 服务")
        logger.warning("PostgreSQL / Redis 均不可用，已降级为本地实现（JSON + 内存）")

    _initialized = True
    logger.info("存储后端就绪: %s", _backend)
    return _backend


def _try_init_pg() -> bool:
    global _pg_pool
    cfg = STORAGE_CONFIG["postgres"]
    try:
        from psycopg_pool import ConnectionPool

        conninfo = (
            f"host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} "
            f"user={cfg['user']} password={cfg['password']} "
            f"connect_timeout={cfg.get('connect_timeout', 5)}"
        )
        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=cfg.get("min_pool", 1),
            max_size=cfg.get("max_pool", 10),
            kwargs={"autocommit": True},
            open=False,
        )
        pool.open(wait=True, timeout=cfg.get("connect_timeout", 5))
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        _pg_pool = pool
        logger.info("PostgreSQL 已连接: %s:%s/%s", cfg["host"], cfg["port"], cfg["dbname"])
        return True
    except Exception as e:
        logger.warning("PostgreSQL 连接失败: %s", e)
        _pg_pool = None
        return False


def _try_init_redis() -> bool:
    global _redis_client
    cfg = STORAGE_CONFIG["redis"]
    try:
        import redis as redis_lib

        client = redis_lib.Redis(
            host=cfg["host"],
            port=cfg["port"],
            db=cfg.get("db", 0),
            password=cfg.get("password"),
            decode_responses=True,
            socket_timeout=cfg.get("socket_timeout", 2.0),
            socket_connect_timeout=cfg.get("socket_timeout", 2.0),
            max_connections=cfg.get("max_connections", 16),
        )
        client.ping()
        _redis_client = client
        logger.info("Redis 已连接: %s:%s db=%s", cfg["host"], cfg["port"], cfg.get("db", 0))
        return True
    except Exception as e:
        logger.warning("Redis 连接失败: %s", e)
        _redis_client = None
        return False


def get_pg_pool():
    if not _initialized:
        init_backends()
    return _pg_pool


def get_redis():
    if not _initialized:
        init_backends()
    return _redis_client


def get_backend() -> str:
    if not _initialized:
        init_backends()
    return _backend


def close_backends() -> None:
    """CLI 退出时释放连接"""
    global _pg_pool, _redis_client, _initialized
    if _pg_pool is not None:
        try:
            _pg_pool.close()
        except Exception:
            pass
        _pg_pool = None
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            pass
        _redis_client = None
    _initialized = False
    logger.info("后端连接已关闭")


# 进程退出时自动关闭连接池。
# 仅在 cli.py 的 exit 命令里调用是不够的：Ctrl+C、未捕获异常、脚本正常结束
# 都不会走那条分支，psycopg 的池线程会拖住解释器退出（每个线程各等 5 秒）并打出
# "couldn't stop thread ... within 5.0 seconds" 警告。close_backends() 是幂等的。
atexit.register(close_backends)