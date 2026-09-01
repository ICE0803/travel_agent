"""
LLM 连接与可用性：重试退避、可重试错误判断、健康检查
"""
import asyncio
import logging
import time
from typing import TypeVar, Callable, Awaitable, Tuple

from .circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retriable_error(exc: BaseException) -> bool:
    """
    判断是否为可重试错误（网络/超时/限流/5xx）。
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    return False


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay_sec: float = 1.0,
    max_delay_sec: float = 30.0,
    jitter: bool = True,
) -> T:
    """
    对异步调用进行重试，使用指数退避（可选抖动）。

    Args:
        coro_factory: 无参可调用，每次返回新的协程（避免重复使用已 consumed 的 generator）
        max_retries: 最大重试次数（不含首次），即最多调用 1 + max_retries 次
        base_delay_sec: 首次退避基数（秒）
        max_delay_sec: 退避上限（秒）
        jitter: 是否在退避时间上加随机抖动

    Returns:
        协程的返回值

    Raises:
        最后一次尝试的异常（若全部失败）
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except CircuitOpenError:
            raise
        except Exception as e:
            last_exc = e
            if attempt == max_retries or not is_retriable_error(e):
                raise
            delay = min(base_delay_sec * (2 ** attempt), max_delay_sec)
            if jitter:
                import random
                delay = delay * (0.5 + random.random())
            logger.warning(
                "LLM call failed (attempt %d/%d), retry in %.1fs: %s",
                attempt + 1, max_retries + 1, delay, e,
            )
            await asyncio.sleep(delay)
    raise last_exc


async def run_health_check(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout_sec: float = 10.0,
) -> Tuple[bool, str]:
    """
    对 LLM 服务做一次最小化健康检查（发一条极简请求）。

    Returns:
        (success, message)
    """
    try:
        from agentscope.model import OpenAIChatModel
    except ImportError:
        return False, "AgentScope not installed"

    try:
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": base_url, "timeout": timeout_sec},
            temperature=0,
            max_tokens=5,
        )
        # 最小请求
        messages = [{"role": "user", "content": "1"}]
        response = await model(messages)
        # 可能是异步生成器或直接结果
        text = ""
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                if isinstance(chunk, str):
                    text = chunk
                    break
                if hasattr(chunk, "content"):
                    text = getattr(chunk, "content", "") or ""
                    break
        elif hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "content"):
            text = response.content or ""
        elif isinstance(response, dict) and "content" in response:
            text = response["content"] or ""

        if text is not None and len(str(text)) >= 0:
            return True, "ok"
        return True, "ok (no content)"
    except Exception as e:
        return False, str(e)
