"""
熔断器 (Circuit Breaker)
在连续失败后暂停调用 LLM，避免雪崩；一段时间后半开试探恢复。
"""
import time
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常调用
    OPEN = "open"          # 拒绝调用，直接降级
    HALF_OPEN = "half_open"  # 试探性放行少量请求


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常，表示当前不执行 LLM 调用"""
    pass


class CircuitBreaker:
    """
    熔断器：连续失败 N 次后打开，拒绝一段时间后进入半开，半开成功 M 次后关闭。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 60.0,
        half_open_successes: int = 2,
    ):
        """
        Args:
            failure_threshold: 连续失败多少次后打开熔断
            recovery_timeout_sec: 打开状态持续多少秒后进入半开
            half_open_successes: 半开状态下连续成功多少次后关闭熔断
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_successes = half_open_successes

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        """当前状态；可能从 OPEN 自动变为 HALF_OPEN"""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_sec:
                logger.info("Circuit breaker: OPEN -> HALF_OPEN (recovery timeout)")
                self._state = CircuitState.HALF_OPEN
                self._half_open_success_count = 0
        return self._state

    def allow_call(self) -> bool:
        """
        是否允许本次调用。
        - CLOSED: 允许
        - OPEN: 不允许
        - HALF_OPEN: 允许（用于试探）
        """
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.OPEN:
            return False
        # HALF_OPEN: 允许
        return True

    def record_success(self) -> None:
        """记录一次成功调用"""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.half_open_successes:
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED (recovered)")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._opened_at = None
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """记录一次失败调用"""
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN (failure in half-open)")
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._failure_count = 0
            return

        if self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                logger.warning(
                    "Circuit breaker: CLOSED -> OPEN (failure_threshold=%d reached)",
                    self.failure_threshold,
                )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def raise_if_open(self) -> None:
        """若熔断器打开则抛出 CircuitOpenError，供上层直接降级"""
        if not self.allow_call():
            raise CircuitOpenError("服务暂时不可用，请稍后再试")

    def get_status(self) -> dict:
        """当前状态摘要，便于日志或 status 命令"""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "opened_at": self._opened_at,
        }
