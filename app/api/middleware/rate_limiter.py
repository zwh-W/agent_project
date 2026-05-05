# app/api/middleware/rate_limiter.py
"""
基于滑动窗口算法的 API 限流中间件

设计动机：
    P1 阶段加了 API Key 鉴权，但鉴权只能防止陌生人调用。
    它无法阻止一个合法用户在 1 秒内发 1000 个请求，
    把你的 LLM 费用打穿，或者把服务打崩。
    限流是鉴权之后必须加的第二道防线。

为什么选滑动窗口而不是固定窗口？
    固定窗口的问题：假设限制"每分钟 60 次"，
    用户在 00:59 发 60 次，01:00 又发 60 次，
    2 秒内打了 120 次请求，完全绕过了限制。
    滑动窗口始终看"过去 60 秒"，没有边界漏洞。

实现方案：
    纯内存 + 滑动窗口，不依赖 Redis。
    生产环境应换成 Redis，实现跨进程/跨节点共享。
    这里的内存版本适合单机部署，也是面试中展示原理的好材料。

限流维度（双层保护）：
    1. 全局维度：整个服务每秒最多处理 N 个请求（防止服务被打崩）
    2. 用户维度：每个 API Key 每分钟最多 M 个请求（防止单用户滥用）

面试价值：
    "你的系统有限流吗？用的什么算法？为什么不用令牌桶？"
    这是面试官区分"会用框架"和"懂系统设计"的标准问题。
"""
import time
import threading
from collections import deque
from typing import Dict, Deque, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 豁免限流的路径
_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class SlidingWindowCounter:
    """
    滑动窗口计数器（线程安全）

    原理：
        维护一个时间戳队列，每次请求进来：
        1. 清除队列中所有"窗口期之前"的时间戳（过期数据）
        2. 检查队列长度是否超过阈值
        3. 如果没超，记录当前时间戳，放行
        4. 如果超了，拒绝

    时间复杂度：O(过期请求数)，均摊 O(1)
    空间复杂度：O(窗口内请求数)
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    def is_allowed(self) -> Tuple[bool, int]:
        """
        检查是否允许本次请求

        Returns:
            (是否允许, 当前窗口内的请求数)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # 清除过期时间戳（窗口左边界之前的全部弹出）
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()

            current_count = len(self._timestamps)

            if current_count >= self.max_requests:
                return False, current_count

            # 记录本次请求时间戳
            self._timestamps.append(now)
            return True, current_count + 1

    def get_retry_after(self) -> float:
        """
        返回需要等待多少秒才能重试
        （最老的请求过期后就有空位了）
        """
        with self._lock:
            if not self._timestamps:
                return 0.0
            oldest = self._timestamps[0]
            return max(0.0, oldest + self.window_seconds - time.time())


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    双层滑动窗口限流中间件

    配置（从 settings 读取，回退到合理默认值）：
        rate_limit.global_rpm: 全局每分钟最大请求数（默认 600）
        rate_limit.per_key_rpm: 每个 API Key 每分钟最大请求数（默认 60）
    """

    def __init__(self, app):
        super().__init__(app)

        # 从配置读取，提供合理默认值
        rate_cfg = getattr(settings, 'rate_limit', None)
        global_rpm = getattr(rate_cfg, 'global_rpm', 600) if rate_cfg else 600
        per_key_rpm = getattr(rate_cfg, 'per_key_rpm', 60) if rate_cfg else 60

        # 全局限流器：整个服务每分钟 N 次
        self._global_limiter = SlidingWindowCounter(
            max_requests=global_rpm,
            window_seconds=60.0,
        )

        # 用户级限流器字典：每个 key 独立计数
        self._key_limiters: Dict[str, SlidingWindowCounter] = {}
        self._key_limiters_lock = threading.Lock()
        self._per_key_rpm = per_key_rpm

        logger.info(
            f"限流中间件已启动 | 全局: {global_rpm} rpm | 单 Key: {per_key_rpm} rpm"
        )

    def _get_key_limiter(self, api_key: str) -> SlidingWindowCounter:
        """获取或创建某个 API Key 的限流器（懒加载）"""
        with self._key_limiters_lock:
            if api_key not in self._key_limiters:
                self._key_limiters[api_key] = SlidingWindowCounter(
                    max_requests=self._per_key_rpm,
                    window_seconds=60.0,
                )
            return self._key_limiters[api_key]

    async def dispatch(self, request: Request, call_next):
        # 豁免路径
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # ── 第一层：全局限流 ──────────────────────────
        global_allowed, global_count = self._global_limiter.is_allowed()
        if not global_allowed:
            retry_after = self._global_limiter.get_retry_after()
            logger.warning(
                f"全局限流触发 | path={request.url.path} | "
                f"当前 rpm={global_count} | retry_after={retry_after:.1f}s"
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
                content={
                    "response_code": 429,
                    "response_msg": f"服务繁忙，请 {int(retry_after) + 1} 秒后重试",
                }
            )

        # ── 第二层：用户级限流 ────────────────────────
        api_key = request.headers.get("X-API-Key", "anonymous")
        key_limiter = self._get_key_limiter(api_key)
        key_allowed, key_count = key_limiter.is_allowed()

        if not key_allowed:
            retry_after = key_limiter.get_retry_after()
            logger.warning(
                f"用户限流触发 | key={api_key[:8]}... | "
                f"当前 rpm={key_count} | retry_after={retry_after:.1f}s"
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
                content={
                    "response_code": 429,
                    "response_msg": f"请求过于频繁，请 {int(retry_after) + 1} 秒后重试",
                }
            )

        # 在响应 Header 里暴露当前用量（方便客户端感知）
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._per_key_rpm)
        response.headers["X-RateLimit-Remaining"] = str(self._per_key_rpm - key_count)

        return response