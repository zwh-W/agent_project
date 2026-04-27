# app/api/middleware/request_context.py
"""
★ [新增模块] 请求 ID 追踪中间件

原因：原项目日志只有 session_id，没有 request_id 透传。
当线上出现问题时，你无法通过日志还原某一次具体请求的完整链路
（因为同一个 session 可能有多个并发请求）。

这是生产系统的基本可观测性要求，也是面试官问
"你的系统出了问题你怎么排查"时的核心答案。

实现原理：
- 用 Python contextvars 将 request_id 绑定到当前协程上下文
- 中间件在每个请求进来时生成并注入 request_id
- 业务代码通过 get_request_id() 随时获取，注入日志
- 响应 Header 里也返回 X-Request-ID，方便客户端上报
"""
import uuid
from contextvars import ContextVar
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger

logger = get_logger(__name__)

# ★ ContextVar：协程安全的"线程局部变量"
# 每个异步请求有自己独立的值，不会串台
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """在任意位置获取当前请求的 request_id"""
    return _request_id_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求注入唯一 request_id，并写入响应 Header"""

    async def dispatch(self, request: Request, call_next):
        # 优先使用客户端传来的 X-Request-ID（便于链路追踪），否则生成新的
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # 注入到当前协程上下文
        token = _request_id_var.set(request_id)

        logger.info(
            f"→ 请求开始 | request_id={request_id} "
            f"| method={request.method} | path={request.url.path}"
        )

        try:
            response = await call_next(request)
            # 在响应 Header 里返回 request_id，方便客户端/网关追踪
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # 重置 ContextVar，防止协程复用时污染
            _request_id_var.reset(token)
