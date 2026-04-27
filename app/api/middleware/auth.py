# app/api/middleware/auth.py
"""
★ [新增模块] API Key 鉴权中间件

原因：原项目完全没有鉴权，allow_origins=["*"] 全开，任何人都能无限次调用
你的 LLM 接口，产生费用损失。这是生产环境的基本安全要求，
也是面试官必问的"你的系统怎么保证安全"的答案。

设计：
- Header 方式：X-API-Key: your-key（RESTful 标准）
- 支持多个合法 key（团队共享场景）
- key 从环境变量读取，不写进代码
- 健康检查 /health 端点豁免鉴权
"""
import os
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger

logger = get_logger(__name__)

# 豁免鉴权的路径（健康检查不需要 key）
_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    API Key 鉴权中间件
    从环境变量 AGENT_API_KEYS 读取合法 key（逗号分隔支持多个）
    示例：AGENT_API_KEYS=key-abc123,key-def456
    """

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        # ★ 支持多 key，逗号分隔
        raw_keys = os.getenv("AGENT_API_KEYS", "")
        self.valid_keys = set(k.strip() for k in raw_keys.split(",") if k.strip())

        if enabled and not self.valid_keys:
            logger.warning(
                "⚠️ 鉴权已启用但 AGENT_API_KEYS 未配置，所有请求将被拒绝！\n"
                "请在 .env 中配置：AGENT_API_KEYS=your-secret-key"
            )

    async def dispatch(self, request: Request, call_next):
        # 不启用鉴权（开发模式）
        if not self.enabled:
            return await call_next(request)

        # 豁免路径
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # 从 Header 读取 API Key
        api_key = request.headers.get("X-API-Key", "").strip()

        if not api_key:
            logger.warning(f"请求缺少 X-API-Key Header | path={request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "response_code": 401,
                    "response_msg": "缺少 API Key，请在请求 Header 中添加 X-API-Key",
                }
            )

        if api_key not in self.valid_keys:
            logger.warning(f"无效的 API Key | path={request.url.path} | key={api_key[:8]}...")
            return JSONResponse(
                status_code=403,
                content={
                    "response_code": 403,
                    "response_msg": "API Key 无效或已过期",
                }
            )

        return await call_next(request)
