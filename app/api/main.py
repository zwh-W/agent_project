# app/api/main.py
"""
FastAPI 服务入口

当前版本：v2.0.2

包含能力：
1. 注册 RequestContext / CORS / APIKey / RateLimiter 中间件。
2. 在 /health 接口暴露工具统计、Prompt 统计、LLM 配置状态。
3. 全局异常处理统一返回 request_id。
4. Swagger UI 支持 X-API-Key Authorize 按钮，方便在 /docs 页面直接测试接口。

中间件执行顺序说明：
  Starlette/FastAPI 中间件通常是“后注册先执行”。

  目标执行顺序：
    RequestContext -> CORS -> APIKey -> RateLimit -> 路由

  设计原因：
    - RequestContext 最外层：保证所有请求和异常都带 request_id。
    - CORS 靠前：优先处理浏览器 OPTIONS 预检请求。
    - APIKey 在 RateLimit 前：非法请求不会进入业务接口。
    - RateLimit 在鉴权后：主要限制合法调用方的请求频率。
"""
import os
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.rate_limiter import RateLimiterMiddleware
from app.api.middleware.request_context import RequestContextMiddleware, get_request_id
from app.api.routers import agent_router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="多智能体 Agent 系统",
    description=(
        "支持 Function Calling / LangGraph / Multi-Agent / MCP 四种 Agent，"
        "含智能自动路由、ReAct 推理链追踪、工具注册中心、Prompt 版本管理、"
        "Human-in-the-loop 工单确认、API Key 鉴权与限流。"
    ),
    version="2.0.2",
)

# ─────────────────────────────────────────────────────────
# 中间件注册
# ─────────────────────────────────────────────────────────
# 由于 Starlette/FastAPI 中间件通常是“后注册先执行”，
# 为了实现：
#   RequestContext -> CORS -> APIKey -> RateLimit -> 路由
# 这里按反向顺序注册：
#   RateLimit -> APIKey -> CORS -> RequestContext

app.add_middleware(RateLimiterMiddleware)

enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
app.add_middleware(APIKeyMiddleware, enabled=enable_auth)

allowed_origins = getattr(settings.app, "allowed_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(RequestContextMiddleware)

# ─────────────────────────────────────────────────────────
# 路由注册
# ─────────────────────────────────────────────────────────
app.include_router(agent_router.router, prefix="/v1", tags=["Agent 对话"])


# ─────────────────────────────────────────────────────────
# Swagger UI API Key 支持
# ─────────────────────────────────────────────────────────
def custom_openapi():
    """
    让 Swagger UI 显示 Authorize 按钮，并把 X-API-Key 自动带到 /v1/* 请求里。

    注意：
      实际鉴权仍然由 APIKeyMiddleware 完成。
      这里仅用于告诉 Swagger UI：请求需要 X-API-Key Header。
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "ApiKeyAuth"
    ] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "请输入你的 API Key，例如：dev-key-123",
    }

    # 只给 /v1/* 接口加安全声明。
    # /health、/docs、/openapi.json、/redoc 仍由中间件豁免或正常公开。
    for path, path_item in openapi_schema.get("paths", {}).items():
        if not path.startswith("/v1/"):
            continue

        for method, operation in path_item.items():
            if method.lower() in {"get", "post", "put", "delete", "patch"}:
                operation["security"] = [{"ApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ─────────────────────────────────────────────────────────
# 全局异常处理
# ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id() or str(uuid.uuid4())
    logger.error(
        f"未捕获异常 | request_id={request_id} | path={request.url.path} | {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "response_code": 500,
            "response_msg": "服务内部错误，请稍后重试",
            "processing_time": 0.0,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = get_request_id() or str(uuid.uuid4())
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id,
            "response_code": exc.status_code,
            "response_msg": exc.detail,
            "processing_time": 0.0,
        },
    )


# ─────────────────────────────────────────────────────────
# 健康检查
# ─────────────────────────────────────────────────────────
@app.get("/health", summary="健康检查")
def health_check():
    from app.core.prompt_manager import prompt_manager
    from app.memory.manager import memory_manager
    from app.tools.registry import tool_registry

    return {
        "status": "ok",
        "version": "2.0.2",
        "model": settings.llm.model,
        "llm_configured": bool(settings.llm.api_key),
        "auth_enabled": enable_auth,
        "active_sessions": memory_manager.active_session_count,
        "registered_tools": [s["tool_name"] for s in tool_registry.get_stats()],
        "loaded_prompts": list(prompt_manager.get_stats().keys()),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("启动 Agent 服务 v2.0.2...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
