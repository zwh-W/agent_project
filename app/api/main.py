# app/api/main.py
"""
FastAPI 服务入口

P0 修复版改动：
  1. 修正中间件注册顺序和注释，确保 request_id 最外层注入。
  2. 保持 CORS 在鉴权之前处理预检请求，避免浏览器 OPTIONS 被 API Key 拦截。
  3. 正常业务请求执行顺序：
       RequestContext -> CORS -> APIKey -> RateLimit -> 路由
"""
import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger
from app.api.routers import agent_router
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.request_context import RequestContextMiddleware, get_request_id
from app.api.middleware.rate_limiter import RateLimiterMiddleware

logger = get_logger(__name__)

app = FastAPI(
    title="多智能体 Agent 系统",
    description=(
        "支持 Function Calling / LangGraph / Multi-Agent / MCP 四种 Agent，"
        "含智能自动路由、ReAct 推理链追踪、工具注册中心、Prompt 版本管理"
    ),
    version="2.0.1",
)

# ── 中间件注册 ─────────────────────────────────────────────
# Starlette/FastAPI 中间件通常是“后注册先执行”。
#
# ★ [P0 FIX] 目标执行顺序：
#   1. RequestContext：最外层，保证所有请求/异常都带 request_id
#   2. CORS：优先处理浏览器预检请求，避免 OPTIONS 被鉴权拦截
#   3. APIKey：校验调用方身份
#   4. RateLimiter：鉴权通过后再计入限流
#   5. 路由
#
# 因此注册顺序要反过来写：RateLimiter -> APIKey -> CORS -> RequestContext

app.add_middleware(RateLimiterMiddleware)

enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
app.add_middleware(APIKeyMiddleware, enabled=enable_auth)

allowed_origins = getattr(settings.app, 'allowed_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(RequestContextMiddleware)

app.include_router(agent_router.router, prefix="/v1", tags=["Agent 对话"])


# ── 全局异常处理 ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id() or str(uuid.uuid4())
    logger.error(
        f"未捕获异常 | request_id={request_id} | path={request.url.path} | {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "response_code": 500,
            "response_msg": "服务内部错误，请稍后重试",
            "processing_time": 0.0,
        }
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
        }
    )


# ── 健康检查 ───────────────────────────────────────────────
@app.get("/health", summary="健康检查")
def health_check():
    from app.memory.manager import memory_manager
    from app.tools.registry import tool_registry
    from app.core.prompt_manager import prompt_manager

    return {
        "status": "ok",
        "version": "2.0.1",
        "model": settings.llm.model,
        "llm_configured": bool(settings.llm.api_key),
        "auth_enabled": enable_auth,
        "active_sessions": memory_manager.active_session_count,
        "registered_tools": [s["tool_name"] for s in tool_registry.get_stats()],
        "loaded_prompts": list(prompt_manager.get_stats().keys()),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 Agent 服务 v2.0.1...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
