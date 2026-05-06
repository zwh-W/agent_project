# app/api/main.py
"""
FastAPI 服务入口

★ 本版改动：
1. 注册限流中间件 RateLimiterMiddleware
2. 在 /health 接口里暴露工具统计和 Prompt 统计
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
from app.api.middleware.rate_limiter import RateLimiterMiddleware   # ★ 新增

logger = get_logger(__name__)

app = FastAPI(
    title="多智能体 Agent 系统",
    description=(
        "支持 Function Calling / LangGraph / Multi-Agent / MCP 四种 Agent，"
        "含智能自动路由、ReAct 推理链追踪、工具注册中心、Prompt 版本管理"
    ),
    version="2.0.0",
)

# ── 中间件注册（后注册先执行）────────────────────────────
# 执行顺序：RequestContext → RateLimit → APIKey → CORS → 路由

app.add_middleware(RequestContextMiddleware)             # 4. 最先执行：注入 request_id

enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
app.add_middleware(APIKeyMiddleware, enabled=enable_auth)  # 3. 鉴权

app.add_middleware(RateLimiterMiddleware)                # 2. 限流（鉴权通过后才限流）

allowed_origins = getattr(settings.app, 'allowed_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)                                                        # 1. 最后执行

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
        "version": "2.0.0",
        "model": settings.llm.model,
        "llm_configured": bool(settings.llm.api_key),
        "auth_enabled": enable_auth,
        "active_sessions": memory_manager.active_session_count,
        # ★ 工具注册情况
        "registered_tools": [s["tool_name"] for s in tool_registry.get_stats()],
        # ★ Prompt 加载情况
        "loaded_prompts": list(prompt_manager.get_stats().keys()),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 Agent 服务 v2.0.0...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )