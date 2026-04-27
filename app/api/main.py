# app/api/main.py
"""
FastAPI 服务入口

启动命令：
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import time
import uuid
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger
from app.api.routers import agent_router
# ★ [新增] 导入中间件
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.request_context import RequestContextMiddleware, get_request_id

logger = get_logger(__name__)

app = FastAPI(
    title="多智能体 Agent 系统",
    description="支持 Function Calling / ReAct / LangGraph / Multi-Agent / MCP 五种 Agent",
    version="1.0.0",
)

# ★ [修改] 中间件注册顺序很重要：后注册的先执行（栈结构）
# 执行顺序：RequestContext → APIKey → CORS → 业务路由
# 1. 先注入 request_id（后续所有日志都能带上它）
# 2. 再做鉴权（鉴权日志里也能带 request_id）

# ★ [新增] 请求 ID 追踪中间件（必须最先执行，所以最后注册）
app.add_middleware(RequestContextMiddleware)

# ★ [新增] API Key 鉴权中间件
# 通过环境变量 ENABLE_AUTH=false 可以在开发环境关闭鉴权
enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
app.add_middleware(APIKeyMiddleware, enabled=enable_auth)

# ★ [修改] CORS 改为从配置读取允许的 origins，而不是 "*"
# 原因："*" 允许任何域名跨域请求，生产环境存在 CSRF 风险
# 开发环境可以在 config.yaml 里设置 allowed_origins: ["*"]
allowed_origins = getattr(settings.app, 'allowed_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],  # ★ [修改] 只开放需要的方法
    allow_headers=["*"],
)

app.include_router(agent_router.router, prefix="/v1", tags=["Agent 对话"])


# ── 全局异常处理 ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # ★ [修改] request_id 从上下文获取，保证和请求链路一致
    # 原来每次 exception 都生成新的 request_id，无法和前面的日志关联
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
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": settings.llm.model,
        "llm_configured": bool(settings.llm.api_key),
        "auth_enabled": enable_auth,
        # ★ [新增] 返回活跃会话数，方便监控
        "active_sessions": memory_manager.active_session_count,
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 Agent 服务...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
