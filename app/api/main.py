# app/api/main.py
"""
FastAPI 服务入口

启动命令：
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import time
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="多智能体 Agent 系统",
    description="支持 Function Calling / ReAct / LangGraph / Multi-Agent / MCP 五种 Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 全局异常处理（和 RAG 项目保持一致）────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
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
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": str(uuid.uuid4()),
            "response_code": exc.status_code,
            "response_msg": exc.detail,
            "processing_time": 0.0,
        }
    )


# ── 健康检查 ───────────────────────────────────────────────
@app.get("/health", summary="健康检查")
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": settings.llm.model,
        "llm_configured": bool(settings.llm.api_key),
    }


# ── 注册路由（后续实现各模块后逐步取消注释）──────────────
# from app.api.routers import agent
# app.include_router(agent.router, prefix="/v1", tags=["Agent 对话"])


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 Agent 服务...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )