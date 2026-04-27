# app/api/routers/agent_router.py
import time
from fastapi import APIRouter
from app.api.schemas import ChatRequest, ChatResponse
from app.agents.function_calling_agent import FunctionCallingAgent
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="与 Agent 进行普通对话")
async def chat_endpoint(req: ChatRequest):
    start_time = time.time()
    # 注意这里加了 .value
    agent_type_str = req.agent_type.value

    logger.info(f"收到 API 请求 | Session: {req.session_id} | Type: {agent_type_str}")

    # 根据 agent_type 决定调用哪个 Agent
    if agent_type_str == "function_calling":
        agent = FunctionCallingAgent(session_id=req.session_id)
        answer = agent.chat(req.message)
    else:
        # 默认兜底
        agent = FunctionCallingAgent(session_id=req.session_id)
        answer = agent.chat(req.message)

    processing_time = round(time.time() - start_time, 2)

    return ChatResponse(
        response_code=200,
        response_msg="success",
        session_id=req.session_id,
        answer=answer,
        agent_type=agent_type_str,  # 这里也改成 .value
        processing_time=processing_time,
        tool_calls=[],
        messages=[]
    )