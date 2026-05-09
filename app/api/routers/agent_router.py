# app/api/routers/agent_router.py
"""
Agent API 路由层

P0 修复版改动：
  1. LangGraphAgent 分支新增 pending_action 解析，修复工具返回 need_confirmation=True
     但 ChatResponse.need_confirmation 仍为 False 的问题。
  2. session_id 绑定改用 PendingActionStore.bind_session_if_unknown()，
     不再直接修改 pending_action_store._store 私有字段。
  3. /v1/confirm 执行流程改为：
       pending -> executing -> confirmed
       pending -> executing -> failed
     避免“已 confirmed 但真实工单创建失败”的状态不一致。
"""
import json
import time
from typing import Optional, Iterable, Any

from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    AgentType,
    ChatRequest, ChatResponse,
    ConfirmRequest, ConfirmResponse,
    PendingActionInfo, ToolCall,
)
from app.agents.function_calling_agent import FunctionCallingAgent
from app.agents.langgraph_agent import LangGraphAgent
from app.agents.multi_agent import MultiAgentSupervisor
from app.agents.mcp_agent import MCPAgent
from app.core.auto_router import agent_auto_router
from app.core.prompt_manager import prompt_manager
from app.tools.registry import tool_registry
from app.services.pending_action_store import pending_action_store
from app.services.ticket_service import ticket_service
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def _build_pending_info_from_tool_output(
    *,
    tool_name: str,
    tool_output: str,
    tool_input: Optional[dict],
    session_id: str,
) -> tuple[bool, Optional[PendingActionInfo]]:
    """
    ★ [P0 FIX] 统一解析工具输出中的 need_confirmation。

    支持来源：
      - FunctionCallingAgent 的 ReAct trace step.observation
      - LangGraphAgent 的 profile.tool_events ev.tool_output

    当 request_create_hr_ticket 工具返回 JSON：
      {
        "need_confirmation": true,
        "action_id": "...",
        ...
      }
    这里负责转成 ChatResponse.pending_action。
    """
    if tool_name != "request_create_hr_ticket":
        return False, None

    if not tool_output:
        return False, None

    try:
        data = json.loads(tool_output)
    except (json.JSONDecodeError, TypeError):
        return False, None

    if not data.get("need_confirmation"):
        return False, None

    action_id = data.get("action_id", "")
    if not action_id:
        return False, None

    # ★ [P0 FIX] 用公开方法绑定真实 session_id，不再直接修改 _store
    action = pending_action_store.bind_session_if_unknown(action_id, session_id)
    if action is None:
        action = pending_action_store.get_pending_action(action_id)

    pending_info = PendingActionInfo(
        action_id=action_id,
        action_type=data.get("action_type", "create_hr_ticket"),
        tool_name=tool_name,
        tool_input=tool_input or {},
        message=data.get("message", "请确认是否执行此操作"),
        created_at=action["created_at"] if action else "",
    )
    return True, pending_info


def _extract_pending_from_trace(trace, session_id: str) -> tuple[bool, Optional[PendingActionInfo]]:
    """
    从 FunctionCallingAgent 的 ReAct trace 中检测是否有 need_confirmation。
    """
    for step in trace.steps:
        ok, info = _build_pending_info_from_tool_output(
            tool_name=step.action_tool or "",
            tool_output=step.observation or "",
            tool_input=step.action_input or {},
            session_id=session_id,
        )
        if ok:
            return True, info

    return False, None


def _extract_pending_from_tool_events(tool_events: Iterable[Any], session_id: str) -> tuple[bool, Optional[PendingActionInfo]]:
    """
    ★ [P0 FIX] 从 LangGraphAgent 的 profile.tool_events 中检测 need_confirmation。

    修复点：
      旧代码只把 tool_events 转成 ToolCall 列表，没有解析 pending_action。
      导致 LangGraph 模式下即使 request_create_hr_ticket 成功生成 action_id，
      ChatResponse.need_confirmation 仍然是 False。
    """
    for ev in tool_events:
        ok, info = _build_pending_info_from_tool_output(
            tool_name=getattr(ev, "tool_name", "") or "",
            tool_output=getattr(ev, "tool_output", "") or "",
            tool_input=getattr(ev, "tool_input", {}) or {},
            session_id=session_id,
        )
        if ok:
            return True, info

    return False, None


def _raise_pending_action_error(err: str):
    """
    将 pending action 的业务异常映射为 HTTP 状态码。
    """
    if "找不到" in err:
        raise HTTPException(status_code=404, detail=err)
    if "无权操作" in err:
        raise HTTPException(status_code=403, detail=err)
    raise HTTPException(status_code=409, detail=err)


# ─────────────────────────────────────────────────────────
# POST /v1/chat
# ─────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="与 Agent 进行对话")
async def chat_endpoint(req: ChatRequest):
    start_time = time.time()

    resolved_type = agent_auto_router.route(
        user_input=req.message,
        explicit_type=req.agent_type,
    )
    agent_type_str = resolved_type.value

    logger.info(
        f"请求 | session={req.session_id[:8]}... "
        f"| 原始 agent_type={req.agent_type.value} "
        f"| 路由后={agent_type_str}"
    )

    tool_calls_result = []
    answer = ""
    need_confirmation = False
    pending_action_info: Optional[PendingActionInfo] = None

    # ─── function_calling ──────────────────────────────────
    if agent_type_str == "function_calling":
        agent = FunctionCallingAgent(session_id=req.session_id)
        answer, trace = agent.chat_with_trace(req.message)

        tool_calls_result = [
            ToolCall(
                tool_name=step.action_tool,
                tool_input=step.action_input or {},
                tool_output=step.observation or "",
            )
            for step in trace.steps
            if step.action_tool is not None
        ]

        need_confirmation, pending_action_info = _extract_pending_from_trace(
            trace, req.session_id
        )

    # ─── langgraph ─────────────────────────────────────────
    elif agent_type_str == "langgraph":
        agent = LangGraphAgent(session_id=req.session_id)
        answer, profile = agent.chat_with_profile(req.message)

        tool_calls_result = [
            ToolCall(
                tool_name=ev.tool_name,
                tool_input=ev.tool_input,
                tool_output=ev.tool_output,
            )
            for ev in profile.tool_events
        ]

        # ★ [P0 FIX] LangGraph 分支也要解析 pending_action
        need_confirmation, pending_action_info = _extract_pending_from_tool_events(
            profile.tool_events,
            req.session_id,
        )

    # ─── multi_agent ───────────────────────────────────────
    elif agent_type_str == "multi_agent":
        agent = MultiAgentSupervisor(session_id=req.session_id)
        answer = agent.chat(req.message)

    # ─── mcp ───────────────────────────────────────────────
    elif agent_type_str == "mcp":
        agent = MCPAgent(session_id=req.session_id)
        answer = await agent.chat(req.message)

    # ─── 兜底 ──────────────────────────────────────────────
    else:
        agent = FunctionCallingAgent(session_id=req.session_id)
        answer, trace = agent.chat_with_trace(req.message)
        tool_calls_result = [
            ToolCall(
                tool_name=step.action_tool,
                tool_input=step.action_input or {},
                tool_output=step.observation or "",
            )
            for step in trace.steps
            if step.action_tool is not None
        ]
        need_confirmation, pending_action_info = _extract_pending_from_trace(
            trace, req.session_id
        )

    processing_time = round(time.time() - start_time, 2)

    return ChatResponse(
        response_code=200,
        response_msg="success",
        session_id=req.session_id,
        answer=answer,
        agent_type=agent_type_str,
        processing_time=processing_time,
        tool_calls=tool_calls_result,
        messages=[],
        need_confirmation=need_confirmation,
        pending_action=pending_action_info,
    )


# ─────────────────────────────────────────────────────────
# POST /v1/confirm — Human-in-the-loop 确认接口
# ─────────────────────────────────────────────────────────

@router.post("/confirm", response_model=ConfirmResponse, summary="确认或取消待执行操作")
def confirm_endpoint(req: ConfirmRequest):
    """
    Human-in-the-loop 确认接口

    流程：
      1. 用户在 /v1/chat 里触发了工单创建意图
      2. 系统返回 need_confirmation=True 和 pending_action（含 action_id）
      3. 前端展示确认对话框
      4. 用户点击确认/取消 → 调用此接口
      5. confirm=False → pending -> cancelled
      6. confirm=True  → pending -> executing -> confirmed
      7. 如果真实业务执行失败 → executing -> failed

    异常处理：
      - action_id 不存在 → 404
      - session_id 不匹配 → 403
      - action 已处理/已过期 → 409
      - action 类型不支持 → 400
      - 业务执行失败 → 500
    """
    start_time = time.time()

    # ── 取消操作 ──────────────────────────────────────────
    if not req.confirm:
        try:
            pending_action_store.cancel_action(req.action_id, req.session_id)
        except ValueError as e:
            _raise_pending_action_error(str(e))

        logger.info(f"操作已取消 | action_id={req.action_id}")
        return ConfirmResponse(
            response_code=200,
            response_msg="操作已取消",
            processing_time=round(time.time() - start_time, 2),
            session_id=req.session_id,
            action_id=req.action_id,
            status="cancelled",
            result=None,
        )

    # ── 确认执行：第一步只标记 executing，不直接 confirmed ──
    try:
        action = pending_action_store.begin_execute_action(req.action_id, req.session_id)
    except ValueError as e:
        _raise_pending_action_error(str(e))

    action_type = action.get("action_type", "")
    tool_input = action.get("tool_input", {})
    result_data = None

    try:
        if action_type == "create_hr_ticket":
            # 唯一可以真正调用 create_ticket 的地方
            ticket = ticket_service.create_ticket(
                ticket_type=tool_input.get("ticket_type", "general_hr"),
                title=tool_input.get("title", "HR 工单"),
                description=tool_input.get("description", ""),
                session_id=req.session_id,
            )
            result_data = ticket

            # ★ [P0 FIX] 真实业务执行成功后，才 confirmed
            pending_action_store.mark_confirmed(req.action_id)
            logger.info(f"工单已创建 | ticket_id={ticket['ticket_id']} | action_id={req.action_id}")

        else:
            # ★ [P0 FIX] 不支持的 action_type 也要落 failed，避免卡在 executing
            pending_action_store.mark_failed(req.action_id, f"不支持的操作类型：{action_type}")
            raise HTTPException(
                status_code=400,
                detail=f"不支持的操作类型：{action_type}。当前仅支持：create_hr_ticket"
            )

    except HTTPException:
        raise

    except Exception as e:
        # ★ [P0 FIX] 业务执行失败时，记录 failed 状态
        try:
            pending_action_store.mark_failed(req.action_id, str(e))
        except Exception as mark_err:
            logger.error(f"标记 pending action failed 失败 | action_id={req.action_id} | error={mark_err}")

        logger.error(f"工单创建失败 | action_id={req.action_id} | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"工单创建失败：{str(e)}，请联系系统管理员"
        )

    return ConfirmResponse(
        response_code=200,
        response_msg="操作已确认执行",
        processing_time=round(time.time() - start_time, 2),
        session_id=req.session_id,
        action_id=req.action_id,
        status="confirmed",
        result=result_data,
    )


# ─────────────────────────────────────────────────────────
# 统计 / 管理接口
# ─────────────────────────────────────────────────────────

@router.get("/stats/tools", summary="工具调用统计")
def get_tool_stats():
    """返回所有工具的调用次数、成功率、平均耗时"""
    return {"response_code": 200, "tool_stats": tool_registry.get_stats()}


@router.get("/tickets", summary="查询工单列表")
def list_tickets(session_id: Optional[str] = None):
    """查询工单，可按 session_id 筛选"""
    tickets = ticket_service.list_tickets(session_id=session_id)
    return {"response_code": 200, "count": len(tickets), "tickets": tickets}


@router.post("/prompts/reload", summary="热加载 Prompt")
def reload_prompt(name: str, version: Optional[str] = None):
    try:
        prompt_manager.reload(name, version)
        return {"response_code": 200, "response_msg": f"Prompt '{name}/{version or 'current'}' 已热加载"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prompts/versions", summary="查看 Prompt 版本列表")
def list_prompt_versions(name: str):
    return {"response_code": 200, "name": name, "versions": prompt_manager.list_versions(name)}


@router.get("/prompts/stats", summary="Prompt 使用统计")
def get_prompt_stats():
    return {"response_code": 200, "prompt_stats": prompt_manager.get_stats()}
