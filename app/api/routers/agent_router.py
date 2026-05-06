# app/api/routers/agent_router.py
"""
Agent 路由层

★ 本版改动：
1. 接入 AgentAutoRouter，支持 agent_type="auto" 自动路由
2. FunctionCallingAgent 的 ReAct trace 写入响应的 tool_calls 字段
3. 新增 /stats 端点，暴露工具调用统计（面试展示用）
4. 新增 /prompts/reload 端点，支持 Prompt 热加载
"""
import time
from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, AgentType, ToolCall
from app.agents.function_calling_agent import FunctionCallingAgent
from app.agents.langgraph_agent import LangGraphAgent
from app.agents.multi_agent import MultiAgentSupervisor
from app.agents.mcp_agent import MCPAgent
from app.core.auto_router import agent_auto_router
from app.core.prompt_manager import prompt_manager
from app.tools.registry import tool_registry
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="与 Agent 进行对话")
async def chat_endpoint(req: ChatRequest):
    start_time = time.time()

    # ★ 智能路由：如果 agent_type 是默认值或明确传 "auto"，使用自动路由
    # 注意：schemas.py 里 default=AgentType.function_calling，
    # 这里约定：用户不传 agent_type 时，自动路由接管
    # （生产里可以加一个 AgentType.auto 枚举值，这里用 function_calling 作为"未指定"的信号）
    resolved_type = agent_auto_router.route(
        user_input=req.message,
        explicit_type=req.agent_type,
    )

    agent_type_str = resolved_type.value
    logger.info(
        f"收到请求 | Session: {req.session_id} | "
        f"原始类型: {req.agent_type.value} | 路由后类型: {agent_type_str}"
    )

    tool_calls_result = []
    answer = ""

    if agent_type_str == "function_calling":
        agent = FunctionCallingAgent(session_id=req.session_id)
        # ★ 使用带 trace 的接口，把推理过程写入响应
        answer, trace = agent.chat_with_trace(req.message)
        # 将 trace 中的工具调用转换为 API 响应格式
        tool_calls_result = [
            ToolCall(
                tool_name=step.action_tool,
                tool_input=step.action_input or {},
                tool_output=step.observation or "",
            )
            for step in trace.steps
            if step.action_tool is not None
        ]

    elif agent_type_str == "langgraph":
        agent = LangGraphAgent(session_id=req.session_id)
        # ★ 使用带 profile 的接口
        answer, profile = agent.chat_with_profile(req.message)
        # LangGraph 的工具调用从 profile 中提取（tools 节点的执行记录）
        # 这里简化处理，只记录调用了哪些节点
        tool_calls_result = []  # TODO: 从 profile 中提取详细工具调用信息

    elif agent_type_str == "multi_agent":
        agent = MultiAgentSupervisor(session_id=req.session_id)
        answer = agent.chat(req.message)

    elif agent_type_str == "mcp":
        agent = MCPAgent(session_id=req.session_id)
        answer = await agent.chat(req.message)

    else:
        agent = FunctionCallingAgent(session_id=req.session_id)
        answer, _ = agent.chat_with_trace(req.message)

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
    )


@router.get("/stats/tools", summary="工具调用统计")
def get_tool_stats():
    """
    ★ 新增端点：返回所有工具的调用统计

    面试展示价值：
        当面试官问"你怎么知道搜索工具的成功率"时，
        你可以打开这个接口，实时展示数据。
    """
    return {
        "response_code": 200,
        "tool_stats": tool_registry.get_stats(),
    }


@router.post("/prompts/reload", summary="热加载 Prompt")
def reload_prompt(name: str, version: str = None):
    """
    ★ 新增端点：热加载指定 Prompt，无需重启服务

    使用场景：
        运营同学修改了 prompts/system/v2.txt 后，
        调用这个接口立即生效，不需要发版。

    参数：
        name:    prompt 名称，如 "system"、"supervisor"
        version: 版本号，如 "v2"（None 表示重载当前版本）
    """
    try:
        prompt_manager.reload(name, version)
        return {
            "response_code": 200,
            "response_msg": f"Prompt '{name}/{version or 'current'}' 已热加载",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prompts/versions", summary="查看 Prompt 版本列表")
def list_prompt_versions(name: str):
    """查看某个 Prompt 的所有可用版本"""
    versions = prompt_manager.list_versions(name)
    return {
        "response_code": 200,
        "name": name,
        "versions": versions,
    }


@router.get("/prompts/stats", summary="Prompt 使用统计")
def get_prompt_stats():
    """查看各 Prompt 版本的使用次数"""
    return {
        "response_code": 200,
        "prompt_stats": prompt_manager.get_stats(),
    }