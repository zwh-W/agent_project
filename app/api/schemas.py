# app/api/schemas.py
"""
API 请求/响应数据结构

变更记录：
  v2.1  新增 AgentType.auto（明确的自动路由枚举值，取代用 function_calling 暗示未指定的做法）
        新增 PendingActionInfo / ConfirmRequest / ConfirmResponse（工单人工确认流程）
        ChatResponse 新增 need_confirmation / pending_action 字段
"""
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────
# 基础响应壳
# ─────────────────────────────────────────────────────────
class BaseResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_code: int
    response_msg: str
    processing_time: float


# ─────────────────────────────────────────────────────────
# ★ [修改] AgentType 新增 auto 枚举值
#
# 原来的做法：default=AgentType.function_calling，用 function_calling
#   同时表示"用户明确要 FC"和"用户没有指定想自动路由"，语义混淆。
# 现在：auto 是独立枚举值，含义清晰：
#   - auto          → 走三层漏斗自动路由
#   - function_calling → 用户明确要 FC，尊重用户意图，跳过路由
#   - langgraph     → 同上
#   - multi_agent   → 同上
#   - mcp           → 同上
# ─────────────────────────────────────────────────────────
class AgentType(str, Enum):
    auto = "auto"  # ★ 新增：自动路由，系统决定用哪个 Agent
    function_calling = "function_calling"  # 手写 Function Calling + ReAct trace
    langgraph = "langgraph"  # LangGraph 图结构 Agent
    multi_agent = "multi_agent"  # Multi-Agent Supervisor 模式
    mcp = "mcp"  # MCP 协议 Agent


# ─────────────────────────────────────────────────────────
# 对话消息
# ─────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(description="user / assistant / system")
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()


# ─────────────────────────────────────────────────────────
# 工具调用溯源
# ─────────────────────────────────────────────────────────
class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_output: str


# ─────────────────────────────────────────────────────────
# ★ 新增：等待用户确认的操作信息（Human-in-the-loop）
#
# 当 Agent 调用 request_create_hr_ticket 工具时，
# 工具不直接创建工单，而是生成一个 PendingAction。
# Agent 在 ChatResponse 里把这个信息返回给前端，
# 前端展示"是否确认创建工单？"给用户，
# 用户点确认后再调 POST /v1/confirm 真正执行。
# ─────────────────────────────────────────────────────────
class PendingActionInfo(BaseModel):
    action_id: str = Field(description="pending action 的唯一 ID")
    action_type: str = Field(description="操作类型，如 create_hr_ticket")
    tool_name: str = Field(description="发起操作的工具名")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="工具的入参")
    message: str = Field(description="展示给用户的确认提示文字")
    created_at: str = Field(description="创建时间 ISO 格式")


# ─────────────────────────────────────────────────────────
# ★ 新增：/v1/confirm 接口的请求体
# ─────────────────────────────────────────────────────────
class ConfirmRequest(BaseModel):
    session_id: str = Field(description="会话 ID，必须与创建 pending action 时一致")
    action_id: str = Field(description="pending action 的 ID")
    confirm: bool = Field(description="true=确认执行，false=取消")

    @field_validator("action_id", "session_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("字段不能为空")
        return v.strip()


# ─────────────────────────────────────────────────────────
# ★ 新增：/v1/confirm 接口的响应体
# ─────────────────────────────────────────────────────────
class ConfirmResponse(BaseResponse):
    session_id: str
    action_id: str
    status: str  # confirmed / cancelled
    result: Optional[Dict[str, Any]] = None  # 成功创建工单时返回工单详情


# ─────────────────────────────────────────────────────────
# 对话请求
# ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(description="用户输入")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话 ID，不传则自动生成"
    )
    # ★ [修改] 默认值从 function_calling 改为 auto
    #    语义更清晰：不传 agent_type = 让系统自动决定
    agent_type: AgentType = Field(
        default=AgentType.auto,
        description="使用哪种 Agent；auto 表示由系统自动路由"
    )

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息不能为空")
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def session_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            return str(uuid.uuid4())
        return v.strip()


# ─────────────────────────────────────────────────────────
# 对话响应
# ─────────────────────────────────────────────────────────
class ChatResponse(BaseResponse):
    session_id: str
    answer: str
    agent_type: str
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用溯源")
    messages: List[ChatMessage] = Field(default_factory=list, description="完整对话历史")

    # ★ 新增：Human-in-the-loop 确认相关字段
    # 当 Agent 生成了一个需要用户二次确认的操作时，这两个字段会被填充。
    # 前端收到 need_confirmation=True 后，应展示确认按钮，
    # 用户点击后调用 POST /v1/confirm。
    need_confirmation: bool = Field(
        default=False,
        description="是否需要用户确认才能执行（如创建工单）"
    )
    pending_action: Optional[PendingActionInfo] = Field(
        default=None,
        description="等待确认的操作详情，need_confirmation=True 时有值"
    )
