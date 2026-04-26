# app/api/schemas.py
"""
API 请求/响应数据结构

设计原则：
1. 所有接口统一返回格式
2. agent_type 控制用哪种 Agent 处理
3. session_id 支持多轮对话
"""
import uuid
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class BaseResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_code: int
    response_msg: str
    processing_time: float


# ── Agent 类型枚举 ─────────────────────────────────────────
class AgentType(str, Enum):
    function_calling = "function_calling"  # 手写 Function Calling（最基础）
    react = "react"  # ReAct 推理链
    langgraph = "langgraph"  # LangGraph 单 Agent
    multi_agent = "multi_agent"  # Multi-Agent Supervisor 模式
    mcp = "mcp"  # MCP 协议 Agent


# ── 对话消息 ───────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(description="user / assistant / system")
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()


# ── 工具调用溯源（Agent 用了哪些工具）────────────────────
class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str


# ── 请求 ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(description="用户输入")
    session_id: Optional[str] = Field(default=None, description="会话ID，多轮对话时传入")
    agent_type: AgentType = Field(default=AgentType.react, description="使用哪种 Agent")

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("消息不能为空")
        return v.strip()


# ── 响应 ───────────────────────────────────────────────────
class ChatResponse(BaseResponse):
    session_id: str
    answer: str
    agent_type: str
    tool_calls: List[ToolCall] = Field(default=[], description="Agent 调用的工具记录（溯源）")
    messages: List[ChatMessage] = Field(default=[], description="完整对话历史")
