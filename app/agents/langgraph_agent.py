# app/agents/langgraph_agent.py
"""
基于 LangGraph 构建的图结构 Agent

变更：
  v2.1  GraphProfile 新增 tool_events 列表，
        tools 节点执行后把 ToolMessage 的 name/content 记录进去，
        修复 agent_router.py 中 tool_calls 永远返回空列表的问题。
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.tools import ALL_TOOLS
from app.graph.state import AgentState

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 性能 Profile 数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class NodeProfile:
    """单个节点的执行性能数据"""
    node_name: str
    enter_time: float = 0.0
    exit_time: float = 0.0
    duration_ms: float = 0.0
    messages_produced: int = 0


# ★ 新增：工具调用事件，记录 tools 节点产生的 ToolMessage 信息
@dataclass
class ToolEvent:
    """tools 节点执行后，记录每条 ToolMessage 的核心信息"""
    tool_name: str        # 工具名
    tool_output: str      # 工具返回内容
    # tool_input 在 ToolMessage 里不直接携带（input 在 AIMessage.tool_calls 里）
    # 简化处理：input 留空，agent_router.py 从 ToolCall 构建时填 {}
    tool_input: dict = field(default_factory=dict)


@dataclass
class GraphProfile:
    """整张图的执行性能数据"""
    session_id: str
    user_input: str
    node_profiles: List[NodeProfile] = field(default_factory=list)
    tool_events: List[ToolEvent] = field(default_factory=list)   # ★ 新增
    total_duration_ms: float = 0.0
    agent_call_count: int = 0
    tool_call_count: int = 0

    def add_profile(self, profile: NodeProfile):
        self.node_profiles.append(profile)
        if profile.node_name == "agent":
            self.agent_call_count += 1
        elif profile.node_name == "tools":
            self.tool_call_count += 1

    def add_tool_event(self, event: ToolEvent):
        """★ 新增：记录一次工具调用结果"""
        self.tool_events.append(event)

    def to_readable(self) -> str:
        lines = [
            f"╔{'═'*60}",
            f"║ LangGraph 执行 Profile  |  会话: {self.session_id[:8]}...",
            f"║ 用户问题: {self.user_input[:50]}",
            f"╠{'═'*60}",
        ]
        for p in self.node_profiles:
            bar_len = int(p.duration_ms / max(self.total_duration_ms, 1) * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"║  [{p.node_name:<6}]  {p.duration_ms:6.0f}ms  {bar}")

        if self.tool_events:
            lines.append(f"╠{'═'*60}")
            lines.append(f"║ 工具调用明细（{len(self.tool_events)} 次）:")
            for ev in self.tool_events:
                preview = ev.tool_output[:60] + "..." if len(ev.tool_output) > 60 else ev.tool_output
                lines.append(f"║   [{ev.tool_name}] → {preview}")

        lines.extend([
            f"╠{'═'*60}",
            f"║  总耗时:    {self.total_duration_ms:.0f}ms",
            f"║  LLM 调用:  {self.agent_call_count} 次",
            f"║  工具调用:  {self.tool_call_count} 次",
            f"╚{'═'*60}",
        ])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Agent 本体
# ─────────────────────────────────────────────────────────

class LangGraphAgent:
    """
    基于 LangGraph 构建的图结构 Agent
    支持：节点耗时监控、工具结果持久化、工具调用事件记录
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.app = self._build_graph()
        self.last_profile: Optional[GraphProfile] = None

    def _build_graph(self):
        def call_model(state: AgentState):
            logger.debug(f"[{self.session_id}] 🧠 LangGraph: agent 节点...")
            response = self.llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        tool_node = ToolNode(ALL_TOOLS)

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def chat(self, user_input: str) -> str:
        answer, _ = self.chat_with_profile(user_input)
        return answer

    def chat_with_profile(self, user_input: str) -> tuple[str, GraphProfile]:
        """
        带性能 Profile 和工具事件记录的对话入口
        """
        logger.info(f"[{self.session_id}] 用户输入 (LangGraph): {user_input}")
        self.memory.add_user_message(user_input)

        profile = GraphProfile(session_id=self.session_id, user_input=user_input)
        total_start = time.time()

        initial_state = {"messages": self.memory.get_context(current_user_input=user_input)}
        final_answer = ""
        node_enter_times: Dict[str, float] = {}

        for event in self.app.stream(
            initial_state,
            {"recursion_limit": settings.agent.max_iterations * 2}
        ):
            for node_name, state_update in event.items():
                now = time.time()

                if node_name not in node_enter_times:
                    node_enter_times[node_name] = now

                node_profile = NodeProfile(
                    node_name=node_name,
                    enter_time=node_enter_times[node_name],
                    exit_time=now,
                    duration_ms=(now - node_enter_times[node_name]) * 1000,
                    messages_produced=len(state_update.get("messages", [])),
                )
                profile.add_profile(node_profile)
                node_enter_times[node_name] = now

                logger.debug(
                    f"[{self.session_id}] 节点 '{node_name}' 完成，"
                    f"耗时 {node_profile.duration_ms:.0f}ms"
                )

                msgs = state_update.get("messages", [])
                if not msgs:
                    continue

                latest_msg = msgs[-1]

                # agent 节点且无工具调用 → 最终答案
                if node_name == "agent" and not getattr(latest_msg, "tool_calls", None):
                    final_answer = latest_msg.content

                # ★ tools 节点：记录 ToolMessage 到 profile.tool_events
                # 同时同步写入持久化 memory
                if node_name == "tools":
                    for msg in msgs:
                        # 同步到 memory（修复原版工具结果丢失问题）
                        self.memory.add_raw_message(msg)
                        # 记录 ToolEvent
                        if isinstance(msg, ToolMessage):
                            content_str = msg.content
                            if isinstance(content_str, list):
                                content_str = " ".join(
                                    b.get("text", "") if isinstance(b, dict) else str(b)
                                    for b in content_str
                                )
                            profile.add_tool_event(ToolEvent(
                                tool_name=msg.name or "unknown",
                                tool_output=str(content_str),
                            ))

        profile.total_duration_ms = (time.time() - total_start) * 1000
        logger.info(f"\n{profile.to_readable()}")

        self.memory.add_ai_message(final_answer)
        logger.info(f"[{self.session_id}] 🎉 LangGraph 最终回答: {final_answer[:50]}...")
        self.last_profile = profile

        return final_answer, profile