# app/agents/langgraph_agent.py
"""
基于 LangGraph 构建的图结构 Agent

优化点：
1. 精确节点耗时监控：
   - 不再依赖 stream 事件间隔估算耗时
   - 在 agent / tools 节点函数内部真正执行前后打点
   - 更准确区分 LLM 调用耗时和工具执行耗时

2. 工具调用结果同步写入持久化 memory：
   - 保存中间 tool_call AIMessage 和 ToolMessage
   - 便于后续多轮对话继续使用工具执行上下文

3. 最大步数防护：
   - 使用 recursion_limit 防止 LangGraph 无限循环
   - 捕获 GraphRecursionError，返回可控错误信息

4. 可观测性增强：
   - 记录 agent 调用次数
   - 记录 tools 节点进入次数
   - 记录真实工具调用次数
   - 输出节点耗时 profile

用 LangGraph 把 Agent 拆成两个节点：
agent 节点负责调用大模型，tools 节点负责执行工具；
如果模型还要调用工具，就继续循环；如果模型不再调用工具，
就结束并返回答案。
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.tools import ALL_TOOLS
from app.graph.state import AgentState

logger = get_logger(__name__)


# 用 ContextVar 保存当前请求的 profile，避免多请求并发时互相污染
_current_profile: ContextVar[Optional["GraphProfile"]] = ContextVar(
    "current_langgraph_profile",
    default=None,
)


# ──────────────────────────────────────────────
# 性能 Profile 数据结构
# ──────────────────────────────────────────────

@dataclass
class NodeProfile:
    """单个节点的执行性能数据"""
    node_name: str
    enter_time: float = 0.0
    exit_time: float = 0.0
    duration_ms: float = 0.0

    # 节点产生的消息数量
    messages_produced: int = 0

    # tools 节点中实际执行的工具调用数量
    tool_call_count: int = 0

    # 节点是否执行成功
    success: bool = True

    # 异常信息
    error: Optional[str] = None


@dataclass
class GraphProfile:
    """整张图的执行性能数据"""
    session_id: str
    user_input: str
    node_profiles: List[NodeProfile] = field(default_factory=list)

    total_duration_ms: float = 0.0

    # agent 节点被调用次数，也就是 LLM 调用次数
    agent_call_count: int = 0

    # tools 节点被进入次数
    tool_node_count: int = 0

    # 实际工具调用次数
    tool_call_count: int = 0

    def add_profile(self, profile: NodeProfile):
        self.node_profiles.append(profile)

        if profile.node_name == "agent":
            self.agent_call_count += 1

        elif profile.node_name == "tools":
            self.tool_node_count += 1
            self.tool_call_count += profile.tool_call_count

    def to_readable(self) -> str:
        """生成人类可读的性能分析文本"""
        lines = [
            f"╔{'═' * 70}",
            f"║ LangGraph 执行 Profile  |  会话: {self.session_id[:8]}...",
            f"║ 用户问题: {self.user_input[:60]}",
            f"╠{'═' * 70}",
        ]

        for p in self.node_profiles:
            if self.total_duration_ms > 0:
                ratio = p.duration_ms / self.total_duration_ms
            else:
                ratio = 0

            bar_len = min(int(ratio * 40), 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            status = "OK" if p.success else "ERR"

            lines.append(
                f"║  [{p.node_name:<6}] "
                f"{p.duration_ms:7.0f}ms  "
                f"{bar}  "
                f"msgs={p.messages_produced:<2} "
                f"tools={p.tool_call_count:<2} "
                f"{status}"
            )

            if p.error:
                lines.append(f"║      error: {p.error[:100]}")

        lines.extend([
            f"╠{'═' * 70}",
            f"║  总耗时:        {self.total_duration_ms:.0f}ms",
            f"║  LLM 调用:      {self.agent_call_count} 次",
            f"║  tools 节点:    {self.tool_node_count} 次",
            f"║  实际工具调用:  {self.tool_call_count} 次",
            f"╚{'═' * 70}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为字典，方便 API 返回或日志落库"""
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "total_duration_ms": self.total_duration_ms,
            "agent_call_count": self.agent_call_count,
            "tool_node_count": self.tool_node_count,
            "tool_call_count": self.tool_call_count,
            "node_profiles": [
                {
                    "node_name": p.node_name,
                    "duration_ms": p.duration_ms,
                    "messages_produced": p.messages_produced,
                    "tool_call_count": p.tool_call_count,
                    "success": p.success,
                    "error": p.error,
                }
                for p in self.node_profiles
            ],
        }


# ──────────────────────────────────────────────
# Agent 本体
# ──────────────────────────────────────────────

class LangGraphAgent:
    """
    基于 LangGraph 构建的图结构 Agent

    图结构：
        START
          ↓
        agent 节点：调用绑定工具后的 LLM
          ↓
        tools_condition：
            - 如果有 tool_calls → tools
            - 如果没有 tool_calls → END
          ↓
        tools 节点：执行工具
          ↓
        回到 agent 节点
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)

        self.llm = get_langchain_llm()
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        self.app = self._build_graph()
        self.last_profile: Optional[GraphProfile] = None

    # ──────────────────────────────────────────
    # 通用节点计时包装器
    # ──────────────────────────────────────────

    def _run_timed_node(
        self,
        node_name: str,
        func: Callable[[AgentState], Dict[str, Any]],
        state: AgentState,
    ) -> Dict[str, Any]:
        """
        在节点内部真实执行前后打点，得到更准确的节点耗时。

        这里记录的是：
        - agent 节点真实 LLM 调用耗时
        - tools 节点真实工具执行耗时
        """
        start = time.perf_counter()
        enter_time = time.time()

        success = True
        error_msg: Optional[str] = None
        result: Dict[str, Any] = {}

        try:
            result = func(state)
            return result

        except Exception as exc:
            success = False
            error_msg = str(exc)
            logger.exception(f"[{self.session_id}] 节点 {node_name} 执行失败: {exc}")
            raise

        finally:
            end = time.perf_counter()
            exit_time = time.time()

            messages = []
            if isinstance(result, dict):
                messages = result.get("messages", []) or []

            tool_call_count = 0
            if node_name == "tools":
                # ToolNode 一般会返回 ToolMessage 列表
                tool_call_count = len(messages)

            profile = _current_profile.get()
            if profile is not None:
                node_profile = NodeProfile(
                    node_name=node_name,
                    enter_time=enter_time,
                    exit_time=exit_time,
                    duration_ms=(end - start) * 1000,
                    messages_produced=len(messages),
                    tool_call_count=tool_call_count,
                    success=success,
                    error=error_msg,
                )
                profile.add_profile(node_profile)

                logger.debug(
                    f"[{self.session_id}] 节点 {node_name} 完成，"
                    f"耗时 {node_profile.duration_ms:.0f}ms，"
                    f"messages={node_profile.messages_produced}，"
                    f"tools={node_profile.tool_call_count}，"
                    f"success={node_profile.success}"
                )

    # ──────────────────────────────────────────
    # 构建 LangGraph
    # ──────────────────────────────────────────

    def _build_graph(self):
        """构建 LangGraph 图结构"""

        def call_model_inner(state: AgentState) -> Dict[str, Any]:
            """agent 节点内部逻辑：调用绑定工具后的 LLM"""
            logger.debug(f"[{self.session_id}] 进入 agent 节点，调用 LLM...")

            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)

            logger.debug(
                f"[{self.session_id}] agent 节点返回："
                f"content_len={len(response.content or '')}, "
                f"tool_calls={len(getattr(response, 'tool_calls', []) or [])}"
            )

            return {"messages": [response]}

        def call_model(state: AgentState) -> Dict[str, Any]:
            """带真实耗时统计的 agent 节点"""
            return self._run_timed_node(
                node_name="agent",
                func=call_model_inner,
                state=state,
            )

        # LangGraph 预置 ToolNode：
        # 自动读取 AIMessage.tool_calls，执行对应工具，并返回 ToolMessage
        tool_node = ToolNode(ALL_TOOLS)

        def call_tools_inner(state: AgentState) -> Dict[str, Any]:
            """tools 节点内部逻辑：执行工具"""
            logger.debug(f"[{self.session_id}] 进入 tools 节点，执行工具...")
            return tool_node.invoke(state)

        def call_tools(state: AgentState) -> Dict[str, Any]:
            """带真实耗时统计的 tools 节点"""
            return self._run_timed_node(
                node_name="tools",
                func=call_tools_inner,
                state=state,
            )

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)

        workflow.add_edge(START, "agent")

        # 如果 agent 输出里有 tool_calls，则进入 tools；
        # 如果没有 tool_calls，则自动进入 END。
        workflow.add_conditional_edges("agent", tools_condition)

        # 工具执行完成后，回到 agent，让模型基于工具结果继续判断或总结。
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # ──────────────────────────────────────────
    # 对外调用入口
    # ──────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """普通对话入口，只返回最终答案"""
        answer, _ = self.chat_with_profile(user_input)
        return answer

    def chat_with_profile(self, user_input: str) -> tuple[str, GraphProfile]:
        """
        带性能 Profile 的对话入口

        Returns:
            (最终答案, 图执行性能数据)
        """
        logger.info(f"[{self.session_id}] 用户输入 (LangGraph): {user_input}")

        profile = GraphProfile(
            session_id=self.session_id,
            user_input=user_input,
        )
        token = _current_profile.set(profile)

        total_start = time.perf_counter()

        final_answer = ""

        try:
            # 1. 写入用户消息
            self.memory.add_user_message(user_input)

            # 2. 构造初始状态
            initial_state = {
                "messages": self.memory.get_context(current_user_input=user_input)
            }

            recursion_limit = max(settings.agent.max_iterations * 2, 2)

            # 3. 执行 LangGraph
            for event in self.app.stream(
                initial_state,
                {"recursion_limit": recursion_limit},
            ):
                logger.debug(f"[{self.session_id}] LangGraph event: {event.keys()}")

                for node_name, state_update in event.items():
                    messages = state_update.get("messages", []) or []

                    if not messages:
                        continue

                    latest_msg = messages[-1]

                    # 4. 持久化中间消息
                    #
                    # agent 节点如果产生 tool_calls，则把这条 AIMessage 写入 memory，
                    # 因为后续 ToolMessage 需要和它的 tool_call_id 对应。
                    if node_name == "agent":
                        if getattr(latest_msg, "tool_calls", None):
                            self.memory.add_raw_message(latest_msg)

                        # agent 节点没有 tool_calls，说明这是最终答案。
                        else:
                            final_answer = latest_msg.content or ""

                    # tools 节点返回 ToolMessage，需要写入 memory。
                    elif node_name == "tools":
                        for msg in messages:
                            self.memory.add_raw_message(msg)

            # 5. 如果没有拿到最终答案，给一个兜底结果
            if not final_answer:
                final_answer = "未能生成有效回答，请尝试重新提问或补充更多信息。"

            # 6. 写入最终 AI 回复
            self.memory.add_ai_message(final_answer)

            return final_answer, profile

        except GraphRecursionError:
            final_answer = (
                "⚠️ Agent 执行超过最大递归 / 迭代限制，"
                "已强制停止。请尝试缩小问题范围或减少工具调用。"
            )
            logger.warning(f"[{self.session_id}] {final_answer}")

            self.memory.add_ai_message(final_answer)
            return final_answer, profile

        except Exception as exc:
            final_answer = f"❌ Agent 执行异常：{str(exc)}"
            logger.exception(f"[{self.session_id}] {final_answer}")

            self.memory.add_ai_message(final_answer)
            return final_answer, profile

        finally:
            profile.total_duration_ms = (time.perf_counter() - total_start) * 1000

            logger.info(f"\n{profile.to_readable()}")
            logger.info(f"[{self.session_id}] LangGraph 最终回答: {final_answer[:80]}...")

            self.last_profile = profile
            _current_profile.reset(token)