# app/agents/multi_agent.py
"""
企业级 Multi-Agent 架构 (主管-员工模式)

★ 改动：Researcher 和 Calculator 各自从 ToolRegistry 按标签取工具，
  而不是两个 Agent 共用同一套工具。

为什么这很重要：
    给 Calculator Agent 绑定 web_search 工具，会让 LLM 在
    "应该用计算器还是去搜索" 之间产生不必要的犹豫，
    增加 token 消耗和出错概率。
    角色职责明确，工具集也应该明确。
"""
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.core.prompt_manager import prompt_manager
from app.memory.manager import memory_manager
from app.graph.state import AgentState
from app.tools.registry import tool_registry   # ★ 改为从注册中心获取

logger = get_logger(__name__)


class MultiAgentSupervisor:
    """企业级 Multi-Agent 架构 (主管-员工模式)"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()

        # ★ 按标签分别获取工具子集
        self.search_tools = tool_registry.get_tools(tags={"search"})
        self.math_tools = tool_registry.get_tools(tags={"math"})

        self.app = self._build_graph()

    def _route_by_supervisor(self, state: AgentState) -> Literal["Researcher", "Calculator", "__end__"]:
        logger.info(f"[{self.session_id}] 👔 主管路由决策中...")

        try:
            supervisor_prompt = prompt_manager.get("supervisor")
        except Exception:
            supervisor_prompt = (
                "你管理 Researcher 和 Calculator。"
                "根据对话决定下一步。完成则输出 FINISH。只输出三词之一。"
            )

        messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
        decision = self.llm.invoke(messages).content.strip().lower()

        logger.info(f"[{self.session_id}] 👔 主管输出: '{decision}'")

        if "researcher" in decision:
            return "Researcher"
        if "calculator" in decision:
            return "Calculator"
        return "__end__"

    def _build_graph(self):
        # ── Researcher：只有搜索工具 ──────────────────
        researcher_llm = self.llm.bind_tools(self.search_tools)

        def researcher_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🔍 研究员开始工作（工具: {[t.name for t in self.search_tools]}）...")
            messages = [
                SystemMessage(content="你是权威的研究员，遇到事实问题必须使用搜索引擎。")
            ] + state["messages"]
            response = researcher_llm.invoke(messages)
            if response.content:
                response.content = f"【研究员汇报】: {response.content}"
            return {"messages": [response]}

        # ── Calculator：只有数学工具 ──────────────────
        calculator_llm = self.llm.bind_tools(self.math_tools)

        def calculator_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🧮 精算师开始工作（工具: {[t.name for t in self.math_tools]}）...")
            messages = [
                SystemMessage(content="你是精算师，必须使用计算工具解决数学问题。")
            ] + state["messages"]
            response = calculator_llm.invoke(messages)
            if response.content:
                response.content = f"【精算师汇报】: {response.content}"
            return {"messages": [response]}

        # ── 组装图 ────────────────────────────────────
        workflow = StateGraph(AgentState)
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("Calculator", calculator_node)

        # ★ 工具节点也按角色分开，各自只能用自己的工具
        workflow.add_node("ResearcherTools", ToolNode(self.search_tools))
        workflow.add_node("CalculatorTools", ToolNode(self.math_tools))

        from langgraph.prebuilt import tools_condition

        workflow.add_conditional_edges(START, self._route_by_supervisor)
        workflow.add_conditional_edges(
            "Researcher",
            tools_condition,
            {"tools": "ResearcherTools", "__end__": "__end__"}
        )
        workflow.add_conditional_edges(
            "Calculator",
            tools_condition,
            {"tools": "CalculatorTools", "__end__": "__end__"}
        )
        workflow.add_conditional_edges("ResearcherTools", self._route_by_supervisor)
        workflow.add_conditional_edges("CalculatorTools", self._route_by_supervisor)

        return workflow.compile()

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)
        initial_state = {"messages": self.memory.get_context(current_user_input=user_input)}

        final_answer = ""
        for event in self.app.stream(initial_state, {"recursion_limit": 15}):
            for node_name, state_update in event.items():
                if "messages" in state_update and state_update["messages"]:
                    latest_msg = state_update["messages"][-1]
                    if latest_msg.content:
                        final_answer += latest_msg.content + "\n"

        if not final_answer:
            final_answer = "任务已完成，但员工没有给出具体的文字报告。"

        self.memory.add_ai_message(final_answer.strip())
        return final_answer.strip()