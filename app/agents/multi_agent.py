# app/agents/multi_agent.py
"""
企业级 Multi-Agent 架构 (主管-员工模式)

★ [修改1] supervisor_node 的返回值和路由解耦
原因：原代码 supervisor_node 返回 {"next_worker": next_step}，
但 AgentState 没有 next_worker 字段（已在 state.py 修复），
且原代码中 supervisor_node 节点根本没被 add_node 进去，
路由完全靠 route_by_supervisor 函数，但这个函数重复调用了一次 LLM，
意味着每次路由都会额外多消耗一次 LLM 调用（2倍成本，2倍延迟）。

★ [修改2] 将双重 LLM 路由调用合并为单次
★ [修改3] Supervisor Prompt 加 few-shot 示例，减少模型输出无关内容
"""
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.graph.state import AgentState
from app.tools.search import web_search
from app.tools.calculator import calculator

logger = get_logger(__name__)


# ★ [新增] Supervisor Prompt 抽离为模块级常量，方便版本管理和测试
# 加入 few-shot 示例，强制约束输出格式，降低大模型乱输出的概率
SUPERVISOR_SYSTEM_PROMPT = """你是一个团队主管，管理两个员工：
- Researcher：负责联网搜索事实、新闻、实时数据
- Calculator：负责数学计算、公式求值

根据对话历史，决定下一步派给谁。任务彻底完成或不需要员工帮忙时回复 FINISH。

【输出规则】只能输出以下三个词之一，不加任何标点或解释：
Researcher
Calculator
FINISH

【示例】
用户问: "今天比特币价格是多少" → Researcher
用户问: "123 乘以 456 等于多少" → Calculator
用户问: "先查一下苹果股价，再算出它比100美元高多少" → Researcher（先搜索）
研究员已汇报苹果股价为182美元 → Calculator（再计算）
计算结果已给出，任务完成 → FINISH"""


class MultiAgentSupervisor:
    """
    企业级 Multi-Agent 架构 (主管-员工模式)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()
        self.app = self._build_graph()

    def _route_by_supervisor(self, state: AgentState) -> Literal["Researcher", "Calculator", "__end__"]:
        """
        ★ [修改] 将路由逻辑从两个函数（supervisor_node + route_by_supervisor）合并为一个
        原因：原代码 supervisor_node 返回 next_worker 到状态字典，
        但路由 edge 函数 route_by_supervisor 又单独再调用一次 LLM 做决策，
        导致每次路由 = 2 次 LLM 调用，既浪费钱又增加延迟。
        现在合并为一次调用，将 LLM 的决策直接用于路由。
        """
        logger.info(f"[{self.session_id}] 👔 主管正在审视全局进展并做路由决策...")

        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        decision = self.llm.invoke(messages).content.strip()

        logger.info(f"[{self.session_id}] 👔 主管原始输出: '{decision}'")

        # 容错：大模型可能在词里加空格或换行
        decision_clean = decision.strip().lower()
        if "researcher" in decision_clean:
            next_step = "Researcher"
        elif "calculator" in decision_clean:
            next_step = "Calculator"
        else:
            next_step = "__end__"

        logger.info(f"[{self.session_id}] 👔 主管决定: {next_step}")
        return next_step

    def _build_graph(self):
        # ──────────────────────────────────────────────
        # 员工节点定义
        # ──────────────────────────────────────────────
        researcher_llm = self.llm.bind_tools([web_search])

        def researcher_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🔍 员工[研究员] 开始工作...")
            messages = [SystemMessage(content="你是权威的研究员，遇到事实问题必须使用搜索引擎，不能凭记忆回答。")] + state["messages"]
            response = researcher_llm.invoke(messages)
            if response.content:
                response.content = f"【研究员汇报】: {response.content}"
            return {"messages": [response]}

        calculator_llm = self.llm.bind_tools([calculator])

        def calculator_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🧮 员工[精算师] 开始工作...")
            messages = [SystemMessage(content="你是精算师，必须使用计算工具解决数学问题，不能心算。")] + state["messages"]
            response = calculator_llm.invoke(messages)
            if response.content:
                response.content = f"【精算师汇报】: {response.content}"
            return {"messages": [response]}

        # ──────────────────────────────────────────────
        # 组装流程图
        # ──────────────────────────────────────────────
        workflow = StateGraph(AgentState)

        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("Calculator", calculator_node)

        # ★ [修改] 路由函数改为 self._route_by_supervisor（实例方法），
        # 同一个 session 的上下文（self.session_id）可以被路由函数访问，方便日志追踪
        workflow.add_conditional_edges(START, self._route_by_supervisor)
        workflow.add_conditional_edges("Researcher", self._route_by_supervisor)
        workflow.add_conditional_edges("Calculator", self._route_by_supervisor)

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
