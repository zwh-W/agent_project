# app/agents/multi_agent.py
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.graph.state import AgentState
from app.tools.search import web_search
from app.tools.calculator import calculator

logger = get_logger(__name__)


class MultiAgentSupervisor:
    """
    企业级 Multi-Agent 架构 (主管-员工模式)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)

        # 准备大脑
        self.llm = get_langchain_llm()

        # 编译流程图
        self.app = self._build_graph()

    def _build_graph(self):
        # ==========================================
        # 1. 定义员工：研究员 (Researcher)
        # ==========================================
        researcher_llm = self.llm.bind_tools([web_search])

        def researcher_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🔍 员工[研究员] 开始工作...")
            # 注入角色人设
            messages = [SystemMessage(content="你是权威的研究员，遇到事实问题必须使用搜索引擎。")] + state["messages"]
            response = researcher_llm.invoke(messages)
            # 在返回消息前加上特定的前缀，方便我们在日志里区分是谁说的
            if response.content:
                response.content = f"【研究员汇报】: {response.content}"
            return {"messages": [response]}

        # ==========================================
        # 2. 定义员工：精算师 (Calculator)
        # ==========================================
        calculator_llm = self.llm.bind_tools([calculator])

        def calculator_node(state: AgentState):
            logger.debug(f"[{self.session_id}] 🧮 员工[精算师] 开始工作...")
            messages = [SystemMessage(content="你是精算师，必须使用计算工具解决数学问题。")] + state["messages"]
            response = calculator_llm.invoke(messages)
            if response.content:
                response.content = f"【精算师汇报】: {response.content}"
            return {"messages": [response]}

        # ==========================================
        # 3. 定义老板：主管 (Supervisor)
        # ==========================================
        def supervisor_node(state: AgentState) -> dict:
            """
            主管不执行工具，只负责输出下一步要派给谁干活
            """
            logger.info(f"[{self.session_id}] 👔 老板[主管] 正在审视全局进展...")

            supervisor_prompt = (
                "你是一个团队主管，管理着两个员工：'Researcher'(负责联网查资料) 和 'Calculator'(负责数学计算)。\n"
                "请根据以下的对话历史，决定接下来需要谁来处理。\n"
                "如果任务已经彻底解决，或者不需要员工帮忙了，请回复 'FINISH'。\n"
                "你只能回复这三个词之一：'Researcher', 'Calculator', 'FINISH'，绝对不要输出任何标点符号或其他文字！"
            )

            messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
            decision = self.llm.invoke(messages).content.strip()

            # 容错处理
            if "Researcher" in decision:
                next_step = "Researcher"
            elif "Calculator" in decision:
                next_step = "Calculator"
            else:
                next_step = "FINISH"

            logger.info(f"[{self.session_id}] 👔 主管决定下一步派给: {next_step}")
            # 注意：主管节点不往 messages 里加消息，它只返回一个用于路由的信号（我们需要在状态字典中临时存一下）
            # 为了不破坏 AgentState 的结构，我们利用 LangGraph 的 node 返回特性，直接在路由函数里处理
            return {"next_worker": next_step}  # 这里只是为了日志打印方便，真正的路由在 edge 里

        # 定义一个特殊的路由函数（Edge Function）
        def route_by_supervisor(state: AgentState) -> Literal["Researcher", "Calculator", "__end__"]:
            # 重新召唤大模型做一次快速的路由判断
            supervisor_prompt = (
                "你管理 'Researcher' 和 'Calculator'。根据对话记录，输出下一步交给谁。完成则输出 'FINISH'。只能输出这三个词之一。"
            )
            messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
            decision = self.llm.invoke(messages).content.strip()
            if "Researcher" in decision: return "Researcher"
            if "Calculator" in decision: return "Calculator"
            return "__end__"

        # ==========================================
        # 4. 组装多智能体流程图
        # ==========================================
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("Calculator", calculator_node)
        # 我们把路由逻辑直接做在 conditional_edges 里，省略单纯的 supervisor 节点以防死循环

        # 强制起点是主管进行路由判断
        workflow.add_conditional_edges(START, route_by_supervisor)

        # 员工干完活之后，必须乖乖把结果交回给主管重新评估
        workflow.add_conditional_edges("Researcher", route_by_supervisor)
        workflow.add_conditional_edges("Calculator", route_by_supervisor)

        return workflow.compile()

    def chat(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)
        initial_state = {"messages": self.memory.get_context()}

        final_answer = ""
        # 限制 recursion_limit 防止大模型抽风死循环
        for event in self.app.stream(initial_state, {"recursion_limit": 15}):
            for node_name, state_update in event.items():
                if "messages" in state_update and state_update["messages"]:
                    latest_msg = state_update["messages"][-1]
                    # 我们过滤掉工具调用的中间日志，只拿自然语言回复
                    if latest_msg.content:
                        final_answer += latest_msg.content + "\n"

        if not final_answer:
            final_answer = "任务已完成，但员工没有给出具体的文字报告。"

        self.memory.add_ai_message(final_answer.strip())
        return final_answer.strip()