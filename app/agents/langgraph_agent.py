# app/agents/langgraph_agent.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.tools import ALL_TOOLS
from app.graph.state import AgentState

logger = get_logger(__name__)


class LangGraphAgent:
    """
    基于 LangGraph 构建的图结构 Agent
    核心优势：支持无限扩展的流程图、天然支持中断与状态恢复
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)

        # 1. 准备大脑和工具
        self.llm = get_langchain_llm()
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # 2. 编译并生成流程图
        self.app = self._build_graph()

    def _build_graph(self):
        """核心编排逻辑：画流程图"""

        # ==========================================
        # 节点 (Nodes)：流程图里的方块
        # ==========================================
        def call_model(state: AgentState):
            """大脑节点：负责思考和决定是否调用工具"""
            logger.debug(f"[{self.session_id}] 🧠 LangGraph: 进入大脑节点...")
            response = self.llm_with_tools.invoke(state["messages"])
            # 返回的值会自动追加到 state["messages"]
            return {"messages": [response]}

        # 工具节点：LangGraph 内置的 ToolNode，自动解析 tool_calls 并执行 ALL_TOOLS
        tool_node = ToolNode(ALL_TOOLS)

        # ==========================================
        # 边 (Edges)：流程图里的箭头
        # ==========================================
        workflow = StateGraph(AgentState)

        # 把方块贴到画板上
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # 画第一根箭头：起点指向 agent
        workflow.add_edge(START, "agent")

        # 画条件箭头：从 agent 出来后去哪？
        # tools_condition 会自动判断：如果模型想调工具，就走向 "tools"；否则走向 END
        workflow.add_conditional_edges("agent", tools_condition)

        # 画最后一根箭头：工具执行完，无条件回到 agent 节点继续思考
        workflow.add_edge("tools", "agent")

        # 编译成可执行的图
        return workflow.compile()

    def chat(self, user_input: str) -> str:
        """对话入口"""
        logger.info(f"[{self.session_id}] 用户输入 (LangGraph): {user_input}")

        # 1. 把用户输入写进记忆
        self.memory.add_user_message(user_input)

        # 2. 拿到当前的完整上下文，作为流程图的初始状态
        initial_state = {"messages": self.memory.get_context(current_user_input=user_input)}

        # 3. 启动流程图机器！(stream 可以让我们看到流转过程)
        final_answer = ""
        for event in self.app.stream(initial_state):
            for node_name, state_update in event.items():
                logger.debug(f"[{self.session_id}] 🔄 流程流转完毕: 节点 '{node_name}'")
                latest_msg = state_update["messages"][-1]

                # 如果是 agent 节点，且没有调工具的意图，说明得出了最终答案
                if node_name == "agent" and not latest_msg.tool_calls:
                    final_answer = latest_msg.content

                # 【可选】：如果想把工具的执行结果同步记录到持久化 memory 里，可以在这里拦截并处理

        # 4. 把最终答案写进记忆
        self.memory.add_ai_message(final_answer)
        logger.info(f"[{self.session_id}] 🎉 LangGraph 最终回答: {final_answer[:50]}...")

        return final_answer