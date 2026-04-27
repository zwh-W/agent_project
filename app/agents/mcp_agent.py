# app/agents/mcp_agent.py
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.graph.state import AgentState
from app.core.config import settings

logger = get_logger(__name__)


class MCPAgent:
    """
    基于 MCP (Model Context Protocol) 协议的跨平台 Agent。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()

    async def chat(self, user_input: str) -> str:
        logger.info(f"[{self.session_id}] 用户输入 (MCP): {user_input}")
        self.memory.add_user_message(user_input)

        initial_state = {"messages": self.memory.get_context()}
        final_answer = ""

        logger.info(f"[{self.session_id}] 🔌 正在连接 MCP 服务器...")

        # 1. 初始化客户端（新版 API：去掉 async with）
        mcp_client = MultiServerMCPClient({
            "my-enterprise-tools": {
                "command": "python",
                "args": [settings.mcp.server_script],
                "transport": settings.mcp.transport,
            }
        })

        try:
            # 2. 动态拉取工具（新版 API：必须加 await）
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"[{self.session_id}] ✅ 成功拉取外部工具: {[t.name for t in mcp_tools]}")

            # 绑定外部工具给大模型
            llm_with_tools = self.llm.bind_tools(mcp_tools)

            # 3. 动态组装 LangGraph
            def agent_node(state: AgentState):
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            workflow = StateGraph(AgentState)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", ToolNode(mcp_tools))

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_edge("tools", "agent")

            app = workflow.compile()

            # 4. 异步运行流转
            async for event in app.astream(initial_state):
                for node_name, state_update in event.items():
                    latest_msg = state_update["messages"][-1]
                    if node_name == "agent" and not latest_msg.tool_calls:
                        final_answer = latest_msg.content

            self.memory.add_ai_message(final_answer)
            logger.info(f"[{self.session_id}] 🎉 MCP 最终回答: {final_answer[:50]}...")
            return final_answer

        finally:
            # 确保请求结束后，关闭子进程连接，防止产生僵尸进程
            if hasattr(mcp_client, "close"):
                mcp_client.close()