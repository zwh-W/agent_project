# app/agents/mcp_agent.py
"""
基于 MCP (Model Context Protocol) 协议的跨平台 Agent

★ [修改] 修复 MCP 客户端关闭方式
原因：原来 finally 块调用 mcp_client.close()（同步），
但 MultiServerMCPClient 的底层是异步 I/O，
在 async 函数里用同步 close() 等于没关，子进程连接会泄漏成僵尸进程。
正确做法是 await mcp_client.aclose()，或使用 async with 上下文管理器。
"""
import asyncio
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
    基于 MCP (Model Context Protocol) 协议的跨平台 Agent
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()

    async def chat(self, user_input: str) -> str:
        logger.info(f"[{self.session_id}] 用户输入 (MCP): {user_input}")
        self.memory.add_user_message(user_input)

        initial_state = {"messages": self.memory.get_context(current_user_input=user_input)}
        final_answer = ""

        logger.info(f"[{self.session_id}] 🔌 正在连接 MCP 服务器...")

        # ★ [修改] 改用 async with 上下文管理器，确保连接一定被正确关闭
        # 原因：原来用 try/finally + mcp_client.close()（同步调用）
        # MultiServerMCPClient 的关闭是异步的，同步调用会直接返回而不等待关闭完成
        # 导致后台子进程变成僵尸进程，长时间运行后系统资源耗尽
        # async with 会自动调用 __aenter__ 和 __aexit__，保证异步安全关闭
        async with MultiServerMCPClient({
            "my-enterprise-tools": {
                "command": "python",
                "args": [settings.mcp.server_script],
                "transport": settings.mcp.transport,
            }
        }) as mcp_client:
            # ★ [修改] get_tools() 加 await（新版 API 要求）
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"[{self.session_id}] ✅ 成功拉取外部工具: {[t.name for t in mcp_tools]}")

            llm_with_tools = self.llm.bind_tools(mcp_tools)

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

            async for event in app.astream(initial_state):
                for node_name, state_update in event.items():
                    latest_msg = state_update["messages"][-1]
                    if node_name == "agent" and not latest_msg.tool_calls:
                        final_answer = latest_msg.content

        # async with 结束后连接已自动关闭
        self.memory.add_ai_message(final_answer)
        logger.info(f"[{self.session_id}] 🎉 MCP 最终回答: {final_answer[:50]}...")
        return final_answer
