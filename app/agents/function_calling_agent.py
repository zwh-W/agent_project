# app/agents/function_calling_agent.py
import json
from langchain_core.messages import ToolMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.tools import ALL_TOOLS

logger = get_logger(__name__)


class FunctionCallingAgent:
    """
    企业级 Function Calling Agent
    具备能力：多轮动态记忆、工具自动分发、错误自我纠正 (Reflection)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        # 获取该用户的专属记忆对象
        self.memory = memory_manager.get_session(session_id)

        # 获取大模型，并将所有工具“绑定”给大脑
        self.llm = get_langchain_llm()
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # 将工具列表转换为 {名字: 函数} 的字典，方便后续根据名字调用
        self.tool_map = {tool.name: tool for tool in ALL_TOOLS}

    def chat(self, user_input: str) -> str:
        """核心对话入口"""
        logger.info(f"[{self.session_id}] 用户输入: {user_input}")

        # 1. 记录用户输入到记忆中枢
        self.memory.add_user_message(user_input)

        # 允许 Agent 最多思考/调工具的轮数（防止陷入死循环）
        max_iterations = settings.agent.max_iterations

        for attempt in range(max_iterations):
            # 获取当前上下文（包含自动注入的 System Prompt 和 Summary 摘要）
            messages = self.memory.get_context()

            # 2. 召唤大模型进行思考
            logger.debug(f"[{self.session_id}] 🧠 正在思考 (第 {attempt + 1}/{max_iterations} 轮)...")
            ai_msg = self.llm_with_tools.invoke(messages)

            # 必须把大模型的原始回复（不管是文字还是想调工具的指令）存入记忆
            self.memory.add_raw_message(ai_msg)

            # 3. 判断是否需要调用工具
            if not ai_msg.tool_calls:
                # 模型没有输出工具调用，直接给出了最终答案，任务完成！
                logger.info(f"[{self.session_id}] 🎉 给出最终回答: {ai_msg.content[:50]}...")
                return ai_msg.content

            # 4. 执行工具调用（大模型可能会一次性想调用多个工具，所以要循环）
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                logger.info(f"[{self.session_id}] 🛠️ 决定调用工具: {tool_name}, 参数: {tool_args}")

                # 执行本地工具代码
                try:
                    tool_func = self.tool_map.get(tool_name)
                    if tool_func:
                        # invoke 是 LangChain Tool 的标准执行方法
                        tool_result = str(tool_func.invoke(tool_args))
                    else:
                        tool_result = f"❌ 错误：找不到名为 {tool_name} 的工具"

                except Exception as e:
                    # 【核心亮点：错误自我纠正】拦截所有代码层面的崩溃，转化为自然语言的观测结果！
                    logger.warning(f"[{self.session_id}] 工具执行异常: {e}")
                    tool_result = f"❌ 工具执行抛出异常: {str(e)}。请检查你的参数是否合法，并反思后重试。"

                logger.debug(f"[{self.session_id}] 👁️ 观察到工具结果: {tool_result[:100]}...")

                # 5. 将工具的执行结果封装成规范的 ToolMessage，存入记忆
                tool_msg = ToolMessage(
                    tool_call_id=tool_id,
                    name=tool_name,
                    content=tool_result
                )
                self.memory.add_raw_message(tool_msg)

            # 本轮执行完毕，循环继续。大模型会在下一轮看到工具的返回结果，继续推理！

        # 如果超出了最大循环次数
        error_msg = "⚠️ 思考过程过于复杂，超过了最大迭代次数，已强制停止。"
        logger.error(f"[{self.session_id}] {error_msg}")
        self.memory.add_ai_message(error_msg)
        return error_msg
