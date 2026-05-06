# app/agents/function_calling_agent.py
"""
企业级 Function Calling Agent
这个 Agent 本质上是一个 “LLM → Tool Call → Tool Result → LLM” 的循环，
直到模型不再调用工具，或者达到最大迭代次数。
核心升级：ReAct 推理链结构化记录
    ReAct = Reasoning + Acting
    每一轮循环都记录：
        Thought（模型的推理）
        Action（决定调用什么工具，参数是什么）
        Observation（工具返回了什么）
    最终生成可读的 trace，方便调试、展示、评测。

为什么这很重要：
    1. 调试价值：出了问题能看到模型在哪一步想错了
    2. 展示价值：面试时能当场展示 Agent 的思考过程
    3. 评测价值：trace 是工具调用准确率评测的数据来源
    4. 产品价值：trace 可以展示给用户，增加信任感
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.messages import ToolMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.memory.manager import memory_manager
from app.tools.registry import tool_registry
logger = get_logger(__name__)


# ──────────────────────────────────────────────
# ReAct 推理链数据结构
# ──────────────────────────────────────────────

@dataclass
class ReActStep:
    """单个 ReAct 步骤（思考-行动-观察 三元组）"""
    step_index: int

    # Thought：模型的推理内容（如果有）
    thought: Optional[str] = None

    # Action：工具调用信息
    action_tool: Optional[str] = None
    action_input: Optional[dict] = None

    # Observation：工具返回结果
    observation: Optional[str] = None

    # 耗时（毫秒）
    duration_ms: float = 0.0


@dataclass
class ReActTrace:
    """完整的 ReAct 推理链"""
    session_id: str
    user_input: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    tool_call_count: int = 0

    def add_step(self, step: ReActStep):
        self.steps.append(step)
        if step.action_tool:
            self.tool_call_count += 1

    def to_readable(self) -> str:
        """
        生成人类可读的推理链文本
        这是面试展示的核心输出
        """
        lines = [
            f"╔{'═' * 60}",
            f"║ ReAct 推理链  |  会话: {self.session_id[:8]}...",
            f"║ 用户问题: {self.user_input[:50]}",
            f"╠{'═' * 60}",
        ]

        for step in self.steps:
            lines.append(f"║ 【第 {step.step_index} 轮】耗时 {step.duration_ms:.0f}ms")

            if step.thought:
                # 截断过长的思考内容
                thought_preview = step.thought[:100] + "..." if len(step.thought) > 100 else step.thought
                lines.append(f"║   🧠 Thought: {thought_preview}")

            if step.action_tool:
                args_str = json.dumps(step.action_input, ensure_ascii=False)
                if len(args_str) > 60:
                    args_str = args_str[:60] + "..."
                lines.append(f"║   ⚡ Action:  {step.action_tool}({args_str})")

            if step.observation:
                obs_preview = step.observation[:80] + "..." if len(step.observation) > 80 else step.observation
                lines.append(f"║   👁  Observe: {obs_preview}")

            lines.append(f"║   {'─' * 56}")

        lines.extend([
            f"║ ✅ Final Answer: {self.final_answer[:80]}",
            f"║ 📊 统计: {len(self.steps)} 轮推理 | {self.tool_call_count} 次工具调用 | 总耗时 {self.total_duration_ms:.0f}ms",
            f"╚{'═' * 60}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为字典，供 API 响应或日志使用"""
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "steps": [
                {
                    "step": s.step_index,
                    "thought": s.thought,
                    "action_tool": s.action_tool,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "total_duration_ms": self.total_duration_ms,
            "tool_call_count": self.tool_call_count,
        }


# ──────────────────────────────────────────────
# Agent 本体
# ──────────────────────────────────────────────

class FunctionCallingAgent:
    """
    企业级 Function Calling Agent
    具备能力：多轮动态记忆、工具自动分发、错误自我纠正、ReAct 推理链追踪
    """

    def __init__(
        self,
        session_id: str,
        tool_tags: Optional[set[str]] = None,
        exclude_tags: Optional[set[str]] = None,
    ):
        self.session_id = session_id
        self.memory = memory_manager.get_session(session_id)
        self.llm = get_langchain_llm()

        # 从工具注册中心获取工具
        # 如果 tool_tags 为 None，则默认获取全部已启用工具
        self.tools = tool_registry.get_tools(
            tags=tool_tags,
            exclude_tags=exclude_tags,
        )

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}

        # 保存最近一次的推理链，供外部读取
        self.last_trace: Optional[ReActTrace] = None

    def chat(self, user_input: str) -> str:
        """核心对话入口，返回最终答案字符串"""
        answer, _ = self.chat_with_trace(user_input)
        return answer

    def chat_with_trace(self, user_input: str) -> tuple[str, ReActTrace]:
        """
        带完整推理链的对话入口

        Returns:
            (最终答案, ReAct推理链)
        """
        logger.info(f"[{self.session_id}] 用户输入: {user_input}")

        trace = ReActTrace(
            session_id=self.session_id,
            user_input=user_input,
        )
        total_start = time.time()

        self.memory.add_user_message(user_input)
        max_iterations = settings.agent.max_iterations

        for attempt in range(max_iterations):
            step = ReActStep(step_index=attempt + 1)
            step_start = time.time()

            messages = self.memory.get_context(current_user_input=user_input)

            logger.debug(f"[{self.session_id}] 🧠 思考中 (第 {attempt + 1}/{max_iterations} 轮)...")
            ai_msg = self.llm_with_tools.invoke(messages)
            self.memory.add_raw_message(ai_msg)

            # 记录 Thought（模型的文本推理，如果有的话）
            if ai_msg.content:
                step.thought = ai_msg.content

            # 没有工具调用 → 得出最终答案
            if not ai_msg.tool_calls:
                step.duration_ms = (time.time() - step_start) * 1000
                trace.add_step(step)

                final_answer = ai_msg.content
                trace.final_answer = final_answer
                trace.total_duration_ms = (time.time() - total_start) * 1000

                # 打印可读推理链到日志
                logger.info(f"\n{trace.to_readable()}")
                self.last_trace = trace

                return final_answer, trace

            # 有工具调用 → 执行工具
            # 注意：模型可能一次想调多个工具（parallel tool calling）
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                # 记录 Action
                step.action_tool = tool_name
                step.action_input = tool_args

                logger.info(f"[{self.session_id}] ⚡ 调用工具: {tool_name}, 参数: {tool_args}")

                try:
                    tool_func = self.tool_map.get(tool_name)
                    if tool_func:
                        tool_result = str(tool_func.invoke(tool_args))
                    else:
                        tool_result = f"❌ 错误：找不到名为 {tool_name} 的工具"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 工具执行异常: {e}")
                    tool_result = (
                        f"❌ 工具执行抛出异常: {str(e)}。"
                        f"请检查参数是否合法，并反思后重试。"
                    )

                # 记录 Observation
                step.observation = tool_result
                logger.debug(f"[{self.session_id}] 👁 工具返回: {tool_result[:100]}...")

                tool_msg = ToolMessage(
                    tool_call_id=tool_id,
                    name=tool_name,
                    content=tool_result
                )
                self.memory.add_raw_message(tool_msg)

            step.duration_ms = (time.time() - step_start) * 1000
            trace.add_step(step)

        # 超出最大迭代次数
        error_msg = "⚠️ 思考过程过于复杂，超过了最大迭代次数，已强制停止。"
        trace.final_answer = error_msg
        trace.total_duration_ms = (time.time() - total_start) * 1000
        logger.error(f"[{self.session_id}] {error_msg}")
        logger.info(f"\n{trace.to_readable()}")

        self.memory.add_ai_message(error_msg)
        self.last_trace = trace
        return error_msg, trace