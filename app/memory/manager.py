# app/memory/manager.py
import threading
from typing import Dict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm  # 引入大模型用来做总结

logger = get_logger(__name__)


class SessionMemory:
    """单个用户的智能记忆管理器（带动态压缩）"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.summary = ""  # 存放记忆摘要
        self.messages: List[BaseMessage] = []
        # 触发压缩的阈值（通常设置大一点，比如 20）
        self.max_messages = settings.memory.short_term_max_messages

        self.base_system_prompt = (
            "你是一个极其聪明的 AI 助手。"
            "遇到不知道的事情，必须使用工具查询，绝不能瞎编。"
        )

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._compress_if_needed()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        self._compress_if_needed()

    def add_raw_message(self, message: BaseMessage):
        self.messages.append(message)
        self._compress_if_needed()

    def get_context(self) -> List[BaseMessage]:
        """获取当前供大模型读取的完整上下文（动态注入摘要）"""
        context = []

        # 动态组装系统人设：基础设定 + 历史摘要
        full_system_prompt = self.base_system_prompt
        if self.summary:
            full_system_prompt += f"\n\n【用户历史背景档案】:\n{self.summary}"

        context.append(SystemMessage(content=full_system_prompt))

        # 加上最近的活跃对话
        context.extend(self.messages)
        return context

    def _compress_if_needed(self):
        """核心：当消息数超标时，召唤 LLM 进行记忆压缩"""
        if len(self.messages) <= self.max_messages:
            return

        logger.info(f"[{self.session_id}] ⚠️ 记忆达到阈值({self.max_messages})，启动记忆压缩引擎...")

        # 抽出需要被压缩的老旧消息（留下最新的 4 条作为上下文连贯缓冲）
        keep_recent = 4
        messages_to_compress = self.messages[:-keep_recent]

        # 把老消息转成文本
        old_dialogue = "\n".join([f"{m.type}: {m.content}" for m in messages_to_compress])

        # 让大模型干活的 Prompt
        compression_prompt = f"""
        你是一个记忆整理专家。请根据【旧的背景档案】和【最近的对话记录】，写出一份最新的、包含所有关键信息的【新背景档案】。
        要求：极度精简，重点保留用户的偏好、核心诉求、关键事实等。

        【旧的背景档案】: {self.summary if self.summary else "无"}

        【最近的对话记录】: 
        {old_dialogue}
        """

        try:
            # 召唤 LLM 生成新摘要
            llm = get_langchain_llm()
            # 注意：这里的思考是在后台默默进行的，不影响主对话历史
            response = llm.invoke([HumanMessage(content=compression_prompt)])

            self.summary = response.content.strip()
            logger.info(f"[{self.session_id}] ✅ 压缩完成。新背景档案: {self.summary[:50]}...")

            # 截断原始消息列表，只保留最近的那几条
            self.messages = self.messages[-keep_recent:]

        except Exception as e:
            logger.error(f"[{self.session_id}] ❌ 记忆压缩失败: {e}")
            # 如果压缩失败，为了防止彻底爆掉，退化为暴力截断
            self.messages = self.messages[-keep_recent:]


class MemoryManager:
    """全局会话管理器"""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> SessionMemory:
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"🆕 创建新会话: {session_id}")
                self._sessions[session_id] = SessionMemory(session_id)
            return self._sessions[session_id]


memory_manager = MemoryManager()