# app/memory/manager.py
import threading
from typing import Dict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm  # 引入大模型用来做总结
from app.memory.long_term import es_memory_db  # ✨ 新增导入 ES 模块

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

    def get_context(self, current_user_input: str = "") -> List[BaseMessage]:
        """获取上下文，并【自动唤醒长期记忆】"""
        context = []
        full_system_prompt = self.base_system_prompt

        # 1. 注入短期摘要
        if self.summary:
            full_system_prompt += f"\n\n【近期背景摘要】:\n{self.summary}"

        # 2. ✨ 核心动作：用当前用户的提问，去 ES 检索前世今生的记忆！
        if current_user_input:
            long_term_facts = es_memory_db.recall_memory(self.session_id, current_user_input)
            if long_term_facts:
                full_system_prompt += f"\n\n【从长期记忆(ES)中唤醒的相关事实】:\n{long_term_facts}"

        context.append(SystemMessage(content=full_system_prompt))
        context.extend(self.messages)
        return context

    def _compress_if_needed(self):
        """核心：当消息数超标时，召唤 LLM 进行记忆压缩"""
        if len(self.messages) <= self.max_messages:
            return

        logger.info(f"[{self.session_id}] ⚠️ 记忆达到阈值，启动记忆压缩引擎...")
        keep_recent = 4
        messages_to_compress = self.messages[:-keep_recent]
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
            llm = get_langchain_llm()
            response = llm.invoke([HumanMessage(content=compression_prompt)])
            self.summary = response.content.strip()

            # ✨ 核心动作：将生成的摘要当做核心事实，永久固化到 ES 中！
            es_memory_db.save_memory(self.session_id, self.summary)

            self.messages = self.messages[-keep_recent:]
        except Exception as e:
            logger.error(f"[{self.session_id}] ❌ 记忆压缩失败: {e}")
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