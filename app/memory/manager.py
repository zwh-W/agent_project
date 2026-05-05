# app/memory/manager.py
"""
会话记忆管理器

★ 本版改动：接入 PromptManager，System Prompt 从文件加载而非硬编码
   这样改 Prompt 不需要改代码，支持热加载和版本回滚
"""
import threading
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
from app.core.prompt_manager import prompt_manager   # ★ 新增导入
from app.memory.long_term import get_es_memory

logger = get_logger(__name__)


class SessionMemory:
    """单个用户的智能记忆管理器（带动态压缩）"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.summary = ""
        self.messages: List[BaseMessage] = []
        self.max_messages = settings.memory.short_term_max_messages

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
        """
        获取上下文，自动唤醒长期记忆，并从 PromptManager 加载 System Prompt

        ★ 改动说明：
        原来 System Prompt 是硬编码字符串，每次改 Prompt 需要改代码重启服务。
        现在从 PromptManager 加载，支持：
          1. 文件热加载（改完 prompts/system/v2.txt 调 reload() 即生效）
          2. 版本切换（可以随时切回 v1，做 A/B 对比）
          3. 变量插值（summary 和 long_term 内容动态注入）
        """
        # 1. 收集动态变量
        summary_section = ""
        long_term_section = ""

        if self.summary:
            summary_section = f"\n\n【近期背景摘要】:\n{self.summary}"

        if current_user_input:
            es = get_es_memory()
            if es is not None:
                try:
                    long_term_facts = es.recall_memory(self.session_id, current_user_input)
                    if long_term_facts:
                        long_term_section = f"\n\n【从长期记忆中唤醒的相关事实】:\n{long_term_facts}"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 长期记忆召回失败（已降级跳过）: {e}")

        # 2. ★ 从 PromptManager 加载 System Prompt，并注入动态变量
        try:
            full_system_prompt = prompt_manager.get(
                "system",
                summary_section=summary_section,
                long_term_section=long_term_section,
            )
        except Exception as e:
            # PromptManager 失败时降级到内联字符串，保证服务不中断
            logger.warning(f"[{self.session_id}] PromptManager 加载失败，使用内联兜底 Prompt: {e}")
            full_system_prompt = (
                "你是一个专业的 AI 助手。遇到不知道的事情必须使用工具查询，绝不能瞎编。"
                + summary_section
                + long_term_section
            )

        return [SystemMessage(content=full_system_prompt)] + self.messages

    def _compress_if_needed(self):
        """当消息数超标时，召唤 LLM 进行记忆压缩"""
        if len(self.messages) <= self.max_messages:
            return

        logger.info(f"[{self.session_id}] ⚠️ 记忆达到阈值，启动记忆压缩...")
        keep_recent = 4
        messages_to_compress = self.messages[:-keep_recent]

        def extract_content(msg: BaseMessage) -> str:
            content = msg.content
            if isinstance(content, list):
                parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                return " ".join(filter(None, parts))
            return str(content) if content else ""

        old_dialogue_lines = []
        for m in messages_to_compress:
            role = m.type if not isinstance(m, ToolMessage) else "tool"
            content_text = extract_content(m)
            if content_text:
                old_dialogue_lines.append(f"{role}: {content_text}")

        old_dialogue = "\n".join(old_dialogue_lines)
        compression_prompt = (
            f"你是记忆整理专家。根据【旧档案】和【对话记录】，输出一份精简的【新档案】。"
            f"重点保留：用户偏好、核心诉求、关键事实、已确认的结论。\n\n"
            f"【旧档案】: {self.summary or '无'}\n\n"
            f"【对话记录】:\n{old_dialogue}\n\n"
            f"【新档案】（直接输出，不要任何前缀）:"
        )

        try:
            llm = get_langchain_llm()
            response = llm.invoke([HumanMessage(content=compression_prompt)])
            self.summary = response.content.strip()

            es = get_es_memory()
            if es is not None:
                try:
                    es.save_memory(self.session_id, self.summary)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 长期记忆保存失败: {e}")

            self.messages = self.messages[-keep_recent:]
            logger.info(f"[{self.session_id}] ✅ 记忆压缩完成，摘要 {len(self.summary)} 字")
        except Exception as e:
            logger.error(f"[{self.session_id}] ❌ 记忆压缩失败: {e}")
            self.messages = self.messages[-keep_recent:]


class MemoryManager:
    """全局会话管理器"""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> SessionMemory:
        if not session_id:
            raise ValueError("session_id 不能为空")
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"🆕 创建新会话: {session_id}")
                self._sessions[session_id] = SessionMemory(session_id)
            return self._sessions[session_id]

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)


memory_manager = MemoryManager()