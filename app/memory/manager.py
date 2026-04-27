# app/memory/manager.py
import threading
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.core.config import settings
from app.core.logger import get_logger
from app.core.llm_client import get_langchain_llm
# ★ [修改] 改为导入懒加载函数，不再在 import 时触发初始化
# 原因：原来 from app.memory.long_term import es_memory_db 会立即执行
# ESLongTermMemory() 构造函数，加载 BGE 模型 + 连接 ES，服务启动即崩
from app.memory.long_term import get_es_memory

logger = get_logger(__name__)


class SessionMemory:
    """单个用户的智能记忆管理器（带动态压缩）"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.summary = ""
        self.messages: List[BaseMessage] = []
        self.max_messages = settings.memory.short_term_max_messages

        # ★ [修改] System Prompt 大幅增强，增加工具使用约束和输出格式说明
        # 原因：原来只有两句话，没有 few-shot、没有输出约束，
        # 大模型很容易不调工具直接瞎编，或者输出格式混乱
        self.base_system_prompt = (
            "你是一个专业、严谨的 AI 助手。\n\n"
            "【核心原则】\n"
            "1. 对于需要实时数据、最新信息、具体计算的问题，你【必须】使用对应工具，绝不能凭记忆编造。\n"
            "2. 如果工具返回了错误，你需要分析原因并重试，而不是直接告诉用户失败了。\n"
            "3. 最终回答必须简洁、准确、有条理。\n\n"
            "【工具使用规则】\n"
            "- 需要搜索事实/新闻/实时信息 → 使用 web_search\n"
            "- 需要数学计算 → 使用 calculator，不要心算\n\n"
            "【输出格式】\n"
            "- 直接给出答案，不要说'根据我的知识'或'我认为'\n"
            "- 如果使用了工具，可以简短说明信息来源\n"
            "- 不要重复用户的问题"
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
        """获取上下文，并自动唤醒长期记忆"""
        context = []
        full_system_prompt = self.base_system_prompt

        if self.summary:
            full_system_prompt += f"\n\n【近期背景摘要】:\n{self.summary}"

        # ★ [修改] 通过懒加载函数获取 ES 实例，失败时降级而非崩溃
        # 原因：原来直接用 es_memory_db.recall_memory()，
        # 如果 ES 未启动，这里会抛异常，整个对话接口返回 500
        if current_user_input:
            es = get_es_memory()
            if es is not None:
                try:
                    long_term_facts = es.recall_memory(self.session_id, current_user_input)
                    if long_term_facts:
                        full_system_prompt += f"\n\n【从长期记忆中唤醒的相关事实】:\n{long_term_facts}"
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 长期记忆召回失败（已降级跳过）: {e}")
            else:
                logger.debug(f"[{self.session_id}] ES 不可用，跳过长期记忆召回")

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

        # ★ [修改] 对 ToolMessage 做特殊处理，提取真实文本内容
        # 原因：ToolMessage.content 可能是 list（多个 content block），
        # 原来直接 m.content 会输出 "[TextContent(...)]" 这种 Python 对象字符串
        def extract_content(msg: BaseMessage) -> str:
            content = msg.content
            if isinstance(content, list):
                # 提取所有 text 类型的内容块
                parts = [block.get("text", "") if isinstance(block, dict) else str(block)
                         for block in content]
                return " ".join(filter(None, parts))
            return str(content) if content else ""

        old_dialogue_lines = []
        for m in messages_to_compress:
            role = m.type if not isinstance(m, ToolMessage) else "tool"
            content_text = extract_content(m)
            if content_text:  # 跳过空内容（如纯工具调用指令消息）
                old_dialogue_lines.append(f"{role}: {content_text}")

        old_dialogue = "\n".join(old_dialogue_lines)

        compression_prompt = f"""
你是一个记忆整理专家。请根据【旧的背景档案】和【最近的对话记录】，写出一份最新的、包含所有关键信息的【新背景档案】。
要求：极度精简，重点保留用户的偏好、核心诉求、关键事实、已确认的结论。

【旧的背景档案】: {self.summary if self.summary else "无"}

【最近的对话记录】: 
{old_dialogue}

【新背景档案】（直接输出，不要任何前缀）:
"""

        try:
            llm = get_langchain_llm()
            response = llm.invoke([HumanMessage(content=compression_prompt)])
            self.summary = response.content.strip()

            # ★ [修改] 通过懒加载函数获取 ES 实例，失败时降级
            es = get_es_memory()
            if es is not None:
                try:
                    es.save_memory(self.session_id, self.summary)
                except Exception as e:
                    logger.warning(f"[{self.session_id}] 长期记忆保存失败（已降级跳过）: {e}")

            self.messages = self.messages[-keep_recent:]
            logger.info(f"[{self.session_id}] ✅ 记忆压缩完成，摘要长度: {len(self.summary)} 字")
        except Exception as e:
            logger.error(f"[{self.session_id}] ❌ 记忆压缩失败: {e}")
            self.messages = self.messages[-keep_recent:]


class MemoryManager:
    """全局会话管理器"""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> SessionMemory:
        # ★ [修改] session_id 不应该是 None，在 schemas 层已经保证，这里加一道防御
        if not session_id:
            raise ValueError("session_id 不能为空，请检查 schemas 层是否正确生成")
        with self._lock:
            if session_id not in self._sessions:
                logger.info(f"🆕 创建新会话: {session_id}")
                self._sessions[session_id] = SessionMemory(session_id)
            return self._sessions[session_id]

    # ★ [新增] 获取当前活跃会话数，用于监控
    @property
    def active_session_count(self) -> int:
        return len(self._sessions)


memory_manager = MemoryManager()
