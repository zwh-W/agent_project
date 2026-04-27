# app/core/llm_client.py
"""
LLM 客户端统一管理

设计原则：
1. 懒加载：第一次调用时才初始化，避免 import 时因 Key 未配置崩溃
2. 单例：整个进程复用同一个客户端
3. 同时提供 openai 原生客户端 和 langchain 封装两种方式

★ [修改] 所有单例操作加线程锁
原因：原代码的 if _xxx is not None: return _xxx / _xxx = ... 序列
在多线程并发下（FastAPI 默认多线程处理请求）存在竞态条件，
可能创建多个实例或读到半初始化对象（Python GIL 不保护对象构造中间状态）
"""
import threading
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── 原生 OpenAI 客户端 ────────────────────────────────────
_openai_client = None
_openai_lock = threading.Lock()  # ★ [新增] 线程锁


def get_openai_client():
    """
    懒加载原生 OpenAI 客户端（线程安全）
    适用于：手写 Function Calling、ReAct 等需要精细控制的场景
    """
    global _openai_client

    # 快速路径：已初始化则直接返回（不加锁，性能更好）
    if _openai_client is not None:
        return _openai_client

    # 慢速路径：加锁初始化
    with _openai_lock:
        # ★ double-check：拿到锁后再判断一次，防止多个线程都通过了快速路径
        if _openai_client is not None:
            return _openai_client

        if not settings.llm.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置！\n"
                "请在 .env 文件中设置：DASHSCOPE_API_KEY=sk-your-key"
            )

        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        logger.info(f"OpenAI 客户端初始化完成，模型：{settings.llm.model}")

    return _openai_client


# ── LangChain ChatOpenAI ──────────────────────────────────
_langchain_llm = None
_langchain_lock = threading.Lock()  # ★ [新增] 线程锁


def get_langchain_llm(streaming: bool = False):
    """
    懒加载 LangChain LLM 封装（线程安全）
    适用于：LangGraph 节点、工具绑定（bind_tools）等场景

    注意：streaming=True 时每次创建新实例（不缓存），
    因为流式客户端状态有差异，不应复用。
    """
    global _langchain_llm

    # streaming 模式每次新建，不走单例路径
    if streaming:
        return _create_langchain_llm(streaming=True)

    if _langchain_llm is not None:
        return _langchain_llm

    with _langchain_lock:
        if _langchain_llm is not None:
            return _langchain_llm
        _langchain_llm = _create_langchain_llm(streaming=False)
        logger.info(f"LangChain LLM 初始化完成，模型：{settings.llm.model}")

    return _langchain_llm


def _create_langchain_llm(streaming: bool = False):
    """内部工厂函数，创建 LangChain LLM 实例"""
    if not settings.llm.api_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置！")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        streaming=streaming,
    )
