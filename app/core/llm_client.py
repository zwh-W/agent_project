# app/core/llm_client.py
"""
LLM 客户端统一管理

设计原则：
1. 懒加载：第一次调用时才初始化，避免 import 时因 Key 未配置崩溃
2. 单例：整个进程复用同一个客户端
3. 同时提供 openai 原生客户端 和 langchain 封装两种方式

为什么要封装这一层？
换模型提供商（千问→OpenAI→智谱）只改 config.yaml，
业务代码一行不用动。
"""
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── 原生 OpenAI 客户端（手写 Function Calling 时用）──────
_openai_client = None


def get_openai_client():
    """
    懒加载原生 OpenAI 客户端
    适用于：手写 Function Calling、ReAct 等需要精细控制的场景

    使用方式：
        from app.core.llm_client import get_openai_client
        client = get_openai_client()
        response = client.chat.completions.create(...)
    """
    global _openai_client
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


# ── LangChain ChatOpenAI（LangGraph 工作流用）────────────
_langchain_llm = None


def get_langchain_llm(streaming: bool = False):
    """
    懒加载 LangChain LLM 封装
    适用于：LangGraph 节点、工具绑定（bind_tools）等场景

    使用方式：
        from app.core.llm_client import get_langchain_llm
        llm = get_langchain_llm()
        llm_with_tools = llm.bind_tools(tools)
    """
    global _langchain_llm
    if _langchain_llm is not None and not streaming:
        return _langchain_llm

    if not settings.llm.api_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置！")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        streaming=streaming,
    )

    if not streaming:
        _langchain_llm = llm
        logger.info(f"LangChain LLM 初始化完成，模型：{settings.llm.model}")

    return llm