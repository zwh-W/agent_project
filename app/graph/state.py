# app/graph/state.py
import operator
from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(dict):
    """
    LangGraph 全局状态字典

    messages：使用 operator.add 作为 Reducer，
              每次节点返回时追加而非覆盖，保留完整对话历史。

    next_worker：MultiAgentSupervisor 路由信号，
                 声明此字段避免 LangGraph 写入时 KeyError。
    """
    # 用 TypedDict 风格注解（LangGraph 通过 __annotations__ 读取 Reducer）
    __annotations__ = {
        "messages":    Annotated[List[BaseMessage], operator.add],
        "next_worker": Optional[str],
    }