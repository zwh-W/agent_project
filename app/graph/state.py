# app/graph/state.py
import operator
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    LangGraph 的全局状态字典

    Annotated[List[BaseMessage], operator.add] 的意思是：
    每次向 messages 更新数据时，不是"覆盖"，而是将新消息"追加(add)"到列表中。
    这正是对话历史需要的特性。

    ★ [修改] 新增 next_worker 字段
    原因：multi_agent.py 的 supervisor_node 返回了 {"next_worker": ...}，
    但原 AgentState 没有声明该字段，LangGraph 写入时会直接抛 KeyError 崩溃。
    Optional[str] + default None 保证未赋值时不影响其他 Agent。
    """
    messages: Annotated[List[BaseMessage], operator.add]
    # ★ [新增字段] Multi-Agent Supervisor 的路由信号
    next_worker: Optional[str]
