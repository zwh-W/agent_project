# app/graph/state.py
import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    LangGraph 的全局状态字典

    Annotated[List[BaseMessage], operator.add] 的意思是：
    每次向 messages 更新数据时，不是“覆盖”，而是将新消息“追加(add)”到列表中。
    这正是对话历史需要的特性。
    """
    messages: Annotated[List[BaseMessage], operator.add]