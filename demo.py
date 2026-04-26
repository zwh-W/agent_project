from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# 1. 工具
@tool
def get_weather(city: str):
    """查天气"""
    return f"{city} 晴天 26度"

# 2. 大模型
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_weather])

# --------------------------
# 3. Agent 思考（ReAct）
# --------------------------
def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --------------------------
# 4. 反思（框架自动处理）
# --------------------------
def reflection(state: MessagesState):
    check = llm.invoke("检查最后回答是否正确，只回复：通过 或 重写").content
    return {"reflection_result": check}  # 直接返回结果

# --------------------------
# 5. 构建流程图
# --------------------------
builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_node("reflection", reflection)

# --------------------------
# ✅ 重点：框架自带路由！
# 完全不用写 if else！
# --------------------------
builder.add_edge("agent", "tools")
builder.add_edge("tools", "reflection")

# 框架自动判断：反思通过→结束，不通过→回到Agent
builder.add_conditional_edges(
    "reflection",
    lambda s: "agent" if s["reflection_result"] == "重写" else END
)

builder.set_entry_point("agent")
graph = builder.compile()

