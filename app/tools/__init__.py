# app/tools/__init__.py
"""
工具初始化模块

★ 改动：通过 ToolRegistry 注册工具，不再是硬编码的 ALL_TOOLS 列表

每个工具注册时声明自己的 tags，调用方可以按 tags 筛选：
    - "math"   : 数学计算类工具
    - "search" : 搜索/信息检索类工具
    - "system" : 系统/内部工具

各 Agent 按需获取工具子集，而不是全部拿走：
    - FunctionCallingAgent: get_tools()            → 全部工具
    - MultiAgent Researcher: get_tools({"search"}) → 只有搜索工具
    - MultiAgent Calculator: get_tools({"math"})   → 只有计算工具
"""
from app.tools.registry import tool_registry
from app.tools.calculator import calculator
from app.tools.search import web_search

# 注册计算工具
tool_registry.register(
    calculator,
    tags={"math", "calculation"},
    description="安全的数学表达式求值",
)

# 注册搜索工具
tool_registry.register(
    web_search,
    tags={"search", "information_retrieval"},
    description="DuckDuckGo 实时网络搜索",
)

# ── 对外暴露的快捷方式 ────────────────────────────────────
# 保持向后兼容，现有 Agent 代码中 from app.tools import ALL_TOOLS 不需要改
ALL_TOOLS = tool_registry.get_tools()