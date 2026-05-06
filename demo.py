# debug_tool_registry.py

from langchain_core.tools import tool
from app.tools.registry import tool_registry


@tool
def debug_add(a: int, b: int) -> int:
    """用于 debug 的加法工具"""
    return a + b


@tool
def debug_fail() -> str:
    """用于 debug 的失败工具"""
    raise RuntimeError("模拟工具执行失败")


def main():
    print("1. 注册工具")
    tool_registry.register(
        debug_add,
        tags={"math", "debug"},
        description="debug 加法工具"
    )

    tool_registry.register(
        debug_fail,
        tags={"debug", "error"},
        description="debug 失败工具"
    )

    print("\n2. 获取全部工具")
    all_tools = tool_registry.get_tools()
    print([t.name for t in all_tools])

    print("\n3. 按标签获取 math 工具")
    math_tools = tool_registry.get_tools(tags={"math"})
    print([t.name for t in math_tools])

    print("\n4. 调用 debug_add 工具")
    result = debug_add.invoke({"a": 3, "b": 5})
    print("debug_add result:", result)

    print("\n5. 调用 debug_fail 工具，观察失败统计")
    try:
        debug_fail.invoke({})
    except Exception as e:
        print("debug_fail error:", e)

    print("\n6. 查看工具统计")
    stats = tool_registry.get_stats()
    for item in stats:
        print(item)

    print("\n7. 禁用 debug_add 工具")
    tool_registry.disable_tool("debug_add")

    print("\n8. 再次获取全部工具")
    all_tools_after_disable = tool_registry.get_tools()
    print([t.name for t in all_tools_after_disable])

    print("\n9. 启用 debug_add 工具")
    tool_registry.enable_tool("debug_add")

    print("\n10. 再次获取全部工具")
    all_tools_after_enable = tool_registry.get_tools()
    print([t.name for t in all_tools_after_enable])


if __name__ == "__main__":
    main()