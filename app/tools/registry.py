# app/tools/registry.py
"""
工具注册中心

设计动机：
    现在的 app/tools/__init__.py 是这样的：
        ALL_TOOLS = [calculator, web_search]
    这是最原始的硬编码列表，存在三个问题：

    问题 1：添加新工具需要改两个文件（工具文件 + __init__.py）
            容易漏改，而且没有任何检查机制

    问题 2：所有 Agent 用同一套工具，无法针对不同 Agent 配置不同工具集
            比如：Calculator Agent 不应该有 web_search 工具，
            让它拥有这个工具只会增加大模型的"选择困难"

    问题 3：工具没有元数据（谁用、用了多少次、失败了多少次），
            无法做任何监控和优化

解决方案：工具注册中心
    1. 装饰器注册：@register_tool(tags=["math"]) 一行即可注册
    2. 按标签筛选：get_tools(tags=["search"]) 获取特定工具集
    3. 调用统计：每次工具被调用，自动记录成功/失败次数和耗时
    4. 健康检查：通过 health_check() 测试工具是否可用

面试价值：
    "你的工具是怎么管理的？如果要给不同 Agent 配不同工具集怎么做？"
    能说出"我有一个工具注册中心，支持按标签筛选"，
    说明你在认真思考系统设计，而不只是把代码凑在一起。
"""
import time
import threading
import functools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from langchain_core.tools import BaseTool
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolStats:
    """工具调用统计数据"""
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_duration_ms": f"{self.avg_duration_ms:.0f}ms",
        }


@dataclass
class ToolRegistration:
    """工具的注册信息"""
    tool: BaseTool
    tags: Set[str]          # 用于分类和筛选的标签
    description: str        # 额外的注册说明（比 tool.description 更简短）
    enabled: bool = True    # 可以动态禁用某个工具（不需要重启服务）
    stats: ToolStats = field(default_factory=lambda: ToolStats(""))

    def __post_init__(self):
        self.stats.tool_name = self.tool.name


class ToolRegistry:
    """
    工具注册中心（单例）

    使用方式：
        # 注册工具
        from app.tools.registry import tool_registry
        tool_registry.register(calculator, tags={"math", "calculation"})

        # 获取全部工具
        all_tools = tool_registry.get_tools()

        # 按标签获取工具子集
        math_tools = tool_registry.get_tools(tags={"math"})

        # 获取统计
        stats = tool_registry.get_stats()
    """

    def __init__(self):
        self._registrations: Dict[str, ToolRegistration] = {}
        self._lock = threading.Lock()

    def register(
        self,
        tool: BaseTool,
        tags: Optional[Set[str]] = None,
        description: str = "",
    ) -> BaseTool:
        """
        注册工具，并自动为其包装调用统计

        Args:
            tool:        LangChain BaseTool 实例
            tags:        标签集合，用于分类筛选
            description: 注册说明

        Returns:
            包装了统计功能的工具（行为不变，仅增加监控）
        """
        tags = tags or set()
        wrapped_tool = self._wrap_with_stats(tool)

        with self._lock:
            reg = ToolRegistration(
                tool=wrapped_tool,
                tags=tags,
                description=description or tool.description[:50],
            )
            self._registrations[tool.name] = reg
            logger.debug(f"工具已注册: {tool.name} | 标签: {tags}")

        return wrapped_tool

    def get_tools(
        self,
        tags: Optional[Set[str]] = None,
        exclude_tags: Optional[Set[str]] = None,
    ) -> List[BaseTool]:
        """
        获取工具列表

        Args:
            tags:         只返回包含这些标签之一的工具（None 表示返回全部）
            exclude_tags: 排除包含这些标签的工具

        Returns:
            符合条件的 BaseTool 列表
        """
        with self._lock:
            result = []
            for name, reg in self._registrations.items():
                if not reg.enabled:
                    continue
                if tags and not tags.intersection(reg.tags):
                    continue
                if exclude_tags and exclude_tags.intersection(reg.tags):
                    continue
                result.append(reg.tool)
        return result

    def disable_tool(self, tool_name: str):
        """动态禁用某个工具（不需要重启服务）"""
        with self._lock:
            if tool_name in self._registrations:
                self._registrations[tool_name].enabled = False
                logger.info(f"工具已禁用: {tool_name}")

    def enable_tool(self, tool_name: str):
        """重新启用某个工具"""
        with self._lock:
            if tool_name in self._registrations:
                self._registrations[tool_name].enabled = True
                logger.info(f"工具已启用: {tool_name}")

    def get_stats(self) -> List[dict]:
        """获取所有工具的调用统计"""
        with self._lock:
            return [reg.stats.to_dict() for reg in self._registrations.values()]

    def _wrap_with_stats(self, tool: BaseTool) -> BaseTool:
        """
        为工具包装调用统计。

        注意：
        LangChain 的 StructuredTool 底层是 Pydantic 对象，
        不能直接 tool.invoke = tracked_invoke，
        否则会触发 Pydantic 的字段校验错误。

        这里使用 object.__setattr__ 绕过 Pydantic 的 __setattr__，
        在实例层面替换 invoke 方法。
        """
        registry_ref = self
        original_invoke = tool.invoke

        @functools.wraps(original_invoke)
        def tracked_invoke(input=None, *args, **kwargs):
            start_time = time.time()

            try:
                result = original_invoke(input, *args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                with registry_ref._lock:
                    if tool.name in registry_ref._registrations:
                        stats = registry_ref._registrations[tool.name].stats
                        stats.call_count += 1
                        stats.success_count += 1
                        stats.total_duration_ms += duration_ms

                logger.debug(
                    f"工具调用成功: {tool.name} | 耗时 {duration_ms:.0f}ms"
                )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                with registry_ref._lock:
                    if tool.name in registry_ref._registrations:
                        stats = registry_ref._registrations[tool.name].stats
                        stats.call_count += 1
                        stats.error_count += 1
                        stats.total_duration_ms += duration_ms

                logger.warning(
                    f"工具调用失败: {tool.name} | 耗时 {duration_ms:.0f}ms | 错误: {e}"
                )
                raise

        # 关键修复点：
        # StructuredTool 是 Pydantic 对象，不能直接 tool.invoke = ...
        # 用 object.__setattr__ 绕过 Pydantic 字段限制。
        object.__setattr__(tool, "invoke", tracked_invoke)

        return tool


# 全局单例
tool_registry = ToolRegistry()