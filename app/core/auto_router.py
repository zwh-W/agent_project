# app/core/auto_router.py
"""
智能 Agent 路由器

设计动机：
    原来用户需要手动指定 agent_type，这不现实——
    普通用户不知道什么是 LangGraph，什么是 MCP。
    真实产品里应该由系统自动判断用哪个 Agent。

路由策略（三层漏斗）：
    第一层：规则路由（零延迟）——明确的关键词直接命中
    第二层：LLM 路由（约 200ms）——复杂意图让模型判断
    第三层：兜底路由——任何异常降级到最稳定的 function_calling

为什么不全用 LLM 路由？
    因为"今天多少度"这种问题用规则一毫秒就判断了，
    让 LLM 再思考一轮是浪费钱和时间。
    规则 + LLM 的混合策略是工程实践中的标准做法。

面试价值：
    这个模块能说明你理解"Agent 框架"不只是调 API，
    而是一个需要精心设计的决策系统。
"""
import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import get_logger
from app.api.schemas import AgentType

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# 第一层：规则路由表
# 结构：(正则模式, 对应的 AgentType, 规则描述)
# 按优先级从高到低排列
# ──────────────────────────────────────────────
_RULE_ROUTING_TABLE = [
    # MCP 相关：明确要调企业内网工具
    (
        re.compile(r"(员工信息|工号|内网|服务器时间|企业系统)", re.IGNORECASE),
        AgentType.mcp,
        "命中企业内网工具关键词"
    ),
    # 复杂多步任务：明确需要多个专业角色协作
    (
        re.compile(r"(先.+再.+|搜索.+计算|查一下.+算|调研.+然后)", re.IGNORECASE),
        AgentType.multi_agent,
        "命中多步骤协作关键词"
    ),
    # 纯计算任务：数字 + 运算符
    (
        re.compile(r"[\d\s]+[+\-*/^%]+[\d\s]|计算|算一下|等于多少|多少钱"),
        AgentType.function_calling,
        "命中纯计算关键词"
    ),
]

# ──────────────────────────────────────────────
# 第二层：LLM 路由的 Prompt
# ──────────────────────────────────────────────
_LLM_ROUTER_PROMPT = """你是一个 Agent 路由专家，负责判断用户的问题应该交给哪种 Agent 处理。

【可选的 Agent 类型】
- function_calling：处理单步任务，如简单计算、直接查询、日常问答
- langgraph：处理需要多步推理的任务，如分析类问题、需要反复思考的问题  
- multi_agent：处理需要多个专业角色协作的任务，如"先搜索再计算"、"调研后总结"
- mcp：处理需要访问企业内网工具的任务，如查询员工信息、获取内部系统数据

【判断原则】
1. 优先选最简单够用的 Agent（function_calling 能搞定就不用 langgraph）
2. 只有明确需要多角色分工才选 multi_agent
3. 有企业内部数据需求才选 mcp

【输出格式】
只输出 Agent 类型名称，不加任何解释。必须是以下之一：
function_calling / langgraph / multi_agent / mcp

【示例】
用户: "帮我计算 100 乘以 3.14" → function_calling
用户: "分析一下电商行业的竞争格局，从多个维度给我建议" → langgraph  
用户: "先查一下今天黄金价格，再帮我算如果买100克要多少钱" → multi_agent
用户: "查一下工号 E1001 的员工信息" → mcp
"""


class AgentAutoRouter:
    """
    三层漏斗式智能路由器

    使用方式：
        router = AgentAutoRouter()
        agent_type = router.route("帮我计算一下 2 的 10 次方")
    """

    def __init__(self):
        # 懒加载 LLM，只有规则层没命中时才初始化
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from app.core.llm_client import get_langchain_llm
            self._llm = get_langchain_llm()
        return self._llm

    def route(self, user_input: str, explicit_type: Optional[AgentType] = None) -> AgentType:
        """
        路由入口

        Args:
            user_input: 用户输入
            explicit_type: 用户明确指定的类型（最高优先级）

        Returns:
            确定要使用的 AgentType
        """
        # 优先级 0：用户明确指定
        if explicit_type and explicit_type != AgentType.function_calling:
            # 注意：默认值是 function_calling，如果用户没改默认值就当没指定
            logger.info(f"路由决策：用户明确指定 → {explicit_type.value}")
            return explicit_type

        # 优先级 1：规则路由（零延迟）
        rule_result = self._rule_route(user_input)
        if rule_result:
            return rule_result

        # 优先级 2：LLM 路由（约 200ms）
        llm_result = self._llm_route(user_input)
        if llm_result:
            return llm_result

        # 优先级 3：兜底
        logger.info("路由决策：兜底降级 → function_calling")
        return AgentType.function_calling

    def _rule_route(self, user_input: str) -> Optional[AgentType]:
        """第一层：规则路由"""
        for pattern, agent_type, description in _RULE_ROUTING_TABLE:
            if pattern.search(user_input):
                logger.info(f"路由决策：规则命中（{description}）→ {agent_type.value}")
                return agent_type
        return None

    def _llm_route(self, user_input: str) -> Optional[AgentType]:
        """第二层：LLM 路由"""
        try:
            llm = self._get_llm()
            messages = [
                SystemMessage(content=_LLM_ROUTER_PROMPT),
                HumanMessage(content=f"用户问题：{user_input}")
            ]
            response = llm.invoke(messages).content.strip().lower()

            # 解析 LLM 输出
            type_map = {
                "function_calling": AgentType.function_calling,
                "langgraph": AgentType.langgraph,
                "multi_agent": AgentType.multi_agent,
                "mcp": AgentType.mcp,
            }

            for key, agent_type in type_map.items():
                if key in response:
                    logger.info(f"路由决策：LLM 判断 → {agent_type.value}（原始输出: '{response}'）")
                    return agent_type

        except Exception as e:
            logger.warning(f"LLM 路由失败，将使用兜底路由: {e}")

        return None


# 全局单例
agent_auto_router = AgentAutoRouter()