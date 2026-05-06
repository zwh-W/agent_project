# app/core/auto_router.py
"""
智能 Agent 路由器（三层漏斗）

变更记录：
  v2.1  修正 auto 判断逻辑：
          原来用"是否等于 function_calling"暗示用户没指定，语义混淆。
          现在：explicit_type 为 None 或 AgentType.auto 才走自动路由，
          其他值一律尊重用户显式指定。
        新增 HR / 制度 / 工单 关键词规则路由。
        更新 LLM Router Prompt，让模型知道 RAG 工具和工单工具的存在。

路由策略（三层漏斗）：
  第一层：规则路由（零延迟）  ── 明确关键词直接命中
  第二层：LLM 路由（≈200ms）  ── 复杂意图让模型判断
  第三层：兜底路由            ── 任何异常降级到 function_calling
"""
import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import get_logger
from app.api.schemas import AgentType

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 第一层：规则路由表
# 结构：(正则模式, 目标 AgentType, 规则描述)
# ★ 规则按优先级从高到低排列，第一个命中的生效
# ─────────────────────────────────────────────────────────
_RULE_ROUTING_TABLE = [

    # ── MCP：企业内网工具（最高优先级，明确的系统操作）──────
    (
        re.compile(r"(员工信息|工号|内网|服务器时间|企业系统)", re.IGNORECASE),
        AgentType.mcp,
        "MCP：企业内网工具关键词"
    ),

    # ── Multi-Agent：明确的多步骤协作任务 ────────────────────
    (
        re.compile(r"(先.{1,10}再.{1,10}|搜索.{1,10}计算|查一下.{1,10}算|调研.{1,10}然后)", re.IGNORECASE),
        AgentType.multi_agent,
        "Multi-Agent：多步骤协作关键词"
    ),

    # ★ ── HR / 制度 / 工单（新增）────────────────────────────
    # 这类问题需要调用 RAG 工具（query_policy_knowledge_base）
    # 或工单工具（request_create_hr_ticket），都走 function_calling，
    # 因为它有 ReAct trace，方便展示推理过程和 Human-in-the-loop。
    (
        re.compile(
            r"(年假|年休假|带薪假|病假|事假|产假|陪产假|婚假|丧假"
            r"|报销|差旅|餐补|交通补贴|住宿标准"
            r"|绩效|KPI|OKR|考核|晋升|调薪"
            r"|入职|离职|离职流程|办理离职|转正|试用期"
            r"|考勤|打卡|迟到|早退|旷工|加班"
            r"|制度|政策|规定|规则|员工手册|HR政策|公司规定"
            r"|创建工单|提交工单|HR工单|申请工单|工单申请"
            r"|招聘|薪资|五险一金|社保|公积金)",
            re.IGNORECASE
        ),
        AgentType.function_calling,
        "FC+RAG/Ticket：HR/制度/工单关键词"
    ),

    # ── 纯计算：数字 + 运算符 ─────────────────────────────────
    (
        re.compile(r"[\d\s]+[+\-*/^%]+[\d\s]|^(计算|算一下|等于多少|多少钱)"),
        AgentType.function_calling,
        "FC：纯计算关键词"
    ),
]


# ─────────────────────────────────────────────────────────
# 第二层：LLM 路由 Prompt
# ★ 更新：让模型知道 RAG 工具和工单工具的使用场景
# ─────────────────────────────────────────────────────────
_LLM_ROUTER_PROMPT = """你是一个 Agent 路由专家，判断用户问题应交给哪种 Agent。

【可选 Agent 类型】
- function_calling：
    · 单步任务：简单计算、日常问答
    · HR/制度/政策问答（会调用 RAG 知识库工具）
    · 创建 HR 工单（会调用工单工具，需要人工确认）
- langgraph：需要多步推理的分析类问题
- multi_agent：明确需要多个专业角色协作，如"先搜索再计算"
- mcp：访问企业内网系统，如查员工信息、获取内部数据

【判断原则】
1. 优先选最简单够用的 Agent（function_calling 能搞定就不用 langgraph）
2. HR/制度/政策/工单类 → function_calling（系统会自动调用 RAG 或工单工具）
3. 只有明确多角色分工才选 multi_agent
4. 有企业内网数据需求才选 mcp

【输出格式】
只输出以下词之一，不加任何解释或标点：
function_calling / langgraph / multi_agent / mcp

【示例】
用户: "帮我计算 100 乘以 3.14" → function_calling
用户: "公司年假政策是什么" → function_calling
用户: "我想创建一个 HR 工单申请补假" → function_calling
用户: "先查黄金价格再算买100克要多少钱" → multi_agent
用户: "查工号 E1001 的员工信息" → mcp
用户: "分析电商行业竞争格局并给出多维度建议" → langgraph
"""


class AgentAutoRouter:
    """
    三层漏斗式智能路由器

    使用方式：
        resolved = agent_auto_router.route("年假有几天", explicit_type=AgentType.auto)
        # → AgentType.function_calling（命中 HR 关键词规则）
    """

    def __init__(self):
        self._llm = None  # 懒加载，只有规则层未命中时才初始化

    def _get_llm(self):
        if self._llm is None:
            from app.core.llm_client import get_langchain_llm
            self._llm = get_langchain_llm()
        return self._llm

    def route(
        self,
        user_input: str,
        explicit_type: Optional[AgentType] = None,
    ) -> AgentType:
        """
        路由入口

        ★ [修改] 判断"是否需要自动路由"的逻辑：
          原来：explicit_type != AgentType.function_calling 才走用户指定
               → 当用户传 function_calling 时，路由器错误地认为"用户没指定"
          现在：explicit_type 为 None 或 AgentType.auto → 自动路由
               explicit_type 是其他具体类型 → 直接尊重用户选择
        """
        # ── 优先级 0：用户明确指定了具体 Agent ──────────────
        if explicit_type is not None and explicit_type != AgentType.auto:
            logger.info(f"路由决策 | 用户明确指定 → {explicit_type.value}")
            return explicit_type

        # ── 优先级 1：规则路由（零延迟）─────────────────────
        rule_result = self._rule_route(user_input)
        if rule_result:
            return rule_result

        # ── 优先级 2：LLM 路由（约 200ms）───────────────────
        llm_result = self._llm_route(user_input)
        if llm_result:
            return llm_result

        # ── 优先级 3：兜底 ────────────────────────────────
        logger.info("路由决策 | 兜底降级 → function_calling")
        return AgentType.function_calling

    def _rule_route(self, user_input: str) -> Optional[AgentType]:
        """第一层：正则规则路由"""
        for pattern, agent_type, description in _RULE_ROUTING_TABLE:
            if pattern.search(user_input):
                logger.info(f"路由决策 | 规则命中（{description}）→ {agent_type.value}")
                return agent_type
        return None

    def _llm_route(self, user_input: str) -> Optional[AgentType]:
        """第二层：LLM 判断路由"""
        try:
            llm = self._get_llm()
            messages = [
                SystemMessage(content=_LLM_ROUTER_PROMPT),
                HumanMessage(content=f"用户问题：{user_input}"),
            ]
            raw = llm.invoke(messages).content.strip().lower()

            type_map = {
                "function_calling": AgentType.function_calling,
                "langgraph":        AgentType.langgraph,
                "multi_agent":      AgentType.multi_agent,
                "mcp":              AgentType.mcp,
            }
            for key, agent_type in type_map.items():
                if key in raw:
                    logger.info(f"路由决策 | LLM 判断 → {agent_type.value}（原始: '{raw}'）")
                    return agent_type

        except Exception as e:
            logger.warning(f"LLM 路由失败，降级到兜底路由: {e}")

        return None


# 全局单例
agent_auto_router = AgentAutoRouter()