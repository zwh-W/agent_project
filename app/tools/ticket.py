# app/tools/ticket.py
"""
HR 工单工具

核心设计：Human-in-the-loop（人机协同）
  这个工具是 LLM 可以调用的，但它绝对不会直接创建工单。
  它的唯一作用是：
    1. 整理用户的意图（title、description、ticket_type）
    2. 在 PendingActionStore 里创建一条等待确认的记录
    3. 把 action_id 和确认提示返回给 Agent
    4. Agent 把这些信息放进 ChatResponse.pending_action
    5. 用户在前端点击"确认"后，调用 POST /v1/confirm
    6. /v1/confirm 接口才真正调用 ticket_service.create_ticket

为什么这样设计？
  企业场景中，LLM 可能误解用户意图，直接执行写操作风险很高。
  "用户说'帮我提个假期申请'不等于真的要提交"。
  强制人工确认是企业 Agent 的基本安全原则。

给 LLM 看到的工具描述必须强调：这个工具不创建工单，只生成待确认操作。

注意：session_id 注入问题
  工具本身无法感知当前是哪个 session 的对话。
  解决方案：在工具函数签名里加 session_id 参数，
  由 FunctionCallingAgent 在调用工具前通过 tool_input 注入。
  但这会暴露 session_id 给 LLM（LLM 可能乱填）。
  更优雅的方案是用 contextvars，但为保持简单，
  当前使用 session_id 作为可选参数，
  agent_router.py 在解析 trace 时会从 req.session_id 补填。
"""
import json
from typing import Optional
from langchain_core.tools import tool

from app.services.pending_action_store import pending_action_store
from app.core.logger import get_logger

logger = get_logger(__name__)

# 支持的工单类型（LLM 参考）
TICKET_TYPES = {
    "leave_request": "休假申请",
    "reimbursement": "报销申请",
    "onboarding": "入职相关",
    "offboarding": "离职相关",
    "salary_inquiry": "薪资咨询",
    "general_hr": "其他 HR 事务",
}


@tool
def request_create_hr_ticket(
        title: str,
        description: str,
        ticket_type: str = "general_hr",
        session_id: str = "unknown",
) -> str:
    """
    发起创建 HR 工单的请求（需要用户确认后才真正创建）。

    此工具不会直接创建工单，而是生成一个待用户确认的操作。
    用户确认后，系统才会真正提交工单。

    适用场景：
    - 用户明确表示要创建/提交/申请 HR 工单
    - 年假申请、报销申请、入职/离职办理等需要正式记录的事项
    - 任何需要 HR 部门人工处理的事务
     重要规则：
    - 当用户明确表示要创建、提交、申请、报销、发起 HR 工单时，必须调用本工具。
    - 即使用户提供的信息不完整，也应先基于已有信息生成待确认工单。
    - 缺失信息可以在 description 中写“待补充”，例如“出差日期待补充”“发票信息待补充”。
    - 不要因为缺少出差时间、发票、工号、城市等信息而只反问用户。
    - 创建工单时应尽量保留用户原始表达，不要擅自添加用户没有说过的事实。

    示例：
    用户说“帮我报销上次出差住宿费1200元”，即使未提供出差日期、城市、发票信息，也应调用本工具：
    title="差旅住宿费报销"
    description="申请报销上次出差住宿费1200元，出差日期、城市、发票信息待补充。"
    ticket_type="reimbursement"

    Args:
        title:       工单标题，简明描述申请事项（20字以内）
        description: 详细描述，包含时间、原因、具体需求等
        ticket_type: 工单类型，可选值：
                     leave_request（休假申请）、reimbursement（报销申请）、
                     onboarding（入职相关）、offboarding（离职相关）、
                     salary_inquiry（薪资咨询）、general_hr（其他 HR 事务）
        session_id:  当前会话 ID（系统自动填入，你不需要修改）

    Returns:
        JSON 字符串，包含 need_confirmation=true 和 action_id，
        等待用户在前端确认后执行。
    """
    logger.info(
        f"request_create_hr_ticket 被调用 | "
        f"session_id={session_id[:8]}... | type={ticket_type} | title={title}"
    )

    # ── 参数校验 ──────────────────────────────────────────
    if not title or not title.strip():
        return json.dumps({
            "success": False,
            "error": "工单标题不能为空，请提供简明的申请标题"
        }, ensure_ascii=False)

    if not description or not description.strip():
        return json.dumps({
            "success": False,
            "error": "工单描述不能为空，请描述具体的申请内容和需求"
        }, ensure_ascii=False)

    # 标准化 ticket_type，非法值降级为 general_hr
    if ticket_type not in TICKET_TYPES:
        logger.warning(f"未知 ticket_type={ticket_type}，降级为 general_hr")
        ticket_type = "general_hr"

    ticket_type_cn = TICKET_TYPES[ticket_type]

    # ── 创建 Pending Action（不创建真实工单）────────────
    try:
        action = pending_action_store.create_pending_action(
            session_id=session_id,
            tool_name="request_create_hr_ticket",
            tool_input={
                "title": title.strip(),
                "description": description.strip(),
                "ticket_type": ticket_type,
            },
            action_type="create_hr_ticket",
        )
    except Exception as e:
        logger.error(f"创建 pending action 失败: {e}")
        return json.dumps({
            "success": False,
            "error": f"系统错误，无法生成工单确认请求: {str(e)}"
        }, ensure_ascii=False)

    # ── 构造返回给 Agent 的确认信息 ───────────────────────
    confirm_message = (
        f"已为您准备好 HR 工单，请确认以下信息后提交：\n\n"
        f"📋 工单类型：{ticket_type_cn}\n"
        f"📌 标题：{title.strip()}\n"
        f"📝 描述：{description.strip()}\n\n"
        f"⚠️ 请注意：点击确认后将正式提交，HR 部门将收到此工单。"
    )

    result = {
        "need_confirmation": True,  # ★ 关键标志，agent_router.py 会检测这个字段
        "action_id": action["action_id"],
        "action_type": "create_hr_ticket",
        "message": confirm_message,
        "pending_action": {
            "ticket_type": ticket_type,
            "ticket_type_cn": ticket_type_cn,
            "title": title.strip(),
            "description": description.strip(),
        },
    }

    logger.info(f"Pending action 已就绪 | action_id={action['action_id']}")
    return json.dumps(result, ensure_ascii=False, indent=2)
