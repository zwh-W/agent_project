# app/services/pending_action_store.py
"""
等待用户确认的操作存储（Human-in-the-loop 核心组件）

设计思路：
  Agent 调用 request_create_hr_ticket 工具时，工具不直接创建工单，
  而是在这里保存一个"pending action"，并把 action_id 返回给前端。
  前端展示确认对话框，用户点击"确认"后，
  前端调用 POST /v1/confirm，后端从这里取出 pending action，
  再真正调用 ticket_service.create_ticket。

  这就是 Human-in-the-loop（HITL）的完整流程：
    LLM 意图识别 → 工具生成 pending → 用户确认 → 真正执行

状态机：
  pending → confirmed（用户确认）
  pending → cancelled（用户取消 或 超时过期）

存储：
  内存存储，生产环境应换 Redis（支持 TTL 自动过期）
  当前通过 _is_expired() 软过期（查询时判断，不主动清除）
"""
import uuid
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# Pending action 的有效期（超过后自动视为 expired）
_EXPIRE_MINUTES = 30


class PendingActionStore:
    """
    内存版 Pending Action 存储

    线程安全，支持创建、查询、确认、取消操作。
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}  # action_id → action_data
        self._lock = threading.Lock()

    def create_pending_action(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict,
        action_type: str,
    ) -> dict:
        """
        创建一条待确认的操作记录

        Args:
            session_id:  会话 ID（确认时用于鉴权，防止跨会话确认）
            tool_name:   发起操作的工具名，如 "request_create_hr_ticket"
            tool_input:  工具的调用参数
            action_type: 操作类型，如 "create_hr_ticket"

        Returns:
            完整的 pending action 字典
        """
        action_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        action = {
            "action_id":   action_id,
            "session_id":  session_id,
            "tool_name":   tool_name,
            "tool_input":  tool_input,
            "action_type": action_type,
            "created_at":  now.isoformat(),
            "expires_at":  (now + timedelta(minutes=_EXPIRE_MINUTES)).isoformat(),
            "status":      "pending",   # pending / confirmed / cancelled / expired
        }

        with self._lock:
            self._store[action_id] = action

        logger.info(
            f"Pending action 已创建 | action_id={action_id} "
            f"| session_id={session_id[:8]}... | type={action_type}"
        )
        return action

    def get_pending_action(self, action_id: str) -> Optional[dict]:
        """
        查询 pending action

        Returns:
            action dict（status 可能已被更新为 expired），或 None（不存在）
        """
        with self._lock:
            action = self._store.get(action_id)
            if action is None:
                return None

            # 软过期检查：查询时判断，不主动删除
            if action["status"] == "pending" and self._is_expired(action):
                action["status"] = "expired"
                logger.info(f"Pending action 已过期 | action_id={action_id}")

            return dict(action)  # 返回副本，防止外部修改内部状态

    def confirm_action(self, action_id: str, session_id: str) -> dict:
        """
        确认执行操作

        Args:
            action_id:  pending action ID
            session_id: 请求方的 session_id，必须与创建时一致

        Returns:
            更新后的 action dict

        Raises:
            ValueError: action 不存在 / 已处理 / session 不匹配
        """
        with self._lock:
            action = self._store.get(action_id)
            self._validate_action(action, action_id, session_id, require_pending=True)

            if self._is_expired(action):
                action["status"] = "expired"
                raise ValueError(f"Pending action 已过期（超过 {_EXPIRE_MINUTES} 分钟），请重新发起操作")

            action["status"] = "confirmed"
            action["confirmed_at"] = datetime.now(timezone.utc).isoformat()
            confirmed = dict(action)

        logger.info(f"Pending action 已确认 | action_id={action_id}")
        return confirmed

    def cancel_action(self, action_id: str, session_id: str) -> dict:
        """
        取消操作

        Args:
            action_id:  pending action ID
            session_id: 请求方的 session_id

        Returns:
            更新后的 action dict

        Raises:
            ValueError: action 不存在 / 已处理 / session 不匹配
        """
        with self._lock:
            action = self._store.get(action_id)
            self._validate_action(action, action_id, session_id, require_pending=True)

            action["status"] = "cancelled"
            action["cancelled_at"] = datetime.now(timezone.utc).isoformat()
            cancelled = dict(action)

        logger.info(f"Pending action 已取消 | action_id={action_id}")
        return cancelled

    # ─────────────────────────────────────────────────────
    # 私有工具方法
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _is_expired(action: dict) -> bool:
        expires_at = datetime.fromisoformat(action["expires_at"])
        return datetime.now(timezone.utc) > expires_at

    @staticmethod
    def _validate_action(
        action: Optional[dict],
        action_id: str,
        session_id: str,
        require_pending: bool = True,
    ):
        """统一校验逻辑，抛出 ValueError 给调用方处理"""
        if action is None:
            raise ValueError(f"找不到 action_id={action_id}，请确认 ID 是否正确")

        if action["session_id"] != session_id:
            # 安全：不暴露 session_id 是否存在，只说"无权操作"
            raise ValueError("无权操作此 action：session_id 不匹配")

        if require_pending and action["status"] != "pending":
            status = action["status"]
            status_desc = {
                "confirmed": "已确认执行",
                "cancelled": "已取消",
                "expired":   "已过期",
            }.get(status, status)
            raise ValueError(f"此操作已处理（当前状态：{status_desc}），无法重复操作")


# 全局单例
pending_action_store = PendingActionStore()