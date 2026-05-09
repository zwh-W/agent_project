# app/services/pending_action_store.py
"""
等待用户确认的操作存储（Human-in-the-loop 核心组件）

P0 修复版改动：
  1. 新增 bind_session_if_unknown()，避免 router 直接修改 _store 私有字段。
  2. 将确认执行拆成两阶段：
       pending -> executing -> confirmed
       pending -> executing -> failed
     避免“action 已 confirmed，但真实业务创建失败”导致状态不一致。
  3. 新增 mark_confirmed() / mark_failed()，由 /v1/confirm 在业务执行成功或失败后调用。
  4. 保留 confirm_action() 作为兼容别处旧调用的别名，但语义改为“开始执行”。
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

    线程安全，支持创建、查询、绑定 session、开始执行、确认完成、标记失败、取消操作。

    状态机：
      pending   -> executing
      executing -> confirmed
      executing -> failed
      pending   -> cancelled
      pending   -> expired
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}  # action_id -> action_data
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
            "status":      "pending",   # pending / executing / confirmed / failed / cancelled / expired
        }

        with self._lock:
            self._store[action_id] = action

        logger.info(
            f"Pending action 已创建 | action_id={action_id} "
            f"| session_id={session_id[:8]}... | type={action_type}"
        )
        return dict(action)

    def bind_session_if_unknown(self, action_id: str, session_id: str) -> Optional[dict]:
        """
        ★ [P0 FIX] 将 session_id 从 unknown 绑定为真实会话 ID。

        背景：
          LLM 调用工具时可能无法获知真实 session_id，ticket 工具会用 "unknown" 占位。
          router 解析工具返回的 action_id 后，需要把 pending action 绑定到真实 session。
          旧实现直接访问 pending_action_store._store，破坏封装且没有锁保护。
        """
        with self._lock:
            action = self._store.get(action_id)
            if action is None:
                return None

            if action.get("session_id") == "unknown":
                action["session_id"] = session_id
                action["bound_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(
                    f"Pending action 已绑定真实 session | action_id={action_id} "
                    f"| session_id={session_id[:8]}..."
                )

            return dict(action)

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
                action["expired_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"Pending action 已过期 | action_id={action_id}")

            return dict(action)  # 返回副本，防止外部修改内部状态

    def begin_execute_action(self, action_id: str, session_id: str) -> dict:
        """
        ★ [P0 FIX] 用户确认后，先把 action 标记为 executing，而不是直接 confirmed。

        Args:
            action_id:  pending action ID
            session_id: 请求方的 session_id，必须与创建时一致

        Returns:
            更新后的 action dict

        Raises:
            ValueError: action 不存在 / 已处理 / session 不匹配 / 已过期
        """
        with self._lock:
            action = self._store.get(action_id)
            self._validate_action(action, action_id, session_id, require_pending=True)

            if self._is_expired(action):
                action["status"] = "expired"
                action["expired_at"] = datetime.now(timezone.utc).isoformat()
                raise ValueError(f"Pending action 已过期（超过 {_EXPIRE_MINUTES} 分钟），请重新发起操作")

            action["status"] = "executing"
            action["executing_at"] = datetime.now(timezone.utc).isoformat()
            executing = dict(action)

        logger.info(f"Pending action 开始执行 | action_id={action_id}")
        return executing

    def confirm_action(self, action_id: str, session_id: str) -> dict:
        """
        兼容旧调用：语义等同 begin_execute_action。

        注意：
          真实业务执行成功后，调用方还必须调用 mark_confirmed(action_id)。
          不建议新代码继续直接依赖这个方法名，推荐用 begin_execute_action。
        """
        return self.begin_execute_action(action_id, session_id)

    def mark_confirmed(self, action_id: str) -> dict:
        """
        ★ [P0 FIX] 真实业务执行成功后，标记 action 为 confirmed。
        """
        with self._lock:
            action = self._store.get(action_id)
            if action is None:
                raise ValueError(f"找不到 action_id={action_id}，请确认 ID 是否正确")

            if action["status"] != "executing":
                raise ValueError(f"无法确认完成：当前状态为 {action['status']}，期望 executing")

            action["status"] = "confirmed"
            action["confirmed_at"] = datetime.now(timezone.utc).isoformat()
            confirmed = dict(action)

        logger.info(f"Pending action 已确认完成 | action_id={action_id}")
        return confirmed

    def mark_failed(self, action_id: str, error: str) -> dict:
        """
        ★ [P0 FIX] 真实业务执行失败后，标记 action 为 failed，避免状态卡死为 confirmed。
        """
        with self._lock:
            action = self._store.get(action_id)
            if action is None:
                raise ValueError(f"找不到 action_id={action_id}，请确认 ID 是否正确")

            action["status"] = "failed"
            action["failed_at"] = datetime.now(timezone.utc).isoformat()
            action["error"] = error
            failed = dict(action)

        logger.warning(f"Pending action 执行失败 | action_id={action_id} | error={error}")
        return failed

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

            if self._is_expired(action):
                action["status"] = "expired"
                action["expired_at"] = datetime.now(timezone.utc).isoformat()
                raise ValueError(f"Pending action 已过期（超过 {_EXPIRE_MINUTES} 分钟），无法取消")

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
                "executing": "正在执行",
                "confirmed": "已确认执行",
                "failed":    "执行失败",
                "cancelled": "已取消",
                "expired":   "已过期",
            }.get(status, status)
            raise ValueError(f"此操作已处理（当前状态：{status_desc}），无法重复操作")


# 全局单例
pending_action_store = PendingActionStore()
