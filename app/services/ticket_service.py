# app/services/ticket_service.py
"""
HR 工单服务

职责：
  真正创建工单的地方。只能由 /v1/confirm 接口在用户确认后调用，
  绝对不暴露给 LLM 直接调用——这是 Human-in-the-loop 的核心边界。

存储方案：
  当前使用内存列表 + JSON 文件双重存储：
  - 内存：提供极速查询（服务重启后靠 JSON 文件恢复）
  - JSON 文件：简单持久化，生产环境应换成数据库

工单 ID 格式：HR-YYYYMMDD-NNNN
  例：HR-20260506-0001
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# 工单持久化文件路径（相对于项目根目录）
_TICKET_FILE = Path(__file__).parent.parent.parent / "data" / "tickets.json"


class TicketService:
    """
    HR 工单服务（内存 + JSON 文件存储）

    设计要点：
      - 线程安全（FastAPI 多线程处理请求）
      - 工单 ID 格式固定，方便人工核对
      - 存储故障不影响服务启动（graceful fallback）
    """

    def __init__(self):
        self._tickets: Dict[str, dict] = {}  # ticket_id → ticket_data
        self._lock = threading.Lock()
        self._daily_counter: Dict[str, int] = {}  # date_str → counter
        self._load_from_file()  # 启动时从文件恢复

    # ─────────────────────────────────────────────────────
    # 公开方法
    # ─────────────────────────────────────────────────────

    def create_ticket(
        self,
        ticket_type: str,
        title: str,
        description: str,
        session_id: str,
        created_by: Optional[str] = None,
    ) -> dict:
        """
        创建工单

        Args:
            ticket_type:  工单类型，如 "leave_request"、"reimbursement"、"general_hr"
            title:        工单标题
            description:  详细描述
            session_id:   会话 ID（用于关联用户）
            created_by:   创建人，可选（如有用户体系可传用户名）

        Returns:
            工单完整信息 dict，包含 ticket_id、status、created_at 等
        """
        with self._lock:
            ticket_id = self._generate_ticket_id()
            now = datetime.now(timezone.utc).isoformat()

            ticket = {
                "ticket_id":   ticket_id,
                "ticket_type": ticket_type,
                "title":       title,
                "description": description,
                "status":      "open",          # open / processing / closed
                "session_id":  session_id,
                "created_by":  created_by or session_id,
                "created_at":  now,
                "updated_at":  now,
            }

            self._tickets[ticket_id] = ticket
            logger.info(f"✅ 工单已创建: {ticket_id} | 类型: {ticket_type} | 标题: {title}")

        # 持久化（在锁外执行，避免 I/O 阻塞锁）
        self._save_to_file()
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        """查询工单"""
        with self._lock:
            return self._tickets.get(ticket_id)

    def list_tickets(self, session_id: Optional[str] = None) -> List[dict]:
        """列出工单（可按 session_id 筛选）"""
        with self._lock:
            tickets = list(self._tickets.values())
        if session_id:
            tickets = [t for t in tickets if t.get("session_id") == session_id]
        return sorted(tickets, key=lambda t: t["created_at"], reverse=True)

    # ─────────────────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────────────────

    def _generate_ticket_id(self) -> str:
        """
        生成工单 ID：HR-YYYYMMDD-NNNN
        格式示例：HR-20260506-0001
        每天计数从 0001 重新开始
        """
        today = datetime.now().strftime("%Y%m%d")
        count = self._daily_counter.get(today, 0) + 1
        self._daily_counter[today] = count
        return f"HR-{today}-{count:04d}"

    def _save_to_file(self):
        """持久化到 JSON 文件"""
        try:
            _TICKET_FILE.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "tickets": list(self._tickets.values()),
                    "daily_counter": self._daily_counter,
                }
            _TICKET_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            # 持久化失败不影响服务正常运行
            logger.error(f"工单持久化失败（内存数据仍然有效）: {e}")

    def _load_from_file(self):
        """从 JSON 文件恢复数据"""
        if not _TICKET_FILE.exists():
            return
        try:
            raw = json.loads(_TICKET_FILE.read_text(encoding="utf-8"))
            for t in raw.get("tickets", []):
                self._tickets[t["ticket_id"]] = t
            self._daily_counter = raw.get("daily_counter", {})
            logger.info(f"从文件恢复 {len(self._tickets)} 条工单记录")
        except Exception as e:
            logger.warning(f"工单文件加载失败（将从空状态启动）: {e}")


# 全局单例
ticket_service = TicketService()