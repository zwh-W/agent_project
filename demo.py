from app.services.pending_action_store import pending_action_store
from app.services.ticket_service import ticket_service


session_id = "debug-session-001"

# 1. 创建 pending action
pending = pending_action_store.create_pending_action(
    session_id=session_id,
    tool_name="request_create_hr_ticket",
    tool_input={
        "ticket_type": "leave_request",
        "title": "年假补发申请",
        "description": "用户希望申请补发年假"
    },
    action_type="create_hr_ticket",
)

print("pending action:")
print(pending)

action_id = pending["action_id"]

# 2. 查询 pending action
found = pending_action_store.get_pending_action(action_id)
print("\nfound action:")
print(found)

# 3. 确认 action
confirmed = pending_action_store.confirm_action(
    action_id=action_id,
    session_id=session_id,
)

print("\nconfirmed action:")
print(confirmed)

# 4. 真正创建工单
tool_input = confirmed["tool_input"]

ticket = ticket_service.create_ticket(
    ticket_type=tool_input["ticket_type"],
    title=tool_input["title"],
    description=tool_input["description"],
    session_id=session_id,
)

print("\ncreated ticket:")
print(ticket)