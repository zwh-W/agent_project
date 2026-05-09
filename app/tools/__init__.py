# app/tools/__init__.py
"""
工具初始化与注册

变更：
  v2.1  新增 query_policy_knowledge_base（RAG 知识库查询）
        新增 request_create_hr_ticket（HR 工单 Human-in-the-loop）

工具分类标签说明：
  math              → 数学计算类
  search            → 实时网络搜索
  rag / policy /
  knowledge_base /
  hr                → 企业知识库 / HR 政策查询
  ticket / hr /
  hitl              → 工单创建（Human-in-the-loop）
"""
from app.tools.registry import tool_registry
from app.tools.calculator import calculator
from app.tools.search import web_search
from app.tools.rag_search import query_policy_knowledge_base   # ★ 新增
from app.tools.ticket import request_create_hr_ticket          # ★ 新增

# ── 注册：数学计算 ────────────────────────────────────────
tool_registry.register(
    calculator,
    tags={"math", "calculation"},
    description="AST 安全数学表达式求值",
)

# ── 注册：网络搜索 ────────────────────────────────────────
tool_registry.register(
    web_search,
    tags={"search", "information_retrieval"},
    description="DuckDuckGo 实时网络搜索",
)

# ── 注册：RAG 知识库查询 ★ 新增 ───────────────────────────
# tags 包含 hr，方便 MultiAgent 中的 HR 专员 Agent 按标签取工具
tool_registry.register(
    query_policy_knowledge_base,
    tags={"rag", "policy", "knowledge_base", "hr"},
    description="企业 HR 政策 / 制度知识库 RAG 查询",
)

# ── 注册：HR 工单（Human-in-the-loop）★ 新增 ─────────────
tool_registry.register(
    request_create_hr_ticket,
    tags={"ticket", "hr", "hitl"},
    description="发起 HR 工单创建请求（需用户确认）",
)

# ── 对外暴露 ──────────────────────────────────────────────
# 保持向后兼容：所有已有 Agent 中的 from app.tools import ALL_TOOLS 无需修改
ALL_TOOLS = tool_registry.get_tools()