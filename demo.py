import re
from enum import Enum


# 1. 定义 Agent 类型（模拟）
class AgentType(Enum):
    mcp = "MCP 企业工具Agent"
    multi_agent = "Multi-Agent 多智能体"
    function_calling = "函数调用Agent"
    auto = "自动LLM路由"


# 2. 你的规则路由表（原样复制）
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


# 3. 路由函数（你写的循环）
def rule_route(user_input: str):
    print(f"\n用户输入：{user_input}")

    # 核心循环！你问的就是这段
    for pattern, agent_type, description in _RULE_ROUTING_TABLE:
        print(f"正在匹配规则：{description}")

        # 正则搜索
        if pattern.search(user_input):
            print(f"✅ 匹配成功 → 使用：{agent_type.value}")
            return agent_type

    # 所有规则都没命中
    print("❌ 规则全部不匹配 → 进入 LLM 自动路由")
    return AgentType.auto


# 4. 测试用例（你可以随便改）
if __name__ == "__main__":
    # 测试 1：MCP
    rule_route("帮我查一下员工信息")

    # 测试 2：Multi-Agent
    rule_route("先搜索一下价格，再算一下总和")

    # 测试 3：计算器
    rule_route("1+1等于多少")

    # 测试 4：无规则命中（走LLM）
    rule_route("你好，今天天气怎么样")