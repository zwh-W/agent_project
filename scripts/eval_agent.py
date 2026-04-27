# scripts/eval_agent.py
"""
★ [新增模块] Agent 基础评测脚本

面试必问："你怎么知道你的 Agent 效果好？用什么指标？"
这个脚本提供了三个核心指标的自动化评测：

1. 工具调用准确率（Tool Call Accuracy）
   - 该用工具的时候用了吗？
   - 用了正确的工具吗？

2. 答案正确率（Answer Accuracy）
   - 最终答案是否包含预期关键词（keyword match）
   - 更严格的可以用 LLM 作为评委（LLM-as-Judge）

3. 延迟（Latency P50/P95）
   - Agent 思考 + 工具调用的整体耗时

使用方式：
    python scripts/eval_agent.py --agent function_calling
    python scripts/eval_agent.py --agent langgraph --verbose
"""
import argparse
import time
import json
import statistics
import requests
from dataclasses import dataclass, field
from typing import List, Optional

# ── 测试集定义 ──────────────────────────────────────────────
# 每个 case 定义：输入、期望工具调用、期望答案关键词
@dataclass
class EvalCase:
    name: str
    user_input: str
    expected_tool: Optional[str] = None       # 期望调用的工具名（None=不期望调工具）
    expected_keywords: List[str] = field(default_factory=list)  # 答案中应包含的关键词
    should_not_keywords: List[str] = field(default_factory=list)  # 答案中不应包含的词（幻觉检测）


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        name="纯计算_简单加法",
        user_input="请计算 123 加上 456 等于多少",
        expected_tool="calculator",
        expected_keywords=["579"],
    ),
    EvalCase(
        name="纯计算_幂运算",
        user_input="2 的 10 次方是多少",
        expected_tool="calculator",
        expected_keywords=["1024"],
    ),
    EvalCase(
        name="常识问题_不需要工具",
        user_input="你好，请自我介绍一下",
        expected_tool=None,  # 不应该调工具
        expected_keywords=["助手", "AI"],
    ),
    EvalCase(
        name="搜索_实时信息",
        user_input="今天是几号？帮我搜索一下今天的天气新闻",
        expected_tool="web_search",
        expected_keywords=[],  # 实时数据不校验具体内容
    ),
    EvalCase(
        name="复合任务_搜索后计算",
        user_input="帮我计算 (100 + 200) * 3 的结果，然后告诉我这个数字用中文怎么说",
        expected_tool="calculator",
        expected_keywords=["900", "九百"],
    ),
]


# ── 评测执行器 ────────────────────────────────────────────
@dataclass
class EvalResult:
    case_name: str
    success: bool
    tool_match: Optional[bool]   # None 表示不检查
    answer_match: bool
    latency_ms: float
    answer: str
    error: Optional[str] = None


def run_single_eval(
    case: EvalCase,
    agent_type: str,
    api_url: str,
    api_key: str,
    session_id: str,
    verbose: bool = False,
) -> EvalResult:
    """执行单个评测 case"""
    start_time = time.time()

    try:
        resp = requests.post(
            f"{api_url}/v1/chat",
            json={
                "message": case.user_input,
                "session_id": session_id,
                "agent_type": agent_type,
            },
            headers={"X-API-Key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        latency_ms = (time.time() - start_time) * 1000

        # 工具调用检查（通过 tool_calls 字段）
        tool_match = None
        tool_calls = data.get("tool_calls", [])
        if case.expected_tool is not None:
            called_tools = [tc.get("tool_name", "") for tc in tool_calls]
            tool_match = case.expected_tool in called_tools

        # 答案关键词检查
        answer_match = True
        if case.expected_keywords:
            answer_match = any(kw in answer for kw in case.expected_keywords)

        # 幻觉检测
        hallucination = any(kw in answer for kw in case.should_not_keywords)
        if hallucination:
            answer_match = False

        success = (tool_match is None or tool_match) and answer_match

        if verbose:
            print(f"\n  答案: {answer[:150]}...")
            print(f"  工具调用: {[tc.get('tool_name') for tc in tool_calls]}")
            print(f"  延迟: {latency_ms:.0f}ms")

        return EvalResult(
            case_name=case.name,
            success=success,
            tool_match=tool_match,
            answer_match=answer_match,
            latency_ms=latency_ms,
            answer=answer,
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return EvalResult(
            case_name=case.name,
            success=False,
            tool_match=None,
            answer_match=False,
            latency_ms=latency_ms,
            answer="",
            error=str(e),
        )


def run_eval(agent_type: str, api_url: str, api_key: str, verbose: bool = False):
    """运行完整评测，输出报告"""
    print(f"\n{'='*60}")
    print(f"  Agent 评测报告")
    print(f"  Agent 类型: {agent_type}")
    print(f"  测试集大小: {len(EVAL_CASES)} cases")
    print(f"{'='*60}")

    results: List[EvalResult] = []
    session_id = f"eval_{agent_type}_{int(time.time())}"

    for i, case in enumerate(EVAL_CASES, 1):
        print(f"\n[{i}/{len(EVAL_CASES)}] {case.name}", end="", flush=True)
        result = run_single_eval(case, agent_type, api_url, api_key, session_id, verbose)
        results.append(result)
        status = "✅" if result.success else "❌"
        print(f" {status} ({result.latency_ms:.0f}ms)", end="")
        if result.error:
            print(f" ⚠️ Error: {result.error}", end="")

    # ── 汇总统计 ────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r.success)
    latencies = [r.latency_ms for r in results if r.error is None]

    tool_cases = [r for r in results if r.tool_match is not None]
    tool_accuracy = sum(1 for r in tool_cases if r.tool_match) / len(tool_cases) if tool_cases else 0

    print(f"\n\n{'='*60}")
    print(f"  📊 评测结果汇总")
    print(f"{'='*60}")
    print(f"  总体通过率:     {passed}/{total} = {passed/total*100:.1f}%")
    print(f"  工具调用准确率: {tool_accuracy*100:.1f}%")
    if latencies:
        print(f"  延迟 P50:       {statistics.median(latencies):.0f}ms")
        print(f"  延迟 P95:       {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  延迟 Max:       {max(latencies):.0f}ms")

    # 失败 case 详情
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n  ❌ 失败 Cases ({len(failed)}):")
        for r in failed:
            print(f"    - {r.case_name}: tool_match={r.tool_match}, answer_match={r.answer_match}")
            if r.error:
                print(f"      Error: {r.error}")

    print(f"{'='*60}\n")

    # 输出 JSON 报告（方便 CI/CD 集成）
    report = {
        "agent_type": agent_type,
        "total": total,
        "passed": passed,
        "accuracy": passed / total,
        "tool_accuracy": tool_accuracy,
        "latency_p50": statistics.median(latencies) if latencies else 0,
        "results": [
            {
                "case": r.case_name,
                "success": r.success,
                "tool_match": r.tool_match,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
    }

    report_file = f"eval_report_{agent_type}_{int(time.time())}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  详细报告已保存: {report_file}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 评测脚本")
    parser.add_argument("--agent", default="function_calling",
                        choices=["function_calling", "langgraph", "multi_agent", "mcp"],
                        help="要评测的 Agent 类型")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--key", default="dev-key-123", help="API Key")
    parser.add_argument("--verbose", action="store_true", help="打印详细输出")
    args = parser.parse_args()

    run_eval(args.agent, args.url, args.key, args.verbose)
