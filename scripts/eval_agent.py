# scripts/eval_agent.py
"""
Agent 自动化评测脚本

测试集覆盖（28 个 case）：
  A. 普通问答，不应调工具            （3 个）
  B. 纯计算，应调 calculator         （4 个）
  C. 网络搜索，应调 web_search        （3 个）
  D. HR / 制度，应调 RAG 工具         （6 个）
  E. 工单创建，应调 ticket 工具       （3 个）
     并检测 need_confirmation=True
  F. 多步骤，先搜索再计算             （3 个）
  G. 反例：不该调工具                 （3 个）
  H. 无结果/无法回答，不应胡编        （3 个）

评测指标：
  - 总体通过率（overall_accuracy）
  - 工具调用准确率（tool_accuracy）
  - 答案关键词命中率（answer_accuracy）
  - 确认标志准确率（confirmation_accuracy）：工单 case 是否返回 need_confirmation
  - 拒绝答题准确率（refusal_accuracy）：无依据问题是否拒答或提示无相关信息
  - 延迟 P50 / P95 / Max
"""
import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


# ─────────────────────────────────────────────────────────
# 测试用例定义
# ─────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    name: str
    user_input: str
    category: str                           # A/B/C/D/E/F/G/H
    expected_tool: Optional[str] = None     # 期望调用的工具（None=不调工具）
    expected_keywords: List[str] = field(default_factory=list)  # 答案中应包含的词
    should_not_keywords: List[str] = field(default_factory=list) # 答案中不应有的词（幻觉检测）
    expect_confirmation: bool = False       # 是否期望 need_confirmation=True
    expect_refusal: bool = False            # 是否期望拒答（答案中含"无法"/"不知道"/"联系HR"等）
    weak_tool_check: bool = False           # True=只检查工具是否被调用（不强校验答案）


EVAL_CASES: List[EvalCase] = [

    # ══════════════════════════════════════════════════════
    # A. 普通问答（不应调工具）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="A1_自我介绍",
        user_input="你好，请简单介绍一下你自己",
        category="A",
        expected_tool=None,
        expected_keywords=["助手", "AI", "帮助"],
    ),
    EvalCase(
        name="A2_常识问题",
        user_input="水的化学式是什么",
        category="A",
        expected_tool=None,
        expected_keywords=["H₂O", "H2O"],
    ),
    EvalCase(
        name="A3_简单问候",
        user_input="今天天气怎么样？就聊聊天",
        category="A",
        expected_tool=None,
        expected_keywords=[],  # 只检查不调工具
    ),

    # ══════════════════════════════════════════════════════
    # B. 纯计算（应调 calculator）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="B1_加法",
        user_input="请计算 123 加上 456 等于多少",
        category="B",
        expected_tool="calculator",
        expected_keywords=["579"],
    ),
    EvalCase(
        name="B2_幂运算",
        user_input="2 的 10 次方是多少",
        category="B",
        expected_tool="calculator",
        expected_keywords=["1024"],
    ),
    EvalCase(
        name="B3_复合运算",
        user_input="请帮我计算 (100 + 200) * 3",
        category="B",
        expected_tool="calculator",
        expected_keywords=["900"],
    ),
    EvalCase(
        name="B4_百分比计算",
        user_input="15000 元的 20% 是多少",
        category="B",
        expected_tool="calculator",
        expected_keywords=["3000"],
    ),

    # ══════════════════════════════════════════════════════
    # C. 网络搜索（应调 web_search，弱校验）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="C1_实时新闻",
        user_input="帮我搜索一下今天有什么重要新闻",
        category="C",
        expected_tool="web_search",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="C2_最新技术",
        user_input="最近 AI 大模型有什么新进展？请搜索一下",
        category="C",
        expected_tool="web_search",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="C3_股价查询",
        user_input="帮我查一下阿里巴巴最近的股价走势",
        category="C",
        expected_tool="web_search",
        expected_keywords=[],
        weak_tool_check=True,
    ),

    # ══════════════════════════════════════════════════════
    # D. HR / 制度 / 政策（应调 query_policy_knowledge_base）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="D1_年假政策",
        user_input="入职满一年有几天年假？请根据公司制度回答",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],   # RAG 可能返回"无相关内容"，弱校验答案
        weak_tool_check=True,
    ),
    EvalCase(
        name="D2_差旅报销标准",
        user_input="出差住宿费用的报销标准是多少？",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="D3_绩效考核周期",
        user_input="公司绩效考核是每季度还是每年？",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="D4_离职流程",
        user_input="员工离职需要提前多少天申请？离职流程是什么？",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="D5_试用期规定",
        user_input="试用期是多长时间？转正需要什么条件？",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="D6_产假政策",
        user_input="女员工产假有多少天？工资怎么发放？",
        category="D",
        expected_tool="query_policy_knowledge_base",
        expected_keywords=[],
        weak_tool_check=True,
    ),

    # ══════════════════════════════════════════════════════
    # E. 工单创建（应调 request_create_hr_ticket + need_confirmation=True）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="E1_申请年假工单",
        user_input="我想创建一个 HR 工单，申请下周一到周三三天年假",
        category="E",
        expected_tool="request_create_hr_ticket",
        expect_confirmation=True,
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="E2_报销工单",
        user_input="帮我提交一个差旅报销工单，上次出差花了1200元住宿费",
        category="E",
        expected_tool="request_create_hr_ticket",
        expect_confirmation=True,
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="E3_通用HR工单",
        user_input="我需要创建一个 HR 工单，问题是关于我的社保缴纳基数",
        category="E",
        expected_tool="request_create_hr_ticket",
        expect_confirmation=True,
        expected_keywords=[],
        weak_tool_check=True,
    ),

    # ══════════════════════════════════════════════════════
    # F. 多步骤任务（先搜索再计算）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="F1_搜索后计算",
        user_input="先搜索一下当前美元对人民币的汇率，再帮我算100美元等于多少人民币",
        category="F",
        expected_tool="web_search",  # 第一步应调搜索
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="F2_查询后比较",
        user_input="搜索一下 Python 和 Java 的就业市场，再计算哪个方向的平均薪资更高",
        category="F",
        expected_tool="web_search",
        expected_keywords=[],
        weak_tool_check=True,
    ),
    EvalCase(
        name="F3_调研后汇总",
        user_input="调研一下国内主要云服务商的价格，然后给我一个对比分析",
        category="F",
        expected_tool="web_search",
        expected_keywords=[],
        weak_tool_check=True,
    ),

    # ══════════════════════════════════════════════════════
    # G. 反例：不该调工具的问题
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="G1_数学知识",
        user_input="请解释什么是勾股定理",
        category="G",
        expected_tool=None,
        expected_keywords=["直角", "斜边", "两边"],
    ),
    EvalCase(
        name="G2_编程概念",
        user_input="Python 里的 list comprehension 是什么意思？举个例子",
        category="G",
        expected_tool=None,
        expected_keywords=["列表", "推导"],
    ),
    EvalCase(
        name="G3_语言问题",
        user_input="'serendipity' 这个英文单词是什么意思",
        category="G",
        expected_tool=None,
        expected_keywords=["意外", "巧合", "偶然"],
    ),

    # ══════════════════════════════════════════════════════
    # H. 无结果/无依据，不应胡编（期望拒答或提示不确定）
    # ══════════════════════════════════════════════════════
    EvalCase(
        name="H1_不存在的制度",
        user_input="公司的'宇宙级别特殊假期'政策是什么？",
        category="H",
        expected_tool="query_policy_knowledge_base",
        expect_refusal=True,
        should_not_keywords=["10天", "20天", "30天"],  # 不能编造假期天数
        weak_tool_check=True,
    ),
    EvalCase(
        name="H2_超出范围",
        user_input="请告诉我公司 CEO 的手机号",
        category="H",
        expected_tool=None,
        expect_refusal=True,
        should_not_keywords=["138", "139", "186"],  # 不能编造手机号
    ),
    EvalCase(
        name="H3_未来预测",
        user_input="明年公司会给我涨多少薪水？",
        category="H",
        expected_tool=None,
        expect_refusal=True,
        should_not_keywords=["30%", "50%", "一定会"],  # 不能做肯定性预测
    ),
]


# ─────────────────────────────────────────────────────────
# 评测结果数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_name: str
    category: str
    success: bool
    tool_match: Optional[bool]
    answer_match: bool
    confirmation_match: Optional[bool]   # None=不检查
    refusal_match: Optional[bool]        # None=不检查
    latency_ms: float
    answer: str
    called_tools: List[str] = field(default_factory=list)
    need_confirmation: bool = False
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────
# 评测执行
# ─────────────────────────────────────────────────────────

def run_single_eval(
    case: EvalCase,
    agent_type: str,
    api_url: str,
    api_key: str,
    session_id: str,
    verbose: bool = False,
) -> EvalResult:
    start = time.time()
    try:
        resp = requests.post(
            f"{api_url}/v1/chat",
            json={
                "message": case.user_input,
                "session_id": session_id,
                "agent_type": agent_type,
            },
            headers={"X-API-Key": api_key},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - start) * 1000

        answer          = data.get("answer", "")
        tool_calls      = data.get("tool_calls", [])
        need_confirm    = data.get("need_confirmation", False)
        called_tools    = [tc.get("tool_name", "") for tc in tool_calls]

        # ── 工具调用检查 ──────────────────────────────────
        tool_match = None
        if case.expected_tool is not None:
            tool_match = case.expected_tool in called_tools

        # ── 答案关键词检查 ────────────────────────────────
        answer_match = True
        if case.expected_keywords:
            answer_match = any(kw in answer for kw in case.expected_keywords)
        # 幻觉检测：不应出现的词
        if case.should_not_keywords:
            if any(kw in answer for kw in case.should_not_keywords):
                answer_match = False

        # ── 确认标志检查（工单 case）──────────────────────
        confirmation_match = None
        if case.expect_confirmation:
            confirmation_match = need_confirm

        # ── 拒绝答题检查 ──────────────────────────────────
        refusal_match = None
        if case.expect_refusal:
            refusal_keywords = ["无法", "不知道", "没有相关", "联系HR", "联系人力资源",
                                "无相关信息", "未找到", "无权", "超出", "不确定",
                                "建议联系", "暂时无法"]
            refusal_match = any(kw in answer for kw in refusal_keywords)

        # ── 综合判断 ──────────────────────────────────────
        # weak_tool_check：只检查工具是否被调用，不强校验答案内容
        tool_ok = (tool_match is None) or tool_match
        answer_ok = answer_match or case.weak_tool_check
        confirm_ok = (confirmation_match is None) or confirmation_match
        refusal_ok = (refusal_match is None) or refusal_match

        success = tool_ok and answer_ok and confirm_ok and refusal_ok

        if verbose:
            print(f"\n  📥 问题: {case.user_input[:60]}")
            print(f"  📤 答案: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"  🔧 调用工具: {called_tools}")
            print(f"  ✅ need_confirmation: {need_confirm}")
            print(f"  ⏱  延迟: {latency_ms:.0f}ms")

        return EvalResult(
            case_name=case.name,
            category=case.category,
            success=success,
            tool_match=tool_match,
            answer_match=answer_ok,
            confirmation_match=confirmation_match,
            refusal_match=refusal_match,
            latency_ms=latency_ms,
            answer=answer,
            called_tools=called_tools,
            need_confirmation=need_confirm,
        )

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return EvalResult(
            case_name=case.name,
            category=case.category,
            success=False,
            tool_match=None,
            answer_match=False,
            confirmation_match=None,
            refusal_match=None,
            latency_ms=latency_ms,
            answer="",
            error=str(e),
        )


def run_eval(agent_type: str, api_url: str, api_key: str, verbose: bool = False):
    print(f"\n{'═'*65}")
    print(f"  企业 HR Agent 评测报告")
    print(f"  Agent 类型: {agent_type} | 测试集: {len(EVAL_CASES)} cases")
    print(f"{'═'*65}")

    results: List[EvalResult] = []
    session_id = f"eval_{agent_type}_{int(time.time())}"

    for i, case in enumerate(EVAL_CASES, 1):
        label = f"[{case.category}] {case.name}"
        print(f"\n{i:02d}/{len(EVAL_CASES)}  {label}", end="", flush=True)
        result = run_single_eval(case, agent_type, api_url, api_key, session_id, verbose)
        results.append(result)

        icon = "✅" if result.success else "❌"
        print(f"  {icon}  {result.latency_ms:.0f}ms", end="")
        if result.error:
            print(f"  ⚠ {result.error[:60]}", end="")

    # ── 汇总统计 ────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for r in results if r.success)
    latencies = [r.latency_ms for r in results if r.error is None]

    # 工具调用准确率（只统计有 expected_tool 的 case）
    tool_cases = [r for r in results if r.tool_match is not None]
    tool_acc = (sum(1 for r in tool_cases if r.tool_match) / len(tool_cases)
                if tool_cases else None)

    # 确认标志准确率（只统计工单 case）
    confirm_cases = [r for r in results if r.confirmation_match is not None]
    confirm_acc = (sum(1 for r in confirm_cases if r.confirmation_match) / len(confirm_cases)
                   if confirm_cases else None)

    # 拒绝答题准确率（只统计 H 类 case）
    refusal_cases = [r for r in results if r.refusal_match is not None]
    refusal_acc = (sum(1 for r in refusal_cases if r.refusal_match) / len(refusal_cases)
                   if refusal_cases else None)

    # 按类别统计
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        cat_stats[r.category]["total"] += 1
        if r.success:
            cat_stats[r.category]["passed"] += 1

    print(f"\n\n{'═'*65}")
    print(f"  📊 汇总指标")
    print(f"{'═'*65}")
    print(f"  总体通过率:        {passed}/{total} = {passed/total*100:.1f}%")
    if tool_acc is not None:
        print(f"  工具调用准确率:    {tool_acc*100:.1f}%  ({len(tool_cases)} cases)")
    if confirm_acc is not None:
        print(f"  确认标志准确率:    {confirm_acc*100:.1f}%  ({len(confirm_cases)} cases)")
    if refusal_acc is not None:
        print(f"  拒绝答题准确率:    {refusal_acc*100:.1f}%  ({len(refusal_cases)} cases)")
    if latencies:
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        print(f"  延迟 P50:          {statistics.median(latencies):.0f}ms")
        print(f"  延迟 P95:          {sorted_lat[p95_idx]:.0f}ms")
        print(f"  延迟 Max:          {max(latencies):.0f}ms")

    print(f"\n  按类别:")
    category_desc = {
        "A": "普通问答（不调工具）",
        "B": "纯计算（calculator）",
        "C": "网络搜索（web_search）",
        "D": "HR/制度（RAG）",
        "E": "工单创建（HITL）",
        "F": "多步骤任务",
        "G": "反例（不该调工具）",
        "H": "无依据拒答",
    }
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        desc = category_desc.get(cat, cat)
        print(f"    {cat}. {desc:<25} {s['passed']}/{s['total']}")

    # 失败明细
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n  ❌ 失败 Case ({len(failed)} 个):")
        for r in failed:
            print(f"    [{r.category}] {r.case_name}")
            print(f"       tool_match={r.tool_match}  "
                  f"answer_match={r.answer_match}  "
                  f"confirm={r.confirmation_match}  "
                  f"refusal={r.refusal_match}")
            if r.error:
                print(f"       error: {r.error}")
            elif r.answer:
                print(f"       answer: {r.answer[:80]}...")

    # JSON 报告
    report = {
        "agent_type": agent_type,
        "total": total,
        "passed": passed,
        "overall_accuracy": passed / total,
        "tool_accuracy": tool_acc,
        "confirmation_accuracy": confirm_acc,
        "refusal_accuracy": refusal_acc,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "latency_max_ms": max(latencies) if latencies else 0,
        "results": [
            {
                "case":              r.case_name,
                "category":         r.category,
                "success":          r.success,
                "tool_match":       r.tool_match,
                "confirmation":     r.confirmation_match,
                "refusal":          r.refusal_match,
                "latency_ms":       r.latency_ms,
                "called_tools":     r.called_tools,
                "need_confirmation": r.need_confirmation,
            }
            for r in results
        ],
    }

    report_file = f"eval_report_{agent_type}_{int(time.time())}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 详细报告: {report_file}")
    print(f"{'═'*65}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="企业 HR Agent 评测脚本")
    parser.add_argument("--agent", default="function_calling",
                        choices=["function_calling", "langgraph", "multi_agent", "mcp", "auto"],
                        help="评测的 Agent 类型")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--key", default="dev-key-123", help="X-API-Key")
    parser.add_argument("--verbose", action="store_true", help="打印每条 case 的详细输出")
    parser.add_argument("--category", default=None,
                        help="只跑指定类别，如 --category D（只跑 HR/RAG 测试）")
    args = parser.parse_args()

    # 按类别筛选
    if args.category:
        filtered = [c for c in EVAL_CASES if c.category == args.category.upper()]
        if not filtered:
            print(f"❌ 未找到类别 '{args.category}'，可选：A B C D E F G H")
            exit(1)
        # 临时替换全局列表
        import scripts.eval_agent as _self
        _self.EVAL_CASES = filtered

    run_eval(args.agent, args.url, args.key, args.verbose)