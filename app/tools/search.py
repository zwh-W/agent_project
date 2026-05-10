# app/tools/search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.tools import tool
from duckduckgo_search import DDGS

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchItem:
    title: str
    url: str
    snippet: str
    query: str


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def _build_fallback_queries(query: str) -> list[str]:
    """
    根据用户原始 query 自动生成 fallback 查询。
    目标：
    1. 去掉过窄年份
    2. 中英文互补
    3. 拆分复杂比较任务
    4. 针对薪资、云价格、汇率等常见场景加专用关键词
    """
    q = query.strip()
    q_lower = q.lower()

    queries: list[str] = [q]

    # 汇率类：普通搜索很容易空，增加中英文和权威来源关键词
    if any(k in q_lower for k in ["usd", "cny", "exchange rate"]) or any(
        k in q for k in ["汇率", "美元", "人民币"]
    ):
        queries += [
            "美元兑人民币 汇率 今日",
            "USD CNY exchange rate today",
            "美元 人民币 汇率 中国银行",
            "美元人民币汇率 中国外汇交易中心",
            "USD to CNY today",
        ]

    # Python / Java 薪资比较：拆开搜，不要只搜“对比”
    if (
        ("python" in q_lower or "Python" in q)
        and ("java" in q_lower or "Java" in q)
        and any(k in q for k in ["薪资", "工资", "平均薪资", "salary", "就业"])
    ):
        queries += [
            "Python 开发工程师 平均薪资",
            "Java 开发工程师 平均薪资",
            "BOSS直聘 Python 工程师 薪资",
            "BOSS直聘 Java 工程师 薪资",
            "猎聘 IT 行业 薪酬报告 Python Java",
            "智联招聘 Python Java 工程师 薪资",
            "Python developer salary China",
            "Java developer salary China",
        ]

    # 云服务价格：拆服务商、拆产品
    if any(k in q for k in ["云服务", "云服务器", "阿里云", "腾讯云", "华为云", "ECS", "CVM"]):
        queries += [
            "阿里云 ECS 价格",
            "腾讯云 CVM 价格",
            "华为云 ECS 价格",
            "阿里云 腾讯云 华为云 云服务器 价格 对比",
            "阿里云 ECS 价格计算器",
            "腾讯云 CVM 价格计算器",
            "华为云 弹性云服务器 价格",
        ]

    # 新闻 / 技术趋势：去掉过窄日期
    if any(k in q for k in ["新闻", "最新", "进展", "趋势"]):
        queries += [
            q.replace("2026年5月10日", "").strip(),
            q.replace("2026年5月", "").strip(),
            q.replace("2026", "").strip(),
        ]

    # 通用：去掉年份，避免“2026”导致搜索引擎结果为空
    for year in ["2026年", "2026", "2025年", "2025"]:
        if year in q:
            queries.append(q.replace(year, "").strip())

    return _unique_keep_order(queries)[:6]


def _search_once(query: str, max_results: int) -> list[SearchItem]:
    items: list[SearchItem] = []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        for res in results:
            title = res.get("title") or ""
            url = res.get("href") or res.get("url") or ""
            snippet = res.get("body") or ""

            if not title and not snippet:
                continue

            items.append(
                SearchItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    query=query,
                )
            )

    except Exception as e:
        logger.warning(f"搜索 query 失败: {query}, error={e}")

    return items


def _format_success(
    original_query: str,
    queries_tried: list[str],
    results: list[SearchItem],
) -> str:
    lines: list[str] = [
        "【搜索状态】",
        "success=true",
        "provider=duckduckgo",
        f"original_query={original_query}",
        f"queries_tried={queries_tried}",
        "",
        "【高相关搜索结果】",
    ]

    for i, item in enumerate(results, start=1):
        lines.extend(
            [
                f"{i}. 标题：{item.title}",
                f"   命中查询：{item.query}",
                f"   URL：{item.url or 'N/A'}",
                f"   摘要：{item.snippet}",
                "",
            ]
        )

    lines.extend(
        [
            "【回答要求】",
            "请只基于以上搜索结果回答。",
            "如果搜索结果不足以完成计算、比较或调研，请明确说明缺口，不要编造。",
            "如果需要比较两个对象，应尽量分别总结每个对象的可用证据，再给出谨慎结论。",
        ]
    )

    return "\n".join(lines)


def _format_no_results(original_query: str, queries_tried: list[str]) -> str:
    return "\n".join(
        [
            "【搜索状态】",
            "success=false",
            "provider=duckduckgo",
            f"original_query={original_query}",
            f"queries_tried={queries_tried}",
            "",
            "【搜索结果】",
            "未找到可靠网页结果。",
            "",
            "【回答要求】",
            "请明确说明当前搜索工具未获得有效结果。",
            "不要编造事实、价格、薪资、汇率、新闻或市场数据。",
            "可以说明已经尝试了哪些查询，并建议用户提供更具体来源或数据。",
        ]
    )


@tool
def web_search(query: str) -> str:
    """
    搜索互联网上的实时信息、新闻、市场数据和未知知识。

    适用场景：
    - 当前新闻、最新技术进展、股价、汇率、薪资、云服务价格等动态信息
    - 需要外部公开网页证据的问题
    - 复杂调研任务中的资料收集步骤

    工具策略：
    - 内部会自动进行 query rewrite 和 fallback 查询
    - 如果搜索不到可靠结果，会明确返回 success=false
    - 调用者不得在无搜索证据时编造事实
    """
    logger.info(f"执行网络搜索: {query}")

    max_results = getattr(settings.tools, "search_max_results", 5)
    queries = _build_fallback_queries(query)

    collected: list[SearchItem] = []
    seen_urls: set[str] = set()
    seen_text: set[str] = set()

    for q in queries:
        results = _search_once(q, max_results=max_results)

        for item in results:
            dedupe_key = item.url or f"{item.title}-{item.snippet[:80]}"
            if dedupe_key in seen_urls or dedupe_key in seen_text:
                continue

            if item.url:
                seen_urls.add(item.url)
            else:
                seen_text.add(dedupe_key)

            collected.append(item)

        # 有足够结果就提前停止，避免过慢
        if len(collected) >= max_results:
            break

    if not collected:
        return _format_no_results(query, queries)

    return _format_success(query, queries, collected[:max_results])