# app/tools/search.py
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str) -> str:
    """
    搜索互联网上的实时信息、新闻和未知知识。
    当你不知道当前事实，或需要获取最新数据时，使用此工具。
    """
    try:
        logger.info(f"执行网络搜索: {query}")
        with DDGS() as ddgs:
            # 读取 config.yaml 中的 search_max_results
            results = list(ddgs.text(query, max_results=settings.tools.search_max_results))

        if not results:
            return f"未找到关于 '{query}' 的相关网页信息。"

        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"[{i + 1}] 标题：{res.get('title')}\n内容摘要：{res.get('body')}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"搜索工具异常: {e}")
        return f"❌ 搜索遇到网络或解析异常: {str(e)}"