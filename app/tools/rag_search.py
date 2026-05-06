# app/tools/rag_search.py
"""
RAG 知识库查询工具

让 Agent 调用外部 RAG 服务，查询企业制度、HR 政策、员工手册等知识库内容。
注意：工具只负责返回证据上下文，最终答案由 Agent 基于证据生成。
"""

import requests
from typing import Optional
from langchain_core.tools import tool

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_RAG_TIMEOUT = 15


def _call_rag_api(query: str, knowledge_id: int) -> dict:
    """
    调用外部 RAG 项目的 /chat 接口。

    RAG 项目接口格式：
    POST /chat
    {
        "knowledge_id": 1,
        "messages": [
            {"role": "user", "content": "..."}
        ]
    }
    """
    rag_url = settings.tools.rag_api_url

    if not rag_url:
        return {
            "success": False,
            "error": "rag_api_url 未配置，请在 Agent 项目的 config.yaml 中配置 tools.rag_api_url"
        }

    payload = {
        "knowledge_id": knowledge_id,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }

    try:
        resp = requests.post(
            rag_url,
            json=payload,
            timeout=_RAG_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        # 你的 RAG 项目标准响应格式：
        # {
        #   "response_code": 200,
        #   "response_msg": "...",
        #   "answer": "...",
        #   "sources": [...],
        #   "messages": [...]
        # }
        if "answer" in data or "sources" in data:
            return {
                "success": True,
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "raw_response": data,
            }

        return {
            "success": False,
            "error": f"RAG 响应格式无法识别: {str(data)[:300]}"
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"无法连接 RAG 服务：{rag_url}。请确认 RAG 项目已启动"
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"RAG 服务请求超时，超过 {_RAG_TIMEOUT} 秒"
        }

    except requests.exceptions.HTTPError as e:
        try:
            error_detail = resp.text[:300]
        except Exception:
            error_detail = str(e)

        return {
            "success": False,
            "error": f"RAG 服务返回 HTTP 错误: {e}，响应内容: {error_detail}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"RAG 调用异常: {str(e)}"
        }


@tool
def query_policy_knowledge_base(query: str, knowledge_id: Optional[int] = None) -> str:
    """
    查询企业知识库、HR 政策、员工手册、制度文档。

    当用户询问以下问题时，必须使用此工具：
    - 年假、病假、事假、产假、婚假、丧假等休假制度
    - 报销、差旅、餐补、交通补贴、住宿标准等政策
    - 绩效、KPI、晋升、调薪、绩效面谈等制度
    - 入职、离职、转正、试用期等流程
    - 考勤、打卡、迟到、早退、加班等规定
    - 五险一金、社保、公积金等福利政策
    - 公司规定、员工手册、制度、政策类问题

    工具只返回知识库证据，最终答案由 Agent 基于证据生成。

    Args:
        query: 用户问题
        knowledge_id: 知识库 ID，不传时使用默认配置
    """
    kid = knowledge_id if knowledge_id is not None else settings.tools.rag_knowledge_id

    logger.info(f"RAG 工具调用 | knowledge_id={kid} | query={query[:80]}")

    result = _call_rag_api(query=query, knowledge_id=kid)

    if not result["success"]:
        error_msg = result["error"]
        logger.warning(f"RAG 查询失败: {error_msg}")

        return (
            f"RAG 查询失败：{error_msg}。\n"
            f"请检查 RAG 服务是否启动、rag_api_url 是否配置为 RAG 项目的 /chat 接口。\n"
            f"如果知识库暂不可用，请告知用户当前无法查询制度依据，不要凭空编造。"
        )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    output_lines = []

    if answer:
        output_lines.append(f"【知识库答案摘要】\n{answer}")

    if sources:
        output_lines.append("\n【检索到的原文证据】")

        for i, src in enumerate(sources, 1):
            document_id = src.get("document_id", "")
            document_name = src.get("document_name", "未知文档")
            page_number = src.get("page_number", "")
            chunk_content = src.get("chunk_content", "")

            header = f"来源 {i}：《{document_name}》"
            if document_id != "":
                header += f"（document_id={document_id}）"
            if page_number != "":
                header += f" 第 {page_number} 页"

            output_lines.append(header)

            if chunk_content:
                preview = (
                    chunk_content[:500] + "..."
                    if len(chunk_content) > 500
                    else chunk_content
                )
                output_lines.append(f"原文片段：{preview}")

    if not output_lines:
        return (
            "知识库中未找到与该问题相关的内容。"
            "请提示用户补充问题或联系 HR 获取准确信息，不要凭空编造。"
        )

    logger.info(f"RAG 查询成功 | sources={len(sources)}")
    return "\n".join(output_lines)