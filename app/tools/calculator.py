# app/tools/calculator.py
from langchain_core.tools import tool
from app.core.logger import get_logger

logger = get_logger(__name__)


@tool
def calculator(expression: str) -> str:
    """
    执行数学计算，支持加减乘除、幂运算等。
    当需要计算具体的数值时，使用此工具。
    示例输入: "100 * 0.15" 或 "2 ** 10"
    """
    try:
        # 安全白名单过滤，防止大模型注入恶意 Python 代码（如 os.system('rm -rf /')）
        allowed_chars = set('0123456789+-*/().,% ^**')
        safe_expr = expression.replace('^', '**')

        if not all(c in allowed_chars or c.isspace() for c in safe_expr):
            return "❌ 计算失败：表达式包含非法字符，请只使用纯数字和数学符号。"

        result = eval(safe_expr)
        logger.debug(f"执行计算: {expression} = {result}")
        return str(result)

    except ZeroDivisionError:
        return "❌ 计算失败：除数不能为 0"
    except Exception as e:
        logger.error(f"计算工具异常: {e}")
        return f"❌ 计算失败：请检查数学表达式是否合法。错误信息: {str(e)}"