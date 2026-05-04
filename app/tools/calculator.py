"""
安全数学计算工具

核心安全设计：
1. 使用 AST 替代 eval()，不执行任意 Python 代码。
2. 只允许数字字面量、括号、加减乘除、幂运算、取模、整除、正负号。
3. 增加表达式长度、AST 节点数、递归深度、数字大小、指数大小、结果大小限制，降低 DoS 风险。
4. 拒绝 bool、str、list、dict、函数调用、属性访问、变量名等非数学表达式。
5. 拒绝 inf / nan，避免返回不可解释结果。
"""

import ast
import math
import operator as op

from langchain_core.tools import tool
from app.core.logger import get_logger

logger = get_logger(__name__)


# =========================
# 安全限制参数
# =========================

# 表达式最大长度，防止超长输入导致解析压力
_MAX_EXPR_LEN = 300

# AST 最大节点数，防止表达式过于复杂
_MAX_AST_NODES = 80

# AST 最大递归深度，防止深层嵌套导致递归溢出
_MAX_DEPTH = 30

# 单个数字字面量的最大绝对值
_MAX_ABS_LITERAL = 10**12

# 最终或中间计算结果的最大绝对值
_MAX_ABS_RESULT = 10**100

# 幂运算指数最大绝对值
_MAX_ABS_EXPONENT = 100

# 估算结果最大十进制位数，主要用于提前拦截大整数幂运算
_MAX_RESULT_DIGITS = 100


# 运算符白名单：只允许这些数学运算符
_ALLOWED_OPERATORS = {
    # 二元运算符
    ast.Add: op.add,             # +
    ast.Sub: op.sub,             # -
    ast.Mult: op.mul,            # *
    ast.Div: op.truediv,         # /
    ast.Pow: op.pow,             # **
    ast.Mod: op.mod,             # %
    ast.FloorDiv: op.floordiv,   # //

    # 一元运算符
    ast.USub: op.neg,            # -x
    ast.UAdd: op.pos,            # +x
}


Number = int | float


def _normalize_expression(expression: str) -> str:
    """
    表达式预处理。

    设计说明：
    - strip() 去除首尾空白。
    - 将 ^ 视为幂运算符，转换为 Python AST 支持的 **。
    """
    if not isinstance(expression, str):
        raise ValueError("表达式必须是字符串")

    cleaned_expr = expression.strip().replace("^", "**")

    if not cleaned_expr:
        raise ValueError("表达式不能为空，请输入合法的数学表达式")

    if len(cleaned_expr) > _MAX_EXPR_LEN:
        raise ValueError(f"表达式过长，最大允许长度为 {_MAX_EXPR_LEN} 个字符")

    return cleaned_expr


def _validate_ast_size(ast_tree: ast.AST) -> None:
    """
    校验 AST 节点总数，防止超复杂表达式造成资源消耗。
    """
    node_count = 0

    for _ in ast.walk(ast_tree):
        node_count += 1
        if node_count > _MAX_AST_NODES:
            raise ValueError(f"表达式过于复杂，最多允许 {_MAX_AST_NODES} 个语法节点")


def _validate_number(value: object, *, is_literal: bool) -> Number:
    """
    校验数字值是否合法。

    注意：
    - bool 是 int 的子类，所以必须显式拒绝。
    - 拒绝 inf / nan。
    - 限制字面量和计算结果大小。
    """
    if isinstance(value, bool):
        raise ValueError("不支持布尔值，仅支持数字")

    if not isinstance(value, (int, float)):
        raise ValueError(f"不支持的数值类型：{type(value).__name__}")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("不支持无穷大或 NaN")

    limit = _MAX_ABS_LITERAL if is_literal else _MAX_ABS_RESULT

    if abs(value) > limit:
        if is_literal:
            raise ValueError(f"数字字面量过大，最大允许绝对值为 {limit}")
        raise ValueError(f"计算结果过大，最大允许绝对值为 {limit}")

    return value


def _estimate_pow_result_digits(base: Number, exponent: int) -> None:
    """
    对幂运算做提前规模估算，避免计算巨大整数后才发现结果过大。

    示例需要拦截：
    - 2 ** 100000000
    - 999999999999 ** 100
    - 0.0001 ** -100
    """
    if isinstance(exponent, bool):
        raise ValueError("指数不能是布尔值")

    if abs(exponent) > _MAX_ABS_EXPONENT:
        raise ValueError(f"指数过大，最大允许绝对值为 {_MAX_ABS_EXPONENT}")

    # 0 ** negative 会由 Python 抛出 ZeroDivisionError，这里不提前吞掉
    if base == 0:
        return

    abs_base = abs(base)

    # 1、-1 的幂不会变大
    if abs_base == 1:
        return

    try:
        # 估算 log10(abs(base ** exponent)) = exponent * log10(abs(base))
        estimated_log10 = exponent * math.log10(abs_base)
    except ValueError:
        raise ValueError("幂运算底数不合法")

    # estimated_log10 > 0 代表结果绝对值大于 1
    if estimated_log10 > _MAX_RESULT_DIGITS:
        raise ValueError(f"幂运算结果过大，最大允许约 {_MAX_RESULT_DIGITS} 位十进制数字")


def _safe_eval(node: ast.AST, depth: int = 0) -> Number:
    """
    递归安全求值核心函数。

    只处理白名单 AST 节点：
    - ast.Expression
    - ast.Constant，且只能是 int / float
    - ast.BinOp，且运算符必须在白名单内
    - ast.UnaryOp，且运算符必须在白名单内

    其他节点一律拒绝，例如：
    - ast.Call：函数调用
    - ast.Name：变量名
    - ast.Attribute：属性访问
    - ast.Subscript：下标访问
    - ast.List / ast.Dict / ast.Tuple：容器
    - ast.Compare：比较表达式
    """
    if depth > _MAX_DEPTH:
        raise ValueError(f"表达式嵌套过深，最大允许深度为 {_MAX_DEPTH}")

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        return _validate_number(node.value, is_literal=True)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的运算符：{op_type.__name__}")

        left_val = _safe_eval(node.left, depth + 1)
        right_val = _safe_eval(node.right, depth + 1)

        # 幂运算单独加固
        if op_type is ast.Pow:
            if isinstance(right_val, bool):
                raise ValueError("指数不能是布尔值")

            if not isinstance(right_val, int):
                raise ValueError("指数必须是整数")

            _estimate_pow_result_digits(left_val, right_val)

        result = _ALLOWED_OPERATORS[op_type](left_val, right_val)
        return _validate_number(result, is_literal=False)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的一元运算符：{op_type.__name__}")

        operand_val = _safe_eval(node.operand, depth + 1)
        result = _ALLOWED_OPERATORS[op_type](operand_val)

        return _validate_number(result, is_literal=False)

    raise ValueError(f"不允许的表达式语法：{type(node).__name__}，仅支持纯数学运算")


def _format_result(result: Number) -> str:
    """
    格式化计算结果。

    - 10.0 返回 10
    - 普通 int / float 直接返回字符串
    """
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return str(result)


@tool
def calculator(expression: str) -> str:
    """
    安全执行数学计算，支持加减乘除、幂运算、取模、整除、正负号。
    当用户需要计算具体数值结果时，必须使用此工具。

    支持示例：
    - 基础运算："123 * 456"
    - 带括号："(3 + 4) * 2 / 5"
    - 幂运算："2 ** 10" 或 "2 ^ 10"
    - 取模/整除："10 % 3" 或 "10 // 3"
    - 负数运算："-10 + 20"

    不支持示例：
    - 函数调用："sum([1, 2, 3])"
    - 变量访问："x + 1"
    - 属性访问："().__class__"
    - 字符串："'1' + '2'"
    - 布尔值："True + 1"
    - 超大幂："2 ** 100000000"
    """
    try:
        # 1. 表达式预处理与长度校验
        cleaned_expr = _normalize_expression(expression)

        # 2. AST 语法解析
        try:
            ast_tree = ast.parse(cleaned_expr, mode="eval")
        except SyntaxError as e:
            return f"❌ 计算失败：表达式语法错误（{e.msg}），请检查括号、运算符是否配对正确"

        # 3. AST 复杂度校验
        _validate_ast_size(ast_tree)

        # 4. 安全递归求值
        result = _safe_eval(ast_tree)

        # 5. 结果格式化
        formatted_result = _format_result(result)

        logger.debug(
            f"安全计算执行成功 | 原始表达式：{expression} | 清洗后表达式：{cleaned_expr} | 结果：{formatted_result}"
        )

        return formatted_result

    except ZeroDivisionError:
        return "❌ 计算失败：除数不能为 0，请修改表达式后重试"

    except OverflowError:
        return "❌ 计算失败：结果数值超出范围，无法计算"

    except RecursionError:
        return "❌ 计算失败：表达式嵌套过深，请简化后重试"

    except ValueError as e:
        return f"❌ 计算失败：{str(e)}"

    except Exception as e:
        # 生产环境不建议把未知异常详情直接返回给用户，避免泄露内部实现细节
        logger.error(
            f"计算工具异常 | 表达式：{expression} | 错误：{str(e)}",
            exc_info=True,
        )
        return "❌ 计算失败：表达式不合法，请检查是否为纯数学运算"