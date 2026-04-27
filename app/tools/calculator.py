# app/tools/calculator.py
"""
计算工具

★ [核心修改] 将 eval() + 字符白名单过滤 改为 AST 安全解析
原因：
1. 原来的 allowed_chars 白名单过滤逻辑有漏洞：
   - set('0123456789+-*/().,% ^**') 确实能过滤大多数恶意字符
   - 但 eval() 本身的风险在于：一旦白名单漏掉某个字符，就完全暴露
   - 例如：如果将来有人绕过白名单，eval() 可以执行任意 Python 代码
2. 正确做法：用 ast.parse() 做语法树级别的安全检查，
   只允许「数字字面量 + 安全运算符」两类 AST 节点，从根本上杜绝代码注入
3. 这个知识点在面试中经常被问到："你的计算工具安全吗？eval 有什么风险？"
"""
import ast
import operator as op
from langchain_core.tools import tool
from app.core.logger import get_logger

logger = get_logger(__name__)

# ★ [新增] 白名单：只允许这些 AST 节点类型
# BinOp: 二元运算 (a + b)
# UnaryOp: 一元运算 (-a)
# Constant: 数字字面量 (42, 3.14)
# Expression: 表达式根节点
_SAFE_AST_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call)

# ★ [新增] 白名单：只允许这些运算符
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,   # 负号
    ast.UAdd: op.pos,   # 正号
}


def _safe_eval(node):
    """
    递归安全求值，只处理白名单 AST 节点
    ★ 这是面试亮点：展示你理解 AST 和代码安全的关系
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"不允许的一元运算符: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)

    else:
        raise ValueError(f"不允许的表达式类型: {type(node).__name__}。只支持纯数学运算。")


@tool
def calculator(expression: str) -> str:
    """
    安全执行数学计算，支持加减乘除、幂运算、取模等。
    当需要计算具体的数值时，使用此工具。
    示例输入: "100 * 0.15" 或 "2 ** 10" 或 "(3 + 4) * 2"
    """
    try:
        # ★ [修改] 预处理：支持 ^ 作为幂运算符（用户友好）
        cleaned = expression.strip().replace('^', '**')

        if not cleaned:
            return "❌ 计算失败：表达式为空"

        # ★ [修改] 第一步：AST 解析（语法检查）
        try:
            tree = ast.parse(cleaned, mode='eval')
        except SyntaxError as e:
            return f"❌ 计算失败：数学表达式语法错误 ({e})，请检查括号和运算符是否正确"

        # ★ [修改] 第二步：节点白名单检查（安全检查）
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_AST_NODES):
                return f"❌ 安全拒绝：表达式包含不允许的操作 ({type(node).__name__})，只支持纯数学运算"

        # ★ [修改] 第三步：安全求值（不使用 eval）
        result = _safe_eval(tree)

        # 格式化输出：整数不显示小数点
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        logger.debug(f"安全计算: {expression} = {result}")
        return str(result)

    except ZeroDivisionError:
        return "❌ 计算失败：除数不能为 0"
    except OverflowError:
        return "❌ 计算失败：结果超出数值范围"
    except ValueError as e:
        return f"❌ 计算失败：{str(e)}"
    except Exception as e:
        logger.error(f"计算工具异常: {e}")
        return f"❌ 计算失败：请检查数学表达式是否合法。错误信息: {str(e)}"
