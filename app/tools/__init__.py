# app/tools/__init__.py

from app.tools.calculator import calculator
from app.tools.search import web_search

# 将所有写好的工具放到这个列表里，暴露给外部统一调用
ALL_TOOLS = [
    calculator,
    web_search
]