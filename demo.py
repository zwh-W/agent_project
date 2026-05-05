# debug_prompt_manager.py
from app.core.prompt_manager import prompt_manager

print("Prompt 目录已初始化")
print(prompt_manager.list_versions("system"))
print(prompt_manager.list_versions("supervisor"))
print(prompt_manager.list_versions("router"))