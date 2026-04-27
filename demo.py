import sys
import os

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. 导入 Agent
from app.agents.langgraph_agent import LangGraphAgent

# 2. 初始化 (随便起个 ID)
print("正在初始化...")
agent = LangGraphAgent(session_id="debug_001")

# 3. 发消息测试
print("\n发送消息中...")
user_input = "你好"
response = agent.chat(user_input)

# 4. 打印结果
print(f"\n[用户]: {user_input}")
print(f"[Agent]: {response}")
print("\nDebug 结束。")