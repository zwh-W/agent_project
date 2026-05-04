"""
极简 Debug 脚本：直接调用你现有的 FunctionCallingAgent
"""
import sys
import os

# 确保能找到 app 模块（根据你的项目结构调整）


from app.agents.function_calling_agent import FunctionCallingAgent

if __name__ == "__main__":
    print("="*70)
    print("🚀 开始 Debug FunctionCallingAgent")
    print("="*70)

    # 1. 初始化 Agent（随便填一个 session_id）
    agent = FunctionCallingAgent(session_id="debug_001")

    # 2. 你的测试问题
    # 可以改成你想测试的任何问题
    user_input = "123乘以456等于多少"

    print(f"\n💬 用户输入: {user_input}")
    print("-"*70)

    # 3. 调用（推荐用 chat_with_trace，能看到完整推理链）
    final_answer, trace = agent.chat_with_trace(user_input)

    print("-"*70)
    print(f"\n🎯 最终答案: {final_answer}")

    # 4. 如果你想看结构化的 JSON Trace，取消下面的注释
    # import json
    # print(f"\n📊 结构化 Trace:")
    # print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))