# test_agent.py
from app.agents.function_calling_agent import FunctionCallingAgent


def main():
    print("================ 启动企业级 Agent 测试 ================")
    session_id = "test_user_001"
    agent = FunctionCallingAgent(session_id=session_id)

    # 考验记忆和计算工具
    print("\n--- 测试开始 ---")
    ans = agent.chat("我叫架构师。帮我算一下 2的10次方减去50是多少？然后用鸭子侦探的语气告诉我结果。")
    print(f"\n最终回复:\n{ans}")


if __name__ == "__main__":
    main()