# debug_langgraph_agent.py

import traceback
from app.agents.langgraph_agent import LangGraphAgent


def run_case(session_id: str, question: str):
    print("=" * 80)
    print(f"Session: {session_id}")
    print(f"Question: {question}")
    print("=" * 80)

    try:
        agent = LangGraphAgent(session_id=session_id)

        answer, profile = agent.chat_with_profile(question)

        print("\n【最终答案】")
        print(answer)

        print("\n【执行 Profile】")
        print(profile.to_readable())

        print("\n【Profile Dict】")
        print(profile.to_dict())

    except Exception as e:
        print("\n【执行异常】")
        print(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    # 测试 1：普通问题，理论上不应该调用工具
    run_case(
        session_id="debug-no-tool-001",
        question="你好，请简单介绍一下你自己。"
    )

    # 测试 2：需要调用工具的问题
    # 这里的问题要根据你项目里已有工具来改。
    # 如果你的工具里有知识库查询工具，可以用类似问题。
    # run_case(
    #     session_id="debug-tool-001",
    #     question="请查询知识库中关于年假制度的规定，并总结入职满一年有几天年假。"
    # )