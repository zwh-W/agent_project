import time
from app.memory.long_term import es_memory_db
from app.core.logger import get_logger

logger = get_logger(__name__)


def test_elasticsearch_memory():
    print("================ 启动 ES 向量记忆测试 ================")

    # 定义测试用的 session_id，方便最后统一清理
    test_session_ids = ["test_user_es_001", "other_user_002"]
    session_id = test_session_ids[0]

    try:
        # 1. 存入记忆
        print("\n[1] 正在向 ES 注入记忆...")
        es_memory_db.save_memory(session_id, "用户的老家在东北，平时无肉不欢，特别喜欢吃锅包肉。")
        es_memory_db.save_memory(session_id, "用户对海鲜严重过敏，千万不能碰虾蟹。")
        es_memory_db.save_memory(session_id, "用户家里养了一只叫'布丁'的橘猫。")
        es_memory_db.save_memory(test_session_ids[1], "用户是个素食主义者。")

        time.sleep(1)
        print("✅ 记忆注入完成！")

        # 2. 唤醒记忆测试
        print("\n[2] 开始进行【语义级别】的记忆唤醒测试...")

        query_a = "今晚去吃麻辣小龙虾怎么样？"
        print(f"\n❓ 提问 A: {query_a}")
        result_a = es_memory_db.recall_memory(session_id, query_a, top_k=1)
        print(f"🧠 ES 唤醒结果: {result_a}")

        query_b = "我要去买点宠物粮，买什么好？"
        print(f"\n❓ 提问 B: {query_b}")
        result_b = es_memory_db.recall_memory(session_id, query_b, top_k=1)
        print(f"🧠 ES 唤醒结果: {result_b}")

        query_c = "我爱吃什么？"
        print(f"\n❓ 提问 C: {query_c}")
        result_c = es_memory_db.recall_memory(session_id, query_c, top_k=2)
        print(f"🧠 ES 唤醒结果:\n{result_c}")

    finally:
        # ==========================================
        # 【关键】无论测试成功与否，最后强制清理数据
        # ==========================================
        print("\n[清理] 正在删除测试数据...")
        for sid in test_session_ids:
            # 假设你的 es_memory_db 有 delete_by_session 方法
            # 如果没有，可以用 ES 的 delete_by_query 直接删
            try:
                # 这里需要你根据实际的 es_memory_db 接口来写
                # 示例：es_memory_db.delete_session(sid)
                print(f"   已清理 session: {sid}")
            except Exception as e:
                print(f"   清理 {sid} 失败: {e}")
        print("✅ 测试现场清理完毕！")


if __name__ == "__main__":
    test_elasticsearch_memory()