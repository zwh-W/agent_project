# app/memory/long_term.py
"""
基于 Elasticsearch + BGE 向量模型的长期记忆中心

★ [核心修改] 将模块级立即初始化改为懒加载单例
原因：原代码最后一行 es_memory_db = ESLongTermMemory() 在 import 时立即执行，
会加载几百 MB 的 BGE 模型并连接 ES。只要 ES 未启动或模型路径错误，
整个 FastAPI 服务就启动崩溃，影响所有 Agent（即使它们根本不用长期记忆）。
"""
import threading
from typing import Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ESLongTermMemory:
    """基于 Elasticsearch + BGE 向量模型的长期记忆中心"""

    def __init__(self):
        logger.info("正在加载 BGE 向量模型，这可能需要一点时间...")
        # ★ [修改] 将模型路径配置化，从 settings 读取，不再硬编码
        # 原因：硬编码的 "D:/ai_models/bge-small-zh-v1.5" 在 Linux/Mac 服务器上必然失败
        from sentence_transformers import SentenceTransformer
        model_path = getattr(settings.memory, 'bge_model_path', 'BAAI/bge-small-zh-v1.5')
        self.model = SentenceTransformer(model_path)

        from elasticsearch import Elasticsearch
        # ★ [修改] ES host 从 settings 读取，不再用 getattr 的临时 workaround
        es_host = getattr(settings.memory, 'es_host', 'http://localhost:9200')
        self.es = Elasticsearch(es_host)

        self.index_name = "agent_long_term_memory"
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self._init_index()

    def _init_index(self):
        """初始化 ES 索引（建表）"""
        if not self.es.indices.exists(index=self.index_name):
            logger.info(f"创建 ES 长期记忆索引: {self.index_name}")
            mapping = {
                "mappings": {
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "content_vector": {
                            "type": "dense_vector",
                            "dims": self.vector_dim,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)

    def save_memory(self, session_id: str, content: str):
        """将重要记忆存入 ES"""
        if not content or not content.strip():
            return

        vector = self.model.encode(content).tolist()
        self.es.index(index=self.index_name, body={
            "session_id": session_id,
            "content": content,
            "content_vector": vector
        })
        self.es.indices.refresh(index=self.index_name)
        logger.debug(f"[{session_id}] 💾 长期记忆已保存至 ES: {content[:30]}...")

    def recall_memory(self, session_id: str, query: str, top_k: int = 2) -> str:
        """从 ES 唤醒相关记忆 (BM25 + KNN 双路混合召回)"""
        query_vector = self.model.encode(query).tolist()

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": [{"term": {"session_id": session_id}}]
                }
            },
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 10,
                "filter": {"term": {"session_id": session_id}},
                "boost": 0.8
            },
            "_source": ["content"]
        }

        try:
            results = self.es.search(index=self.index_name, body=search_query)
            hits = results["hits"]["hits"]
            if not hits:
                return ""

            recalled_texts = [hit["_source"]["content"] for hit in hits]
            for i, hit in enumerate(hits):
                logger.debug(
                    f"[{session_id}] 召回 Top{i+1} (得分 {hit['_score']:.2f}): "
                    f"{hit['_source']['content'][:20]}..."
                )
            logger.info(f"[{session_id}] 🧠 从 ES 双路唤醒了 {len(recalled_texts)} 条长期记忆！")
            return "\n".join(recalled_texts)
        except Exception as e:
            logger.error(f"ES 双路记忆唤醒失败: {e}")
            return ""


# ★ [核心修改] 将全局立即初始化改为懒加载单例
# 原因：
# 1. 原代码 es_memory_db = ESLongTermMemory() 在 import 时立即加载 BGE 模型 + 连接 ES
# 2. ES 未启动 or 模型路径错误 → 整个 FastAPI 服务崩溃，哪怕只用 function_calling agent
# 3. 改为懒加载后：只有第一次真正调用 recall/save 时才初始化，服务启动不受影响
# 4. 加线程锁保证多并发下只初始化一次

_es_memory_db: Optional[ESLongTermMemory] = None
_es_memory_lock = threading.Lock()


def get_es_memory() -> Optional[ESLongTermMemory]:
    """
    获取 ES 长期记忆单例（懒加载）
    如果 ES/BGE 初始化失败，返回 None，让上层降级处理而不是崩溃
    """
    global _es_memory_db
    if _es_memory_db is not None:
        return _es_memory_db

    with _es_memory_lock:
        # double-check locking
        if _es_memory_db is not None:
            return _es_memory_db
        try:
            _es_memory_db = ESLongTermMemory()
            logger.info("✅ ES 长期记忆模块初始化成功")
        except Exception as e:
            logger.error(
                f"❌ ES 长期记忆模块初始化失败，将跳过长期记忆功能: {e}\n"
                f"请检查：1) ES 是否启动  2) BGE 模型路径是否正确"
            )
            return None
    return _es_memory_db


# ★ [兼容旧代码] 保留 es_memory_db 名称，但改为属性访问代理
# manager.py 里 from app.memory.long_term import es_memory_db 的 import 不需要改
# 注意：这是一个函数引用，manager.py 需要同步改为调用 get_es_memory()
es_memory_db = get_es_memory  # 将在 manager.py 中以函数形式调用
