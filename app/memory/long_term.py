# app/memory/long_term.py
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ESLongTermMemory:
    """基于 Elasticsearch + BGE 向量模型的长期记忆中心"""

    def __init__(self):
        logger.info("正在加载 BGE 向量模型，这可能需要一点时间...")
        # 替换为你本地的 BGE 模型路径，如 "BAAI/bge-small-zh-v1.5"
        self.model = SentenceTransformer("D:/ai_models/bge-small-zh-v1.5")

        # 连接 ES（如果你的 ES 有密码，请加上 basic_auth=("elastic", "password")）
        # 这里假设你在 config.yaml 或者 .env 里配置了 ES_HOST，如果没有配置，默认用 localhost
        es_host = getattr(settings, 'es_host', 'http://localhost:9200')
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
                            "index": True,  # <--- 就加这一行！！！
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
        # 刷新索引使其立即可查
        self.es.indices.refresh(index=self.index_name)
        logger.debug(f"[{session_id}] 💾 长期记忆已保存至 ES: {content[:30]}...")

    def recall_memory(self, session_id: str, query: str, top_k: int = 2) -> str:
        """从 ES 唤醒相关记忆 (企业级：BM25 + KNN 双路混合召回)"""
        query_vector = self.model.encode(query).tolist()

        # 混合检索 (Hybrid Search)
        search_query = {
            "size": top_k,
            # 第一路：传统的 BM25 关键词匹配
            "query": {
                "bool": {
                    "must":[
                        {
                            "match": {
                                "content": query
                            }
                        }
                    ],
                    "filter":[
                        {"term": {"session_id": session_id}}
                    ]
                }
            },
            # 第二路：BGE 向量语义匹配
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 10,
                "filter": {
                    "term": {"session_id": session_id}
                },
                "boost": 0.8  # 权重微调：让向量的得分稍微主导一点
            },
            "_source": ["content"]
        }

        try:
            results = self.es.search(index=self.index_name, body=search_query)
            hits = results["hits"]["hits"]
            if not hits:
                return ""

            recalled_texts = [hit["_source"]["content"] for hit in hits]

            # 【高级调试日志】打印一下双路召回的具体得分，方便你以后调参
            for i, hit in enumerate(hits):
                logger.debug(f"[{session_id}] 召回 Top{i+1} (得分 {hit['_score']:.2f}): {hit['_source']['content'][:20]}...")

            logger.info(f"[{session_id}] 🧠 从 ES 双路唤醒了 {len(recalled_texts)} 条长期记忆！")
            return "\n".join(recalled_texts)
        except Exception as e:
            logger.error(f"ES 双路记忆唤醒失败: {e}")
            return ""


# 全局单例，防止模型重复加载撑爆显存
es_memory_db = ESLongTermMemory()