from app.tools.rag_search import query_policy_knowledge_base

result = query_policy_knowledge_base.invoke({
    "query": "学生有什么权利",
    "knowledge_id": 2
})

print(result)
