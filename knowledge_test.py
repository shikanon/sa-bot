import os
import logging
import time
from elasticsearch7 import Elasticsearch
import knowledge


logging.basicConfig(level=logging.DEBUG)

es_url = os.environ.get("ES_URL")
print("es url:",es_url)
es_table = "knowledge"
# es = Elasticsearch(
#             hosts=[es_url],
#             verify_certs=False, 
#         )
# if es.indices.exists(es_table):
#     es.indices.delete(es_table)
emb = knowledge.MaaSKnowledgeEmbedding(model="bge-large-zh", model_version="1.0")
db = knowledge.ESKnnVectorDB(es_url, es_table, emb)
db.delete()
time.sleep(2)

with open("Readme.md", "r", encoding="utf-8") as fr:
    doc = fr.read()
kg_unit = doc.split("/n")
print(len(kg_unit))
db.bulk_insert(kg_unit)
time.sleep(2)

value = db.query("如何使用 es 作为知识库?")
print("如何使用 es 作为知识库?\n答案是：\n",value)
