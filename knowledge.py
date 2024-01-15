import hashlib
from typing import List
from volcengine.maas import MaasService, MaasException, ChatRole
from elasticsearch7 import Elasticsearch

import logger

# 知识库中存储文本的字段
KG_FIELD_TEXT = "text"
KG_FIELD_VECTOR = "embedding"

class TextEmbeddingVector:
    def __init__(self, text, vector):
        self._text = text
        self._vector = vector
        self._id = hashlib.md5(text.encode(encoding='UTF-8')).hexdigest()
    
    def get_id(self):
        return self._id
    
    def get_text(self):
        return self._text

    def get_vector(self):
        return self._vector

class MaaSKnowledgeEmbedding:
    def __init__(self, model, model_version) -> None:
        self.maas = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
        self.model = model
        self.model_version = model_version
        self.debug_logger = logger.DebugLogger(self.__class__.__name__)
    
    def encode(self, texts: List[str]) -> List[TextEmbeddingVector]:
        req = {
            "model": {
                "name": self.model,
                "version": self.model_version
            },
            "input": texts
        }
        self.debug_logger.debug(req)
        resp = self.maas.embeddings(req)
        result = []
        for i in range(len(texts)):
            emb = TextEmbeddingVector(texts[i], resp.data[i].embedding)
            result.append(emb)
        return result

class VectorDB:
    def __init__(self, url:str, table:str) -> None:
        pass
    
    def bulk_insert(self, data: List[TextEmbeddingVector]):
        pass

    def query(self):
        pass


class ESKnnVectorDB(VectorDB):
    def __init__(self, url:str, table:str, embedding: MaaSKnowledgeEmbedding) -> None:
        '''url: http://<用户名>:<密码>@<域名/ip地址>:<端口>
        table: vector存储表,

        '''
        self.debug_logger = logger.DebugLogger(self.__class__.__name__)
        self.embedding = embedding
        self.es = Elasticsearch(
            hosts=[url],
            verify_certs=False, 
        )
        self.table = table
        if not self.es.indices.exists(self.table):
            self.es.indices.create(
                index=self.table,
                body={
                    "mappings": {
                        "properties": {
                            KG_FIELD_TEXT: { "type": "text" },
                            KG_FIELD_VECTOR: { "type": "knn_vector", "dimension": 1024 }
                        }
                    },
                    "settings": {
                        "index": {
                            "refresh_interval": "10s",
                            "knn": True,
                            "knn.space_type": "cosinesimil",
                            "number_of_replicas": "1"
                        }
                    }
                }
            )
            self.debug_logger.debug("成功创建index: %s"%self.table)

    def query(self, text: str):
        vectors = self.embedding.encode([text])
        return self.query_vector(vectors[0])

    def bulk_insert(self, data: List[str])->None:
        vectors = self.embedding.encode(texts=data)
        self.bulk_insert_vector(vectors)


    def bulk_insert_vector(self, vectors: List[TextEmbeddingVector])->None:
        data = []
        for tv in vectors:
            # 确保写入唯一id
            data.append({"index": {"_index": self.table, "_id": tv.get_id()}})
            # 写入数据
            data.append({
                KG_FIELD_TEXT: tv.get_text(),
                KG_FIELD_VECTOR: tv.get_vector(),
            })
        self.debug_logger.debug(data)
        self.es.bulk(data)

    def query_vector(self, vector:TextEmbeddingVector):
        res = self.es.search(
            body={
                "size": 1,
                "query": {
                    "knn": {
                        KG_FIELD_VECTOR: {
                            "vector": vector.get_vector(), "k": 1
                            }
                        }
                    },
                "_source": ["text"],
            },
            index=self.table,
        )
        self.debug_logger.debug(res)
        return res['hits']['hits']

    def delete(self):
        self.es.delete_by_query(
            index=self.table,
            body={
                "query": {
                    "match_all": {}
                }
            }
        )
        self.debug_logger.debug("完成数据表(%s)的数据删除工作"%self.table)


class KnowledgeDB:
    def __init__(self, embedding: MaaSKnowledgeEmbedding,) -> None:
        self.embedding = MaaSKnowledgeEmbedding

    def load_and_save_data(self, filepath):
        pass
    
    def save_data(self, text, sep="\n"):
        units = text.split(sep)
        vectors = self.encode(units)
        print(vectors)
    
    def similarity_search(self, text):
        pass