from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from pathlib import Path
from typing import List, Dict
import time
from index_base import IndexBase

class ElasticsearchIndexer(IndexBase):
    def __init__(self):
        super().__init__(
            core='ESIndex',
            info='TFIDF',  # Elasticsearch uses TF-IDF by default
            dstore='DB1',  # Elasticsearch as datastore
            qproc='TERMatat',
            compr='CLIB',  # Elasticsearch uses compression
            optim='Null'
        )
        
        # Connect to Elasticsearch (assumes ES is running on localhost:9200)
        self.es = Elasticsearch(
            ['http://localhost:9200'],
            verify_certs=False,
            ssl_show_warn=False
        )
        self.index_name = 'esindex-v1-0'  # ES requires lowercase and no dots
        try:
            if not self.es.ping():
                print("Warning: Cannot connect to Elasticsearch")
        except Exception as e:
            print(f"Warning: Connection check failed: {e}")
    
    def create_index(self, index_id: str = None, files: List[tuple] = None) -> None:
        if index_id:
            self.index_name = index_id.lower().replace('.', '-')
        
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        
        # Create index with settings
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "porter_stem", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "tokens_processed": {"type": "keyword"}
                }
            }
        }
        
        self.es.indices.create(index=self.index_name, **index_settings)
        
        if files is None:
            data_path = Path("data/processed/wikipedia_processed.json")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            # Convert files format to documents
            documents = [
                {
                    'id': doc_id,
                    'title': '',
                    'original_text': content,
                    'tokens_processed': content.split()
                }
                for doc_id, content in files
            ]
        
        print(f"Indexing {len(documents):,} documents...")
        actions = []
        for i, doc in enumerate(documents):
            action = {
                "_index": self.index_name,
                "_id": doc.get('id', f'doc_{i}'),
                "_source": {
                    "doc_id": doc.get('id', f'doc_{i}'),
                    "title": doc.get('title', ''),
                    "text": doc.get('original_text', ''),
                    "tokens_processed": doc.get('tokens_processed', [])
                }
            }
            actions.append(action)
            
            if len(actions) >= 1000:
                bulk(self.es, actions)
                actions = []
        
        if actions:
            bulk(self.es, actions)
        
        self.es.indices.refresh(index=self.index_name)
        doc_count = self.es.indices.stats(index=self.index_name)['_all']['primaries']['docs']['count']
        print(f"Indexed {doc_count:,} documents")
    
    def load_index(self, serialized_index_dump: str = None) -> None:
        if serialized_index_dump:
            self.index_name = serialized_index_dump
        if not self.es.indices.exists(index=self.index_name):
            print(f"Index not found: {self.index_name}")
    
    def query(self, query_text: str, size: int = 10) -> str:
        search_body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["title^2", "text"],
                    "type": "best_fields"
                }
            },
            "size": size
        }
        
        response = self.es.search(index=self.index_name, **search_body)
        
        results = {
            "query": query_text,
            "total_hits": response['hits']['total']['value'],
            "results": [
                {
                    "doc_id": hit['_source']['doc_id'],
                    "title": hit['_source']['title'],
                    "score": hit['_score']
                }
                for hit in response['hits']['hits']
            ]
        }
        
        return json.dumps(results, indent=2)
    
    def update_index(self, index_id: str = None, 
                    remove_files: List[tuple] = None, 
                    add_files: List[tuple] = None) -> None:
        """Update index by removing and adding documents"""
        if index_id:
            self.index_name = index_id
        
        # Remove documents
        if remove_files:
            for doc_id, _ in remove_files:
                self.es.delete(index=self.index_name, id=doc_id, ignore=[404])
        
        if add_files:
            actions = []
            for doc_id, content in add_files:
                action = {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {
                        "doc_id": doc_id,
                        "title": "",
                        "text": content,
                        "tokens_processed": content.split()
                    }
                }
                actions.append(action)
            
            if actions:
                bulk(self.es, actions)
                self.es.indices.refresh(index=self.index_name)
    
    def delete_index(self, index_id: str = None) -> None:
        if index_id:
            self.index_name = index_id
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
    
    def list_indices(self) -> List[str]:
        return list(self.es.indices.get_alias(index="*").keys())
    
    def list_indexed_files(self, index_id: str = None) -> List[str]:
        if index_id:
            self.index_name = index_id
        doc_ids = []
        response = self.es.search(
            index=self.index_name,
            query={"match_all": {}},
            source=["doc_id"],
            scroll='2m',
            size=1000
        )
        
        scroll_id = response['_scroll_id']
        
        while len(response['hits']['hits']) > 0:
            for hit in response['hits']['hits']:
                doc_ids.append(hit['_source']['doc_id'])
            
            response = self.es.scroll(scroll_id=scroll_id, scroll='2m')
        
        return doc_ids


def main():
    indexer = ElasticsearchIndexer()
    print(f"Indexer: {indexer}")
    indexer.create_index(index_id='esindex-v1.0')
    print("Testing query...")
    results = indexer.query("machine learning")
    print(results)


if __name__ == "__main__":
    main()
