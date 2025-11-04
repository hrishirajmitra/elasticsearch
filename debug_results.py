"""
Debug script to check why precision/recall is low
"""
import json
from elasticsearch import Elasticsearch
from selfindex_q_daat import SelfIndexDAAT

# Initialize
es = Elasticsearch(['http://localhost:9200'])
indexer = SelfIndexDAAT()
indexer.load_index('SelfIndex-v1.300D0')

# Test a few queries
test_queries = [
    "machine learning",
    "climate change",
    "quantum computing"
]

print("="*80)
print("DEBUGGING: Comparing ES vs SelfIndex Results")
print("="*80)

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"Query: '{query}'")
    print(f"{'='*80}")
    
    # Get ES results
    es_response = es.search(
        index="esindex-v1-0",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"]
                }
            },
            "size": 10
        }
    )
    
    es_docs = [hit['_id'] for hit in es_response['hits']['hits']]
    es_scores = [hit['_score'] for hit in es_response['hits']['hits']]
    
    # Get SelfIndex results
    self_results = json.loads(indexer.query(query, size=10))
    self_docs = [r['doc_id'] for r in self_results['results']]
    self_scores = [r['score'] for r in self_results['results']]
    
    print(f"\nES Results (BM25):")
    for i, (doc, score) in enumerate(zip(es_docs[:5], es_scores[:5]), 1):
        print(f"  {i}. Doc {doc}: {score:.4f}")
    
    print(f"\nSelfIndex Results (TF-IDF):")
    for i, (doc, score) in enumerate(zip(self_docs[:5], self_scores[:5]), 1):
        print(f"  {i}. Doc {doc}: {score:.4f}")
    
    # Calculate overlap
    overlap = set(es_docs) & set(self_docs)
    precision = len(overlap) / len(self_docs) if self_docs else 0
    recall = len(overlap) / len(es_docs) if es_docs else 0
    
    print(f"\nOverlap: {len(overlap)}/10 documents")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Common docs: {sorted(overlap)}")

print("\n" + "="*80)
