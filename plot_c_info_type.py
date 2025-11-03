#!/usr/bin/env python3
"""
Plot C: Information Type Variants (x=1,2,3)
Builds SelfIndex-v1.x00T0 and generates memory footprint comparison plot
"""
import sys
import json
import pickle
import math
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Setup
sns.set_style("whitegrid")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SelfIndex:
    """SelfIndex implementation for Plot C"""
    
    def __init__(self, x: int):
        """x: 1=Boolean, 2=WordCount, 3=TF-IDF"""
        self.x = x
        self.version = f"1.{x}00T0"
        self.index_name = f"SelfIndex-v{self.version}"
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.doc_lengths = {}
        self.num_documents = 0
        self.term_doc_freq = defaultdict(int)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def build_index(self, documents: List[Tuple[str, List[str]]]):
        """Build index"""
        self.num_documents = len(documents)
        
        for doc_id, tokens in documents:
            self.documents[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            term_positions = defaultdict(list)
            
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                count = len(positions)
                if self.x == 1:
                    # Boolean: store only doc_id (presence/absence)
                    self.inverted_index[term].append(doc_id)
                elif self.x == 2:
                    self.inverted_index[term].append((doc_id, count))
                elif self.x == 3:
                    self.inverted_index[term].append((doc_id, count))
                    self.term_doc_freq[term] += 1
        
        if self.x == 3:
            for term, postings in self.inverted_index.items():
                df = self.term_doc_freq[term]
                idf = math.log(self.num_documents / df) if df > 0 else 0
                new_postings = []
                for doc_id, count in postings:
                    tf = count / self.doc_lengths[doc_id]
                    new_postings.append((doc_id, tf * idf))
                self.inverted_index[term] = new_postings
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search"""
        query_terms = [self.stemmer.stem(t.lower()) for t in query.split() 
                      if t.lower() not in self.stop_words]
        
        if self.x == 1:
            # Boolean: inverted_index stores doc_ids directly
            docs = set()
            for term in query_terms:
                if term in self.inverted_index:
                    docs.update(self.inverted_index[term])
            return [(doc_id, 1.0) for doc_id in list(docs)[:top_k]]
        
        scores = defaultdict(float)
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, score in self.inverted_index[term]:
                    scores[doc_id] += score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def save_index(self, index_id: str):
        """Save to disk"""
        Path("indices").mkdir(exist_ok=True)
        with open(f"indices/{index_id}.pkl", 'wb') as f:
            pickle.dump({
                'x': self.x,
                'inverted_index': dict(self.inverted_index),
                'documents': self.documents,
                'doc_lengths': self.doc_lengths,
                'num_documents': self.num_documents,
                'term_doc_freq': dict(self.term_doc_freq)
            }, f)


def measure_metrics(index, index_id, queries):
    """Measure performance metrics"""
    filepath = f"indices/{index_id}.pkl"
    memory_mb = os.path.getsize(filepath) / (1024**2) if os.path.exists(filepath) else 0
    
    latencies = []
    for query in queries:
        start = time.time()
        index.search(query, top_k=10)
        latencies.append((time.time() - start) * 1000)
    
    latencies.sort()
    n = len(latencies)
    return {
        'x': index.x,
        'memory_mb': memory_mb,
        'latency_p95': latencies[int(n*0.95)] if n > 0 else 0,
        'num_docs': index.num_documents,
        'num_terms': len(index.inverted_index)
    }


def generate_plot(results):
    """Generate Plot C"""
    labels = ['Boolean\n(x=1)', 'Word Count\n(x=2)', 'TF-IDF\n(x=3)']
    memory = [r['memory_mb'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, memory, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Memory Footprint (MB)', fontsize=14, fontweight='bold')
    ax.set_title('Plot C: Memory Footprint vs Information Type', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    stats = f"Docs: {results[0]['num_docs']:,}\nTerms: {results[0]['num_terms']:,}"
    ax.text(0.98, 0.97, stats, transform=ax.transAxes, fontsize=10,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/plots/plot_c_info_type.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: data/plots/plot_c_info_type.png")
    plt.show()


def stream_load_documents(filepath, sample_size=None):
    """Stream load documents from large JSON array file"""
    import re
    documents = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip opening bracket
        line = f.readline()
        
        doc_buffer = []
        in_doc = False
        brace_count = 0
        
        for line in f:
            line = line.strip()
            
            if line.startswith('{'):
                in_doc = True
                doc_buffer = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_doc:
                doc_buffer.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    # Complete document
                    doc_str = '\n'.join(doc_buffer).rstrip(',')
                    try:
                        doc = json.loads(doc_str)
                        # Validate required fields
                        if 'id' not in doc:
                            print(f"\r  Warning: Skipped document without 'id' field")
                            continue
                        if 'tokens_processed' not in doc:
                            print(f"\r  Warning: Skipped document {doc.get('id', 'unknown')} without 'tokens_processed' field")
                            continue
                        
                        documents.append((doc['id'], doc['tokens_processed']))
                        
                        if len(documents) % 1000 == 0:
                            print(f"\r  Loaded {len(documents)} documents...", end='', flush=True)
                        
                        if sample_size and len(documents) >= sample_size:
                            print(f"\r  Loaded {len(documents)} documents   ")
                            return documents
                    except json.JSONDecodeError as e:
                        print(f"\r  Warning: Skipped malformed JSON document: {e}")
                    except Exception as e:
                        print(f"\r  Warning: Skipped document due to error: {e}")
                    
                    in_doc = False
                    doc_buffer = []
    
    print(f"\r  Loaded {len(documents)} documents   ")
    return documents


def main(sample_size=None):
    print("="*80)
    print("PLOT C: INFORMATION TYPE COMPARISON")
    print("="*80)
    
    # Load data with streaming
    print("\nLoading documents...")
    documents = stream_load_documents("data/processed/wikipedia_processed.json", sample_size)
    print(f"Loaded {len(documents)} documents")
    
    # Test queries
    queries = ["machine learning", "artificial intelligence", "neural network",
               "deep learning", "natural language processing"]
    
    # Build indices
    results = []
    for x in [1, 2, 3]:
        info_name = ['Boolean', 'Word Count', 'TF-IDF'][x-1]
        index_id = f"SelfIndex-v1.{x}00T0"
        
        print(f"\n[x={x}] {info_name}")
        index = SelfIndex(x=x)
        index.build_index(documents)
        index.save_index(index_id)
        
        metrics = measure_metrics(index, index_id, queries)
        results.append(metrics)
        print(f"  Memory: {metrics['memory_mb']:.2f} MB | p95: {metrics['latency_p95']:.2f} ms")
    
    # Generate plot
    generate_plot(results)
    
    # Save results
    with open("plot_c_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Complete! Results: plot_c_results.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(sample_size=sample)
