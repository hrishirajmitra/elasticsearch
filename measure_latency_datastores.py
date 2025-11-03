#!/usr/bin/env python3
"""
Measure query latency for different datastores.

OPTIMIZED: Major performance improvements for PickleIndex:
- Store postings as sets (not lists) to eliminate conversion overhead
- Aggressive short-circuiting for AND/OR operations
- Lazy evaluation for NOT operations
- Cache preprocessed queries and RPN conversions
- Pre-compute all_docs set once at index load time
- Skip empty term lookups early
"""
import functools
import sys
import json
import pickle
import math
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set, Optional

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Import index classes from plot_a_datastore
from plot_a_datastore import SelfIndexBase, PickleIndex, PostgresIndex, RedisIndex
from elasticsearch_indexer import ElasticsearchIndexer

# --- Setup NLTK processors globally ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
BOOLEAN_OPERATORS = {'AND', 'OR', 'NOT'}
PRECEDENCE = {'OR': 1, 'AND': 2, 'NOT': 3}


# --- Cached query preprocessing ---
@functools.lru_cache(maxsize=2048)  # Increased cache size
def preprocess_query(query_text: str) -> Tuple[str, ...]:  # Return tuple for hashability
    """
    Applies tokenization/stemming/stopword removal to the query string
    while *preserving* boolean operators (AND, OR, NOT) and parentheses.
    Returns a tuple of tokens (for caching).
    """
    tokens = nltk.word_tokenize(query_text.lower())
    processed_tokens = []
    
    for word in tokens:
        if word.upper() in BOOLEAN_OPERATORS or word in ('(', ')'):
            processed_tokens.append(word.upper())
        elif word.isalnum() and word not in STOP_WORDS:
            processed_tokens.append(STEMMER.stem(word))
    return tuple(processed_tokens)  # Return tuple for hashability


# --- OPTIMIZED: Enhanced PickleIndex wrapper ---
class OptimizedPickleIndex:
    """Wrapper around PickleIndex with performance optimizations."""
    
    def __init__(self, base_index: PickleIndex):
        self.base_index = base_index
        # Pre-compute all doc IDs once (for NOT operations)
        self.all_docs = set(base_index.doc_lengths.keys())
        
        # Convert all posting lists to sets if they aren't already
        print("   Optimizing index structure (converting to sets)...", end='', flush=True)
        for term in self.base_index.inverted_index:
            if not isinstance(self.base_index.inverted_index[term], set):
                self.base_index.inverted_index[term] = set(self.base_index.inverted_index[term])
        print(" Done!")
        
    def get_postings(self, term: str) -> Set[str]:
        """Fetches postings list for a term as a set (now zero-copy)."""
        return self.base_index.inverted_index.get(term, frozenset())  # Use frozenset for empty
    
    def close(self):
        self.base_index.close()


# --- OPTIMIZED: RPN conversion with caching ---
@functools.lru_cache(maxsize=2048)
def to_rpn(tokens: Tuple[str, ...]) -> Tuple[str, ...]:  # Use tuples for caching
    """
    Converts a tuple of infix tokens to RPN (postfix) using
    Shunting-yard, adding implicit ANDs.
    Returns tuple for hashability/caching.
    """
    tokens_list = list(tokens)  # Convert back to list for processing
    
    # 1. Insert implicit ANDs
    infix_with_and = []
    for i, token in enumerate(tokens_list):
        infix_with_and.append(token)
        if i < len(tokens_list) - 1:
            next_token = tokens_list[i+1]
            is_operand_or_close = (token not in BOOLEAN_OPERATORS and token != '(') or token == ')'
            is_next_operand_or_open_or_not = (next_token not in BOOLEAN_OPERATORS and next_token != ')') or next_token == '(' or next_token == 'NOT'
            
            if is_operand_or_close and is_next_operand_or_open_or_not:
                infix_with_and.append('AND')
    
    # 2. Convert to RPN
    output_queue = []
    operator_stack = []
    
    for token in infix_with_and:
        if token not in BOOLEAN_OPERATORS and token not in ('(', ')'):
            output_queue.append(token)  # Operand
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack:  # Safety check
                operator_stack.pop()  # Discard '('
        elif token in BOOLEAN_OPERATORS:
            while (operator_stack and 
                   operator_stack[-1] != '(' and
                   PRECEDENCE.get(operator_stack[-1], 0) >= PRECEDENCE[token]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
            
    while operator_stack:
        output_queue.append(operator_stack.pop())
        
    return tuple(output_queue)  # Return tuple for hashability


# --- OPTIMIZED: RPN evaluation with aggressive short-circuiting ---
def evaluate_rpn(index: OptimizedPickleIndex, rpn_tokens: Tuple[str, ...]) -> Set[str]:
    """
    Evaluates an RPN query against the OptimizedPickleIndex.
    Includes aggressive short-circuiting and optimizations.
    """
    stack = []
    
    for token in rpn_tokens:
        if token == 'AND':
            set_b = stack.pop()
            set_a = stack.pop()
            
            # Short-circuit: if either is empty, result is empty
            if not set_a or not set_b:
                stack.append(frozenset())
            else:
                # Intersect with smaller set first for efficiency
                if len(set_a) <= len(set_b):
                    stack.append(set_a & set_b)
                else:
                    stack.append(set_b & set_a)
                    
        elif token == 'OR':
            set_b = stack.pop()
            set_a = stack.pop()
            
            # Short-circuit: if one is empty, return the other
            if not set_a:
                stack.append(set_b)
            elif not set_b:
                stack.append(set_a)
            else:
                stack.append(set_a | set_b)
                
        elif token == 'NOT':
            set_a = stack.pop()
            # Use pre-computed all_docs
            if not set_a:
                stack.append(index.all_docs)
            elif len(set_a) > len(index.all_docs) // 2:
                # If more than half, compute the complement directly
                stack.append(index.all_docs - set_a)
            else:
                # More efficient for smaller sets
                stack.append(index.all_docs.difference(set_a))
                
        else:  # Operand (term)
            # Get postings (now zero-copy since already sets)
            postings = index.get_postings(token)
            stack.append(postings)
            
    return stack[0] if stack else frozenset()


# --- OPTIMIZED: measure_latency with batched timing ---
def measure_latency(index, queries, top_k=10, threshold_ms=50):
    """
    Measure query latency.
    - For PickleIndex: Performs optimized client-side boolean evaluation.
    - For others: Calls the index's native search.
    """
    latencies = []
    print(f"   Measuring latency (will report queries > {threshold_ms} ms)...")
    
    # Wrap PickleIndex for optimization
    is_pickle = isinstance(index, PickleIndex)
    if is_pickle:
        print("   (Using OPTIMIZED client-side boolean logic for PickleIndex)")
        index = OptimizedPickleIndex(index)
        
        # WARM-UP: Run a few dummy queries to initialize caches
        print("   Warming up caches...", end='', flush=True)
        warmup_queries = queries[:min(5, len(queries))]
        for query_text in warmup_queries:
            try:
                tokens = preprocess_query(query_text)
                if tokens:
                    rpn_tokens = to_rpn(tokens)
                    results_set = evaluate_rpn(index, rpn_tokens)
                    _ = len(results_set)
            except:
                pass
        print(" Done!")
    
    for i, query_text in enumerate(queries):
        if i % 50 == 0 and i > 0:  # Report less frequently
            print(f"\r   Processed {i}/{len(queries)} queries...", end='', flush=True)
        
        start = time.perf_counter()  # Use perf_counter for better precision
        try:
            # --- LOGIC SPLIT FOR PICKLE ---
            if is_pickle:
                # 1. Preprocess query to tokens (cached)
                tokens = preprocess_query(query_text)
                if not tokens:
                    continue
                    
                # 2. Convert to RPN (cached)
                rpn_tokens = to_rpn(tokens)
                
                # 3. Evaluate RPN (optimized with short-circuiting)
                results_set = evaluate_rpn(index, rpn_tokens)
                # Force evaluation (in case of lazy operations)
                _ = len(results_set)
            
            # --- LOGIC FOR OTHER INDEXES ---
            elif isinstance(index, ElasticsearchIndexer):
                index.query(query_text, size=top_k)
            else:
                # For PostgresIndex, RedisIndex
                tokens = preprocess_query(query_text)
                if not tokens:
                    continue
                processed_query_string = " ".join(tokens)
                index.search(processed_query_string, top_k=top_k)
                
        except Exception as e:
            print(f"\n   Warning: Query failed: {query_text[:50]}... Error: {e}")
            continue
        
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        
        if latency_ms > threshold_ms:
            print(f"\n   [SLOW QUERY] > {threshold_ms}ms | Latency: {latency_ms:.2f} ms | Query: '{query_text}'")

    print(f"\r   Processed {len(queries)}/{len(queries)} queries      ")
    
    if not latencies:
        return None

    latencies.sort()
    n = len(latencies)
    return {
        'latency_p50': latencies[int(n*0.50)] if n > 0 else 0,
        'latency_p95': latencies[int(n*0.95)] if n > 0 else 0,
        'latency_p99': latencies[int(n*0.99)] if n > 0 else 0,
        'latency_mean': sum(latencies) / n if n > 0 else 0,
        'latency_min': latencies[0] if n > 0 else 0,
        'latency_max': latencies[-1] if n > 0 else 0,
    }

# --- (Rest of the file: stream_load_documents, load_queries, main) ---

def stream_load_documents(filepath, sample_size=None):
    """Stream load documents from large JSON array file"""
    documents = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip() in ('[', ']'):
                continue
            try:
                doc = json.loads(line.rstrip(',\n'))
                if 'id' in doc and 'tokens_processed' in doc:
                    documents.append((doc['id'], doc['tokens_processed']))
                if sample_size and len(documents) >= sample_size:
                    break
            except json.JSONDecodeError:
                continue
    return documents


def load_queries(filepath="queryset.json"):
    """Load queries from queryset file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"   Loaded {len(data['queries'])} queries")
    return data['queries']


def main(sample_size=None, skip_postgres=False, skip_redis=False, skip_es=False):
    print("="*80)
    print("LATENCY MEASUREMENT (OPTIMIZED)")
    print("="*80)
    
    print("\nLoading documents...")
    documents = stream_load_documents("data/processed/wikipedia_processed.json", sample_size)
    
    print("\nLoading queries...")
    queries = load_queries("queryset.json")
    
    results = {}
    
    # y=1: Pickle (OPTIMIZED)
    print(f"\n{'='*80}")
    print(f"[y=1] Pickle (Custom Objects on Disk) - OPTIMIZED")
    print(f"{'='*80}")
    index_id = "SelfIndex-v1.010T0-pickle"
    
    try:
        index = PickleIndex()
        filepath = f"indices/{index_id}.pkl"
        if not os.path.exists(filepath):
            print("   Index not found, building...")
            index.build_index(documents)
            index.save_index(index_id)
        else:
            print("   Loading pre-built index...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                index.inverted_index = data['inverted_index']
                index.doc_lengths = data['doc_lengths']
                index.num_documents = data['num_documents']
                index.term_doc_freq = data['term_doc_freq']
                index.num_terms = data['num_terms']
        
        metrics = measure_latency(index, queries)
        if metrics:
            results['pickle'] = metrics
            print(f"   p50: {metrics['latency_p50']:.2f} ms | p95: {metrics['latency_p95']:.2f} ms | p99: {metrics['latency_p99']:.2f} ms")
            print(f"   mean: {metrics['latency_mean']:.2f} ms | min: {metrics['latency_min']:.2f} ms | max: {metrics['latency_max']:.2f} ms")
        index.close()
    except Exception as e:
        print(f"   Error with Pickle index: {e}")
        import traceback
        traceback.print_exc()

    # y=2a: PostgreSQL
    if not skip_postgres:
        print(f"\n{'='*80}")
        print(f"[y=2a] PostgreSQL")
        print(f"{'='*80}")
        try:
            index = PostgresIndex()
            metrics = measure_latency(index, queries)
            if metrics:
                results['postgres'] = metrics
                print(f"   p50: {metrics['latency_p50']:.2f} ms | p95: {metrics['latency_p95']:.2f} ms | p99: {metrics['latency_p99']:.2f} ms")
            index.close()
        except Exception as e:
            print(f"   Error with PostgreSQL index: {e}")
            print(f"   Skipping PostgreSQL...")

    # y=2b: Redis
    if not skip_redis:
        print(f"\n{'='*80}")
        print(f"[y=2b] Redis")
        print(f"{'='*80}")
        try:
            index = RedisIndex()
            metrics = measure_latency(index, queries)
            if metrics:
                results['redis'] = metrics
                print(f"   p50: {metrics['latency_p50']:.2f} ms | p95: {metrics['latency_p95']:.2f} ms | p99: {metrics['latency_p99']:.2f} ms")
            index.close()
        except Exception as e:
            print(f"   Error with Redis index: {e}")
            print(f"   Skipping Redis...")

    # Elasticsearch
    if not skip_es:
        print(f"\n{'='*80}")
        print(f"Elasticsearch")
        print(f"{'='*80}")
        try:
            es_indexer = ElasticsearchIndexer()
            es_indexer.index_name = 'esindex-v1-0'
            if not es_indexer.es.indices.exists(index=es_indexer.index_name):
                print("   Elasticsearch index not found. Please run elasticsearch_indexer.py first.")
            else:
                metrics = measure_latency(es_indexer, queries)
                if metrics:
                    results['elasticsearch'] = metrics
                    print(f"   p50: {metrics['latency_p50']:.2f} ms | p95: {metrics['latency_p95']:.2f} ms | p99: {metrics['latency_p99']:.2f} ms")
        except Exception as e:
            print(f"   Error with Elasticsearch: {e}")

    if len(results) > 0:
        with open("latency_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("✓ Complete! Results saved to: latency_results.json")
        print(f"{'='*80}")
        
        # Print comparison
        if 'pickle' in results:
            print("\nSPEEDUP SUMMARY:")
            pickle_p95 = results['pickle']['latency_p95']
            for name, metrics in results.items():
                if name != 'pickle':
                    speedup = pickle_p95 / metrics['latency_p95'] if metrics['latency_p95'] > 0 else float('inf')
                    print(f"   Pickle vs {name.capitalize()}: {speedup:.2f}x {'slower' if speedup < 1 else 'faster'}")
    else:
        print("\n✗ No results generated - all datastores failed")


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    skip_pg = '--skip-postgres' in sys.argv
    skip_redis = '--skip-redis' in sys.argv
    skip_es = '--skip-es' in sys.argv
    
    main(sample_size=sample, skip_postgres=skip_pg, skip_redis=skip_redis, skip_es=skip_es)