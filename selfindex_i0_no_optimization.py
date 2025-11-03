"""
SelfIndex WITHOUT Skip Pointers (i=0) - Wrapper for existing index
Uses the already built SelfIndex-v1.100T0
Plot.A: Index optimization comparison - Baseline (no optimization)
"""
import json
import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict
import time
from index_base import IndexBase


class SelfIndexI0(IndexBase):
    """Self-implemented inverted index WITHOUT skip pointers (i=0) - Baseline"""
    
    def __init__(self):
        super().__init__(
            core='SelfIndex',
            info='TFIDF',
            dstore='CUSTOM',
            qproc='TERMatat',
            compr='NONE',
            optim='Null'  # No optimization
        )
        
        self.index_dir = Path("indices")
        
        # Inverted index: term -> list of doc indices
        self.inverted_index: Dict[str, List[int]] = {}
        
        # Document metadata
        self.doc_metadata: Dict[str, Dict] = {}
        self.doc_id_list: List[str] = []
        
        # Statistics
        self.doc_count = 0
        self.avg_doc_length = 0
        self.term_doc_freq: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
    
    def create_index(self, index_id: str = None, files: Iterable[Tuple[str, str]] = None) -> None:
        """Not used - i=0 uses existing index SelfIndex-v1.100T0"""
        print("i=0 uses existing index. Use load_index() instead.")
    
    def load_index(self, serialized_index_dump: str = None) -> None:
        """Load index from disk"""
        if serialized_index_dump is None:
            serialized_index_dump = 'SelfIndex-v1.300T0'
        
        index_path = self.index_dir / f"{serialized_index_dump}.pkl"
        
        if not index_path.exists():
            print(f"Index not found: {index_path}")
            return
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        # Check if it's a TF-IDF index (x=3) with pre-computed scores
        sample_term = next(iter(index_data['inverted_index'].keys()), None)
        is_tfidf_index = False
        if sample_term and index_data['inverted_index'][sample_term]:
            first_posting = index_data['inverted_index'][sample_term][0]
            is_tfidf_index = isinstance(first_posting, tuple)
        
        self.term_doc_freq = index_data['term_doc_freq']
        self.doc_count = index_data['num_documents']
        
        # Handle different index formats
        if 'documents' in index_data:
            # Old format: documents = {doc_id: [token, token, ...]}
            from collections import Counter
            self.doc_id_list = [str(doc_id) for doc_id in index_data['documents'].keys()]
            doc_lengths = index_data.get('doc_lengths', {})
            
            # Build doc_id to index mapping
            doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_id_list)}
            
            # Build doc_metadata from documents
            print("  Building metadata from document tokens...")
            for doc_id, tokens in index_data['documents'].items():
                # Compute term frequencies from token list
                term_freq = dict(Counter(tokens))
                doc_len = doc_lengths.get(doc_id, len(tokens))
                self.doc_metadata[str(doc_id)] = {
                    'doc_id': str(doc_id),
                    'length': doc_len,
                    'unique_terms': len(term_freq),
                    'term_freq': term_freq
                }
            
            self.avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
            
            if is_tfidf_index:
                # TF-IDF index (x=3): postings are (doc_id, score) tuples
                # Convert to doc_idx format for consistency with our query methods
                print("  Converting TF-IDF index to index format...")
                temp_inverted_index = {}
                for term, postings in index_data['inverted_index'].items():
                    # Convert doc_ids to indices
                    doc_indices = [doc_id_to_idx.get(str(doc_id), -1) for doc_id, score in postings]
                    # Filter out invalid indices
                    doc_indices = [idx for idx in doc_indices if idx >= 0]
                    temp_inverted_index[term] = doc_indices
                self.inverted_index = temp_inverted_index
            else:
                self.inverted_index = index_data['inverted_index']
        else:
            # New format
            self.doc_metadata = index_data['doc_metadata']
            self.doc_id_list = index_data.get('doc_id_list', list(self.doc_metadata.keys()))
            self.avg_doc_length = index_data.get('avg_doc_length', 0)
        
        # Build IDF cache
        self.idf_cache = {}
        for term, df in self.term_doc_freq.items():
            self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        print(f"✓ Loaded index: {serialized_index_dump}")
        print(f"  Documents: {self.doc_count:,}")
        print(f"  Unique terms: {len(self.inverted_index):,}")
    
    def query(self, query_text: str, size: int = 10) -> str:
        """Query the index - handles both Boolean and TF-IDF queries WITHOUT skip pointers"""
        # Check if it's a Boolean query
        if any(op in query_text for op in [' AND ', ' OR ', ' NOT ', '(', ')']):
            return self._boolean_query(query_text, size)
        else:
            return self._tfidf_query(query_text, size)
    
    def _boolean_query(self, query_text: str, size: int = 10) -> str:
        """Handle Boolean queries (AND, OR, NOT) WITHOUT skip pointers"""
        # Simplified Boolean query parser
        query_lower = query_text.lower()
        
        # Get all terms (remove operators and parentheses)
        import re
        terms = [t.strip() for t in re.split(r'\s+(?:AND|OR|NOT)\s+|\(|\)', query_lower) if t.strip() and t.strip() not in ['and', 'or', 'not', '']]
        
        # For complex Boolean queries, use simple evaluation
        if ' AND ' in query_text:
            # AND operation: intersect posting lists
            result_docs = None
            for term in terms:
                if term not in self.inverted_index:
                    result_docs = set()
                    break
                posting_list = set(self.inverted_index[term])
                if result_docs is None:
                    result_docs = posting_list
                else:
                    # Manual intersection without skip pointers
                    result_docs = result_docs.intersection(posting_list)
            
            if result_docs is None:
                result_docs = set()
            
        elif ' OR ' in query_text:
            # OR operation: union posting lists
            result_docs = set()
            for term in terms:
                if term in self.inverted_index:
                    result_docs.update(self.inverted_index[term])
        
        elif ' NOT ' in query_text:
            # NOT operation
            parts = query_text.split(' NOT ')
            if len(parts) == 2:
                include_term = parts[0].strip().lower()
                exclude_term = parts[1].strip().lower()
                
                result_docs = set()
                if include_term in self.inverted_index:
                    result_docs = set(self.inverted_index[include_term])
                if exclude_term in self.inverted_index:
                    result_docs -= set(self.inverted_index[exclude_term])
            else:
                result_docs = set()
        else:
            # Fallback to TF-IDF for complex queries
            return self._tfidf_query(query_text, size)
        
        # Convert to list and limit
        result_list = list(result_docs)[:size]
        
        # Check if result contains indices or doc_ids
        if result_list and isinstance(result_list[0], int):
            # Convert indices to doc_ids
            result_docs_formatted = [{'doc_id': self.doc_id_list[idx], 'score': 1.0} for idx in result_list]
        else:
            # Already doc_ids
            result_docs_formatted = [{'doc_id': str(doc_id), 'score': 1.0} for doc_id in result_list]
        
        results = {
            'query': query_text,
            'total_hits': len(result_docs),
            'results': result_docs_formatted
        }
        
        return json.dumps(results)
    
    def _tfidf_query(self, query_text: str, size: int = 10) -> str:
        """TF-IDF scoring query WITHOUT skip pointers"""
        query_terms = query_text.lower().split()
        
        # Calculate scores for all documents
        doc_scores = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            # Get posting list (simple list, no skip pointers)
            doc_indices = self.inverted_index[term]
            
            # Get pre-computed IDF
            idf = self.idf_cache.get(term, 0)
            if idf == 0:
                continue
            
            # Calculate TF-IDF for each document
            for doc_idx in doc_indices:
                doc_id = self.doc_id_list[doc_idx]
                doc_meta = self.doc_metadata[doc_id]
                
                tf = doc_meta['term_freq'].get(term, 0)
                doc_len = doc_meta['length']
                
                if doc_len > 0:
                    tf_normalized = tf / doc_len
                    if doc_idx in doc_scores:
                        doc_scores[doc_idx] += tf_normalized * idf
                    else:
                        doc_scores[doc_idx] = tf_normalized * idf
        
        # Sort by score and get top-k
        if not doc_scores:
            return json.dumps({'query': query_text, 'total_hits': 0, 'results': []})
        
        # Fast top-k selection
        if len(doc_scores) <= size:
            ranked_docs = [(self.doc_id_list[idx], score) for idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
        else:
            import heapq
            top_indices = heapq.nlargest(size, doc_scores.items(), key=lambda x: x[1])
            ranked_docs = [(self.doc_id_list[idx], score) for idx, score in top_indices]
        
        # Format results
        results = {
            'query': query_text,
            'total_hits': len(doc_scores),
            'results': [{'doc_id': doc_id, 'score': score} for doc_id, score in ranked_docs]
        }
        
        return json.dumps(results)
    
    def update_index(self, index_id: str = None, 
                    remove_files: Iterable[Tuple[str, str]] = None,
                    add_files: Iterable[Tuple[str, str]] = None) -> None:
        """Update index (simplified - rebuild for now)"""
        print("Update not fully implemented - use create_index to rebuild")
    
    def delete_index(self, index_id: str = None) -> None:
        """Delete index from disk"""
        if index_id is None:
            index_id = 'SelfIndex-v1.100T0'
        
        index_path = self.index_dir / f"{index_id}.pkl"
        if index_path.exists():
            index_path.unlink()
            print(f"✓ Deleted index: {index_id}")
    
    def list_indices(self) -> List[str]:
        """List all available indices"""
        if not self.index_dir.exists():
            return []
        return [f.stem for f in self.index_dir.glob("SelfIndex-v1.1*.pkl")]
    
    def list_indexed_files(self, index_id: str = None) -> List[str]:
        """List all document IDs in the index"""
        return list(self.doc_metadata.keys())
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'identifier': self.identifier_short,
            'optimization': 'None',
            'doc_count': self.doc_count,
            'unique_terms': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length
        }


def main():
    indexer = SelfIndexI0()
    print(f"Indexer: {indexer}")
    
    # Load existing index
    print("\nLoading existing index SelfIndex-v1.100T0...")
    indexer.load_index('SelfIndex-v1.100T0')
    
    # Test query
    print("\nTesting query...")
    results = indexer.query("machine learning")
    print(results)
    
    # Print stats
    print("\nIndex Statistics:")
    stats = indexer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
