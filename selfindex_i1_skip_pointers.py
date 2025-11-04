"""
SelfIndex with Skip Pointers Optimization (i=1)
Implements skip pointers for faster query processing on posting lists
Plot.A: Index optimization comparison - Skip pointers vs No skip pointers
"""
import json
import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional
from collections import defaultdict, Counter
import time
from index_base import IndexBase
from query_preprocessing import preprocess_query


class SkipPointer:
    """Skip pointer structure for faster posting list traversal"""
    
    def __init__(self, doc_id: int, position: int):
        self.doc_id = doc_id  # Document ID at this skip point
        self.position = position  # Position in posting list
    
    def __repr__(self):
        return f"Skip(doc={self.doc_id}, pos={self.position})"


class PostingListWithSkips:
    """Posting list with skip pointers"""
    
    def __init__(self, postings: List[int], skip_interval: Optional[int] = None):
        """
        Args:
            postings: Sorted list of document IDs
            skip_interval: Number of postings between skip pointers (default: sqrt of list length)
        """
        self.postings = postings
        self.length = len(postings)
        
        # Calculate optimal skip interval (larger = fewer skip pointers, less overhead)
        if skip_interval is None:
            # Use sqrt(n) but with minimum of 8 to reduce overhead
            skip_interval = max(int(math.sqrt(self.length)), 8)
        
        self.skip_interval = skip_interval
        self.skip_pointers: List[SkipPointer] = []
        
        # Build skip pointers only for larger lists
        if self.length > 20:  # Don't build for small lists
            self._build_skip_pointers()
    
    def _build_skip_pointers(self):
        """Build skip pointers at regular intervals"""
        for i in range(0, self.length, self.skip_interval):
            if i < self.length:
                self.skip_pointers.append(SkipPointer(self.postings[i], i))
    
    def get_postings(self) -> List[int]:
        """Get the raw posting list"""
        return self.postings
    
    def find_skip_point(self, target_doc_id: int) -> int:
        """
        Find the best skip point to start searching for target_doc_id
        Returns the position to start linear search from
        """
        if not self.skip_pointers:
            return 0
        
        # Binary search on skip pointers
        left, right = 0, len(self.skip_pointers) - 1
        result_pos = 0
        
        while left <= right:
            mid = (left + right) // 2
            if self.skip_pointers[mid].doc_id <= target_doc_id:
                result_pos = self.skip_pointers[mid].position
                left = mid + 1
            else:
                right = mid - 1
        
        return result_pos
    
    def intersect_with(self, other: 'PostingListWithSkips') -> List[int]:
        """
        Intersect two posting lists using skip pointers
        More efficient than linear merge when lists have different sizes
        """
        # For small lists, use simple set intersection (faster)
        if self.length < 50 or other.length < 50:
            return list(set(self.postings) & set(other.postings))
        
        result = []
        i, j = 0, 0
        
        while i < self.length and j < other.length:
            if self.postings[i] == other.postings[j]:
                result.append(self.postings[i])
                i += 1
                j += 1
            elif self.postings[i] < other.postings[j]:
                # Use skip pointer to jump ahead (only if we have them and it helps)
                if self.skip_pointers and other.postings[j] - self.postings[i] > self.skip_interval:
                    skip_pos = self.find_skip_point(other.postings[j])
                    if skip_pos > i:
                        i = skip_pos
                        continue
                i += 1
            else:
                # Use skip pointer to jump ahead (only if we have them and it helps)
                if other.skip_pointers and self.postings[i] - other.postings[j] > other.skip_interval:
                    skip_pos = other.find_skip_point(self.postings[i])
                    if skip_pos > j:
                        j = skip_pos
                        continue
                j += 1
        
        return result
    
    def serialize(self) -> bytes:
        """Serialize posting list with skip pointers"""
        data = {
            'postings': self.postings,
            'skip_interval': self.skip_interval
        }
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data: bytes) -> 'PostingListWithSkips':
        """Deserialize posting list with skip pointers"""
        obj = pickle.loads(data)
        return PostingListWithSkips(obj['postings'], obj['skip_interval'])


class SelfIndexI1(IndexBase):
    """Self-implemented inverted index WITH skip pointers (i=1)"""
    
    def __init__(self):
        super().__init__(
            core='SelfIndex',
            info='TFIDF',
            dstore='CUSTOM',
            qproc='TERMatat',
            compr='NONE',
            optim='SKIP'  # Skip pointers optimization
        )
        
        self.index_dir = Path("indices")
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        # Inverted index: term -> PostingListWithSkips
        self.inverted_index: Dict[str, PostingListWithSkips] = {}
        
        # Document metadata
        self.doc_metadata: Dict[str, Dict] = {}
        self.doc_id_list: List[str] = []
        
        # Statistics
        self.doc_count = 0
        self.avg_doc_length = 0
        self.term_doc_freq: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        
        # Skip pointer statistics
        self.skip_stats = {
            'total_skip_pointers': 0,
            'avg_skip_interval': 0,
            'skip_overhead_bytes': 0
        }
    
    def create_index(self, index_id: str = None, files: Iterable[Tuple[str, str]] = None) -> None:
        """Create inverted index WITH skip pointers"""
        print(f"Creating {self.identifier_short} index with skip pointers...")
        
        if files is None:
            # Load from processed data
            data_path = Path("data/processed/wikipedia_processed.json")
            with open(data_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            files = [
                (doc['id'], ' '.join(doc['tokens_processed']))
                for doc in documents
            ]
        
        # Build inverted index
        term_postings: Dict[str, List[int]] = defaultdict(list)
        doc_lengths = []
        
        print(f"Processing {len(files):,} documents...")
        start_time = time.time()
        
        for doc_idx, (doc_id, content) in enumerate(files):
            tokens = content.split()
            doc_lengths.append(len(tokens))
            
            # Store document metadata
            term_freq = Counter(tokens)
            self.doc_metadata[str(doc_id)] = {
                'doc_id': str(doc_id),
                'length': len(tokens),
                'unique_terms': len(term_freq),
                'term_freq': dict(term_freq)
            }
            
            # Build postings
            for term in set(tokens):  # Use set to avoid duplicates
                term_postings[term].append(doc_idx)
            
            if (doc_idx + 1) % 10000 == 0:
                print(f"  Processed {doc_idx + 1:,} documents...")
        
        self.doc_count = len(files)
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        self.doc_id_list = list(self.doc_metadata.keys())
        
        # Create posting lists with skip pointers
        print("Building posting lists with skip pointers...")
        total_skips = 0
        skip_intervals = []
        
        for term, posting_list in term_postings.items():
            posting_list.sort()  # Ensure sorted
            
            # Create posting list with skip pointers
            pl_with_skips = PostingListWithSkips(posting_list)
            self.inverted_index[term] = pl_with_skips
            self.term_doc_freq[term] = len(posting_list)
            
            # Track statistics
            total_skips += len(pl_with_skips.skip_pointers)
            skip_intervals.append(pl_with_skips.skip_interval)
        
        build_time = time.time() - start_time
        
        # Calculate skip pointer statistics
        self.skip_stats['total_skip_pointers'] = total_skips
        self.skip_stats['avg_skip_interval'] = sum(skip_intervals) / len(skip_intervals) if skip_intervals else 0
        # Each skip pointer stores: doc_id (4 bytes) + position (4 bytes) = 8 bytes
        self.skip_stats['skip_overhead_bytes'] = total_skips * 8
        
        # Pre-compute IDF scores
        print("Pre-computing IDF scores...")
        for term, df in self.term_doc_freq.items():
            self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        print(f"\n✓ Index created successfully!")
        print(f"  Documents: {self.doc_count:,}")
        print(f"  Unique terms: {len(self.inverted_index):,}")
        print(f"  Total skip pointers: {self.skip_stats['total_skip_pointers']:,}")
        print(f"  Avg skip interval: {self.skip_stats['avg_skip_interval']:.1f}")
        print(f"  Skip pointer overhead: {self.skip_stats['skip_overhead_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Build time: {build_time:.2f}s")
        
        # Save index to disk
        self._save_index(index_id or 'SelfIndex-v1.10010')
    
    def _save_index(self, index_id: str):
        """Persist index to disk"""
        index_path = self.index_dir / f"{index_id}.pkl"
        
        # Serialize posting lists
        serialized_index = {}
        for term, pl_with_skips in self.inverted_index.items():
            serialized_index[term] = pl_with_skips.serialize()
        
        index_data = {
            'inverted_index': serialized_index,
            'doc_metadata': self.doc_metadata,
            'doc_id_list': self.doc_id_list,
            'doc_count': self.doc_count,
            'avg_doc_length': self.avg_doc_length,
            'term_doc_freq': self.term_doc_freq,
            'idf_cache': self.idf_cache,
            'skip_stats': self.skip_stats
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"  Saved to: {index_path}")
    
    def load_index(self, serialized_index_dump: str = None) -> None:
        """Load index from disk"""
        if serialized_index_dump is None:
            serialized_index_dump = 'SelfIndex-v1.10010'
        
        index_path = self.index_dir / f"{serialized_index_dump}.pkl"
        
        if not index_path.exists():
            print(f"Index not found: {index_path}")
            return
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        # Deserialize posting lists
        self.inverted_index = {}
        for term, serialized_pl in index_data['inverted_index'].items():
            self.inverted_index[term] = PostingListWithSkips.deserialize(serialized_pl)
        
        self.doc_metadata = index_data['doc_metadata']
        self.doc_id_list = index_data['doc_id_list']
        self.doc_count = index_data['doc_count']
        self.avg_doc_length = index_data['avg_doc_length']
        self.term_doc_freq = index_data['term_doc_freq']
        self.idf_cache = index_data['idf_cache']
        self.skip_stats = index_data.get('skip_stats', {})
        
        print(f"✓ Loaded index: {serialized_index_dump}")
        print(f"  Skip pointers: {self.skip_stats.get('total_skip_pointers', 0):,}")
    
    def query(self, query_text: str, size: int = 10) -> str:
        """Query the index - handles both Boolean and TF-IDF queries WITH skip pointers"""
        # Check if it's a Boolean query
        if any(op in query_text for op in [' AND ', ' OR ', ' NOT ', '(', ')']):
            return self._boolean_query(query_text, size)
        else:
            return self._tfidf_query(query_text, size)
    
    def _boolean_query(self, query_text: str, size: int = 10) -> str:
        """Handle Boolean queries (AND, OR, NOT) WITH skip pointer optimization"""
        # Simplified Boolean query parser
        query_lower = query_text.lower()
        
        # Get all terms (remove operators and parentheses)
        import re
        terms = [t.strip() for t in re.split(r'\s+(?:AND|OR|NOT)\s+|\(|\)', query_lower) if t.strip() and t.strip() not in ['and', 'or', 'not', '']]
        
        # For complex Boolean queries, use skip pointer-optimized evaluation
        if ' AND ' in query_text:
            # AND operation: intersect posting lists using skip pointers
            result_docs = None
            posting_lists = []
            
            for term in terms:
                if term not in self.inverted_index:
                    result_docs = []
                    break
                posting_lists.append(self.inverted_index[term])
            
            if result_docs is None and posting_lists:
                # Sort by length (shortest first for faster intersection)
                posting_lists.sort(key=lambda pl: pl.length)
                
                # Use skip pointer intersection for first two lists
                if len(posting_lists) >= 2:
                    result_docs = posting_lists[0].intersect_with(posting_lists[1])
                    # Intersect with remaining lists using simple algorithm (avoid overhead)
                    for pl in posting_lists[2:]:
                        if not result_docs:  # Early termination
                            break
                        # Simple set intersection (faster than creating temp PostingList)
                        result_set = set(result_docs)
                        result_docs = [doc_id for doc_id in pl.get_postings() if doc_id in result_set]
                elif len(posting_lists) == 1:
                    result_docs = posting_lists[0].get_postings()
                else:
                    result_docs = []
            
            if result_docs is None:
                result_docs = []
            
        elif ' OR ' in query_text:
            # OR operation: union posting lists
            result_docs = set()
            for term in terms:
                if term in self.inverted_index:
                    result_docs.update(self.inverted_index[term].get_postings())
        
        elif ' NOT ' in query_text:
            # NOT operation
            parts = query_text.split(' NOT ')
            if len(parts) == 2:
                include_term = parts[0].strip().lower()
                exclude_term = parts[1].strip().lower()
                
                result_docs = set()
                if include_term in self.inverted_index:
                    result_docs = set(self.inverted_index[include_term].get_postings())
                if exclude_term in self.inverted_index:
                    result_docs -= set(self.inverted_index[exclude_term].get_postings())
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
        """TF-IDF scoring query WITH skip pointers (though not beneficial for scoring)"""
        # Preprocess query (stem, remove stopwords)
        processed_query = preprocess_query(query_text)
        query_terms = processed_query.split() if processed_query else []
        
        # Calculate scores for all documents
        doc_scores = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            # Get posting list with skip pointers
            pl_with_skips = self.inverted_index[term]
            doc_indices = pl_with_skips.get_postings()
            
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
            index_id = 'SelfIndex-v1.10010'
        
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
            'optimization': 'Skip Pointers',
            'doc_count': self.doc_count,
            'unique_terms': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length,
            **self.skip_stats
        }


def main():
    indexer = SelfIndexI1()
    print(f"Indexer: {indexer}")
    
    # Create index
    indexer.create_index(index_id='SelfIndex-v1.10010')
    
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
