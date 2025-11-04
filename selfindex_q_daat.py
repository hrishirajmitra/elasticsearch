"""
SelfIndex with Document-at-a-Time Query Processing (q=D0)
Implements DAAT query processing for comparison with TAAT
Plot.AC: Query processing comparison - Term-at-a-time vs Document-at-a-time
"""
import json
import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict, Counter
import time
from index_base import IndexBase
from query_preprocessing import preprocess_query


class SelfIndexDAAT(IndexBase):
    """Self-implemented inverted index with Document-at-a-Time query processing
    
    Args:
        scoring: Scoring function ('tfidf' or 'bm25'), default='tfidf'
        k1: BM25 parameter for term frequency saturation, default=1.2
        b: BM25 parameter for document length normalization, default=0.75
    """
    
    def __init__(self, scoring='tfidf', k1=1.2, b=0.75):
        # Update info field based on scoring
        info_field = 'BM25' if scoring == 'bm25' else 'TFIDF'
        
        super().__init__(
            core='SelfIndex',
            info=info_field,
            dstore='CUSTOM',
            qproc='DOCatat',
            compr='NONE',
            optim='Null'
        )
        
        self.index_dir = Path("indices")
        
        # Scoring parameters
        self.scoring = scoring
        self.k1 = k1  # BM25 term frequency saturation
        self.b = b    # BM25 document length normalization
        
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
        """Create inverted index for DAAT processing"""
        print(f"Creating {self.identifier_short} index with DAAT query processing...")
        
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
            for term in set(tokens):
                term_postings[term].append(doc_idx)
            
            if (doc_idx + 1) % 10000 == 0:
                print(f"  Processed {doc_idx + 1:,} documents...")
        
        self.doc_count = len(files)
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        self.doc_id_list = list(self.doc_metadata.keys())
        
        # Store posting lists
        print("Building posting lists...")
        for term, posting_list in term_postings.items():
            posting_list.sort()
            self.inverted_index[term] = posting_list
            self.term_doc_freq[term] = len(posting_list)
        
        build_time = time.time() - start_time
        
        # Pre-compute IDF scores
        print("Pre-computing IDF scores...")
        for term, df in self.term_doc_freq.items():
            self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        print(f"\n✓ Index created successfully!")
        print(f"  Documents: {self.doc_count:,}")
        print(f"  Unique terms: {len(self.inverted_index):,}")
        print(f"  Build time: {build_time:.2f}s")
        
        # Save index to disk
        self._save_index(index_id or 'SelfIndex-v1.300D0')
    
    def _save_index(self, index_id: str):
        """Persist index to disk"""
        index_path = self.index_dir / f"{index_id}.pkl"
        
        index_data = {
            'inverted_index': self.inverted_index,
            'doc_metadata': self.doc_metadata,
            'doc_id_list': self.doc_id_list,
            'doc_count': self.doc_count,
            'avg_doc_length': self.avg_doc_length,
            'term_doc_freq': self.term_doc_freq,
            'idf_cache': self.idf_cache
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"  Saved to: {index_path}")
    
    def load_index(self, serialized_index_dump: str = None) -> None:
        """Load index from disk"""
        if serialized_index_dump is None:
            serialized_index_dump = 'SelfIndex-v1.300D0'
        
        index_path = self.index_dir / f"{serialized_index_dump}.pkl"
        
        if not index_path.exists():
            print(f"Index not found: {index_path}")
            return
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.inverted_index = index_data['inverted_index']
        self.doc_metadata = index_data['doc_metadata']
        self.doc_id_list = index_data.get('doc_id_list', list(self.doc_metadata.keys()))
        self.doc_count = index_data['doc_count']
        self.avg_doc_length = index_data['avg_doc_length']
        self.term_doc_freq = index_data['term_doc_freq']
        self.idf_cache = index_data.get('idf_cache', {})
        
        # Build IDF cache if not present
        if not self.idf_cache:
            for term, df in self.term_doc_freq.items():
                self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        print(f"✓ Loaded index: {serialized_index_dump}")
        print(f"  Documents: {self.doc_count:,}")
        print(f"  Unique terms: {len(self.inverted_index):,}")
    
    def query(self, query_text: str, size: int = 10) -> str:
        """
        Document-at-a-Time (DAAT) query processing
        Process one document at a time, accumulating scores for all query terms in that document
        """
        # Preprocess query (stem, remove stopwords)
        processed_query = preprocess_query(query_text)
        query_terms = processed_query.split() if processed_query else []
        
        # Get all documents that contain at least one query term
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        if not candidate_docs:
            return json.dumps({'query': query_text, 'total_hits': 0, 'results': []})
        
        # Pre-compute query term IDFs
        query_idfs = {}
        for term in query_terms:
            if term in self.idf_cache:
                query_idfs[term] = self.idf_cache[term]
        
        # DAAT: Process each document, computing its score for all query terms
        doc_scores = {}
        for doc_idx in candidate_docs:
            doc_id = self.doc_id_list[doc_idx]
            doc_meta = self.doc_metadata[doc_id]
            doc_len = doc_meta['length']
            
            if doc_len == 0:
                continue
            
            # Accumulate score for this document across all query terms
            score = 0.0
            for term in query_terms:
                if term in query_idfs:
                    tf = doc_meta['term_freq'].get(term, 0)
                    if tf > 0:
                        idf = query_idfs[term]
                        
                        # Calculate score based on selected scoring function
                        if self.scoring == 'bm25':
                            # BM25 scoring with term saturation and length normalization
                            numerator = tf * (self.k1 + 1)
                            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                            score += idf * (numerator / denominator)
                        else:
                            # TF-IDF scoring (normalized by document length)
                            tf_normalized = tf / doc_len
                            score += tf_normalized * idf
            
            if score > 0:
                doc_scores[doc_idx] = score
        
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
            index_id = 'SelfIndex-v1.300D0'
        
        index_path = self.index_dir / f"{index_id}.pkl"
        if index_path.exists():
            index_path.unlink()
            print(f"✓ Deleted index: {index_id}")
    
    def list_indices(self) -> List[str]:
        """List all available indices"""
        if not self.index_dir.exists():
            return []
        return [f.stem for f in self.index_dir.glob("SelfIndex-v1.3*D*.pkl")]
    
    def list_indexed_files(self, index_id: str = None) -> List[str]:
        """List all document IDs in the index"""
        return list(self.doc_metadata.keys())
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'identifier': self.identifier_short,
            'query_processing': 'Document-at-a-Time (DAAT)',
            'doc_count': self.doc_count,
            'unique_terms': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length
        }


def main():
    indexer = SelfIndexDAAT()
    print(f"Indexer: {indexer}")
    
    # Create index
    print("\nCreating DAAT index...")
    indexer.create_index('SelfIndex-v1.300D0')
    
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
