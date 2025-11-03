"""
SelfIndex with Simple Compression (z=1)
Implements a custom inverted index with simple variable byte encoding compression
Plot.AB: Compression method comparison - Simple code implementation
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict, Counter
import time
import struct
from index_base import IndexBase

class SimpleCompression:
    """Simple Variable Byte (VByte) encoding for compression"""
    
    @staticmethod
    def encode_number(n: int) -> bytes:
        """Encode a single number using variable byte encoding"""
        if n == 0:
            return bytes([0x80])
        
        bytes_list = []
        while n > 0:
            bytes_list.append(n & 0x7F)
            n >>= 7
        
        # Set the high bit of the last byte
        bytes_list[0] |= 0x80
        
        # Reverse to get big-endian order
        return bytes(reversed(bytes_list))
    
    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        """Encode a list of numbers using variable byte encoding"""
        result = bytearray()
        for num in numbers:
            result.extend(SimpleCompression.encode_number(num))
        return bytes(result)
    
    @staticmethod
    def decode_list(data: bytes) -> List[int]:
        """Decode a list of numbers from variable byte encoding"""
        numbers = []
        current = 0
        
        for byte in data:
            if byte & 0x80:  # Last byte of number
                current = (current << 7) | (byte & 0x7F)
                numbers.append(current)
                current = 0
            else:
                current = (current << 7) | byte
        
        return numbers
    
    @staticmethod
    def encode_delta(numbers: List[int]) -> bytes:
        """Encode using delta encoding + variable byte"""
        if not numbers:
            return b''
        
        deltas = [numbers[0]]
        for i in range(1, len(numbers)):
            deltas.append(numbers[i] - numbers[i-1])
        
        return SimpleCompression.encode_list(deltas)
    
    @staticmethod
    def decode_delta(data: bytes) -> List[int]:
        """Decode delta encoded data"""
        if not data:
            return []
        
        deltas = SimpleCompression.decode_list(data)
        numbers = [deltas[0]]
        
        for i in range(1, len(deltas)):
            numbers.append(numbers[-1] + deltas[i])
        
        return numbers


class SelfIndexZ1(IndexBase):
    """Self-implemented inverted index with simple compression"""
    
    def __init__(self):
        super().__init__(
            core='SelfIndex',
            info='TFIDF',
            dstore='CUSTOM',
            qproc='TERMatat',
            compr='CODE',  # Simple code compression
            optim='Null'
        )
        
        self.index_dir = Path("indices/selfindex_z1")
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        # Inverted index: term -> compressed postings list
        self.inverted_index: Dict[str, bytes] = {}
        
        # Document metadata
        self.doc_metadata: Dict[str, Dict] = {}
        self.doc_id_list: List[str] = []  # Fast index-to-id mapping
        
        # Statistics
        self.doc_count = 0
        self.avg_doc_length = 0
        self.term_doc_freq: Dict[str, int] = {}  # Number of docs containing term
        self.idf_cache: Dict[str, float] = {}  # Pre-computed IDF scores
        
        # Memory tracking
        self.memory_stats = {
            'inverted_index_size': 0,
            'doc_metadata_size': 0,
            'total_postings': 0,
            'compressed_size': 0,
            'uncompressed_size': 0
        }
    
    def create_index(self, index_id: str = None, files: Iterable[Tuple[str, str]] = None) -> None:
        """Create inverted index with simple compression"""
        print(f"Creating {self.identifier_short} index...")
        
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
        term_positions: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        
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
            for position, term in enumerate(tokens):
                if doc_idx not in term_postings[term]:
                    term_postings[term].append(doc_idx)
                term_positions[term][doc_idx].append(position)
            
            if (doc_idx + 1) % 10000 == 0:
                print(f"  Processed {doc_idx + 1:,} documents...")
        
        self.doc_count = len(files)
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        # Create fast doc_id lookup list
        self.doc_id_list = list(self.doc_metadata.keys())
        
        # Compress postings lists
        print("Compressing postings lists...")
        for term, posting_list in term_postings.items():
            posting_list.sort()  # Ensure sorted for delta encoding
            
            # Store uncompressed size for comparison
            uncompressed_size = len(posting_list) * 4  # 4 bytes per int
            self.memory_stats['uncompressed_size'] += uncompressed_size
            
            # Compress using simple delta + vbyte encoding
            compressed_data = SimpleCompression.encode_delta(posting_list)
            self.inverted_index[term] = compressed_data
            
            # Track statistics
            self.term_doc_freq[term] = len(posting_list)
            self.memory_stats['compressed_size'] += len(compressed_data)
            self.memory_stats['total_postings'] += len(posting_list)
        
        build_time = time.time() - start_time
        
        # Pre-compute IDF scores for fast query processing
        print("Pre-computing IDF scores...")
        import math
        for term, df in self.term_doc_freq.items():
            self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        # Calculate memory usage
        self.memory_stats['inverted_index_size'] = sum(len(k.encode()) + len(v) for k, v in self.inverted_index.items())
        self.memory_stats['doc_metadata_size'] = len(json.dumps(self.doc_metadata).encode())
        
        compression_ratio = (1 - self.memory_stats['compressed_size'] / self.memory_stats['uncompressed_size']) * 100
        
        print(f"\n✓ Index created successfully!")
        print(f"  Documents: {self.doc_count:,}")
        print(f"  Unique terms: {len(self.inverted_index):,}")
        print(f"  Total postings: {self.memory_stats['total_postings']:,}")
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print(f"  Uncompressed size: {self.memory_stats['uncompressed_size'] / 1024 / 1024:.2f} MB")
        print(f"  Compressed size: {self.memory_stats['compressed_size'] / 1024 / 1024:.2f} MB")
        
        # Save index to disk
        self._save_index(index_id or 'selfindex-z1-v1.0')
    
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
            'idf_cache': self.idf_cache,
            'memory_stats': self.memory_stats
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"  Saved to: {index_path}")
    
    def load_index(self, serialized_index_dump: str = None) -> None:
        """Load index from disk"""
        if serialized_index_dump is None:
            serialized_index_dump = 'selfindex-z1-v1.0'
        
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
        self.memory_stats = index_data['memory_stats']
        
        # Build IDF cache if not present
        if not self.idf_cache:
            import math
            for term, df in self.term_doc_freq.items():
                self.idf_cache[term] = math.log((self.doc_count + 1) / (df + 1)) if df > 0 else 0
        
        print(f"✓ Loaded index: {serialized_index_dump}")
    
    def query(self, query_text: str, size: int = 10) -> str:
        """Query the index using TF-IDF scoring - highly optimized for speed"""
        query_terms = query_text.lower().split()
        
        # Calculate scores for all documents - use dict with int keys for speed
        doc_scores = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            # Decompress postings (happens before query processing)
            doc_indices = SimpleCompression.decode_delta(self.inverted_index[term])
            
            # Get pre-computed IDF
            idf = self.idf_cache.get(term, 0)
            if idf == 0:
                continue
            
            # Calculate TF-IDF for each document using index-based lookup
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
        
        # Format results - minimal JSON for speed
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
            index_id = 'selfindex-z1-v1.0'
        
        index_path = self.index_dir / f"{index_id}.pkl"
        if index_path.exists():
            index_path.unlink()
            print(f"✓ Deleted index: {index_id}")
    
    def list_indices(self) -> List[str]:
        """List all available indices"""
        if not self.index_dir.exists():
            return []
        return [f.stem for f in self.index_dir.glob("*.pkl")]
    
    def list_indexed_files(self, index_id: str = None) -> List[str]:
        """List all document IDs in the index"""
        return list(self.doc_metadata.keys())
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'identifier': self.identifier_short,
            'compression_method': 'Simple Variable Byte + Delta Encoding',
            'doc_count': self.doc_count,
            'unique_terms': len(self.inverted_index),
            'avg_doc_length': self.avg_doc_length,
            **self.memory_stats
        }


def main():
    indexer = SelfIndexZ1()
    print(f"Indexer: {indexer}")
    
    # Create index
    indexer.create_index(index_id='selfindex-z1-v1.0')
    
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
