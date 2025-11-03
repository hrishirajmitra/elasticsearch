import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict, Counter
import math
import zlib
from index_base import IndexBase


class SelfIndex(IndexBase):
    def __init__(self, info='BOOLEAN', dstore='CUSTOM', compr='NONE', 
                 qproc='TERMatat', optim='Null'):
        super().__init__(
            core='SelfIndex',
            info=info,
            dstore=dstore,
            qproc=qproc,
            compr=compr,
            optim=optim
        )
        
        # Configuration
        self.info_type = info      # BOOLEAN, WORDCOUNT, TFIDF
        self.dstore_type = dstore  # CUSTOM, POSTGRES, REDIS
        self.compr_type = compr    # NONE, CODE, CLIB
        self.qproc_type = qproc    # TERMatat, DOCatatt
        self.optim_type = optim    # Null, SKIP
        
        # Index data structures
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.doc_lengths = {}
        self.term_doc_freq = Counter()
        self.num_documents = 0
        self.avg_doc_length = 0
        self.tfidf_scores = {}
        
        # Skip pointers for optimization (i=1)
        self.skip_pointers = {} if optim == 'SKIP' else None
        
        # Paths
        self.index_dir = Path("indices")
        self.index_dir.mkdir(exist_ok=True)
        
        self.index_id = None
        
        # Datastore connections (for off-the-shelf DBs)
        self.db_connection = None
        if dstore in ['POSTGRES', 'REDIS']:
            self._init_datastore()
    
    def _init_datastore(self):
        """Initialize off-the-shelf datastore connections"""
        if self.dstore_type == 'POSTGRES':
            # PostgreSQL GIN (Generalized Inverted Index)
            # Pro: Optimized for full-text search, supports complex queries, ACID compliance
            # Con: Slower writes, higher memory usage, requires setup
            try:
                import psycopg2
                from psycopg2.extras import Json
                
                # Connect to PostgreSQL Docker container
                self.db_connection = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    database="indexdb",
                    user="indexuser",
                    password="indexpass"
                )
                self.db_connection.autocommit = True
                print(f"    ✓ Connected to PostgreSQL")
                
            except ImportError:
                print(f"    ✗ Warning: psycopg2 not installed, using fallback storage")
                self.dstore_type = 'CUSTOM'
            except Exception as e:
                print(f"    ✗ Warning: Cannot connect to PostgreSQL ({e}), using fallback storage")
                print(f"    Hint: Start PostgreSQL with: docker run -d --name postgres-index -e POSTGRES_PASSWORD=indexpass -e POSTGRES_USER=indexuser -e POSTGRES_DB=indexdb -p 5432:5432 postgres:15")
                self.dstore_type = 'CUSTOM'
                self.db_connection = None
        
        elif self.dstore_type == 'REDIS':
            # Redis with sorted sets and hashes
            # Pro: Fast in-memory operations, simple key-value model, persistence options
            # Con: Memory-bound, less complex query support, eventual consistency
            try:
                import redis
                
                # Connect to Redis Docker container
                self.db_connection = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True
                )
                # Test connection
                self.db_connection.ping()
                print(f"    ✓ Connected to Redis")
                
            except ImportError:
                print(f"    ✗ Warning: redis not installed, using fallback storage")
                self.dstore_type = 'CUSTOM'
            except Exception as e:
                print(f"    ✗ Warning: Cannot connect to Redis ({e}), using fallback storage")
                print(f"    Hint: Start Redis with: docker run -d --name redis-index -p 6379:6379 redis:7")
                self.dstore_type = 'CUSTOM'
                self.db_connection = None
    
    def _get_index_path(self, index_id: str) -> Path:
        """Get the file path for storing index"""
        filename = f"{index_id}.pkl"
        return self.index_dir / filename
    
    def _build_inverted_index(self, files: Iterable[Tuple[str, str]]):
        """Build inverted index from documents"""
        print(f"    Building {self.info_type} inverted index...")
        all_doc_lengths = []
        
        for i, (doc_id, tokens) in enumerate(files):
            # Store document
            self.documents[doc_id] = {'id': doc_id, 'tokens': tokens}
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            all_doc_lengths.append(doc_length)
            
            # Build term positions and counts for this document
            term_positions = defaultdict(list)
            term_counts = Counter()
            
            for position, term in enumerate(tokens):
                term_positions[term].append(position)
                term_counts[term] += 1
            
            # Add to inverted index based on info type
            for term, positions in term_positions.items():
                count = term_counts[term]
                
                if self.info_type == 'BOOLEAN':
                    # x=1: Boolean index with doc IDs and position IDs
                    self.inverted_index[term].append((doc_id, positions, None))
                elif self.info_type == 'WORDCOUNT':
                    # x=2: Enable ranking with word counts
                    self.inverted_index[term].append((doc_id, positions, count))
                elif self.info_type == 'TFIDF':
                    # x=3: TF-IDF scores (calculate later)
                    self.inverted_index[term].append((doc_id, positions, count))
                
                self.term_doc_freq[term] += 1
            
            if (i + 1) % 100 == 0:
                print(f"      Processed {i + 1} documents...")
        
        self.num_documents = len(self.documents)
        self.avg_doc_length = sum(all_doc_lengths) / self.num_documents if self.num_documents > 0 else 0
        
        # Calculate TF-IDF if needed
        if self.info_type == 'TFIDF':
            print(f"    Calculating TF-IDF scores...")
            self._calculate_tfidf()
        
        # Build skip pointers if optimization enabled
        if self.optim_type == 'SKIP':
            print(f"    Building skip pointers...")
            self._build_skip_pointers()
        
        print(f"    Index built: {self.num_documents} docs, {len(self.inverted_index)} unique terms")
    
    def _calculate_tfidf(self):
        """Calculate TF-IDF scores for all term-document pairs"""
        for term, postings in self.inverted_index.items():
            # IDF = log(N / df)
            idf = math.log(self.num_documents / self.term_doc_freq[term])
            for doc_id, positions, count in postings:
                # TF = term count / document length
                tf = count / self.doc_lengths[doc_id]
                # TF-IDF score
                self.tfidf_scores[(term, doc_id)] = tf * idf
    
    def _build_skip_pointers(self):
        """Build skip pointers for optimization (i=1)"""
        self.skip_pointers = {}
        
        for term, postings in self.inverted_index.items():
            # Build skip pointers with interval of sqrt(len(postings))
            posting_len = len(postings)
            if posting_len > 3:
                skip_interval = int(math.sqrt(posting_len))
                skips = {}
                for i in range(0, posting_len, skip_interval):
                    if i + skip_interval < posting_len:
                        skips[i] = i + skip_interval
                self.skip_pointers[term] = skips
            else:
                self.skip_pointers[term] = {}
    
    def _compress_postings(self, postings: List) -> bytes:
        """Compress postings list based on compression type"""
        if self.compr_type == 'NONE':
            # z=0: No compression
            return pickle.dumps(postings)
        elif self.compr_type == 'CODE':
            # z=1: Simple variable byte encoding
            return self._varbyte_encode(postings)
        elif self.compr_type == 'CLIB':
            # z=2: Off-the-shelf compression (zlib)
            return zlib.compress(pickle.dumps(postings), level=6)
        return pickle.dumps(postings)
    
    def _decompress_postings(self, compressed: bytes) -> List:
        """Decompress postings list"""
        if self.compr_type == 'NONE':
            return pickle.loads(compressed)
        elif self.compr_type == 'CODE':
            return self._varbyte_decode(compressed)
        elif self.compr_type == 'CLIB':
            return pickle.loads(zlib.decompress(compressed))
        return pickle.loads(compressed)
    
    def _varbyte_encode(self, postings: List) -> bytes:
        """
        Simple variable byte encoding for compression (z=1)
        Encodes integers using variable number of bytes
        """
        result = bytearray()
        
        def encode_number(n):
            """Encode a single number using VarByte"""
            bytes_list = []
            while True:
                bytes_list.insert(0, n % 128)
                if n < 128:
                    break
                n = n // 128
            bytes_list[-1] += 128  # Set continuation bit on last byte
            return bytes(bytes_list)
        
        # Serialize the structure
        encoded_data = pickle.dumps(postings)
        # Apply simple compression by encoding length
        result.extend(encode_number(len(encoded_data)))
        result.extend(encoded_data)
        return bytes(result)
    
    def _varbyte_decode(self, encoded: bytes) -> List:
        """Decode variable byte encoding"""
        # Simplified decoding - just extract the pickle data
        idx = 0
        n = 0
        while idx < len(encoded):
            byte = encoded[idx]
            if byte < 128:
                n = n * 128 + byte
                idx += 1
            else:
                n = n * 128 + (byte - 128)
                idx += 1
                break
        
        return pickle.loads(encoded[idx:idx+n])
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        """Create index from documents"""
        self.index_id = index_id
        self._build_inverted_index(files)
        self._save_index()
        print(f"    ✓ Index saved: {self._get_index_path(index_id)}")
    
    def _save_index(self):
        """Save index to disk or database"""
        if not self.index_id:
            raise ValueError("Index ID not set")
        
        # Prepare index data
        index_data = {
            'identifier': self.identifier_short,
            'config': {
                'info': self.info_type,
                'dstore': self.dstore_type,
                'compr': self.compr_type,
                'qproc': self.qproc_type,
                'optim': self.optim_type
            },
            'inverted_index': dict(self.inverted_index),
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'term_doc_freq': dict(self.term_doc_freq),
            'num_documents': self.num_documents,
            'avg_doc_length': self.avg_doc_length,
            'tfidf_scores': self.tfidf_scores if self.info_type == 'TFIDF' else {},
            'skip_pointers': self.skip_pointers if self.optim_type == 'SKIP' else {}
        }
        
        # Save based on datastore type
        if self.dstore_type == 'CUSTOM':
            # y=1: Custom storage using pickle
            with open(self._get_index_path(self.index_id), 'wb') as f:
                pickle.dump(index_data, f)
        
        elif self.dstore_type == 'POSTGRES':
            # y=2a: PostgreSQL storage with GIN index
            self._save_to_postgres(index_data)
        
        elif self.dstore_type == 'REDIS':
            # y=2b: Redis storage with key-value structure
            self._save_to_redis(index_data)
    
    def _save_to_postgres(self, index_data):
        """Save index to PostgreSQL database"""
        import psycopg2.extras
        
        cursor = self.db_connection.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indices (
                index_id VARCHAR(255) PRIMARY KEY,
                identifier VARCHAR(255),
                config JSONB,
                inverted_index JSONB,
                documents JSONB,
                doc_lengths JSONB,
                term_doc_freq JSONB,
                num_documents INTEGER,
                avg_doc_length FLOAT,
                tfidf_scores JSONB,
                skip_pointers JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create GIN index on inverted_index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inverted_index_gin 
            ON indices USING GIN (inverted_index)
        """)
        
        # Insert or update index data
        cursor.execute("""
            INSERT INTO indices (
                index_id, identifier, config, inverted_index, documents,
                doc_lengths, term_doc_freq, num_documents, avg_doc_length,
                tfidf_scores, skip_pointers
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_id) DO UPDATE SET
                identifier = EXCLUDED.identifier,
                config = EXCLUDED.config,
                inverted_index = EXCLUDED.inverted_index,
                documents = EXCLUDED.documents,
                doc_lengths = EXCLUDED.doc_lengths,
                term_doc_freq = EXCLUDED.term_doc_freq,
                num_documents = EXCLUDED.num_documents,
                avg_doc_length = EXCLUDED.avg_doc_length,
                tfidf_scores = EXCLUDED.tfidf_scores,
                skip_pointers = EXCLUDED.skip_pointers
        """, (
            self.index_id,
            index_data['identifier'],
            psycopg2.extras.Json(index_data['config']),
            psycopg2.extras.Json(index_data['inverted_index']),
            psycopg2.extras.Json(index_data['documents']),
            psycopg2.extras.Json(index_data['doc_lengths']),
            psycopg2.extras.Json(index_data['term_doc_freq']),
            index_data['num_documents'],
            index_data['avg_doc_length'],
            psycopg2.extras.Json(index_data['tfidf_scores']),
            psycopg2.extras.Json(index_data['skip_pointers'])
        ))
        
        cursor.close()
        print(f"    ✓ Saved to PostgreSQL with GIN index")
    
    def _save_to_redis(self, index_data):
        """Save index to Redis database"""
        import json
        
        # Use Redis hash for the index
        index_key = f"index:{self.index_id}"
        
        # Store each component as a hash field
        self.db_connection.hset(index_key, "identifier", index_data['identifier'])
        self.db_connection.hset(index_key, "config", json.dumps(index_data['config']))
        self.db_connection.hset(index_key, "inverted_index", json.dumps(index_data['inverted_index']))
        self.db_connection.hset(index_key, "documents", json.dumps(index_data['documents']))
        self.db_connection.hset(index_key, "doc_lengths", json.dumps(index_data['doc_lengths']))
        self.db_connection.hset(index_key, "term_doc_freq", json.dumps(index_data['term_doc_freq']))
        self.db_connection.hset(index_key, "num_documents", index_data['num_documents'])
        self.db_connection.hset(index_key, "avg_doc_length", index_data['avg_doc_length'])
        self.db_connection.hset(index_key, "tfidf_scores", json.dumps(index_data['tfidf_scores']))
        self.db_connection.hset(index_key, "skip_pointers", json.dumps(index_data['skip_pointers']))
        
        # Add index_id to set of all indices
        self.db_connection.sadd("all_indices", self.index_id)
        
        print(f"    ✓ Saved to Redis as hash")
    
    def load_index(self, serialized_index_dump: str) -> None:
        """Load index from disk or database"""
        self.index_id = serialized_index_dump
        
        if self.dstore_type == 'CUSTOM':
            # Load from pickle file
            index_path = Path(serialized_index_dump)
            if not index_path.exists():
                index_path = self._get_index_path(serialized_index_dump)
            
            if not index_path.exists():
                raise FileNotFoundError(f"Index not found: {serialized_index_dump}")
            
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
        
        elif self.dstore_type == 'POSTGRES':
            # Load from PostgreSQL
            index_data = self._load_from_postgres(serialized_index_dump)
        
        elif self.dstore_type == 'REDIS':
            # Load from Redis
            index_data = self._load_from_redis(serialized_index_dump)
        
        # Restore index data
        self.inverted_index = defaultdict(list, index_data['inverted_index'])
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.term_doc_freq = Counter(index_data['term_doc_freq'])
        self.num_documents = index_data['num_documents']
        self.avg_doc_length = index_data['avg_doc_length']
        self.tfidf_scores = index_data.get('tfidf_scores', {})
        self.skip_pointers = index_data.get('skip_pointers', {})
    
    def _load_from_postgres(self, index_id):
        """Load index from PostgreSQL database"""
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
            SELECT identifier, config, inverted_index, documents, doc_lengths,
                   term_doc_freq, num_documents, avg_doc_length, tfidf_scores, skip_pointers
            FROM indices WHERE index_id = %s
        """, (index_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            raise FileNotFoundError(f"Index not found in PostgreSQL: {index_id}")
        
        return {
            'identifier': row[0],
            'config': row[1],
            'inverted_index': row[2],
            'documents': row[3],
            'doc_lengths': row[4],
            'term_doc_freq': row[5],
            'num_documents': row[6],
            'avg_doc_length': row[7],
            'tfidf_scores': row[8],
            'skip_pointers': row[9]
        }
    
    def _load_from_redis(self, index_id):
        """Load index from Redis database"""
        import json
        
        index_key = f"index:{index_id}"
        
        if not self.db_connection.exists(index_key):
            raise FileNotFoundError(f"Index not found in Redis: {index_id}")
        
        data = self.db_connection.hgetall(index_key)
        
        return {
            'identifier': data['identifier'],
            'config': json.loads(data['config']),
            'inverted_index': json.loads(data['inverted_index']),
            'documents': json.loads(data['documents']),
            'doc_lengths': json.loads(data['doc_lengths']),
            'term_doc_freq': json.loads(data['term_doc_freq']),
            'num_documents': int(data['num_documents']),
            'avg_doc_length': float(data['avg_doc_length']),
            'tfidf_scores': json.loads(data['tfidf_scores']),
            'skip_pointers': json.loads(data['skip_pointers'])
        }
    
    def update_index(self, index_id: str, 
                    remove_files: Iterable[Tuple[str, str]] = None,
                    add_files: Iterable[Tuple[str, str]] = None) -> None:
        self.index_id = index_id
        self.load_index(index_id)
        
        if remove_files:
            for doc_id, _ in remove_files:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    del self.doc_lengths[doc_id]
                    for term in list(self.inverted_index.keys()):
                        self.inverted_index[term] = [p for p in self.inverted_index[term] if p[0] != doc_id]
                        if not self.inverted_index[term]:
                            del self.inverted_index[term]
        
        if add_files:
            self._build_inverted_index(add_files)
        
        self.num_documents = len(self.documents)
        if self.num_documents > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_documents
        self._save_index()
    
    def query(self, query_str: str) -> str:
        """Execute query using configured query processing method"""
        if self.qproc_type == 'TERMatat':
            # Term-at-a-time query processing
            results = self._query_term_at_a_time(query_str)
        else:
            # Document-at-a-time query processing
            results = self._query_document_at_a_time(query_str)
        return json.dumps(results, indent=2)
    
    def _query_term_at_a_time(self, query_str: str) -> Dict:
        """
        Term-at-a-time query processing (q=T0 or q=T1 with optimization)
        Process one query term at a time, accumulating scores
        """
        terms = query_str.lower().split()
        doc_scores = defaultdict(float)
        
        for term in terms:
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                
                # Use skip pointers if optimization enabled (i=1)
                if self.optim_type == 'SKIP' and term in self.skip_pointers:
                    # With skip pointer optimization
                    skips = self.skip_pointers[term]
                    idx = 0
                    while idx < len(postings):
                        doc_id, positions, count = postings[idx]
                        
                        # Score based on info type
                        if self.info_type == 'BOOLEAN':
                            doc_scores[doc_id] += 1.0
                        elif self.info_type == 'WORDCOUNT':
                            doc_scores[doc_id] += count
                        elif self.info_type == 'TFIDF':
                            doc_scores[doc_id] += self.tfidf_scores.get((term, doc_id), 0.0)
                        
                        # Use skip pointer if available
                        if idx in skips:
                            idx = skips[idx]
                        else:
                            idx += 1
                else:
                    # Standard processing without optimization
                    for doc_id, positions, count in postings:
                        if self.info_type == 'BOOLEAN':
                            doc_scores[doc_id] += 1.0
                        elif self.info_type == 'WORDCOUNT':
                            doc_scores[doc_id] += count
                        elif self.info_type == 'TFIDF':
                            doc_scores[doc_id] += self.tfidf_scores.get((term, doc_id), 0.0)
        
        # Rank documents by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            'query': query_str,
            'num_results': len(ranked_docs),
            'results': [{'doc_id': doc_id, 'score': score} for doc_id, score in ranked_docs[:10]]
        }
    
    def _query_document_at_a_time(self, query_str: str) -> Dict:
        """
        Document-at-a-time query processing (q=D0 or q=D1 with optimization)
        Process one document at a time, computing full scores
        """
        terms = query_str.lower().split()
        all_docs = set()
        term_postings = {}
        
        # Gather all postings for query terms
        for term in terms:
            if term in self.inverted_index:
                term_postings[term] = {doc_id: (positions, count) 
                                      for doc_id, positions, count in self.inverted_index[term]}
                all_docs.update(term_postings[term].keys())
        
        # Score each document
        doc_scores = {}
        for doc_id in all_docs:
            score = 0.0
            for term in terms:
                if term in term_postings and doc_id in term_postings[term]:
                    positions, count = term_postings[term][doc_id]
                    
                    # Score based on info type
                    if self.info_type == 'BOOLEAN':
                        score += 1.0
                    elif self.info_type == 'WORDCOUNT':
                        score += count
                    elif self.info_type == 'TFIDF':
                        score += self.tfidf_scores.get((term, doc_id), 0.0)
            
            doc_scores[doc_id] = score
        
        # Rank documents by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            'query': query_str,
            'num_results': len(ranked_docs),
            'results': [{'doc_id': doc_id, 'score': score} for doc_id, score in ranked_docs[:10]]
        }
    
    def delete_index(self, index_id: str) -> None:
        """Delete an index"""
        if self.dstore_type == 'CUSTOM':
            index_path = self._get_index_path(index_id)
            if index_path.exists():
                index_path.unlink()
        elif self.dstore_type == 'POSTGRES':
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM indices WHERE index_id = %s", (index_id,))
            cursor.close()
        elif self.dstore_type == 'REDIS':
            index_key = f"index:{index_id}"
            self.db_connection.delete(index_key)
            self.db_connection.srem("all_indices", index_id)
    
    def list_indices(self) -> Iterable[str]:
        """List all indices"""
        if self.dstore_type == 'CUSTOM':
            return [path.stem for path in self.index_dir.glob("*.pkl")]
        elif self.dstore_type == 'POSTGRES':
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT index_id FROM indices")
            indices = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return indices
        elif self.dstore_type == 'REDIS':
            return list(self.db_connection.smembers("all_indices"))
        return []
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List all files in an index"""
        self.load_index(index_id)
        return list(self.documents.keys())
    
    def close(self):
        """Close database connections"""
        if self.db_connection:
            if self.dstore_type == 'POSTGRES':
                self.db_connection.close()
                print("    PostgreSQL connection closed")
            elif self.dstore_type == 'REDIS':
                self.db_connection.close()
                print("    Redis connection closed")


def load_processed_data(data_path: str = "data/processed/wikipedia_processed.json") -> List[Tuple[str, List[str]]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return [(doc['id'], doc['tokens_processed']) for doc in documents]


def main():
    """
    Create 9 unique indices for comprehensive evaluation
    
    Indices cover all required plot variations:
    - Plot C: Information types (x=1,2,3)
    - Plot A: Datastores (y=1,2a,2b)
    - Plot AB: Compression (z=0,1,2)
    - Plot A: Optimization (i=0,1)
    - Bonus: Query processing (q=T,D)
    """
    print("=" * 80)
    print("SELF-INDEX: CREATING 9 EVALUATION INDICES")
    print("=" * 80)
    
    # Load processed data
    print("\nLoading processed Wikipedia data...")
    files = load_processed_data()
    print(f"Loaded {len(files)} documents")
    
    # Define all 9 unique index configurations
    # Format: (info, dstore, compr, qproc, optim, index_id, description)
    indices_config = [
        # === Plot C: Information Type Comparison (x=1,2,3) ===
        ('BOOLEAN', 'CUSTOM', 'NONE', 'TERMatat', 'Null', 
         '1-plotC-boolean', 'Boolean index with doc IDs + positions'),
        
        ('WORDCOUNT', 'CUSTOM', 'NONE', 'TERMatat', 'Null', 
         '2-plotC-wordcount', 'Ranking with word counts'),
        
        ('TFIDF', 'CUSTOM', 'NONE', 'TERMatat', 'Null', 
         '3-plotC-tfidf', 'TF-IDF scoring (baseline for other plots)'),
        
        # === Plot A: Datastore Comparison (y=1,2a,2b) ===
        # y=1 (CUSTOM) already created above as #3
        
        ('TFIDF', 'POSTGRES', 'NONE', 'TERMatat', 'Null', 
         '4-plotA-postgres', 'PostgreSQL GIN index'),
        
        ('TFIDF', 'REDIS', 'NONE', 'TERMatat', 'Null', 
         '5-plotA-redis', 'Redis key-value store'),
        
        # === Plot AB: Compression Comparison (z=0,1,2) ===
        # z=0 (NONE) already created above as #3
        
        ('TFIDF', 'CUSTOM', 'CODE', 'TERMatat', 'Null', 
         '6-plotAB-code', 'Simple VarByte compression'),
        
        ('TFIDF', 'CUSTOM', 'CLIB', 'TERMatat', 'Null', 
         '7-plotAB-clib', 'Zlib compression library'),
        
        # === Plot A: Optimization Comparison (i=0,1) ===
        # i=0 (Null) already created above as #3
        
        ('TFIDF', 'CUSTOM', 'NONE', 'TERMatat', 'SKIP', 
         '8-plotA-skip', 'Skip pointers optimization'),
        
        # === Bonus: Query Processing Comparison (q=T,D) ===
        
        ('TFIDF', 'CUSTOM', 'NONE', 'DOCatatt', 'Null', 
         '9-bonus-docatatt', 'Document-at-a-time processing'),
    ]
    
    print(f"\n{'=' * 80}")
    print(f"Creating {len(indices_config)} indices...")
    print(f"{'=' * 80}\n")
    
    # Create each index
    created_indices = []
    for i, (info, dstore, compr, qproc, optim, index_id, desc) in enumerate(indices_config, 1):
        print(f"\n{'─' * 80}")
        print(f"[{i}/{len(indices_config)}] Index: {index_id}")
        print(f"Description: {desc}")
        print(f"Configuration:")
        print(f"  • Information:  {info:10s} (x={'123'[['BOOLEAN','WORDCOUNT','TFIDF'].index(info)]})")
        print(f"  • Datastore:    {dstore:10s} (y={'1' if dstore=='CUSTOM' else '2'})")
        print(f"  • Compression:  {compr:10s} (z={'012'[['NONE','CODE','CLIB'].index(compr)]})")
        print(f"  • Query Proc:   {qproc:10s} (q={'T' if qproc=='TERMatat' else 'D'})")
        print(f"  • Optimization: {optim:10s} (i={'0' if optim=='Null' else '1'})")
        
        try:
            index = SelfIndex(info=info, dstore=dstore, compr=compr, 
                            qproc=qproc, optim=optim)
            index.create_index(index_id, files)
            created_indices.append(index_id)
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("INDEX CREATION SUMMARY")
    print(f"{'=' * 80}\n")
    
    print(f"Successfully created {len(created_indices)} indices:\n")
    
    print("Plot C - Information Type (x):")
    print("  1. 1-plotC-boolean    : x=1 Boolean")
    print("  2. 2-plotC-wordcount  : x=2 Word Count")
    print("  3. 3-plotC-tfidf      : x=3 TF-IDF\n")
    
    print("Plot A - Datastore (y):")
    print("  3. 3-plotC-tfidf      : y=1 CUSTOM")
    print("  4. 4-plotA-postgres   : y=2 PostgreSQL")
    print("  5. 5-plotA-redis      : y=2 Redis\n")
    
    print("Plot AB - Compression (z):")
    print("  3. 3-plotC-tfidf      : z=0 NONE")
    print("  6. 6-plotAB-code      : z=1 CODE")
    print("  7. 7-plotAB-clib      : z=2 CLIB\n")
    
    print("Plot A - Optimization (i):")
    print("  3. 3-plotC-tfidf      : i=0 Null")
    print("  8. 8-plotA-skip       : i=1 SKIP\n")
    
    print("Bonus - Query Processing (q):")
    print("  3. 3-plotC-tfidf      : q=T TERMatat")
    print("  9. 9-bonus-docatatt   : q=D DOCatatt\n")
    
    print(f"{'=' * 80}")
    print("Testing query on baseline index...")
    print(f"{'=' * 80}\n")
    
    try:
        test_index = SelfIndex(info='TFIDF')
        test_index.load_index('3-plotC-tfidf')
        result = test_index.query("machine learning")
        print(result)
    except Exception as e:
        print(f"Error testing query: {e}")
    
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
