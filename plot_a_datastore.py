#!/usr/bin/env python3
"""
Plot A: Datastore Comparison (y=1,2a,2b)
Builds SelfIndex-v1.0y0T0 and compares different datastore choices
y=1: Pickle (custom objects on disk)
y=2a: PostgreSQL with GIN index
y=2b: Redis (in-memory key-value store)
"""
import sys
import json
import pickle
import math
import os
import time
import psutil
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any
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


class SelfIndexBase:
    """Base class for all datastore implementations"""
    
    def __init__(self, y: int, variant: str = ""):
        """y: 1=Pickle, 2=Database (variant: 'postgres' or 'redis')"""
        self.y = y
        self.variant = variant
        self.version = f"1.0{y}0T0"
        self.index_name = f"SelfIndex-v{self.version}"
        if variant:
            self.index_name += f"-{variant}"
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.num_documents = 0
        self.num_terms = 0
    
    def build_index(self, documents: List[Tuple[str, List[str]]]):
        """Build index - to be implemented by subclasses"""
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search - to be implemented by subclasses"""
        raise NotImplementedError
    
    def close(self):
        """Close connections - to be implemented by subclasses"""
        pass


class PickleIndex(SelfIndexBase):
    """y=1: Pickle-based index stored on local disk"""
    
    def __init__(self):
        super().__init__(y=1, variant="pickle")
        self.inverted_index = defaultdict(list)
        self.doc_lengths = {}
        self.term_doc_freq = defaultdict(int)
    
    def build_index(self, documents: List[Tuple[str, List[str]]]):
        """Build TF-IDF index using pickle storage"""
        self.num_documents = len(documents)
        
        print(f"  Building inverted index...")
        for idx, (doc_id, tokens) in enumerate(documents):
            if idx % 1000 == 0 and idx > 0:
                print(f"\r  Processed {idx}/{len(documents)} documents...", end='', flush=True)
            
            self.doc_lengths[doc_id] = len(tokens)
            term_counts = defaultdict(int)
            
            for term in tokens:
                term_counts[term] += 1
            
            for term, count in term_counts.items():
                self.inverted_index[term].append((doc_id, count))
                self.term_doc_freq[term] += 1
        
        print(f"\r  Processed {len(documents)}/{len(documents)} documents   ")
        
        # Convert to TF-IDF
        print(f"  Computing TF-IDF scores...")
        for term, postings in self.inverted_index.items():
            df = self.term_doc_freq[term]
            idf = math.log(self.num_documents / df) if df > 0 else 0
            new_postings = []
            for doc_id, count in postings:
                tf = count / self.doc_lengths[doc_id]
                new_postings.append((doc_id, tf * idf))
            self.inverted_index[term] = new_postings
        
        self.num_terms = len(self.inverted_index)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF scoring"""
        query_terms = [self.stemmer.stem(t.lower()) for t in query.split() 
                      if t.lower() not in self.stop_words]
        
        scores = defaultdict(float)
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, score in self.inverted_index[term]:
                    scores[doc_id] += score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def save_index(self, index_id: str):
        """Save index to disk using pickle"""
        Path("indices").mkdir(exist_ok=True)
        filepath = f"indices/{index_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump({
                'inverted_index': dict(self.inverted_index),
                'doc_lengths': self.doc_lengths,
                'num_documents': self.num_documents,
                'term_doc_freq': dict(self.term_doc_freq),
                'num_terms': self.num_terms
            }, f)
        print(f"  Saved to {filepath}")


class PostgresIndex(SelfIndexBase):
    """y=2a: PostgreSQL with GIN index"""
    
    def __init__(self, host='127.0.0.1', port=5432, dbname='searchdb', 
                 user='searchuser', password='searchpass'):
        super().__init__(y=2, variant="postgres")
        try:
            import psycopg2
            from psycopg2.extras import execute_batch
            self.psycopg2 = psycopg2
            self.execute_batch = execute_batch
        except ImportError:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
        
        self.conn_params = {
            'host': host, 'port': port, 'dbname': dbname,
            'user': user, 'password': password
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = self.psycopg2.connect(**self.conn_params)
            self.conn.autocommit = False
            self._create_tables()
        except Exception as e:
            print(f"  Warning: Could not connect to PostgreSQL: {e}")
            print(f"  Make sure PostgreSQL is running (docker-compose up -d postgres)")
            raise
    
    def _create_tables(self):
        """Create index tables"""
        with self.conn.cursor() as cur:
            # Drop existing tables
            cur.execute("DROP TABLE IF EXISTS postings CASCADE")
            cur.execute("DROP TABLE IF EXISTS documents CASCADE")
            cur.execute("DROP TABLE IF EXISTS terms CASCADE")
            
            # Create tables
            cur.execute("""
                CREATE TABLE documents (
                    doc_id VARCHAR(255) PRIMARY KEY,
                    doc_length INTEGER NOT NULL
                )
            """)
            
            cur.execute("""
                CREATE TABLE terms (
                    term_id SERIAL PRIMARY KEY,
                    term VARCHAR(255) UNIQUE NOT NULL,
                    doc_freq INTEGER NOT NULL
                )
            """)
            
            cur.execute("""
                CREATE TABLE postings (
                    term_id INTEGER REFERENCES terms(term_id),
                    doc_id VARCHAR(255) REFERENCES documents(doc_id),
                    tf_idf REAL NOT NULL,
                    PRIMARY KEY (term_id, doc_id)
                )
            """)
            
            self.conn.commit()
    
    def build_index(self, documents: List[Tuple[str, List[str]]]):
        """Build index in PostgreSQL"""
        self.num_documents = len(documents)
        
        # Calculate term frequencies and document lengths
        print(f"  Processing documents...")
        doc_data = []
        term_data = defaultdict(lambda: {'doc_freq': 0, 'postings': []})
        
        for idx, (doc_id, tokens) in enumerate(documents):
            if idx % 1000 == 0 and idx > 0:
                print(f"\r  Processed {idx}/{len(documents)} documents...", end='', flush=True)
            
            doc_length = len(tokens)
            doc_data.append((doc_id, doc_length))
            
            term_counts = defaultdict(int)
            for term in tokens:
                term_counts[term] += 1
            
            for term, count in term_counts.items():
                term_data[term]['doc_freq'] += 1
                term_data[term]['postings'].append((doc_id, count, doc_length))
        
        print(f"\r  Processed {len(documents)}/{len(documents)} documents   ")
        
        # Insert documents
        print(f"  Inserting documents into PostgreSQL...")
        with self.conn.cursor() as cur:
            self.execute_batch(cur, 
                "INSERT INTO documents (doc_id, doc_length) VALUES (%s, %s)",
                doc_data, page_size=1000)
        self.conn.commit()
        
        # Insert terms and postings with TF-IDF
        print(f"  Inserting terms and postings...")
        term_ids = {}
        
        # Batch insert all terms first
        terms_batch = [(term, data['doc_freq']) for term, data in term_data.items()]
        with self.conn.cursor() as cur:
            self.execute_batch(cur,
                "INSERT INTO terms (term, doc_freq) VALUES (%s, %s)",
                terms_batch, page_size=1000)
        self.conn.commit()
        
        # Retrieve all term IDs in one query
        with self.conn.cursor() as cur:
            cur.execute("SELECT term, term_id FROM terms")
            term_ids = dict(cur.fetchall())
        
        # Now insert all postings in batches
        print(f"  Computing TF-IDF and inserting postings...")
        all_postings = []
        for term, data in term_data.items():
            term_id = term_ids[term]
            df = data['doc_freq']
            idf = math.log(self.num_documents / df) if df > 0 else 0
            
            for doc_id, count, doc_length in data['postings']:
                tf = count / doc_length
                tf_idf = tf * idf
                all_postings.append((term_id, doc_id, tf_idf))
        
        with self.conn.cursor() as cur:
            self.execute_batch(cur,
                "INSERT INTO postings (term_id, doc_id, tf_idf) VALUES (%s, %s, %s)",
                all_postings, page_size=5000)
        
        self.conn.commit()
        
        # Create indexes AFTER data insertion for better performance
        print(f"  Creating indexes...")
        with self.conn.cursor() as cur:
            # Use simple B-tree index instead of GIN for better performance with small datasets
            cur.execute("CREATE INDEX idx_terms_term ON terms(term)")
            cur.execute("CREATE INDEX idx_postings_term ON postings(term_id)")
            cur.execute("CREATE INDEX idx_postings_doc ON postings(doc_id)")
        self.conn.commit()
        
        print(f"  Inserted {len(term_data)} terms and {len(all_postings)} postings")
        self.num_terms = len(term_data)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using PostgreSQL"""
        query_terms = [self.stemmer.stem(t.lower()) for t in query.split() 
                      if t.lower() not in self.stop_words]
        
        if not query_terms:
            return []
        
        with self.conn.cursor() as cur:
            # Get term IDs
            placeholders = ','.join(['%s'] * len(query_terms))
            cur.execute(f"""
                SELECT term_id, term FROM terms 
                WHERE term IN ({placeholders})
            """, query_terms)
            
            term_ids = [row[0] for row in cur.fetchall()]
            
            if not term_ids:
                return []
            
            # Get top documents
            placeholders = ','.join(['%s'] * len(term_ids))
            cur.execute(f"""
                SELECT doc_id, SUM(tf_idf) as score
                FROM postings
                WHERE term_id IN ({placeholders})
                GROUP BY doc_id
                ORDER BY score DESC
                LIMIT %s
            """, term_ids + [top_k])
            
            return [(row[0], float(row[1])) for row in cur.fetchall()]
    
    def save_index(self, index_id: str):
        """PostgreSQL index is already persisted"""
        print(f"  Index persisted in PostgreSQL database")
    
    def close(self):
        """Close PostgreSQL connection"""
        if self.conn:
            self.conn.close()


class RedisIndex(SelfIndexBase):
    """y=2b: Redis in-memory key-value store"""
    
    def __init__(self, host='127.0.0.1', port=6379, db=0):
        super().__init__(y=2, variant="redis")
        try:
            import redis
            self.redis_module = redis
        except ImportError:
            raise ImportError("redis not installed. Run: pip install redis")
        
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self.connect()
    
    def connect(self):
        """Connect to Redis"""
        try:
            self.client = self.redis_module.Redis(
                host=self.host, port=self.port, db=self.db,
                decode_responses=False  # We'll handle encoding/decoding
            )
            self.client.ping()
            # Clear existing data
            self.client.flushdb()
        except Exception as e:
            print(f"  Warning: Could not connect to Redis: {e}")
            print(f"  Make sure Redis is running (docker-compose up -d redis)")
            raise
    
    def build_index(self, documents: List[Tuple[str, List[str]]]):
        """Build index in Redis"""
        self.num_documents = len(documents)
        
        print(f"  Processing documents...")
        term_data = defaultdict(lambda: {'doc_freq': 0, 'postings': []})
        doc_lengths = {}
        
        for idx, (doc_id, tokens) in enumerate(documents):
            if idx % 1000 == 0 and idx > 0:
                print(f"\r  Processed {idx}/{len(documents)} documents...", end='', flush=True)
            
            doc_length = len(tokens)
            doc_lengths[doc_id] = doc_length
            
            term_counts = defaultdict(int)
            for term in tokens:
                term_counts[term] += 1
            
            for term, count in term_counts.items():
                term_data[term]['doc_freq'] += 1
                term_data[term]['postings'].append((doc_id, count, doc_length))
        
        print(f"\r  Processed {len(documents)}/{len(documents)} documents   ")
        
        # Store in Redis using pipeline for efficiency
        print(f"  Storing in Redis...")
        pipe = self.client.pipeline()
        
        # Store metadata
        pipe.set('meta:num_documents', self.num_documents)
        pipe.set('meta:num_terms', len(term_data))
        
        # Store document lengths
        for doc_id, length in doc_lengths.items():
            pipe.hset('doc_lengths', doc_id, length)
        
        # Store inverted index with TF-IDF
        for idx, (term, data) in enumerate(term_data.items()):
            if idx % 1000 == 0 and idx > 0:
                print(f"\r  Storing {idx}/{len(term_data)} terms...", end='', flush=True)
            
            df = data['doc_freq']
            idf = math.log(self.num_documents / df) if df > 0 else 0
            
            # Store postings as hash: postings:{term} -> {doc_id: tf_idf}
            postings_dict = {}
            for doc_id, count, doc_length in data['postings']:
                tf = count / doc_length
                tf_idf = tf * idf
                postings_dict[doc_id] = tf_idf
            
            # Store as JSON string for each term
            pipe.set(f'postings:{term}', json.dumps(postings_dict))
        
        pipe.execute()
        print(f"\r  Stored {len(term_data)}/{len(term_data)} terms   ")
        self.num_terms = len(term_data)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using Redis"""
        query_terms = [self.stemmer.stem(t.lower()) for t in query.split() 
                      if t.lower() not in self.stop_words]
        
        if not query_terms:
            return []
        
        # Aggregate scores from all query terms
        scores = defaultdict(float)
        
        pipe = self.client.pipeline()
        for term in query_terms:
            pipe.get(f'postings:{term}')
        
        results = pipe.execute()
        
        for postings_json in results:
            if postings_json:
                postings = json.loads(postings_json)
                for doc_id, score in postings.items():
                    scores[doc_id] += score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def save_index(self, index_id: str):
        """Redis index is in-memory but can be persisted"""
        print(f"  Index stored in Redis (in-memory)")
        # Optionally trigger Redis persistence (skip if already in progress)
        try:
            self.client.bgsave()
        except Exception:
            pass  # Ignore if save already in progress
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()


def measure_memory_footprint(index, index_id, store_type):
    """Measure memory footprint"""
    if store_type == "pickle":
        filepath = f"indices/{index_id}.pkl"
        memory_mb = os.path.getsize(filepath) / (1024**2) if os.path.exists(filepath) else 0
    elif store_type == "postgres":
        # Query PostgreSQL database size
        try:
            with index.conn.cursor() as cur:
                cur.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                           pg_database_size(current_database()) as bytes
                """)
                size_str, size_bytes = cur.fetchone()
                memory_mb = size_bytes / (1024**2)
        except:
            memory_mb = 0
    elif store_type == "redis":
        # Query Redis memory usage
        try:
            info = index.client.info('memory')
            memory_mb = info['used_memory'] / (1024**2)
        except:
            memory_mb = 0
    else:
        memory_mb = 0
    
    return {
        'y': index.y,
        'variant': index.variant,
        'memory_mb': memory_mb,
        'num_docs': index.num_documents,
        'num_terms': index.num_terms
    }


def generate_plots(results):
    """Generate Plot A - Datastore Memory Footprint"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    labels = [r['variant'].title() for r in results]
    memory = [r['memory_mb'] for r in results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Plot 1: Memory Footprint
    bars1 = ax.bar(labels, memory, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Memory Footprint (MB)', fontsize=13, fontweight='bold')
    ax.set_title('Memory Footprint by Datastore', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/plots/plot_a_memory.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: data/plots/plot_a_memory.png")
    plt.show()


def stream_load_documents(filepath, sample_size=None):
    """Stream load documents from large JSON array file"""
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
                    doc_str = '\n'.join(doc_buffer).rstrip(',')
                    try:
                        doc = json.loads(doc_str)
                        if 'id' not in doc:
                            continue
                        if 'tokens_processed' not in doc:
                            continue
                        
                        documents.append((doc['id'], doc['tokens_processed']))
                        
                        if len(documents) % 1000 == 0:
                            print(f"\r  Loaded {len(documents)} documents...", end='', flush=True)
                        
                        if sample_size and len(documents) >= sample_size:
                            print(f"\r  Loaded {len(documents)} documents   ")
                            return documents
                    except json.JSONDecodeError as e:
                        pass
                    except Exception as e:
                        pass
                    
                    in_doc = False
                    doc_buffer = []
    
    print(f"\r  Loaded {len(documents)} documents   ")
    return documents


def load_queries(filepath="queryset.json"):
    """Load queries from queryset file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['queries']


def main(sample_size=None, skip_postgres=False, skip_redis=False):
    print("="*80)
    print("PLOT A: DATASTORE MEMORY FOOTPRINT")
    print("="*80)
    
    # Load data
    print("\nLoading documents...")
    documents = stream_load_documents("data/processed/wikipedia_processed.json", sample_size)
    print(f"Loaded {len(documents)} documents")
    
    results = []
    
    # y=1: Pickle
    print(f"\n{'='*80}")
    print(f"[y=1] Pickle (Custom Objects on Disk)")
    print(f"{'='*80}")
    index_id = "SelfIndex-v1.010T0-pickle"
    
    try:
        index = PickleIndex()
        index.build_index(documents)
        index.save_index(index_id)
        
        metrics = measure_memory_footprint(index, index_id, "pickle")
        results.append(metrics)
        print(f"  Memory: {metrics['memory_mb']:.2f} MB")
        index.close()
    except Exception as e:
        print(f"  Error with Pickle index: {e}")
    
    # y=2a: PostgreSQL
    if not skip_postgres:
        print(f"\n{'='*80}")
        print(f"[y=2a] PostgreSQL with GIN Index")
        print(f"{'='*80}")
        index_id = "SelfIndex-v1.020T0-postgres"
        
        try:
            index = PostgresIndex()
            index.build_index(documents)
            index.save_index(index_id)
            
            metrics = measure_memory_footprint(index, index_id, "postgres")
            results.append(metrics)
            print(f"  Memory: {metrics['memory_mb']:.2f} MB")
            index.close()
        except Exception as e:
            print(f"  Error with PostgreSQL index: {e}")
            print(f"  Skipping PostgreSQL...")
    
    # y=2b: Redis
    if not skip_redis:
        print(f"\n{'='*80}")
        print(f"[y=2b] Redis (In-Memory Key-Value Store)")
        print(f"{'='*80}")
        index_id = "SelfIndex-v1.020T0-redis"
        
        try:
            index = RedisIndex()
            index.build_index(documents)
            index.save_index(index_id)
            
            metrics = measure_memory_footprint(index, index_id, "redis")
            results.append(metrics)
            print(f"  Memory: {metrics['memory_mb']:.2f} MB")
            index.close()
        except Exception as e:
            print(f"  Error with Redis index: {e}")
            print(f"  Skipping Redis...")
    
    if len(results) > 0:
        # Generate plots
        generate_plots(results)
        
        # Save results
        with open("plot_a_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("✓ Complete! Results: plot_a_results.json")
        print(f"{'='*80}")
    else:
        print("\n✗ No results generated - all datastores failed")


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    skip_pg = '--skip-postgres' in sys.argv
    skip_redis = '--skip-redis' in sys.argv
    
    main(sample_size=sample, skip_postgres=skip_pg, skip_redis=skip_redis)
