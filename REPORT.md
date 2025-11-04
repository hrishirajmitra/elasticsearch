# IRE Assignment 1 Report


## 1. Data Preprocessing & Elasticsearch Indexing

**Implementation:** `preprocess_data.py`, `elasticsearch_indexer.py`

Applied Porter stemming, stopword removal (NLTK), and punctuation handling to 50,000 Wikipedia articles. Indexed into Elasticsearch (esindex-v1-0) using BM25 scoring.

**Results:**
- Without preprocessing: 626,087 unique tokens
- With preprocessing: 486,898 unique tokens (22% reduction)

![Word Frequency Comparison](data/plots/word_frequencies_comparison.png)
![Frequency Distribution](data/plots/frequency_distribution_zipf.png)

---

## 2. Plot C: Information Type Comparison (x=1,2,3)

**Implementation:** `plot_c_info_type.py`

Built three index variants:
- **x=1**: Boolean index (presence/absence only)
- **x=2**: WordCount index (term frequencies)
- **x=3**: TF-IDF index (pre-computed scores)

![Information Type Comparison](data/plots/plot_c_info_type.png)

**Key Findings:**
- Boolean (x=1): 263 MB
- WordCount (x=2): 308 MB
- TF-IDF (x=3): 389 MB
---

## 3. Plot A/Y: Datastore Comparison (y=1,2)

**Implementation:** `plot_a_datastore.py`, `measure_latency_datastores.py`

Compared three datastores:
- **y=1**: Pickle (Python serialization on disk)
- **y=2a**: PostgreSQL with GIN index
- **y=2b**: Redis (in-memory key-value store)

![Datastore Latency Comparison](data/plots/datastore_latency_comparison.png)
![Datastore Memory Usage](data/plots/datastore_mem.png)


---

## 4. Plot AB: Compression Comparison (z=1,2)

**Implementation:** `selfindex_z1_simple_compression.py`, `selfindex_z2_lib_compression.py`, `plot_ab_compression.py`

Compared two compression methods:
- **z=1**: Variable Byte Encoding (VByte) with delta encoding
- **z=2**: Zlib (DEFLATE algorithm)

![Compression Comparison](data/plots/plot_ab_compression_comparison.png)

**Key Findings:**
- VByte (z=1): 68.94% compression ratio, 243 QPS
- Zlib (z=2): 53.67% compression ratio, 304 QPS
---

## 5. Plot A/I: Skip Pointer Optimization (i=0,1)

**Implementation:** `selfindex_i0_no_optimization.py`, `selfindex_i1_skip_pointers.py`, `plot_a_skip_optimization.py`

Implemented skip pointers for posting list intersection optimization.

**Skip Pointer Strategy:**
- Interval: max(8, sqrt(list_length))
- Small lists (<50): No skip pointers (overhead not worth it)
- Minimum interval: 8 (avoids excessive overhead)

![Skip Pointer Comparison](data/plots/plot_a_skip_pointer_comparison.png)

**Key Findings:**
- Baseline (i=0): Mean 1.85ms, p95: 12.03ms
- Skip Pointers (i=1): Mean 1.32ms, p95: 9.53ms
- **29% faster mean latency, 21% faster p95**
- Most effective for high-frequency terms with long posting lists

---

## 6. Plot AC: Query Processing Comparison (q=T,D)

**Implementation:** `selfindex_q_taat.py`, `selfindex_q_daat.py`, `plot_ac_query_processing.py`

Compared two query processing strategies:
- **q=T**: Term-at-a-Time (TAAT) - process one term fully, accumulate scores
- **q=D**: Document-at-a-Time (DAAT) - process one document fully across all terms

![Query Processing Comparison](data/plots/plot_ac_query_processing_comparison.png)

**Key Findings:**
- TAAT: Mean 4.78ms, Median 1.00ms, p95: 17.78ms
- DAAT: Mean 5.20ms, Median 0.78ms, p95: 18.73ms
- **DAAT 22.6% faster at median** (better typical case)
- TAAT better worst-case performance (p99)
- Choice depends on query characteristics and parallelization needs

---

## 7. Plot D: Functional Metrics Evaluation

**Implementation:** `plot_d_functional_metrics.py`, `query_preprocessing.py`

Evaluated 8 configurations against Elasticsearch ground truth:
1. TAAT (TF-IDF)
2. TAAT (BM25 k1=1.2, b=0.75)
3. DAAT (TF-IDF)
4. DAAT (BM25 k1=1.2, b=0.75)
5. Baseline (i=0, no optimization)
6. Skip Pointers (i=1)
7. TAAT (BM25 k1=2.0) - aggressive saturation
8. TAAT (BM25 b=0.5) - reduced length normalization

**Metrics:** Precision@K, Recall@K, MAP, MRR, NDCG@K, F1

![Functional Metrics Comparison](data/plots/plot_d_functional_metrics.png)

### Results Summary:

**MAP (Mean Average Precision):**
```
DAAT (BM25)              93.08%   Best
TAAT (BM25)              93.05%
TAAT (BM25 k1=2.0)       88.15%
TAAT (BM25 b=0.5)        78.58%
DAAT (TF-IDF)            21.22%
TAAT (TF-IDF)            21.17%
Skip Pointers (i=1)      18.54%
Baseline (i=0)            0.10% 
```

**Precision@10:**
```
TAAT (BM25)              93.60%
DAAT (BM25)              93.60%
TAAT (BM25 k1=2.0)       89.30%
TAAT (BM25 b=0.5)        81.20%
TAAT (TF-IDF)            29.60%
DAAT (TF-IDF)            29.70%
Skip Pointers (i=1)      23.50%
Baseline (i=0)            0.10%
```

**MRR (Mean Reciprocal Rank):**
```
All BM25 variants        100.0%  (Perfect first result!)
TAAT (TF-IDF)            45.83%
DAAT (TF-IDF)            45.74%
Skip Pointers (i=1)      32.57%
Baseline (i=0)            1.00%
```

### Key Insights:

1. **Scoring Function Dominates Performance**
   - BM25 vs TF-IDF: 4.4x improvement (93% vs 21% MAP)
   - Query processing (TAAT vs DAAT): Minimal difference (0.03% MAP)
   - **Conclusion:** Scoring function matters far more than query strategy

2. **BM25 Parameter Sensitivity**
   - Standard (k1=1.2, b=0.75): 93% MAP 
   - High saturation (k1=2.0): 88% MAP (-5%)
   - Low normalization (b=0.5): 79% MAP (-14%)
   - **Conclusion:** Standard parameters are well-tuned

3. **Optimization Trade-offs**
   - Skip Pointers: 29% faster but 12% lower quality (18.54% vs 21.17% MAP)
   - Baseline (i=0): Nearly unusable (0.1% precision)
   - **Conclusion:** Speed optimizations can reduce quality

4. **Ground Truth Alignment**
   - Elasticsearch uses BM25 → SelfIndex with BM25: 93% agreement
   - SelfIndex with TF-IDF: 21% agreement
   - **Conclusion:** Matching scoring functions crucial for evaluation

5. **Critical Bug Fixed**
   - Initial results: 6% precision (queries not preprocessed, documents were)
   - After fix: 30% TF-IDF, 93% BM25 precision
   - **Lesson:** Query and document preprocessing must match exactly

---