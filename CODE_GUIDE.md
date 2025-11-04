# Python Files Overview

This document maps each Python file to its purpose and corresponding assignment requirement.

---

## Data Preprocessing & Indexing

| File                               | Purpose                                                                        | Maps To                              |
| ---------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `preprocess_data.py`               | Performs Porter stemming, stopword removal, and tokenization on Wikipedia data | Activity 1: Data preprocessing       |
| `download_wikipedia.py`            | Downloads Wikipedia dataset from HuggingFace                                   | Activity 1: Data sources             |
| `generate_word_frequency_plots.py` | Creates word frequency plots before and after preprocessing                    | Activity 1: Word frequency plots     |
| `elasticsearch_indexer.py`         | Indexes preprocessed data into Elasticsearch (esindex-v1-0)                    | Activity 1: Index into Elasticsearch |
| `query_preprocessing.py`           | Preprocesses user queries with same stemming/stopword rules as documents       | Support for query processing         |

---

## Information Type Variants (x=1,2,3)

| File                  | Purpose                                                                      | Maps To                   |
| --------------------- | ---------------------------------------------------------------------------- | ------------------------- |
| `plot_c_info_type.py` | Builds and compares Boolean (x=1), WordCount (x=2), and TF-IDF (x=3) indices | Plot.C: Information types |

**Assignment Mapping:**

- x=1: Boolean index with document IDs
- x=2: Enable ranking with word counts
- x=3: Evaluate gains from TF-IDF scores

---

## Datastore Comparison (y=1,2)

| File                            | Purpose                                                                              | Maps To                      |
| ------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------- |
| `plot_a_datastore.py`           | Implements and compares Pickle (y=1), PostgreSQL (y=2a), and Redis (y=2b) datastores | Plot.A: Datastore choices    |
| `measure_latency_datastores.py` | Measures latency metrics (p50, p95, p99) for different datastores                    | Plot.A: System response time |
| `plot_latency_datastores.py`    | Generates visualization plots for datastore latency comparison                       | Plot.A: Visualization        |
| `test_connections.py`           | Tests PostgreSQL and Redis Docker connections                                        | Support for y=2 datastores   |

**Assignment Mapping:**

- y=1: Custom objects (Pickle) on disk
- y=2: PostgreSQL GIN + Redis (off-the-shelf)

---

## Compression Methods (z=1,2)

| File                                 | Purpose                                                           | Maps To                         |
| ------------------------------------ | ----------------------------------------------------------------- | ------------------------------- |
| `selfindex_z1_simple_compression.py` | Implements Variable Byte Encoding with delta encoding             | Plot.AB: Simple code (z=1)      |
| `selfindex_z2_lib_compression.py`    | Implements Zlib compression for posting lists                     | Plot.AB: Library (z=2)          |
| `plot_ab_compression.py`             | Compares compression ratio, memory, and throughput for z=1 vs z=2 | Plot.AB: Compression comparison |
| `verify_compression.py`              | Validates correctness of compression/decompression algorithms     | Support for compression         |

**Assignment Mapping:**

- z=1: Simple custom code (VByte)
- z=2: Off-the-shelf library (Zlib)

---

## Index Optimization (i=0,1)

| File                              | Purpose                                                            | Maps To                        |
| --------------------------------- | ------------------------------------------------------------------ | ------------------------------ |
| `selfindex_i0_no_optimization.py` | Baseline index without skip pointers (i=0)                         | Plot.A: Baseline (i=0)         |
| `selfindex_i1_skip_pointers.py`   | Index with skip pointer optimization for faster intersection (i=1) | Plot.A: Optimization (i=1)     |
| `plot_a_skip_optimization.py`     | Measures and compares latency between i=0 and i=1                  | Plot.A: Skipping with pointers |

**Assignment Mapping:**

- i=0: No optimization (baseline)
- i=1: Skip pointers for faster posting list intersection

---

## Query Processing Strategies (q=T,D)

| File                          | Purpose                                               | Maps To                              |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------ |
| `selfindex_q_taat.py`         | Implements Term-at-a-Time query processing (q=T)      | Plot.AC: TAAT (q=Tn)                 |
| `selfindex_q_daat.py`         | Implements Document-at-a-Time query processing (q=D)  | Plot.AC: DAAT (q=Dn)                 |
| `plot_ac_query_processing.py` | Compares latency and throughput between TAAT and DAAT | Plot.AC: Query processing comparison |

**Assignment Mapping:**

- q=T: Term-at-a-time processing
- q=D: Document-at-a-time processing

---

## Functional Metrics Evaluation (Plot D)

| File                           | Purpose                                                                        | Maps To                    |
| ------------------------------ | ------------------------------------------------------------------------------ | -------------------------- |
| `plot_d_functional_metrics.py` | Evaluates Precision, Recall, MAP, MRR, NDCG against Elasticsearch ground truth | Plot.D: Functional metrics |
| `debug_results.py`             | Debugs low precision/recall issues (found query preprocessing bug)             | Support for Plot D         |

**Assignment Mapping:**

- Measures precision, recall, ranking quality
- Compares SelfIndex variants vs Elasticsearch

---

## Support Files

| File                 | Purpose                                                                     | Maps To                  |
| -------------------- | --------------------------------------------------------------------------- | ------------------------ |
| `index_base.py`      | Abstract base class defining index interface with versioning (xyziq format) | Core framework           |
| `queryset.json`      | 100 diverse queries for testing across all plots                            | Query set for evaluation |
| `create_sample.py`   | Creates sample dataset for testing                                          | Development support      |
| `docker-compose.yml` | Sets up PostgreSQL, Redis, and Elasticsearch containers                     | Infrastructure           |

---

## Assignment Coverage Summary

| Requirement                          | Files                                                                                             | Status   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- | -------- |
| Activity 1: Data preprocessing       | `preprocess_data.py`, `generate_word_frequency_plots.py`                                          | Complete |
| Activity 1: Elasticsearch indexing   | `elasticsearch_indexer.py`                                                                        | Complete |
| Plot.C (x=1,2,3): Information types  | `plot_c_info_type.py`                                                                             | Complete |
| Plot.A (y=1,2): Datastore comparison | `plot_a_datastore.py`, `measure_latency_datastores.py`                                            | Complete |
| Plot.AB (z=1,2): Compression         | `selfindex_z1_simple_compression.py`, `selfindex_z2_lib_compression.py`, `plot_ab_compression.py` | Complete |
| Plot.A (i=0,1): Skip pointers        | `selfindex_i0_no_optimization.py`, `selfindex_i1_skip_pointers.py`, `plot_a_skip_optimization.py` | Complete |
| Plot.AC (q=T,D): Query processing    | `selfindex_q_taat.py`, `selfindex_q_daat.py`, `plot_ac_query_processing.py`                       | Complete |
| Plot.D: Functional metrics           | `plot_d_functional_metrics.py`                                                                    | Complete |

---

## File Naming Convention

All SelfIndex implementations follow the versioning format: `SelfIndex-v1.xyziq`

- **x**: Information type (1=Boolean, 2=WordCount, 3=TF-IDF)
- **y**: Datastore (1=Pickle, 2=PostgreSQL/Redis)
- **z**: Compression (1=VByte, 2=Zlib)
- **i**: Optimization (0=None, 1=Skip pointers)
- **q**: Query processing (T=TAAT, D=DAAT)

Example: `SelfIndex-v1.300T0` = TF-IDF index, custom pickle storage, no compression, TAAT, no optimization
