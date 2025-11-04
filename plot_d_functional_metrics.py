"""
Plot D: Functional Metrics Evaluation
Evaluate precision, recall, and ranking metrics using Elasticsearch as ground truth
Compare SelfIndex implementations against ES baseline
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
from elasticsearch import Elasticsearch
from selfindex_q_taat import SelfIndexTAAT
from selfindex_q_daat import SelfIndexDAAT
from selfindex_i1_skip_pointers import SelfIndexI1
from selfindex_i0_no_optimization import SelfIndexI0


class FunctionalMetricsEvaluator:
    """Evaluate functional metrics using ES as ground truth"""
    
    def __init__(self, query_file: str = "queryset.json", es_host: str = "http://localhost:9200"):
        self.query_file = query_file
        self.queries = self._load_queries()
        self.es = Elasticsearch([es_host])
        self.es_index = "esindex-v1-0"
        self.results_dir = Path("data/plots")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_queries(self) -> List[str]:
        """Load queries from queryset"""
        with open(self.query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        queries = data.get('queries', [])
        print(f"Loaded {len(queries)} queries from queryset")
        return queries
    
    def get_es_results(self, query: str, size: int = 10) -> List[str]:
        """Get results from Elasticsearch (ground truth)"""
        try:
            response = self.es.search(
                index=self.es_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text"]
                        }
                    },
                    "size": size
                }
            )
            
            # Extract document IDs
            doc_ids = [str(hit['_id']) for hit in response['hits']['hits']]
            return doc_ids
        except Exception as e:
            print(f"Error querying ES: {e}")
            return []
    
    def get_selfindex_results(self, indexer, query: str, size: int = 10) -> List[str]:
        """Get results from SelfIndex"""
        try:
            results_json = indexer.query(query, size=size)
            results = json.loads(results_json)
            doc_ids = [result['doc_id'] for result in results.get('results', [])]
            return doc_ids
        except Exception as e:
            print(f"Error querying SelfIndex: {e}")
            return []
    
    def calculate_precision(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        """
        Calculate Precision@K
        Precision = |Retrieved ∩ Relevant| / |Retrieved|
        """
        if not retrieved:
            return 0.0
        
        if k is not None:
            retrieved = retrieved[:k]
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        tp = len(retrieved_set & relevant_set)
        precision = tp / len(retrieved_set) if retrieved_set else 0.0
        
        return precision
    
    def calculate_recall(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        """
        Calculate Recall@K
        Recall = |Retrieved ∩ Relevant| / |Relevant|
        """
        if not relevant:
            return 0.0
        
        if k is not None:
            retrieved = retrieved[:k]
        
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        
        tp = len(retrieved_set & relevant_set)
        recall = tp / len(relevant_set) if relevant_set else 0.0
        
        return recall
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        Calculate F1 Score
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ap(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Average Precision (AP)
        AP = (1/|Relevant|) * Σ(Precision@k * rel(k))
        where rel(k) is 1 if item at rank k is relevant, 0 otherwise
        """
        if not relevant or not retrieved:
            return 0.0
        
        relevant_set = set(relevant)
        score = 0.0
        num_hits = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                num_hits += 1
                precision_at_k = num_hits / k
                score += precision_at_k
        
        ap = score / len(relevant_set) if relevant_set else 0.0
        return ap
    
    def calculate_rr(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Reciprocal Rank (RR)
        RR = 1 / rank_of_first_relevant_item
        """
        if not relevant or not retrieved:
            return 0.0
        
        relevant_set = set(relevant)
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg(self, retrieved: List[str], relevant: List[str], k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        DCG = Σ(rel_i / log2(i + 1))
        NDCG = DCG / IDCG
        """
        if not relevant or not retrieved:
            return 0.0
        
        if k is not None:
            retrieved = retrieved[:k]
        
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved, 1):
            rel = 1.0 if doc_id in relevant_set else 0.0
            dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG - all relevant docs at top)
        idcg = 0.0
        for i in range(1, min(len(relevant), len(retrieved)) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def evaluate_indexer(self, indexer, name: str, queries: List[str]) -> Dict:
        """Evaluate an indexer against ES ground truth"""
        print(f"\n{'='*70}")
        print(f"EVALUATING: {name}")
        print(f"{'='*70}")
        
        metrics = {
            'precision_at_5': [],
            'precision_at_10': [],
            'recall_at_5': [],
            'recall_at_10': [],
            'f1_at_5': [],
            'f1_at_10': [],
            'average_precision': [],
            'reciprocal_rank': [],
            'ndcg_at_5': [],
            'ndcg_at_10': []
        }
        
        num_queries = len(queries)
        print(f"Processing {num_queries} queries...")
        
        for i, query in enumerate(queries):
            # Get ground truth from ES
            es_results = self.get_es_results(query, size=10)
            
            # Get results from SelfIndex
            self_results = self.get_selfindex_results(indexer, query, size=10)
            
            if not es_results:  # Skip if ES returns no results
                continue
            
            # Calculate metrics
            p5 = self.calculate_precision(self_results, es_results, k=5)
            p10 = self.calculate_precision(self_results, es_results, k=10)
            r5 = self.calculate_recall(self_results, es_results, k=5)
            r10 = self.calculate_recall(self_results, es_results, k=10)
            f1_5 = self.calculate_f1(p5, r5)
            f1_10 = self.calculate_f1(p10, r10)
            ap = self.calculate_ap(self_results, es_results)
            rr = self.calculate_rr(self_results, es_results)
            ndcg5 = self.calculate_ndcg(self_results, es_results, k=5)
            ndcg10 = self.calculate_ndcg(self_results, es_results, k=10)
            
            metrics['precision_at_5'].append(p5)
            metrics['precision_at_10'].append(p10)
            metrics['recall_at_5'].append(r5)
            metrics['recall_at_10'].append(r10)
            metrics['f1_at_5'].append(f1_5)
            metrics['f1_at_10'].append(f1_10)
            metrics['average_precision'].append(ap)
            metrics['reciprocal_rank'].append(rr)
            metrics['ndcg_at_5'].append(ndcg5)
            metrics['ndcg_at_10'].append(ndcg10)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_queries} queries...")
        
        # Calculate mean for each metric
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                results[metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        # Calculate aggregate metrics
        results['map'] = float(np.mean(metrics['average_precision'])) if metrics['average_precision'] else 0.0
        results['mrr'] = float(np.mean(metrics['reciprocal_rank'])) if metrics['reciprocal_rank'] else 0.0
        
        print(f"\n✓ Evaluation complete!")
        print(f"  MAP (Mean Average Precision): {results['map']:.4f}")
        print(f"  MRR (Mean Reciprocal Rank): {results['mrr']:.4f}")
        print(f"  Precision@10: {results['precision_at_10']['mean']:.4f}")
        print(f"  Recall@10: {results['recall_at_10']['mean']:.4f}")
        print(f"  NDCG@10: {results['ndcg_at_10']['mean']:.4f}")
        
        return results
    
    def run_evaluation(self):
        """Run complete evaluation for all indexers"""
        print("="*70)
        print("PLOT D: FUNCTIONAL METRICS EVALUATION")
        print("Using Elasticsearch as Ground Truth")
        print("="*70)
        
        # Verify ES connection
        print(f"\n✓ Connected to Elasticsearch")
        print(f"  Index: {self.es_index}")
        doc_count = self.es.count(index=self.es_index)['count']
        print(f"  Documents: {doc_count:,}")
        
        # Initialize indexers
        print("\nInitializing SelfIndex implementations...")
        
        indexers = []
        
        # TAAT indexer - TF-IDF
        print("\n--- Loading TAAT Indexer (TF-IDF) ---")
        taat_tfidf = SelfIndexTAAT(scoring='tfidf')
        taat_tfidf.load_index('SelfIndex-v1.300T0')
        indexers.append(('TAAT (TF-IDF)', taat_tfidf))
        
        # TAAT indexer - BM25
        print("\n--- Loading TAAT Indexer (BM25) ---")
        taat_bm25 = SelfIndexTAAT(scoring='bm25', k1=1.2, b=0.75)
        taat_bm25.load_index('SelfIndex-v1.300T0')
        indexers.append(('TAAT (BM25)', taat_bm25))
        
        # DAAT indexer - TF-IDF
        print("\n--- Loading DAAT Indexer (TF-IDF) ---")
        daat_tfidf = SelfIndexDAAT(scoring='tfidf')
        daat_tfidf.load_index('SelfIndex-v1.300D0')
        indexers.append(('DAAT (TF-IDF)', daat_tfidf))
        
        # DAAT indexer - BM25
        print("\n--- Loading DAAT Indexer (BM25) ---")
        daat_bm25 = SelfIndexDAAT(scoring='bm25', k1=1.2, b=0.75)
        daat_bm25.load_index('SelfIndex-v1.300D0')
        indexers.append(('DAAT (BM25)', daat_bm25))
        
        # Baseline indexer - No optimization (i=0)
        print("\n--- Loading Baseline Indexer (i=0) ---")
        baseline = SelfIndexI0()
        baseline.load_index('SelfIndex-v1.100T0')
        indexers.append(('Baseline (i=0)', baseline))
        
        # Skip pointer indexer (i=1)
        print("\n--- Loading Skip Pointer Indexer (i=1) ---")
        skip_indexer = SelfIndexI1()
        skip_indexer.load_index('SelfIndex-v1.10010')
        indexers.append(('Skip Pointers (i=1)', skip_indexer))
        
        # Test BM25 with different parameters
        print("\n--- Loading TAAT with BM25 (k1=2.0, aggressive saturation) ---")
        taat_bm25_k2 = SelfIndexTAAT(scoring='bm25', k1=2.0, b=0.75)
        taat_bm25_k2.load_index('SelfIndex-v1.300T0')
        indexers.append(('TAAT (BM25 k1=2.0)', taat_bm25_k2))
        
        print("\n--- Loading TAAT with BM25 (b=0.5, less length norm) ---")
        taat_bm25_b05 = SelfIndexTAAT(scoring='bm25', k1=1.2, b=0.5)
        taat_bm25_b05.load_index('SelfIndex-v1.300T0')
        indexers.append(('TAAT (BM25 b=0.5)', taat_bm25_b05))
        
        # Evaluate each indexer
        all_results = {}
        
        for name, indexer in indexers:
            results = self.evaluate_indexer(indexer, name, self.queries)
            all_results[name] = results
        
        # Compile final results
        final_results = {
            'metadata': {
                'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'query_set_size': len(self.queries),
                'ground_truth': 'Elasticsearch (esindex-v1-0)',
                'es_document_count': doc_count
            },
            'indexers': all_results
        }
        
        # Save results
        results_path = self.results_dir / "plot_d_functional_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Print comparison summary
        self._print_summary(final_results)
        
        # Generate plots
        self._generate_plots(final_results)
        
        return final_results
    
    def _print_summary(self, results: Dict):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("FUNCTIONAL METRICS COMPARISON SUMMARY")
        print("="*70)
        
        indexers = results['indexers']
        
        print("\n--- Precision & Recall ---")
        print(f"{'Indexer':<25} {'P@5':<10} {'P@10':<10} {'R@5':<10} {'R@10':<10}")
        print("-" * 70)
        
        for name, metrics in indexers.items():
            print(f"{name:<25} "
                  f"{metrics['precision_at_5']['mean']:<10.4f} "
                  f"{metrics['precision_at_10']['mean']:<10.4f} "
                  f"{metrics['recall_at_5']['mean']:<10.4f} "
                  f"{metrics['recall_at_10']['mean']:<10.4f}")
        
        print("\n--- Ranking Metrics ---")
        print(f"{'Indexer':<25} {'MAP':<10} {'MRR':<10} {'NDCG@5':<10} {'NDCG@10':<10}")
        print("-" * 70)
        
        for name, metrics in indexers.items():
            print(f"{name:<25} "
                  f"{metrics['map']:<10.4f} "
                  f"{metrics['mrr']:<10.4f} "
                  f"{metrics['ndcg_at_5']['mean']:<10.4f} "
                  f"{metrics['ndcg_at_10']['mean']:<10.4f}")
        
        print("\n--- F1 Scores ---")
        print(f"{'Indexer':<25} {'F1@5':<10} {'F1@10':<10}")
        print("-" * 70)
        
        for name, metrics in indexers.items():
            print(f"{name:<25} "
                  f"{metrics['f1_at_5']['mean']:<10.4f} "
                  f"{metrics['f1_at_10']['mean']:<10.4f}")
        
        print("\n" + "="*70)
    
    def _generate_plots(self, results: Dict):
        """Generate comparison plots"""
        print("\nGenerating plots...")
        
        indexers = results['indexers']
        names = list(indexers.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Plot D: Functional Metrics Evaluation (vs Elasticsearch Ground Truth)', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Precision & Recall comparison
        ax1 = axes[0, 0]
        x = np.arange(len(names))
        width = 0.2
        
        p5_vals = [indexers[name]['precision_at_5']['mean'] for name in names]
        p10_vals = [indexers[name]['precision_at_10']['mean'] for name in names]
        r5_vals = [indexers[name]['recall_at_5']['mean'] for name in names]
        r10_vals = [indexers[name]['recall_at_10']['mean'] for name in names]
        
        ax1.bar(x - 1.5*width, p5_vals, width, label='P@5', color='#2E86AB')
        ax1.bar(x - 0.5*width, p10_vals, width, label='P@10', color='#A23B72')
        ax1.bar(x + 0.5*width, r5_vals, width, label='R@5', color='#F18F01')
        ax1.bar(x + 1.5*width, r10_vals, width, label='R@10', color='#C73E1D')
        
        ax1.set_ylabel('Score')
        ax1.set_title('Precision & Recall')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Plot 2: Ranking metrics (MAP, MRR, NDCG)
        ax2 = axes[0, 1]
        map_vals = [indexers[name]['map'] for name in names]
        mrr_vals = [indexers[name]['mrr'] for name in names]
        ndcg10_vals = [indexers[name]['ndcg_at_10']['mean'] for name in names]
        
        ax2.bar(x - width, map_vals, width, label='MAP', color='#2E86AB')
        ax2.bar(x, mrr_vals, width, label='MRR', color='#A23B72')
        ax2.bar(x + width, ndcg10_vals, width, label='NDCG@10', color='#F18F01')
        
        ax2.set_ylabel('Score')
        ax2.set_title('Ranking Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # Plot 3: F1 Scores
        ax3 = axes[1, 0]
        f1_5_vals = [indexers[name]['f1_at_5']['mean'] for name in names]
        f1_10_vals = [indexers[name]['f1_at_10']['mean'] for name in names]
        
        ax3.bar(x - width/2, f1_5_vals, width, label='F1@5', color='#2E86AB')
        ax3.bar(x + width/2, f1_10_vals, width, label='F1@10', color='#A23B72')
        
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Scores')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=15, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1.0)
        
        # Plot 4: Overall comparison (radar/spider chart alternative - horizontal bars)
        ax4 = axes[1, 1]
        
        # Show top performer for each metric
        metrics_to_show = ['MAP', 'MRR', 'P@10', 'R@10', 'NDCG@10']
        best_scores = []
        
        for name in names:
            score = (indexers[name]['map'] + 
                    indexers[name]['mrr'] + 
                    indexers[name]['precision_at_10']['mean'] + 
                    indexers[name]['recall_at_10']['mean'] + 
                    indexers[name]['ndcg_at_10']['mean']) / 5
            best_scores.append(score)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        ax4.barh(names, best_scores, color=colors[:len(names)])
        ax4.set_xlabel('Average Score')
        ax4.set_title('Overall Performance (Average of Key Metrics)')
        ax4.set_xlim(0, 1.0)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / "plot_d_functional_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
        plt.close()


def main():
    print("Starting Plot D evaluation...")
    evaluator = FunctionalMetricsEvaluator(query_file="queryset.json")
    results = evaluator.run_evaluation()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/plots/plot_d_functional_metrics.json")
    print("  - data/plots/plot_d_functional_metrics.png")


if __name__ == "__main__":
    main()
