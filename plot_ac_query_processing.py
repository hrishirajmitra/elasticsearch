"""
Plot AC: Query Processing Comparison - Term-at-a-Time vs Document-at-a-Time
Benchmark latency (Plot A) and memory footprint (Plot C) for TAAT vs DAAT
"""
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from selfindex_q_taat import SelfIndexTAAT
from selfindex_q_daat import SelfIndexDAAT


class PlotACBenchmark:
    """Benchmark TAAT vs DAAT query processing"""
    
    def __init__(self, query_file: str = "queryset.json"):
        self.query_file = query_file
        self.queries = self._load_queries()
        self.results_dir = Path("data/plots")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_queries(self) -> List[str]:
        """Load queries from queryset"""
        with open(self.query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        queries = data.get('queries', [])
        print(f"Loaded {len(queries)} queries from queryset")
        return queries
    
    def measure_latency(self, indexer, queries: List[str], name: str) -> Dict:
        """Measure query latency with percentiles"""
        print(f"\nMeasuring latency for {name}...")
        
        # Warmup phase
        print("  Warmup phase (3 rounds)...")
        for _ in range(3):
            for query in queries[:10]:
                indexer.query(query, size=10)
        
        # Actual measurement
        print(f"  Executing {len(queries)} queries...")
        latencies = []
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            indexer.query(query, size=10)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(queries)} queries...")
        
        # Calculate statistics
        latencies_array = np.array(latencies)
        stats = {
            'mean': float(np.mean(latencies_array)),
            'median': float(np.median(latencies_array)),
            'std': float(np.std(latencies_array)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'latencies': latencies
        }
        
        print(f"  ✓ Latency measurement complete!")
        print(f"    Mean: {stats['mean']:.2f} ms")
        print(f"    Median (p50): {stats['p50']:.2f} ms")
        print(f"    p95: {stats['p95']:.2f} ms")
        print(f"    p99: {stats['p99']:.2f} ms")
        
        return stats
    
    def run_benchmark(self):
        """Run complete benchmark comparing TAAT vs DAAT"""
        print("="*70)
        print("PLOT AC: QUERY PROCESSING COMPARISON (TAAT vs DAAT)")
        print("="*70)
        
        print("\nInitializing indexers...")
        
        # Initialize TAAT indexer (q=T0)
        taat_indexer = SelfIndexTAAT()
        
        # Initialize DAAT indexer (q=D0)
        daat_indexer = SelfIndexDAAT()
        
        # Load indices
        print("\n--- Loading Index q=T0 (Term-at-a-Time) ---")
        taat_indexer.load_index('SelfIndex-v1.300T0')
        
        print("\n--- Loading Index q=D0 (Document-at-a-Time) ---")
        # Check if DAAT index exists, if not create it
        daat_index_path = Path("indices/SelfIndex-v1.300D0.pkl")
        if not daat_index_path.exists():
            print("DAAT index not found. Creating...")
            daat_indexer.create_index('SelfIndex-v1.300D0')
        else:
            daat_indexer.load_index('SelfIndex-v1.300D0')
        
        # Get stats
        taat_stats = taat_indexer.get_stats()
        daat_stats = daat_indexer.get_stats()
        
        # Benchmark TAAT
        print("\n" + "="*70)
        print(f"BENCHMARKING: {taat_indexer.identifier_short}")
        print("="*70)
        taat_latency = self.measure_latency(taat_indexer, self.queries, taat_indexer.identifier_short)
        
        # Benchmark DAAT
        print("\n" + "="*70)
        print(f"BENCHMARKING: {daat_indexer.identifier_short}")
        print("="*70)
        daat_latency = self.measure_latency(daat_indexer, self.queries, daat_indexer.identifier_short)
        
        # Compile results
        results = {
            'metadata': {
                'benchmark_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'query_set_size': len(self.queries),
                'query_set_file': self.query_file
            },
            'taat': {
                'identifier': taat_indexer.identifier_short,
                'query_processing': 'Term-at-a-Time',
                'index_stats': taat_stats,
                'latency': {
                    'mean_ms': taat_latency['mean'],
                    'median_ms': taat_latency['median'],
                    'p50_ms': taat_latency['p50'],
                    'p95_ms': taat_latency['p95'],
                    'p99_ms': taat_latency['p99'],
                    'std_ms': taat_latency['std'],
                    'min_ms': taat_latency['min'],
                    'max_ms': taat_latency['max']
                }
            },
            'daat': {
                'identifier': daat_indexer.identifier_short,
                'query_processing': 'Document-at-a-Time',
                'index_stats': daat_stats,
                'latency': {
                    'mean_ms': daat_latency['mean'],
                    'median_ms': daat_latency['median'],
                    'p50_ms': daat_latency['p50'],
                    'p95_ms': daat_latency['p95'],
                    'p99_ms': daat_latency['p99'],
                    'std_ms': daat_latency['std'],
                    'min_ms': daat_latency['min'],
                    'max_ms': daat_latency['max']
                }
            }
        }
        
        # Save results
        results_path = self.results_dir / "plot_ac_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Save detailed results
        detailed_results = {
            'metadata': results['metadata'],
            'taat_latencies': taat_latency['latencies'],
            'daat_latencies': daat_latency['latencies']
        }
        detailed_path = self.results_dir / "plot_ac_detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✓ Detailed results saved to: {detailed_path}")
        
        # Print comparison summary
        self._print_summary(results)
        
        # Generate plots
        self._generate_plots(results, taat_latency, daat_latency)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("QUERY PROCESSING COMPARISON SUMMARY")
        print("="*70)
        
        taat = results['taat']
        daat = results['daat']
        
        print("\n--- Index Statistics ---")
        print(f"{'Metric':<40} {'TAAT (q=T0)':<20} {'DAAT (q=D0)':<20}")
        print("-" * 80)
        print(f"{'Document Count':<40} {taat['index_stats']['doc_count']:<20,} {daat['index_stats']['doc_count']:<20,}")
        print(f"{'Unique Terms':<40} {taat['index_stats']['unique_terms']:<20,} {daat['index_stats']['unique_terms']:<20,}")
        print(f"{'Avg Doc Length':<40} {taat['index_stats']['avg_doc_length']:<20.2f} {daat['index_stats']['avg_doc_length']:<20.2f}")
        
        print("\n--- Latency Statistics (Plot A) ---")
        print(f"{'Metric':<40} {'TAAT (q=T0)':<20} {'DAAT (q=D0)':<20} {'Improvement':<20}")
        print("-" * 95)
        
        metrics = [
            ('Mean Latency (ms)', 'mean_ms'),
            ('Median Latency (ms)', 'median_ms'),
            ('p50 Latency (ms)', 'p50_ms'),
            ('p95 Latency (ms)', 'p95_ms'),
            ('p99 Latency (ms)', 'p99_ms')
        ]
        
        for label, key in metrics:
            taat_val = taat['latency'][key]
            daat_val = daat['latency'][key]
            improvement = ((taat_val - daat_val) / taat_val * 100) if taat_val > 0 else 0
            print(f"{label:<40} {taat_val:<20.2f} {daat_val:<20.2f} {improvement:+.1f}%")
        
        print("\n" + "="*70)
    
    def _generate_plots(self, results: Dict, taat_latency: Dict, daat_latency: Dict):
        """Generate comparison plots"""
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Plot AC: Query Processing Comparison (TAAT vs DAAT)', fontsize=14, fontweight='bold')
        
        # Plot 1: Latency comparison bar chart
        ax1 = axes[0, 0]
        metrics = ['Mean', 'Median', 'p95', 'p99']
        taat_values = [
            taat_latency['mean'],
            taat_latency['median'],
            taat_latency['p95'],
            taat_latency['p99']
        ]
        daat_values = [
            daat_latency['mean'],
            daat_latency['median'],
            daat_latency['p95'],
            daat_latency['p99']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, taat_values, width, label='TAAT (q=T0)', color='#2E86AB')
        ax1.bar(x + width/2, daat_values, width, label='DAAT (q=D0)', color='#A23B72')
        
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Latency distribution
        ax2 = axes[0, 1]
        ax2.boxplot([taat_latency['latencies'], daat_latency['latencies']],
                    labels=['TAAT', 'DAAT'])
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency Distribution (Box Plot)')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: CDF comparison
        ax3 = axes[1, 0]
        taat_sorted = np.sort(taat_latency['latencies'])
        daat_sorted = np.sort(daat_latency['latencies'])
        taat_cdf = np.arange(1, len(taat_sorted) + 1) / len(taat_sorted)
        daat_cdf = np.arange(1, len(daat_sorted) + 1) / len(daat_sorted)
        
        ax3.plot(taat_sorted, taat_cdf, label='TAAT (q=T0)', linewidth=2, color='#2E86AB')
        ax3.plot(daat_sorted, daat_cdf, label='DAAT (q=D0)', linewidth=2, color='#A23B72')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function (CDF)')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Performance improvement
        ax4 = axes[1, 1]
        improvements = []
        for i in range(len(metrics)):
            if taat_values[i] > 0:
                imp = ((taat_values[i] - daat_values[i]) / taat_values[i] * 100)
                improvements.append(imp)
            else:
                improvements.append(0)
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax4.barh(metrics, improvements, color=colors, alpha=0.7)
        ax4.set_xlabel('Improvement (%)')
        ax4.set_title('DAAT Performance Improvement over TAAT')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / "plot_ac_query_processing_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
        plt.close()


def main():
    print("Starting Plot AC benchmark...")
    benchmark = PlotACBenchmark(query_file="queryset.json")
    results = benchmark.run_benchmark()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/plots/plot_ac_results.json")
    print("  - data/plots/plot_ac_detailed_results.json")
    print("  - data/plots/plot_ac_query_processing_comparison.png")


if __name__ == "__main__":
    main()
