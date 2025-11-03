"""
Plot A: Index Optimization Comparison (Skip Pointers)
Compares latency for:
- i=0: No optimization (baseline) - SelfIndex-v1.100T0
- i=1: Skip pointers optimization - SelfIndex-v1.101T0

Metrics:
- Plot A: Response time with p95 and p99 percentiles
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# Import the two index implementations
from selfindex_i0_no_optimization import SelfIndexI0
from selfindex_i1_skip_pointers import SelfIndexI1


class SkipPointerBenchmark:
    """Benchmark skip pointer optimization for inverted indices"""
    
    def __init__(self):
        self.queryset_path = Path("queryset.json")
        self.results_path = Path("data/plots/plot_a_skip_results.json")
        self.detailed_results_path = Path("data/plots/plot_a_skip_detailed_results.json")
        
        # Load query set
        with open(self.queryset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.queries = data['queries']
            self.metadata = data['metadata']
        
        print(f"Loaded {len(self.queries)} queries from queryset")
    
    def measure_latency(self, indexer, queries: List[str], warmup_rounds: int = 3) -> Dict:
        """
        Measure query latency with proper warmup
        Returns: dict with latencies, percentiles, and stats
        """
        print(f"\nMeasuring latency for {indexer.identifier_short}...")
        
        # Warmup phase
        print(f"  Warmup phase ({warmup_rounds} rounds)...")
        for _ in range(warmup_rounds):
            for query in queries[:10]:
                indexer.query(query, size=10)
        
        # Actual measurement
        print(f"  Executing {len(queries)} queries...")
        latencies = []
        query_details = []
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            result = indexer.query(query, size=10)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Parse result
            try:
                result_dict = json.loads(result)
                total_hits = result_dict.get('total_hits', 0)
            except:
                total_hits = 0
            
            query_details.append({
                'query': query,
                'latency_ms': latency_ms,
                'total_hits': total_hits
            })
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(queries)} queries...")
        
        # Calculate statistics
        latencies_array = np.array(latencies)
        
        results = {
            'indexer': indexer.identifier_short,
            'total_queries': len(queries),
            'latencies_ms': latencies,
            'query_details': query_details,
            'statistics': {
                'mean_ms': float(np.mean(latencies_array)),
                'median_ms': float(np.median(latencies_array)),
                'std_ms': float(np.std(latencies_array)),
                'min_ms': float(np.min(latencies_array)),
                'max_ms': float(np.max(latencies_array)),
                'p50_ms': float(np.percentile(latencies_array, 50)),
                'p95_ms': float(np.percentile(latencies_array, 95)),
                'p99_ms': float(np.percentile(latencies_array, 99)),
                'p999_ms': float(np.percentile(latencies_array, 99.9))
            }
        }
        
        print(f"  ✓ Latency measurement complete!")
        print(f"    Mean: {results['statistics']['mean_ms']:.2f} ms")
        print(f"    Median (p50): {results['statistics']['p50_ms']:.2f} ms")
        print(f"    p95: {results['statistics']['p95_ms']:.2f} ms")
        print(f"    p99: {results['statistics']['p99_ms']:.2f} ms")
        
        return results
    
    def run_benchmark(self):
        """Run complete benchmark for skip pointer optimization"""
        
        print("="*70)
        print("PLOT A: SKIP POINTER OPTIMIZATION BENCHMARK")
        print("="*70)
        
        # Initialize indexers
        print("\nInitializing indexers...")
        i0_indexer = SelfIndexI0()
        i1_indexer = SelfIndexI1()
        
        # Load indices
        print("\n--- Loading Index i=0 (No Optimization) ---")
        i0_indexer.load_index('SelfIndex-v1.300T0')
        
        print("\n--- Loading Index i=1 (Skip Pointers) ---")
        # Check if skip pointer index exists, if not create it
        skip_index_path = Path("indices/SelfIndex-v1.10010.pkl")
        if not skip_index_path.exists():
            print("Skip pointer index not found. Creating...")
            i1_indexer.create_index('SelfIndex-v1.10010')
        else:
            i1_indexer.load_index('SelfIndex-v1.10010')
        
        if not i0_indexer.inverted_index or not i1_indexer.inverted_index:
            print("\n❌ Error: One or both indices not loaded properly!")
            return
        
        # Store all results
        all_results = {
            'metadata': {
                'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'query_set_size': len(self.queries),
                'query_set_metadata': self.metadata,
                'optimization': 'Skip Pointers'
            },
            'indices': {}
        }
        
        detailed_results = {
            'metadata': all_results['metadata'].copy(),
            'indices': {}
        }
        
        # Benchmark each indexer
        for indexer in [i0_indexer, i1_indexer]:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING: {indexer.identifier_short}")
            print(f"{'='*70}")
            
            # Get index stats
            index_stats = indexer.get_stats()
            
            # Measure latency (Plot A)
            latency_results = self.measure_latency(indexer, self.queries)
            
            # Store results
            identifier = indexer.identifier_short
            
            all_results['indices'][identifier] = {
                'index_stats': index_stats,
                'latency': {
                    'statistics': latency_results['statistics'],
                    'total_queries': latency_results['total_queries']
                }
            }
            
            detailed_results['indices'][identifier] = {
                'index_stats': index_stats,
                'latency': latency_results
            }
        
        # Save results
        self.results_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(self.results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {self.results_path}")
        
        with open(self.detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✓ Detailed results saved to: {self.detailed_results_path}")
        
        # Print comparison summary
        self.print_comparison(all_results)
        
        # Generate plots
        self.generate_plots(detailed_results)
        
        return all_results
    
    def print_comparison(self, results: Dict):
        """Print side-by-side comparison"""
        print("\n" + "="*70)
        print("SKIP POINTER OPTIMIZATION COMPARISON SUMMARY")
        print("="*70)
        
        indices = results['indices']
        index_keys = list(indices.keys())
        i0_key = [k for k in index_keys if 'o0' in k][0]  # o0 = No optimization
        i1_key = [k for k in index_keys if 'osp' in k][0]  # osp = Skip pointers
        
        i0 = indices[i0_key]
        i1 = indices[i1_key]
        
        print("\n--- Index Statistics ---")
        print(f"{'Metric':<40} {'i=0 (No Opt)':<20} {'i=1 (Skip)':<20}")
        print("-" * 80)
        
        stats_to_compare = [
            ('doc_count', 'Document Count', lambda x: f"{x:,}"),
            ('unique_terms', 'Unique Terms', lambda x: f"{x:,}"),
            ('avg_doc_length', 'Avg Doc Length', lambda x: f"{x:.2f}")
        ]
        
        for key, label, fmt in stats_to_compare:
            v0 = i0['index_stats'].get(key, 0)
            v1 = i1['index_stats'].get(key, 0)
            print(f"{label:<40} {fmt(v0):<20} {fmt(v1):<20}")
        
        # Skip pointer stats
        if 'total_skip_pointers' in i1['index_stats']:
            print(f"{'Skip Pointers':<40} {'N/A':<20} {i1['index_stats']['total_skip_pointers']:,}")
            print(f"{'Avg Skip Interval':<40} {'N/A':<20} {i1['index_stats']['avg_skip_interval']:.1f}")
            print(f"{'Skip Overhead (MB)':<40} {'N/A':<20} {i1['index_stats']['skip_overhead_bytes']/1024/1024:.2f}")
        
        print("\n--- Latency Statistics (Plot A) ---")
        print(f"{'Metric':<40} {'i=0 (No Opt)':<20} {'i=1 (Skip)':<20} {'Improvement':<15}")
        print("-" * 95)
        
        latency_metrics = [
            ('mean_ms', 'Mean Latency (ms)'),
            ('median_ms', 'Median Latency (ms)'),
            ('p50_ms', 'p50 Latency (ms)'),
            ('p95_ms', 'p95 Latency (ms)'),
            ('p99_ms', 'p99 Latency (ms)')
        ]
        
        for key, label in latency_metrics:
            v0 = i0['latency']['statistics'][key]
            v1 = i1['latency']['statistics'][key]
            improvement = ((v0 - v1) / v0 * 100) if v0 > 0 else 0
            sign = '+' if improvement > 0 else ''
            print(f"{label:<40} {v0:<20.2f} {v1:<20.2f} {sign}{improvement:.1f}%")
        
        print("\n" + "="*70)
    
    def generate_plots(self, results: Dict):
        """Generate visualization plots"""
        print("\nGenerating plots...")
        
        indices = results['indices']
        index_keys = list(indices.keys())
        i0_key = [k for k in index_keys if 'o0' in k][0]
        i1_key = [k for k in index_keys if 'osp' in k][0]
        
        i0 = indices[i0_key]
        i1 = indices[i1_key]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot A: Skip Pointer Optimization Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Latency Percentiles (Plot A)
        ax1 = axes[0, 0]
        percentiles = ['p50', 'p95', 'p99']
        percentile_labels = ['p50 (Median)', 'p95', 'p99']
        i0_values = [i0['latency']['statistics'][f'{p}_ms'] for p in percentiles]
        i1_values = [i1['latency']['statistics'][f'{p}_ms'] for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax1.bar(x - width/2, i0_values, width, label='i=0 (No Opt)', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, i1_values, width, label='i=1 (Skip Pointers)', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Percentile', fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontweight='bold')
        ax1.set_title('Plot A: Query Latency Percentiles', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(percentile_labels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Latency Distribution (Box Plot)
        ax2 = axes[0, 1]
        i0_latencies = i0['latency']['latencies_ms']
        i1_latencies = i1['latency']['latencies_ms']
        
        bp = ax2.boxplot([i0_latencies, i1_latencies], 
                          tick_labels=['i=0 (No Opt)', 'i=1 (Skip Pointers)'],
                          patch_artist=True,
                          showfliers=False)
        
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Latency (ms)', fontweight='bold')
        ax2.set_title('Latency Distribution (Box Plot)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Latency Improvement
        ax3 = axes[1, 0]
        metrics = ['Mean', 'Median', 'p95', 'p99']
        metric_keys = ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms']
        improvements = []
        
        for key in metric_keys:
            v0 = i0['latency']['statistics'][key]
            v1 = i1['latency']['statistics'][key]
            improvement = ((v0 - v1) / v0 * 100) if v0 > 0 else 0
            improvements.append(improvement)
        
        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax3.bar(metrics, improvements, alpha=0.8, color=colors_imp)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Improvement (%)', fontweight='bold')
        ax3.set_title('Latency Improvement with Skip Pointers', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # Plot 4: Skip Pointer Statistics
        ax4 = axes[1, 1]
        if 'total_skip_pointers' in i1['index_stats']:
            skip_info = [
                ('Total Skip\nPointers', i1['index_stats']['total_skip_pointers']),
                ('Avg Skip\nInterval', i1['index_stats']['avg_skip_interval']),
                ('Memory\nOverhead (MB)', i1['index_stats']['skip_overhead_bytes']/1024/1024)
            ]
            
            labels = [x[0] for x in skip_info]
            values = [x[1] for x in skip_info]
            
            bars = ax4.bar(labels, values, alpha=0.8, color='teal')
            ax4.set_title('Skip Pointer Statistics', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if val > 1000:
                    label = f'{val:,.0f}'
                else:
                    label = f'{val:.2f}'
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        label,
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Skip pointer stats not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Skip Pointer Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("data/plots/plot_a_skip_pointer_comparison.png")
        plot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
        
        plt.close()


def main():
    """Main execution"""
    benchmark = SkipPointerBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {benchmark.results_path}")
    print(f"  - {benchmark.detailed_results_path}")
    print(f"  - data/plots/plot_a_skip_pointer_comparison.png")


if __name__ == "__main__":
    main()
