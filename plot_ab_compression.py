"""
Plot AB: Compression Method Comparison
Measures latency (A) and throughput (B) for compression methods:
- z=1: Simple Variable Byte encoding (CODE)
- z=2: Library compression with zlib (CLIB)

Metrics:
- Plot A: Response time with p95 and p99 percentiles
- Plot B: Throughput in queries/second
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the two index implementations
from selfindex_z1_simple_compression import SelfIndexZ1
from selfindex_z2_lib_compression import SelfIndexZ2


class CompressionBenchmark:
    """Benchmark compression methods for inverted indices"""
    
    def __init__(self):
        self.queryset_path = Path("queryset.json")
        self.results_path = Path("data/plots/plot_ab_results.json")
        self.detailed_results_path = Path("data/plots/plot_ab_detailed_results.json")
        
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
        
        # Warmup phase - run queries to warm up caches
        print(f"  Warmup phase ({warmup_rounds} rounds)...")
        for _ in range(warmup_rounds):
            for query in queries[:10]:  # Use subset for warmup
                indexer.query(query, size=10)
        
        # Actual measurement
        print(f"  Executing {len(queries)} queries...")
        
        # --- OPTIMIZATION 1: Pre-allocate NumPy array ---
        # This avoids costly list.append() and list-to-array conversion
        num_queries = len(queries)
        latencies = np.empty(num_queries, dtype=np.float64)
        
        query_details = []
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            result = indexer.query(query, size=10)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies[i] = latency_ms # Assign directly to the array
            
            # Parse result to get hit count
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
            
            # --- OPTIMIZATION 2: Removed print statement from hot loop ---
            # Printing every 20 queries adds significant I/O overhead 
            # and slows down the benchmark execution.
            # if (i + 1) % 20 == 0:
            #     print(f"    Processed {i + 1}/{len(queries)} queries...")
        
        print(f"  ...all {num_queries} queries processed.") # Single confirmation print

        # Calculate statistics
        # No need for np.array(latencies) as we already have an array
        
        results = {
            'indexer': indexer.identifier_short,
            'total_queries': len(queries),
            # Convert back to list for JSON serialization
            'latencies_ms': latencies.tolist(), 
            'query_details': query_details,
            'statistics': {
                'mean_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'p50_ms': float(np.percentile(latencies, 50)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
                'p999_ms': float(np.percentile(latencies, 99.9))
            }
        }
        
        print(f"  ✓ Latency measurement complete!")
        print(f"    Mean: {results['statistics']['mean_ms']:.2f} ms")
        print(f"    Median (p50): {results['statistics']['p50_ms']:.2f} ms")
        print(f"    p95: {results['statistics']['p95_ms']:.2f} ms")
        print(f"    p99: {results['statistics']['p99_ms']:.2f} ms")
        
        return results
    
    def measure_throughput(self, indexer, queries: List[str], duration_seconds: int = 30) -> Dict:
        """
        Measure query throughput (queries per second)
        Runs queries continuously for specified duration
        """
        print(f"\nMeasuring throughput for {indexer.identifier_short}...")
        print(f"  Running queries for {duration_seconds} seconds...")
        
        query_count = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        # Cycle through queries
        query_idx = 0
        
        while time.perf_counter() < end_time:
            query = queries[query_idx % len(queries)]
            indexer.query(query, size=10)
            query_count += 1
            query_idx += 1
            
            # --- OPTIMIZATION: Removed print statement from hot loop ---
            # Printing here is a major performance bottleneck. It's an I/O operation
            # that "steals" time from the benchmark, directly reducing the
            # measured queries_per_second (QPS).
            # if query_count % 100 == 0:
            #     elapsed = time.perf_counter() - start_time
            #     current_qps = query_count / elapsed
            #     print(f"    {query_count} queries, {elapsed:.1f}s elapsed, {current_qps:.1f} QPS...")
        
        actual_duration = time.perf_counter() - start_time
        qps = query_count / actual_duration
        
        results = {
            'indexer': indexer.identifier_short,
            'total_queries': query_count,
            'duration_seconds': actual_duration,
            'queries_per_second': qps,
            'avg_query_time_ms': (actual_duration / query_count) * 1000
        }
        
        print(f"  ✓ Throughput measurement complete!")
        print(f"    Total queries: {query_count:,}")
        print(f"    Duration: {actual_duration:.2f}s")
        print(f"    Throughput: {qps:.2f} queries/second")
        
        return results
    
    def run_benchmark(self, measure_throughput: bool = True, throughput_duration: int = 30):
        """Run complete benchmark for both compression methods"""
        
        print("="*70)
        print("PLOT AB: COMPRESSION METHOD COMPARISON BENCHMARK")
        print("="*70)
        
        # Initialize indexers
        print("\nInitializing indexers...")
        z1_indexer = SelfIndexZ1()
        z2_indexer = SelfIndexZ2()
        
        # Load indices
        print("\n--- Loading Index z=1 (Simple Compression) ---")
        z1_indexer.load_index('selfindex-z1-v1.0')
        
        print("\n--- Loading Index z=2 (Library Compression) ---")
        z2_indexer.load_index('selfindex-z2-v1.0')
        
        if not z1_indexer.inverted_index or not z2_indexer.inverted_index:
            print("\n❌ Error: One or both indices not loaded properly!")
            print("Please run the index creation scripts first:")
            print("  python selfindex_z1_simple_compression.py")
            print("  python selfindex_z2_lib_compression.py")
            return
        
        # Store all results
        all_results = {
            'metadata': {
                'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'query_set_size': len(self.queries),
                'query_set_metadata': self.metadata
            },
            'indices': {}
        }
        
        detailed_results = {
            'metadata': all_results['metadata'].copy(),
            'indices': {}
        }
        
        # Benchmark each indexer
        for indexer in [z1_indexer, z2_indexer]:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING: {indexer.identifier_short}")
            print(f"{'='*70}")
            
            # Get index stats
            index_stats = indexer.get_stats()
            
            # Measure latency (Plot A)
            latency_results = self.measure_latency(indexer, self.queries)
            
            # Measure throughput (Plot B)
            if measure_throughput:
                throughput_results = self.measure_throughput(
                    indexer, 
                    self.queries, 
                    duration_seconds=throughput_duration
                )
            else:
                throughput_results = None
            
            # Store results
            identifier = indexer.identifier_short
            
            all_results['indices'][identifier] = {
                'index_stats': index_stats,
                'latency': {
                    'statistics': latency_results['statistics'],
                    'total_queries': latency_results['total_queries']
                },
                'throughput': throughput_results
            }
            
            detailed_results['indices'][identifier] = {
                'index_stats': index_stats,
                'latency': latency_results,
                'throughput': throughput_results
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
        print("COMPRESSION METHOD COMPARISON SUMMARY")
        print("="*70)
        
        indices = results['indices']
        # Get keys based on compression type in identifier
        index_keys = list(indices.keys())
        z1_key = [k for k in index_keys if 'c2' in k][0]  # c2 = CODE compression
        z2_key = [k for k in index_keys if 'c3' in k][0]  # c3 = CLIB compression
        
        z1 = indices[z1_key]
        z2 = indices[z2_key]
        
        print("\n--- Index Statistics ---")
        print(f"{'Metric':<40} {'z=1 (CODE)':<20} {'z=2 (CLIB)':<20}")
        print("-" * 80)
        
        stats_to_compare = [
            ('doc_count', 'Document Count', lambda x: f"{x:,}"),
            ('unique_terms', 'Unique Terms', lambda x: f"{x:,}"),
            ('total_postings', 'Total Postings', lambda x: f"{x:,}"),
            ('compressed_size', 'Compressed Size (MB)', lambda x: f"{x/1024/1024:.2f}"),
            ('uncompressed_size', 'Uncompressed Size (MB)', lambda x: f"{x/1024/1024:.2f}")
        ]
        
        for key, label, fmt in stats_to_compare:
            v1 = z1['index_stats'].get(key, 0)
            v2 = z2['index_stats'].get(key, 0)
            print(f"{label:<40} {fmt(v1):<20} {fmt(v2):<20}")
        
        # Compression ratio
        cr1 = (1 - z1['index_stats']['compressed_size'] / z1['index_stats']['uncompressed_size']) * 100
        cr2 = (1 - z2['index_stats']['compressed_size'] / z2['index_stats']['uncompressed_size']) * 100
        print(f"{'Compression Ratio (%)':<40} {cr1:<20.2f} {cr2:<20.2f}")
        
        print("\n--- Latency Statistics (Plot A) ---")
        print(f"{'Metric':<40} {'z=1 (CODE)':<20} {'z=2 (CLIB)':<20} {'Improvement':<15}")
        print("-" * 95)
        
        latency_metrics = [
            ('mean_ms', 'Mean Latency (ms)'),
            ('median_ms', 'Median Latency (ms)'),
            ('p50_ms', 'p50 Latency (ms)'),
            ('p95_ms', 'p95 Latency (ms)'),
            ('p99_ms', 'p99 Latency (ms)')
        ]
        
        for key, label in latency_metrics:
            v1 = z1['latency']['statistics'][key]
            v2 = z2['latency']['statistics'][key]
            improvement = ((v1 - v2) / v1 * 100) if v1 > 0 else 0
            sign = '+' if improvement > 0 else ''
            print(f"{label:<40} {v1:<20.2f} {v2:<20.2f} {sign}{improvement:.1f}%")
        
        if z1.get('throughput') and z2.get('throughput'):
            print("\n--- Throughput Statistics (Plot B) ---")
            print(f"{'Metric':<40} {'z=1 (CODE)':<20} {'z=2 (CLIB)':<20} {'Improvement':<15}")
            print("-" * 95)
            
            qps1 = z1['throughput']['queries_per_second']
            qps2 = z2['throughput']['queries_per_second']
            improvement = ((qps2 - qps1) / qps1 * 100) if qps1 > 0 else 0
            sign = '+' if improvement > 0 else ''
            
            print(f"{'Queries per Second':<40} {qps1:<20.2f} {qps2:<20.2f} {sign}{improvement:.1f}%")
            print(f"{'Avg Query Time (ms)':<40} {z1['throughput']['avg_query_time_ms']:<20.2f} {z2['throughput']['avg_query_time_ms']:<20.2f}")
        
        print("\n" + "="*70)
    
    def generate_plots(self, results: Dict):
        """Generate visualization plots"""
        print("\nGenerating plots...")
        
        indices = results['indices']
        # Get keys based on compression type in identifier
        index_keys = list(indices.keys())
        z1_key = [k for k in index_keys if 'c2' in k][0]  # c2 = CODE compression
        z2_key = [k for k in index_keys if 'c3' in k][0]  # c3 = CLIB compression
        
        z1 = indices[z1_key]
        z2 = indices[z2_key]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot AB: Compression Method Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Latency Percentiles (Plot A)
        ax1 = axes[0, 0]
        percentiles = ['p50', 'p95', 'p99']
        percentile_labels = ['p50 (Median)', 'p95', 'p99']
        z1_values = [z1['latency']['statistics'][f'{p}_ms'] for p in percentiles]
        z2_values = [z2['latency']['statistics'][f'{p}_ms'] for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax1.bar(x - width/2, z1_values, width, label='z=1 (CODE)', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, z2_values, width, label='z=2 (CLIB)', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Percentile', fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontweight='bold')
        ax1.set_title('Plot A: Query Latency Percentiles', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(percentile_labels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Latency Distribution (Box Plot)
        ax2 = axes[0, 1]
        z1_latencies = z1['latency']['latencies_ms']
        z2_latencies = z2['latency']['latencies_ms']
        
        bp = ax2.boxplot([z1_latencies, z2_latencies], 
                            labels=['z=1 (CODE)', 'z=2 (CLIB)'],
                            patch_artist=True,
                            showfliers=False)
        
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Latency (ms)', fontweight='bold')
        ax2.set_title('Latency Distribution (Box Plot)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Compression Comparison
        ax3 = axes[1, 0]
        methods = ['z=1 (CODE)', 'z=2 (CLIB)']
        compressed_sizes = [
            z1['index_stats']['compressed_size'] / 1024 / 1024,
            z2['index_stats']['compressed_size'] / 1024 / 1024
        ]
        uncompressed_size = z1['index_stats']['uncompressed_size'] / 1024 / 1024
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax3.bar(x - width/2, [uncompressed_size] * 2, width, 
                label='Uncompressed', alpha=0.8, color='lightgray')
        ax3.bar(x + width/2, compressed_sizes, width, 
                label='Compressed', alpha=0.8, color='green')
        
        ax3.set_ylabel('Size (MB)', fontweight='bold')
        ax3.set_title('Index Size Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Throughput Comparison (Plot B)
        ax4 = axes[1, 1]
        if z1.get('throughput') and z2.get('throughput'):
            methods = ['z=1 (CODE)', 'z=2 (CLIB)']
            throughputs = [
                z1['throughput']['queries_per_second'],
                z2['throughput']['queries_per_second']
            ]
            
            bars = ax4.bar(methods, throughputs, alpha=0.8, color=['steelblue', 'coral'])
            ax4.set_ylabel('Queries per Second', fontweight='bold')
            ax4.set_title('Plot B: Query Throughput', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Throughput measurement not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Plot B: Query Throughput', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("data/plots/plot_ab_compression_comparison.png")
        plot_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
        
        plt.close()


def main():
    """Main execution"""
    benchmark = CompressionBenchmark()
    
    # Run benchmark with throughput measurement
    # Set throughput_duration lower for faster testing (default: 30 seconds)
    results = benchmark.run_benchmark(
        measure_throughput=True,
        throughput_duration=30  # Adjust as needed
    )
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {benchmark.results_path}")
    print(f"  - {benchmark.detailed_results_path}")
    print(f"  - data/plots/plot_ab_compression_comparison.png")


if __name__ == "__main__":
    main()