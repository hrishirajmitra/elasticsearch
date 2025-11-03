#!/usr/bin/env python3
"""
Plot query latency for different datastores from latency_results.json.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_latency_plot(results):
    """Generate Plot for Datastore Latency Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [r.title() for r in results.keys()]
    latency_p95 = [r['latency_p95'] for r in results.values()]
    latency_p99 = [r['latency_p99'] for r in results.values()]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    
    x = range(len(labels))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], latency_p95, width, 
                    label='p95', color=colors[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], latency_p99, width,
                    label='p99', color=colors[1], alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Query Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Query Latency by Datastore', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("data/plots/latency_comparison.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: data/plots/latency_comparison.png")
    plt.show()


def main():
    print("="*80)
    print("PLOTTING LATENCY RESULTS")
    print("="*80)

    try:
        with open("latency_results.json", 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: latency_results.json not found.")
        print("Please run measure_latency.py first to generate the latency data.")
        return

    if results:
        generate_latency_plot(results)
    else:
        print("No latency data to plot.")

if __name__ == "__main__":
    main()
