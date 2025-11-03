"""
Generate word frequency plots from already processed data
This script creates visualizations comparing word frequencies with and without text preprocessing
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")

def load_word_frequencies(data_dir: str = "data"):
    """Load word frequencies from processed data"""
    processed_dir = Path(data_dir) / "processed"
    
    with open(processed_dir / "word_frequencies.json", 'r', encoding='utf-8') as f:
        freq_data = json.load(f)
    
    freq_without = Counter(freq_data['without_preprocessing'])
    freq_with = Counter(freq_data['with_preprocessing'])
    
    return freq_without, freq_with

def generate_plots(freq_without: Counter, freq_with: Counter, output_dir: str = "data/plots", top_n: int = 30):
    """Generate all word frequency plots"""
    plots_dir = Path(output_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating word frequency plots...")
    print(f"Unique tokens: {len(freq_without):,} raw, {len(freq_with):,} processed")
    
    # Plot 1: Top N words comparison
    print(f"  Creating top {top_n} words comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Without preprocessing
    words_raw, counts_raw = zip(*freq_without.most_common(top_n))
    ax1.barh(range(len(words_raw)), counts_raw, color='skyblue')
    ax1.set_yticks(range(len(words_raw)))
    ax1.set_yticklabels(words_raw, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency', fontsize=11)
    ax1.set_title(f'Top {top_n} Words WITHOUT Preprocessing\n(Basic tokenization only)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # With preprocessing
    words_proc, counts_proc = zip(*freq_with.most_common(top_n))
    ax2.barh(range(len(words_proc)), counts_proc, color='lightcoral')
    ax2.set_yticks(range(len(words_proc)))
    ax2.set_yticklabels(words_proc, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Frequency', fontsize=11)
    ax2.set_title(f'Top {top_n} Words WITH Preprocessing\n(Stemming + Stopword removal)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file1 = plots_dir / "word_frequencies_comparison.png"
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file1}")
    plt.close()
    
    # Plot 2: Frequency distribution (Zipf's Law) - Log scale
    print("  Creating Zipf's law plot (log scale)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    freqs_raw = sorted(freq_without.values(), reverse=True)
    freqs_proc = sorted(freq_with.values(), reverse=True)
    
    ax.plot(range(1, min(10000, len(freqs_raw)) + 1), 
            freqs_raw[:10000], 
            label='Without Preprocessing', 
            alpha=0.7, 
            linewidth=2)
    ax.plot(range(1, min(10000, len(freqs_proc)) + 1), 
            freqs_proc[:10000], 
            label='With Preprocessing', 
            alpha=0.7, 
            linewidth=2)
    
    ax.set_xlabel('Rank (log scale)', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Word Frequency Distribution (Zipf\'s Law)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plot_file2 = plots_dir / "frequency_distribution_zipf.png"
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file2}")
    plt.close()
    
    # Plot 3: Linear scale distribution
    print("  Creating frequency distribution (linear scale)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Show top 1000 for non-log scale
    top_n_dist = 1000
    ax.plot(range(1, min(top_n_dist, len(freqs_raw)) + 1), 
            freqs_raw[:top_n_dist], 
            label='Without Preprocessing', 
            alpha=0.7, 
            linewidth=2,
            marker='o',
            markersize=2)
    ax.plot(range(1, min(top_n_dist, len(freqs_proc)) + 1), 
            freqs_proc[:top_n_dist], 
            label='With Preprocessing', 
            alpha=0.7, 
            linewidth=2,
            marker='s',
            markersize=2)
    
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Word Frequency Distribution - Top {top_n_dist} (Linear Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plot_file3 = plots_dir / "frequency_distribution_linear.png"
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file3}")
    plt.close()
    
    # Plot 4: Statistics comparison
    print("  Creating preprocessing statistics plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Unique\nTokens', 'Total\nTokens', 'Avg Token\nLength']
    without_values = [
        len(freq_without),
        sum(freq_without.values()),
        sum(len(word) * count for word, count in freq_without.items()) / sum(freq_without.values())
    ]
    with_values = [
        len(freq_with),
        sum(freq_with.values()),
        sum(len(word) * count for word, count in freq_with.items()) / sum(freq_with.values())
    ]
    
    x = range(len(categories))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], without_values, width, 
           label='Without Preprocessing', color='skyblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], with_values, width, 
           label='With Preprocessing', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Count / Average', fontsize=12)
    ax.set_title('Preprocessing Impact Statistics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(without_values, with_values)):
        ax.text(i - width/2, v1, f'{v1:,.0f}' if i < 2 else f'{v1:.1f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2, f'{v2:,.0f}' if i < 2 else f'{v2:.1f}', 
               ha='center', va='bottom', fontsize=9)
    
    plot_file4 = plots_dir / "preprocessing_statistics.png"
    plt.savefig(plot_file4, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file4}")
    plt.close()
    
    print(f"\n✓ All plots generated successfully in {plots_dir}/")
    print(f"  - word_frequencies_comparison.png")
    print(f"  - frequency_distribution_zipf.png")
    print(f"  - frequency_distribution_linear.png")
    print(f"  - preprocessing_statistics.png")

def main():
    print("=" * 60)
    print("WORD FREQUENCY PLOTS GENERATOR")
    print("=" * 60)
    
    # Load word frequencies from processed data
    freq_without, freq_with = load_word_frequencies()
    
    # Generate all plots
    generate_plots(freq_without, freq_with)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
