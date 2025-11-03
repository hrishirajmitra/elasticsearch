"""
Preprocess Wikipedia data: tokenization, stemming, stopword removal, punctuation handling
Generate word frequency plots with and without preprocessing
"""
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Set plotting style
sns.set_style("whitegrid")

class TextPreprocessor:
    def __init__(self, data_dir: str = "data"):
        """Initialize the preprocessor"""
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.plots_dir = self.data_dir / "plots"
        
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Download NLTK resources
        self._download_nltk_resources()
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def load_raw_data(self, filename: str = "wikipedia_raw.json") -> List[Dict]:
        with open(self.raw_data_dir / filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def tokenize_without_preprocessing(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def tokenize_with_preprocessing(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        tokens = [token for token in tokens if not token.isdigit() and len(token) >= 2]
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def process_documents(self, documents: List[Dict]) -> Tuple[List[Dict], Counter, Counter]:
        print("Processing documents...")
        
        processed_docs = []
        all_tokens_raw = []
        all_tokens_processed = []
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '')
            title = doc.get('title', '')
            full_text = f"{title}. {text}"
            
            # Tokenize without preprocessing (minimal processing)
            tokens_raw = self.tokenize_without_preprocessing(full_text)
            all_tokens_raw.extend(tokens_raw)
            
            # Tokenize with full preprocessing
            tokens_processed = self.tokenize_with_preprocessing(full_text)
            all_tokens_processed.extend(tokens_processed)
            
            processed_doc = {
                'id': doc.get('id'),
                'title': title,
                'original_text': text,
                'tokens_raw': tokens_raw,
                'tokens_processed': tokens_processed
            }
            processed_docs.append(processed_doc)
            
        freq_without = Counter(all_tokens_raw)
        freq_with = Counter(all_tokens_processed)
        
        print(f"Unique tokens: {len(freq_without):,} raw, {len(freq_with):,} processed")
        
        return processed_docs, freq_without, freq_with
    
    def plot_word_frequencies(self, freq_without: Counter, freq_with: Counter, top_n: int = 30):
        print("Generating plots...")
        
        # Plot 1: Top N words comparison
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
        plot_file1 = self.plots_dir / "word_frequencies_comparison.png"
        plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_file1}")
        plt.close()
        
        # Plot 2: Frequency distribution (Zipf's Law)
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
        
        plt.savefig(self.plots_dir / "frequency_distribution_zipf.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2b: Non-log
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
        
        plt.savefig(self.plots_dir / "frequency_distribution_linear.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Statistics
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
        
        plt.savefig(self.plots_dir / "preprocessing_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_processed_data(self, processed_docs: List[Dict], 
                           freq_without: Counter, freq_with: Counter):
        with open(self.processed_data_dir / "wikipedia_processed.json", 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
        freq_data = {
            'without_preprocessing': dict(freq_without.most_common(10000)),
            'with_preprocessing': dict(freq_with.most_common(10000))
        }
        with open(self.processed_data_dir / "word_frequencies.json", 'w', encoding='utf-8') as f:
            json.dump(freq_data, f, indent=2)
        
        # Save statistics summary
        stats = {
            'num_documents': len(processed_docs),
            'without_preprocessing': {
                'unique_tokens': len(freq_without),
                'total_tokens': sum(freq_without.values()),
                'avg_token_length': sum(len(w) * c for w, c in freq_without.items()) / sum(freq_without.values()),
                'top_10_words': freq_without.most_common(10)
            },
            'with_preprocessing': {
                'unique_tokens': len(freq_with),
                'total_tokens': sum(freq_with.values()),
                'avg_token_length': sum(len(w) * c for w, c in freq_with.items()) / sum(freq_with.values()),
                'top_10_words': freq_with.most_common(10)
            }
        }
        
        with open(self.processed_data_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    
    def run(self):
        print("TEXT PREPROCESSING PIPELINE")
        documents = self.load_raw_data()
        
        processed_docs, freq_without, freq_with = self.process_documents(documents)
        self.plot_word_frequencies(freq_without, freq_with)
        self.save_processed_data(processed_docs, freq_without, freq_with)
        print(f"Complete: {len(processed_docs):,} documents")


def main():
    preprocessor = TextPreprocessor(data_dir="data")
    preprocessor.run()


if __name__ == "__main__":
    main()
