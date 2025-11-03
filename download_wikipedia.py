"""
Simple script to download Wikipedia dataset from Hugging Face
No preprocessing - just download and save raw data
"""
import json
from pathlib import Path
from datasets import load_dataset

def download_wikipedia(split: str = "20231101.en", output_dir: str = "data", num_samples: int = 50000):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    raw_data_dir = output_path / "raw"
    raw_data_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {num_samples:,} Wikipedia articles...")
    
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            split,
            split="train",
            streaming=True
        )
        
        # Download all documents into memory
        documents = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
                
            doc = {
                'id': item.get('id', f'doc_{i}'),
                'url': item.get('url', ''),
                'title': item.get('title', ''),
                'text': item.get('text', '')
            }
            documents.append(doc)
        
        with open(raw_data_dir / "wikipedia_raw.json", 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(documents):,} documents")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    download_wikipedia()
