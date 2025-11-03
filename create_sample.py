#!/usr/bin/env python3
"""Create a smaller sample from the large JSON file for testing"""

import json

print("Creating sample dataset...")
print("Reading first 5000 documents from large file...")

with open("data/processed/wikipedia_processed.json", 'r', encoding='utf-8') as f:
    # Read opening bracket
    f.read(1)
    
    docs = []
    buffer = ""
    brace_count = 0
    in_doc = False
    
    for line in f:
        buffer += line
        
        # Count braces to detect complete documents
        for char in line:
            if char == '{':
                brace_count += 1
                in_doc = True
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and in_doc:
                    # Complete document
                    try:
                        doc_str = buffer.strip().rstrip(',')
                        doc = json.loads(doc_str)
                        docs.append(doc)
                        buffer = ""
                        in_doc = False
                        
                        if len(docs) >= 5000:
                            break
                    except:
                        pass
        
        if len(docs) >= 5000:
            break

print(f"Extracted {len(docs)} documents")

# Save sample
with open("data/processed/wikipedia_sample_5k.json", 'w', encoding='utf-8') as f:
    json.dump(docs, f)

print("✓ Saved to: data/processed/wikipedia_sample_5k.json")
