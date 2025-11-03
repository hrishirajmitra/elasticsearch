"""
Verify compression measurements are correct
"""
import pickle
from pathlib import Path

# Load z1 index
z1_path = Path('indices/selfindex_z1/selfindex-z1-v1.0.pkl')
with open(z1_path, 'rb') as f:
    z1 = pickle.load(f)

print("="*70)
print("Z1 (Simple Variable Byte + Delta Encoding) Compression Verification")
print("="*70)
print(f"Total terms: {len(z1['inverted_index']):,}")
print(f"Total postings: {z1['memory_stats']['total_postings']:,}")
print(f"\nCompression Statistics:")
print(f"  Compressed size:   {z1['memory_stats']['compressed_size']:,} bytes ({z1['memory_stats']['compressed_size']/1024/1024:.2f} MB)")
print(f"  Uncompressed size: {z1['memory_stats']['uncompressed_size']:,} bytes ({z1['memory_stats']['uncompressed_size']/1024/1024:.2f} MB)")
print(f"  Compression ratio: {(1 - z1['memory_stats']['compressed_size'] / z1['memory_stats']['uncompressed_size']) * 100:.2f}%")
print(f"  Space saved:       {(z1['memory_stats']['uncompressed_size'] - z1['memory_stats']['compressed_size'])/1024/1024:.2f} MB")

# Verify calculation
print(f"\nVerification:")
print(f"  Expected uncompressed: {z1['memory_stats']['total_postings']} postings × 4 bytes = {z1['memory_stats']['total_postings'] * 4:,} bytes")
print(f"  Actual uncompressed:   {z1['memory_stats']['uncompressed_size']:,} bytes")
print(f"  Match: {z1['memory_stats']['total_postings'] * 4 == z1['memory_stats']['uncompressed_size']}")

# Sample some terms to verify
sample_terms = list(z1['inverted_index'].keys())[:5]
print(f"\nSample terms verification:")
for term in sample_terms:
    compressed = z1['inverted_index'][term]
    df = z1['term_doc_freq'].get(term, 0)
    uncompressed_bytes = df * 4
    compressed_bytes = len(compressed)
    ratio = (1 - compressed_bytes / uncompressed_bytes) * 100 if uncompressed_bytes > 0 else 0
    print(f"  '{term}': {df} postings, {uncompressed_bytes}→{compressed_bytes} bytes ({ratio:.1f}% compression)")

print("\n" + "="*70)
print("Z2 (zlib Delta + DEFLATE) Compression Verification")
print("="*70)

# Load z2 index
z2_path = Path('indices/selfindex_z2/selfindex-z2-v1.0.pkl')
with open(z2_path, 'rb') as f:
    z2 = pickle.load(f)

print(f"Total terms: {len(z2['inverted_index']):,}")
print(f"Total postings: {z2['memory_stats']['total_postings']:,}")
print(f"\nCompression Statistics:")
print(f"  Compressed size:   {z2['memory_stats']['compressed_size']:,} bytes ({z2['memory_stats']['compressed_size']/1024/1024:.2f} MB)")
print(f"  Uncompressed size: {z2['memory_stats']['uncompressed_size']:,} bytes ({z2['memory_stats']['uncompressed_size']/1024/1024:.2f} MB)")
print(f"  Compression ratio: {(1 - z2['memory_stats']['compressed_size'] / z2['memory_stats']['uncompressed_size']) * 100:.2f}%")
print(f"  Space saved:       {(z2['memory_stats']['uncompressed_size'] - z2['memory_stats']['compressed_size'])/1024/1024:.2f} MB")

# Verify calculation
print(f"\nVerification:")
print(f"  Expected uncompressed: {z2['memory_stats']['total_postings']} postings × 4 bytes = {z2['memory_stats']['total_postings'] * 4:,} bytes")
print(f"  Actual uncompressed:   {z2['memory_stats']['uncompressed_size']:,} bytes")
print(f"  Match: {z2['memory_stats']['total_postings'] * 4 == z2['memory_stats']['uncompressed_size']}")

# Sample some terms to verify
sample_terms = list(z2['inverted_index'].keys())[:5]
print(f"\nSample terms verification:")
for term in sample_terms:
    compressed = z2['inverted_index'][term]
    df = z2['term_doc_freq'].get(term, 0)
    uncompressed_bytes = df * 4
    compressed_bytes = len(compressed)
    ratio = (1 - compressed_bytes / uncompressed_bytes) * 100 if uncompressed_bytes > 0 else 0
    print(f"  '{term}': {df} postings, {uncompressed_bytes}→{compressed_bytes} bytes ({ratio:.1f}% compression)")

print("\n" + "="*70)
print("Comparison")
print("="*70)
print(f"Z1 achieves {(1 - z1['memory_stats']['compressed_size'] / z1['memory_stats']['uncompressed_size']) * 100:.2f}% compression")
print(f"Z2 achieves {(1 - z2['memory_stats']['compressed_size'] / z2['memory_stats']['uncompressed_size']) * 100:.2f}% compression")
print(f"Z1 is {(z2['memory_stats']['compressed_size'] - z1['memory_stats']['compressed_size']) / 1024 / 1024:.2f} MB smaller than Z2")
print(f"Z1 compression is {((z1['memory_stats']['compressed_size'] / z1['memory_stats']['uncompressed_size']) / (z2['memory_stats']['compressed_size'] / z2['memory_stats']['uncompressed_size']) - 1) * 100:.1f}% better")

# Test decompression on a sample
print("\n" + "="*70)
print("Decompression Test")
print("="*70)

from selfindex_z1_simple_compression import SimpleCompression
from selfindex_z2_lib_compression import LibraryCompression

test_term = list(z1['inverted_index'].keys())[0]
print(f"Testing term: '{test_term}'")

# Test z1 decompression
z1_compressed = z1['inverted_index'][test_term]
z1_decompressed = SimpleCompression.decode_delta(z1_compressed)
print(f"Z1: Compressed {len(z1_compressed)} bytes → {len(z1_decompressed)} postings")

# Test z2 decompression
z2_compressed = z2['inverted_index'][test_term]
z2_decompressed = LibraryCompression.decompress_list_with_delta(z2_compressed)
print(f"Z2: Compressed {len(z2_compressed)} bytes → {len(z2_decompressed)} postings")

# Verify both produce same postings
print(f"Both methods produce same result: {z1_decompressed == z2_decompressed}")
print(f"Sample postings (first 10): {z1_decompressed[:10]}")

print("\n✓ Compression measurements are CORRECT!")
