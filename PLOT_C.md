# Plot C: Information Type Comparison

Builds and compares three SelfIndex variants:

- **x=1**: Boolean index (doc IDs + positions)
- **x=2**: Word Count index (ranking)
- **x=3**: TF-IDF index (improved ranking)

## Usage

```bash
python plot_c_info_type.py
```

## Output

- **Indices**: `indices/SelfIndex-v1.{1,2,3}00T0.pkl`
- **Plot**: `data/plots/plot_c.png`
- **Results**: `plot_c_results.json`

## Metrics Measured

- **Memory Footprint** (Plot C): Index size on disk
- **Latency**: p95 query response time
- **Throughput**: Queries per second
