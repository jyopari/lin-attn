# lin-attn

Linear attention mechanisms and benchmarks.

## Profiling

To profile the code with Nsight Compute, use:

```bash
ncu --set full --import-source yes --nvtx --profile-from-start off -f -o out python ncu.py
```

## Reproducing Results

1. Run `get_stats.py` to obtain the performance data
2. Run `make_plots.py` to reproduce the figures in the paper
