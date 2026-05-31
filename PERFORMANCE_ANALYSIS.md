# Performance Analysis: CPU vs GPU for Spatial Boxcounting

## Hardware Configuration
- **GPU:** NVIDIA GeForce RTX 4060 Ti (16 GB VRAM, Compute Capability 8.9)
- **CUDA:** 12.9 (CuPy 14.1.0)
- **CPU:** Docker container on CachyOS Linux

## Benchmark Results (boxsize=2, iteration=0, 8-bit data)

| Image Size | CPU (Numba JIT) | GPU (CuPy) | Notes |
|-----------|-----------------|------------|-------|
| 128×128   | 0.010 s         | 2.48 s     | GPU launch overhead dominates |
| 256×256   | 0.042 s         | 9.93 s     | Python loop bottleneck |
| 512×512   | 0.093 s*        | —          | GPU timed out |
| 1024×1024 | 0.383 s*        | —          | — |

*\* historical baseline (CPU-bound Numba JIT)*

## Current GPU Bottleneck

The `spacialBoxcount_gpu` function processes each sliding window in a **Python-level nested loop**, launching one GPU kernel per window:

- 128×128, boxsize=2 → ~4,096 kernel launches
- 512×512, boxsize=2 → ~65,536 kernel launches

Each launch incurs GPU scheduling overhead. The `Z_boxcount_gpu` function itself is fully vectorized (via `cp.bincount`), but the outer loop kills performance.

### Fix Plan (next release)
Replace the Python-level sliding window loop with a **batched 4D tensor approach**:

```python
# Instead of: for each window, call Z_boxcount_gpu()
# Do:               reshape into (ny, nx, bs, bs) and process all windows at once
```

Expected: **2–10× speedup** over CPU for images ≥ 512×512.

## Performance Optimization Tips

### CPU (Numba JIT)
- First run is slower due to JIT compilation — subsequent runs use cached code
- Multi-scale processing: run multiple iterations in parallel with Python threads
- Numba `nopython=True` releases the GIL

### GPU (CuPy)
- **Current limitation:** only beneficial for batch processing where the Python loop overhead is amortised
- For per-image processing: CPU (Numba JIT) is faster until GPU loop is batched
- Install matching CuPy version: `pip install spacial_boxcounting[gpu]` (NVIDIA) or `spacial_boxcounting[gpu-amd]` (AMD ROCm)

### Cross-Platform
| Platform | GPU Support | Notes |
|----------|------------|-------|
| Linux    | NVIDIA ✓, AMD ✓ (ROCm) | Best performance |
| Windows  | NVIDIA ✓ | CUDA drivers required |
| macOS    | CPU only | No NVIDIA/AMD GPU support |

## Memory Considerations
- GPU: images must fit in VRAM (~16 GB available on RTX 4060 Ti)
- CPU: limited by system RAM
- Future batched GPU processing will increase memory usage proportionally to batch size
