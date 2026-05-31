# Performance Analysis: CPU vs GPU for Spatial Boxcounting

## Hardware Configuration
- **GPU:** NVIDIA GeForce RTX 4060 Ti (16 GB VRAM, Compute Capability 8.9)
- **CUDA:** 12.9 (CuPy 14.1.0)
- **CPU:** Docker container on CachyOS Linux

## Benchmark Results (v0.2.0, batched GPU)

All windows are processed in a **single kernel launch** via 4D-tensor reshaping
and vectorised CuPy operations. No Python-level loops in the GPU path.

### boxsize=2 (iteration=0)

| Image Size | CPU (Numba JIT) | GPU (CuPy batched) | Speedup |
|-----------|-----------------|--------------------|---------|
| 128×128   | 0.010 s         | 0.001 s            | **9×** 🚀 |
| 256×256   | 0.041 s         | 0.005 s            | **9×** 🚀 |
| 512×512   | 0.164 s         | 0.049 s            | **3×** 🚀 |
| 1024×1024 | 0.661 s         | 0.255 s            | **2.6×** 🚀 |

### boxsize=4 (iteration=1)

| Image Size | CPU (Numba JIT) | GPU (CuPy batched) | Speedup |
|-----------|-----------------|--------------------|---------|
| 128×128   | 0.005 s         | 0.001 s            | **5×** 🚀 |
| 256×256   | 0.021 s         | 0.001 s            | **18×** 🚀 |
| 512×512   | 0.085 s         | 0.007 s            | **12×** 🚀 |
| 1024×1024 | 0.338 s         | 0.031 s            | **11×** 🚀 |

### boxsize=16 (iteration=3)

| Image Size | CPU (Numba JIT) | GPU (CuPy batched) | Speedup |
|-----------|-----------------|--------------------|---------|
| 128×128   | 0.001 s         | 0.001 s            | 1× (tie) |
| 256×256   | 0.004 s         | 0.001 s            | **4×** 🚀 |
| 512×512   | 0.017 s         | 0.001 s            | **16×** 🚀 |
| 1024×1024 | 0.066 s         | 0.002 s            | **35×** 🚀 |

### Key insight

GPU advantage grows with **larger boxsizes** (fewer windows, more pixels per
window → better GPU utilisation) and **larger images**. For boxsize=2 on
1024×1024 the speedup is modest (2.6×) because there are 262k windows each
containing only 4 pixels — the GPU is under-utilised. At boxsize=16 on the
same image: only 4k windows with 256 pixels each → **35× speedup**.

For images below 128×128, CPU (Numba JIT) is competitive and often simpler.

## Architecture: Batched 4D Tensor

The `spacialBoxcount_gpu` function in v0.2.0 uses a single batched kernel
launch:

```python
# 1. Reshape image into (ny, nx, boxsize, boxsize) 4D tensor
# 2. Floor division → integer box indices for ALL windows
# 3. Per-window unique count via sort + diff
# 4. Per-window histogram via offset cp.bincount (single call)
# 5. Lacunarity via cp.mean / cp.std on stacked histograms
```

This replaces the v0.1.0 approach of 4k–65k individual kernel launches
per image (which was 200–2000× slower than CPU).

## Performance Optimization Tips

### CPU (Numba JIT)
- First run is slower due to JIT compilation — subsequent runs use cached code
- Multi-scale processing: run multiple iterations in parallel with Python threads
- Numba `nopython=True` releases the GIL

### GPU (CuPy)
- Install matching CuPy version: `pip install spacial_boxcounting[gpu]` (NVIDIA) or `spacial_boxcounting[gpu-amd]` (AMD ROCm)
- For maximum throughput: process multiple images in a batch

### Cross-Platform
| Platform | GPU Support | Notes |
|----------|------------|-------|
| Linux    | NVIDIA ✓, AMD ✓ (ROCm) | Best performance |
| Windows  | NVIDIA ✓ | CUDA drivers required |
| macOS    | CPU only | No NVIDIA/AMD GPU support |

## Memory Considerations
- GPU: images and intermediate tensors must fit in VRAM (~16 GB on RTX 4060 Ti)
- The batched histogram has shape `(N_windows, MaxValue/boxsize)` — for 1024×1024
  with boxsize=2 this is (262k, 128) ≈ 268 MB. Acceptable for 16 GB VRAM.
- CPU: limited by system RAM
