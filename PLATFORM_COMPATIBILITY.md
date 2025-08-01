"""
Platform Compatibility Guide for Spatial Boxcounting
=====================================================

This document provides guidance for making the spatial boxcounting package
accessible to users with different hardware configurations including AMD GPUs.

Universal Compatibility Strategy:
---------------------------------

1. CPU-Only Baseline (Works everywhere):
   - Primary implementation using Numba JIT compilation
   - Multi-threading support for better CPU utilization
   - Optimized memory usage for large arrays

2. NVIDIA GPU Support (CUDA):
   - CuPy-based implementation for parallel processing
   - Requires: NVIDIA GPU + CUDA drivers + CUDA toolkit
   - Installation: pip install cupy-cuda12x

3. AMD GPU Support (ROCm - Experimental):
   - CuPy has experimental ROCm support
   - Requires: AMD GPU + ROCm stack
   - Installation: pip install cupy-rocm-5.0 (version depends on ROCm version)

4. Web-Based Alternative:
   - Cloud-based processing for users without GPUs
   - Jupyter notebook with Google Colab integration
   - REST API for remote processing

Platform-Specific Installation:
------------------------------

For NVIDIA Users:
```bash
# Install CUDA drivers and toolkit first
pip install cupy-cuda12x  # or appropriate version
pip install spacial_boxcounting[gpu]
```

For AMD Users:
```bash
# Install ROCm first (system dependent)
pip install cupy-rocm-5.0  # or appropriate version
pip install spacial_boxcounting[gpu]
```

For CPU-Only Users:
```bash
# Works on any system
pip install spacial_boxcounting
```

Performance Expectations:
------------------------

Hardware       | Image Size  | Processing Time | Recommended For
---------------|-------------|-----------------|-----------------
CPU (Modern)   | Small- Medium| Fast           | All users, small datasets
CPU (High-end) | Large        | Moderate       | Users without GPU
NVIDIA GPU     | Any          | Fastest        | Performance-critical tasks
AMD GPU (ROCm) | Medium-Large | Fast           | AMD users with setup
Web/Cloud      | Any          | Varies         | No local GPU, large batches

Cross-Platform Recommendations:
------------------------------

1. Provide Clear Detection:
   ```python
   # Auto-detect available processing backends
   def detect_backend():
       if cuda_available():
           return "gpu-nvidia"
       elif rocm_available():
           return "gpu-amd"
       else:
           return "cpu"
   ```

2. Graceful Fallback:
   - Always provide CPU fallback when GPU unavailable
   - Clear error messages with installation instructions
   - Performance warnings for suboptimal configurations

3. Documentation for All Platforms:
   - Platform-specific installation guides
   - Performance comparison charts
   - Troubleshooting common issues per platform

Making It Work for Everyone:
---------------------------

1. For AMD Users Without ROCm:
   - Provide CPU-optimized version with clear performance expectations
   - Document when GPU would be beneficial (large images, batch processing)
   - Recommend cloud alternatives for heavy processing needs

2. For Users Without Any GPU:
   - Emphasize CPU optimization features (Numba JIT)
   - Provide multi-core processing options
   - Recommend image size considerations for reasonable processing times

3. For Cloud/Enterprise Users:
   - Docker images with all dependencies pre-configured
   - Kubernetes deployment options for distributed processing
   - Batch job processing APIs

Sample Performance Comparison:
-----------------------------

| Image Size | CPU Time | GPU Time (NVIDIA) | GPU Time (AMD/ROCm) | Speedup (NVIDIA) | Speedup (AMD) |
|------------|----------|------------------|---------------------|------------------|---------------|
| 256x256    | 0.08s    | 0.03s            | 0.04s               | 2.7x             | 2.0x          |
| 512x512    | 0.32s    | 0.08s            | 0.10s               | 4.0x             | 3.2x          |
| 1024x1024  | 1.28s    | 0.20s            | 0.25s               | 6.4x             | 5.1x          |
| 2048x2048  | 5.12s    | 0.50s            | 0.60s               | 10.2x            | 8.5x          |

Optimization Checklist:
----------------------

□ Auto-detect available hardware backends
□ Provide clear performance expectations per platform
□ Document installation procedures for all systems
□ Implement graceful degradation when GPU not available
□ Test with various hardware configurations
□ Provide examples and benchmarks for typical use cases
□ Create platform-specific troubleshooting guides
□ Offer web-based alternatives for users without GPU
"""
