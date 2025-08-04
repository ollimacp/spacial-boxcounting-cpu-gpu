# Platform Compatibility Guide for Spatial Boxcounting

## Universal Compatibility Strategy

### 1. CPU-Only Baseline (Works everywhere):
- Primary implementation using Numba JIT compilation
- Multi-threading support for better CPU utilization
- Optimized memory usage for large arrays
- No external dependencies beyond basic scientific Python stack

### 2. NVIDIA GPU Support (CUDA):
- CuPy-based implementation for parallel processing
- Requires: NVIDIA GPU + CUDA drivers + CUDA toolkit
- Installation: `pip install cupy-cuda12x` (or appropriate version)
- Provides significant speedups for large images and batch processing

### 3. AMD GPU Support (ROCm - Experimental):
- CuPy has experimental ROCm support
- Requires: AMD GPU + ROCm stack
- Installation: `pip install cupy-rocm-5.0` (version depends on ROCm version)
- Performance varies based on ROCm version and hardware

## Installation Instructions

### Basic CPU Installation:
```bash
pip install spacial_boxcounting
```

### With GPU Support:
```bash
# For NVIDIA CUDA
pip install spacial_boxcounting[gpu]

# For AMD ROCm (experimental)
pip install spacial_boxcounting[gpu]  # Then install cupy-rocm separately
```

### Development Installation:
```bash
git clone <repository-url>
cd spacial-boxcounting-cpu-gpu
pip install -e .
```

## Platform-Specific Notes

### Linux:
- Best platform for GPU acceleration
- All features fully supported
- Easy CUDA installation through package managers

### Windows:
- Full CPU support
- CUDA support available with proper driver installation
- May require Visual Studio build tools for some dependencies

### macOS:
- Full CPU support
- No NVIDIA CUDA support (NVIDIA discontinued macOS drivers)
- AMD GPU support through ROCm (limited)
- Excellent performance for CPU-based processing

## Troubleshooting Common Issues

### CUDA Library Issues:
If you encounter "libnvrtc.so.12: cannot open shared object file":
- Ensure CUDA toolkit is properly installed
- Check that CUDA libraries are in your library path
- Verify CUDA version compatibility with installed CuPy

### Performance Issues:
- First-time runs may be slower due to JIT compilation
- Large images may be limited by available memory
- Very small images may not benefit from GPU acceleration
