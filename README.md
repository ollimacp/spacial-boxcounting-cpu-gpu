# spacial-boxcounting: Spatial Boxcount Algorithm & Fractal Analysis

An implementation of a spatial boxcount algorithm for fractal analysis, with both CPU and GPU support for accelerated computation.

## Abstract
This project implements a spatial boxcount algorithm that characterizes 2D arrays by topological complexity and spatial heterogeneity. With both CPU and GPU support, it enables spatial similarity search, edge detection, and statistical analysis of image datasets.

## Key Features
- **Spatial Box Counting**: Produces 2D maps of box count ratios and lacunarity
- **Fractal Dimension Analysis**: Multi-scale fractal dimension computation
- **Multiple Processing Modes**: Spatial maps or single-value results
- **CPU & GPU Support**: Numba JIT compilation and CuPy acceleration — switch via `backend="cpu"|"gpu"`
- **Batch Processing**: Process entire directories of images
- **Multiple Input Formats**: JPEG, BMP, PNG, and binary files
- **Hilbert Curve Mapping**: Preserves data locality for binary file analysis
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Installation
Install via pip:

```bash
# Basic CPU-only installation
pip install spacial_boxcounting

# With GPU support (NVIDIA CUDA)
pip install spacial_boxcounting[gpu]

# With GPU support (AMD ROCm — experimental)
pip install spacial_boxcounting[gpu-amd]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

### Processing from a Numpy Array

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

# Spatial processing (CPU)
result = boxcount_from_array(arr, mode='spatial')
print('Spatial Result shape:', [r.shape for r in result])

# Single value processing (CPU)
result = boxcount_from_array(arr, mode='single')
print('Box Count & Lacunarity:', result)

# GPU-accelerated — just add backend='gpu'
result_gpu = boxcount_from_array(arr, mode='single', backend='gpu')
print('GPU Result:', result_gpu)

# Fractal dimension (CPU or GPU)
fd_cpu = fractal_dimension_from_array(arr)
fd_gpu = fractal_dimension_from_array(arr, backend='gpu')
print(f'Fractal Dimension: CPU={fd_cpu:.3f}, GPU={fd_gpu:.3f}')
```

### Processing a File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension

# CPU
result = boxcount_from_file('path/to/image.jpg', mode='spatial')
fd = fractal_dimension('path/to/image.jpg')

# GPU
result = boxcount_from_file('path/to/image.jpg', mode='spatial', backend='gpu')
fd = fractal_dimension('path/to/image.jpg', backend='gpu')
```

## Command-Line Interface

```bash
# CPU (default)
spacial-boxcount single --file path/to/image.jpg --mode spatial

# GPU — just add --backend gpu
spacial-boxcount single --file path/to/image.jpg --mode spatial --backend gpu

# Batch processing with GPU
spacial-boxcount batch --folder path/to/images/ --mode single --backend gpu

# With Hilbert curve mapping (for binary files)
spacial-boxcount single --file path/to/data.bin --mode spatial --hilbert
```

## GPU Acceleration

GPU acceleration is available via CuPy. Use the `backend="gpu"` parameter:

```python
from spacial_boxcounting.api import boxcount_from_array

# All API functions accept backend='cpu' (default) or backend='gpu'
result = boxcount_from_array(arr, mode='single', backend='gpu')
```

If CuPy is not installed, requesting `backend="gpu"` raises a clear `ImportError`:

```
ImportError: GPU backend requested but CuPy is not installed.
Install with: pip install spacial_boxcounting[gpu]
```

### GPU Backend Detection

```python
from spacial_boxcounting import CUPY_AVAILABLE, GPU_BACKEND

print(f'CuPy available: {CUPY_AVAILABLE}')
print(f'GPU backend: {GPU_BACKEND}')  # 'cuda', 'rocm', or None
```

### Performance

Measured on RTX 4060 Ti (batched 4D-tensor implementation):

| Image Size | Boxsize=2 | Boxsize=4 | Boxsize=16 |
|------------|-----------|-----------|------------|
| 128×128    | 9×        | 5×        | 1×         |
| 256×256    | 9×        | 18×       | 4×         |
| 512×512    | 3×        | 12×       | 16×        |
| 1024×1024  | 2.6×      | 11×       | **35×**    |

- **Small images (< 128×128)**: CPU may be faster due to GPU transfer overhead
- **Large images (> 512×512)**: GPU provides 2–35× speedup
- **AMD users**: ROCm support is experimental (`pip install spacial_boxcounting[gpu-amd]`)

## Hilbert Curve Mapping for Binary Data
For binary files, the Hilbert curve mapping preserves data locality when converting 1D data streams to 2D arrays for spatial analysis:

```python
result = boxcount_from_file('data.bin', mode='spatial', hilbert=True)
fd = fractal_dimension('data.bin', hilbert=True)
```

## Batch Processing
Process multiple images with progress tracking:

```python
from spacial_boxcounting.batch import batch_boxcount

results = batch_boxcount('path/to/images/', mode='single')
for filename, result in results.items():
    print(f'{filename}: {result}')
```

## License
See [LICENSE.txt](LICENSE.txt) for details.