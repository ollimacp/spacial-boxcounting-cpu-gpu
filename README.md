# spacial-boxcounting: Spatial Boxcount Algorithm & Fractal Analysis

An implementation of a spatial boxcount algorithm for fractal analysis, with both CPU and GPU support for accelerated computation.

## Abstract
This project implements a spatial boxcount algorithm that characterizes 2D arrays by topological complexity and spatial heterogeneity. With both CPU and GPU support, it enables spatial similarity search, edge detection, and statistical analysis of image datasets.

## Key Features
- **Spatial Box Counting**: Produces 2D maps of box count ratios and lacunarity
- **Fractal Dimension Analysis**: Multi-scale fractal dimension computation
- **Multiple Processing Modes**: Spatial maps or single-value results
- **CPU & GPU Support**: Numba JIT compilation and optional CuPy acceleration
- **Batch Processing**: Process entire directories of images
- **Multiple Input Formats**: JPEG, BMP, PNG, and binary files
- **Hilbert Curve Mapping**: Preserves data locality for binary file analysis
- **Cross-Platform**: Works on Windows, Linux, and macOS (NVIDIA GPU support)

## Installation
Install via pip:

```bash
# Basic CPU-only installation
pip install spacial_boxcounting

# With GPU support (NVIDIA CUDA)
pip install spacial_boxcounting[gpu]

# Development installation
pip install -e .
```

Ensure dependencies are installed: numpy, numba, Pillow, matplotlib, hilbertcurve, pandas, and optionally cupy for GPU acceleration.

## Quick Start
### Processing a Single File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension_from_file

# Get spatial box count map (2D maps of box count ratios and lacunarity)
result_spatial = boxcount_from_file('path/to/your/image.jpg', mode='spatial')
print('Spatial Box Count Map shape:', [r.shape for r in result_spatial])

# Get overall box count & lacunarity
result_single = boxcount_from_file('path/to/your/image.jpg', mode='single')
print('Box Count & Lacunarity:', result_single)

# Compute fractal dimension
fd = fractal_dimension_from_file('path/to/your/image.jpg')
print('Fractal Dimension:', fd)
```

### Processing from a Numpy Array

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

# Spatial processing
result_spatial = boxcount_from_array(arr, mode='spatial')
print('Spatial Result shape:', [r.shape for r in result_spatial])

# Single value processing
result_single = boxcount_from_array(arr, mode='single')
print('Single Result:', result_single)

# Fractal dimension
fd = fractal_dimension_from_array(arr)
print('Fractal Dimension:', fd)
```

## Command-Line Interface
Process images directly from the command line:

```bash
# Process a single file
spacial-boxcount single --file path/to/image.jpg --mode spatial

# Process all images in a directory
spacial-boxcount batch --folder path/to/images/ --mode single

# Process with Hilbert curve mapping (for binary files)
spacial-boxcount single --file path/to/data.bin --mode spatial --hilbert
```

## Hilbert Curve Mapping for Binary Data
For binary files, the Hilbert curve mapping preserves data locality when converting 1D data streams to 2D arrays for spatial analysis:

```python
# Process binary file with Hilbert curve mapping
result = boxcount_from_file('data.bin', mode='spatial', hilbert=True)
fd = fractal_dimension_from_file('data.bin', hilbert=True)
```

## GPU Acceleration
If Cupy is installed with CUDA support, GPU accelerated functions will automatically be used:

```python
import numpy as np
from spacial_boxcounting.core import spacialBoxcount_gpu

arr = np.random.randint(0, 256, size=(512, 512)).astype(np.uint8)
# GPU processing for large images (significant speedup)
result_gpu = spacialBoxcount_gpu(arr, iteration=2, MaxValue=256)  # box size 8
print('GPU spatial result shape:', [r.shape for r in result_gpu])
```

## Batch Processing
Process multiple images with progress tracking:

```python
from spacial_boxcounting.batch import batch_boxcount

# Process all images in a directory
results = batch_boxcount('path/to/images/', mode='single')
for filename, result in results.items():
    print(f'{filename}: {result}')
```

## Performance
Performance varies by hardware and image size:
- **Small images (< 256x256)**: CPU often faster due to GPU overhead
- **Large images (> 512x512)**: GPU provides 2-10x speedup
- **Batch processing**: GPU provides 5-50x speedup for large batches
- **AMD users**: CPU optimization available (ROCm support experimental)

## License
See [LICENSE.txt](LICENSE.txt) for details.
