# spacial-boxcounting Tutorial

This tutorial demonstrates how to install and use the package for spatial box counting and fractal analysis.

## Installation

Clone the repository and install dependencies:

```bash
pip install .
```

Ensure that all dependencies (numpy, numba, Pillow, matplotlib, hilbertcurve, cupy (optional for GPU), pytest) are installed.

## Basic Usage

### Processing a Single File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension_from_file

# For obtaining spatial box count maps:
result_spatial = boxcount_from_file('path/to/your/image.jpg', mode='spatial')
print('Spatial Box Count Map:', result_spatial)

# For a single overall box count and lacunarity:
result_single = boxcount_from_file('path/to/your/image.jpg', mode='single')
print('Box Count & Lacunarity:', result_single)

# Estimating fractal dimension:
fd = fractal_dimension_from_file('path/to/your/image.jpg')
print('Fractal Dimension:', fd)
```

### Processing from Numpy Arrays

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

# Create sample data
arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

# Get spatial results
spatial_result = boxcount_from_array(arr, mode='spatial')
print('Spatial result shape:', [r.shape for r in spatial_result])

# Get single value results  
single_result = boxcount_from_array(arr, mode='single')
print('Single result:', single_result)

# Compute fractal dimension
fd = fractal_dimension_from_array(arr)
print('Fractal dimension:', fd)
```

## Command Line Usage

The package includes a command-line interface for processing files directly:

```bash
# Process single file
spacial-boxcount single --file input.jpg --mode spatial

# Process directory of files
spacial-boxcount batch --folder images/ --mode single

# Process with Hilbert curve mapping (for binary files)
spacial-boxcount single --file binary_data.bin --mode spatial --hilbert
```

## Hilbert Curve Mapping for Binary Data

For binary files where there's no natural 2D structure, the Hilbert curve preserves sequential locality:

```python
# Process binary file with Hilbert curve mapping
result = boxcount_from_file('data.bin', mode='spatial', hilbert=True)
fd = fractal_dimension_from_file('data.bin', hilbert=True)
```

## GPU Acceleration

If CuPy is installed with CUDA support, GPU acceleration is available:

```python
from spacial_boxcounting.core import spacialBoxcount_gpu
import numpy as np

arr = np.random.randint(0, 256, size=(512, 512)).astype(np.uint8)
result = spacialBoxcount_gpu(arr, iteration=2, MaxValue=256)  # box size 8
```

## Batch Processing

Process multiple files with progress tracking:

```python
from spacial_boxcounting.batch import batch_boxcount

# Process all images in directory
results = batch_boxcount('path/to/images/', mode='single')
for filename, result in results.items():
    print(f'{filename}: {result}')
```
