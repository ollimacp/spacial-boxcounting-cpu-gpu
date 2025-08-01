import numpy as np
import pytest

try:
    from spacial_boxcounting.core import spacialBoxcount_gpu, Z_boxcount_gpu
    import cupy as cp
    # Check if CuPy can actually work (CUDA available)
    try:
        # Try to create a simple array and perform an operation that requires CUDA
        test_arr = cp.array([1, 2, 3])
        _ = test_arr * 2  # Simple operation
        # Try to access GPU functions that would fail without CUDA
        CUPY_AVAILABLE = True
    except Exception:
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not installed")
def test_Z_boxcount_gpu():
    # Create a simple 2D array
    arr = np.array([[10, 20], [30, 40]], dtype=np.float32)
    boxsize = 2
    maxvalue = 256
    counted, lacunarity = Z_boxcount_gpu(arr, boxsize, maxvalue)
    assert isinstance(counted, int)
    assert isinstance(lacunarity, float)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not installed")
def test_spacialBoxcount_gpu():
    # Create a simple 2D array of size 8x8
    arr = np.random.randint(0, 256, size=(8, 8)).astype(np.uint8)
    iteration = 0  # using the smallest boxsize=2
    maxvalue = 256
    result = spacialBoxcount_gpu(arr, iteration, maxvalue)
    # result should be a list of two arrays
    assert isinstance(result, list) and len(result) == 2
    assert result[0].ndim == 2
    assert result[1].ndim == 2
