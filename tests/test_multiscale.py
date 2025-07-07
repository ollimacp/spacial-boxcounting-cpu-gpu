import numpy as np
import pytest

from spacial_boxcounting.api import global_boxcount_from_array, multi_scale_fractal_dimension_from_array


def test_global_boxcount_from_array():
    # Create a random image array
    arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)
    result = global_boxcount_from_array(arr, scales=range(5), maxvalue=256)
    assert isinstance(result, dict)
    # Check that we have box counts for specified scales
    expected_box_sizes = [2, 4, 8, 16, 32]
    for bs in expected_box_sizes:
        assert bs in result
        assert isinstance(result[bs], int)


def test_multi_scale_fractal_dimension_from_array():
    # Create a random image array
    arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)
    fd = multi_scale_fractal_dimension_from_array(arr, scales=range(5), maxvalue=256)
    assert isinstance(fd, float)
    # Fractal dimension should be a non-negative value
    assert fd >= 0

if __name__ == '__main__':
    pytest.main([__file__])
