import os
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension

def test_boxcount_from_file_single_mode():
    filepath = os.path.join('0Data', 'Images', 'test_image.jpg')
    result = boxcount_from_file(filepath, mode='single')
    assert isinstance(result, dict)
    assert 'boxcount' in result


def test_boxcount_from_file_spatial_mode():
    filepath = os.path.join('0Data', 'Images', 'test_image.jpg')
    result = boxcount_from_file(filepath, mode='spatial')
    # result should be a list of two arrays
    assert isinstance(result, list) and len(result) == 2


def test_fractal_dimension():
    filepath = os.path.join('0Data', 'Images', 'test_image.jpg')
    fd = fractal_dimension(filepath)
    assert isinstance(fd, float)
