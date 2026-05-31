"""Tests for the public API functions using generated test data."""

import numpy as np
import pytest
from PIL import Image

from spacial_boxcounting.api import (
    boxcount_from_array,
    boxcount_from_file,
    fractal_dimension,
    fractal_dimension_from_array,
    global_boxcount_from_array,
    multi_scale_fractal_dimension_from_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_array() -> np.ndarray:
    """A reproducible 2D test array (256×256, uint8)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)


@pytest.fixture
def test_image_path(tmp_path) -> str:
    """Save a small test image to a temporary file and return its path."""
    rng = np.random.default_rng(seed=123)
    arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img_path = tmp_path / "test_image.png"
    img.save(str(img_path))
    return str(img_path)


# ---------------------------------------------------------------------------
# boxcount_from_file
# ---------------------------------------------------------------------------

def test_boxcount_from_file_single_mode(test_image_path: str) -> None:
    result = boxcount_from_file(test_image_path, mode="single")
    assert isinstance(result, dict)
    assert "boxcount" in result
    assert "lacunarity" in result
    assert isinstance(result["boxcount"], (int, np.integer))
    assert isinstance(result["lacunarity"], float)


def test_boxcount_from_file_spatial_mode(test_image_path: str) -> None:
    result = boxcount_from_file(test_image_path, mode="spatial")
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in result)


# ---------------------------------------------------------------------------
# boxcount_from_array
# ---------------------------------------------------------------------------

def test_boxcount_from_array_single_mode(test_array: np.ndarray) -> None:
    result = boxcount_from_array(test_array, mode="single")
    assert isinstance(result, dict)
    assert "boxcount" in result
    assert "lacunarity" in result


def test_boxcount_from_array_spatial_mode(test_array: np.ndarray) -> None:
    result = boxcount_from_array(test_array, mode="spatial")
    assert isinstance(result, list)
    assert len(result) == 2


def test_boxcount_from_array_invalid_mode(test_array: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        boxcount_from_array(test_array, mode="invalid")


# ---------------------------------------------------------------------------
# fractal_dimension
# ---------------------------------------------------------------------------

def test_fractal_dimension_from_file(test_image_path: str) -> None:
    fd = fractal_dimension(test_image_path)
    assert isinstance(fd, float)
    assert fd >= 0


def test_fractal_dimension_from_array(test_array: np.ndarray) -> None:
    fd = fractal_dimension_from_array(test_array)
    assert isinstance(fd, float)
    assert fd >= 0


# ---------------------------------------------------------------------------
# multi_scale_fractal_dimension
# ---------------------------------------------------------------------------

def test_multi_scale_fractal_dimension(test_array: np.ndarray) -> None:
    fd = multi_scale_fractal_dimension_from_array(test_array, scales=range(5), maxvalue=256)
    assert isinstance(fd, float)
    assert fd >= 0


# ---------------------------------------------------------------------------
# global_boxcount
# ---------------------------------------------------------------------------

def test_global_boxcount_from_array(test_array: np.ndarray) -> None:
    result = global_boxcount_from_array(test_array, scales=range(5), maxvalue=256)
    assert isinstance(result, dict)
    expected_box_sizes = [2, 4, 8, 16, 32]
    for bs in expected_box_sizes:
        assert bs in result
        assert isinstance(result[bs], (int, np.integer))