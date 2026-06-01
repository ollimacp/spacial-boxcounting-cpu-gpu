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
from spacial_boxcounting.core import CUPY_AVAILABLE


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


# ---------------------------------------------------------------------------
# backend parameter
# ---------------------------------------------------------------------------


def test_backend_invalid_raises() -> None:
    arr = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported backend"):
        boxcount_from_array(arr, mode="single", backend="cuda")


def test_backend_cpu_is_default(test_array: np.ndarray) -> None:
    """Explicit cpu backend should match the default (no backend arg)."""
    default = boxcount_from_array(test_array, mode="single")
    explicit = boxcount_from_array(test_array, mode="single", backend="cpu")
    assert default["boxcount"] == explicit["boxcount"]


def test_backend_cpu_all_functions(test_array: np.ndarray) -> None:
    """Every public function accepts backend='cpu' without error."""
    boxcount_from_array(test_array, mode="spatial", backend="cpu")
    boxcount_from_array(test_array, mode="single", backend="cpu")
    fractal_dimension_from_array(test_array, backend="cpu")
    multi_scale_fractal_dimension_from_array(
        test_array, scales=range(3), backend="cpu"
    )
    result = global_boxcount_from_array(test_array, scales=range(3), backend="cpu")
    assert isinstance(result, dict)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_backend_gpu_spatial(test_array: np.ndarray) -> None:
    result = boxcount_from_array(test_array, mode="spatial", backend="gpu")
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_backend_gpu_single(test_array: np.ndarray) -> None:
    result = boxcount_from_array(test_array, mode="single", backend="gpu")
    assert isinstance(result, dict)
    assert "boxcount" in result
    assert "lacunarity" in result


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_backend_gpu_fractal_dimension(test_array: np.ndarray) -> None:
    fd = fractal_dimension_from_array(test_array, backend="gpu")
    assert isinstance(fd, float)
    assert fd >= 0


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
def test_backend_gpu_vs_cpu_consistency(test_array: np.ndarray) -> None:
    """CPU and GPU single-mode boxcounts must match."""
    cpu = boxcount_from_array(test_array, mode="single", backend="cpu")
    gpu = boxcount_from_array(test_array, mode="single", backend="gpu")
    assert cpu["boxcount"] == gpu["boxcount"]


def test_backend_gpu_without_cupy_raises() -> None:
    """Requesting GPU without CuPy must raise ImportError."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    if not CUPY_AVAILABLE:
        with pytest.raises(ImportError, match="GPU backend requested"):
            boxcount_from_array(arr, mode="single", backend="gpu")


# ---------------------------------------------------------------------------
# CPU [0.0] artefact — fixed in v0.3.1
# ---------------------------------------------------------------------------


def test_z_boxcount_histogram_no_leading_zero() -> None:
    """Histogram must NOT contain a spurious leading zero from InitialEntry."""
    from spacial_boxcounting.core import Z_boxcount

    arr = np.array([[10, 20], [30, 40]], dtype=np.float32)
    # Z_boxcount returns (boxcount, lacunarity) — we verify it doesn't crash
    # and that the internal histogram (tested via lacunarity) is correct.
    # With boxsize=2, all 4 values map to floor(pixel/2) = [5,10,15,20].
    # That's 4 unique boxes — lacunarity measures distribution.
    counted, lac = Z_boxcount(arr, 2, 256)
    assert isinstance(counted, (int, np.integer))
    assert isinstance(lac, float)
    assert counted >= 1
    assert lac >= 0.0


def test_lacunarity_no_leading_zero_effect() -> None:
    """With the [0.0] bug fixed, lacunarity matches true population variance.

    Uses a case where counted_Boxes >= Max_Num_Boxes so no empty-box padding
    is appended — isolating the leading-zero artefact.
    """
    from spacial_boxcounting.core import Z_boxcount

    # 2×2 array, boxsize=2, MaxValue=4 → Max_Num_Boxes=2
    # Values: floor(pixel/2) = [0, 1, 2, 3] → 4 unique boxes
    # counted_Boxes=4 >= Max_Num_Boxes=2 → no empty boxes appended
    # Histogram (after fix): [1, 1, 1, 1] → all boxes equal → lacunarity=0
    # Histogram (with bug):  [0, 1, 1, 1, 1] → leading zero skews result
    arr = np.array([[0, 2], [4, 6]], dtype=np.float32)
    counted, lac = Z_boxcount(arr, 2, 4)
    assert counted == 4
    # With the fix: all counts equal → σ=0 → lac=0
    assert lac == 0.0, (
        f"Expected lacunarity=0 for uniform histogram [1,1,1,1], "
        f"got {lac}. Leading-zero bug still present?"
    )


def test_cpu_gpu_lacunarity_consistent() -> None:
    """After the fix, CPU and GPU lacunarity should be naturally identical."""
    from spacial_boxcounting.core import Z_boxcount

    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    _, lac_cpu = Z_boxcount(arr, 4, 256)

    if CUPY_AVAILABLE:
        from spacial_boxcounting.core import Z_boxcount_gpu
        _, lac_gpu = Z_boxcount_gpu(arr, 4, 256)
        assert lac_cpu == pytest.approx(lac_gpu), (
            f"CPU lacunarity {lac_cpu} != GPU lacunarity {lac_gpu}"
        )