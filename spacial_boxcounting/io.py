from __future__ import annotations

import os

import numpy as np
from PIL import Image
from hilbertcurve.hilbertcurve import HilbertCurve


def load_file_as_ndarray(
    filepath: str,
    mode: str = "auto",
    hilbert: bool = False,
) -> np.ndarray:
    """Load a file as a 2D numpy array.

    Supports images (PNG, JPG, BMP, TIFF), .npy files, and arbitrary
    binary files. Binary files can optionally be mapped to 2D via a
    Hilbert curve to preserve data locality.

    Parameters
    ----------
    filepath : str
        Path to the input file.
    mode : str
        ``"auto"`` (default): detect from file extension.
        ``"image"``: load as grayscale image.
        ``"npy"``: load as .npy array.
        ``"binary"``: load binary file, reshape to square or map via Hilbert.
    hilbert : bool
        If True and mode is ``"binary"``, use Hilbert curve mapping.
        Default is False.

    Returns
    -------
    np.ndarray
        2D array of uint8 data.
    """
    if mode == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            mode = "image"
        elif ext == ".npy":
            mode = "npy"
        else:
            mode = "binary"

    if mode == "image":
        img = Image.open(filepath).convert("L")
        arr: np.ndarray = np.array(img, dtype=np.uint8)
    elif mode == "npy":
        arr = np.load(filepath).astype(np.uint8)
    elif mode == "binary":
        with open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        if hilbert:
            arr = _map_bytes_to_hilbert(data)
        else:
            # Fallback: reshape to square, truncating excess
            side = int(np.floor(np.sqrt(data.size)))
            arr = data[: side * side].reshape(side, side)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return arr


def _map_bytes_to_hilbert(data: np.ndarray) -> np.ndarray:
    """Map a 1D byte array to a 2D array via a Hilbert curve.

    The Hilbert curve preserves spatial locality, making it suitable
    for spatial box-counting analysis of arbitrary binary data.
    The output is a square of size 2^p × 2^p.

    Parameters
    ----------
    data : np.ndarray
        1D array of bytes.

    Returns
    -------
    np.ndarray
        2D array with dimensions 2^p × 2^p.
    """
    length = data.size

    # Determine minimal p such that 2^(2p) >= length
    p = 0
    while (2 ** (2 * p)) < length:
        p += 1
    side = 2**p
    total = side * side

    # Pad to full square size if needed
    if total > length:
        data = np.pad(data, (0, total - length), mode="constant", constant_values=0)

    # Generate Hilbert-curve coordinates (n=2 dimensions, p iterations)
    hc = HilbertCurve(p, 2)
    distances = np.arange(total)
    coords = np.array(hc.points_from_distances(distances))  # shape (total, 2)

    # Map byte values into the 2D grid
    arr2d = np.zeros((side, side), dtype=data.dtype)
    arr2d[coords[:, 0], coords[:, 1]] = data
    return arr2d