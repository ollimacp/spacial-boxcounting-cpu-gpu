import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_curve_transform(arr):
    """Transform a 2D numpy array using a Hilbert curve mapping.

    Parameters:
        arr (np.ndarray): 2D array input
    Returns:
        np.ndarray: Transformed 2D array via Hilbert curve ordering
    """
    # Flatten the array and compute the side length for a square
    flat = arr.flatten()
    n = flat.size
    side = int(np.ceil(np.sqrt(n)))
    # Next power of 2 for Hilbert curve
    # find p s.t. 2^(p*2) >= n
    p = 1
    while (2**(2*p)) < n:
        p += 1
    hilbert = HilbertCurve(p, 2)
    total_points = 2**(2*p)
    # Get Hilbert indices and sort points accordingly
    # Build mapping from index to position
    points = hilbert.points_from_distances(range(total_points))
    # Retain only as many points as data length
    points = points[:n]

    # Create an empty array of shape (side, side)
    transformed = np.zeros((side, side), dtype=arr.dtype)
    for idx, point in enumerate(points):
        x, y = point
        # Ensure indices are within bounds of our square
        if y < side and x < side:
            transformed[y, x] = flat[idx]
    return transformed
