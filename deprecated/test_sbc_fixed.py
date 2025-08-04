import os
import numpy as np
import pytest
import traceback

def debug_print(header, content):
    print(f"\n=== {header} ===")
    print(content)

def test_from_file():
    """Test processing from file - uses an actual test image"""
    try:
        from spacial_boxcounting.api import boxcount_from_file, fractal_dimension_from_file

        # Use a test image that should exist
        image_path = "0Data/Images/test_image.jpg"
        if not os.path.exists(image_path):
            # Fallback to another image
            image_path = "0Data/Images/test.bmp"
            
        if not os.path.exists(image_path):
            # If no test images found, skip the test
            pytest.skip("No test images found")

        debug_print("Testing Spatial Mode from File", image_path)
        result_spatial = boxcount_from_file(image_path, mode='spatial')
        debug_print("Spatial Box Count Map", result_spatial)

        debug_print("Testing Single Mode from File", image_path)
        result_single = boxcount_from_file(image_path, mode='single')
        debug_print("Single Box Count & Lacunarity", result_single)

        debug_print("Testing Fractal Dimension from File", image_path)
        fd = fractal_dimension_from_file(image_path)
        debug_print("Fractal Dimension", fd)

    except Exception as e:
        print("\n[ERROR] Failed processing image file.")
        traceback.print_exc()
        pytest.fail(f"Failed processing image file: {e}")

def test_from_array():
    """Test processing from numpy array"""
    try:
        from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

        debug_print("Creating Random Test Array", "Shape: (256, 256)")
        arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

        debug_print("Testing Spatial Mode from Array", "")
        result = boxcount_from_array(arr, mode='spatial')
        debug_print("Spatial Result from Array", result)

        debug_print("Testing Fractal Dimension from Array", "")
        fd = fractal_dimension_from_array(arr)
        debug_print("Fractal Dimension from Array", fd)

    except Exception as e:
        print("\n[ERROR] Failed processing numpy array.")
        traceback.print_exc()
        pytest.fail(f"Failed processing numpy array: {e}")

def test_gpu():
    """Test GPU functionality if available"""
    try:
        import cupy
        from spacial_boxcounting.core import spacialBoxcount_gpu

        debug_print("Creating Random Test Array for GPU", "Shape: (64, 64)")
        arr = np.random.randint(0, 256, size=(64, 64)).astype(np.uint8)

        debug_print("Testing GPU Boxcount", "")
        result_gpu = spacialBoxcount_gpu(arr, iteration=0, MaxValue=256)
        debug_print("GPU Spatial Result", result_gpu)

    except ImportError:
        print("[INFO] cupy not installed, skipping GPU test.")
        pytest.skip("cupy not installed")
    except Exception as e:
        print("\n[ERROR] Failed during GPU boxcount.")
        traceback.print_exc()
        pytest.fail(f"Failed during GPU boxcount: {e}")

if __name__ == '__main__':
    print("=== spacial-boxcounting Test Runner ===")

    # Optional: change this path to an existing image to test from file
    test_image_path = "/home/raghat/projects/spacial-boxcounting-cpu-gpu/0Data/Images/12_3_3700x.bmp"  # <-- Replace with actual image path
    # /home/raghat/projects/spacial-boxcounting-cpu-gpu/0Data/Images/
    test_from_array()
    if os.path.exists(test_image_path):
        test_from_file()
    #test_gpu()
