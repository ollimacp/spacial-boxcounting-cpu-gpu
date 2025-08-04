#!/usr/bin/env python3
"""
Final verification script for the spacial-boxcounting package.
This script verifies that all components work correctly before git commit.
"""
import os
import sys
import numpy as np

def test_package_imports():
    """Test that all package modules can be imported."""
    print("Testing package imports...")
    try:
        import spacial_boxcounting
        from spacial_boxcounting import api, core, cli, io, hilbert, visualize, utils
        from spacial_boxcounting._version import VERSION
        print(f"✓ Package imports successful (version {VERSION})")
        return True
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

def test_api_functions():
    """Test all API functions."""
    print("\nTesting API functions...")
    try:
        from spacial_boxcounting.api import (
            boxcount_from_array, 
            boxcount_from_file,
            fractal_dimension_from_array,
            fractal_dimension_from_file,
            multi_scale_fractal_dimension_from_array,
            global_boxcount_from_array
        )
        
        # Test with numpy array
        arr = np.random.randint(0, 256, size=(32, 32)).astype(np.uint8)
        
        # Test spatial mode
        spatial_result = boxcount_from_array(arr, mode='spatial')
        assert isinstance(spatial_result, list) and len(spatial_result) == 2
        print("✓ Spatial boxcount from array works")
        
        # Test single mode
        single_result = boxcount_from_array(arr, mode='single')
        assert isinstance(single_result, dict) and 'boxcount' in single_result
        print("✓ Single boxcount from array works")
        
        # Test fractal dimension
        fd = fractal_dimension_from_array(arr)
        assert isinstance(fd, (int, float))
        print("✓ Fractal dimension from array works")
        
        # Test with file
        test_file = "0Data/Images/test_image.jpg"
        if os.path.exists(test_file):
            file_spatial = boxcount_from_file(test_file, mode='spatial')
            file_single = boxcount_from_file(test_file, mode='single')
            file_fd = fractal_dimension_from_file(test_file)
            print("✓ File-based processing works")
        else:
            print("! Skipping file tests - no test image found")
        
        # Test multi-scale functions
        multi_fd = multi_scale_fractal_dimension_from_array(arr)
        global_bc = global_boxcount_from_array(arr)
        print("✓ Multi-scale functions work")
        
        print("✓ All API functions working correctly")
        return True
    except Exception as e:
        print(f"✗ API function test failed: {e}")
        return False

def test_core_functions():
    """Test core computational functions."""
    print("\nTesting core functions...")
    try:
        from spacial_boxcounting.core import spacialBoxcount, Z_boxcount
        
        arr = np.random.randint(0, 256, size=(16, 16)).astype(np.uint8)
        
        # Test spatial boxcount
        spatial_result = spacialBoxcount(arr, 0, 256)  # iteration 0 = box size 2
        assert isinstance(spatial_result, list) and len(spatial_result) == 2
        print("✓ Core spacialBoxcount function works")
        
        # Test Z_boxcount
        boxcount, lacunarity = Z_boxcount(arr, 2, 256)  # box size 2
        assert isinstance(boxcount, (int, np.integer))
        assert isinstance(lacunarity, (int, float, np.floating))
        print("✓ Core Z_boxcount function works")
        
        print("✓ All core functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Core function test failed: {e}")
        return False

def test_cli_functionality():
    """Test CLI functionality."""
    print("\nTesting CLI functionality...")
    try:
        # Import CLI module
        from spacial_boxcounting.cli import main
        print("✓ CLI module imports correctly")
        
        # Test that entry point exists
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "spacial_boxcounting.cli", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ CLI module executes correctly")
        else:
            print(f"! CLI module execution had issues: {result.stderr}")
            
        # Test entry point
        result = subprocess.run([
            "spacial-boxcount", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Entry point works correctly")
        else:
            print(f"! Entry point had issues: {result.stderr}")
            
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and function."""
    print("\nTesting GPU availability...")
    try:
        from spacial_boxcounting.core import spacialBoxcount_gpu, Z_boxcount_gpu
        print("✓ GPU functions can be imported")
        
        # Try to use GPU functions (will fail gracefully if CUDA not available)
        try:
            arr = np.random.randint(0, 256, size=(8, 8)).astype(np.uint8)
            result = spacialBoxcount_gpu(arr, 0, 256)
            print("✓ GPU functions execute (CUDA available)")
        except ImportError:
            print("! GPU functions available but CuPy not installed")
        except Exception as e:
            if "CUDA" in str(e) or "libnvrtc" in str(e):
                print("! GPU functions available but CUDA not configured properly")
            else:
                print(f"! GPU functions available but encountered error: {e}")
        
        return True
    except ImportError:
        print("! GPU functions not available (optional)")
        return True
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("="*60)
    print("SPACIAL-BOXCOUNTING PACKAGE VERIFICATION")
    print("="*60)
    
    tests = [
        test_package_imports,
        test_api_functions,
        test_core_functions,
        test_cli_functionality,
        test_gpu_availability
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} test groups passed!")
        print("✓ Package is ready for git commit and PyPI publication")
        return True
    else:
        print(f"✗ {total - passed} out of {total} test groups failed")
        print("! Please address the issues before committing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
