# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-01

### Added
- Initial release of spacial-boxcounting package
- Spatial boxcount algorithm implementation with CPU and GPU support
- API functions for processing files and numpy arrays
- Command-line interface for batch processing
- Fractal dimension computation capabilities
- Multi-scale analysis support
- Hilbert curve transformation option
- Comprehensive test suite with 5/7 tests passing (2 GPU tests skipped when CUDA unavailable)
- Performance benchmarking and analysis tools
- Cross-platform compatibility documentation
- Detailed installation guides for all platforms

### Changed
- Enhanced README.md with comprehensive quick start guide and feature overview
- Improved package metadata in setup.py with complete classifiers and dependencies
- Added CLI entry points for easy command-line usage
- Updated documentation structure for better user experience

### Fixed
- Resolved duplicate function definitions in API
- Fixed test import errors and failures
- Standardized parameter naming conventions across all functions
- Enhanced GPU test skipping logic to properly handle CUDA availability
- Corrected package structure with proper __init__.py files

### Security
- Verified no known security vulnerabilities in dependencies
- Implemented secure file handling and input validation
- Added comprehensive security audit with bandit
- No hardcoded credentials or sensitive information in code

### Documentation
- Added detailed README with installation and usage instructions
- Created performance analysis documentation with expected speedups
- Developed platform compatibility guide for NVIDIA, AMD, and CPU-only users
- Documentation organization plan for comprehensive API reference

## Pre-release Development

### Features Implemented
- Core spatial boxcounting algorithm with Numba JIT optimization
- GPU acceleration support with CuPy (NVIDIA CUDA)
- File I/O with support for JPEG, BMP, PNG, and other common formats
- Batch processing capabilities for directories of images
- Multi-scale fractal dimension analysis
- 2D spatial maps of box count ratios and lacunarity
- Single-value overall analysis mode
- Hilbert curve transformation for optimized processing order

### Testing
- Unit tests for API functions covering various scenarios
- GPU functionality tests with proper fallback handling
- Integration tests for complete workflow validation
- Cross-platform compatibility validation

### Packaging
- Proper Python package structure for pip installation
- Complete setup.py with metadata and dependencies
- CLI entry points for command-line usage
- Version management system
- Development and GPU extras for optional features

[Unreleased]: https://github.com/yourusername/spacial-boxcounting-cpu-gpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/spacial-boxcounting-cpu-gpu/releases/tag/v0.1.0
