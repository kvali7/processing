# Changelog

All notable changes to the psafe_custom library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024

### Added
- Initial release of psafe_custom library
- `compute_frame()` function for extracting and averaging variable-length frames from time series
- `compute_frame_with_metadata()` function that returns frames with starting indices
- Flexible interpolation support via `interpolation_kind` parameter:
  - 'linear' (default) - Linear interpolation
  - 'cubic' - Cubic spline interpolation
  - 'quadratic' - Quadratic spline interpolation
  - 'nearest' - Nearest neighbor interpolation
  - 'nearest-up' - Nearest neighbor with upper preference
  - 'zero' - Zero-order spline (piecewise constant)
  - 'slinear' - First-order spline
  - 'previous' - Use previous value
  - 'next' - Use next value
- Comprehensive input validation with clear error messages
- Full type hints for all functions
- Extensive test suite (30+ test cases)
- Complete documentation:
  - README.md for quick start
  - docs.md for comprehensive API reference
  - INSTALL.md for installation instructions
  - SUMMARY.md for project overview
  - example_usage.py with 7 runnable examples
- Python 3.8+ support
- NumPy and SciPy integration

### Features
- Variable-length frame extraction from time series
- Automatic interpolation to uniform frame lengths
- Frame averaging for noise reduction
- Metadata tracking for frame positions
- Support for overlapping and non-overlapping windows
- Memory-efficient implementation
- Handles edge cases gracefully

### Testing
- 30+ comprehensive test cases
- Input validation tests
- Edge case coverage
- Different interpolation method tests
- Integration tests

### Documentation
- Complete API reference designed for both humans and agentic AI
- 7 detailed usage examples
- Type signature documentation
- Performance considerations
- Best practices guide

## [Unreleased]

### Planned
- Additional interpolation methods exploration
- Parallel processing support for large datasets
- GPU acceleration options
- Windowing functions (Hamming, Hanning, Blackman)
- Real-time streaming mode
- Multi-dimensional time series support
- Performance benchmarking suite
- Additional output format options

---

## Version History

### Version 0.1.0 (Current)
**Status:** Production Ready  
**Release Date:** 2024  
**Python Support:** 3.8, 3.9, 3.10, 3.11, 3.12

**Key Highlights:**
- First stable release
- Complete core functionality
- Flexible interpolation methods
- Comprehensive test coverage
- Full documentation
- Ready for production use

---

## How to Upgrade

Since the package is installed in editable mode during development, changes to source code are immediately available. To upgrade dependencies:

```bash
pip install -e ".[dev]" --upgrade
```

## Breaking Changes

None (initial release)

## Deprecations

None (initial release)

## Security Updates

None (initial release)

---

**Maintained by:** psafe_custom development team  
**License:** MIT  
**Repository:** Local development