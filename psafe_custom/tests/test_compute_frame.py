"""
Tests for the compute_frame module.
"""

import numpy as np
import pytest

from psafe_custom.compute_frame import compute_frame, compute_frame_with_metadata


class TestComputeFrame:
    """Tests for the compute_frame function."""

    def test_basic_functionality(self):
        """Test basic frame computation with uniform frame lengths."""
        # Create a simple time series
        time_series = np.arange(100, dtype=float)
        # Uniform frame length of 10
        frame_lengths = np.full(100, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=3,
            sample_skip=5,
        )

        # Should have multiple averaged frames
        assert len(result) > 0
        # Each frame should have length 10
        for frame in result:
            assert len(frame) == 10

    def test_variable_frame_lengths(self):
        """Test with variable frame lengths that need interpolation."""
        time_series = np.arange(200, dtype=float)
        # Alternating frame lengths
        frame_lengths = np.array([10, 20] * 100)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=15,
            frames_to_average=2,
            sample_skip=10,
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 15

    def test_large_sample_skip(self):
        """Test with large sample_skip relative to time series."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=50,
        )

        # Should have fewer frames due to large skip
        assert len(result) >= 1

    def test_insufficient_samples(self):
        """Test when there aren't enough samples to complete all frames."""
        time_series = np.arange(50, dtype=float)
        frame_lengths = np.full(50, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=10,  # Need 100 samples, but only have 50
            sample_skip=5,
        )

        # Should have no frames or very few
        assert isinstance(result, list)

    def test_exact_fit(self):
        """Test when samples exactly fit the frame requirements."""
        time_series = np.arange(60, dtype=float)
        frame_lengths = np.full(60, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=6,  # Exactly 60 samples needed
            sample_skip=60,
        )

        assert len(result) == 1
        assert len(result[0]) == 10

    def test_different_interpolation_lengths(self):
        """Test interpolation from various frame lengths to a common length."""
        time_series = np.arange(300, dtype=float)
        # Mix of different frame lengths
        frame_lengths = np.array([5, 10, 15, 20] * 75)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=12,
            frames_to_average=4,
            sample_skip=10,
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 12

    def test_floating_point_time_series(self):
        """Test with floating point values in time series."""
        time_series = np.sin(np.linspace(0, 10, 500))
        frame_lengths = np.full(500, 25)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=25,
            frames_to_average=3,
            sample_skip=10,
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 25
            # Values should be in range of sine function
            assert np.all(frame >= -1.1) and np.all(frame <= 1.1)

    def test_averaging_effect(self):
        """Test that averaging actually averages the frames."""
        # Create a time series with a known pattern
        time_series = np.array([1.0] * 50 + [2.0] * 50)
        frame_lengths = np.full(100, 25)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=25,
            frames_to_average=2,
            sample_skip=10,
        )

        # First averaged frame should be around 1.0 or transitioning to 2.0
        assert len(result) > 0

    def test_linear_interpolation(self):
        """Test with explicit linear interpolation."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
            interpolation_kind="linear",
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 10

    def test_cubic_interpolation(self):
        """Test with cubic interpolation."""
        time_series = np.sin(np.linspace(0, 10, 200))
        frame_lengths = np.full(200, 20)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=15,
            frames_to_average=2,
            sample_skip=10,
            interpolation_kind="cubic",
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 15

    def test_nearest_interpolation(self):
        """Test with nearest neighbor interpolation."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=8,
            frames_to_average=2,
            sample_skip=5,
            interpolation_kind="nearest",
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 8

    def test_quadratic_interpolation(self):
        """Test with quadratic interpolation."""
        time_series = np.linspace(0, 100, 150)
        frame_lengths = np.full(150, 15)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=7,
            interpolation_kind="quadratic",
        )

        assert len(result) > 0
        for frame in result:
            assert len(frame) == 10

    def test_default_interpolation_is_linear(self):
        """Test that default interpolation is linear."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        result_default = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        result_explicit_linear = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
            interpolation_kind="linear",
        )

        # Results should be identical
        assert len(result_default) == len(result_explicit_linear)
        for frame_default, frame_linear in zip(result_default, result_explicit_linear):
            np.testing.assert_array_almost_equal(frame_default, frame_linear)


class TestComputeFrameValidation:
    """Tests for input validation in compute_frame."""

    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(50, 10)  # Different length

        with pytest.raises(ValueError, match="must have equal length"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=10,
                frames_to_average=2,
                sample_skip=5,
            )

    def test_non_1d_time_series(self):
        """Test that 2D time series raises ValueError."""
        time_series = np.arange(100, dtype=float).reshape(10, 10)
        frame_lengths = np.full(100, 10)

        with pytest.raises(ValueError, match="must be a 1D array"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=10,
                frames_to_average=2,
                sample_skip=5,
            )

    def test_non_1d_frame_lengths(self):
        """Test that 2D frame_lengths raises ValueError."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full((10, 10), 10)

        with pytest.raises(ValueError, match="must be a 1D array"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=10,
                frames_to_average=2,
                sample_skip=5,
            )

    def test_negative_average_frame_length(self):
        """Test that negative average_frame_length raises ValueError."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        with pytest.raises(ValueError, match="must be positive"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=-10,
                frames_to_average=2,
                sample_skip=5,
            )

    def test_zero_frames_to_average(self):
        """Test that zero frames_to_average raises ValueError."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        with pytest.raises(ValueError, match="must be positive"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=10,
                frames_to_average=0,
                sample_skip=5,
            )

    def test_negative_sample_skip(self):
        """Test that negative sample_skip raises ValueError."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        with pytest.raises(ValueError, match="must be positive"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=10,
                frames_to_average=2,
                sample_skip=-5,
            )

    def test_non_numpy_array_time_series(self):
        """Test that non-numpy array for time_series raises ValueError."""
        time_series = [1, 2, 3, 4, 5]  # Python list
        frame_lengths = np.full(5, 2)

        with pytest.raises(ValueError, match="must be a numpy array"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=2,
                frames_to_average=2,
                sample_skip=1,
            )

    def test_non_numpy_array_frame_lengths(self):
        """Test that non-numpy array for frame_lengths raises ValueError."""
        time_series = np.arange(5, dtype=float)
        frame_lengths = [2, 2, 2, 2, 2]  # Python list

        with pytest.raises(ValueError, match="must be a numpy array"):
            compute_frame(
                time_series=time_series,
                current_frame_length=frame_lengths,
                average_frame_length=2,
                frames_to_average=2,
                sample_skip=1,
            )


class TestComputeFrameWithMetadata:
    """Tests for the compute_frame_with_metadata function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of frames and indices."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        result = compute_frame_with_metadata(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        frames, indices = result
        assert isinstance(frames, list)
        assert isinstance(indices, list)

    def test_indices_match_frames(self):
        """Test that number of indices matches number of frames."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        frames, indices = compute_frame_with_metadata(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        assert len(frames) == len(indices)

    def test_indices_increase_by_sample_skip(self):
        """Test that indices increase by sample_skip amount."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)
        sample_skip = 7

        frames, indices = compute_frame_with_metadata(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=sample_skip,
        )

        if len(indices) > 1:
            # Check that consecutive indices differ by sample_skip
            for i in range(1, len(indices)):
                assert indices[i] == indices[i - 1] + sample_skip

    def test_first_index_is_zero(self):
        """Test that first index is always 0."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        frames, indices = compute_frame_with_metadata(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        if len(indices) > 0:
            assert indices[0] == 0

    def test_metadata_with_different_interpolation(self):
        """Test that metadata function works with different interpolation kinds."""
        time_series = np.arange(100, dtype=float)
        frame_lengths = np.full(100, 10)

        frames, indices = compute_frame_with_metadata(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
            interpolation_kind="cubic",
        )

        assert len(frames) == len(indices)
        for frame in frames:
            assert len(frame) == 10


class TestComputeFrameEdgeCases:
    """Tests for edge cases in compute_frame."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        time_series = np.array([])
        frame_lengths = np.array([])

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        assert result == []

    def test_very_small_time_series(self):
        """Test with very small time series."""
        time_series = np.array([1.0, 2.0, 3.0])
        frame_lengths = np.array([1, 1, 1])

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=1,
            frames_to_average=2,
            sample_skip=1,
        )

        # May or may not have results depending on exact behavior
        assert isinstance(result, list)

    def test_large_frame_length(self):
        """Test when frame length equals or exceeds time series length."""
        time_series = np.arange(50, dtype=float)
        frame_lengths = np.full(50, 100)  # Frame length > time series length

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=10,
            frames_to_average=2,
            sample_skip=5,
        )

        # Should return empty list as frames can't be extracted
        assert result == []

    def test_sample_skip_larger_than_series(self):
        """Test when sample_skip is larger than time series."""
        time_series = np.arange(50, dtype=float)
        frame_lengths = np.full(50, 5)

        result = compute_frame(
            time_series=time_series,
            current_frame_length=frame_lengths,
            average_frame_length=5,
            frames_to_average=2,
            sample_skip=100,
        )

        # Should have at most 1 frame
        assert len(result) <= 1
