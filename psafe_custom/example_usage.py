"""
Example usage script for psafe_custom library.

This script demonstrates the complete pipeline for loading sheep PPG data
and computing frames using all utility functions in apply_pasfe.py.
"""

import numpy as np

from psafe_custom import (
    compute_frames_from_df,
    get_sheep_dataframe,
    get_sorted_ac_columns,
)


def complete_pipeline_example():
    """
    Complete Pipeline: Loading Sheep Data and Computing Frames

    This example demonstrates the full workflow for processing sheep PPG data:
    1. Select dataset index
    2. Get sorted AC column names
    3. Select channels to process
    4. Load sheep dataframe (includes PPG, fSaO2, and FHR)
    5. Compute frames using heart rate data
    6. Analyze results
    """
    print("=" * 70)
    print(" " * 15 + "COMPLETE PIPELINE EXAMPLE")
    print("=" * 70)
    print()

    # Step 1: Select a dataset index (0-12 corresponding to 13 sheep experiments)
    idx = 7
    print(f"Step 1: Selected dataset index: {idx}")
    print()

    # Step 2: Get the sorted AC column names
    # Returns a dictionary with 'wv1' and 'wv2' keys containing channel names
    ac_columns = get_sorted_ac_columns()

    print("Step 2: Available AC columns:")
    print(f"  Wavelength 1 channels: {ac_columns['wv1']}")
    print(f"  Wavelength 2 channels: {ac_columns['wv2']}")
    print()

    # Step 3: Select specific channels to process
    # Here we select the first 2 channels from wavelength 1
    selected_channels = ac_columns["wv1"][:2]

    print(f"Step 3: Selected channels for processing: {selected_channels}")
    print()

    # Step 4: Load the sheep dataframe
    # This returns a DataFrame with PPG channels, fSaO2, and FHR (in Hz)
    print("Step 4: Loading sheep dataframe...")
    data = get_sheep_dataframe(idx)

    print(f"  DataFrame shape: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Data length: {len(data)} samples")
    print()

    # Step 5: Process the selected channels using compute_frames_from_df
    print("Step 5: Computing frames...")
    print("  Parameters:")
    print(f"    - Time series columns: {selected_channels}")
    print("    - Heart rate column: 'FHR' (in Hz)")
    print("    - Sampling rate: 80.0 Hz")
    print("    - Average frame length: 100 samples")
    print("    - Frames to average: 120")
    print("    - Sample skip: 80 samples")
    print("    - Interpolation: linear")
    print()

    frames_dict = compute_frames_from_df(
        dataframe=data,
        time_series_columns=selected_channels,
        heartrate_column="FHR",
        sampling_rate=80.0,
        average_frame_length=100,
        frames_to_average=120,
        sample_skip=80,
        interpolation_kind="linear",
    )

    # Step 6: Analyze the results
    print("Step 6: Results summary:")
    print("-" * 70)

    for channel_name, frames_list in frames_dict.items():
        print(f"\nChannel: {channel_name}")
        print(f"  Number of averaged frames: {len(frames_list)}")
        print(f"  Each frame shape: {frames_list[0].shape}")
        print("  First frame statistics:")
        print(f"    - Mean: {np.mean(frames_list[0]):.6f}")
        print(f"    - Std: {np.std(frames_list[0]):.6f}")
        print(f"    - Min: {np.min(frames_list[0]):.6f}")
        print(f"    - Max: {np.max(frames_list[0]):.6f}")

    print()
    print("=" * 70)
    print()

    return frames_dict


def process_all_datasets_example():
    """
    Example: Process all 13 sheep datasets

    Demonstrates how to iterate through all available datasets
    and extract basic information.
    """
    print("=" * 70)
    print(" " * 15 + "PROCESS ALL DATASETS EXAMPLE")
    print("=" * 70)
    print()

    ac_columns = get_sorted_ac_columns()
    selected_channels = ac_columns["wv1"][:1]  # Just use first channel for speed

    print(f"Processing channel: {selected_channels[0]}")
    print()
    print("Dataset | Samples | Frames | FHR Range (Hz)")
    print("-" * 70)

    for idx in range(13):
        try:
            # Load data
            data = get_sheep_dataframe(idx)

            # Get FHR statistics
            fhr_min = data["FHR"].min()
            fhr_max = data["FHR"].max()

            # Compute frames
            frames_dict = compute_frames_from_df(
                dataframe=data,
                time_series_columns=selected_channels,
                heartrate_column="FHR",
                sampling_rate=80.0,
                average_frame_length=100,
                frames_to_average=120,
                sample_skip=80,
                interpolation_kind="linear",
            )

            num_frames = len(frames_dict[selected_channels[0]])

            print(
                f"   {idx:2d}   | {len(data):7d} | {num_frames:6d} | {fhr_min:.3f} - {fhr_max:.3f}"
            )

        except Exception as e:
            print(f"   {idx:2d}   | ERROR: {str(e)}")

    print()
    print("=" * 70)
    print()


def compare_wavelengths_example():
    """
    Example: Compare processing both wavelengths

    Demonstrates how to process channels from both wavelengths
    and compare their characteristics.
    """
    print("=" * 70)
    print(" " * 15 + "COMPARE WAVELENGTHS EXAMPLE")
    print("=" * 70)
    print()

    idx = 7
    ac_columns = get_sorted_ac_columns()

    # Select first channel from each wavelength
    wv1_channel = [ac_columns["wv1"][0]]
    wv2_channel = [ac_columns["wv2"][0]]

    print(f"Dataset index: {idx}")
    print(f"Wavelength 1 channel: {wv1_channel[0]}")
    print(f"Wavelength 2 channel: {wv2_channel[0]}")
    print()

    # Load data
    data = get_sheep_dataframe(idx)

    # Process both wavelengths
    print("Processing wavelength 1...")
    frames_wv1 = compute_frames_from_df(
        dataframe=data,
        time_series_columns=wv1_channel,
        heartrate_column="FHR",
        sampling_rate=80.0,
        average_frame_length=100,
        frames_to_average=120,
        sample_skip=80,
        interpolation_kind="linear",
    )

    print("Processing wavelength 2...")
    frames_wv2 = compute_frames_from_df(
        dataframe=data,
        time_series_columns=wv2_channel,
        heartrate_column="FHR",
        sampling_rate=80.0,
        average_frame_length=100,
        frames_to_average=120,
        sample_skip=80,
        interpolation_kind="linear",
    )

    # Compare results
    print()
    print("Comparison Results:")
    print("-" * 70)

    wv1_frames = frames_wv1[wv1_channel[0]]
    wv2_frames = frames_wv2[wv2_channel[0]]

    print(f"\nWavelength 1 ({wv1_channel[0]}):")
    print(f"  Number of frames: {len(wv1_frames)}")
    print(f"  Average frame mean: {np.mean([np.mean(f) for f in wv1_frames]):.6f}")
    print(f"  Average frame std: {np.mean([np.std(f) for f in wv1_frames]):.6f}")

    print(f"\nWavelength 2 ({wv2_channel[0]}):")
    print(f"  Number of frames: {len(wv2_frames)}")
    print(f"  Average frame mean: {np.mean([np.mean(f) for f in wv2_frames]):.6f}")
    print(f"  Average frame std: {np.mean([np.std(f) for f in wv2_frames]):.6f}")

    print()
    print("=" * 70)
    print()


def custom_parameters_example():
    """
    Example: Using custom processing parameters

    Demonstrates how different parameter choices affect the output.
    """
    print("=" * 70)
    print(" " * 15 + "CUSTOM PARAMETERS EXAMPLE")
    print("=" * 70)
    print()

    idx = 7
    ac_columns = get_sorted_ac_columns()
    selected_channel = [ac_columns["wv1"][0]]

    data = get_sheep_dataframe(idx)

    print(f"Dataset: {idx}")
    print(f"Channel: {selected_channel[0]}")
    print(f"Total samples: {len(data)}")
    print()

    # Test different parameter combinations
    parameter_sets = [
        {
            "frames_to_average": 60,
            "sample_skip": 40,
            "description": "More frames, less overlap",
        },
        {
            "frames_to_average": 120,
            "sample_skip": 80,
            "description": "Standard processing",
        },
        {
            "frames_to_average": 240,
            "sample_skip": 160,
            "description": "More averaging, less frames",
        },
    ]

    print("Comparing different parameter settings:")
    print("-" * 70)

    for params in parameter_sets:
        frames_dict = compute_frames_from_df(
            dataframe=data,
            time_series_columns=selected_channel,
            heartrate_column="FHR",
            sampling_rate=80.0,
            average_frame_length=100,
            frames_to_average=params["frames_to_average"],
            sample_skip=params["sample_skip"],
            interpolation_kind="linear",
        )

        frames = frames_dict[selected_channel[0]]

        print(f"\n{params['description']}:")
        print(f"  frames_to_average: {params['frames_to_average']}")
        print(f"  sample_skip: {params['sample_skip']}")
        print(f"  Number of output frames: {len(frames)}")
        print(
            f"  Frame SNR estimate: {np.mean([np.mean(f) / np.std(f) for f in frames]):.2f}"
        )

    print()
    print("=" * 70)
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print(
        "║"
        + " " * 10
        + "PSAFE_CUSTOM LIBRARY - COMPLETE PIPELINE EXAMPLES"
        + " " * 8
        + "║"
    )
    print("╚" + "═" * 68 + "╝")
    print("\n")

    # Run the main complete pipeline example
    complete_pipeline_example()

    # Additional examples
    print("\n" + "+" * 70)
    print("Additional Examples:")
    print("+" * 70 + "\n")

    process_all_datasets_example()
    compare_wavelengths_example()
    custom_parameters_example()

    print("=" * 70)
    print(" " * 20 + "All examples completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • Use get_sheep_dataframe() to load pre-configured sheep data")
    print("  • Use get_sorted_ac_columns() to get standardized channel names")
    print("  • Use compute_frames_from_df() to process multiple channels at once")
    print("  • FHR (heart rate) is automatically converted to Hz and included")
    print("  • Adjust parameters based on your signal characteristics and goals")
    print("=" * 70)


if __name__ == "__main__":
    main()
