"""
SQI-Filtered Correlation Analysis: AC Ratios vs. Saturation

This script computes the correlation between AC ratios (for all 5 channels)
and averaged saturation values, using Signal Quality Index (SQI) filtering
to exclude low-quality signal segments.
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr

from psafe_custom import (
    compute_averaged_fsao2,
    get_sheep_dataframe,
    get_sorted_ac_columns,
)
from psafe_custom.apply_pasfe import compute_ac_ratio, compute_frames_from_df
from psafe_custom.sqi import get_sqi


def compute_sqi_per_channel(
    frames_dict: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """
    Compute SQI for each frame in each channel.

    Args:
        frames_dict: Dictionary where keys are channel names and values are lists of frames

    Returns:
        Dictionary where keys are channel names and values are numpy arrays of SQI values
    """
    sqi_dict = {}

    for channel_name, frames_list in frames_dict.items():
        # Compute SQI for each frame
        sqi_values = np.zeros(len(frames_list))
        for idx, frame in enumerate(frames_list):
            sqi_values[idx] = get_sqi(frame, sqi_type="perfusion")

        sqi_dict[channel_name] = sqi_values

    return sqi_dict


def average_sqi_per_channel(sqi_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Average SQI values across both wavelengths for each channel.

    Args:
        sqi_dict: Dictionary with keys like "ch1voltsWL1", "ch1voltsWL2", etc.

    Returns:
        Dictionary with keys like "ch1", "ch2", etc., and values are averaged SQI arrays
    """
    averaged_sqi = {}

    for ch_num in range(1, 6):  # ch1 to ch5
        wl1_key = None
        wl2_key = None

        # Find WL1 and WL2 keys for this channel
        for key in sqi_dict.keys():
            if f"ch{ch_num}" in key and "WL1" in key:
                wl1_key = key
            elif f"ch{ch_num}" in key and "WL2" in key:
                wl2_key = key

        # Average if both exist
        if wl1_key is not None and wl2_key is not None:
            sqi_wl1 = sqi_dict[wl1_key]
            sqi_wl2 = sqi_dict[wl2_key]

            # Average across wavelengths
            avg_sqi = (sqi_wl1 + sqi_wl2) / 2.0
            averaged_sqi[f"ch{ch_num}"] = avg_sqi

    return averaged_sqi


def display_comparison_table(
    idx: int,
    ac_ratios: dict[str, np.ndarray],
    avg_saturation: np.ndarray,
    averaged_sqi: dict[str, np.ndarray],
    sqi_threshold: float,
):
    """
    Display comparison table with unfiltered and filtered correlations.

    Args:
        idx: Sheep experiment index
        ac_ratios: Dictionary with keys like "Ratio ch1" and values are AC ratio arrays
        avg_saturation: Array of averaged saturation values
        averaged_sqi: Dictionary with keys like "ch1" and values are SQI arrays
        sqi_threshold: SQI threshold (keep samples with SQI < threshold)
    """
    console = Console()

    # Header
    console.print("[bold cyan]SQI-Filtered Correlation Analysis[/bold cyan]")
    console.print(f"[yellow]Experiment Index:[/yellow] {idx}")
    console.print(f"[yellow]SQI Threshold:[/yellow] < {sqi_threshold}")
    console.print(f"[yellow]Total Samples:[/yellow] {len(avg_saturation)}")
    console.print()

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Channel", style="cyan", justify="center")
    table.add_column("Unfiltered r", justify="right")
    table.add_column("Filtered r", justify="right")
    table.add_column("% Kept", justify="right")

    # Compute correlations for each channel
    for ch_num in range(1, 6):
        ratio_key = f"Ratio ch{ch_num}"
        sqi_key = f"ch{ch_num}"

        if ratio_key in ac_ratios and sqi_key in averaged_sqi:
            # Unfiltered correlation
            ratio_values = ac_ratios[ratio_key]
            unfilt_r, _ = pearsonr(ratio_values, avg_saturation)

            # Filter by SQI
            sqi_values = averaged_sqi[sqi_key]
            good_quality_mask = sqi_values > sqi_threshold
            filtered_ratio = ratio_values[good_quality_mask]
            filtered_sat = avg_saturation[good_quality_mask]

            # Filtered correlation (if enough samples)
            if len(filtered_ratio) > 2:
                filt_r, _ = pearsonr(filtered_ratio, filtered_sat)
                filt_r_str = f"{filt_r:+.4f}"
            else:
                filt_r_str = "[red]N/A[/red]"

            # Percentage kept
            pct_kept = (np.sum(good_quality_mask) / len(good_quality_mask)) * 100

            table.add_row(
                f"ch{ch_num}",
                f"{unfilt_r:+.4f}",
                filt_r_str,
                f"{pct_kept:.1f}%",
            )
        else:
            table.add_row(
                f"ch{ch_num}",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
            )

    console.print(table)
    console.print()


def main(idx: int = 3, sqi_threshold: float = 0.3):
    """
    Main function to run SQI-filtered correlation analysis.

    Args:
        idx: Sheep experiment index
        sqi_threshold: SQI threshold (keep samples with SQI < threshold)
    """
    # Parameters
    sampling_rate = 80.0  # Hz
    frames_to_average = 120
    sample_skip = 80

    console = Console()
    console.print(f"[bold]Processing sheep experiment {idx}...[/bold]")

    # Load sheep data
    data = get_sheep_dataframe(idx)

    # Get all AC columns (both wavelengths for all 5 channels)
    ac_columns = get_sorted_ac_columns()
    all_channels = ac_columns["wv1"] + ac_columns["wv2"]

    # Compute frames for all channels
    frames_dict = compute_frames_from_df(
        dataframe=data,
        time_series_columns=all_channels,
        heartrate_column="FHR",
        sampling_rate=sampling_rate,
        average_frame_length=100,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
        interpolation_kind="linear",
    )

    # Compute SQI for each frame
    sqi_dict = compute_sqi_per_channel(frames_dict)

    # Average SQI per channel (across wavelengths)
    averaged_sqi = average_sqi_per_channel(sqi_dict)

    # Compute AC ratios
    ac_ratios = compute_ac_ratio(
        dataframe=data,
        time_series_columns=all_channels,
        heartrate_column="FHR",
        sampling_rate=sampling_rate,
        average_frame_length=100,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
        interpolation_kind="linear",
    )

    # Compute averaged saturation
    avg_saturation = compute_averaged_fsao2(
        dataframe=data,
        heartrate_column="FHR",
        sampling_rate=sampling_rate,
        frames_to_average=frames_to_average,
        sample_skip=sample_skip,
    )

    console.print("[bold green]Done![/bold green]")
    console.print()

    # Display comparison table
    display_comparison_table(idx, ac_ratios, avg_saturation, averaged_sqi, sqi_threshold)


if __name__ == "__main__":
    # You can change the experiment index and SQI threshold here
    main(idx=3, sqi_threshold=0.3)
