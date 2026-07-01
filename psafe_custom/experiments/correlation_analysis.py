"""
Correlation Analysis: AC Ratios vs. Saturation

This script computes the correlation between AC ratios (for all 5 channels)
and averaged saturation values for a given sheep experiment.
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr

from psafe_custom import (
    compute_ac_ratio,
    compute_averaged_fsao2,
    get_sheep_dataframe,
    get_sorted_ac_columns,
)


def compute_correlations(ac_ratios: dict[str, np.ndarray], avg_saturation: np.ndarray) -> dict:
    """
    Compute correlations between AC ratios and saturation for all channels.

    Args:
        ac_ratios: Dictionary where keys are "Ratio chX" and values are AC ratio arrays
        avg_saturation: Array of averaged saturation values

    Returns:
        Dictionary containing correlation results for each channel
    """
    results = {}

    for ch_num in range(1, 6):  # ch1 to ch5
        ratio_key = f"Ratio ch{ch_num}"

        if ratio_key in ac_ratios:
            ratio_values = ac_ratios[ratio_key]

            # Compute Pearson correlation
            pearson_corr, pearson_pval = pearsonr(ratio_values, avg_saturation)

            # Compute Spearman correlation
            spearman_corr, spearman_pval = spearmanr(ratio_values, avg_saturation)

            results[f"ch{ch_num}"] = {
                "pearson_r": pearson_corr,
                "pearson_p": pearson_pval,
                "spearman_r": spearman_corr,
                "spearman_p": spearman_pval,
                "n_samples": len(ratio_values),
            }

    return results


def display_correlation_table(idx: int, results: dict, avg_saturation: np.ndarray):
    """
    Display correlation results in a Rich table.

    Args:
        idx: Sheep experiment index
        results: Dictionary of correlation results
        avg_saturation: Array of averaged saturation values
    """
    console = Console()

    # Find top 2 Pearson correlations by absolute value
    sorted_channels = sorted(
        results.keys(), key=lambda k: abs(results[k]["pearson_r"]), reverse=True
    )
    top_1 = sorted_channels[0] if len(sorted_channels) > 0 else None
    top_2 = sorted_channels[1] if len(sorted_channels) > 1 else None

    # Create header with experiment info
    console.print(
        "[bold cyan]Correlation Analysis: AC Ratios vs. Saturation[/bold cyan]"
    )
    console.print(f"[yellow]Experiment Index:[/yellow] {idx}")
    console.print(f"[yellow]Number of samples:[/yellow] {len(avg_saturation)}")
    console.print(f"[yellow]Saturation mean:[/yellow] {np.mean(avg_saturation):.2f}%")
    console.print(f"[yellow]Saturation std:[/yellow] {np.std(avg_saturation):.2f}%")
    console.print()

    # Create table
    table = Table(
        title="Correlation Results", show_header=True, header_style="bold magenta"
    )

    # Add columns
    table.add_column("Channel", style="cyan", justify="center")
    table.add_column("Pearson r", justify="right")
    table.add_column("Pearson p", justify="right")
    table.add_column("Spearman ρ", justify="right")
    table.add_column("Spearman p", justify="right")

    # Add rows for each channel
    for ch_num in range(1, 6):
        ch_key = f"ch{ch_num}"

        if ch_key in results:
            res = results[ch_key]

            # Color code Pearson r based on ranking
            if ch_key == top_1:
                pearson_color = "green"
            elif ch_key == top_2:
                pearson_color = "dark_orange"
            else:
                pearson_color = "white"

            # Format p-values with scientific notation if very small
            pearson_p_str = (
                f"{res['pearson_p']:.2e}"
                if res["pearson_p"] < 0.001
                else f"{res['pearson_p']:.4f}"
            )
            spearman_p_str = (
                f"{res['spearman_p']:.2e}"
                if res["spearman_p"] < 0.001
                else f"{res['spearman_p']:.4f}"
            )

            # Add significance markers
            pearson_sig = (
                "***"
                if res["pearson_p"] < 0.001
                else "**"
                if res["pearson_p"] < 0.01
                else "*"
                if res["pearson_p"] < 0.05
                else ""
            )
            spearman_sig = (
                "***"
                if res["spearman_p"] < 0.001
                else "**"
                if res["spearman_p"] < 0.01
                else "*"
                if res["spearman_p"] < 0.05
                else ""
            )

            table.add_row(
                f"ch{ch_num}",
                f"[{pearson_color}]{res['pearson_r']:+.4f}[/{pearson_color}] {pearson_sig}",
                pearson_p_str,
                f"{res['spearman_r']:+.4f} {spearman_sig}",
                spearman_p_str,
            )
        else:
            table.add_row(
                f"ch{ch_num}",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
            )

    console.print(table)
    console.print()

    # Add legend
    console.print("[bold]Significance levels:[/bold]")
    console.print("  * p < 0.05   ** p < 0.01   *** p < 0.001")
    console.print()
    console.print("[bold]Color coding:[/bold]")
    console.print(
        "  [green]Green[/green] = Best correlation   [dark_orange]Orange[/dark_orange] = Second best correlation"
    )
    console.print()


def main(idx: int = 7):
    """
    Main function to run correlation analysis.
    """
    # Parameters
    sampling_rate = 80.0  # Hz
    frames_to_average = 120
    sample_skip = 80

    # Load sheep data
    data = get_sheep_dataframe(idx)

    # Get all AC columns (both wavelengths for all 5 channels)
    ac_columns = get_sorted_ac_columns()
    all_channels = ac_columns["wv1"] + ac_columns["wv2"]

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

    # Compute correlations
    results = compute_correlations(ac_ratios, avg_saturation)

    # Display results
    display_correlation_table(idx, results, avg_saturation)

    # Display best correlating channel
    console = Console()
    best_channel = max(results.keys(), key=lambda k: abs(results[k]["pearson_r"]))
    best_corr = results[best_channel]["pearson_r"]

    console.print(
        f"[bold green]Best correlating channel:[/bold green] {best_channel} (r = {best_corr:+.4f})"
    )
    console.print()


if __name__ == "__main__":
    main(3)
