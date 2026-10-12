"""Global 2023 surface precipitation histogram: HiRO-ACE v1 vs. v2 perfect
prediction, against the shared X-SHiELD reference.

v1 only ran a precip-only perfect-prediction rollout, so its histogram is a
one-off precomputed file rather than living alongside the other v2 variables
under ``2026-08-03-hiro-perfect-pred-global-2023``.
"""

import argparse
from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from plot_cache import SUPPLEMENTS_DIR, cached_arrays
from series_colors import PERFECT_PRED_COLOR, REFERENCE_COLOR

HERE = Path(__file__).parent

ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-hiro-perfect-pred-global-2023"
)
REFERENCE_PATH = f"{ROOT}/reference_histograms/PRATEsfc_histogram.nc"
V2_PATH = f"{ROOT}/prediction_histograms/PRATEsfc_histogram.nc"
V1_PATH = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-hiro-perfect-pred-global-2023/"
    "perfect_prediction_hiro_v1_precip_only_histograms/PRATEsfc_histogram.nc"
)
OUT_PATH = SUPPLEMENTS_DIR / "histogram-v1-vs-v2-precip.png"
# raw (untrimmed, kg/m2/s) counts and edges of each source, as
# "<name>_counts" / "<name>_edges"; raw so --drop-negative works from the cache
CACHE_NAME = "figx-v1-to-v2-precip-histogram/PRATEsfc_histograms.npz"

# Okabe-Ito vermillion; distinct from REFERENCE_COLOR (green) and
# PERFECT_PRED_COLOR (blue), which are reused here for v2's reference/prediction
V1_COLOR = "#D55E00"

SOURCES = {
    "reference": (REFERENCE_PATH, REFERENCE_COLOR, "X-SHiELD reference"),
    "v2": (V2_PATH, PERFECT_PRED_COLOR, "v2: this work"),
    "v1": (V1_PATH, V1_COLOR, "v1"),
}
PERCENTILES = {99: "-", 99.99: "--", 99.9999: "--"}


def trim_empty_bins(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``counts`` and ``edges`` without their all-zero leading and trailing bins.

    Only the contiguous runs of empty bins at the two ends are dropped; empty
    bins that have populated bins on both sides are kept, so gaps within the
    distribution stay visible. Returned unchanged if every bin is empty.
    """
    counts, edges = np.asarray(counts), np.asarray(edges)
    (populated,) = np.nonzero(counts)
    if len(populated) == 0:
        return counts, edges
    first, last = populated[0], populated[-1]
    # bin i spans edges[i] to edges[i + 1], so keeping bins first..last needs
    # one more edge than bins
    return counts[first : last + 1], edges[first : last + 2]


def clamp_below_zero(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``counts`` and ``edges`` with all bins below zero merged into one bin at 0.

    The histograms' bin edges dip slightly negative near zero (float noise in
    the binning, not physical negative precipitation). Every bin whose upper
    edge is <= 0, plus the bin straddling zero, are summed into a single bin
    spanning 0 to that straddling bin's upper edge.
    """
    counts, edges = np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    (nonnegative,) = np.nonzero(edges[1:] > 0)
    first = nonnegative[0]
    clamped_count = counts[: first + 1].sum()
    counts = np.concatenate(([clamped_count], counts[first + 1 :]))
    edges = np.concatenate(([0.0], edges[first + 1 :]))
    return counts, edges


def drop_negative_bins(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``counts`` and ``edges`` with bins entirely below zero dropped outright.

    Unlike ``clamp_below_zero``, the dropped bins' counts are discarded rather
    than folded into the zero-straddling bin, which is left as-is.
    """
    counts, edges = np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    (nonnegative,) = np.nonzero(edges[1:] > 0)
    first = nonnegative[0]
    return counts[first:].copy(), edges[first:].copy()


def histogram_percentile(
    counts: np.ndarray, edges: np.ndarray, percentile: float
) -> float:
    """Value at ``percentile`` of the distribution described by the histogram.

    The bin the percentile lands in is interpolated linearly, which assumes the
    samples are spread uniformly within a bin; resolution is therefore bounded
    by the bin width, and the far tail is only meaningful if the sample count is
    well above 100 / (100 - percentile).
    """
    counts = np.asarray(counts, dtype=float)
    cdf = np.cumsum(counts)
    target = cdf[-1] * percentile / 100
    i = min(int(np.searchsorted(cdf, target, side="left")), len(counts) - 1)
    below = cdf[i - 1] if i > 0 else 0.0
    within = (target - below) / counts[i] if counts[i] > 0 else 0.0
    return edges[i] + within * (edges[i + 1] - edges[i])


def print_percentile_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Percentile values (mm/day) for each source, one row per percentile."""
    names = list(series)
    header = f"{'percentile':>12}" + "".join(f"{SOURCES[n][2]:>20}" for n in names)
    print(header)
    print("-" * len(header))
    for percentile in PERCENTILES:
        row = f"{percentile:>12g}"
        for name in names:
            counts, edges = series[name]
            row += f"{histogram_percentile(counts, edges, percentile):>20.2f}"
        print(row)
    print()


def count_negative_bins(counts: np.ndarray, edges: np.ndarray) -> int:
    """Number of samples in bins entirely below zero (excludes the
    zero-straddling bin, which isn't purely negative)."""
    counts, edges = np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    (nonnegative,) = np.nonzero(edges[1:] > 0)
    first = nonnegative[0]
    return int(counts[:first].sum())


def load_raw_histograms(refresh: bool = False) -> dict[str, np.ndarray]:
    """Each source's raw counts and edges, from the plot-data cache if present."""

    def compute() -> dict[str, np.ndarray]:
        arrays = {}
        for name, (path, _color, _label) in SOURCES.items():
            ds = xr.open_dataset(path)
            arrays[f"{name}_counts"] = ds.counts.values
            arrays[f"{name}_edges"] = ds.edges.values
        return arrays

    return cached_arrays(CACHE_NAME, compute, refresh)


def load_series(
    drop_negative: bool = False, refresh: bool = False
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    negative_bin_handler = drop_negative_bins if drop_negative else clamp_below_zero
    verb = "dropped" if drop_negative else "clamped to zero"
    raw = load_raw_histograms(refresh=refresh)
    series = {}
    for name, (_path, _color, label) in SOURCES.items():
        raw_counts, raw_edges = raw[f"{name}_counts"], raw[f"{name}_edges"]
        n_negative = count_negative_bins(raw_counts, raw_edges)
        print(f"{label}: {n_negative} negative-bin counts {verb}")
        counts, edges = negative_bin_handler(raw_counts, raw_edges)
        counts, edges = trim_empty_bins(counts, edges)
        edges = edges * 86400  # kg/m2/s -> mm/day
        series[name] = (counts, edges)
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop-negative",
        action="store_true",
        help="exclude bins entirely below zero instead of clamping them into "
        "the zero-straddling bin",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompute the cached histograms from the source histogram files",
    )
    args = parser.parse_args()

    series = load_series(drop_negative=args.drop_negative, refresh=args.refresh)
    print_percentile_table(series)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (counts, edges) in series.items():
        color, label = SOURCES[name][1], SOURCES[name][2]
        # density, not mass per bin: the three sources have different bin
        # widths, so counts / counts.sum() is not comparable between them
        density = counts / (counts.sum() * np.diff(edges))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, density, color=color, label=label)

    x_span = np.subtract(*ax.get_xlim()[::-1])
    for percentile, style in PERCENTILES.items():
        values = {
            name: histogram_percentile(counts, edges, percentile)
            for name, (counts, edges) in series.items()
        }
        for name, value in values.items():
            ax.axvline(
                value,
                color=SOURCES[name][1],
                linestyle=style,
                linewidth=1,
                alpha=0.8,
            )
        # label the group once, just right of its rightmost line; x is in data
        # coordinates and y in axes fraction, so the text sits mid-plot
        ax.text(
            max(values.values()) + 0.01 * x_span,
            0.5,
            f"{percentile:g}th",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="0.35",
        )
    #ax.set_xscale('log')
    ax.set_yscale("log")
    ax.set_xlim(10, None)
    ax.set_xlabel("mm/day")
    ax.set_ylabel("density (mm/day)$^{-1}$")
    ax.legend()
    ax.set_title("Surface precipitation rate: v1 vs. v2, global 2023")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, transparent=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
