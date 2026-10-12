"""Combined TC track figure from the coarse track-detection ``tracks.csv`` files.

Row 1: global TC track density (cyclone-days/yr) for each dataset.
Row 2: the density bias of the model against the reference, plus the
distribution of per-track maximum 10m wind speed.

Merges ``figx-coarse-datasets-tc-track-density.py`` (maps) and
``figx-coarse-tc-max-wind-histograms.py`` (histogram), which read the same
two CSVs.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from series_colors import HIRO_ACE_COLOR, REFERENCE_COLOR

HERE = Path(__file__).parent
DATASETS = {
    "X-SHiELD": HERE / "track_detection" / "xshield-reference" / "tracks.csv",
    "ACE2S-X-SHiELD": HERE / "track_detection" / "ace2s-shieldplus-best-inf-ckpt-10yr-rollout" / "tracks.csv",
}
REFERENCE = "X-SHiELD"
COLORS = {
    "X-SHiELD": REFERENCE_COLOR,
    "ACE2S-X-SHiELD": HIRO_ACE_COLOR,
}

# 4x4 degree bins: fine enough to resolve ocean-basin structure in TC track
# density without the count-per-bin noise of a finer grid at ~10 years of data.
BIN_SIZE_DEG = 4.0
LON_EDGES = np.arange(0, 360 + BIN_SIZE_DEG, BIN_SIZE_DEG)
LAT_EDGES = np.arange(-90, 90 + BIN_SIZE_DEG, BIN_SIZE_DEG)
# 6-hourly track points -> cyclone-days per bin per year.
HOURS_PER_POINT = 6.0

WIND_BIN_WIDTH = 2.0  # m/s
# Colorbar geometry, in figure-width fractions.
CBAR_WIDTH = 0.012
CBAR_PAD = 0.012
# Room for a colorbar's rotated label plus the next panel's y-axis.
CBAR_LABEL_GAP = 0.105
OUT_PATH = HERE / "main_figures" / "coarse_tc_track_comparison.png"


def track_days_per_year(df: pd.DataFrame) -> np.ndarray:
    n_years = (df["time"].max() - df["time"].min()).days / 365.25
    hist, _, _ = np.histogram2d(df["lon"], df["lat"], bins=[LON_EDGES, LAT_EDGES])
    return hist * (HOURS_PER_POINT / 24.0) / n_years


def yearly_track_counts(df: pd.DataFrame) -> pd.Series:
    """Number of distinct tracks starting in each calendar year."""
    track_years = df.groupby("track_id")["time"].min().dt.year
    return track_years.value_counts().sort_index()


def plot_map(ax, field: np.ndarray, title: str, subtitle: str, **kwargs) -> object:
    ax.set_global()
    mesh = ax.pcolormesh(
        LON_EDGES,
        LAT_EDGES,
        field.T,
        transform=ccrs.PlateCarree(),
        zorder=0,
        **kwargs,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black", zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
    return mesh


def plot_wind_histogram(ax, frames: dict) -> None:
    peaks = {
        name: df.groupby("track_id")["wind"].max() for name, df in frames.items()
    }
    all_values = pd.concat(peaks.values())
    n_bins = int((all_values.max() // WIND_BIN_WIDTH) + 2)
    bins = [i * WIND_BIN_WIDTH for i in range(n_bins + 1)]

    for name, values in peaks.items():
        color = COLORS[name]
        ax.hist(
            values,
            bins=bins,
            density=False,
            label=f"{name} (N={len(values)})",
            color=to_rgba(color, 0.4),
            edgecolor=color,
            linewidth=1.3,
        )
    ax.set_xlabel("Max 10m wind speed per track (m/s)")
    ax.set_ylabel("Counts")
    ax.set_xlim(10, None)
    ax.legend(loc="upper right", fontsize=12)


def main() -> None:
    frames = {}
    densities = {}
    track_stats = {}
    for title, path in DATASETS.items():
        # xshield-output-tracks/tracks.csv has already had its first year
        # (2013-01-12 to 2014-01-12) trimmed so both runs cover a comparable
        # span starting 2014-01-12.
        df = pd.read_csv(path, parse_dates=["time"])
        frames[title] = df
        densities[title] = track_days_per_year(df)
        counts = yearly_track_counts(df)
        track_stats[title] = (counts.mean(), counts.std())

    reference = densities[REFERENCE]
    biases = {
        title: density - reference
        for title, density in densities.items()
        if title != REFERENCE
    }

    # A handful of bins (e.g. a spurious quasi-stationary detection over land)
    # can be far denser than any real ocean-basin hotspot and would otherwise
    # swamp the color scale; cap at the 99th percentile of nonzero bins and
    # let those outliers saturate instead.
    all_nonzero = np.concatenate([d[d > 0].ravel() for d in densities.values()])
    vmax = np.percentile(all_nonzero, 99)
    # Symmetric color scale capped the same way so a few outlier bins don't
    # wash out the rest of the bias map.
    all_abs = np.concatenate([np.abs(b).ravel() for b in biases.values()])
    bias_vmax = np.percentile(all_abs[all_abs > 0], 99)

    map_proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(14.0, 7.0))
    # Row 2 is taller than the 2:1 maps in row 1 so the histogram is not
    # squashed; the bias map keeps its own aspect and centers in its cell.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.04, wspace=0.18)

    density_axes = [fig.add_subplot(gs[0, i], projection=map_proj) for i in range(2)]
    for ax, (title, density) in zip(density_axes, densities.items()):
        mean, std = track_stats[title]
        mesh = plot_map(
            ax,
            density,
            title,
            f"{mean:.1f} ± {std:.1f} tracks/yr",
            cmap="inferno_r",
            vmin=0,
            vmax=vmax,
        )

    bias_ax = fig.add_subplot(gs[1, 0], projection=map_proj)
    (bias_title, bias), = biases.items()
    bias_mesh = plot_map(
        bias_ax,
        bias,
        f"{bias_title} minus {REFERENCE} track density bias",
        "",
        cmap="RdBu_r",
        vmin=-bias_vmax,
        vmax=bias_vmax,
    )

    hist_ax = fig.add_subplot(gs[1, 1])
    plot_wind_histogram(hist_ax, frames)

    density_cbar = fig.colorbar(
        mesh,
        ax=density_axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        extend="max",
        label=f"Cyclone-days per year ({BIN_SIZE_DEG:g}° x {BIN_SIZE_DEG:g}° bin)",
    )
    bias_cbar = fig.colorbar(
        bias_mesh,
        ax=bias_ax,
        orientation="vertical",
        fraction=0.04,
        pad=0.02,
        extend="both",
        label=f"Bias ({BIN_SIZE_DEG:g}° x {BIN_SIZE_DEG:g}° bin)",
    )
    # Cartopy shrinks a map to its 2:1 aspect only at draw time, so lay the
    # row out against the drawn extents rather than against the gridspec cells.
    fig.canvas.draw()

    # Each colorbar hugs the right edge of the map it describes, spanning
    # exactly that map's height.
    for cbar, ax in ((density_cbar, density_axes[1]), (bias_cbar, bias_ax)):
        pos = ax.get_position()
        cbar.ax.set_aspect("auto")
        cbar.ax.set_position([pos.x1 + CBAR_PAD, pos.y0, CBAR_WIDTH, pos.height])

    # The histogram then starts clear of the bias colorbar and its rotated
    # label, and ends flush with the map above it.
    bias_pos = bias_ax.get_position()
    hist_x0 = bias_cbar.ax.get_position().x1 + CBAR_LABEL_GAP
    hist_ax.set_position(
        [
            hist_x0,
            bias_pos.y0,
            density_axes[1].get_position().x1 - hist_x0,
            bias_pos.height,
        ]
    )

    # Panel labels, each just above the top-left corner of its own axes
    # frame. The panel titles are centered, so nothing collides.
    panels = [density_axes[0], density_axes[1], bias_ax, hist_ax]
    # d) takes its x from b) rather than from the histogram's own left edge,
    # which is inset to clear the bias colorbar, so the two line up in a column.
    label_x = [ax.get_position().x0 for ax in panels]
    label_x[3] = label_x[1]
    for letter, ax, x in zip("abcd", panels, label_x):
        pos = ax.get_position()
        fig.text(
            x,
            pos.y1 + 0.008,
            f"{letter})",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
