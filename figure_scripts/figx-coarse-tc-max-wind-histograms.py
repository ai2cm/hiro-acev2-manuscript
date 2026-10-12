"""Histogram of per-track maximum 10m wind speed: X-SHiELD reference vs.
HiRO-ACE, from the coarse track-detection ``tracks.csv`` files (c.f.
``figx-coarse-datasets-tc-track-density.py``, which reads the same files for
track density).

Unlike ``figx-max-wind-ike-histograms.py`` (which reads per-track max-snapshot
netCDF files), each track here is already a full 6-hourly time series in one
CSV, so a track's peak wind is simply the max of its own ``wind`` column --
no ensemble dimension, no snapshot-file lookup.

Each dataset's per-track peaks are cached in
``plot_data_cache/figx-coarse-tc-max-wind-histograms/<track dir>_per_track_max_wind.csv``;
pass ``--refresh`` to recompute them from the ``tracks.csv`` files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

from plot_cache import SUPPLEMENTS_DIR, cached_frame
from series_colors import HIRO_ACE_COLOR, REFERENCE_COLOR

HERE = Path(__file__).parent
# Track CSVs are read from the parent figure directory, only when the cache
# below is missing.
PARENT = HERE.parent
DATASETS = {
    "X-SHiELD reference": PARENT / "track_detection" / "xshield-reference" / "tracks.csv",
    "HiRO-ACEv2": PARENT / "track_detection" / "ace2s-shieldplus-best-inf-ckpt-10yr-rollout" / "tracks.csv",
}
COLORS = {
    "X-SHiELD reference": REFERENCE_COLOR,
    "HiRO-ACEv2": HIRO_ACE_COLOR,
}

BIN_WIDTH = 2.  # m/s
OUT_PATH = SUPPLEMENTS_DIR / "coarse-tc-max-wind-histogram.png"
CACHE_SUBDIR = Path(__file__).stem


def per_track_max_wind(path: Path, refresh: bool = False) -> pd.Series:
    def compute() -> pd.DataFrame:
        df = pd.read_csv(path, usecols=["track_id", "wind"])
        return df.groupby("track_id")["wind"].max().reset_index()

    # Named after the track-detection directory, e.g. "xshield-reference".
    name = f"{CACHE_SUBDIR}/{path.parent.name}_per_track_max_wind.csv"
    return cached_frame(name, compute, refresh).set_index("track_id")["wind"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="Recompute the cached per-track peaks."
    )
    args = parser.parse_args()

    peaks = {
        name: per_track_max_wind(path, args.refresh) for name, path in DATASETS.items()
    }

    all_values = pd.concat(peaks.values())
    n_bins = int((all_values.max() // BIN_WIDTH) + 2)
    bins = [i * BIN_WIDTH for i in range(n_bins + 1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_yscale("log")
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
    ax.set_title("Distribution of per-track maximum 10m wind speed")
    ax.set_xlim(10, None)
    ax.legend(loc="upper right", fontsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
