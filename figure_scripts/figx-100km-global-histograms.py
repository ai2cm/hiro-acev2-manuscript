"""Global 10-year (2014-2023) field histograms at 100 km resolution: the
X-SHiELD reference vs. the coarse ACE2S rollout, computed directly from the
underlying zarr stores (no precomputed histogram files at this resolution).

The loading, histogramming, caching and panel drawing all live in
``global_histogram_common.py``, so that ``figx-global-coarse-res-multifield-bias.py``
can draw the same distribution beside each field's maps and
``compute-100km-member-histograms.py`` can fill the cache for the rest of the
ensemble. What is left here is this figure's own two series and its layout.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from global_histogram_common import (
    CACHE,
    END,
    MEMBER_CACHE_DIR,
    PREDICTION_SOURCE,
    REFERENCE_SOURCE,
    START,
    histogram_source,
    is_source_cached,
    load_cached_histograms,
    plot_histogram,
    write_source_cache,
)
from plot_cache import SUPPLEMENTS_DIR

REFERENCE_DATASET = "/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr"
ACE_DATASET = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-coarse-ace2s-10yr-inference/output_6hourly_predictions_ic0000.zarr"
)
SOURCES = {REFERENCE_SOURCE: REFERENCE_DATASET, PREDICTION_SOURCE: ACE_DATASET}

OUT_PATH = SUPPLEMENTS_DIR / "global-100km-10yr-histograms.png"


def load_histograms(refresh: bool = False) -> dict[str, np.ndarray]:
    """Cached histograms, computing this figure's own two series if missing.

    ``refresh`` recomputes both series into the per-series cache, whose files
    take precedence over the single-file cache when the two layers are merged.
    """
    for source, path in SOURCES.items():
        if is_source_cached(source) and not refresh:
            continue
        print(f"computing histograms for {source}: {path}", flush=True)
        print(f"wrote {write_source_cache(source, histogram_source(source, path))}")
    print(f"loading plot data from {CACHE} and {MEMBER_CACHE_DIR}", flush=True)
    return load_cached_histograms()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="recompute and overwrite this figure's two cached series",
    )
    args = parser.parse_args()
    histograms = load_histograms(args.refresh)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 6)
    top_row = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]
    bottom_row = [gs[1, 1:3], gs[1, 3:5]]

    for i, (spec, var) in enumerate(
        zip(top_row, ["PRATEsfc", "wind_speed", "PRMSL"])
    ):
        plot_histogram(fig.add_subplot(spec), var, histograms, show_legend=(i == 0))
    for spec, var in zip(
        bottom_row, ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]
    ):
        plot_histogram(fig.add_subplot(spec), var, histograms)

    fig.suptitle(
        f"Global field histograms, {START[:4]}-{END[:4]}, 100 km resolution",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, transparent=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
