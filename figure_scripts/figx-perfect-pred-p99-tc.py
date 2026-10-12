"""
Predicted-vs-target scatter of each track's extreme-tail percentile, for the
HiRO-ACE v2 "perfect prediction" max-snapshot run (see
``tc_snapshot_common.py`` module docstring): one point per
``tc_<idx>_<YYYYMMDDTHH>.nc`` file in
``2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot``.

Restricted to snapshots timestamped in 2023 (``YEAR``, via
``tc_snapshot_common.filter_by_year``).

Three subplots, one per variable:

  * PRATEsfc (mm/day) and wind speed (m/s) -- 99.99th percentile, the upper
    tail that matters for a storm's peak rain/wind.
  * PRMSL (hPa) -- 0.01st percentile, since a TC's intensity shows up as how
    low central pressure gets, not how high (see ``figx-perfect-pred-Phl-tc.py``).

The predicted field carries an 8-member ``sample`` ensemble dimension, the
target does not; each ensemble member's own percentile is plotted as its own
point (so a track contributes up to 8 predicted points, all against the same
target value), rather than collapsing the ensemble to one mean point per
track, so the ensemble spread is visible directly in the scatter.

The per-variable scatter points are cached in
``plot_data_cache/figx-perfect-pred-p99-tc/percentiles_<YEAR>.npz``; pass
``--refresh`` to recompute them from the snapshot files.

Usage:
    python figx-perfect-pred-p99-tc.py
    python figx-perfect-pred-p99-tc.py --refresh
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_cache import SUPPLEMENTS_DIR, cached_arrays
from series_colors import PERFECT_PRED_COLOR
from tc_snapshot_common import (
    filter_by_year,
    find_track_files,
    prmsl_field,
    target_prmsl_field,
    target_wind_speed_field,
    wind_speed_field,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = Path(
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot"
)
OUT_PATH = SUPPLEMENTS_DIR / "perfect-pred-percentile-scatter.png"
CACHE_SUBDIR = Path(__file__).stem
MAX_WORKERS = 16
YEAR = 2023

KG_M2_S_TO_MM_DAY = 86400.0


def _prate_predicted(ds: xr.Dataset) -> xr.DataArray | None:
    return ds["PRATEsfc_predicted"] * KG_M2_S_TO_MM_DAY if "PRATEsfc_predicted" in ds else None


def _prate_target(ds: xr.Dataset) -> xr.DataArray | None:
    return ds["PRATEsfc_target"] * KG_M2_S_TO_MM_DAY if "PRATEsfc_target" in ds else None


# (variable key, title, units, tail percentile, predicted field, target field)
VARS = [
    ("PRATEsfc", "Surface precipitation rate", "mm/day", 99.99, _prate_predicted, _prate_target),
    ("wind_speed", "Wind speed", "m/s", 99.99, wind_speed_field, target_wind_speed_field),
    ("PRMSL", "Sea level pressure", "hPa", 0.01, prmsl_field, target_prmsl_field),
]


def _per_sample_percentiles(predicted: xr.DataArray, pctl: float) -> np.ndarray:
    """Each ensemble sample's own percentile, as a 1D array (length 1 if
    ``predicted`` has no ``sample`` dimension)."""
    if "sample" in predicted.dims:
        by_sample = predicted.values.reshape(predicted.sizes["sample"], -1)
        return np.nanpercentile(by_sample, pctl, axis=1)
    return np.array([np.nanpercentile(predicted.values, pctl)])


def process_track(item: tuple) -> dict | None:
    """Worker: ``{var: (target_pctl, predicted_pctls)}`` for one track's
    snapshot, skipping any variable missing its predicted or target field.
    ``predicted_pctls`` holds one value per ensemble sample."""
    idx, path = item
    result = {}
    with xr.open_dataset(path) as ds:
        for var, _title, _units, pctl, predicted_fn, target_fn in VARS:
            predicted = predicted_fn(ds)
            target = target_fn(ds)
            if predicted is None or target is None:
                continue
            target_vals = target.values
            target_pctl = float(np.nanpercentile(target_vals[np.isfinite(target_vals)], pctl))
            predicted_pctls = _per_sample_percentiles(predicted.load(), pctl)
            result[var] = (target_pctl, predicted_pctls)
    return (idx, result) if result else None


def collect_percentiles(root: Path, max_workers: int, year: int | None = None) -> dict:
    """``{var: (target_values, predicted_values, n_tracks)}``: one predicted
    point per ensemble sample, with its track's target value repeated
    alongside it, so ensemble spread is visible in the scatter. ``year``, if
    given, restricts to snapshots timestamped in that year (see
    ``filter_by_year``)."""
    files = find_track_files(root)
    if year is not None:
        files = filter_by_year(files, year)
    items = list(files.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(process_track, items)

    per_var = {var: ([], [], 0) for var, *_ in VARS}
    n_tracks = 0
    for result in filter(None, results):
        _idx, by_var = result
        n_tracks += 1
        for var, (target_pctl, predicted_pctls) in by_var.items():
            targets, predicteds, n = per_var[var]
            targets.extend([target_pctl] * len(predicted_pctls))
            predicteds.extend(predicted_pctls)
            per_var[var] = (targets, predicteds, n + 1)
    logger.info("Collected percentiles for %d of %d track(s)", n_tracks, len(files))
    return {
        var: (np.array(targets), np.array(predicteds), n)
        for var, (targets, predicteds, n) in per_var.items()
    }


def load_percentiles(refresh: bool = False) -> dict:
    """Cached ``collect_percentiles(ROOT, MAX_WORKERS, year=YEAR)``, flattened
    to ``<var>_target``/``<var>_predicted``/``<var>_n_tracks`` npz arrays."""

    def compute() -> dict:
        arrays = {}
        for var, (targets, predicteds, n) in collect_percentiles(
            ROOT, MAX_WORKERS, year=YEAR
        ).items():
            arrays[f"{var}_target"] = targets
            arrays[f"{var}_predicted"] = predicteds
            arrays[f"{var}_n_tracks"] = np.array(n)
        return arrays

    arrays = cached_arrays(f"{CACHE_SUBDIR}/percentiles_{YEAR}.npz", compute, refresh)
    return {
        var: (arrays[f"{var}_target"], arrays[f"{var}_predicted"], int(arrays[f"{var}_n_tracks"]))
        for var, *_ in VARS
    }


def plot_scatter(
    ax,
    targets: np.ndarray,
    predicteds: np.ndarray,
    n_tracks: int,
    title: str,
    units: str,
    pctl: float,
    show_legend: bool = True,
) -> None:
    ax.scatter(
        targets,
        predicteds,
        s=10,
        color=PERFECT_PRED_COLOR,
        alpha=0.4,
        edgecolors="none",
        label="Predicted (ensemble members)",
    )
    lo = min(targets.min(), predicteds.min())
    hi = max(targets.max(), predicteds.max())
    margin = 0.05 * (hi - lo)
    lims = (lo - margin, hi + margin)
    ax.plot(lims, lims, linestyle="--", color="black", lw=1.2, label="1:1")
    slope, intercept = np.polyfit(targets, predicteds, 1)
    ax.plot(
        lims,
        [slope * x + intercept for x in lims],
        linestyle="-",
        color=PERFECT_PRED_COLOR,
        lw=1.2,
        label="Best fit",
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Target ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(f"{title}\n({pctl:g}th percentile, N={n_tracks} tracks)", fontsize=10)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Recompute the cached percentiles."
    )
    args = parser.parse_args()

    per_var = load_percentiles(args.refresh)

    fig, axes = plt.subplots(1, len(VARS), figsize=(4.2 * len(VARS), 4.2))
    for i, (ax, (var, title, units, pctl, _predicted_fn, _target_fn)) in enumerate(zip(axes, VARS)):
        targets, predicteds, n_tracks = per_var[var]
        plot_scatter(ax, targets, predicteds, n_tracks, title, units, pctl, show_legend=(i == 0))

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
