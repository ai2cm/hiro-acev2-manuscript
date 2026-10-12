#!/usr/bin/env python
"""
Perfect-prediction TC event time series for north_atlantic_000 (2023 holdout):
maximum 10m wind speed, minimum sea-level pressure, and IKE_TS of the
downscaled ensemble (mean, members, min-max spread) against the X-SHiELD 3km
reference, one panel per diagnostic.

Same figure as
``2026-07-16-tc-tracks-prod/event_timeseries/perfect-pred-2023-holdout/north_atlantic_000_timeseries.png``,
made by ``2026-07-16-tc-tracks-prod/plot_event_timeseries.py``; the series are
computed with that script's own functions (imported from it, so the numbers
are identical) and the plotting is reproduced here to save at 300 dpi on a
transparent background.

The series (the slow part -- every reference box is a read from the 3km
zarrs) are cached to CACHE_PATH; the figure is redrawn from the cache on every
run. Delete the cache file to recompute.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

TC_TRACKS_PROD_DIR = Path("/climate-default/home/annak/explore2/annak/2026-07-16-tc-tracks-prod")
EVENT_DIR = Path("/climate-default/home/annak/2026-07-17-perfect-pred-tc-tracks-2023-holdout")
EVENT = "north_atlantic_000"

FIG_DIR = Path(__file__).parent
OUT_PATH = FIG_DIR / "supplements" / "perfect_pred_tc_timeseries.png"
CACHE_PATH = FIG_DIR / "tc_timeseries_cache" / f"perfect_pred_{EVENT}_timeseries.nc"

_spec = importlib.util.spec_from_file_location(
    "plot_event_timeseries", TC_TRACKS_PROD_DIR / "plot_event_timeseries.py"
)
pet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pet)

DIAGNOSTICS = pet.diagnostics_for(pet.IkeSpec())

# matplotlib defaults (10 pt base: title 12, labels/ticks 10, suptitle 12) + 2 pt
FONT_SIZES = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
}


def compute_series() -> xr.Dataset:
    """Prediction (mean/lo/hi/per-member) and reference series per diagnostic,
    exactly as plot_event_timeseries.main() builds them with default options."""
    files = pet.group_event_files(EVENT_DIR)[EVENT]
    ike = pet.IkeSpec()

    ref_stores = [xr.open_zarr(pet.DEFAULT_REF_ZARR), xr.open_zarr(pet.DEFAULT_REF_PRMSL_ZARR)]
    data_vars = {}
    for diag in DIAGNOSTICS:
        mean, lo, hi, per_member = pet.prediction_series(files, diag, None, ike)
        data_vars[f"{diag.key}_pred_mean"] = ("time", mean)
        data_vars[f"{diag.key}_pred_lo"] = ("time", lo)
        data_vars[f"{diag.key}_pred_hi"] = ("time", hi)
        data_vars[f"{diag.key}_pred_members"] = (("time", "member"), per_member)
        ref = next(store for store in ref_stores if diag.field(store) is not None)
        data_vars[f"{diag.key}_ref"] = ("time", pet.reference_series(files, ref, diag, ike))
    times = pd.DatetimeIndex([t for t, _ in files])
    return xr.Dataset(data_vars, coords={"time": times})


def load_series() -> xr.Dataset:
    if CACHE_PATH.exists():
        print(f"Loading cached series: {CACHE_PATH}")
        return xr.open_dataset(CACHE_PATH).load()
    series = compute_series()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    series.to_netcdf(CACHE_PATH)
    print(f"Saved cache: {CACHE_PATH}")
    return series


def plot(series: xr.Dataset) -> None:
    """Mirrors plot_event_timeseries.plot_event."""
    plt.rcParams.update(FONT_SIZES)
    fig, axes = plt.subplots(len(DIAGNOSTICS), 1, figsize=(11, 3.2 * len(DIAGNOSTICS)), sharex=True)
    x = series.time.values
    for ax, diag in zip(axes, DIAGNOSTICS):
        mean = series[f"{diag.key}_pred_mean"].values
        lo = series[f"{diag.key}_pred_lo"].values
        hi = series[f"{diag.key}_pred_hi"].values
        per_member = series[f"{diag.key}_pred_members"].values
        ref = series[f"{diag.key}_ref"].values

        if per_member.shape[1] > 1:
            for m in range(per_member.shape[1]):
                ax.plot(x, per_member[:, m], color="C0", alpha=0.15, linewidth=0.8,
                        label="ensemble members" if m == 0 else None, zorder=1)
        if np.any(hi > lo):
            ax.fill_between(x, lo, hi, color="C0", alpha=0.2, label="prediction spread", zorder=2)
        ax.plot(x, mean, color="C0", marker="o", markersize=3, label="prediction", zorder=3)

        ax.plot(x, ref, color="C1", marker="s", markersize=3, label="X-SHiELD 3km", zorder=4)

        ax.set_ylabel(diag.ylabel)
        ax.set_title("Integrated kinetic energy" if diag.reduce == "ike_box" else diag.title)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    start, end = pd.Timestamp(x[0]), pd.Timestamp(x[-1])
    fig.suptitle(f"North Atlantic hurricane, {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


def main():
    plot(load_series())


if __name__ == "__main__":
    main()
