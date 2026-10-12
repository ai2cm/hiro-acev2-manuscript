"""
Bias of the HiRO-ACE v2 "perfect prediction" downscaling model's per-track
peak 10m wind speed and integrated kinetic energy (IKE) against the X-SHiELD
reference, for every ``tc_<idx>_<YYYYMMDDTHH>.nc`` snapshot in
``2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot`` (see
``tc_snapshot_common.py`` module docstring). Restricted to snapshots
timestamped in 2023 (``YEAR``, via ``tc_snapshot_common.filter_by_year``), as
in ``figx-perfect-pred-p99-tc.py``.

Each snapshot holds the downscaling model's ``*_predicted`` fields (an
8-member ``sample`` ensemble) and the colocated X-SHiELD ``*_target`` fields
(no ensemble dimension) at that track's peak-intensity time. For each track:

  * Peak 10m wind speed is the spatial max of the wind speed field (per
    ensemble sample for the prediction).
  * IKE is ``integrated_kinetic_energy`` (Powell and Reinhold 2007) over an
    8 deg storm-centered box, restricted to winds above 18 m/s -- the same
    ``IkeSpec``/``regional_reduce`` definition as
    ``figx-max-wind-ike-histograms.py``, copied in unmodified below since
    this is (like that script) the only one of this directory's scripts that
    needs it.

Bias is ``predicted - target`` (absolute) and ``(predicted - target) /
target`` (relative). The printed biases are *medians*, not means -- IKE for a
weak system sits right at the edge of the 18 m/s band that defines it, so a
small absolute error there can be a huge fraction of a near-zero target,
and a mean lets a handful of such tracks dominate the summary. The median is
robust to that. Each track's ensemble samples are first averaged down to a
single per-track predicted value -- so tracks with larger ensembles don't
outweigh tracks with smaller ones (the same 1/N-per-track principle as
``figx-max-wind-ike-histograms.peaks_to_values_weights``). The scatter
figure instead plots every ensemble sample as its own point against its
track's target (as in ``figx-perfect-pred-p99-tc.py``), so ensemble spread is
visible directly.

Two figures come out of the same computation:

  * ``main_figures/perfect_pred_wind_ike_bias_scatter.png`` -- the two
    predicted-vs-target bias panels, labelled a) IKE and b) peak 10m wind
    speed.
  * ``supplements/perfect_pred_ike_vs_wind_scatter.png`` -- IKE against peak
    wind speed directly (not predicted vs. target): one X-SHiELD reference
    point per track, and one HiRO-ACE prediction point per ensemble sample,
    so the model's wind/IKE relationship can be compared against the
    reference relationship and the ensemble spread around it is visible.
    One panel, so it carries no letter label.

A track whose target IKE is exactly 0 (no cell in the box exceeds the wind
threshold) is kept in every statistic -- its absolute bias is well-defined,
and only its relative bias (undefined, 0 in the denominator) is treated as
NaN and skipped via ``nanmedian`` rather than dropping the track outright.

Each variable's columns are, in order: median absolute bias, median relative
bias, absolute RMSE, and relative RMSE (the RMS of ``(predicted - target) /
target``). The two "bias" columns are medians and the two "RMSE" columns are
root-*mean*-squares, so the RMSEs are the outlier-sensitive pair -- see
``bias_stats``.

Biases are reported as a table with one row per group: all tracks, tracks
whose target peak wind exceeds 18 m/s (tropical-storm force), and tracks
exceeding 33 m/s (hurricane force) -- see ``WIND_GROUPS``. Weak systems below
these thresholds are the ones whose near-zero target IKE makes relative bias
noisy, so seeing progressively more intense subsets shows whether the
model's bias on genuine storms differs from its bias on marginal ones.

Usage:
    python figx-windspeed-ike-perfect-pred-tc-bias.py
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from series_colors import PERFECT_PRED_COLOR, REFERENCE_COLOR
from tc_snapshot_common import (
    _coord_names,
    filter_by_year,
    find_track_files,
    target_wind_speed_field,
    wind_speed_field,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot"
)
# Two figures out of the same computation: the IKE predicted-vs-target panel
# alone for the paper, and all three panels (that same IKE panel first) for
# the supplement.
DEFAULT_OUT = "main_figures/perfect_pred_wind_ike_bias_scatter.png"
DEFAULT_SUPPLEMENTAL_OUT = "supplements/perfect_pred_ike_vs_wind_scatter.png"
PANEL_LABELS = ("a)", "b)")
PANEL_LABEL_FONTSIZE = 12
DEFAULT_MAX_WORKERS = 16
# Restricted to 2023 snapshots only, as in figx-perfect-pred-p99-tc.py's YEAR.
YEAR = 2023
# (label, min target peak wind in m/s or None) rows for the printed bias
# table, from least to most restrictive -- None ("All tracks") first, then
# the standard tropical-storm and hurricane wind-speed cutoffs -- so the
# table shows whether bias on genuine storms differs from marginal ones.
WIND_GROUPS = [
    ("All tracks", None),
    ("Target max wind > 18 m/s (tropical storm)", 18.0),
    ("Target max wind > 33 m/s (hurricane)", 33.0),
    ("Target max wind > 50 m/s", 50.0),
]

# --------------------------------------------------------------------------- #
# Integrated kinetic energy (Powell and Reinhold 2007), copied unmodified from
# figx-max-wind-ike-histograms.py -- this is the only other script in this
# directory that needs the IKE definition, so it is kept local here rather
# than shared via tc_snapshot_common.py (see that module's own docstring).
# --------------------------------------------------------------------------- #
_EARTH_RADIUS_M = 6.371e6
DENSITY_OF_AIR = 1.0  # kg m^-3
IKE_LAYER_DEPTH_M = 1.0
_JOULES_PER_TJ = 1.0e12
DEFAULT_IKE_BOX_DEG = 8.0
DEFAULT_IKE_MIN_WIND = 18.0


@dataclass(frozen=True)
class IkeSpec:
    box_deg: float = DEFAULT_IKE_BOX_DEG
    min_wind: float | None = DEFAULT_IKE_MIN_WIND
    max_wind: float | None = None


def _area_weights(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    lat = da[lat_name]
    lon = da[lon_name]
    dlat = np.deg2rad(float(np.abs(np.diff(lat.values)).mean()))
    dlon = np.deg2rad(float(np.abs(np.diff(lon.values)).mean()))
    coslat = np.cos(np.deg2rad(lat))
    return _EARTH_RADIUS_M**2 * coslat * dlat * dlon


def integrated_kinetic_energy(
    speed: xr.DataArray,
    lat_name: str,
    lon_name: str,
    min_wind: float | None = DEFAULT_IKE_MIN_WIND,
    max_wind: float | None = None,
) -> xr.DataArray:
    cell_area = _area_weights(speed, lat_name, lon_name)
    if min_wind is not None:
        speed = speed.where(speed > min_wind)
    if max_wind is not None:
        speed = speed.where(speed < max_wind)
    energy_density = 0.5 * DENSITY_OF_AIR * IKE_LAYER_DEPTH_M * speed**2
    return (energy_density * cell_area).sum([lat_name, lon_name])


def _patch_center(da: xr.DataArray, lat_name: str, lon_name: str) -> tuple:
    return tuple(
        float((da[name].values.min() + da[name].values.max()) / 2.0)
        for name in (lat_name, lon_name)
    )


def _centered_box(
    da: xr.DataArray, lat_name: str, lon_name: str, lat0: float, lon0: float, box_deg: float
) -> xr.DataArray:
    half = box_deg / 2.0
    slices = {}
    for name, center in ((lat_name, lat0), (lon_name, lon0)):
        coord = da[name]
        descending = float(coord[0]) > float(coord[-1])
        lo, hi = center - half, center + half
        slices[name] = slice(hi, lo) if descending else slice(lo, hi)
    return da.sel(slices)


def _ike_box(da: xr.DataArray, lat_name: str, lon_name: str, box_deg: float) -> xr.DataArray:
    lat0, lon0 = _patch_center(da, lat_name, lon_name)
    return _centered_box(da, lat_name, lon_name, lat0, lon0, box_deg)


def track_ike(da: xr.DataArray, lat_name: str, lon_name: str, ike: IkeSpec) -> xr.DataArray:
    """IKE (TJ) over ``ike``'s storm-centered box and wind band, preserving
    any leading ``sample`` dim. Equivalent to
    ``figx-max-wind-ike-histograms.regional_reduce(..., "ike_box", ...)``."""
    box = _ike_box(da, lat_name, lon_name, ike.box_deg)
    joules = integrated_kinetic_energy(box, lat_name, lon_name, ike.min_wind, ike.max_wind)
    return joules / _JOULES_PER_TJ


# --------------------------------------------------------------------------- #
# Per-track predicted vs. target: peak wind and IKE
# --------------------------------------------------------------------------- #
def _process_track(item: tuple, ike: IkeSpec) -> tuple | None:
    """Worker: (idx, pred_wind_per_sample, target_wind, pred_ike_per_sample,
    target_ike) for one track, or None if either wind component is missing.
    ``item`` is (idx, path)."""
    idx, path = item
    with xr.open_dataset(path) as ds:
        pred_speed = wind_speed_field(ds)
        target_speed = target_wind_speed_field(ds)
        if pred_speed is None or target_speed is None:
            return None
        lat_name, lon_name = _coord_names(ds)
        pred_speed = pred_speed.load()
        target_speed = target_speed.load()

        if "sample" in pred_speed.dims:
            reduce_dims = [d for d in pred_speed.dims if d != "sample"]
            pred_wind = pred_speed.max(dim=reduce_dims, skipna=True).values
        else:
            pred_wind = np.array([float(pred_speed.max(skipna=True).values)])
        target_wind = float(target_speed.max(skipna=True).values)

        pred_ike = np.atleast_1d(
            np.asarray(track_ike(pred_speed, lat_name, lon_name, ike).values, dtype=float)
        )
        target_ike = float(
            np.atleast_1d(np.asarray(track_ike(target_speed, lat_name, lon_name, ike).values, dtype=float))[0]
        )
    return idx, np.atleast_1d(pred_wind), target_wind, pred_ike, target_ike


def collect_bias(directory: Path, ike: IkeSpec, max_workers: int, year: int | None = None) -> dict:
    """Return per-track and per-sample arrays needed for the bias figure and
    summary statistics:

    ``{"wind_target": ..., "wind_predicted": ..., "ike_target": ...,
    "ike_predicted": ...}`` -- one entry per (track, ensemble sample), target
    repeated across a track's samples (for the scatter figure) -- plus
    ``{"track_target_wind": ..., "track_pred_wind": ...,
    "track_target_ike": ..., "track_pred_ike": ...}`` -- one entry per track
    (the predicted value averaged over its ensemble samples first; see module
    docstring), for ``bias_stats`` to turn into the printed medians -- plus
    ``"track_n_samples"``, the ensemble size (number of ``*_predicted``
    samples) of each of those tracks, for the scatter figure's legend (see
    ``_ensemble_size_label``). Every track is kept in these, including one
    with zero target IKE. ``year``, if given, restricts to snapshots
    timestamped in that year (see ``filter_by_year``).
    """
    files = find_track_files(directory)
    if year is not None:
        files = filter_by_year(files, year)
    items = list(files.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_process_track, items, [ike] * len(items))

    wind_target, wind_predicted = [], []
    ike_target, ike_predicted = [], []
    track_target_wind, track_pred_wind = [], []
    track_target_ike, track_pred_ike = [], []
    track_n_samples = []
    n_tracks = 0
    n_zero_target_ike = 0
    for result in filter(None, results):
        _idx, pred_wind, tgt_wind, pred_ike, tgt_ike = result
        n_tracks += 1

        wind_target.extend([tgt_wind] * len(pred_wind))
        wind_predicted.extend(pred_wind)
        ike_target.extend([tgt_ike] * len(pred_ike))
        ike_predicted.extend(pred_ike)

        track_target_wind.append(tgt_wind)
        track_pred_wind.append(float(np.mean(pred_wind)))
        track_target_ike.append(tgt_ike)
        track_pred_ike.append(float(np.mean(pred_ike)))
        track_n_samples.append(len(pred_wind))
        if tgt_ike == 0:
            n_zero_target_ike += 1

    logger.info("%s: computed predicted/target peak wind and IKE for %d of %d track(s)",
                directory, n_tracks, len(files))
    if n_zero_target_ike:
        logger.info(
            "%d of %d track(s) have zero target IKE; kept for absolute bias, "
            "relative bias treated as undefined (NaN) for those",
            n_zero_target_ike, n_tracks,
        )

    return {
        "wind_target": np.array(wind_target),
        "wind_predicted": np.array(wind_predicted),
        "ike_target": np.array(ike_target),
        "ike_predicted": np.array(ike_predicted),
        "track_target_wind": np.array(track_target_wind),
        "track_pred_wind": np.array(track_pred_wind),
        "track_target_ike": np.array(track_target_ike),
        "track_pred_ike": np.array(track_pred_ike),
        "track_n_samples": np.array(track_n_samples),
        "n_tracks": n_tracks,
    }


# --------------------------------------------------------------------------- #
# Plotting: predicted-vs-target scatter, one subplot per variable, in the
# style of figx-perfect-pred-p99-tc.py's plot_scatter.
# --------------------------------------------------------------------------- #
def _ensemble_size_label(track_n_samples: np.ndarray) -> str:
    """``"8 ensemble members"``, or ``"8-16 ensemble members"`` if tracks
    don't all share the same ensemble size."""
    sizes = np.unique(track_n_samples)
    n = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes.min()}-{sizes.max()}"
    return f"{n} ensemble member{'s' if sizes.max() != 1 else ''}"


def plot_scatter(
    ax, target: np.ndarray, predicted: np.ndarray, n_tracks: int, title: str, units: str,
    ensemble_label: str, show_legend: bool = True,
) -> None:
    ax.scatter(
        target,
        predicted,
        s=10,
        color=PERFECT_PRED_COLOR,
        alpha=0.4,
        edgecolors="none",
        label=f"Predicted \n({ensemble_label})",
    )

    lo = min(target.min(), predicted.min())
    hi = max(target.max(), predicted.max())
    margin = 0.05 * (hi - lo)
    lims = (lo - margin, hi + margin)
    ax.plot(lims, lims, linestyle="--", color="black", lw=1.2, label="1:1")
    slope, intercept = np.polyfit(target, predicted, 1)
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
    ax.set_title(f"{title}\n(N={n_tracks} tracks)", fontsize=10)
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")


def plot_ike_vs_wind(ax, data: dict, n_tracks: int) -> None:
    """IKE vs. peak 10m wind speed for every track: one X-SHiELD reference
    point per track (``track_target_*``) and one HiRO-ACE prediction point
    per ensemble sample (``wind_predicted``/``ike_predicted``, already one
    entry per (track, sample) with no track-averaging), so ensemble spread
    is visible directly, as in ``plot_scatter`` above."""
    ax.scatter(
        data["wind_predicted"],
        data["ike_predicted"],
        s=10,
        color=PERFECT_PRED_COLOR,
        alpha=0.4,
        edgecolors="none",
        label="Predicted (ensemble members)",
    )
    ax.scatter(
        data["track_target_wind"],
        data["track_target_ike"],
        marker='s',
        s=16,
        color=REFERENCE_COLOR,
        alpha=1.,
        edgecolors="none",
        label="X-SHiELD (reference)",
    )
    ax.set_yscale('log')
    ax.set_xlabel("Peak 10m wind speed (m/s)")
    ax.set_ylabel("Integrated kinetic energy (TJ)")
    ax.set_title(f"IKE vs. peak wind speed\n(N={n_tracks} tracks)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


def _plot_ike_bias_panel(ax, data: dict, ensemble_label: str) -> None:
    """The IKE predicted-vs-target scatter -- the main figure's panel a)."""
    plot_scatter(
        ax, data["ike_target"], data["ike_predicted"], data["n_tracks"],
        "Integrated kinetic energy", "TJ", ensemble_label,
    )


def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def make_main_figure(data: dict, out_path: Path) -> None:
    """Main-text figure: the two predicted-vs-target bias panels, labelled
    a) IKE and b) peak 10m wind speed."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.5 * 1.25, 3.0 * 1.25),
        gridspec_kw={"wspace": 0.05},
    )
    ensemble_label = _ensemble_size_label(data["track_n_samples"])
    _plot_ike_bias_panel(axes[0], data, ensemble_label)
    plot_scatter(
        axes[1], data["wind_target"], data["wind_predicted"], data["n_tracks"],
        "Peak 10m wind speed", "m/s", ensemble_label, show_legend=False,
    )
    for ax, label in zip(axes, PANEL_LABELS):
        ax.set_title(label, loc="left", fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold")
    _save(fig, out_path)


def make_supplemental_figure(data: dict, out_path: Path) -> None:
    """Supplemental figure: the IKE-vs-peak-wind scatter on its own, so it
    carries no panel label (there is nothing to label it against)."""
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4))
    plot_ike_vs_wind(ax, data, data["n_tracks"])
    _save(fig, out_path)


def bias_stats(target: np.ndarray, predicted: np.ndarray) -> dict:
    """Median absolute (``predicted - target``) and relative
    (``(predicted - target) / target``) bias, plus absolute RMSE and relative
    RMSE, over a set of tracks. A target of exactly 0 makes the relative
    ratio undefined -- kept as NaN (not dropped from ``target``/``predicted``)
    and skipped via ``nanmedian``/``nanmean``, so only the relative statistics
    lose that track, not the absolute ones.

    The two RMSEs are *means* of squared errors, unlike the medians above, so
    they are deliberately outlier-sensitive -- ``rel_rmse`` especially: it is
    ``sqrt(mean(((predicted - target) / target)**2))``, so a track whose
    target is small but nonzero (an IKE right at the edge of the 18 m/s band)
    can dominate it even though its absolute error is tiny. Read it alongside
    ``median_rel_bias`` (robust) rather than instead of it."""
    if len(target) == 0:
        return {
            "n": 0, "n_rel": 0,
            "median_abs_bias": float("nan"), "median_rel_bias": float("nan"),
            "rmse": float("nan"), "rel_rmse": float("nan"),
        }
    abs_bias = predicted - target
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_bias = np.where(target != 0, (predicted - target) / target, np.nan)
    n_rel = int(np.sum(~np.isnan(rel_bias)))
    return {
        "n": len(target),
        "n_rel": n_rel,
        "median_abs_bias": float(np.median(abs_bias)),
        "median_rel_bias": float(np.nanmedian(rel_bias)),
        "rmse": float(np.sqrt(np.mean(abs_bias**2))),
        "rel_rmse": float(np.sqrt(np.nanmean(rel_bias**2))) if n_rel else float("nan"),
    }


def _format_table(header: list, rows: list) -> str:
    """A plain-text table: ``header`` and each of ``rows`` are lists of
    already-formatted cell strings, first column left-aligned (the group
    label), the rest right-aligned to their column's widest cell."""
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    lines = []
    for row in [header] + rows:
        cells = [str(row[0]).ljust(widths[0])] + [
            str(cell).rjust(w) for cell, w in zip(row[1:], widths[1:])
        ]
        lines.append("  ".join(cells))
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def print_bias_summary(data: dict) -> None:
    header = [
        "Group", "N",
        "Wind abs bias (m/s)", "Wind rel bias (%)", "Wind RMSE (m/s)", "Wind rel RMSE (%)",
        "IKE N (nonzero)",
        "IKE abs bias (TJ)", "IKE rel bias (%)", "IKE RMSE (TJ)", "IKE rel RMSE (%)",
    ]
    rows = []
    for label, min_wind in WIND_GROUPS:
        mask = (
            np.ones_like(data["track_target_wind"], dtype=bool)
            if min_wind is None
            else data["track_target_wind"] > min_wind
        )
        wind_stats = bias_stats(data["track_target_wind"][mask], data["track_pred_wind"][mask])
        ike_stats = bias_stats(data["track_target_ike"][mask], data["track_pred_ike"][mask])
        rows.append([
            label,
            wind_stats["n"],
            f"{wind_stats['median_abs_bias']:.3f}",
            f"{100 * wind_stats['median_rel_bias']:.2f}",
            f"{wind_stats['rmse']:.3f}",
            f"{100 * wind_stats['rel_rmse']:.2f}",
            ike_stats["n_rel"],
            f"{ike_stats['median_abs_bias']:.3f}",
            f"{100 * ike_stats['median_rel_bias']:.2f}",
            f"{ike_stats['rmse']:.3f}",
            f"{100 * ike_stats['rel_rmse']:.2f}",
        ])
    print(_format_table(header, rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", default=DEFAULT_ROOT,
        help=f"Perfect-prediction max-snapshot directory (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output path for the main-text figure -- the IKE and peak-wind bias "
        f"scatters (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--supplemental-out", default=DEFAULT_SUPPLEMENTAL_OUT,
        help=f"Output path for the supplemental IKE-vs-wind figure "
        f"(default: {DEFAULT_SUPPLEMENTAL_OUT}).",
    )
    parser.add_argument(
        "--ike-box-deg", type=float, default=DEFAULT_IKE_BOX_DEG,
        help=f"Width of the square storm-centered IKE domain, in degrees (default: {DEFAULT_IKE_BOX_DEG:g}).",
    )
    parser.add_argument(
        "--ike-min-wind", type=float, default=DEFAULT_IKE_MIN_WIND,
        help=f"Only cells with sustained wind above this, in m/s, contribute to IKE "
        f"(default: {DEFAULT_IKE_MIN_WIND:g}). Pass 0 for every cell.",
    )
    parser.add_argument("--ike-max-wind", type=float, default=None, help="Optional upper bound on the IKE wind band.")
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Worker processes for reading/reducing tracks in parallel (default: {DEFAULT_MAX_WORKERS}).",
    )
    args = parser.parse_args()

    ike = IkeSpec(
        box_deg=args.ike_box_deg,
        min_wind=args.ike_min_wind if args.ike_min_wind else None,
        max_wind=args.ike_max_wind,
    )

    data = collect_bias(Path(args.root), ike, args.max_workers, year=YEAR)
    make_main_figure(data, Path(args.out))
    make_supplemental_figure(data, Path(args.supplemental_out))
    print_bias_summary(data)


if __name__ == "__main__":
    main()
