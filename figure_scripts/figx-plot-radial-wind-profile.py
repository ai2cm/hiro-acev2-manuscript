"""
Two side-by-side-panel composite figures of TC radial 10m wind structure, one
per comparison:

  * ``radial_wind_profile_perfect_pred.png`` -- 2023 holdout year: "Perfect
    prediction" vs its colocated "X-SHiELD reference" (the ``*_target``
    fields stored alongside each perfect-prediction max-intensity snapshot in
    --perfect-pred-dir), restricted to --year 2023. The free-running HiRO-ACE
    series is dropped here -- this figure is about how well a perfect
    coarse-resolution forecast downscales, not about the free-running
    rollout.

  * ``radial_wind_profile_hiro_ace.png`` -- full set: the same
    --perfect-pred-dir's colocated "X-SHiELD reference", this time over *all*
    its tracks (no --year restriction), vs "HiRO-ACE" composited from its own
    free-running rollout tracks (--hiro-ace-dir), also over all years. The
    two composites are *not* time/box-matched to each other -- HiRO-ACE's
    rollout dates don't correspond to real X-SHiELD storms -- so this figure
    compares two independent storm populations' mean radial structure, not
    paired snapshots.

Each figure's left panel is the composite azimuthally-averaged wind speed vs.
radius (one line per dataset, pooling every sample together before averaging,
binned into radius bins). The right panel breaks that same pooled average
back out into one point per individual sample: its own raw radius of maximum
wind -- the straight-line distance from its snapshot's PRMSL-minimum centroid
to its single strongest grid cell, read directly off the native grid rather
than off the left panel's radius bins -- vs. that grid cell's wind speed,
plus a per-dataset 2D kernel density contour. So the right panel shows the
unbinned sample-to-sample scatter (and its density) that the left panel's
binned average smooths over.

For the two predicted datasets (Perfect prediction, HiRO-ACE), each snapshot
holds an ensemble of members; every member is treated as its own independent
sample in both panels -- never averaged into a single ensemble-mean field
first -- so a snapshot with 8 members contributes 8 points to the right panel
and 8 histogram draws to the left (all relative to the same ensemble-mean
PRMSL centroid, since the storm's position is one per snapshot even though
its intensity/structure varies by member). The X-SHiELD reference has no
ensemble dimension, so it contributes exactly one sample per snapshot either
way.

Both directories are max-snapshot runs named ``tc_<idx>_<YYYYMMDDTHH>.nc``;
``find_track_files``'s filename regex already excludes any ``_ens32``-suffixed
variant (e.g. --perfect-pred-dir's one-off ``tc_743_..._ens32.nc``), since
that suffix keeps it from matching the plain ``tc_<idx>_<YYYYMMDDTHH>`` stem.

Field/coordinate lookup, centroid-finding, radius/azimuth binning, and
max-snapshot file discovery are shared with this directory's other
max-snapshot scripts via ``tc_snapshot_common.py`` (this script previously
reimplemented them locally; see that module's docstring for what else uses
each piece). The ensemble-independent-sample composite/scatter logic and the
density-contour plotting below are unique to this script and stay local.

Each series' composite histogram and scatter points (all either panel draws
for it) are cached under ``plot_data_cache/figx-plot-radial-wind-profile/``
(see ``plot_cache.py``), one file per series, keyed by --year and
--max-radius-deg; later runs draw both figures from the cache without opening
any snapshot. Pass ``--refresh`` to recompute (also needed after pointing
--perfect-pred-dir/--hiro-ace-dir somewhere else, which the cache names don't
track).

Usage:
    python figx-plot-radial-wind-profile.py
    python figx-plot-radial-wind-profile.py --out-dir main_figures
    python figx-plot-radial-wind-profile.py --refresh
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from plot_cache import MAIN_FIGURES_DIR, cached_arrays  # noqa: E402
from series_colors import HIRO_ACE_COLOR, PERFECT_PRED_COLOR, REFERENCE_COLOR  # noqa: E402
from tc_snapshot_common import (  # noqa: E402
    _azimuthal_mean,
    _coord_names,
    bin_edges,
    compute_centroid,
    filter_by_year,
    find_track_files,
    prmsl_field,
    radius_azimuth,
    target_prmsl_field,
    target_wind_speed_field,
    wind_speed_field,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bin counts/radius for this script's composites -- deliberately different
# from plot_radial_wind_profile_max_snapshot.py's own values (20/36/1.0), so
# these are kept as local constants rather than centralized in
# tc_snapshot_common.py.
DEFAULT_MAX_RADIUS_DEG = 1.75
N_R_BINS = 25
N_THETA_BINS = 36

DEFAULT_PERFECT_PRED_DIR = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot"
)
DEFAULT_HIRO_ACE_DIR = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-19-hiro-ace2s-rollout-all-tcs-max-snapshot"
)
DEFAULT_YEAR = 2023
DEFAULT_OUT_DIR = MAIN_FIGURES_DIR
CACHE_SUBDIR = Path(__file__).stem
PERFECT_PRED_FILENAME = "radial_wind_profile_perfect_pred.png"
HIRO_ACE_FILENAME = "radial_wind_profile_hiro_ace.png"

PERFECT_PRED_LABEL = "Perfect prediction"
HIRO_ACE_LABEL = "HiRO-ACEv2"
REFERENCE_LABEL = "X-SHiELD reference"
SERIES_COLOR = {
    PERFECT_PRED_LABEL: PERFECT_PRED_COLOR,
    HIRO_ACE_LABEL: HIRO_ACE_COLOR,
    REFERENCE_LABEL: REFERENCE_COLOR,
}

# Panel labels, in the figures' left-to-right order; each figure is
# untitled -- which comparison it shows is carried by its filename and its
# per-series legend, and its caption in the paper.
PANEL_LABELS = ("a)", "b)")
PANEL_LABEL_FONTSIZE = 12

SCATTER_ALPHA = 0.4
SCATTER_SIZE = 6
# HiRO-ACE has far more samples than the other series (8 ensemble members x
# many more tracks), so its scatter dots dominate/overplot the panel; drawn
# at a lower opacity (all points still plotted) for legibility.
HIRO_ACE_SCATTER_ALPHA = SCATTER_ALPHA / 3

DENSITY_GRID_N = 100j
DENSITY_LEVELS = 4
DENSITY_PAD_FRAC = 0.05
# Bands below this fraction of a dataset's own peak density are left
# unfilled (fully transparent), so two datasets' low-density tails don't
# wash the whole panel in a faint, indistinguishable tint -- only each
# dataset's actual core shows up, in its own color.
DENSITY_MIN_FRAC = 0.1
DENSITY_MAX_ALPHA = 0.75
DENSITY_LINEWIDTH = 1.
# Exponent of Scott's rule for a 2D KDE: bandwidth factor = n ** (-1 / (d + 4)).
SCOTT_EXPONENT_2D = -1 / 6


# --------------------------------------------------------------------------- #
# Scatter-point helpers unique to this script (the right panel's unbinned
# radius/peak per sample, read straight off the native grid rather than off
# the left panel's radius bins).
# --------------------------------------------------------------------------- #
def _raw_max_wind_point(r: np.ndarray, speed_vals: np.ndarray, max_radius_km: float) -> tuple | None:
    """The (radius_km, wind_speed) of the single strongest finite grid cell
    within ``max_radius_km``, read straight off the native grid -- not binned
    into the left panel's radius bins."""
    mask = np.isfinite(speed_vals) & (r <= max_radius_km)
    if not np.any(mask):
        return None
    masked_speed = np.where(mask, speed_vals, -np.inf)
    idx = np.unravel_index(np.argmax(masked_speed), masked_speed.shape)
    return float(r[idx]), float(speed_vals[idx])


def _collect_rmw_peak(points: list) -> tuple:
    if not points:
        return np.array([]), np.array([])
    rmws, peaks = zip(*points)
    return np.array(rmws), np.array(peaks)


# --------------------------------------------------------------------------- #
# Per-snapshot processing: left-panel composite histogram + right-panel raw
# scatter point(s), both from a single file read.
# --------------------------------------------------------------------------- #
def _predicted_all_members(path: Path, max_radius_deg: float) -> tuple | None:
    """Every ensemble member's own contribution for one predicted-field
    snapshot, relative to the ensemble-mean PRMSL-minimum centroid (the
    storm's position is one per snapshot even though its intensity/structure
    varies by member): the member's wind speed binned into the shared
    (azimuth, radius) histogram for the left panel's composite, and the
    member's own raw (unbinned) radius/peak of its single strongest grid cell
    for the right panel's scatter. Members are never averaged together
    first.

    Returns ``(sum_grid, count_grid, n_members, points)`` where ``points`` is
    a list of ``(radius_km, peak_wind)``, one per member with a finite peak,
    or ``None`` if the file lacks PRMSL or wind.
    """
    theta_edges, r_edges = bin_edges(max_radius_deg, N_THETA_BINS, N_R_BINS)
    with xr.open_dataset(path) as ds:
        prmsl = prmsl_field(ds)
        speed = wind_speed_field(ds)
        if prmsl is None or speed is None:
            return None
        lat_name, lon_name = _coord_names(ds)
        prmsl_mean = prmsl.mean("sample") if "sample" in prmsl.dims else prmsl
        lat0, lon0 = compute_centroid(prmsl_mean.load(), lat_name, lon_name)
        r, theta = radius_azimuth(ds[lat_name].values, ds[lon_name].values, lat0, lon0)
        speed = speed.load()
    members = (
        [speed.isel(sample=i).values for i in range(speed.sizes["sample"])]
        if "sample" in speed.dims
        else [speed.values]
    )

    sum_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    count_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    points = []
    for member_speed in members:
        speed_vals = np.asarray(member_speed, dtype=float)
        mask = np.isfinite(speed_vals) & (r <= r_edges[-1])
        if np.any(mask):
            s, _, _ = np.histogram2d(
                theta[mask], r[mask], bins=[theta_edges, r_edges], weights=speed_vals[mask]
            )
            c, _, _ = np.histogram2d(theta[mask], r[mask], bins=[theta_edges, r_edges])
            sum_grid += s
            count_grid += c
        point = _raw_max_wind_point(r, speed_vals, r_edges[-1])
        if point is not None:
            points.append(point)
    return sum_grid, count_grid, len(members), points


def predicted_composite_and_scatter(name: str, paths: list, max_radius_deg: float, max_workers: int) -> tuple:
    """Composite (left-panel) entry and scatter (right-panel) entry for a
    predicted-field dataset, treating every ensemble member as its own independent sample
    (never averaged into a single ensemble-mean field first) -- both derived
    from a single per-snapshot file read via ``_predicted_all_members``."""
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_predicted_all_members, paths, [max_radius_deg] * len(paths)))

    sum_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    count_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    n_snapshots = 0
    n_samples = 0
    all_points = []
    for result in results:
        if result is None:
            continue
        s, c, n_members, points = result
        sum_grid += s
        count_grid += c
        n_snapshots += 1
        n_samples += n_members
        all_points.extend(points)

    composite_entry = (name, sum_grid, count_grid, n_snapshots, n_samples)
    scatter_entry = (name, *_collect_rmw_peak(all_points))
    return composite_entry, scatter_entry


def _target_snapshot(path: Path, max_radius_deg: float) -> tuple | None:
    """One X-SHiELD reference (``_target``) snapshot's contribution: its wind
    speed binned into the (azimuth, radius) histogram for the left panel's
    composite, and the raw (unbinned) radius/peak of its single strongest grid
    cell for the right panel's scatter. No ensemble dimension here, so exactly
    one sample.

    Returns ``(sum_grid, count_grid, point)``, or ``None`` if the file lacks
    target PRMSL or wind.
    """
    theta_edges, r_edges = bin_edges(max_radius_deg, N_THETA_BINS, N_R_BINS)
    with xr.open_dataset(path) as ds:
        prmsl = target_prmsl_field(ds)
        speed = target_wind_speed_field(ds)
        if prmsl is None or speed is None:
            return None
        lat_name, lon_name = _coord_names(ds)
        prmsl = prmsl.load()
        speed_vals = np.asarray(speed.load().values, dtype=float)
        lat0, lon0 = compute_centroid(prmsl, lat_name, lon_name)
        r, theta = radius_azimuth(ds[lat_name].values, ds[lon_name].values, lat0, lon0)

    mask = np.isfinite(speed_vals) & (r <= r_edges[-1])
    sum_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    count_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    if np.any(mask):
        sum_grid, _, _ = np.histogram2d(
            theta[mask], r[mask], bins=[theta_edges, r_edges], weights=speed_vals[mask]
        )
        count_grid, _, _ = np.histogram2d(theta[mask], r[mask], bins=[theta_edges, r_edges])
    point = _raw_max_wind_point(r, speed_vals, r_edges[-1])
    return sum_grid, count_grid, point


def target_composite_and_scatter(name: str, paths: list, max_radius_deg: float, max_workers: int) -> tuple:
    """Composite (left-panel) entry and scatter (right-panel) entry for the
    X-SHiELD reference dataset, both derived from a single per-snapshot file read via
    ``_target_snapshot``."""
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_target_snapshot, paths, [max_radius_deg] * len(paths)))

    sum_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    count_grid = np.zeros((N_THETA_BINS, N_R_BINS))
    n_snapshots = 0
    points = []
    for result in results:
        if result is None:
            continue
        s, c, point = result
        sum_grid += s
        count_grid += c
        n_snapshots += 1
        if point is not None:
            points.append(point)

    composite_entry = (name, sum_grid, count_grid, n_snapshots, n_snapshots)
    scatter_entry = (name, *_collect_rmw_peak(points))
    return composite_entry, scatter_entry


# --------------------------------------------------------------------------- #
# Plot-data cache: one npz per series holding its composite entry's grids/counts
# and its scatter entry's points, so both panels can be drawn without the
# snapshot directories.
# --------------------------------------------------------------------------- #
def load_series(cache_key: str, max_radius_deg: float, compute, refresh: bool) -> tuple:
    """``(composite_entry, scatter_entry)`` for one series, as returned by
    ``compute()`` (one of the ``*_composite_and_scatter`` functions above), from
    ``plot_data_cache/<CACHE_SUBDIR>/<cache_key>_r<max_radius_deg>.npz`` if present."""

    def compute_arrays() -> dict:
        (name, sum_grid, count_grid, n_snapshots, n_samples), (_, rmw, peak) = compute()
        return {
            "name": np.array(name), "sum_grid": sum_grid, "count_grid": count_grid,
            "n_snapshots": np.array(n_snapshots), "n_samples": np.array(n_samples),
            "rmw": rmw, "peak": peak,
        }

    arrays = cached_arrays(f"{CACHE_SUBDIR}/{cache_key}_r{max_radius_deg:g}.npz", compute_arrays, refresh)
    name = str(arrays["name"])
    composite_entry = (
        name, arrays["sum_grid"], arrays["count_grid"],
        int(arrays["n_snapshots"]), int(arrays["n_samples"]),
    )
    scatter_entry = (name, arrays["rmw"], arrays["peak"])
    return composite_entry, scatter_entry


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_composite_panel(ax, results: list, r_centers: np.ndarray) -> None:
    """Left panel: one line per (name, sum_grid, count_grid, n_snapshots,
    n_samples) entry in ``results`` -- the pooled composite azimuthal-mean
    wind speed vs. radius -- colored by ``SERIES_COLOR``."""
    for name, sum_grid, count_grid, n_snapshots, n_samples in results:
        profile = _azimuthal_mean(sum_grid, count_grid)
        ax.plot(
            r_centers,
            profile,
            marker="s" if name == REFERENCE_LABEL else "o",
            markersize=3,
            markerfacecolor="none" if name == REFERENCE_LABEL else None,
            color=SERIES_COLOR.get(name),
            label=f"{name}\n(N={n_snapshots} snapshots, {n_samples} samples)",
        )
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc="lower center", fontsize=7)
    for line in legend.get_lines():
        line.set_linestyle("None")


def _add_density_contours(ax, series: list, bw_factor: float) -> None:
    """Draw one 2D kernel-density contour per dataset with enough points,
    evaluated on a grid shared across ``series`` (padded from their combined
    data range) so densities from different datasets in the same panel are
    on the same axes. Each dataset is normalized to its *own* peak density
    (not compared in absolute terms across datasets, matching the left panel's
    per-dataset composite lines) and rendered as unfilled color lines only --
    no shading between levels -- so overlapping datasets blend by color
    instead of one dataset's fill occluding another's.

    Every dataset's KDE uses the same ``bw_factor`` (see
    ``_shared_bandwidth_factor``) rather than scipy's per-dataset Scott's rule,
    which would shrink the kernel for whichever dataset has more points and
    make its contours look tighter purely from sample count."""
    all_rmw = np.concatenate([rmw for _, rmw, _ in series if len(rmw)])
    all_peak = np.concatenate([peak for _, _, peak in series if len(peak)])
    if all_rmw.size == 0:
        return
    pad_x = DENSITY_PAD_FRAC * (all_rmw.max() - all_rmw.min())
    pad_y = DENSITY_PAD_FRAC * (all_peak.max() - all_peak.min())
    # Radius is bounded below by 0, but the Gaussian kernels sitting on the
    # many near-zero-RMW samples spill across that boundary, so the padded
    # grid would otherwise draw contours at physically impossible negative
    # radii. Clamping the grid's left edge at 0 stops the fill there (it does
    # not correct the KDE's boundary bias -- density just inside 0 is still
    # under-estimated by the mass that leaked out).
    x_lo = max(0.0, all_rmw.min() - pad_x)
    xx, yy = np.mgrid[
        x_lo : all_rmw.max() + pad_x : DENSITY_GRID_N,
        all_peak.min() - pad_y : all_peak.max() + pad_y : DENSITY_GRID_N,
    ]
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    for name, rmw, peak in series:
        if len(rmw) < 5:
            continue
        kde = gaussian_kde(np.vstack([rmw, peak]), bw_method=bw_factor)
        density = kde(grid_points).reshape(xx.shape)
        levels = np.linspace(DENSITY_MIN_FRAC * density.max(), density.max(), DENSITY_LEVELS)
        color = SERIES_COLOR.get(name)
        ax.contour(xx, yy, density, levels=levels, colors=[color], linewidths=DENSITY_LINEWIDTH, alpha=DENSITY_MAX_ALPHA)


def _shared_bandwidth_factor(composite: list) -> float:
    """One KDE bandwidth factor for every dataset in a panel: Scott's rule
    evaluated at the smallest dataset's *snapshot* count (``n_snapshots`` in
    each ``composite`` entry). Snapshots, not points, because an ensemble's
    members within one snapshot are the same storm and far from independent,
    so counting each member would overstate the effective sample size."""
    n_eff = min(n_snapshots for _, _, _, n_snapshots, _ in composite)
    return n_eff**SCOTT_EXPONENT_2D


def _plot_scatter_panel(ax, series: list, bw_factor: float) -> None:
    """Right panel: one dataset-colored density contour (background) plus a small
    scatter on top (radius of maximum wind vs. peak wind, one point per sample),
    per ``(name, rmw_array, peak_array)`` entry in ``series``. X-SHiELD reference
    points are drawn fully opaque; the other series stay semi-transparent so dense
    overlapping points still read as a cloud."""
    _add_density_contours(ax, series, bw_factor)
    for name, rmw, peak in series:
        color = SERIES_COLOR.get(name)
        if name == REFERENCE_LABEL:
            alpha, marker, facecolor, edgecolor = 1.0, "s", "none", color
        elif name == HIRO_ACE_LABEL:
            alpha, marker, facecolor, edgecolor = HIRO_ACE_SCATTER_ALPHA, "o", color, "none"
        else:
            alpha, marker, facecolor, edgecolor = SCATTER_ALPHA, "o", color, "none"
        ax.scatter(
            rmw, peak, alpha=alpha, s=SCATTER_SIZE, facecolor=facecolor,
            edgecolor=edgecolor, marker=marker,
        )
    ax.set_xlabel("Radius of maximum wind (km)")
    ax.set_yscale('log')
    ax.set_ylim(10, 110)
    ax.set_yticks([10, 20, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=0.3)


def plot_comparison(
    composite: list,
    scatter: list,
    max_radius_deg: float,
    out_path: Path,
) -> None:
    """One comparison's two panels side by side, labelled a) and b): the
    binned composite profile on the left, the per-sample RMW/peak scatter
    (and its density) on the right."""
    _, r_edges = bin_edges(max_radius_deg, N_THETA_BINS, N_R_BINS)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    fig, (ax_composite, ax_scatter) = plt.subplots(1, 2, figsize=(8, 3.5))

    _plot_composite_panel(ax_composite, composite, r_centers)
    ax_composite.set_xlabel("Radius (km)")
    ax_composite.set_ylabel("Mean 10m wind speed (m/s)")

    _plot_scatter_panel(ax_scatter, scatter, _shared_bandwidth_factor(composite))
    ax_scatter.set_ylabel("Peak 10m wind speed (m/s)")

    for ax, label in zip((ax_composite, ax_scatter), PANEL_LABELS):
        ax.set_title(label, loc="left", fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--perfect-pred-dir",
        default=DEFAULT_PERFECT_PRED_DIR,
        help="Perfect-prediction max-snapshot directory, also supplying the X-SHiELD "
        f"reference (its '_target' fields) for both figures (default: {DEFAULT_PERFECT_PRED_DIR}).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"{PERFECT_PRED_FILENAME}: restrict --perfect-pred-dir (and its X-SHiELD "
        f"reference) to this holdout year (default: {DEFAULT_YEAR}). "
        f"{HIRO_ACE_FILENAME} always uses the full set.",
    )
    parser.add_argument(
        "--hiro-ace-dir",
        default=DEFAULT_HIRO_ACE_DIR,
        help=f"{HIRO_ACE_FILENAME}: HiRO-ACE free-running rollout max-snapshot directory, "
        f"over all years (default: {DEFAULT_HIRO_ACE_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write {PERFECT_PRED_FILENAME} and {HIRO_ACE_FILENAME} into "
        f"(default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--max-radius-deg",
        type=float,
        default=DEFAULT_MAX_RADIUS_DEG,
        help=f"Radial extent of the composite, in degrees (default: {DEFAULT_MAX_RADIUS_DEG:g}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Worker processes for reading/binning snapshots in parallel (default: 16).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute every series' plot data and overwrite its plot_data_cache/ file.",
    )
    args = parser.parse_args()

    # File discovery only runs when some series' cache is missing (or --refresh).
    perfect_pred_dir = Path(args.perfect_pred_dir)

    def all_perfect_pred_paths() -> list:
        return list(find_track_files(perfect_pred_dir).values())

    def perfect_pred_2023_paths() -> list:
        all_perfect_pred_files = find_track_files(perfect_pred_dir)
        perfect_pred_2023_files = (
            filter_by_year(all_perfect_pred_files, args.year) if args.year else all_perfect_pred_files
        )
        return list(perfect_pred_2023_files.values())

    hiro_ace_dir = Path(args.hiro_ace_dir)

    def hiro_ace_paths() -> list:
        paths = list(find_track_files(hiro_ace_dir).values())
        logger.info("%s: %d track(s)", hiro_ace_dir, len(paths))
        return paths

    out_dir = Path(args.out_dir)
    year_key = args.year or "all"

    # Figure 1: 2023 holdout, perfect prediction vs. its colocated X-SHiELD reference.
    pp_2023_composite, pp_2023_scatter = load_series(
        f"perfect_pred_{year_key}", args.max_radius_deg,
        lambda: predicted_composite_and_scatter(
            PERFECT_PRED_LABEL, perfect_pred_2023_paths(), args.max_radius_deg, args.max_workers
        ),
        args.refresh,
    )
    ref_2023_composite, ref_2023_scatter = load_series(
        f"reference_{year_key}", args.max_radius_deg,
        lambda: target_composite_and_scatter(
            REFERENCE_LABEL, perfect_pred_2023_paths(), args.max_radius_deg, args.max_workers
        ),
        args.refresh,
    )
    plot_comparison(
        [pp_2023_composite, ref_2023_composite],
        [pp_2023_scatter, ref_2023_scatter],
        args.max_radius_deg,
        out_dir / PERFECT_PRED_FILENAME,
    )

    # Figure 2: full set, the same directory's X-SHiELD reference (unrestricted by
    # year) vs. HiRO-ACE's own free-running rollout tracks -- an independent,
    # unmatched population (see the module docstring).
    ref_full_composite, ref_full_scatter = load_series(
        "reference_all", args.max_radius_deg,
        lambda: target_composite_and_scatter(
            REFERENCE_LABEL, all_perfect_pred_paths(), args.max_radius_deg, args.max_workers
        ),
        args.refresh,
    )
    ace_composite, ace_scatter = load_series(
        "hiro_ace_all", args.max_radius_deg,
        lambda: predicted_composite_and_scatter(
            HIRO_ACE_LABEL, hiro_ace_paths(), args.max_radius_deg, args.max_workers
        ),
        args.refresh,
    )
    plot_comparison(
        [ref_full_composite, ace_composite],
        [ref_full_scatter, ace_scatter],
        args.max_radius_deg,
        out_dir / HIRO_ACE_FILENAME,
    )


if __name__ == "__main__":
    main()
