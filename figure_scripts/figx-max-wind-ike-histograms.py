"""
Histograms of per-track peak 10m wind speed and integrated kinetic energy
(IKE) for the HiRO-ACE v2 "max snapshot" runs, in the style of
``2026-07-16-tc-tracks-prod/plot_max_wind_histogram.py``.

Unlike that script's inputs (many snapshot files per track, spanning each
track's lifetime), each run here already stores exactly one file per track --
``tc_<idx>_<YYYYMMDDTHH>.nc``, pre-selected at that track's peak-intensity
time (see ``track_detection/downscale_highest_max_wind.py``). So there is no
per-track "max over lifetime" step: a track's peak value is simply the max
over its one snapshot's grid (and, for the ensemble series, computed
per sample).

Two series are plotted:

  * "HiRO-ACE" -- the ``*_predicted`` fields of ``--hiro-ace-dir`` (default:
    the 2026-08-19 ACE2S rollout + downscaling run).
  * "X-SHiELD reference" -- the ``*_target`` fields stored alongside the
    ``--perfect-pred-dir`` snapshots (default: the 2026-08-10 perfect-pred
    run), i.e. the true 3km X-SHiELD state at the same time and lat/lon box as
    each perfect-prediction snapshot, matched to HiRO-ACE's tracks by
    ``tc_<idx>`` (the two runs pick each track's peak time independently, so
    timestamps differ). Unlike the HiRO-ACE series this has no ensemble
    dimension. Pass --no-reference to skip it.

``--perfect-pred-dir``'s own ``*_predicted`` fields (the downscaling model's
best-case/ceiling behaviour, given the true coarse X-SHiELD state as input)
are not plotted, only its ``_target`` fields.

The predicted and IKE fields reuse the field-detection helpers shared with
this directory's other max-snapshot scripts (``wind_speed_field``,
``target_wind_speed_field``, ...; see ``tc_snapshot_common.py``) so the
figure sets stay physically consistent. The IKE definition itself
(``IkeSpec``, ``regional_reduce``, ``integrated_kinetic_energy``) and the
histogram-binning helpers (``bins_covering``, ``peaks_to_values_weights``,
...) are used only by this script among this directory's max-snapshot
scripts, so -- rather than importing them from
``2026-07-16-tc-tracks-prod/plot_event_timeseries.py`` and
``plot_max_wind_histogram.py`` -- they are copied in below, unmodified.
Ensemble members are weighted 1/N per track (see ``peaks_to_values_weights``),
so a track's samples never outweigh a track with a smaller ensemble.

A second figure per quantity (``--basin-out``/``--ike-basin-out``) breaks the
same histogram down into one subplot per basin. Unlike
``select_top_tcs.py``'s ``<basin_slug>_<rank>`` event names, these tracks are
named ``tc_<idx>`` with no basin encoded in the name, so each track's basin is
instead derived from its own snapshot's mean (lon, lat) -- the patches are
storm-centered, so that mean position is the storm's location -- via the same
lon/lat boxes as ``select_top_tcs.assign_basin`` (``select_top_tcs.py``'s
``BASINS`` list is a small hardcoded set of boxes, inlined below as
``BASINS`` rather than imported). A track outside every basin box (the
omitted "Australian region") is dropped from the by-basin figures only. Pass
--no-basin-figure to skip both by-basin figures.

Every panel plots a weighted track count (``density=False``), not a
normalized probability density -- each sample is weighted 1/n_samples so an
ensemble's samples never outweigh a track with fewer of them (see
``peaks_to_values_weights``), but the two series' bar heights are otherwise
directly comparable only when their track counts are similar (e.g. a track
present in one directory but not the other skews the comparison).

All figures use a log-scaled density axis, except the by-basin max-wind
figure (``--basin-out``), which stays linear -- its per-basin y-ranges are
already tight (see ``sharey=False`` in ``plot_histogram_by_basin``), so a log
scale would add little while making the dominant low-wind bin harder to read
at a glance.

Each series' per-track peaks (wind and IKE per sample, plus the track's basin)
are cached in
``plot_data_cache/figx-max-wind-ike-histograms/<run dir>_<predicted|target>_<IKE spec>.npz``,
keyed on the run directory and the IKE options, so the figures redraw without
opening the snapshot files. Pass --refresh to recompute them.

Usage:
    python plot_max_wind_ike_histograms.py
    python plot_max_wind_ike_histograms.py --out max_wind_histogram.png --ike-out ike_histogram.png
    python plot_max_wind_ike_histograms.py --no-reference --max-workers 32
    python plot_max_wind_ike_histograms.py --no-basin-figure
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from plot_cache import MAIN_FIGURES_DIR, SUPPLEMENTS_DIR, cached_arrays  # noqa: E402
from series_colors import HIRO_ACE_COLOR, PERFECT_PRED_COLOR, REFERENCE_COLOR  # noqa: E402
from tc_snapshot_common import (  # noqa: E402
    _coord_names,
    find_track_files,
    target_wind_speed_field,
    wind_speed_field,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PERFECT_PRED_DIR = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot"
)
DEFAULT_HIRO_ACE_DIR = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-19-hiro-ace2s-rollout-all-tcs-max-snapshot"
)
DEFAULT_OUT = str(MAIN_FIGURES_DIR / "max_wind_histogram.png")
DEFAULT_BASIN_OUT = str(SUPPLEMENTS_DIR / "max_wind_histogram_by_basin.png")
DEFAULT_IKE_OUT = str(MAIN_FIGURES_DIR / "ike_histogram.png")
DEFAULT_IKE_BASIN_OUT = str(SUPPLEMENTS_DIR / "ike_histogram_by_basin.png")
CACHE_SUBDIR = Path(__file__).stem

PERFECT_PRED_LABEL = "Perfect prediction"
HIRO_ACE_LABEL = "HiRO-ACEv2"
REFERENCE_LABEL = "X-SHiELD reference"

# Assigned in the same (perfect-pred, HiRO-ACE, reference) order as
# plot_max_wind_histogram.py's default series_colors, so a series keeps its
# slot even when perfect-prediction is left unplotted.
SERIES_COLORS = (PERFECT_PRED_COLOR, HIRO_ACE_COLOR, REFERENCE_COLOR)


def series_colors(names: list) -> dict:
    """Map series label -> colour, assigned in order over the full series list
    (see ``SERIES_COLORS``)."""
    return {name: SERIES_COLORS[i % len(SERIES_COLORS)] for i, name in enumerate(names)}


# --------------------------------------------------------------------------- #
# Integrated kinetic energy (Powell and Reinhold 2007), copied unmodified from
# 2026-07-16-tc-tracks-prod/plot_event_timeseries.py -- this script is the
# only one of this directory's max-snapshot scripts that needs IKE, so it is
# kept local here rather than in tc_snapshot_common.py.
# --------------------------------------------------------------------------- #
_EARTH_RADIUS_M = 6.371e6

DENSITY_OF_AIR = 1.0  # kg m^-3
# Depth of the surface layer the 10m wind field is taken to represent. Powell
# and Reinhold (2007) integrate the surface wind field over a 1 m depth, which
# is what puts IKE in the conventional tens-to-hundreds-of-TJ range. Held fixed
# (rather than tied to the grid spacing) so the value is a property of the
# storm, not of the resolution -- without that, prediction and reference would
# not be comparable on grids of different spacing.
IKE_LAYER_DEPTH_M = 1.0
_JOULES_PER_TJ = 1.0e12
# Powell and Reinhold take their volume elements "over a storm-centered 8 deg
# latitude domain", so that is the domain width here too (comfortably inside the
# ~17 deg storm-following patches, so the box always fits).
DEFAULT_IKE_BOX_DEG = 8.0
# ...and they integrate only over "portions of the storm domain volume (V)
# containing sustained surface wind speeds (U) within specific ranges" -- there
# is no unthresholded IKE in the paper. 18 m/s is tropical-storm force, giving
# their IKE_TS, the quantity they validate against storm surge and wave
# destructive potential and the one their published values (e.g. Camille at
# 124 TJ) refer to.
#
# The threshold is what makes IKE size-aware: it follows the storm's own wind
# footprint, so a sprawling storm contributes more area than a compact one, and
# the calm environment is excluded no matter how wide the domain is.
DEFAULT_IKE_MIN_WIND = 18.0
# Named wind bands from the paper, for labelling (m/s lower bound -> name).
_IKE_BAND_NAMES = {10.0: "IKE>10", 18.0: "IKE_TS", 33.0: "IKE_H"}


@dataclass(frozen=True)
class IkeSpec:
    """How IKE is integrated: the storm-centered domain and the wind-speed band.

    Mirrors Powell and Reinhold's parameterisation -- an 8 deg domain, and a band
    of sustained wind speeds that the integral is restricted to. ``min_wind=None``
    integrates every cell, which is *not* a quantity the paper defines and which
    the calm environment dominates; it exists only to reproduce older figures.
    """

    box_deg: float = DEFAULT_IKE_BOX_DEG
    min_wind: float | None = DEFAULT_IKE_MIN_WIND
    max_wind: float | None = None  # set for the paper's damage bands (e.g. 41-54)

    @property
    def name(self) -> str:
        """Short name, using the paper's own where one applies."""
        if self.min_wind is None:
            return "IKE"
        if self.max_wind is not None:
            return f"IKE{self.min_wind:g}-{self.max_wind:g}"
        return _IKE_BAND_NAMES.get(self.min_wind, f"IKE>{self.min_wind:g}")

    @property
    def band(self) -> str:
        """Human-readable wind band, e.g. 'winds > 18 m/s'."""
        if self.min_wind is None:
            return "all cells, no wind threshold"
        if self.max_wind is not None:
            return f"winds {self.min_wind:g}-{self.max_wind:g} m/s"
        return f"winds > {self.min_wind:g} m/s"

    def describe(self) -> str:
        return f"{self.name}: {self.band}, {self.box_deg:g}-deg storm-centered domain"


def _area_weights(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """Per-cell area (m^2) for a regular lat/lon grid, broadcast over the field.

    ``area = R^2 * cos(lat) * dlat * dlon`` (angles in radians). Used to weight
    a spatial mean so cells nearer the pole (smaller area) count less,
    comparable across grids of different resolution.
    """
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
    """Integrated kinetic energy (J) of a 10m wind speed field, c.f. Powell and
    Reinhold 2007: ``IKE = int 0.5 * rho * U^2 dV``.

    Only the lowest level is available, so the integral is the sum over the 2d
    field of ``0.5 * rho * U^2`` times each cell's volume -- its true spherical
    area (``R^2 cos(lat) dlat dlon``, via ``_area_weights``, so cells nearer the
    pole count less) times the ``IKE_LAYER_DEPTH_M`` surface layer depth.

    Following the paper, only cells whose sustained wind speed lies in
    ``[min_wind, max_wind)`` contribute; cells outside the band are dropped, so a
    field with no winds above ``min_wind`` integrates to exactly 0. Pass
    ``min_wind=None`` to integrate every cell (not a quantity the paper defines).

    Any leading dims (``sample``) are preserved; only lat/lon are reduced.
    """
    cell_area = _area_weights(speed, lat_name, lon_name)
    if min_wind is not None:
        speed = speed.where(speed > min_wind)
    if max_wind is not None:
        speed = speed.where(speed < max_wind)
    energy_density = 0.5 * DENSITY_OF_AIR * IKE_LAYER_DEPTH_M * speed**2
    # .sum skips the masked-out cells, so an empty band gives 0 rather than nan.
    return (energy_density * cell_area).sum([lat_name, lon_name])


def _patch_center(da: xr.DataArray, lat_name: str, lon_name: str) -> tuple:
    """Midpoint of a patch's lat/lon extent.

    Used as the storm center for IKE -- a more robust center than a PRMSL-
    minimum argmin, which on a patch holding no coherent low returns an
    arbitrary pixel (often on the boundary). It is also identical across
    ensemble members and between prediction and reference, so IKE differences
    reflect the wind field and never storm placement.
    """
    return tuple(
        float((da[name].values.min() + da[name].values.max()) / 2.0)
        for name in (lat_name, lon_name)
    )


def _centered_box(
    da: xr.DataArray, lat_name: str, lon_name: str, lat0: float, lon0: float, box_deg: float
) -> xr.DataArray:
    """The ``box_deg`` x ``box_deg`` lat/lon box centered on ``(lat0, lon0)``,
    clipped to whatever of it lies inside ``da`` (handles descending coords)."""
    half = box_deg / 2.0
    slices = {}
    for name, center in ((lat_name, lat0), (lon_name, lon0)):
        coord = da[name]
        descending = float(coord[0]) > float(coord[-1])
        lo, hi = center - half, center + half
        slices[name] = slice(hi, lo) if descending else slice(lo, hi)
    return da.sel(slices)


def _ike_box(da: xr.DataArray, lat_name: str, lon_name: str, box_deg: float) -> xr.DataArray:
    """The region IKE is integrated over: the central ``box_deg`` of the patch."""
    lat0, lon0 = _patch_center(da, lat_name, lon_name)
    return _centered_box(da, lat_name, lon_name, lat0, lon0, box_deg)


def regional_reduce(
    da: xr.DataArray,
    reduce: str,
    lat_name: str,
    lon_name: str,
    ike: "IkeSpec | None" = None,
) -> xr.DataArray:
    """Reduce a field over its lat/lon dims, preserving any leading dims (sample).

    ``max``/``min`` take the spatial extremum; ``ike_box`` integrates kinetic
    energy over ``ike``'s domain and wind band and returns TJ. (Only these two
    reduce kinds are needed here; ``plot_event_timeseries.py``'s
    ``area_mean`` kind is not used by this script.)
    """
    spatial = [lat_name, lon_name]
    if reduce == "max":
        return da.max(spatial)
    if reduce == "min":
        return da.min(spatial)
    if reduce == "ike_box":
        ike = ike or IkeSpec()
        box = _ike_box(da, lat_name, lon_name, ike.box_deg)
        # Reported in TJ, the conventional unit for IKE, so the panel/histogram
        # axes match the values Powell and Reinhold 2007 quote.
        joules = integrated_kinetic_energy(
            box, lat_name, lon_name, ike.min_wind, ike.max_wind
        )
        return joules / _JOULES_PER_TJ
    raise ValueError(f"Unknown reduce {reduce!r}")


# --------------------------------------------------------------------------- #
# Basin assignment. select_top_tcs.py's BASINS list is a small hardcoded set
# of (name, slug, lon_min, lon_max, lat_min, lat_max) boxes on the 0-360
# longitude convention; it is only needed by this script among this
# directory's max-snapshot scripts, so it is inlined here rather than
# imported.
# --------------------------------------------------------------------------- #
BASINS = [
    ("North Atlantic", "north_atlantic", 260.0, 360.0, 0.0, 60.0),
    ("NE Pacific", "ne_pacific", 180.0, 260.0, 0.0, 60.0),
    ("NW Pacific", "nw_pacific", 100.0, 180.0, 0.0, 60.0),
    ("N Indian Ocean", "n_indian", 30.0, 100.0, 0.0, 30.0),
    ("SW Indian Ocean", "sw_indian", 20.0, 90.0, -60.0, 0.0),
    ("S Pacific", "s_pacific", 160.0, 240.0, -60.0, 0.0),
]
BASIN_ORDER = [slug for _name, slug, *_ in BASINS]
BASIN_DISPLAY_NAME = {slug: name for name, slug, *_ in BASINS}


def assign_basin_slug(lon: float, lat: float) -> str | None:
    """Basin slug for a (lon, lat) point, using the same lon/lat boxes -- and
    first-match-wins order -- as ``select_top_tcs.assign_basin`` (which
    returns the display name instead of the slug)."""
    for _name, slug, lon_min, lon_max, lat_min, lat_max in BASINS:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return slug
    return None


def track_basin(ds: xr.Dataset) -> str | None:
    """Basin slug of a track's patch, from the mean (lon, lat) of its
    snapshot's domain -- the patches are storm-centered, so this mean
    position is the storm's own location."""
    lat_name, lon_name = _coord_names(ds)
    lon = float(ds[lon_name].values.mean())
    lat = float(ds[lat_name].values.mean())
    return assign_basin_slug(lon, lat)


def group_peaks_by_basin(peaks: dict, basin_by_idx: dict) -> dict:
    """Split ``{idx: peak_per_sample}`` into ``{basin_slug: {idx: peak}}``,
    dropping any track whose basin is unknown (outside every basin box)."""
    groups: dict = {}
    n_dropped = 0
    for idx, peak in peaks.items():
        basin = basin_by_idx.get(idx)
        if basin is None:
            n_dropped += 1
            continue
        groups.setdefault(basin, {})[idx] = peak
    if n_dropped:
        logger.info("Dropped %d track(s) with no matching basin from the by-basin figure", n_dropped)
    return groups


def _ordered_basins(basin_sets: list) -> list:
    """Union of basin slugs across ``basin_sets``, in canonical BASINS order,
    with any unrecognized slugs appended at the end."""
    all_basins = {basin for basins in basin_sets for basin in basins}
    return sorted(
        all_basins, key=lambda b: BASIN_ORDER.index(b) if b in BASIN_ORDER else len(BASIN_ORDER)
    )


# --------------------------------------------------------------------------- #
# Predicted series: peak wind and peak IKE, per ensemble sample
# --------------------------------------------------------------------------- #
def _process_track_predicted(item: tuple, ike: IkeSpec) -> tuple | None:
    """Worker: (basin, peak_wind_per_sample, peak_ike_per_sample) for one
    track's predicted ensemble. ``item`` is (idx, path)."""
    idx, path = item
    with xr.open_dataset(path) as ds:
        basin = track_basin(ds)
        speed = wind_speed_field(ds)
        if speed is None:
            return None
        if "sample" in speed.dims:
            reduce_dims = [d for d in speed.dims if d != "sample"]
            peak_wind = speed.max(dim=reduce_dims, skipna=True).values
        else:
            peak_wind = np.array([float(speed.max(skipna=True).values)])

        lat_name, lon_name = _coord_names(ds)
        speed = speed.load()
        reduced = regional_reduce(speed, "ike_box", lat_name, lon_name, ike)
        peak_ike = np.atleast_1d(np.asarray(reduced.values, dtype=float))
    return idx, basin, np.atleast_1d(peak_wind), peak_ike


def collect_track_predicted(
    directory: Path, ike: IkeSpec, max_workers: int
) -> tuple[dict, dict, dict]:
    """Return ``({idx: peak_wind_per_sample}, {idx: peak_ike_per_sample},
    {idx: basin_slug})`` for every track in ``directory`` that has predicted
    wind components."""
    files = find_track_files(directory)
    items = list(files.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_process_track_predicted, items, [ike] * len(items))
    wind_peaks, ike_peaks, basin_by_idx = {}, {}, {}
    for result in filter(None, results):
        idx, basin, wind, ike_val = result
        wind_peaks[idx] = wind
        ike_peaks[idx] = ike_val
        basin_by_idx[idx] = basin
    logger.info(
        "%s: computed predicted peak wind/IKE for %d of %d track(s)",
        directory,
        len(wind_peaks),
        len(files),
    )
    return wind_peaks, ike_peaks, basin_by_idx


# --------------------------------------------------------------------------- #
# X-SHiELD reference series: peak wind and peak IKE (no ensemble dimension)
# --------------------------------------------------------------------------- #
def _process_track_target(item: tuple, ike: IkeSpec) -> tuple | None:
    """Worker: (basin, peak_wind, peak_ike) for one track's X-SHiELD reference
    snapshot. ``item`` is (idx, path)."""
    idx, path = item
    with xr.open_dataset(path) as ds:
        basin = track_basin(ds)
        speed = target_wind_speed_field(ds)
        if speed is None:
            return None
        lat_name, lon_name = _coord_names(ds)
        speed = speed.load()
        peak_wind = np.array([float(speed.max(skipna=True).values)])
        reduced = regional_reduce(speed, "ike_box", lat_name, lon_name, ike)
        peak_ike = np.atleast_1d(np.asarray(reduced.values, dtype=float))
    return idx, basin, peak_wind, peak_ike


def collect_track_target(
    directory: Path, ike: IkeSpec, max_workers: int
) -> tuple[dict, dict, dict]:
    """Return ``({idx: peak_wind}, {idx: peak_ike}, {idx: basin_slug})``
    (single-value arrays) of the X-SHiELD reference for every track in
    ``directory`` that has target wind components."""
    files = find_track_files(directory)
    items = list(files.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_process_track_target, items, [ike] * len(items))
    wind_peaks, ike_peaks, basin_by_idx = {}, {}, {}
    for result in filter(None, results):
        idx, basin, wind, ike_val = result
        wind_peaks[idx] = wind
        ike_peaks[idx] = ike_val
        basin_by_idx[idx] = basin
    logger.info(
        "%s: computed X-SHiELD reference peak wind/IKE for %d of %d track(s)",
        directory,
        len(wind_peaks),
        len(files),
    )
    return wind_peaks, ike_peaks, basin_by_idx


# --------------------------------------------------------------------------- #
# Plot-data cache: one npz per series, holding each track's peaks flattened
# in track order (split back apart by the per-track sample counts).
# --------------------------------------------------------------------------- #
def _ike_tag(ike: IkeSpec) -> str:
    """Cache-name fragment for every IkeSpec field, so non-default IKE options
    never pick up another definition's cached values."""
    def fmt(value):
        return "none" if value is None else f"{value:g}"

    return f"ike_box{ike.box_deg:g}_min{fmt(ike.min_wind)}_max{fmt(ike.max_wind)}"


def _peaks_to_arrays(wind_peaks: dict, ike_peaks: dict, basin_by_idx: dict) -> dict:
    idxs = list(wind_peaks)
    return {
        "idx": np.array(idxs, dtype=str),
        # "" stands for a track outside every basin box (basin None).
        "basin": np.array([basin_by_idx[idx] or "" for idx in idxs], dtype=str),
        "wind_n": np.array([len(wind_peaks[idx]) for idx in idxs]),
        "wind": np.concatenate([wind_peaks[idx] for idx in idxs]),
        "ike_n": np.array([len(ike_peaks[idx]) for idx in idxs]),
        "ike": np.concatenate([ike_peaks[idx] for idx in idxs]),
    }


def _arrays_to_peaks(arrays: dict) -> tuple[dict, dict, dict]:
    idxs = [str(idx) for idx in arrays["idx"]]
    wind = np.split(arrays["wind"], np.cumsum(arrays["wind_n"])[:-1])
    ike_val = np.split(arrays["ike"], np.cumsum(arrays["ike_n"])[:-1])
    return (
        dict(zip(idxs, wind)),
        dict(zip(idxs, ike_val)),
        {idx: (str(basin) or None) for idx, basin in zip(idxs, arrays["basin"])},
    )


def load_track_peaks(
    kind: str, directory: Path, ike: IkeSpec, max_workers: int, refresh: bool = False
) -> tuple[dict, dict, dict]:
    """Cached ``collect_track_predicted`` (``kind="predicted"``) or
    ``collect_track_target`` (``kind="target"``) for ``directory``."""
    collect = {"predicted": collect_track_predicted, "target": collect_track_target}[kind]
    arrays = cached_arrays(
        f"{CACHE_SUBDIR}/{directory.name}_{kind}_{_ike_tag(ike)}.npz",
        lambda: _peaks_to_arrays(*collect(directory, ike, max_workers)),
        refresh,
    )
    return _arrays_to_peaks(arrays)


# --------------------------------------------------------------------------- #
# Histogram-binning helpers, copied unmodified from
# 2026-07-16-tc-tracks-prod/plot_max_wind_histogram.py -- used only by this
# script among this directory's max-snapshot scripts.
# --------------------------------------------------------------------------- #
DEFAULT_BIN_WIDTH = 5.0
DEFAULT_IKE_BIN_WIDTH = 5.0  # TJ


def peaks_to_values_weights(peaks: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``{event: peak_per_sample}`` into (values, weights): each
    sample is its own histogram entry, weighted 1/n_samples so a track's
    samples sum to a weight of 1 (count tracks, not samples)."""
    if not peaks:
        return np.array([]), np.array([])
    values = np.concatenate(list(peaks.values()))
    weights = np.concatenate([np.full(len(peak), 1.0 / len(peak)) for peak in peaks.values()])
    return values, weights


def bins_covering(peaks_by_dir: list, bin_width: float, what: str) -> np.ndarray:
    """Uniform ``bin_width`` bins from 0 up past the largest value in any series,
    so every series is histogrammed on the same axis."""
    all_values = np.concatenate(
        [peaks_to_values_weights(peaks)[0] for _name, peaks in peaks_by_dir]
    )
    if all_values.size == 0:
        raise SystemExit(f"No tracks with {what} data found in either directory.")
    n_bins = int(np.ceil(float(all_values.max()) / bin_width)) + 1
    return np.arange(n_bins + 1) * bin_width


# Percentile markers overlaid on each series' distribution in the aggregated
# (non-basin) figures only -- the by-basin figures already split each series
# into small subplots, where percentile lines on top of a handful of tracks
# add clutter without much signal. Distinguished by linestyle rather than
# color, since color is already spoken for by series (see SERIES_COLORS) --
# the median (P50) is solid and heavier so it reads as the "center" line at a
# glance, the rest are progressively lighter dashes.
PERCENTILES = (50, 75, 95, 99)
PERCENTILE_LINESTYLES = {
    50: "-",
    75: (0, (3, 1, 1, 1)),
    95: (0, (5, 1)),
    99: (0, (1, 1)),
}
PERCENTILE_LINEWIDTHS = {50: 1.4, 75: 1.0, 95: 1.0, 99: 1.0}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted ``q``-th percentile (0-100) of ``values``, honoring the same
    per-sample weights the histogram itself is drawn with (1/n_samples per
    track; see ``peaks_to_values_weights``) so an ensemble's samples don't
    outweigh a track with fewer of them."""
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cum_weights = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cum_weights /= sorted_weights.sum()
    return float(np.interp(q / 100.0, cum_weights, sorted_values))


def _annotate_percentiles(
    ax, series_values: dict, colors: dict, percentiles: tuple
) -> None:
    """Draw each series' percentile line, then one small text label per
    percentile placed just before its leftmost line, at the top of the axes
    (label hangs down from there, following the line) -- adapted from the
    axvline + text style in ``figx-global-2023-histograms.py``'s
    ``plot_histogram`` (label positioned in data-x/axes-fraction-y
    coordinates so its height sits at the top regardless of the y autoscale).

    ``series_values`` is ``{name: (values, weights)}`` (see
    ``peaks_to_values_weights``); a series with no values is skipped.
    """
    x_span = np.subtract(*ax.get_xlim()[::-1])
    for q in percentiles:
        line_values = []
        for name, (values, weights) in series_values.items():
            if values.size == 0:
                continue
            value = weighted_percentile(values, weights, q)
            line_values.append(value)
            ax.axvline(
                value,
                color=colors[name],
                linestyle=PERCENTILE_LINESTYLES[q],
                linewidth=PERCENTILE_LINEWIDTHS[q],
                alpha=0.8,
            )
        if not line_values:
            continue
        ax.text(
            min(line_values) - 0.003 * x_span,
            0.98,
            f"P{q:g}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="0.35",
        )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_histograms(
    ax,
    peaks_by_dir: list,
    bins: np.ndarray,
    colors: dict | None = None,
    percentiles: tuple = (),
    density: bool = False,
) -> None:
    """Draw the series as overlapping bars, one ``hist`` call per series,
    normalized to a probability *density* (area 1) rather than raw counts --
    each series is normalized by its own total weight, so the two series'
    shapes are directly comparable even though they cover different numbers
    of tracks (753 X-SHiELD reference vs. 690 HiRO-ACE, since a few tracks
    only have HiRO-ACE or only have a reference).

    ``percentiles`` (e.g. ``PERCENTILES``) additionally overlays each series'
    weighted percentiles at those q-values as vertical lines, each labelled
    with a text annotation (see ``_annotate_percentiles``). Empty by default
    -- pass explicitly where wanted (see ``plot_histogram``).

    Otherwise identical to ``plot_max_wind_histogram.py``'s ``_plot_histograms``
    -- reimplemented locally only because that version hard-codes raw counts.
    """
    if not any(len(peaks) for _name, peaks in peaks_by_dir):
        return
    colors = colors or series_colors([name for name, _ in peaks_by_dir])
    series_values = {}
    for i, (name, peaks) in enumerate(peaks_by_dir):
        values, weights = peaks_to_values_weights(peaks)
        color = colors[name]
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            density=density,
            label=f"{name} (N={len(peaks)})",
            color=to_rgba(color, 0.4),
            edgecolor=color,
            linewidth=1.3,
            histtype="bar",
            zorder=2 + i,
        )
        series_values[name] = (values, weights)
    # Headroom for the legend, whose box is wide (full directory names) and would
    # otherwise clip the tallest bars.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    if percentiles:
        _annotate_percentiles(ax, series_values, colors, percentiles)


def plot_histogram(
    peaks_by_dir: list,
    bins: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    colors: dict | None = None,
    log_y: bool = False,
    xmin=None,
    density: bool = False,
    figsize: tuple = (8, 6),
    show_title: bool = True,
) -> None:
    """Single-panel histogram of a per-track peak value, one series per
    directory. ``log_y`` log-scales the density axis, which keeps the long
    tail of intense storms visible against the much larger low-intensity bins
    that dominate a linear axis.

    ``density`` controls both the underlying ``hist`` call and the y-label:
    with ``density=False`` (the default) the bars are weighted counts, so the
    axis is always labelled "Count" regardless of ``ylabel``; only when
    ``density=True`` does the caller's ``ylabel`` (a density unit, e.g.
    "density (m/s)$^{-1}$") apply.

    Also draws the 50/75/95/99th percentile lines (``PERCENTILES``), each
    labelled with a text annotation rather than a legend entry (see
    ``_annotate_percentiles``) -- the by-basin figures skip these (see
    ``plot_histogram_by_basin``)."""
    fig, ax = plt.subplots(figsize=figsize)
    # set_yscale must precede _plot_histograms: it ends with set_ylim(top=...)
    # to make legend room, and that top is only sane once autoscaling has
    # already happened in log space -- computed first on a linear scale, it
    # can come out at/below the bottom of the eventual log axis.
    if log_y:
        ax.set_yscale("log")
    _plot_histograms(ax, peaks_by_dir, bins, colors, percentiles=PERCENTILES, density=density)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if density else "Count")
    if show_title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(xmin, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_histogram_by_basin(
    peaks_by_dir: list,
    bins: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    colors: dict | None = None,
    log_y: bool = False,
    density: bool = False,
) -> None:
    """Same histogram as ``plot_histogram``, but with one subplot per basin.
    ``density`` follows the same convention as ``plot_histogram``: with the
    default ``density=False``, the y-label is always "Count" regardless of
    ``ylabel``.

    ``peaks_by_dir`` is a list of ``(label, peaks, basin_by_idx)``, since here
    (unlike ``plot_max_wind_histogram.py``'s ``<basin_slug>_<rank>`` event
    names) each series' basin lookup is external to the peaks dict -- derived
    per track from its own snapshot's coordinates (see ``track_basin``).

    ``log_y`` is applied per-subplot (each basin has its own y-scale; see
    ``sharey=False`` below), and like ``plot_histogram`` must be set before
    that subplot's ``_plot_histograms`` call.

    Unlike ``plot_histogram``, no percentile lines are drawn here -- with up
    to six small subplots already carrying their own series legend, per-track
    percentiles over a handful of tracks would add clutter without much
    signal.
    """
    basin_groups = [
        (name, group_peaks_by_basin(peaks, basin_by_idx)) for name, peaks, basin_by_idx in peaks_by_dir
    ]
    basins = _ordered_basins([groups for _name, groups in basin_groups])

    ncols = 2 if len(basins) > 1 else 1
    nrows = -(-len(basins) // ncols)
    # sharey=False: each basin's density peak has a different height (the
    # series within a basin are already made comparable to each other by the
    # density normalization, but basins are not comparable to one another --
    # a sharp, concentrated basin peaks higher than a spread-out one
    # regardless of track count), so a shared axis leaves most panels mostly
    # empty under whichever basin happens to peak highest.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=False, squeeze=False
    )
    flat_axes = list(axes.flat)

    for ax, basin in zip(flat_axes, basins):
        basin_peaks_by_dir = [(name, groups.get(basin, {})) for name, groups in basin_groups]
        if log_y:
            ax.set_yscale("log")
        _plot_histograms(ax, basin_peaks_by_dir, bins, colors, density=density)
        ax.set_title(BASIN_DISPLAY_NAME.get(basin, basin), fontsize=10)
        ax.legend(loc="best", fontsize=7)
    for ax in flat_axes[len(basins):]:
        ax.axis("off")

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in flat_axes[: len(basins)]:
        ax.set_ylabel(ylabel if density else "Count")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        help=f"Perfect-prediction max-snapshot directory (default: {DEFAULT_PERFECT_PRED_DIR}). "
        "Also supplies the X-SHiELD reference series (its '_target' fields).",
    )
    parser.add_argument(
        "--hiro-ace-dir",
        default=DEFAULT_HIRO_ACE_DIR,
        help=f"HiRO-ACE rollout max-snapshot directory (default: {DEFAULT_HIRO_ACE_DIR}).",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output path for the max-wind histogram.")
    parser.add_argument(
        "--basin-out",
        default=DEFAULT_BASIN_OUT,
        help="Output path for the by-basin max-wind histogram figure.",
    )
    parser.add_argument(
        "--ike-out", default=DEFAULT_IKE_OUT, help="Output path for the IKE histogram."
    )
    parser.add_argument(
        "--ike-basin-out",
        default=DEFAULT_IKE_BASIN_OUT,
        help="Output path for the by-basin IKE histogram figure.",
    )
    parser.add_argument(
        "--no-basin-figure", action="store_true", help="Skip both by-basin figures."
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the X-SHiELD reference series.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help=f"Wind histogram bin width, in m/s (default: {DEFAULT_BIN_WIDTH:g}).",
    )
    parser.add_argument(
        "--ike-box-deg",
        type=float,
        default=DEFAULT_IKE_BOX_DEG,
        help="Width of the square storm-centered lat/lon domain IKE is integrated "
        f"over, in degrees (default: {DEFAULT_IKE_BOX_DEG:g}, the Powell and "
        "Reinhold domain).",
    )
    parser.add_argument(
        "--ike-min-wind",
        type=float,
        default=DEFAULT_IKE_MIN_WIND,
        help="Only grid cells with sustained 10m wind above this, in m/s, "
        f"contribute to IKE (default: {DEFAULT_IKE_MIN_WIND:g}, tropical-storm "
        "force, giving Powell and Reinhold's IKE_TS). Pass 0 for every cell.",
    )
    parser.add_argument(
        "--ike-max-wind",
        type=float,
        default=None,
        help="Optional upper bound on the IKE wind band, in m/s.",
    )
    parser.add_argument(
        "--ike-bin-width",
        type=float,
        default=DEFAULT_IKE_BIN_WIDTH,
        help=f"IKE histogram bin width, in TJ (default: {DEFAULT_IKE_BIN_WIDTH:g}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Worker processes for reading/reducing tracks in parallel (default: 16).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute the cached per-track peaks from the snapshot files.",
    )
    args = parser.parse_args()

    # 0 spells "no threshold" on the command line.
    ike = IkeSpec(
        box_deg=args.ike_box_deg,
        min_wind=args.ike_min_wind if args.ike_min_wind else None,
        max_wind=args.ike_max_wind,
    )

    perfect_pred_dir = Path(args.perfect_pred_dir)
    hiro_ace_dir = Path(args.hiro_ace_dir)

    # Colours are assigned over every series this script could plot -- including
    # the unplotted perfect-prediction series -- so HiRO-ACE and the reference
    # keep the same hue they would if perfect-prediction were added back later.
    labels = [PERFECT_PRED_LABEL, HIRO_ACE_LABEL, REFERENCE_LABEL]
    colors = series_colors(labels)

    hiro_ace_wind, hiro_ace_ike, hiro_ace_basin = load_track_peaks(
        "predicted", hiro_ace_dir, ike, args.max_workers, args.refresh
    )

    wind_by_series = [(HIRO_ACE_LABEL, hiro_ace_wind)]
    ike_by_series = [(HIRO_ACE_LABEL, hiro_ace_ike)]
    wind_by_series_basin = [(HIRO_ACE_LABEL, hiro_ace_wind, hiro_ace_basin)]
    ike_by_series_basin = [(HIRO_ACE_LABEL, hiro_ace_ike, hiro_ace_basin)]

    if not args.no_reference:
        reference_wind, reference_ike, reference_basin = load_track_peaks(
            "target", perfect_pred_dir, ike, args.max_workers, args.refresh
        )
        wind_by_series.append((REFERENCE_LABEL, reference_wind))
        ike_by_series.append((REFERENCE_LABEL, reference_ike))
        wind_by_series_basin.append((REFERENCE_LABEL, reference_wind, reference_basin))
        ike_by_series_basin.append((REFERENCE_LABEL, reference_ike, reference_basin))

    wind_ylabel = "density (m/s)$^{{-1}}$"
    wind_bins = bins_covering(wind_by_series, args.bin_width, "wind")
    plot_histogram(
        wind_by_series,
        wind_bins,
        Path(args.out),
        xlabel="Max 10m wind speed at peak wind speed snapshot (m/s)",
        ylabel=wind_ylabel,
        title="Distribution of per-track maximum 10m wind speed",
        colors=colors,
        log_y=True,
        xmin=15
    )
    if not args.no_basin_figure:
        plot_histogram_by_basin(
            wind_by_series_basin,
            wind_bins,
            Path(args.basin_out),
            xlabel="Max 10m wind speed at peak wind speed snapshot (m/s)",
            ylabel=wind_ylabel,
            title="Distribution of per-track maximum 10m wind speed, by basin",
            colors=colors,
        )

    logger.info("IKE definition: %s", ike.describe())
    ike_bins = bins_covering(ike_by_series, args.ike_bin_width, "IKE")
    ike_xlabel = f"Integrated kinetic energy at peak wind speed snapshot (TJ)"
    ike_ylabel = "density (TJ)$^{{-1}}$"
    plot_histogram(
        ike_by_series,
        ike_bins,
        Path(args.ike_out),
        xlabel=ike_xlabel,
        ylabel=ike_ylabel,
        title=(
            "Distribution of per-track peak integrated kinetic energy"
        ),
        colors=colors,
        log_y=True,
        figsize=(6.5, 5),
        show_title=False,
    )
    if not args.no_basin_figure:
        plot_histogram_by_basin(
            ike_by_series_basin,
            ike_bins,
            Path(args.ike_basin_out),
            xlabel=ike_xlabel,
            ylabel=ike_ylabel,
            log_y=True,
            title=(
                "Distribution of per-track peak integrated kinetic energy, by basin"
            ),
            colors=colors,
        )


if __name__ == "__main__":
    main()
