"""
Shared helpers for this directory's HiRO-ACE v2 "max snapshot" figure
scripts: ``plot_max_wind_ike_histograms.py``,
``plot_radial_wind_profile_max_snapshot.py``, and
``plot_radial_wind_profile_two_panel.py``.

Those three scripts all read the same kind of file -- ``tc_<idx>_<YYYYMMDDTHH>.nc``,
one pre-selected peak-intensity snapshot per track (see
``track_detection/downscale_highest_max_wind.py``), holding both a downscaling
model's ``*_predicted`` fields and (in the perfect-prediction runs) the
colocated X-SHiELD ``*_target`` fields -- so they need the same field-name
lookup, centroid-finding, and radius/azimuth-binning primitives. This module
collects exactly the pieces used by two or more of them, so that logic lives
in one place instead of being copied by hand (and drifting) across files.

Only genuinely shared code lives here. Anything used by just one of the three
scripts (e.g. the IKE integral, the histogram-binning helpers, or the
composite/classification machinery built on top of ``process_snapshot``-style
functions) is kept local to that script instead, to avoid bloating this module
with single-consumer code -- see each script's own module docstring for what
it inlines.

Values that differ between the three scripts by design -- e.g. the default
composite radius (``DEFAULT_MAX_RADIUS_DEG``) and bin counts (``N_R_BINS``,
``N_THETA_BINS``), which are *not* the same across all three -- are kept as
each script's own local constants rather than centralized here, and passed
explicitly into ``bin_edges``.

This module deliberately has no imports from, or ``sys.path`` reach into,
anything outside this directory.
"""

import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Field name candidates, across the predicted and X-SHiELD reference ("_target")
# naming conventions used by these max-snapshot runs.
# --------------------------------------------------------------------------- #
U10_CANDIDATES = (
    "UGRD10m_predicted",
    # Newer downscaling runs (e.g. the 25km->3km outputs) name the predicted
    # fields after the reference variables rather than the GRIB short names.
    "eastward_wind_at_ten_meters_predicted",
    "eastward_wind_at_ten_meters",
    "UGRD10m",
    "U10",
)
V10_CANDIDATES = (
    "VGRD10m_predicted",
    "northward_wind_at_ten_meters_predicted",
    "northward_wind_at_ten_meters",
    "VGRD10m",
    "V10",
)
PRMSL_CANDIDATES = ("PRMSL_predicted", "PRMSL", "PRMSLmsl", "PRES_msl")

# The reference (truth) fields are stored as "<field>_target" alongside each
# perfect-prediction snapshot -- not covered by the *_CANDIDATES above, which
# only look for "_predicted" (or bare) names.
U10_TARGET_CANDIDATES = ("eastward_wind_at_ten_meters_target", "UGRD10m_target")
V10_TARGET_CANDIDATES = ("northward_wind_at_ten_meters_target", "VGRD10m_target")
TARGET_PRMSL_CANDIDATES = ("PRMSL_target",)

_PA_UNITS = {"pa", "pascal", "pascals"}
_HPA_UNITS = {"hpa", "mb", "mbar", "millibar", "millibars"}
# Above this magnitude a "pressure" field is taken to be in Pa and converted to
# hPa; the prediction files already store hPa (values ~1000).
_PA_THRESHOLD = 10_000.0


def _first_present(ds: xr.Dataset, names: tuple) -> str | None:
    """Return the first of ``names`` present in ``ds``, or None."""
    for name in names:
        if name in ds:
            return name
    return None


def _is_pascals(da: xr.DataArray) -> bool:
    """Whether a pressure field is in Pa (vs hPa/mb); prefers the ``units``
    attribute, falling back to a small corner sample's magnitude."""
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in _PA_UNITS:
        return True
    if units in _HPA_UNITS:
        return False
    corner = da.isel({dim: slice(0, 4) for dim in da.dims})
    vals = np.asarray(corner.values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return finite.size > 0 and float(np.nanmedian(finite)) > _PA_THRESHOLD


def wind_speed_field(ds: xr.Dataset) -> xr.DataArray | None:
    """Predicted 10m wind speed |(u, v)|, or None if absent."""
    u_name = _first_present(ds, U10_CANDIDATES)
    v_name = _first_present(ds, V10_CANDIDATES)
    if u_name is None or v_name is None:
        return None
    return np.hypot(ds[u_name], ds[v_name])


def target_wind_speed_field(ds: xr.Dataset) -> xr.DataArray | None:
    """X-SHiELD reference 10m wind speed |(u, v)|, or None if absent."""
    u_name = _first_present(ds, U10_TARGET_CANDIDATES)
    v_name = _first_present(ds, V10_TARGET_CANDIDATES)
    if u_name is None or v_name is None:
        return None
    return np.hypot(ds[u_name], ds[v_name])


def prmsl_field(ds: xr.Dataset) -> xr.DataArray | None:
    """Predicted sea-level pressure in hPa, or None if absent."""
    name = _first_present(ds, PRMSL_CANDIDATES)
    if name is None:
        return None
    da = ds[name]
    return da / 100.0 if _is_pascals(da) else da


def target_prmsl_field(ds: xr.Dataset) -> xr.DataArray | None:
    """X-SHiELD reference sea-level pressure in hPa, or None if absent."""
    name = _first_present(ds, TARGET_PRMSL_CANDIDATES)
    if name is None:
        return None
    da = ds[name]
    return da / 100.0 if _is_pascals(da) else da


# --------------------------------------------------------------------------- #
# Coordinate lookup and centroid-finding
# --------------------------------------------------------------------------- #
def _coord_names(ds: xr.Dataset) -> tuple:
    for lat_name, lon_name in (("lat", "lon"), ("grid_yt", "grid_xt"), ("latitude", "longitude")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
    raise KeyError(f"Could not find lat/lon coords in {list(ds.coords)}")


def compute_centroid(prmsl: xr.DataArray, lat_name: str, lon_name: str) -> tuple:
    """(lat, lon) of the PRMSL-minimum pixel -- a bare argmin, reliable here
    since these are max-snapshot files preselected to hold a storm."""
    vals = np.asarray(prmsl.values, dtype=float)
    iy, ix = np.unravel_index(np.nanargmin(vals), vals.shape)
    return float(prmsl[lat_name].values[iy]), float(prmsl[lon_name].values[ix])


# --------------------------------------------------------------------------- #
# Radius/azimuth binning
# --------------------------------------------------------------------------- #
KM_PER_DEG_LAT = 111.32


def radius_azimuth(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple:
    """Local equirectangular (radius_km, azimuth_rad) of every (lat, lon)
    pixel relative to (lat0, lon0). Azimuth is the standard math angle (0 =
    east, increasing counterclockwise), matching matplotlib's polar
    convention."""
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    dy = (lat2d - lat0) * KM_PER_DEG_LAT
    dx = (lon2d - lon0) * KM_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    return np.hypot(dx, dy), np.arctan2(dy, dx)


def bin_edges(max_radius_deg: float, n_theta_bins: int, n_r_bins: int) -> tuple:
    """(theta_edges, r_edges) for a polar (azimuth x radius) histogram out to
    ``max_radius_deg``. ``n_theta_bins``/``n_r_bins`` are passed explicitly
    (rather than fixed module constants) because the three scripts sharing
    this helper use different bin counts and radii."""
    r_max_km = max_radius_deg * KM_PER_DEG_LAT
    return (
        np.linspace(-np.pi, np.pi, n_theta_bins + 1),
        np.linspace(0.0, r_max_km, n_r_bins + 1),
    )


def _azimuthal_mean(sum_grid: np.ndarray, count_grid: np.ndarray) -> np.ndarray:
    """Radius-binned profile: sum-of-sums / sum-of-counts across all theta
    bins at each radius, so it weights by actual pixel counts rather than
    treating sparsely sampled theta bins the same as well-sampled ones."""
    total_sum = sum_grid.sum(axis=0)
    total_count = count_grid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_count > 0, total_sum / total_count, np.nan)


# --------------------------------------------------------------------------- #
# File discovery: one max-snapshot file per track, optionally restricted by year
# --------------------------------------------------------------------------- #
# "tc_<idx>_<YYYYMMDDTHH>.nc"; excludes any suffixed variant (e.g. a one-off
# "_ens32" rerun), since those don't match this pattern's end anchor. The idx
# (not the timestamp) is what matches a track between the perfect-prediction
# and HiRO-ACE directories, since the two runs pick each track's
# peak-intensity time independently.
_TC_RE = re.compile(r"^tc_(?P<idx>\d+)_\d{8}T\d{2}$")
# "..._<YYYYMMDDTHH>.nc" -> the stamp's year, for restricting a directory to a
# single (e.g. holdout) year.
_YEAR_RE = re.compile(r"_(?P<year>\d{4})\d{4}T\d{2}$")


def find_track_files(directory: Path) -> dict:
    """Map track index (e.g. "000") -> its single max-snapshot file."""
    files = {}
    for path in sorted(directory.glob("tc_*.nc")):
        match = _TC_RE.match(path.stem)
        if match:
            files[match["idx"]] = path
    return files


def year_of_path(path: Path) -> int | None:
    match = _YEAR_RE.search(path.stem)
    return int(match["year"]) if match else None


def filter_by_year(files: dict, year: int) -> dict:
    """Restrict a ``{idx: path}`` mapping (see ``find_track_files``) to tracks
    whose snapshot timestamp falls in ``year``."""
    filtered = {idx: path for idx, path in files.items() if year_of_path(path) == year}
    logger.info("Restricted %d of %d track(s) to year %d", len(filtered), len(files), year)
    return filtered
