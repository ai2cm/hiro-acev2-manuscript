"""Shared 100 km global-histogram panel, so two figures draw the same panel.

``figx-100km-global-histograms.py`` draws all five fields as a figure of its
own; ``figx-global-coarse-res-multifield-bias.py`` draws one per row, beside
that field's maps. Both read the same cached counts through the same plotting
function, so a distribution shown in one figure is the same distribution, drawn
the same way, as in the other.

Counts are cached in two layers, both read by ``load_cached_histograms``:
``global_100km_10yr_histograms.npz`` holds the reference and ic0000, written
before the rest of the ensemble was computed, and ``global_100km_histogram_cache/``
holds one npz per series added since. One file per series rather than one file
for all of them is what makes a run resumable -- each store is a decade of
6-hourly global fields, so a job that dies partway through must not lose the
members it already finished -- and it means a reader never has to rewrite a file
another process may be writing.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

from series_colors import HIRO_ACE_COLOR, REFERENCE_COLOR

HERE = Path(__file__).parent

# the ACE2S rollouts only cover this decade; the reference store spans a wider
# 2013-2024 window, so both are sliced to the same period before histogramming
START, END = "2014-01-01", "2023-12-31"
NUM_BINS = 500

# computing these histograms means loading a decade of 6-hourly global fields
# into memory, so the result is cached to disk and reused on later runs
CACHE = HERE / "plot_data_cache" / "global_100km_10yr_histograms.npz"
MEMBER_CACHE_DIR = HERE / "plot_data_cache" / "global_100km_histogram_cache"

# Keys as written into the cache, in plotting order. These name the series in
# the stored arrays; what a figure *calls* them is its own ``labels`` argument.
REFERENCE_SOURCE = "reference"
PREDICTION_SOURCE = "HiRO-ACE"
SOURCE_KEYS = (REFERENCE_SOURCE, PREDICTION_SOURCE)

# ic0000 was cached under the name the figures called it by before the rest of
# the ensemble existed; every member added since is keyed by its own tag. Kept
# as a mapping rather than renamed so the existing cache stays readable as-is.
MEMBER_SOURCE_KEYS = {"ic0000": PREDICTION_SOURCE}

COLORS = {REFERENCE_SOURCE: REFERENCE_COLOR, PREDICTION_SOURCE: HIRO_ACE_COLOR}
LABELS = {
    REFERENCE_SOURCE: "X-SHiELD reference (100 km)",
    PREDICTION_SOURCE: "HiRO-ACEv2 (100 km)",
}
TITLES = {
    "PRATEsfc": "Surface precipitation rate",
    "wind_speed": "Wind speed",
    "PRMSL": "Sea level pressure",
    "eastward_wind_at_ten_meters": "Eastward 10m wind",
    "northward_wind_at_ten_meters": "Northward 10m wind",
}
# one linestyle per percentile, so the two lines of a group are distinguishable
# from the other group as well as from each other by color
PERCENTILE_STYLES = {99: "--", 99.99: "--", 99.9999: "--"}
# the 10 m wind components are signed and roughly symmetric, so the lower tail
# is as interesting as the upper one; these mirror the upper percentiles above
LOWER_PERCENTILE_STYLES = {1: ":", 0.01: ":", 0.0001: ":"}


def member_source_key(tag: str) -> str:
    """Cache key for the member store tagged ``tag`` (ic0000, ic0001, ...)."""
    return MEMBER_SOURCE_KEYS.get(tag, tag)


# the same variable is named differently across output sets; either name may
# be on disk, so treat the members of each group as interchangeable
VAR_ALIASES = {
    "UGRD10m": "eastward_wind_at_ten_meters",
    "eastward_wind_at_ten_meters": "UGRD10m",
    "VGRD10m": "northward_wind_at_ten_meters",
    "northward_wind_at_ten_meters": "VGRD10m",
}


def is_wind_component(var: str) -> bool:
    """True for the eastward/northward 10 m wind, in either naming convention."""
    return var.endswith(("GRD10m", "_wind_at_ten_meters"))


def load_arrays(path: str) -> dict[str, np.ndarray]:
    """Finite values over [START, END] for every histogram variable available
    in the zarr store at ``path``, keyed by canonical variable name.

    ``wind_speed`` is derived from the two wind components (whichever naming
    convention the store uses), computed before either component is raveled
    and filtered so the two stay aligned.
    """
    ds = xr.open_zarr(path).sel(time=slice(START, END))
    raw = {}
    for var in (
        "PRATEsfc",
        "PRMSL",
        "eastward_wind_at_ten_meters",
        "northward_wind_at_ten_meters",
    ):
        for candidate in (var, VAR_ALIASES.get(var)):
            if candidate and candidate in ds.data_vars:
                raw[var] = ds[candidate].compute().values
                break
    if "PRATEsfc" in raw:
        raw["PRATEsfc"] = raw["PRATEsfc"] * 86400  # kg/m2/s -> mm/day
    if "eastward_wind_at_ten_meters" in raw and "northward_wind_at_ten_meters" in raw:
        raw["wind_speed"] = np.hypot(
            raw["eastward_wind_at_ten_meters"], raw["northward_wind_at_ten_meters"]
        )
    return {
        var: values.ravel()[np.isfinite(values.ravel())] for var, values in raw.items()
    }


def cached_edges(source: str) -> dict[str, np.ndarray]:
    """The bin edges an already-cached series used, keyed by variable."""
    prefix, suffix = f"{source}__", "__edges"
    return {
        key[len(prefix) : -len(suffix)]: value
        for key, value in load_cached_histograms().items()
        if key.startswith(prefix) and key.endswith(suffix)
    }


def histogram_source(
    source: str, path: str, edges: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Counts/edges for every histogram variable in one store.

    Flat dict keyed by ``f"{source}__{var}__counts"`` / ``"...__edges"``
    (rather than a Dataset) since bin counts differ per variable.

    Without ``edges`` a series is binned over its own range, which resolves its
    own tails but makes it incommensurable with every other series. Passing the
    edges of another series instead puts them on one grid, so their counts can
    be summed or compared bin by bin -- what an ensemble needs.

    A shared grid cannot hold what falls outside it, and ``np.histogram`` drops
    such values silently, so each variable binned this way also records how many
    landed below and above the grid and the range it actually spanned, under
    ``...__outside`` and ``...__range``. Whether the truncation matters is then
    a number someone can look at rather than an assumption.
    """
    result = {}
    for var, values in load_arrays(path).items():
        shared = None if edges is None else edges.get(var)
        counts, bin_edges = np.histogram(
            values, bins=NUM_BINS if shared is None else shared
        )
        result[f"{source}__{var}__counts"] = counts
        result[f"{source}__{var}__edges"] = bin_edges
        if shared is not None:
            # np.histogram closes the last bin on the right, so only values
            # strictly outside the grid are the ones it dropped.
            result[f"{source}__{var}__outside"] = np.array(
                [int((values < bin_edges[0]).sum()), int((values > bin_edges[-1]).sum())]
            )
            result[f"{source}__{var}__range"] = np.array(
                [values.min(), values.max()], dtype=float
            )
    return result


def source_cache_path(source: str) -> Path:
    # The window is part of the name, so changing START/END misses the cache
    # rather than quietly reusing counts taken over a different period.
    return MEMBER_CACHE_DIR / f"{source}_{START}_{END}.npz"


def cached_sources() -> set[str]:
    """Every series with counts on disk, across both cache layers."""
    return {key.split("__")[0] for key in load_cached_histograms()}


def legacy_cached_sources() -> set[str]:
    """Series present in the single-file cache written before the ensemble."""
    if not CACHE.exists():
        return set()
    with np.load(CACHE) as npz:
        return {key.split("__")[0] for key in npz.files}


def is_source_cached(source: str) -> bool:
    return source_cache_path(source).exists() or source in legacy_cached_sources()


def write_source_cache(source: str, result: dict[str, np.ndarray]) -> Path:
    """Write one series' counts, atomically.

    A reader merging the cache directory may run at any time, and a job over
    stores this size is killed often enough to matter, so the file appears only
    once it is complete rather than growing in place.
    """
    MEMBER_CACHE_DIR.mkdir(exist_ok=True)
    out = source_cache_path(source)
    tmp = out.with_suffix(".npz.partial")
    # Written through an open handle, not by name: np.savez appends ".npz" to a
    # path that lacks it, which would leave the data under a name the rename
    # below does not look for.
    with open(tmp, "wb") as handle:
        np.savez(handle, **result)
    os.replace(tmp, out)
    return out


def load_cached_histograms() -> dict[str, np.ndarray]:
    """Every cached series, from both cache layers, for a caller that must not
    trigger a recompute.

    ``figx-100km-global-histograms.py`` and
    ``compute-100km-member-histograms.py`` own building the cache; every figure
    only borrows the result, and a decade-long pass over these stores is not
    something to start as a side effect of drawing.
    """
    merged = {}
    if CACHE.exists():
        with np.load(CACHE) as npz:
            merged.update({key: npz[key] for key in npz.files})
    for path in sorted(MEMBER_CACHE_DIR.glob(f"*_{START}_{END}.npz")):
        with np.load(path) as npz:
            merged.update({key: npz[key] for key in npz.files})
    if not merged:
        raise FileNotFoundError(
            f"no cached histograms in {CACHE} or {MEMBER_CACHE_DIR}; "
            "run figx-100km-global-histograms.py to build them"
        )
    return merged


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


def plot_histogram(
    ax,
    var,
    histograms,
    show_legend=False,
    *,
    labels=None,
    show_title=True,
    fontsize=8,
    legend_kwargs=None,
):
    """Draw one variable's two densities, with their tail percentiles, on ``ax``.

    ``labels`` overrides the series names in the legend, for a figure whose
    other columns already name the same two series differently; ``show_title``
    is dropped by a caller that labels the panel from outside (a row label, a
    column header) rather than above the axes.
    """
    labels = LABELS if labels is None else labels
    series = {}
    for source in SOURCE_KEYS:
        counts_key, edges_key = f"{source}__{var}__counts", f"{source}__{var}__edges"
        if counts_key not in histograms:
            continue
        series[source] = trim_empty_bins(histograms[counts_key], histograms[edges_key])

    for name, (counts, edges) in series.items():
        # density, not mass per bin: the two sources have different bin
        # widths, so counts / counts.sum() is not comparable between them
        density = counts / (counts.sum() * np.diff(edges))
        ax.step(
            edges[:-1], density, where="post", color=COLORS[name], label=labels[name]
        )

    if var == "PRMSL":
        # low pressure is the extreme of interest (storms); high pressure isn't
        percentile_styles = dict(LOWER_PERCENTILE_STYLES)
    else:
        percentile_styles = dict(PERCENTILE_STYLES)
        if is_wind_component(var):
            percentile_styles.update(LOWER_PERCENTILE_STYLES)
    # offset as a fraction of the axis span, not of the value itself: PRMSL's
    # percentile values (~900) are far from zero, so a multiplicative offset
    # would shift the label far from the lines it's meant to label
    x_span = np.subtract(*ax.get_xlim()[::-1])
    for percentile, style in percentile_styles.items():
        values = [
            histogram_percentile(counts, edges, percentile)
            for counts, edges in series.values()
        ]
        for name, value in zip(series, values):
            ax.axvline(
                value, color=COLORS[name], linestyle=style, linewidth=1, alpha=0.8
            )
        # label the group once, just right of its rightmost line; x is in data
        # coordinates and y in axes fraction, so the text sits mid-plot
        ax.text(
            max(values) + 0.01 * x_span,
            0.5,
            f"{percentile:g}th",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            fontsize=fontsize,
            color="0.35",
        )
    ax.set_yscale("log")
    if var == "PRATEsfc":
        units = "mm/day"
        ax.set_xlim(1, None)
    elif var == "PRMSL":
        units = "hPa"
    else:
        units = "m/s"
    ax.set_xlabel(units)
    ax.set_ylabel(f"density ({units})$^{{-1}}$")
    if show_legend:
        ax.legend(**(legend_kwargs or {}))
    if show_title:
        ax.set_title(TITLES[var])
