"""Ten-year (2014-2023) time-mean surface fields: X-SHiELD reference, coarse ACE2S, and bias.

The multi-field companion to ``figx-global-coarse-res-precip-bias.py``, which
makes the same three-column comparison for precipitation alone. Here four
surface fields are stacked as rows of one figure --

    precipitation, 10 m zonal wind, 10 m meridional wind,
    sea-level pressure

-- so that a bias in one can be read against the others. The wind and pressure
rows are what make the precipitation row interpretable: a shifted ITCZ shows up
as a precipitation dipole, a low-level convergence bias, and a sea-level
pressure bias at the same longitudes, and only seeing all three together
distinguishes those from independent errors.

``main_figures/global_coarse_res_multifield_bias.png`` -- four rows of three map
columns: the reference climatology, the ACE2S ensemble-mean climatology, and
their difference. Within a row the first two columns share one colorbar so they
can be read against each other; the bias gets its own diverging scale. Rows do
NOT share scales -- the units differ, so every colorbar is labelled with its
own.

``time_mean_cache/`` -- one small netCDF per (source, store), named for the
averaging window so a change to it cannot silently reuse stale means. Each store
is 14600 6-hourly steps on a latency-bound network filesystem, so a full pass
costs tens of minutes while the result is a single 180x360 field. Caching per
pair makes the job resumable: an interrupted run redoes only what is missing.
The precipitation means are read from ``precip_time_means.nc`` when that file
is present and records the same averaging window, since the sibling script
computes exactly the same quantity.

The two stores are on the identical 1-degree grid and Julian calendar and differ
only in coordinate naming (grid_yt/grid_xt vs lat/lon), so the reference is
renamed rather than regridded. Both are sliced to the same 2014-2023 window
before averaging -- the reference store spans 2013-2024, so taking its mean
without slicing would compare different periods.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_cache import CACHE_DIR as PLOT_DATA_CACHE, MAIN_FIGURES_DIR, cached_frame

HERE = Path(__file__).parent
COARSE_ACE2S_DATASET_ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-coarse-ace2s-10yr-inference"
)
COARSENED_XSHIELD_REFERENCE_DATASET = (
    "/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr"
)

REFERENCE_LABEL = "X-SHiELD"
PREDICTION_LABEL = "ACE2S-X-SHiELD"
START, END = "2014-01-01", "2023-12-31"
# PRATEsfc is kg/m**2/s; 86400 converts to the conventional mm/day.
SECONDS_PER_DAY = 86400.0

CACHE_DIR = PLOT_DATA_CACHE / "time_mean_cache"
PRECIP_CACHE = PLOT_DATA_CACHE / "precip_time_means.nc"
# The member stores found under COARSE_ACE2S_DATASET_ROOT, recorded so that
# drawing from the cache never has to list the source directory.
MEMBERS_CACHE = "figx-global-coarse-res-multifield-bias/members.csv"

# The reference store names its horizontal coordinates after the FV3 grid; the
# prediction store already uses lat/lon.
REFERENCE_RENAME = {"grid_yt": "lat", "grid_xt": "lon"}

# Reading one 180x360 timestep costs far more in latency than in bytes on this
# filesystem, so the reduction is scaled by concurrency, not by chunk size.
CHUNK_STEPS = 25
THREADS = 128

# Sequential (magnitude): one hue, light to dark. Diverging (polarity): two hues
# either side of a neutral midpoint -- used both for the signed wind components
# and for every bias column.
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"
# Precipitation and sea-level pressure's reference/prediction columns use
# turbo instead of the scale-driven cmap above; their bias column keeps
# DIVERGING_CMAP.
TURBO_FIELDS = {"precip", "slp"}


@dataclass(frozen=True)
class Field:
    """One row of the figure: what to average, and how to draw it."""

    key: str
    label: str
    units: str
    # How the two climatology columns are coloured.
    #   "magnitude" -- sequential, one hue light to dark (precipitation)
    #   "signed"    -- diverging about zero; the sign is the point (u, v)
    #   "anomaly"   -- diverging about the field's own global mean, for a field
    #                  whose absolute value is uninformative but whose departure
    #                  from the global mean is the pattern (sea-level pressure)
    scale: str
    # Which pass computes it. "wind" reads UGRD10m and VGRD10m together so the
    # two wind rows cost one traversal of the store rather than two.
    source: str
    # Percentiles clipped off the climatology colour scale. The default keeps
    # essentially the whole field; a field whose tails are a small, unrepresen-
    # tative population needs them cut harder or the bulk washes out.
    mean_percentiles: tuple[float, float] = (0.1, 99.9)


FIELDS = (
    Field("precip", "Precipitation", "mm/day", scale="magnitude", source="precip"),
    Field("u10", "10 m zonal wind", "m/s", scale="signed", source="wind"),
    Field("v10", "10 m meridional wind", "m/s", scale="signed", source="wind"),
    # Sea-level pressure sits within a few percent of 1010 hPa everywhere, so a
    # ramp spanning its absolute range is a flat wash; what carries the
    # information is the departure from the global mean -- subtropical highs one
    # way, the Aleutian, Icelandic and circumpolar lows the other. Hence a
    # diverging scale about that mean. Its tail is dominated by the reduction to
    # sea level over the Antarctic ice sheet, an extrapolation below 2-3 km of
    # rock and ice rather than a pressure anyone measures, so the limit is set
    # from the 95th percentile of the departure and that tail saturates.
    Field(
        "slp", "Sea-level pressure", "hPa", scale="anomaly", source="slp",
        mean_percentiles=(0.0, 95.0),
    ),
)
FIELDS_BY_SOURCE = {
    source: [f for f in FIELDS if f.source == source] for source in ("precip", "wind", "slp")
}


def _open(path: Path | str, is_reference: bool) -> xr.Dataset:
    ds = xr.open_zarr(path)
    if is_reference:
        ds = ds.rename(REFERENCE_RENAME)
    return ds.sel(time=slice(START, END)).chunk({"time": CHUNK_STEPS})


def compute_source(path: Path | str, source: str, is_reference: bool) -> xr.Dataset:
    """Time means over [START, END] for every field drawn from one traversal."""
    ds = _open(path, is_reference)
    if source == "precip":
        fields = {"precip": ds.PRATEsfc * SECONDS_PER_DAY}
    elif source == "slp":
        # PRMSL is stored in mb, which is hPa; no conversion needed.
        fields = {"slp": ds.PRMSL}
    elif source == "wind":
        fields = {"u10": ds.UGRD10m, "v10": ds.VGRD10m}
    else:
        raise ValueError(f"unknown source {source!r}")

    means = xr.Dataset({key: da.mean("time") for key, da in fields.items()})
    return means.compute(scheduler="threads", num_workers=THREADS)


def _cache_path(tag: str, source: str) -> Path:
    # The window is part of the name: changing START/END must miss the cache
    # rather than quietly reuse means taken over a different period.
    return CACHE_DIR / f"{tag}_{source}_{START}_{END}.nc"


def cached_source(
    path: Path | str, tag: str, source: str, is_reference: bool, refresh: bool = False
) -> xr.Dataset:
    """``compute_source`` memoised on disk, so an interrupted run resumes."""
    out = _cache_path(tag, source)
    if out.exists() and not refresh:
        print(f"loading plot data from {out}", flush=True)
        return xr.open_dataset(out).load()
    print(f"computing {source} time mean: {tag}", flush=True)
    means = compute_source(path, source, is_reference)
    CACHE_DIR.mkdir(exist_ok=True)
    means.to_netcdf(out)
    print(f"wrote {out}", flush=True)
    return means


def _seed_precip_cache_from_sibling() -> None:
    """Reuse ``precip_time_means.nc`` if the sibling script already built it.

    It holds the reference and per-member time-mean precipitation computed the
    same way -- recomputing it would cost a full extra pass over all six stores
    for a field we already have. It is only reusable if it was taken over this
    script's window, which it records in its own attributes; on any other window
    the shortcut is skipped and precipitation is averaged like the other fields.
    """
    if not PRECIP_CACHE.exists():
        return
    cached = xr.open_dataset(PRECIP_CACHE).load()
    if (cached.attrs.get("start"), cached.attrs.get("end")) != (START, END):
        print(
            f"{PRECIP_CACHE.name} covers "
            f"{cached.attrs.get('start')}-{cached.attrs.get('end')}, not {START}-{END}; "
            "computing precipitation means instead of reusing it",
            flush=True,
        )
        return
    CACHE_DIR.mkdir(exist_ok=True)
    entries = [("reference", cached["reference"])]
    entries += [
        (str(tag), cached["members"].sel(member=tag)) for tag in cached["members"].member.values
    ]
    for tag, da in entries:
        out = _cache_path(tag, "precip")
        if not out.exists():
            xr.Dataset({"precip": da.drop_vars("member", errors="ignore")}).to_netcdf(out)
            print(f"seeded {out} from {PRECIP_CACHE.name}", flush=True)


def _task(args: tuple[str, str, str, bool, bool]) -> str:
    path, tag, source, is_reference, refresh = args
    cached_source(path, tag, source, is_reference, refresh)
    return f"{tag}:{source}"


def member_stores(refresh: bool = False) -> pd.DataFrame:
    """Tag and path of every ACE2S member store, one row each, in tag order."""

    def compute() -> pd.DataFrame:
        # ic???? rather than ic*: the root also holds output_6hourly_predictions_
        # ic0000_2023-onward.zarr, a 1-year continuation of ic0000, not a member.
        paths = sorted(
            Path(COARSE_ACE2S_DATASET_ROOT).glob("output_6hourly_predictions_ic????.zarr")
        )
        return pd.DataFrame(
            {"tag": [p.stem.split("_")[-1] for p in paths], "path": [str(p) for p in paths]}
        )

    return cached_frame(MEMBERS_CACHE, compute, refresh, dtype=str)


def gather_time_means(workers: int, refresh: bool = False) -> tuple[xr.Dataset, xr.Dataset]:
    """Reference and per-member time means for every field, as two Datasets.

    Members are stacked along a ``member`` dimension labelled by initial-condition
    tag (ic0000, ...), so the arrays are self-describing.
    """
    members_table = member_stores(refresh)
    if not refresh:
        _seed_precip_cache_from_sibling()

    sources = tuple(FIELDS_BY_SOURCE)
    tasks = [
        (COARSENED_XSHIELD_REFERENCE_DATASET, "reference", s, True, refresh) for s in sources
    ]
    tasks += [
        (path, tag, s, False, refresh)
        for tag, path in zip(members_table.tag, members_table.path)
        for s in sources
    ]
    pending = [t for t in tasks if refresh or not _cache_path(t[1], t[2]).exists()]

    if pending:
        # The reduction is latency-bound, so the passes overlap in separate
        # processes; each still uses its own thread pool internally.
        print(f"{len(pending)} of {len(tasks)} (source, store) means to compute", flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            for done in pool.map(_task, pending):
                print(f"done {done}", flush=True)

    def collect(path: Path | str, tag: str, is_reference: bool) -> xr.Dataset:
        parts = [cached_source(path, tag, s, is_reference) for s in sources]
        # Caches written before a field was dropped still carry it; keeping only
        # the plotted fields makes old and new caches merge to the same set.
        return xr.merge(parts)[[f.key for f in FIELDS]]

    reference = collect(COARSENED_XSHIELD_REFERENCE_DATASET, "reference", True)
    labels = list(members_table.tag)
    members = xr.concat(
        [collect(path, tag, False) for path, tag in zip(members_table.path, labels)],
        dim="member",
    ).assign_coords(member=labels)
    return reference, members


def area_weighted(da: xr.DataArray) -> float:
    """Area mean with cos(latitude) weights."""
    return float(da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon")))


def plot_map(ax, field: xr.DataArray, title: str, **kwargs) -> object:
    ax.set_global()
    mesh = ax.pcolormesh(
        field.lon,
        field.lat,
        field,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=0,
        **kwargs,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black", zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(title, fontsize=12)
    return mesh


def _mean_limits(field: Field, reference: xr.DataArray, prediction: xr.DataArray) -> tuple:
    """Colour limits shared by the two climatology columns of one row.

    Returned as (pcolormesh kwargs, colorbar extend). The two columns must share
    them or the eye cannot compare the panels, so both fields set the range.
    """
    both = np.concatenate([reference.values.ravel(), prediction.values.ravel()])
    hi = float(field.mean_percentiles[1])
    cmap = "turbo" if field.key in TURBO_FIELDS else None

    if field.scale in ("signed", "anomaly"):
        # Diverging: two hues either side of a neutral midpoint. Zero for a
        # signed field; the reference global mean for an anomaly field, so the
        # midpoint marks "average pressure" rather than an arbitrary value.
        center = 0.0 if field.scale == "signed" else area_weighted(reference)
        half = float(np.percentile(np.abs(both - center), hi))
        return (
            dict(cmap=cmap or DIVERGING_CMAP, vmin=center - half, vmax=center + half),
            "both",
        )

    lo, vmax = np.percentile(both, [float(field.mean_percentiles[0]), hi])
    # Precipitation and wind speed are bounded below by zero and come close to
    # it, so the ramp starts there rather than at a percentile just above it.
    vmin = 0.0 if 0 < float(lo) < 0.05 * float(vmax) else float(lo)
    return (
        dict(cmap=cmap or SEQUENTIAL_CMAP, vmin=vmin, vmax=float(vmax)),
        "max" if vmin == 0.0 else "both",
    )


def plot_multifield(reference: xr.Dataset, members: xr.Dataset) -> Path:
    """One row per field: reference and prediction on a shared scale, plus their bias."""
    proj = ccrs.PlateCarree(central_longitude=180)
    nrows = len(FIELDS)
    height = 2.95 * nrows
    fig = plt.figure(figsize=(16.5, height))
    # Explicit margins: the per-row colorbars and the column headers are placed
    # by hand, so the layout must not be resized mid-construction. The header
    # band is a fixed height in inches, not a fraction, so adding or removing a
    # row moves the maps without also re-spacing the titles above them.
    top = 1 - 1.115 / height
    header_y = 1 - 0.78 / height
    gs = fig.add_gridspec(
        nrows, 3, hspace=0.36, wspace=0.16, left=0.055, right=0.93, bottom=0.015, top=top
    )

    for row, field in enumerate(FIELDS):
        ref = reference[field.key]
        prediction = members[field.key].mean("member")
        bias = prediction - ref

        mean_kwargs, mean_extend = _mean_limits(field, ref, prediction)
        # Symmetric, so zero bias sits on the neutral midpoint of the ramp.
        bias_vmax = float(np.percentile(np.abs(bias.values), 99.9))

        ref_mean = area_weighted(ref)
        pred_mean = area_weighted(prediction)
        rmsb = np.sqrt(area_weighted(bias**2))

        axes = [fig.add_subplot(gs[row, col], projection=proj) for col in range(3)]
        if row == 0:
            top_row = axes
        mesh = plot_map(axes[0], ref, f"{ref_mean:.2f} {field.units} global mean", **mean_kwargs)
        plot_map(
            axes[1], prediction, f"{pred_mean:.2f} {field.units} global mean", **mean_kwargs
        )
        bias_mesh = plot_map(
            axes[2],
            bias,
            f"Global: mean bias {pred_mean - ref_mean:+.2f} {field.units}, "
            f"RMSB {rmsb:.2f} {field.units}",
            cmap=DIVERGING_CMAP,
            vmin=-bias_vmax,
            vmax=bias_vmax,
        )

        # Row label on the left margin: naming the field once per row beats
        # repeating it across that row's three panel titles. The units are on
        # the colorbars, where the numbers they apply to are.
        axes[0].text(
            -0.045,
            0.5,
            field.label,
            transform=axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

        # One bar for the two climatology columns, one for the bias column. Both
        # are labelled with the row's units: the rows do not share a scale, so a
        # bar read without them says nothing, and a reader tracing a number back
        # from a panel should not have to find the row label to know what it is.
        fig.colorbar(
            mesh, ax=axes[:2], orientation="vertical", fraction=0.02, pad=0.02,
            extend=mean_extend, label=field.units,
        )
        fig.colorbar(
            bias_mesh, ax=axes[2], orientation="vertical", fraction=0.04, pad=0.03,
            extend="both", label=f"Bias ({field.units})",
        )

    headers = (
        REFERENCE_LABEL,
        f"{PREDICTION_LABEL} ensemble mean ({members.sizes['member']} members)",
        f"{PREDICTION_LABEL} minus {REFERENCE_LABEL}",
    )
    # Above the first-row titles, so the
    # columns are named once rather than in all twelve panels. Placed after a
    # layout pass: a map axes is narrower than its cell once cartopy has applied
    # the 2:1 aspect, so centring on the cell would leave every header off its
    # column by half that difference.
    fig.draw_without_rendering()
    for ax, header in zip(top_row, headers):
        box = ax.get_position()
        fig.text(
            box.x0 + box.width / 2, header_y, header,
            ha="center", va="bottom", fontsize=14,
        )

    out_path = MAIN_FIGURES_DIR / "global_coarse_res_multifield_bias.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="how many (source, store) passes to run concurrently",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompute and overwrite the cached member list and time means",
    )
    args = parser.parse_args()

    reference, members = gather_time_means(args.workers, args.refresh)
    print(f"Wrote {plot_multifield(reference, members)}")


if __name__ == "__main__":
    main()
