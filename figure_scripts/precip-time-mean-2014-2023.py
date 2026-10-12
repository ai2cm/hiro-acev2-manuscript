#!/usr/bin/env python
"""2014-2023 time-mean precipitation over the western United States at both
resolutions.

Two rows on shared colorbars. Top: 100 km -- 1 degree X-SHiELD, the coarse
ACE2S run that drives the downscale (ic0000, the same initial condition as the
3 km rollout) and its bias. Bottom: the same period and box at 3 km --
X-SHiELD, HiRO-ACE and the HiRO-ACE bias.

Annual and seasonal (JFM / AMJ / JAS / OND) means are cached under
time_mean_cache/, so re-plotting never re-reads the 3 km stores. Each store is
read exactly once per run: the four seasonal means are built in a single dask
graph and the annual mean is their count-weighted combination. The 3 km caches
keep only what a figure draws -- the requested period, cropped to the displayed
longitudes -- so another period recomputes them; the 1 degree caches keep the
whole box and every period.

    python3 precip-time-mean-2014-2023.py                 # annual figure
    python3 precip-time-mean-2014-2023.py --mask-states   # clipped to WA/OR/CA
    python3 precip-time-mean-2014-2023.py --period JAS    # one season
    python3 precip-time-mean-2014-2023.py --workers 160

Reading the 10 years of 3 km X-SHiELD takes roughly half an hour at 144
workers; the 1 degree stores take a couple of minutes. Everything after that
comes out of the cache.

Note: HiRO-ACE is a free run initialised in 2014, so it is not synchronised
with X-SHiELD. Its bias is a climate bias -- a difference of time means -- not
a paired forecast error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import shapely
import xarray as xr
from cartopy.io import shapereader

from plot_cache import CACHE_DIR as PLOT_DATA_CACHE, SUPPLEMENTS_DIR

HERE = Path(__file__).parent
CACHE_DIR = PLOT_DATA_CACHE / "time_mean_cache"
OUT_DIR = SUPPLEMENTS_DIR

# PRATEsfc is kg/m**2/s; 86400 converts to the conventional mm/day.
SECONDS_PER_DAY = 86400.0
MEAN_CMAP = "Blues"
BIAS_CMAP = "RdBu_r"
# Fixed across every panel, period and rerun, so figures stay comparable
# instead of each one setting its own scale from its own percentiles.
MEAN_VMAX = 10.0
BIAS_VMAX = 2.0

START, STOP = "2014-01-01", "2023-12-31"
PERIOD_TAG = "2014-2023"
# The HiRO-ACE WUS run covers this box on the X-SHiELD native 3 km grid, so
# slicing X-SHiELD to the same box lands on exactly the same 512x512 cells and
# the bias needs no regridding. The 1 degree stores are sliced to the same box.
LAT0, LAT1 = 32.92, 48.85
LON0, LON1 = 235.0, 251.0
# Displayed -- and averaged -- longitudes. The eastern 2 degrees of the box are
# dropped so the panels cover the same window as the coarse ACE2S bias maps.
# The caches always hold the full box, so this crop never forces a recompute.
PLOT_LON0, PLOT_LON1 = 235.0, 249.0
STATES = {"California", "Oregon", "Washington"}

SEASONS = {"JFM": (1, 2, 3), "AMJ": (4, 5, 6),
           "JAS": (7, 8, 9), "OND": (10, 11, 12)}
PERIODS = ["annual", *SEASONS]

ROOT = "/climate-default/2026-08-03-hiro-ace-v2-outputs"
SOURCES = {
    "xshield": dict(
        label="X-SHiELD 3 km",
        path="/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr",
        lat="grid_yt", lon="grid_xt", ensemble=False),
    "hiro_ace": dict(
        label="HiRO-ACEv2 3 km",
        path=f"{ROOT}/2026-09-15-hiro-ace2s-downscale-WUS-10yr-ic0000/"
             "WUS_ace2s_downscaled_2014-2023.zarr",
        lat="latitude", lon="longitude", ensemble=True),
    "xshield_100km": dict(
        label="X-SHiELD 100 km",
        path="/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr",
        lat="grid_yt", lon="grid_xt", ensemble=False),
    "ace2s_100km": dict(
        label="ACE2S 100 km",
        path=f"{ROOT}/2026-08-03-coarse-ace2s-10yr-inference/"
             "output_6hourly_predictions_ic0000.zarr",
        lat="lat", lon="lon", ensemble=False),
}

# (reference, prediction) per figure row, fine beneath coarse.
ROWS = [("xshield_100km", "ace2s_100km"), ("xshield", "hiro_ace")]


# ---------------------------------------------------------------- time means
def _compute_source(key: str, workers: int) -> xr.Dataset:
    """Seasonal + annual mean PRATEsfc (mm/day) over the box, for one store."""
    src = SOURCES[key]
    ds = xr.open_zarr(src["path"])
    da = ds.PRATEsfc
    if src["ensemble"]:
        da = da.isel(ensemble=0)
    da = da.sel(time=slice(START, STOP)).sel(
        {src["lat"]: slice(LAT0, LAT1), src["lon"]: slice(LON0, LON1)})
    da = da.rename({src["lat"]: "lat", src["lon"]: "lon"})

    months = da.time.dt.month
    parts, counts = {}, {}
    for name, mons in SEASONS.items():
        sub = da.isel(time=np.isin(months.values, mons))
        counts[name] = int(sub.sizes["time"])
        parts[name] = sub.mean("time")

    # One compute for all four: every timestep is decompressed exactly once.
    with dask.config.set(scheduler="threads", num_workers=workers):
        values = dask.compute(*[parts[s] for s in SEASONS])

    out = {s: v * SECONDS_PER_DAY for s, v in zip(SEASONS, values)}
    total = sum(counts.values())
    annual = sum(out[s] * counts[s] for s in SEASONS) / total
    out["annual"] = annual

    result = xr.Dataset(out)
    for name in PERIODS:
        result[name].attrs = {"units": "mm/day", "long_name":
                              f"{PERIOD_TAG} {name} mean precipitation"}
    result.attrs = {
        "source": src["path"], "label": src["label"],
        "period": f"{START} to {STOP}",
        "region": f"lat {LAT0}-{LAT1}, lon {LON0}-{LON1}",
        "n_timesteps": total,
        "n_per_season": " ".join(f"{s}={counts[s]}" for s in SEASONS),
    }
    return result


# The fine row's stores: their caches hold one period over the displayed box.
FINE_SOURCES = set(ROWS[-1])


def _cache_path(key: str, period: str) -> Path:
    if key in FINE_SOURCES:
        return CACHE_DIR / f"precip_{PERIOD_TAG}_westus_{key}_{period}_plotted.nc"
    return CACHE_DIR / f"precip_{PERIOD_TAG}_westus_{key}.nc"


def plotted_subset(result: xr.Dataset, period: str) -> xr.Dataset:
    """``period`` alone, over the displayed longitudes: all a figure draws."""
    sub = result[[period]].sel(lon=slice(PLOT_LON0, PLOT_LON1))
    sub.attrs = {**result.attrs,
                 "region": f"lat {LAT0}-{LAT1}, lon {PLOT_LON0}-{PLOT_LON1}"}
    return sub


def time_means(key: str, workers: int, refresh: bool = False,
               period: str = "annual") -> xr.Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(key, period)
    if cache.exists() and not refresh:
        print(f"  {key}: loading plot data from {cache}")
        return xr.open_dataset(cache)
    print(f"  {key}: computing (reads the store once) ...", flush=True)
    result = _compute_source(key, workers)
    if key in FINE_SOURCES:
        result = plotted_subset(result, period)
    result.to_netcdf(cache)
    print(f"  {key}: wrote {cache.name}  ({result.attrs['n_per_season']})")
    return result


# ------------------------------------------------------------------- masking
def state_mask(like: xr.DataArray) -> xr.DataArray:
    """True where a grid cell centre falls inside WA, OR or CA."""
    shp = shapereader.natural_earth(resolution="50m", category="cultural",
                                    name="admin_1_states_provinces")
    geoms = [r.geometry for r in shapereader.Reader(shp).records()
             if r.attributes.get("name") in STATES
             and r.attributes.get("admin") == "United States of America"]
    union = shapely.union_all(geoms)
    lon2d, lat2d = np.meshgrid(((like.lon.values + 180) % 360) - 180,
                               like.lat.values)
    inside = shapely.contains_xy(union, lon2d, lat2d)
    return xr.DataArray(inside, dims=("lat", "lon"),
                        coords={"lat": like.lat, "lon": like.lon})


def area_weighted(da: xr.DataArray, mask: xr.DataArray) -> float:
    """cos(latitude)-weighted mean over the masked cells."""
    weights = np.cos(np.deg2rad(da.lat)) * mask
    return float(da.weighted(weights).mean(("lat", "lon")))


# -------------------------------------------------------------------- figure
# Label spacing in degrees. The graticule itself stays on cartopy's own
# 2-degree choice; only every second parallel and meridian is annotated, which
# keeps the labels from running into each other around the narrow panels.
LON_LABEL_STEP = 4
LAT_LABEL_STEP = 4


def _plot_lon_bounds() -> tuple[float, float]:
    """The displayed longitudes on the -180..180 convention."""
    return (((PLOT_LON0 + 180) % 360) - 180, ((PLOT_LON1 + 180) % 360) - 180)


def _ticks(lo: float, hi: float, step: int) -> np.ndarray:
    """Multiples of ``step`` inside [lo, hi]."""
    return np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)


def draw(ax, field, *, cmap, vmin, vmax, title,
         left_labels=False, bottom_labels=False):
    ax.set_extent((((PLOT_LON0 + 180) % 360) - 180,
                   ((PLOT_LON1 + 180) % 360) - 180,
                   LAT0, LAT1), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.94", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef3f8", zorder=0)
    mesh = ax.pcolormesh(field.lon, field.lat, field.values, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="nearest",
                         transform=ccrs.PlateCarree(), zorder=2)
    ax.coastlines(lw=0.6, color="0.35", zorder=5)
    ax.add_feature(cfeature.BORDERS, lw=0.5, edgecolor="0.45", zorder=5)
    ax.add_feature(cfeature.STATES, lw=0.5, edgecolor="0.45", zorder=5)
    ax.set_title(title, fontsize=10.5, color="0.12")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    # Labels come from a second, invisible graticule so the longitudes can be
    # annotated every other meridian without thinning the grid itself. Ticks
    # only on the outside edges: every panel shares the same box, so repeating
    # the labels six times would just add clutter.
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0,
                      x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator(_ticks(*_plot_lon_bounds(), LON_LABEL_STEP))
    gl.ylocator = mticker.FixedLocator(_ticks(LAT0, LAT1, LAT_LABEL_STEP))
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style = {"size": 8, "color": "0.3"}
    gl.ylabel_style = {"size": 8, "color": "0.3"}
    return mesh


def difference(prediction: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """prediction - reference on the reference's coordinates.

    The two members of a row sit on the same grid, but the coordinates come
    from different stores and can disagree in the last float32 bit, which would
    turn xarray's automatic alignment into an inner join on nothing. Reusing
    the reference's coordinates keeps the subtraction elementwise.
    """
    if prediction.shape != reference.shape:
        raise ValueError(f"grid mismatch: {prediction.shape} vs {reference.shape}")
    return xr.DataArray(prediction.values - reference.values,
                        dims=reference.dims, coords=reference.coords)


def make_figure(period: str, workers: int, refresh: bool, out: Path,
                mask_states: bool = False):
    print(f"Time means ({period}):")
    means = {k: time_means(k, workers, refresh, period) for k in SOURCES}
    # The caches cover the whole box; the panels show the cropped longitudes,
    # and every statistic below is taken over exactly what is shown.
    fields = {k: means[k][period].sel(lon=slice(PLOT_LON0, PLOT_LON1))
              for k in SOURCES}

    # One mask per grid: the rows are at different resolutions.
    masks, shown = {}, {}
    for ref, pred in ROWS:
        if mask_states:
            mask = state_mask(fields[ref])
        else:
            mask = xr.ones_like(fields[ref], dtype=bool)
        for k in (ref, pred):
            masks[k] = mask
            shown[k] = fields[k].where(mask) if mask_states else fields[k]
    domain = "WA+OR+CA" if mask_states else "domain"

    biases = {ref: difference(shown[pred], shown[ref]) for ref, pred in ROWS}

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(13.5, 11.0),
                             subplot_kw=dict(projection=ccrs.PlateCarree()))
    last_row = len(ROWS) - 1
    for r, (row, (ref, pred)) in enumerate(zip(axes, ROWS)):
        ticks = dict(bottom_labels=(r == last_row))
        for c, (ax, key) in enumerate(zip(row[:2], (ref, pred))):
            m = area_weighted(fields[key], masks[key])
            mesh = draw(ax, shown[key], cmap=MEAN_CMAP, vmin=0, vmax=MEAN_VMAX,
                        title=f"{SOURCES[key]['label']}\n"
                              f"{m:.2f} mm/day {domain} mean",
                        left_labels=(c == 0), **ticks)
        bias = biases[ref]
        mb = area_weighted(difference(fields[pred], fields[ref]), masks[ref])
        rmsb = np.sqrt(area_weighted(
            difference(fields[pred], fields[ref]) ** 2, masks[ref]))
        bmesh = draw(row[2], bias, cmap=BIAS_CMAP,
                     vmin=-BIAS_VMAX, vmax=BIAS_VMAX,
                     title=f"{SOURCES[pred]['label']} - {SOURCES[ref]['label']}\n"
                           f"mean bias {mb:+.2f}, RMSB {rmsb:.2f} mm/day",
                     **ticks)

    cb1 = fig.colorbar(mesh, ax=axes[:, :2], location="bottom", shrink=0.62,
                       pad=0.04, aspect=40, extend="max")
    cb1.set_label("Mean precipitation (mm/day)", fontsize=10)
    cb2 = fig.colorbar(bmesh, ax=axes[:, 2], location="bottom", shrink=0.9,
                       pad=0.04, aspect=20, extend="both")
    cb2.set_label("Bias (mm/day)", fontsize=10)

    # The default run is the plain title; the flags that change what is shown
    # say so, so a seasonal or clipped figure is never mislabelled.
    title = f"{PERIOD_TAG} time-mean surface precipitation rate"
    if period != "annual":
        title += f" ({period})"
    if mask_states:
        title += " -- WA / OR / CA"
    fig.suptitle(title, fontsize=14, y=0.94)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="annual", choices=PERIODS)
    ap.add_argument("--workers", type=int, default=144)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the cached time means")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mask-states", action="store_true",
                    help="clip the panels to WA/OR/CA instead of the full box")
    a = ap.parse_args()
    out = a.out or (OUT_DIR / (f"precip-time-mean-{PERIOD_TAG}.png"
                               if a.period == "annual" else
                               f"precip-time-mean-{PERIOD_TAG}-{a.period}.png"))
    make_figure(a.period, a.workers, a.refresh, out, a.mask_states)


if __name__ == "__main__":
    main()
