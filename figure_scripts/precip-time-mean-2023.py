#!/usr/bin/env python
"""2023 time-mean precipitation over the western United States (WA / OR / CA)
for 3 km X-SHiELD, the perfect prediction, and HiRO-ACE.

Row 1: time-mean precipitation. Row 2: bias relative to X-SHiELD.

Annual and seasonal (JFM / AMJ / JAS / OND) means are cached under
time_mean_cache/, so re-plotting never re-reads the 3 km stores. Each store is
read exactly once per run: the four seasonal means are built in a single dask
graph and the annual mean is their count-weighted combination. Only the
requested period is cached, so another period recomputes it.

    python3 precip-time-mean-2023.py                 # annual figure, full box
    python3 precip-time-mean-2023.py --mask-states   # clipped to WA/OR/CA
    python3 precip-time-mean-2023.py --period JAS    # one season
    python3 precip-time-mean-2023.py --workers 64

Note: HiRO-ACE is a free run initialised in 2014, so its "2023" is not
synchronised with X-SHiELD's. Its bias is a climate bias -- a difference of
time means -- not a paired forecast error like the perfect prediction's.
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
REFERENCE = "xshield"

YEAR = "2023"
# Box around WA/OR/CA with a margin; the panels are masked to the three states,
# but the cache keeps the whole box so it can be reused for a wider view.
LAT0, LAT1 = 32.0, 49.5
LON0, LON1 = 234.5, 246.0
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
    "perfect_pred": dict(
        label="Perfect prediction",
        path=f"{ROOT}/2026-08-03-hiro-perfect-pred-global-2023/global_perfect_prediction.zarr",
        lat="latitude", lon="longitude", ensemble=True),
    "hiro_ace": dict(
        label="HiRO-ACEv2",
        path=f"{ROOT}/2026-08-05-hiro-ace2s-downscale-global-2023-ic0000/"
             "global_ace2s_downscaled_2023.zarr",
        lat="latitude", lon="longitude", ensemble=True),
}


# ---------------------------------------------------------------- time means
def _compute_source(key: str, workers: int) -> xr.Dataset:
    """Seasonal + annual mean PRATEsfc (mm/day) over the box, for one store."""
    src = SOURCES[key]
    ds = xr.open_zarr(src["path"])
    da = ds.PRATEsfc
    if src["ensemble"]:
        da = da.isel(ensemble=0)
    da = da.sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31")).sel(
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
                              f"{YEAR} {name} mean precipitation"}
    result.attrs = {
        "source": src["path"], "label": src["label"], "year": YEAR,
        "region": f"lat {LAT0}-{LAT1}, lon {LON0}-{LON1}",
        "n_timesteps": total,
        "n_per_season": " ".join(f"{s}={counts[s]}" for s in SEASONS),
    }
    return result


def time_means(key: str, workers: int, refresh: bool = False,
               period: str = "annual") -> xr.Dataset:
    # Only the period drawn is kept: the panels show exactly the box, so the
    # cache needs no further crop, but another period recomputes it.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"precip_{YEAR}_westus_{key}_{period}.nc"
    if cache.exists() and not refresh:
        print(f"  {key}: loading plot data from {cache}")
        return xr.open_dataset(cache)
    print(f"  {key}: computing (reads the 3 km store once) ...", flush=True)
    result = _compute_source(key, workers)[[period]]
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
def draw(ax, field, *, cmap, vmin, vmax, title):
    ax.set_extent((((LON0 + 180) % 360) - 180, ((LON1 + 180) % 360) - 180,
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
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
    return mesh


def make_figure(period: str, workers: int, refresh: bool, out: Path,
                mask_states: bool = False):
    print(f"Time means ({period}):")
    means = {k: time_means(k, workers, refresh, period) for k in SOURCES}
    fields = {k: means[k][period] for k in SOURCES}
    # The cache always covers the whole box; the WA/OR/CA mask is display-only.
    if mask_states:
        mask = state_mask(fields[REFERENCE])
        shown = {k: v.where(mask) for k, v in fields.items()}
        domain = "WA+OR+CA"
    else:
        mask = xr.ones_like(fields[REFERENCE], dtype=bool)
        shown = dict(fields)
        domain = "domain"

    ref = shown[REFERENCE]
    others = [k for k in SOURCES if k != REFERENCE]
    vmax = float(np.nanpercentile(
        np.concatenate([v.values[np.isfinite(v.values)] for v in shown.values()]), 99.5))
    biases = {k: shown[k] - ref for k in others}
    bias_vmax = float(np.nanpercentile(
        np.abs(np.concatenate([b.values[np.isfinite(b.values)]
                               for b in biases.values()])), 99.0))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.6),
                             subplot_kw=dict(projection=ccrs.PlateCarree()))
    for ax, key in zip(axes[0], SOURCES):
        m = area_weighted(fields[key], mask)
        mesh = draw(ax, shown[key], cmap=MEAN_CMAP, vmin=0, vmax=vmax,
                    title=f"{SOURCES[key]['label']}\n{m:.2f} mm/day {domain} mean")
    axes[1, 0].set_visible(False)
    bmesh = None
    for ax, key in zip(axes[1, 1:], others):
        b = biases[key]
        mb = area_weighted(fields[key] - fields[REFERENCE], mask)
        rmsb = np.sqrt(area_weighted((fields[key] - fields[REFERENCE]) ** 2, mask))
        bmesh = draw(ax, b, cmap=BIAS_CMAP, vmin=-bias_vmax, vmax=bias_vmax,
                     title=f"{SOURCES[key]['label']} - X-SHiELD\n"
                           f"mean bias {mb:+.2f}, RMSB {rmsb:.2f} mm/day")

    cb1 = fig.colorbar(mesh, ax=axes[0], shrink=0.82, pad=0.02, extend="max")
    cb1.set_label("Mean precipitation (mm/day)", fontsize=10)
    cb2 = fig.colorbar(bmesh, ax=axes[1], shrink=0.82, pad=0.02, extend="both")
    cb2.set_label("Bias (mm/day)", fontsize=10)

    label = "annual" if period == "annual" else period
    extent = "WA / OR / CA" if mask_states else "full domain"
    fig.suptitle(f"{YEAR} {label} mean precipitation, western United States "
                 f"({extent})", fontsize=14, y=0.94)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="annual", choices=PERIODS)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the cached time means")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mask-states", action="store_true",
                    help="clip the panels to WA/OR/CA instead of the full box")
    a = ap.parse_args()
    out = a.out or (OUT_DIR / (f"precip-time-mean-{YEAR}.png" if a.period == "annual"
                               else f"precip-time-mean-{YEAR}-{a.period}.png"))
    make_figure(a.period, a.workers, a.refresh, out, a.mask_states)


if __name__ == "__main__":
    main()
