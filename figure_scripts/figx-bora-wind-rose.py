#!/usr/bin/env python
"""
Wind roses of 2014-2023 10m wind at Senj, the classic Bora reference site,
from the 3km X-SHiELD reference, perfect prediction (X-SHiELD 1deg downscaled)
and HiRO-ACE (ACE2S downscaled over the Croatia/Bora region). One polar panel
per dataset; petals are the frequency of wind blowing *from* each of 16
sectors, stacked by speed bin.

The per-site u/v time series are the expensive part (every X-SHiELD timestep
is a full ~57MB chunk read), so they're cached to CACHE_PATH; the rose
histograms themselves are recomputed from the cache on every run. X-SHiELD is
extracted a year at a time into XSHIELD_PART_DIR so an interrupted run
resumes where it left off. Delete the cache files (or pass --refresh) to
re-extract.

Unlike the parent directory's version (whose cache also holds Trieste for
ad_hoc/figx-bora-wind-rose-3panel.py), only the drawn site is extracted and
cached, and the per-year X-SHiELD parts are deleted once the merged cache is
written. All series share the timestamps common to every dataset in
CACHE_DATASETS.
"""

import argparse
from pathlib import Path

import dask
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import xarray as xr

from plot_cache import CACHE_DIR as PLOT_DATA_CACHE_DIR, MAIN_FIGURES_DIR

BORA_REGION_DIR = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-10-01-downscale-bora-region"
)
HIRO_ACE_ZARR = f"{BORA_REGION_DIR}/croatia_ace2s_downscaled_2014-2023_ic0000.zarr"
PERFECT_PRED_ZARR = f"{BORA_REGION_DIR}/croatia_perfect_pred_downscaled_2014-2023_ic0000.zarr"
XSHIELD_ZARR = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr"

START, STOP = "2014-01-01", "2023-12-31 23:59"
YEARS = range(2014, 2024)

FIG_DIR = Path(__file__).parent
OUT_DIR = MAIN_FIGURES_DIR
OUT_PATH = OUT_DIR / "hiro_ace_wind_rose.png"
CACHE_DIR = PLOT_DATA_CACHE_DIR / "wind_rose_cache"
CACHE_PATH = CACHE_DIR / "senj_wind_2014-2023.nc"
XSHIELD_PART_DIR = CACHE_DIR / "xshield_2014-2023_parts"

# Same site coordinates as figx-downslope-wind-events.py's BORA_CITIES.
SITES = {
    "Senj": (44.99, 14.90),
}

# label -> (zarr path, u name, v name, lat name, lon name)
CACHE_DATASETS = {
    "X-SHiELD": (XSHIELD_ZARR, "eastward_wind_at_ten_meters", "northward_wind_at_ten_meters", "grid_yt", "grid_xt"),
    "Perfect prediction": (PERFECT_PRED_ZARR, "UGRD10m", "VGRD10m", "latitude", "longitude"),
    "HiRO-ACE": (HIRO_ACE_ZARR, "UGRD10m", "VGRD10m", "latitude", "longitude"),
}
# Datasets drawn in this (main) figure, in column order, and its site rows.
DATASETS = ["X-SHiELD", "Perfect prediction", "HiRO-ACE"]
# Panel titles, where they differ from the cache keys above (which stay as-is
# so the existing cache remains readable).
DISPLAY_NAMES = {"HiRO-ACE": "HiRO-ACEv2"}
PLOT_SITES = ["Senj"]

N_SECTORS = 16
SPEED_BINS = [0, 3, 6, 10, 15, 20, np.inf]  # m/s
# Reversed viridis samples (light yellow -> dark purple = calm -> strong):
# still monotonic in lightness, but the hue shift at every step keeps the
# upper speed bins distinct from each other, which a single-hue ramp doesn't.
SPEED_COLORS = ["#cae11f", "#58c765", "#1e9c89", "#2d718e", "#424086", "#440154"]
DASK_WORKERS = 64


def to_lon360(lon):
    return lon + 360 if lon < 0 else lon


def open_period(path: str) -> xr.Dataset:
    ds = xr.open_zarr(path)
    if "ensemble" in ds.dims:
        ds = ds.isel(ensemble=0)
    return ds.sel(time=slice(START, STOP))


def site_points(name: str, ds: xr.Dataset) -> dict:
    _, u_name, v_name, lat_name, lon_name = CACHE_DATASETS[name]
    out = {}
    for site, (lat, lon) in SITES.items():
        point = ds[[u_name, v_name]].sel(
            {lat_name: lat, lon_name: to_lon360(lon)}, method="nearest"
        )
        print(
            f"{name} {site}: nearest cell "
            f"({float(point[lat_name]):.3f}, {float(point[lon_name]):.3f})"
        )
        out[f"{name}_{site}_u"] = point[u_name].drop_vars([lat_name, lon_name])
        out[f"{name}_{site}_v"] = point[v_name].drop_vars([lat_name, lon_name])
    return out


def compute(series: dict) -> xr.Dataset:
    with dask.config.set(scheduler="threads", num_workers=DASK_WORKERS):
        (computed,) = dask.compute(series)
    return xr.Dataset({k: v.load() for k, v in computed.items()})


def extract_xshield(ds: xr.Dataset, refresh: bool = False) -> xr.Dataset:
    """X-SHiELD site series, one cached part per year. All sites share one
    X-SHiELD chunk per timestep, so computing them in a single call lets dask
    read each chunk once for every series."""
    XSHIELD_PART_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in YEARS:
        part_path = XSHIELD_PART_DIR / f"{year}.nc"
        if refresh or not part_path.exists():
            year_ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31 23:59"))
            print(f"Extracting X-SHiELD {year} ({year_ds.sizes['time']} timesteps)")
            compute(site_points("X-SHiELD", year_ds)).to_netcdf(part_path)
        parts.append(xr.open_dataset(
            part_path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
        ).load())
    return xr.concat(parts, dim="time")


def extract_site_winds(refresh: bool = False) -> xr.Dataset:
    """u/v at the nearest grid cell to each site, on the timestamps all
    datasets share in 2014-2023."""
    opened = {name: open_period(spec[0]) for name, spec in CACHE_DATASETS.items()}
    common = sorted(set.intersection(*(set(ds.time.values) for ds in opened.values())))
    print(f"{len(common)} common 2014-2023 timesteps")

    out = [extract_xshield(opened["X-SHiELD"].sel(time=common), refresh)]
    for name, ds in opened.items():
        if name != "X-SHiELD":
            out.append(compute(site_points(name, ds.sel(time=common))))
    return xr.merge(out, join="exact")


def load_site_winds(refresh: bool = False) -> xr.Dataset:
    if CACHE_PATH.exists() and not refresh:
        print(f"Loading cached winds: {CACHE_PATH}")
        return xr.open_dataset(CACHE_PATH, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    winds = extract_site_winds(refresh)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    winds.to_netcdf(CACHE_PATH)
    print(f"Saved cache: {CACHE_PATH}")
    # The parts only exist to make an interrupted extraction resumable; once
    # the merged cache is written they duplicate it.
    for part_path in XSHIELD_PART_DIR.glob("*.nc"):
        part_path.unlink()
    XSHIELD_PART_DIR.rmdir()
    return winds


def wind_rose_frequencies(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(n_speed_bins, N_SECTORS) percent of timesteps, by meteorological
    (blowing-from) direction sector centered on N, NNE, ..."""
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    width = 360.0 / N_SECTORS
    sector = ((direction + width / 2) // width).astype(int) % N_SECTORS
    speed_bin = np.digitize(speed, SPEED_BINS[1:-1])
    counts = np.zeros((len(SPEED_BINS) - 1, N_SECTORS))
    np.add.at(counts, (speed_bin, sector), 1)
    return 100.0 * counts / speed.size


def plot_rose(ax, freqs: np.ndarray, rmax: float) -> None:
    theta = np.radians(np.arange(N_SECTORS) * 360.0 / N_SECTORS)
    width = np.radians(360.0 / N_SECTORS) * 0.9
    bottom = np.zeros(N_SECTORS)
    for i, color in enumerate(SPEED_COLORS):
        ax.bar(theta, freqs[i], width=width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.6)
        bottom += freqs[i]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylim(0, rmax)
    ticks = np.linspace(0, rmax, 5)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0f}%" for t in ticks], fontsize=10, color="0.35")
    ax.set_rlabel_position(292.5)
    ax.grid(color="0.8", linewidth=0.5)
    ax.spines["polar"].set_color("0.6")


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.02, 1.1, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")


def make_figure(
    datasets: list[str], out_path: Path, sites: list[str] = list(SITES), refresh: bool = False
) -> None:
    """One row of roses per site; the site name is left out of the panel
    titles when there's only one row."""
    winds = load_site_winds(refresh)

    n_rows, n_roses = len(sites), len(datasets)
    fig = plt.figure(figsize=(3.8 * n_roses + 1.2, 4.1 * n_rows))
    # Roses on the left (one subfigure per site row), and a narrow right-hand
    # subfigure holding the speed-bin colorbar.
    roses_fig, cbar_fig = fig.subfigures(
        1, 2, width_ratios=[3.8 * n_roses, 1.2], wspace=0.0
    )
    site_figs = roses_fig.subfigures(n_rows, 1, hspace=0.0) if n_rows > 1 else [roses_fig]
    panel_labels = iter(f"{chr(ord('a') + i)})" for i in range(n_rows * n_roses))
    for site_fig, site in zip(site_figs, sites):
        grid = site_fig.add_gridspec(
            1, n_roses, left=0.06, right=0.94, top=0.86, bottom=0.08, wspace=0.3,
        )
        freqs = {
            name: wind_rose_frequencies(
                winds[f"{name}_{site}_u"].values, winds[f"{name}_{site}_v"].values
            )
            for name in datasets
        }
        # Shared radial scale per site so the datasets compare directly.
        peak = max(f.sum(axis=0).max() for f in freqs.values())
        rmax = np.ceil(peak / 4) * 4
        for col, name in enumerate(datasets):
            ax = site_fig.add_subplot(grid[0, col], projection="polar")
            plot_rose(ax, freqs[name], rmax)
            title = DISPLAY_NAMES.get(name, name)
            ax.set_title(f"{site}: {title}" if n_rows > 1 else title, pad=18,
                         fontsize=16)
            add_panel_label(ax, next(panel_labels))

    # Discrete colorbar: one equal-width block per speed bin (spacing="uniform"
    # despite the uneven bin edges), with the open-ended top bin as the
    # extend-max triangle.
    cmap = ListedColormap(SPEED_COLORS[:-1]).with_extremes(over=SPEED_COLORS[-1])
    edges = SPEED_BINS[:-1]
    norm = BoundaryNorm(edges, cmap.N)
    cax = cbar_fig.add_axes([0.1, 0.5 - 0.3 / n_rows, 0.16, 0.6 / n_rows])
    cbar = cbar_fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
        orientation="vertical", extend="max", spacing="uniform", ticks=edges,
        extendfrac="auto",
    )
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("10m wind speed (m/s)", fontsize=13)

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="re-extract the cached site winds"
    )
    args = parser.parse_args()
    make_figure(DATASETS, OUT_PATH, PLOT_SITES, refresh=args.refresh)


if __name__ == "__main__":
    main()
