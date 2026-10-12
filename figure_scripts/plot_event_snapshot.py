#!/usr/bin/env python
"""
Given a path to a local netCDF event file (as saved by
`fme.downscaling.evaluator`, containing target/predicted[/coarse] variable
pairs), generate a map plot for each variable similar to the per-event map
output of ace/scripts/downscaling/plot_events.py.

Usage:
    python plot_event_snapshot.py <nc_path> [--output-dir <path>]
        [--variables VAR1 VAR2 ...] [--crop-degrees N]
        [--center-lat LAT] [--center-lon LON]
        [--coarse-data PATH ...] [--no-coarse]

To include the coarse (model input resolution) field alongside target and
predicted maps, the event's date is looked up in a `config.yaml` saved
alongside the event .nc file (falling back to parsing the date from the
filename), and coarse data is pulled from local zarr stores at
--coarse-data (or DEFAULT_COARSE_PATHS below) at that date, regridded to
the event's fine grid via nearest-neighbor lookup.
"""

import argparse
import datetime
import math
import re
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from cartopy.feature import ShapelyFeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

warnings.filterwarnings("ignore")

UNITS = {
    "PRMSL": "hPa",
    "PRATEsfc": "kg/m^2/s",
    "eastward_wind_at_ten_meters": "m/s",
    "northward_wind_at_ten_meters": "m/s",
    "wind_speed": "m/s",
    "air_temperature_at_two_meters": "K",
}

# Prefer these suffixes in this left-to-right order when present.
SUFFIX_ORDER = ["coarse", "target"]

DEFAULT_COARSE_PATHS = [
    "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/100km.zarr",
    "/climate-default/2026-02-03-downscaling-data-X-SHiELD-AMIP-pressure-data/prmsl_100km.zarr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a map plot from a single netCDF event file"
    )
    parser.add_argument(
        "nc_path",
        help="Path to the netCDF event file",
    )
    parser.add_argument(
        "--output-dir",
        default="./event_snapshots",
        help="Output directory for figures (default: ./event_snapshots)",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Filter to only these variables (default: all eligible variables)",
    )
    parser.add_argument(
        "--crop-degrees",
        type=float,
        default=None,
        help=(
            "If set, crop the plotted domain to an N x N degree box "
            "centered on the domain's lat/lon centroid, or on --center-lat/"
            "--center-lon if given"
        ),
    )
    parser.add_argument(
        "--center-lat",
        type=float,
        default=None,
        help="Latitude to center the --crop-degrees box on (default: domain centroid)",
    )
    parser.add_argument(
        "--center-lon",
        type=float,
        default=None,
        help=(
            "Longitude to center the --crop-degrees box on, in either "
            "-180/180 or 0/360 convention (default: domain centroid)"
        ),
    )
    parser.add_argument(
        "--coarse-data",
        nargs="*",
        default=None,
        help=(
            "Path(s) to zarr store(s) with coarse-resolution fields, merged "
            "together (default: DEFAULT_COARSE_PATHS)"
        ),
    )
    parser.add_argument(
        "--no-coarse",
        action="store_true",
        help="Skip including the coarse field in the output plots",
    )
    return parser.parse_args()


# Create a STATES feature with no fill
shpfilename = shapereader.natural_earth(
    resolution="50m", category="cultural", name="admin_1_states_provinces"
)
reader = shapereader.Reader(shpfilename)
states_feature = ShapelyFeature(
    reader.geometries(),
    ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="lightgrey",
    linewidth=0.35,
    linestyle="--",
)


def add_outer_latlon_grid(ax, *, show_left, show_bottom):
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False

    if show_left:
        gl.left_labels = True
    if show_bottom:
        gl.bottom_labels = True

    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_aspect("equal", adjustable="box")


def crop_to_box(
    ds: xr.Dataset,
    size_degrees: float,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> xr.Dataset:
    """Crop a dataset to a size_degrees x size_degrees box centered on
    (center_lat, center_lon), or on the centroid of its lat/lon coordinates
    if either is not given. center_lon is accepted in either -180/180 or
    0/360 convention and is converted to match ds.lon."""
    lat_center = float(ds.lat.mean()) if center_lat is None else center_lat
    lon_center = float(ds.lon.mean()) if center_lon is None else center_lon
    if lon_center < ds.lon.min() and ds.lon.max() > 180:
        lon_center += 360
    half = size_degrees / 2.0
    return ds.sel(
        lat=slice(lat_center - half, lat_center + half),
        lon=slice(lon_center - half, lon_center + half),
    )


def detect_variable_pairs(ds: xr.Dataset) -> list[str]:
    """Detect variables that have both _predicted and _target versions."""
    predicted = {
        v[: -len("_predicted")] for v in ds.data_vars if v.endswith("_predicted")
    }
    target = {v[: -len("_target")] for v in ds.data_vars if v.endswith("_target")}
    return sorted(predicted & target)


def filename_to_datetime(filename: str) -> cftime.DatetimeJulian:
    match = re.search(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2}))?", filename)
    if match is None:
        raise ValueError(f"Could not parse date from filename: {filename}")
    return cftime.DatetimeJulian(
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4) or 0),
    )


def load_event_date(nc_path: Path) -> cftime.DatetimeJulian | None:
    """Look up the event's date in a sibling config.yaml (as saved by
    fme.downscaling.evaluator), falling back to parsing the filename."""
    config_path = nc_path.parent / "config.yaml"
    if config_path.is_file():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for event in config.get("events", []):
            if event["name"] == nc_path.stem:
                dt = datetime.datetime.fromisoformat(event["date"])
                return cftime.DatetimeJulian(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute
                )
    try:
        return filename_to_datetime(nc_path.name)
    except ValueError:
        return None


def get_coarse_data(paths: list[str]) -> xr.Dataset:
    return xr.merge([xr.open_zarr(p) for p in paths])


def merge_coarse(
    event: xr.Dataset, coarse: xr.Dataset, date: cftime.DatetimeJulian
) -> xr.Dataset:
    """Add a "<var>_coarse" field to event for each detected variable pair
    that is also present in coarse, regridded onto the event's fine grid via
    nearest-neighbor lookup."""
    coarse_at_time = coarse.sel(time=date, method="nearest")
    pad = 2.0
    coarse_crop = coarse_at_time.sel(
        grid_yt=slice(float(event.lat.min()) - pad, float(event.lat.max()) + pad),
        grid_xt=slice(float(event.lon.min()) - pad, float(event.lon.max()) + pad),
    ).rename({"grid_yt": "lat", "grid_xt": "lon"})
    coarse_regridded = coarse_crop.reindex(lat=event.lat, lon=event.lon, method="nearest")

    for var in detect_variable_pairs(event):
        if var in coarse_regridded.data_vars:
            event[f"{var}_coarse"] = coarse_regridded[var]
    return event


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    variables = detect_variable_pairs(ds)
    if (
        "eastward_wind_at_ten_meters" in variables
        and "northward_wind_at_ten_meters" in variables
    ):
        for suffix in ["target", "predicted", "coarse"]:
            e_var = f"eastward_wind_at_ten_meters_{suffix}"
            n_var = f"northward_wind_at_ten_meters_{suffix}"
            if e_var in ds and n_var in ds:
                ds[f"wind_speed_{suffix}"] = np.sqrt(ds[e_var] ** 2 + ds[n_var] ** 2)
    return ds


def plot_histogram_lines(ds: xr.Dataset, var_name: str, title_prefix: str, save_path: Path) -> None:
    """Plot a histogram comparing each ensemble member's distribution
    (thin lines), their average, and the target distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    target_data = ds[f"{var_name}_target"]
    predicted_data = ds[f"{var_name}_predicted"]
    all_data = np.concatenate([target_data.values[None], predicted_data.values])

    bins = 50
    data_min, data_max = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    sample_data = predicted_data.values
    if sample_data.ndim != 3:
        raise ValueError(
            f"Expected predicted data to be 3D (samples, lat, lon), "
            f"got shape {sample_data.shape}"
        )

    lower_bounds = np.percentile(sample_data, 0.01, axis=(1, 2))
    upper_bounds = np.percentile(sample_data, 99.99, axis=(1, 2))
    lower_bound_2p5 = np.percentile(lower_bounds, 2.5)
    lower_bound_97p5 = np.percentile(lower_bounds, 97.5)
    upper_bound_2p5 = np.percentile(upper_bounds, 2.5)
    upper_bound_97p5 = np.percentile(upper_bounds, 97.5)

    target_lower_0p01_percentile = np.percentile(target_data.values, 0.01)
    target_upper_99p99_percentile = np.percentile(target_data.values, 99.99)
    counts, _ = np.histogram(target_data.values, bins=bin_edges)

    ylim_min = 0.1
    ylim_max = 10 ** (np.log10(np.max(counts)) + 1)

    all_counts = []
    num_samples = sample_data.shape[0]
    for i in range(num_samples):
        sample_flat = sample_data[i].flatten()
        sample_counts, _ = np.histogram(sample_flat, bins=bin_edges)
        ax.step(
            bin_centers,
            sample_counts,
            where="mid",
            alpha=0.1,
            label="Samples" if i == 0 else None,
        )
        all_counts.append(sample_counts)
    all_counts = np.stack(all_counts)

    ax.step(bin_centers, counts, where="mid", color="black", linewidth=2, label="Target")
    ax.axvline(
        target_lower_0p01_percentile,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label="Target 0.01%",
    )
    ax.axvline(
        target_upper_99p99_percentile,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label="Target 99.99%",
    )
    ax.fill_betweenx(
        [ylim_min, ylim_max],
        upper_bound_2p5,
        upper_bound_97p5,
        color="gray",
        alpha=0.2,
        label="Pred upper tail 95% CI",
    )
    ax.fill_betweenx(
        [ylim_min, ylim_max],
        lower_bound_2p5,
        lower_bound_97p5,
        color="gray",
        alpha=0.2,
        label="Pred lower tail 95% CI",
    )

    avg_counts = np.mean(all_counts, axis=0)
    ax.step(
        bin_centers,
        avg_counts,
        where="mid",
        color="C0",
        linewidth=2,
        label="Average Predicted",
    )
    var_label = var_name.replace("_", " ").title()
    ax.set_xlabel(f"{var_label} [{UNITS.get(var_name, '')}]")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"{title_prefix} distribution: {var_label}")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_event(ds, var_name, samples=None, n_cols=5, **plot_kwargs):
    single_suffixes = [s for s in SUFFIX_ORDER if f"{var_name}_{s}" in ds.data_vars]
    predicted_var = f"{var_name}_predicted"

    if samples is None:
        samples = list(range(ds.sample.size))
    if len(samples) == 0:
        samples = [0]

    N = len(single_suffixes) + len(samples)
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    axes = np.atleast_1d(axes).ravel()
    for ax in axes[N:]:
        ax.set_visible(False)

    var_names = [f"{var_name}_{s}" for s in single_suffixes] + [predicted_var]
    ds_ = ds[var_names]

    arr = ds_.to_array()
    vals = arr.values
    if hasattr(vals, "compute"):
        vals = vals.compute()
    data_min = np.nanmin(vals)
    data_max = np.nanmax(vals)
    if data_min < -0.2:
        plot_kwargs["cmap"] = "RdBu_r"
        plot_kwargs.setdefault("vmin", data_min)
    else:
        plot_kwargs["cmap"] = "turbo"
        plot_kwargs["vmin"] = max(0, data_min)
    if "vmax" not in plot_kwargs:
        plot_kwargs["vmax"] = data_max

    img = None
    for i, suffix in enumerate(single_suffixes):
        ax = axes[i]
        da = ds_[f"{var_name}_{suffix}"]
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)

        ax.set_title(suffix, fontsize=10)
        ax.add_feature(states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey")
        ax.coastlines(color="lightgrey")

        row = i // n_cols
        col = i % n_cols
        add_outer_latlon_grid(ax, show_left=(col == 0), show_bottom=(row == n_rows - 1))

    offset = len(single_suffixes)
    for i, s in enumerate(samples):
        ax = axes[offset + i]
        da = ds_[predicted_var].isel(sample=s)

        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title(f"predicted {s}", fontsize=10)
        ax.add_feature(states_feature)
        ax.coastlines(color="lightgrey")
        ax.add_feature(cfeature.BORDERS, linestyle="-", color="lightgrey")
        row = (offset + i) // n_cols
        col = (offset + i) % n_cols
        add_outer_latlon_grid(ax, show_left=(col == 0), show_bottom=(row == n_rows - 1))

    cbar_ax = fig.add_axes([0.99, 0.25, 0.01, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(f"{var_name} [{UNITS.get(var_name, '')}]")

    return fig, axes


def main():
    args = parse_args()
    nc_path = Path(args.nc_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event = xr.open_dataset(nc_path)
    if args.crop_degrees is not None:
        event = crop_to_box(event, args.crop_degrees, args.center_lat, args.center_lon)

    if not args.no_coarse:
        event_date = load_event_date(nc_path)
        if event_date is None:
            print(f"Could not determine event date for {nc_path.name}; skipping coarse field")
        else:
            coarse = get_coarse_data(args.coarse_data or DEFAULT_COARSE_PATHS)
            event = merge_coarse(event, coarse, event_date)

    event = add_wind_speed(event)

    variables = detect_variable_pairs(event)
    if args.variables is not None:
        variables = [v for v in variables if v in args.variables]

    if not variables:
        print(f"No variable pairs found in {nc_path.name}")
        return

    for var in variables:
        fig, axes = plot_event(event, var)
        out_path = output_dir / f"{nc_path.stem}_{var}_maps.png"
        fig.savefig(out_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        hist_path = output_dir / f"{nc_path.stem}_{var}_histogram.png"
        plot_histogram_lines(event, var, nc_path.stem, save_path=hist_path)
        print(f"Saved: {hist_path}")

    event.close()
    print("Done!")


if __name__ == "__main__":
    main()
