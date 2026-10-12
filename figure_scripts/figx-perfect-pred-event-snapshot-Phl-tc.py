import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("/climate-default/home/annak/ace/scripts/downscaling")
from plot_events import add_wind_speed, add_outer_latlon_grid, states_feature
from precip_scale import (
    LAND_COLOR,
    OCEAN_COLOR,
    TC_TOP_PERCENTILE,
    dry_masked,
    precip_norm,
)
from series_colors import PERFECT_PRED_COLOR
from bicubic_interp import bicubic_to_grid

# Phl tc
FILENAME='tc_743_20231027T00.nc'

# Widen the source box before nearest-selecting onto the prediction grid, so
# prediction cells near the patch edge always have a coarse cell to fall back
# on (a 25km cell is ~0.3 deg wide, a 1deg coarse cell ~1 deg).
BOX_MARGIN_DEG = 2.0

# Names to look for in a coarse store, per canonical variable, in order. The
# 3km reference stores use the long names while some 1deg coarse stores use
# the GRIB ones, so both spellings have to be tried.
ALIASES = {
    "eastward_wind_at_ten_meters": ("eastward_wind_at_ten_meters", "UGRD10m"),
    "northward_wind_at_ten_meters": ("northward_wind_at_ten_meters", "VGRD10m"),
}


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the (lat, lon) coordinate names used by this dataset."""
    for lat_name, lon_name in (("lat", "lon"), ("grid_yt", "grid_xt"), ("latitude", "longitude")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
    raise KeyError(f"Could not find lat/lon coords in {list(ds.coords)}")


def _select_ref_time(ref: xr.Dataset, time: pd.Timestamp) -> xr.Dataset:
    """Select the reference snapshot nearest ``time``, handling a cftime calendar."""
    times = ref.indexes["time"]
    if len(times) and isinstance(times[0], cftime.datetime):
        target = type(times[0])(time.year, time.month, time.day, time.hour)
    else:
        target = np.datetime64(time)
    return ref.sel(time=target, method="nearest")


def _select_ref_box(
    ref: xr.Dataset,
    time: pd.Timestamp,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lat_name: str,
    lon_name: str,
) -> xr.Dataset:
    """Select the nearest reference time and the given lat/lon box.

    Handles a cftime reference calendar and a box that straddles the 0/360 lon
    seam (rebuilt as a contiguous slice with the western part wrapped negative).
    """
    snap = _select_ref_time(ref, time)

    lat = snap[lat_name]
    lat_slice = slice(lat_bounds[0], lat_bounds[1])
    if float(lat[0]) > float(lat[-1]):  # descending latitude
        lat_slice = slice(lat_bounds[1], lat_bounds[0])

    lo, hi = lon_bounds
    if hi - lo > 180.0:  # box wraps the seam; stitch west (wrapped) + east halves
        west = snap.sel({lon_name: slice(lo, 360.0)})
        west = west.assign_coords({lon_name: west[lon_name] - 360.0})
        east = snap.sel({lon_name: slice(0.0, hi - 360.0)})
        snap = xr.concat([west, east], dim=lon_name)
        return snap.sel({lat_name: lat_slice})
    return snap.sel({lat_name: lat_slice, lon_name: slice(lo, hi)})


def _match_lon_convention(lon: np.ndarray, box_lon: np.ndarray) -> np.ndarray:
    """Re-express ``lon`` in the same convention as ``box_lon``.

    ``_select_ref_box`` wraps the western half of a seam-straddling box to
    negative longitudes, so the prediction lons have to follow before they can
    be matched against it.
    """
    if box_lon.min() < 0.0:
        return np.where(lon > 180.0, lon - 360.0, lon)
    return lon


def select_on_event_grid(
    store: xr.Dataset,
    var: str,
    time: pd.Timestamp,
    event: xr.Dataset,
    margin_deg: float = BOX_MARGIN_DEG,
) -> xr.DataArray:
    """``store[var]`` at the nearest time to ``time``, nearest-neighbour
    selected onto ``event``'s lat/lon grid.

    On the 3km grid this is an exact match (same grid the predictions live on);
    on a coarser grid it is a block upsample, which is what makes the coarse
    panel directly comparable to the fine ones -- same axes, same extent, same
    colorbar.
    """
    lat_bounds = (
        float(event.lat.min()) - margin_deg,
        float(event.lat.max()) + margin_deg,
    )
    lon_bounds = (
        float(event.lon.min()) - margin_deg,
        float(event.lon.max()) + margin_deg,
    )
    lat_name, lon_name = _coord_names(store)
    box = _select_ref_box(store, time, lat_bounds, lon_bounds, lat_name, lon_name)[var]
    lon_vals = _match_lon_convention(event.lon.values, box[lon_name].values)
    on_grid = box.sel(
        {lat_name: event.lat.values, lon_name: lon_vals}, method="nearest"
    )
    # Rebuild on the prediction coords so the panels align exactly (the source
    # coords differ by O(1e-3) deg on the 3km grid and wholesale on a coarser one).
    return xr.DataArray(
        np.asarray(on_grid.values),
        dims=["lat", "lon"],
        coords={"lat": event.lat, "lon": event.lon},
        attrs=box.attrs,
    )


def bicubic_on_event_grid(
    store: xr.Dataset,
    var: str,
    time: pd.Timestamp,
    event: xr.Dataset,
    margin_deg: float = BOX_MARGIN_DEG,
) -> xr.DataArray:
    """``store[var]`` at the nearest time to ``time``, bicubically interpolated
    onto ``event``'s lat/lon grid (see bicubic_interp.py). The margin leaves
    the kernel's outer neighbours inside the source box at the patch edges.
    """
    lat_bounds = (
        float(event.lat.min()) - margin_deg,
        float(event.lat.max()) + margin_deg,
    )
    lon_bounds = (
        float(event.lon.min()) - margin_deg,
        float(event.lon.max()) + margin_deg,
    )
    lat_name, lon_name = _coord_names(store)
    box = _select_ref_box(store, time, lat_bounds, lon_bounds, lat_name, lon_name)[var]
    lon_vals = _match_lon_convention(event.lon.values, box[lon_name].values)
    return xr.DataArray(
        bicubic_to_grid(box, lat_name, lon_name, event.lat.values, lon_vals),
        dims=["lat", "lon"],
        coords={"lat": event.lat, "lon": event.lon},
        attrs=box.attrs,
    )


def _find_in_stores(
    stores: list[xr.Dataset], var: str
) -> tuple[xr.Dataset, str] | None:
    """First (store, name) holding ``var`` under any of its ``ALIASES``."""
    for store in stores:
        for candidate in ALIASES.get(var, (var,)):
            if candidate in store:
                return store, candidate
    return None


def open_stores(paths: list[str]) -> list[xr.Dataset]:
    stores = []
    for path in paths:
        ds = xr.open_zarr(path)
        if "sample" in ds.dims:
            ds = ds.isel(sample=0)
        stores.append(ds)
    return stores

root="/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-10-hiro-perfect-pred-all-tcs-max-snapshot"
ds=xr.open_dataset(f"{root}/{FILENAME}")

# This run downscales the 100km stores to 3km (see config.yaml's data.coarse),
# so those are the coarse input fields; the .nc file already carries the
# fine-resolution target and predicted samples.
COARSE_ZARRS = [
    "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/100km.zarr",
    "/climate-default/2026-02-03-downscaling-data-X-SHiELD-AMIP-pressure-data/prmsl_100km.zarr",
]

event_time = pd.to_datetime("20231027T00", format="%Y%m%dT%H")
coarse_stores = open_stores(COARSE_ZARRS)

for var in ["PRATEsfc", "PRMSL", "eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]:
    store, store_var = _find_in_stores(coarse_stores, var)
    ds[f"{var}_coarse"] = select_on_event_grid(store, store_var, event_time, ds)
    ds[f"{var}_bicubic"] = bicubic_on_event_grid(store, store_var, event_time, ds)

ds = add_wind_speed(ds)
# The left column of the main figure and of the ensemble supplement is the
# bicubic baseline, with wind speed derived from the interpolated components
# rather than interpolated itself.
ds["wind_speed_bicubic"] = np.sqrt(
    ds.eastward_wind_at_ten_meters_bicubic**2
    + ds.northward_wind_at_ten_meters_bicubic**2
)


CROP_DEG = 6.0
center_lat = float(ds.lat.mean())
center_lon = float(ds.lon.mean())
ds_crop = ds.sel(
    lat=slice(center_lat - CROP_DEG / 2, center_lat + CROP_DEG / 2),
    lon=slice(center_lon - CROP_DEG / 2, center_lon + CROP_DEG / 2),
)
for suffix in ["coarse", "bicubic", "target", "predicted"]:
    ds_crop[f"PRATEsfc_{suffix}"] = ds_crop[f"PRATEsfc_{suffix}"] * 86400  # kg/m^2/s -> mm/day

# All three fields are physically non-negative, so turbo (with vmin clipped to
# 0) suits them all -- unlike plot_events.plot_event's auto RdBu_r/turbo
# switch, which misfires on precip's small negative interpolation noise.
# Precipitation is the exception to the linear ramp: it gets the shared
# log-spaced band scale from precip_scale, the same one the event-snapshot
# figures use, so a colour means the same rain rate across the figure set.
PRECIP_VAR = "PRATEsfc"
ROWS = [
    ("PRATEsfc", "Surface precipitation rate", "mm/day"),
    ("PRMSL", "Sea level pressure", "hPa"),
    ("wind_speed", "Wind speed", "m/s"),
]
COL_LABELS = ["Bicubic", "Target", "Predicted"]

# Panel letters: one per map row (on its leftmost panel) and one per
# histogram, in reading order down each column.
MAP_PANEL_LABELS = ["a)", "b)", "c)"]
HIST_PANEL_LABELS = ["d)", "e)", "f)"]

# Decades of empty space added above the data in the (log-scaled) top-row
# histogram, so its legend has somewhere to sit that isn't on top of the
# curves.
LEGEND_HEADROOM_DECADES = 1.0

fig = plt.figure(figsize=(10.0, 2.6 * len(ROWS)))
# One shared GridSpec for all 4 columns, so every row's top/bottom edges line
# up exactly and the histogram column sits at the same wspace as the map
# columns. The histogram axes still need per-row tick-label room that the map
# axes don't (see the set_position inset below), but that's handled by
# shrinking each histogram axis within its own row slot, not by giving the
# histogram column a different (and therefore misaligned) hspace/wspace.
# Dedicated columns for the colorbar and a blank spacer, between the maps and
# the histogram: fig.colorbar(..., ax=map_axes) steals space from the map
# axes' own column instead, which -- now that the histogram column sits right
# next to it in the same grid -- collided with the histogram's y-axis label.
# The blank spacer column is a hard, code-guaranteed gap: relying on
# wspace/pad alone to keep the colorbar's tick labels clear of the
# histogram's own y-axis text turned out to be too fragile (their actual
# rendered widths vary per row and easily eat through a thin gap).
gs = fig.add_gridspec(len(ROWS), 6, hspace=0.08, wspace=0.1, width_ratios=[1, 1, 1, 0.35, 0.5, 1.3])

for row, (var, title, units) in enumerate(ROWS):
    names = [f"{var}_bicubic", f"{var}_target", f"{var}_predicted"]
    vals = ds_crop[names].to_array().values
    data_min, data_max = np.nanmin(vals), np.nanmax(vals)
    if var == PRECIP_VAR:
        norm, cmap, precip_ticks = precip_norm(vals, percentile=TC_TOP_PERCENTILE)
        plot_kwargs = {"cmap": cmap, "norm": norm}
    else:
        precip_ticks = None
        plot_kwargs = {"cmap": "turbo", "vmin": max(0, data_min), "vmax": data_max}

    map_axes = []
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        map_axes.append(ax)
        da = ds_crop[name]
        if "sample" in da.dims:
            da = da.isel(sample=0)
        if var == PRECIP_VAR:
            da = dry_masked(da)
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title("")  # xarray auto-titles predicted panels "sample = 0"
        if var == PRECIP_VAR:
            # Under the shading (zorder 0), so the dry cells the floor leaves
            # unshaded read as sea or land rather than as page.
            ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey")
        ax.coastlines(color="lightgrey")
        add_outer_latlon_grid(ax, show_left=(col == 0), show_bottom=(row == len(ROWS) - 1))
        if row == len(ROWS) - 1:
            # cartopy hides the native x-axis (and with it any xlabel) in
            # favor of the gridliner's own tick labels; re-show just the
            # (tickless) axis so the label renders beneath those ticks.
            ax.xaxis.set_visible(True)
            ax.set_xticks([])
            ax.set_xlabel(COL_LABELS[col], fontsize=10, labelpad=15)

    map_axes[0].text(
        0.0,
        1.08,
        MAP_PANEL_LABELS[row],
        transform=map_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    map_axes[1].text(
        0.5,
        1.08,
        title,
        transform=map_axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    # The bar sits at the *left* edge of its reserved column (normal
    # orientation, ticks/label to its right), same as before the histogram
    # column existed; the blank spacer column (index 4) is what actually
    # keeps its tick labels clear of the histogram's y-axis.
    cax = fig.add_subplot(gs[row, 3])
    cax_pos = cax.get_position()
    v_inset = cax_pos.height * 0.1
    bar_frac = 0.25
    cax.set_position(
        [
            cax_pos.x0,
            cax_pos.y0 + v_inset,
            cax_pos.width * bar_frac,
            cax_pos.height - 2 * v_inset,
        ]
    )
    # The precipitation bands are unevenly spaced, so the ticks have to be the
    # band edges; a locator picking round numbers would label places where the
    # colour does not change.
    cbar = fig.colorbar(img, cax=cax, **({"ticks": precip_ticks} if precip_ticks else {}))
    cbar.set_label(units, labelpad=2)

    # --- histogram column: target vs. pooled-ensemble predicted PDFs, over
    # the same cropped region as the maps, with the tail percentile that
    # matters for this variable (see ROWS above). ---
    hist_ax = fig.add_subplot(gs[row, 5])
    # Shrink from the bottom to leave room for this axis's own tick/label
    # text (ROW_HSPACE is shared with the map rows to keep the slots the same
    # height, so that room has to come out of the axis itself, not the gap
    # between rows), then nudge the whole axis down a few percent of the
    # slot height so it doesn't sit flush against the row above.
    pos = hist_ax.get_position()
    label_margin = pos.height * 0.16
    shift_down = pos.height * 0.08
    hist_ax.set_position(
        [pos.x0, pos.y0 + label_margin - shift_down, pos.width, pos.height - label_margin]
    )

    hist_ax.text(
        0.0,
        1.02,
        HIST_PANEL_LABELS[row],
        transform=hist_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    target_vals = ds_crop[f"{var}_target"].values.ravel()
    target_vals = target_vals[np.isfinite(target_vals)]

    pred_da = ds_crop[f"{var}_predicted"]
    pred_by_sample = pred_da.values.reshape(pred_da.sizes["sample"], -1)
    pred_all = pred_by_sample[np.isfinite(pred_by_sample)]

    bins = np.linspace(
        min(target_vals.min(), pred_all.min()),
        max(target_vals.max(), pred_all.max()),
        60,
    )
    hist_ax.hist(target_vals, bins=bins, density=True, histtype="step", color="black", lw=1.3, label="Target")
    for sample_vals in pred_by_sample:
        sample_vals = sample_vals[np.isfinite(sample_vals)]
        hist_ax.hist(sample_vals, bins=bins, density=True, histtype="step", color=PERFECT_PRED_COLOR, alpha=0.4, lw=0.7)
    hist_ax.hist(pred_all, bins=bins, density=True, histtype="step", color=PERFECT_PRED_COLOR, lw=1.3, label="Predicted (ensemble)")
    hist_ax.set_yscale("log")

    # Matching hspace to the map rows (so histogram rows are the same height
    # and line up with them) leaves little room between rows for x-tick
    # labels, hence the small font / tight padding / few ticks below.
    hist_ax.set_xlabel(units, fontsize=8, labelpad=2)
    hist_ax.set_ylabel("density", fontsize=8, labelpad=2)
    hist_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    hist_ax.tick_params(labelsize=7, pad=1)
    # The legend entries are the same in every row, so it's drawn once, on
    # the top row, rather than covering up all three histograms. The y-axis
    # is log, so add decades of empty headroom above the data for the legend
    # to sit in instead of letting it overlap the curves.
    if row == 0:
        y_lo, y_hi = hist_ax.get_ylim()
        hist_ax.set_ylim(y_lo, y_hi * 10 ** LEGEND_HEADROOM_DECADES)
        hist_ax.legend(fontsize=6, loc="upper right")

plt.savefig('main_figures/perfect_pred_Phl_tc.png', dpi=300, transparent=True, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Ensemble supplement: same rows/variables, but every predicted sample gets
# its own column (in place of the single-sample "Predicted" column) and the
# histogram column is dropped -- for eyeballing ensemble spread per variable.
# ---------------------------------------------------------------------------

n_samples = ds_crop.sizes["sample"]
ENSEMBLE_COL_LABELS = ["Bicubic", "Target"] + [f"Predicted {i}" for i in range(n_samples)]
n_ecols = len(ENSEMBLE_COL_LABELS)

fig_e, axes_e = plt.subplots(
    len(ROWS),
    n_ecols,
    figsize=(2.0 * n_ecols, 2.6 * len(ROWS)),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

for row, (var, title, units) in enumerate(ROWS):
    bicubic_target_vals = ds_crop[[f"{var}_bicubic", f"{var}_target"]].to_array().values
    pred_vals = ds_crop[f"{var}_predicted"].values
    data_min = min(np.nanmin(bicubic_target_vals), np.nanmin(pred_vals))
    data_max = max(np.nanmax(bicubic_target_vals), np.nanmax(pred_vals))
    if var == PRECIP_VAR:
        norm, cmap, precip_ticks = precip_norm(
            bicubic_target_vals, pred_vals, percentile=TC_TOP_PERCENTILE
        )
        plot_kwargs = {"cmap": cmap, "norm": norm}
    else:
        precip_ticks = None
        plot_kwargs = {"cmap": "turbo", "vmin": max(0, data_min), "vmax": data_max}

    img = None
    for col in range(n_ecols):
        ax = axes_e[row, col]
        if col == 0:
            da = ds_crop[f"{var}_bicubic"]
        elif col == 1:
            da = ds_crop[f"{var}_target"]
        else:
            da = ds_crop[f"{var}_predicted"].isel(sample=col - 2)
        if var == PRECIP_VAR:
            da = dry_masked(da)
        img = da.plot(ax=ax, add_colorbar=False, **plot_kwargs)
        ax.set_title(ENSEMBLE_COL_LABELS[col] if row == 0 else "", fontsize=9)
        if var == PRECIP_VAR:
            # Under the shading (zorder 0), so the dry cells the floor leaves
            # unshaded read as sea or land rather than as page.
            ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey")
        ax.coastlines(color="lightgrey")
        add_outer_latlon_grid(ax, show_left=(col == 0), show_bottom=(row == len(ROWS) - 1))

    cbar = fig_e.colorbar(
        img, ax=axes_e[row, :].tolist(), orientation="vertical", fraction=0.01,
        pad=0.008, aspect=30, **({"ticks": precip_ticks} if precip_ticks else {}),
    )
    cbar.set_label(units, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    axes_e[row, 0].text(
        -0.35,
        0.5,
        title,
        transform=axes_e[row, 0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        rotation=90,
    )

plt.savefig(
    'supplements/perfect_pred_Phl_tc_ensemble.png', dpi=300, transparent=True, bbox_inches="tight"
)


# ---------------------------------------------------------------------------
# Pressure-contour supplement: the target and predicted wind-speed panels of
# the main figure, zoomed onto the eye and overlaid with each panel's own sea
# level pressure field.
#
# Both panels are cut to the same box, centred on the *target* pressure
# minimum rather than on each panel's own: the predicted eye is free to sit
# off-centre, and a per-panel centre would hide exactly that displacement.
# ---------------------------------------------------------------------------

ZOOM_DEG = 2.0            # width of the box centred on the target pressure min
ZOOM_SAMPLE = 0           # the sample drawn in the main figure's "Predicted" column
# White reads on both ends of turbo -- the dark blue eye and the red eyewall.
ZOOM_CONTOUR_COLOR = "white"
# Candidate contour intervals in hPa, coarsest last; the first that keeps the
# count at or below ZOOM_MAX_CONTOURS wins.
ZOOM_CONTOUR_STEPS = (1, 2, 5, 10, 20)
ZOOM_MAX_CONTOURS = 8


def _pressure_min(field: xr.DataArray) -> tuple[float, float, float]:
    """(lat, lon, value) at the minimum of a 2D lat/lon field."""
    i, j = np.unravel_index(int(np.nanargmin(field.values)), field.shape)
    return float(field.lat[i]), float(field.lon[j]), float(field.values[i, j])


def _contour_levels(fields: list[xr.DataArray]) -> np.ndarray:
    """One set of levels for every panel, so the contours are comparable."""
    lo = min(float(np.nanmin(f)) for f in fields)
    hi = max(float(np.nanmax(f)) for f in fields)
    for step in ZOOM_CONTOUR_STEPS:
        levels = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if len(levels) <= ZOOM_MAX_CONTOURS:
            return levels
    return levels


# The centre comes from the full event box, before any cropping.
zoom_lat, zoom_lon, zoom_min = _pressure_min(ds.PRMSL_target)
_half = ZOOM_DEG / 2
ds_zoom = ds.sel(
    lat=slice(zoom_lat - _half, zoom_lat + _half),
    lon=slice(zoom_lon - _half, zoom_lon + _half),
)
print(
    f"target pressure minimum {zoom_min:.1f} hPa at {zoom_lat:.2f} N, "
    f"{zoom_lon:.2f} E -> {ZOOM_DEG} deg box "
    f"({ds_zoom.sizes['lat']} x {ds_zoom.sizes['lon']} cells)"
)

zoom_panels = [
    ("Target", ds_zoom.wind_speed_target, ds_zoom.PRMSL_target),
    (
        "Predicted",
        ds_zoom.wind_speed_predicted.isel(sample=ZOOM_SAMPLE),
        ds_zoom.PRMSL_predicted.isel(sample=ZOOM_SAMPLE),
    ),
]

# Shared scales across the two panels, as in the main figure's rows.
_zoom_winds = [p[1] for p in zoom_panels]
zoom_vmin = max(0.0, min(float(np.nanmin(w)) for w in _zoom_winds))
zoom_vmax = max(float(np.nanmax(w)) for w in _zoom_winds)
zoom_levels = _contour_levels([p[2] for p in zoom_panels])

fig_z, axes_z = plt.subplots(
    1, 2, figsize=(8.2, 4.4), subplot_kw={"projection": ccrs.PlateCarree()}
)
for col, (ax, (label, wind, prmsl)) in enumerate(zip(axes_z, zoom_panels)):
    img = wind.plot(
        ax=ax, add_colorbar=False, cmap="turbo", vmin=zoom_vmin, vmax=zoom_vmax
    )
    ax.set_title("")  # xarray auto-titles the predicted panel "sample = 0"
    cs = ax.contour(
        prmsl.lon,
        prmsl.lat,
        prmsl.values,
        levels=zoom_levels,
        colors=ZOOM_CONTOUR_COLOR,
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%.0f", fontsize=6, inline=True, colors=ZOOM_CONTOUR_COLOR)
    ax.add_feature(states_feature)
    ax.add_feature(cfeature.BORDERS, color="lightgrey")
    ax.coastlines(color="lightgrey")
    add_outer_latlon_grid(ax, show_left=(col == 0), show_bottom=True)
    ax.xaxis.set_visible(True)
    ax.set_xticks([])
    ax.set_xlabel(label, fontsize=10, labelpad=15)

# The panels are square (equal aspect on a square box), so matplotlib shrinks
# them inside their slots at draw time. Reading the *drawn* position back is
# what lets the bar match the panel height exactly -- fraction/aspect on
# fig.colorbar sizes the bar against the slot, not against the panel.
fig_z.canvas.draw()
_panel_pos = axes_z[1].get_position()
cax_z = fig_z.add_axes(
    [_panel_pos.x1 + 0.02, _panel_pos.y0, 0.018, _panel_pos.height]
)
cbar_z = fig_z.colorbar(img, cax=cax_z)
cbar_z.set_label("m/s", labelpad=2)

plt.savefig(
    'supplements/perfect_pred_Phl_tc_press_contours.png',
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)
