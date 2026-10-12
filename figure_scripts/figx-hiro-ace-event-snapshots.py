"""HiRO-ACE downscale snapshots for three simulated-2023 events.

Five figures, each a single 3 km panel except ar2's, which sets its 10-m
wind speed and sea-level pressure beside the precipitation --

    main_figures/hiro_ace_ar2_2023-01-31_precip.png   AR, strongest transport
    main_figures/hiro_ace_sa1_2023-02-22_wind.png     Santa Ana, Transverse Range
    main_figures/hiro_ace_bora1_2023-01-29_wind.png   Adriatic Bora, Gulf of Rijeka
    supplements/hiro_ace_ar1_2023-01-27_precip.png    AR, California landfall

-- plus ``supplements/hiro_ace_ar2_2023-01-31_slp.png``, ar2's sea-level
pressure twin. Promoted from the exploratory ``ad_hoc/hiro_ace_snapshots/``
scripts, which own the wider catalogue those events were picked from, and which
keep the coarse-vs-downscaled pairs these once showed. The style there is kept
(turbo shading, topography contours, city markers, high-wind arrows) with two
changes: no suptitle, and every panel carrying labelled lat/lon axes.

The two stores share an identical 1453-step 2023 time axis, so a timestamp
indexes straight into each. The downscale carries PRATEsfc/PRMSL/UGRD10m/VGRD10m
only, so precipitation and 10-m wind speed carry the comparison.

Free run initialised from 2014 ICs: simulated "2023" is model-internal weather
and does NOT correspond to real-world 2023 events.

The high-wind arrows mark blobs over land only. Both events are downslope
windstorms whose damage is a land phenomenon, and the Bora in particular puts
its strongest jets out over the Adriatic, where arrows would crowd out the
coastal maxima that are the point of the panel. Land is tested against the same
Natural Earth 10 m polygons the panels draw their coastline from, so an arrow
never lands on the wrong side of a drawn coast.

The precipitation panel shades a 6-hour mean rate, not an instantaneous one, on
the boundary scale the catalogue uses; below 0.5 mm/day the land/ocean basemap
shows through. The two wind events are each cropped to the edges of a 2x3 block
of whole coarse cells, the same construction as
``figx-downslope-wind-events.py``'s corridor box, so the coarse panel shows
complete cells rather than cells clipped by the frame.

The figures are picked by key on the command line, all five by default::

    python figx-hiro-ace-event-snapshots.py                 # all five
    python figx-hiro-ace-event-snapshots.py ar2             # main_figures/hiro_ace_ar2_2023-01-31_precip.png
    python figx-hiro-ace-event-snapshots.py ar1 ar2_slp     # the two supplements

The plot data -- each event's 3 km fields cropped to its panel, and for the
wind events the field-of-view box and the topography under it -- is cached in
plot_data_cache/figx-hiro-ace-event-snapshots/, one file per event (ar2 and
ar2_slp share theirs); pass --refresh to recompute the selected figures' from
the stores. The wind events' land mask still comes from cartopy's Natural Earth
polygons, as do the coastlines drawn.
"""

import argparse
import os
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import shapely
import shapely.geometry as sgeom
import xarray as xr
from cartopy.feature import ShapelyFeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from scipy import ndimage
from shapely.ops import unary_union

from plot_cache import MAIN_FIGURES_DIR, SUPPLEMENTS_DIR, cached_dataset
from precip_scale import (
    LAND_COLOR,
    OCEAN_COLOR,
    PRECIP_FLOOR,
    PRECIP_LADDER,
    PRECIP_TOP_PERCENTILE,
    precip_norm,
)

HERE = Path(__file__).parent

# Every figure here is named for the model, whichever directory it lands in;
# the keys are the catalogue's, which say nothing about where they came from.
PREFIX = "hiro_ace_"
MAIN_FIGURES = MAIN_FIGURES_DIR
SUPPLEMENTS = SUPPLEMENTS_DIR
CACHE_SUBDIR = "figx-hiro-ace-event-snapshots"
DESTINATIONS = {
    "ar1": SUPPLEMENTS,
    "ar2": MAIN_FIGURES,
    "sa1": MAIN_FIGURES,
    "bora1": MAIN_FIGURES,
}

ROOT = "/climate-default/2026-08-03-hiro-ace-v2-outputs"
COARSE = f"{ROOT}/2026-08-03-coarse-ace2s-10yr-inference/output_6hourly_predictions_ic0000.zarr"
FINE = f"{ROOT}/2026-08-05-hiro-ace2s-downscale-global-2023-ic0000/global_ace2s_downscaled_2023.zarr"
TOPO = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr"

# Every figure is a single HiRO-ACE panel, so nothing is titled per panel and
# the width is set once here; the height follows each event's own extent.
PANEL_WIDTH_IN = 7.5

FINE_LAT_MAX = 64.8  # downscale domain limit

# --- wind style constants, matching figx-downslope-wind-events.py ------------
TOPO_INTERVAL_M = 300
WIND_BLOB_PERCENTILE = 95
WIND_BLOB_MIN_CELLS = 5
N_WIND_BLOBS = 6
WIND_ARROW_LENGTH_FRAC = 1 / 14
WIND_ARROW_WIDTH_FRAC = 0.006
WIND_BLOB_MIN_SEPARATION_FRAC = 1 / 11

AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13

states_feature = ShapelyFeature(
    shapereader.Reader(
        shapereader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lines",
        )
    ).geometries(),
    ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="lightgrey",
    linewidth=0.35,
    linestyle="--",
)


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------
def event_time_label(stamp):
    """``"2023-01-27 18"`` -> ``"2023-01-27 18:00 UTC"``, for the suptitle."""
    date, hour = stamp.split()
    return f"{date} {int(hour):02d}:00 UTC"


def tindex(ds, stamp):
    hits = [k for k, v in enumerate(ds.time.values) if str(v).startswith(stamp)]
    if not hits:
        raise SystemExit(f"no timestep {stamp!r}")
    return hits[0]


def to_signed(da):
    """Re-express a 0-360 lon axis as -180..180 and sort, so the data sits in
    a plain PlateCarree axes' native x range and set_extent works."""
    lon = ((da.lon.values + 180) % 360) - 180
    return da.assign_coords(lon=lon).sortby("lon")


# ``ax.coastlines(resolution="auto")`` resolves to 10 m at both wind domains'
# zoom, so testing land against the same scale keeps the arrows consistent with
# the coastline actually drawn beneath them.
LAND_RESOLUTION = "10m"


@lru_cache(maxsize=1)
def _land_shapefile():
    return shapereader.natural_earth(
        resolution=LAND_RESOLUTION, category="physical", name="land"
    )


def land_mask(lats, lons):
    """Boolean (lat, lon) mask, True on land.

    Only the polygons meeting the panel are unioned -- the global 10 m land
    layer is thousands of them, and unioning the lot to test a few thousand
    points would dominate the runtime of the whole figure.
    """
    bbox = sgeom.box(
        float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())
    )
    geoms = [
        g for g in shapereader.Reader(_land_shapefile()).geometries() if g.intersects(bbox)
    ]
    if not geoms:
        return np.zeros((len(lats), len(lons)), dtype=bool)
    lon2d, lat2d = np.meshgrid(np.asarray(lons), np.asarray(lats))
    return shapely.contains_xy(unary_union(geoms), lon2d, lat2d)


def add_latlon_axes(ax, *, show_left, show_bottom, label_axes=True):
    """Dashed graticule with edge tick labels, plus named axes.

    Tick labels go only where they are not repeated -- longitudes along the
    bottom, latitudes on the leftmost panel -- and the axis names follow them,
    so a panel without latitude ticks does not get a stray "Latitude".
    """
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = gl.right_labels = gl.left_labels = gl.bottom_labels = False
    gl.left_labels = bool(show_left)
    gl.bottom_labels = bool(show_bottom)
    gl.xlabel_style = {"size": TICK_LABEL_SIZE}
    gl.ylabel_style = {"size": TICK_LABEL_SIZE}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if label_axes:
        # A GeoAxes ships with both axis artists hidden, so set_xlabel alone is
        # stored and never drawn. Re-showing them with no ticks of their own
        # draws the names and nothing else; the gridliner keeps supplying the
        # tick labels. Both names are always set, blank where the side carries
        # no ticks, because xarray's .plot() leaves "lat"/"lon" behind on the
        # panels that would otherwise keep whatever it wrote.
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Longitude" if show_bottom else "", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Latitude" if show_left else "", fontsize=AXIS_LABEL_SIZE)
    ax.set_aspect("equal", adjustable="box")
    return gl


def fit_axis_label_pads(fig, panels, margin_pt=7):
    """Push each axis name clear of the gridliner labels it sits behind.

    The gap needed is the width (or height) of the tick labels themselves,
    which depends on the panel's extent and font -- "34.75°N" needs half again
    what "51°N" does -- so it is measured from the drawn labels rather than
    guessed at, and the pads then hold across events.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def span(artists, attr):
        shown = [t for t in artists if t.get_visible() and t.get_text()]
        if not shown:
            return 0.0
        pixels = max(getattr(t.get_window_extent(renderer), attr) for t in shown)
        return pixels * 72 / fig.dpi  # points, the unit labelpad is in

    for ax, gl in panels:
        if ax.get_ylabel():
            ax.yaxis.labelpad = span(gl.left_label_artists, "width") + margin_pt
        if ax.get_xlabel():
            ax.xaxis.labelpad = span(gl.bottom_label_artists, "height") + margin_pt
    fig.canvas.draw()


# --- sea-level pressure isobars, for the pressure figure -------------------
SLP_CONTOUR_HPA = 4  # isobar interval
# Isobars are smoothed on the 3 km panel only, over roughly one coarse cell, so
# the two panels' contours describe the same scale of feature. The shading is
# left unsmoothed, so the panel still shows the field the model produced.
SLP_FINE_SMOOTH_CELLS = 12


def isobar_levels(values):
    """Whole multiples of the interval spanning the pair of panels, so both
    panels carry the same isobars and can be read against each other."""
    lo = np.floor(np.min(values) / SLP_CONTOUR_HPA) * SLP_CONTOUR_HPA
    hi = np.ceil(np.max(values) / SLP_CONTOUR_HPA) * SLP_CONTOUR_HPA
    return np.arange(lo, hi + SLP_CONTOUR_HPA, SLP_CONTOUR_HPA)


def draw_isobars(ax, sub, levels, smooth_cells, color="0.62", linewidth=0.7,
                 label=True):
    field = sub.PRMSL.values
    if smooth_cells:
        field = ndimage.gaussian_filter(field, smooth_cells)
    cs = ax.contour(
        sub.lon.values, sub.lat.values, field, levels=levels, colors=color,
        linewidths=linewidth, transform=ccrs.PlateCarree(), zorder=5,
    )
    if not label:
        return cs
    # The labels sit over the shaded field, so they carry a halo.
    for t in ax.clabel(cs, fmt="%d", fontsize=9, inline=True):
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    return cs


# --------------------------------------------------------------------------
# precipitation figure
# --------------------------------------------------------------------------
def load_panel(path, stamp, extent, fine):
    x0, x1, y0, y1 = extent
    ds = xr.open_zarr(path)
    sl = (
        dict(latitude=slice(y0, y1), longitude=slice(x0, x1))
        if fine
        else dict(lat=slice(y0, y1), lon=slice(x0, x1))
    )
    sub = ds.isel(time=tindex(ds, stamp)).sel(**sl)
    if fine:
        sub = sub.isel(ensemble=0).rename(latitude="lat", longitude="lon")
    return sub.load()


# The 3 km variables each event's figures draw; ar2's file also serves ar2_slp.
FINE_VARS = {
    "ar1": ["PRATEsfc"],
    "ar2": ["PRATEsfc", "UGRD10m", "VGRD10m", "PRMSL"],
}


def load_fine_panel(key, stamp, extent, refresh=False):
    """``load_panel`` on the 3 km store, through the plot-data cache, keeping
    only ``FINE_VARS[key]``. The file is named for the event, not the extent,
    so --refresh after changing an event's extent."""
    name = f"{CACHE_SUBDIR}/{key}_{stamp[:10]}T{stamp[11:]}_fine.nc"
    return cached_dataset(
        name,
        lambda: load_panel(FINE, stamp, extent, True)[FINE_VARS[key]],
        refresh,
    )


def draw_precip_panel(ax, sub, extent, clon, norm, cmap):
    x0, x1, y0, y1 = extent
    ax.set_extent(
        (x0 - clon, x1 - clon, y0, y1), crs=ccrs.PlateCarree(central_longitude=clon)
    )
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    v = (sub.PRATEsfc * 86400).values
    mesh = ax.pcolormesh(
        sub.lon.values,
        sub.lat.values,
        np.ma.masked_less(v, PRECIP_FLOOR),
        norm=norm,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        zorder=2,
        shading="nearest",
    )
    ax.coastlines(lw=0.6, color="0.25", zorder=6)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.45", zorder=6)
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.55", zorder=6)
    return mesh


def build_precip(key, stamp, extent, refresh=False):
    x0, x1, y0, y1 = extent
    y1 = min(y1, FINE_LAT_MAX)
    extent = (x0, x1, y0, y1)
    clon = 0.5 * (x0 + x1)
    proj = ccrs.PlateCarree(central_longitude=clon)

    # The scale is this event's own, now set from the one panel that is drawn.
    sub = load_fine_panel(key, stamp, extent, refresh)
    rate = (sub.PRATEsfc * 86400).values
    norm, cmap, levels = precip_norm(rate, rate)

    h = PANEL_WIDTH_IN * (y1 - y0) / (x1 - x0)
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_IN + 1.0, max(3.0, min(h, 9.0))),
        subplot_kw=dict(projection=proj),
    )
    mesh = draw_precip_panel(ax, sub, extent, clon, norm, cmap)
    gridliners = [(ax, add_latlon_axes(ax, show_left=True, show_bottom=True))]
    print(f"  {key}: precip bands {levels[0]:g}-{levels[-1]:g} mm/day")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.015, ticks=levels)
    cb.set_label("Precipitation [mm/day]", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    fit_axis_label_pads(fig, gridliners)
    ax.set_title(event_time_label(stamp), fontsize=15, color="0.15")

    return save(fig, key, stamp, "precip")


# --------------------------------------------------------------------------
# sea-level pressure figure
# --------------------------------------------------------------------------
# Sequential, one hue, dark at the low end: the deep centre is the feature, and
# a rainbow would invent structure in a field that is monotone by nature. The
# palest step is dropped so the high-pressure end stays distinct from the page.
SLP_CMAP = LinearSegmentedColormap.from_list(
    "slp", plt.get_cmap("Blues_r")(np.linspace(0.02, 0.90, 256))
)
def draw_slp_panel(ax, sub, extent, clon, norm, levels, smooth_cells,
                   *, cmap=SLP_CMAP, isobars=False, isobar_color="0.62"):
    x0, x1, y0, y1 = extent
    ax.set_extent(
        (x0 - clon, x1 - clon, y0, y1), crs=ccrs.PlateCarree(central_longitude=clon)
    )
    lon, lat, v = sub.lon.values, sub.lat.values, sub.PRMSL.values
    mesh = ax.pcolormesh(
        lon, lat, v, norm=norm, cmap=cmap,
        transform=ccrs.PlateCarree(), zorder=2, shading="nearest",
    )
    if isobars:
        draw_isobars(ax, sub, levels, smooth_cells, color=isobar_color)
    ax.coastlines(lw=0.6, color="0.25", zorder=6)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.45", zorder=6)
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.55", zorder=6)
    return mesh


def build_slp(key, stamp, extent, out_dir, refresh=False):
    """The precip figure's twin: same event and extent, shading sea-level
    pressure instead."""
    x0, x1, y0, y1 = extent
    y1 = min(y1, FINE_LAT_MAX)
    extent = (x0, x1, y0, y1)
    clon = 0.5 * (x0 + x1)
    proj = ccrs.PlateCarree(central_longitude=clon)

    sub = load_fine_panel(key, stamp, extent, refresh)
    pressure = sub.PRMSL.values
    norm = Normalize(vmin=np.floor(pressure.min()), vmax=np.ceil(pressure.max()))
    levels = isobar_levels(pressure)

    h = PANEL_WIDTH_IN * (y1 - y0) / (x1 - x0)
    fig, ax = plt.subplots(
        figsize=(PANEL_WIDTH_IN + 1.0, max(3.0, min(h, 9.0))),
        subplot_kw=dict(projection=proj),
    )
    mesh = draw_slp_panel(ax, sub, extent, clon, norm, levels, SLP_FINE_SMOOTH_CELLS)
    gridliners = [(ax, add_latlon_axes(ax, show_left=True, show_bottom=True))]
    print(f"  {key}: SLP {norm.vmin:g}-{norm.vmax:g} hPa")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.015)
    cb.set_label("Sea-level pressure [hPa]", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    fit_axis_label_pads(fig, gridliners)
    ax.set_title(event_time_label(stamp), fontsize=15, color="0.15")

    return save(fig, key, stamp, "slp", out_dir=out_dir)


# --------------------------------------------------------------------------
# precipitation / wind speed / pressure panels
# --------------------------------------------------------------------------
AR_PANEL_WIDTH_IN = 5.0
# Diverging, centred on zero: the sign is the direction (east/north positive),
# which a sequential ramp could not show.
WIND_COMPONENT_CMAP = "RdBu_r"


def draw_wind_component_panel(ax, sub, var, extent, clon, norm):
    """One signed 10 m wind component. Not drawn here any more; kept for
    ad_hoc/hiro_ace_snapshots/ar2_precip_wind_components.py."""
    x0, x1, y0, y1 = extent
    ax.set_extent(
        (x0 - clon, x1 - clon, y0, y1), crs=ccrs.PlateCarree(central_longitude=clon)
    )
    mesh = ax.pcolormesh(
        sub.lon.values, sub.lat.values, sub[var].values,
        cmap=WIND_COMPONENT_CMAP, norm=norm,
        transform=ccrs.PlateCarree(), zorder=2, shading="nearest",
    )
    ax.coastlines(lw=0.6, color="0.25", zorder=6)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.45", zorder=6)
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.55", zorder=6)
    return mesh


def draw_wind_speed_panel(ax, sub, extent, clon, norm):
    """10 m wind speed from the UGRD10m/VGRD10m components, in turbo like the
    wind figures."""
    x0, x1, y0, y1 = extent
    ax.set_extent(
        (x0 - clon, x1 - clon, y0, y1), crs=ccrs.PlateCarree(central_longitude=clon)
    )
    mesh = ax.pcolormesh(
        sub.lon.values, sub.lat.values,
        np.hypot(sub.UGRD10m.values, sub.VGRD10m.values),
        cmap="turbo", norm=norm,
        transform=ccrs.PlateCarree(), zorder=2, shading="nearest",
    )
    ax.coastlines(lw=0.6, color="0.25", zorder=6)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.45", zorder=6)
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.55", zorder=6)
    return mesh


def build_ar_panels(key, stamp, extent, field="precip", refresh=False):
    """The precip panel with the 10 m wind speed and the sea-level pressure
    beside it: same event, extent and timestep, one field per column."""
    x0, x1, y0, y1 = extent
    y1 = min(y1, FINE_LAT_MAX)
    extent = (x0, x1, y0, y1)
    clon = 0.5 * (x0 + x1)
    proj = ccrs.PlateCarree(central_longitude=clon)

    sub = load_fine_panel(key, stamp, extent, refresh)
    rate = (sub.PRATEsfc * 86400).values
    norm, cmap, levels = precip_norm(rate, rate)
    # The same saturating percentile as the precip scale, so a few extreme
    # cells do not flatten the rest of the jet.
    speed = np.hypot(sub.UGRD10m.values, sub.VGRD10m.values)
    wind_max = float(np.ceil(np.percentile(speed, PRECIP_TOP_PERCENTILE)))
    wind_norm = Normalize(vmin=0, vmax=wind_max)
    pressure = sub.PRMSL.values
    slp_norm = Normalize(vmin=np.floor(pressure.min()), vmax=np.ceil(pressure.max()))

    panel_h = AR_PANEL_WIDTH_IN * (y1 - y0) / (x1 - x0)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(3 * AR_PANEL_WIDTH_IN + 1.0, panel_h + 1.9),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(wspace=0.06),
    )
    ax_p, ax_w, ax_s = axes

    meshes = [
        draw_precip_panel(ax_p, sub, extent, clon, norm, cmap),
        draw_wind_speed_panel(ax_w, sub, extent, clon, wind_norm),
        # Turbo like the other two panels; no isobars.
        draw_slp_panel(ax_s, sub, extent, clon, slp_norm, None, None, cmap="turbo"),
    ]
    gridliners = [
        (ax, add_latlon_axes(ax, show_left=(i == 0), show_bottom=True))
        for i, ax in enumerate(axes)
    ]
    print(
        f"  {key}: precip bands {levels[0]:g}-{levels[-1]:g} mm/day, "
        f"wind speed 0-{wind_max:g} m/s, SLP {slp_norm.vmin:g}-{slp_norm.vmax:g} hPa"
    )

    # One horizontal bar under each panel, so the three share a row height and
    # the columns stay the width of the maps. Laid flat, the bar has no room
    # for the forced-on top label beside a stride label one rung below it
    # (100 | 150), so the stride label gives way.
    ticks = list(levels)
    if len(ticks) > 2 and PRECIP_LADDER.index(ticks[-1]) - PRECIP_LADDER.index(ticks[-2]) == 1:
        del ticks[-2]
    # The field name titles each panel; its bar carries only the units.
    bars = [
        (dict(ticks=ticks, format="%g", extend="max"), "Precipitation", "mm/day"),
        (dict(extend="max"), "10m wind speed", "m/s"),
        (dict(), "Sea-level pressure", "hPa"),
    ]
    for ax, mesh, (kw, title, units) in zip(axes, meshes, bars):
        ax.set_title(title, fontsize=13, color="0.15")
        cb = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", shrink=0.9, pad=0.13,
            aspect=30, **kw,
        )
        cb.set_label(units, fontsize=12)
        cb.ax.tick_params(labelsize=10)
    fit_axis_label_pads(fig, gridliners)
    # Above the panel titles, not just the map frames, so the two don't overlap.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    top = max(
        to_fig.transform(ax.title.get_window_extent())[1, 1] for ax in axes
    )
    fig.suptitle(event_time_label(stamp), fontsize=15, color="0.15", y=top + 0.02, va="bottom")

    return save(fig, key, stamp, field)


# --------------------------------------------------------------------------
# wind figures
# --------------------------------------------------------------------------
def to_lon360(lon):
    return lon + 360 if lon < 0 else lon


def _coarse_cell_edges(centers):
    """Cell boundaries implied by a 1D array of coarse grid-cell centres."""
    return np.concatenate(
        [
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        ]
    )


def _cell_index(edges, val):
    return int(np.searchsorted(edges, val, side="right") - 1)


def six_coarse_cell_box(center_lat, center_lon, n_rows=2, n_cols=3):
    """Field of view snapped to the edges of an n_rows x n_cols block of coarse
    cells around (center_lat, center_lon) -- 6 whole cells by default. The
    coarse panel it was sized for is gone, but keeping the box makes these
    figures the same field of view as figx-downslope-wind-events.py's."""
    ds = xr.open_zarr(COARSE)
    lat_e = _coarse_cell_edges(ds.lat.values)
    lon_e = _coarse_cell_edges(ds.lon.values)
    i = _cell_index(lat_e, center_lat)
    j = _cell_index(lon_e, to_lon360(center_lon))
    mid = 0.5 * (lat_e[i] + lat_e[i + 1])
    r0 = i if center_lat >= mid else i - 1
    c0 = j - (n_cols // 2)
    return (
        float(lat_e[r0]),
        float(lat_e[r0 + n_rows]),
        float(lon_e[c0]),
        float(lon_e[c0 + n_cols]),
    )


def load_topography(lat0, lat1, lon0, lon1):
    ds = xr.open_zarr(TOPO)
    topo = (
        ds.HGTsfc.sel(grid_yt=slice(lat0, lat1), grid_xt=slice(lon0, lon1))
        .rename({"grid_yt": "lat", "grid_xt": "lon"})
        .compute()
    )
    return to_signed(topo)


def draw_topography_contours(ax, topo, *, interval_m=TOPO_INTERVAL_M,
                             color="dimgray", linewidth=0.85):
    levels = np.arange(interval_m, float(topo.max()), interval_m)
    if len(levels) == 0:
        return
    ax.contour(
        topo.lon, topo.lat, topo.values, levels=levels, colors=color,
        linewidths=linewidth, alpha=0.85, transform=ccrs.PlateCarree(), zorder=3,
    )


def high_wind_blob_peaks(speed, threshold, *, n_blobs=N_WIND_BLOBS,
                         min_cells=WIND_BLOB_MIN_CELLS,
                         min_separation_frac=WIND_BLOB_MIN_SEPARATION_FRAC):
    speed = speed.transpose("lat", "lon")
    mask = speed.values >= threshold
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labels, ids)
    peaks = ndimage.maximum(speed.values, labels, ids)
    cand = ids[sizes >= min_cells]
    if len(cand) == 0:
        return []
    cand = cand[np.argsort(peaks[cand - 1])[::-1]]
    lats, lons = speed.lat.values, speed.lon.values
    min_sep = min_separation_frac * (lons.max() - lons.min())
    kept = []
    for i, j in ndimage.maximum_position(speed.values, labels, list(cand)):
        lat, lon = float(lats[i]), float(lons[j])
        if any(np.hypot(lon - kl, lat - ka) < min_sep for ka, kl, _, _ in kept):
            continue
        kept.append((lat, lon, i, j))
        if len(kept) == n_blobs:
            break
    return kept


def draw_wind_blob_arrows(ax, speed, u_da, v_da, threshold,
                          *, color="white", edge_color="black"):
    """Draw one arrow per high-wind blob; returns how many were drawn."""
    peaks = high_wind_blob_peaks(speed, threshold)
    if not peaks:
        return 0
    ue = u_da.transpose("lat", "lon").values
    ve = v_da.transpose("lat", "lon").values
    u = np.array([ue[i, j] for _, _, i, j in peaks])
    v = np.array([ve[i, j] for _, _, i, j in peaks])
    mag = np.hypot(u, v)
    mag[mag == 0] = 1.0
    speed = speed.transpose("lat", "lon")
    la, lo = speed.lat.values, speed.lon.values
    inset = WIND_ARROW_LENGTH_FRAC * (lo.max() - lo.min())
    lats = np.clip([p[0] for p in peaks], la.min() + inset, la.max() - inset)
    lons = np.clip([p[1] for p in peaks], lo.min() + inset, lo.max() - inset)
    ax.quiver(
        lons, lats, u / mag, v / mag, transform=ccrs.PlateCarree(),
        scale=1 / WIND_ARROW_LENGTH_FRAC, scale_units="width",
        width=WIND_ARROW_WIDTH_FRAC, headwidth=4.0, headlength=4.5,
        headaxislength=4.0, pivot="tail", color=color, edgecolor=edge_color,
        linewidth=0.65, zorder=6,
    )
    return len(peaks)


def draw_city_dots(ax, cities, color="white", size=120, edge_color="black"):
    for lat, lon in cities.values():
        ax.scatter(
            lon, lat, s=size, color=color, edgecolors=edge_color or "none",
            linewidths=0.8, marker="o", zorder=5, transform=ccrs.PlateCarree(),
        )


def draw_city_labels(ax, cities, fontsize=15, color="white", dot_size=120):
    stroke = "black" if color != "black" else None
    draw_city_dots(ax, cities, color=color, size=dot_size, edge_color=stroke)
    for label, (lat, lon) in cities.items():
        t = ax.annotate(
            label, xy=(lon, lat), xytext=(6, -10), textcoords="offset points",
            fontsize=fontsize, color=color, fontweight="bold", zorder=6,
            transform=ccrs.PlateCarree(),
        )
        if stroke:
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=stroke)])


def load_wind_panel(path, stamp, box, fine, pad=0.0):
    # Pad the read so the outermost cells straddle the panel edge; set_extent
    # clips back to the box. Without it the coarse pcolormesh stops half a
    # cell short and leaves a blank strip along the frame.
    lat0, lat1, lon0, lon1 = (box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad)
    ds = xr.open_zarr(path)
    sl = (
        dict(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
        if fine
        else dict(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    )
    sub = ds.isel(time=tindex(ds, stamp)).sel(**sl)
    if fine:
        sub = sub.isel(ensemble=0).rename(latitude="lat", longitude="lon")
    sub = sub[["UGRD10m", "VGRD10m"]].load()
    sub["wind_speed"] = np.hypot(sub.UGRD10m, sub.VGRD10m)
    return to_signed(sub)


def compute_wind_panel(stamp, center):
    """The 3 km wind panel, carrying its coarse-cell box as an attribute (the
    box needs the coarse store's grid, which the cache spares a read of)."""
    box = six_coarse_cell_box(*center)
    # The speed is rederived on load, so only the components are kept.
    fine = load_wind_panel(FINE, stamp, box, True, pad=0.1).drop_vars("wind_speed")
    fine.attrs["box"] = np.array(box)
    return fine


def build_wind(key, stamp, center, cities, refresh=False):
    tag = f"{CACHE_SUBDIR}/{key}_{stamp[:10]}T{stamp[11:]}"
    fine = cached_dataset(
        f"{tag}_wind.nc", lambda: compute_wind_panel(stamp, center), refresh
    )
    fine["wind_speed"] = np.hypot(fine.UGRD10m, fine.VGRD10m)
    box = tuple(float(v) for v in fine.attrs["box"])
    lat0, lat1, lon0, lon1 = box
    topo = cached_dataset(
        f"{tag}_topo.nc",
        lambda: load_topography(lat0 - 0.2, lat1 + 0.2, lon0 - 0.2, lon1 + 0.2).to_dataset(),
        refresh,
    ).HGTsfc
    vmax = float(fine.sel(lat=slice(lat0, lat1)).wind_speed.max())
    lo0, lo1 = ((lon0 + 180) % 360) - 180, ((lon1 + 180) % 360) - 180
    inbox = fine.sel(lat=slice(lat0, lat1), lon=slice(lo0, lo1))
    # The intensity bar stays the panel's own 95th percentile, over water as
    # well as land, so an arrow means the same wind speed at both events; what
    # the mask changes is only where one may be placed.
    thresh = float(np.percentile(inbox.wind_speed.values, WIND_BLOB_PERCENTILE))
    on_land = xr.DataArray(
        land_mask(inbox.lat.values, inbox.lon.values),
        dims=("lat", "lon"),
        coords={"lat": inbox.lat, "lon": inbox.lon},
    )
    # Sea cells go below any threshold rather than to NaN: the blob labelling
    # and its maximum_position both compare values, and NaN propagates through
    # them instead of simply losing.
    land_speed = inbox.wind_speed.transpose("lat", "lon").where(on_land, -1.0)

    panel_w = 8
    fig_h = panel_w * ((lat1 - lat0) / (lon1 - lon0)) + 1.8
    fig = plt.figure(figsize=(panel_w * 1.30, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.28], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    img = fine.wind_speed.plot(ax=ax, cmap="turbo", vmin=0, vmax=vmax,
                               add_colorbar=False)
    ax.add_feature(states_feature)
    ax.add_feature(cfeature.BORDERS, color="lightgrey", linewidth=1.0)
    ax.coastlines(color="lightgrey", linewidth=1.5)
    draw_topography_contours(ax, topo)
    # This also overwrites the "ensemble = ..., time = ..." title xarray's
    # .plot() writes for itself.
    ax.set_title(event_time_label(stamp), fontsize=22)
    ax.set_extent((lo0, lo1, lat0, lat1), crs=ccrs.PlateCarree())
    gridliners = [(ax, add_latlon_axes(ax, show_left=True, show_bottom=True))]
    # The city name used to ride on the coarse panel, which is gone, so the
    # remaining panel carries the label rather than a bare dot.
    draw_city_labels(ax, cities)
    n = draw_wind_blob_arrows(ax, land_speed, inbox.UGRD10m, inbox.VGRD10m, thresh)
    print(f"  {key}: {n} land wind-blob arrow(s)")

    fig.subplots_adjust(top=0.86)
    fig.canvas.draw()
    pos = ax.get_position()
    _, _, lefts, rights = gs.get_grid_positions(fig)
    sl_, sr_ = lefts[1], rights[1]
    cbar_ax = fig.add_axes(
        [sl_ + 0.05 * (sr_ - sl_), pos.y0, 0.15 * (sr_ - sl_), pos.height]
    )
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("10m wind speed [m/s]", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    fit_axis_label_pads(fig, gridliners)

    return save(fig, key, stamp, "wind")


# --------------------------------------------------------------------------
def save(fig, key, stamp, field, out_dir=None):
    out_dir = out_dir or DESTINATIONS[key]
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{PREFIX}{key}_{stamp[:10]}_{field}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# key -> how to build it. The two downslope block centres are the catalogue's,
# so those figures stay the same events as the ad_hoc versions they were
# promoted from.
#
# The AR events no longer share the catalogue's 30 x 24 domain, nor the 3:2
# shape of the wind panels' 2x3 coarse-cell block: each is now cropped onto its
# own landfalling plume, north of 35N, so the two AR figures no longer register
# against each other. ar2 keeps the catalogue's western edge at 140W, so its
# panels still reach the open-ocean end of the plume.
AR1_EXTENT = (225, 243, 35, 52)
AR2_EXTENT = (220, 245, 35, 52)
SA_CENTER = (34.0, -117.7)  # LA basin / Transverse Range passes
BORA_CENTER = (45.0, 14.7)  # Gulf of Rijeka / northern Dalmatian coast
LA = {"Los Angeles": (34.05, -118.24)}
RIJEKA = {"Rijeka": (45.33, 14.44)}

FIGURES = {
    "ar1": lambda refresh: build_precip("ar1", "2023-01-27 18", AR1_EXTENT, refresh=refresh),
    # Three columns: precip, 10 m wind speed, sea-level pressure.
    "ar2": lambda refresh: build_ar_panels("ar2", "2023-01-31 00", AR2_EXTENT, refresh=refresh),
    # ar2's pressure twin, a supplement to the precip figure in main_figures.
    "ar2_slp": lambda refresh: build_slp("ar2", "2023-01-31 00", AR2_EXTENT, SUPPLEMENTS, refresh=refresh),
    "sa1": lambda refresh: build_wind("sa1", "2023-02-22 18", SA_CENTER, LA, refresh=refresh),
    "bora1": lambda refresh: build_wind("bora1", "2023-01-29 06", BORA_CENTER, RIJEKA, refresh=refresh),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*", help=f"figure keys, of {sorted(FIGURES)}; default all")
    parser.add_argument(
        "--refresh", action="store_true", help="recompute the cached plot data"
    )
    args = parser.parse_args()

    only = set(args.figures) or set(FIGURES)
    unknown = only - set(FIGURES)
    if unknown:
        raise SystemExit(f"unknown figure(s) {sorted(unknown)}; have {sorted(FIGURES)}")
    for key in FIGURES:
        if key in only:
            FIGURES[key](args.refresh)


if __name__ == "__main__":
    main()
