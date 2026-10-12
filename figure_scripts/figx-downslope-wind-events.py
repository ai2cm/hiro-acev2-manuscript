#!/usr/bin/env python
"""
Produces the "region" figure for each downslope-wind case study: a
coarse/target/predicted map of the whole cropped region plus its matching
wind-speed histogram, laid out as four columns of one figure. No
per-corridor/per-pass annotation (that lives in
tmp_highlight_downslope_winds.py's other outputs for the Santa Ana event).

Covers two case studies, sharing the same layout/build code:
    - Santa Ana event (2023-02-12T18:00): Southern California downslope
      windstorm, warm/foehn-driven.
    - Bora event (2023-01-26T12:00): Adriatic katabatic outflow through the
      Gulf of Rijeka, cold/gravity-driven -- see
      scripts/find_katabatic_cold_winds.py for the detection that found it.

The cropped event fields and terrain each figure draws are cached per event in
plot_data_cache/figx-downslope-wind-events/; pass --refresh to recompute them
from the event/coarse/topography stores.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from scipy import ndimage

# A copy of wind_analysis/plot_event_snapshot.py, kept next to this script.
import plot_event_snapshot as pes

from bicubic_interp import bicubic_to_grid
from plot_cache import MAIN_FIGURES_DIR as OUT_DIR, SUPPLEMENTS_DIR, cached_dataset

CACHE_SUBDIR = "figx-downslope-wind-events"
EVENTS_DIR = Path(
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-07-hiro-perfect-pred-events"
)
# Static 3km surface height (HGTsfc), same source find_katabatic_cold_winds.py
# and find_santa_ana_winds.py use for their terrain-slope detection.
TOPO_ZARR_PATH = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr"

# Sample-distribution step lines in the region histograms: kept faint so 16
# overlapping lines don't turn to mud, but a bit more visible than 0.1.
HISTOGRAM_SAMPLE_ALPHA = 0.18

# Wind-vector arrows: how the high-wind "blobs" they sit on are picked. The
# threshold is this percentile of the *target* field, so a blob means the same
# wind speed in every panel of a figure and the target/predicted arrows are
# directly comparable. Blobs below the minimum size are single-cell specks
# whose peak isn't a meaningful local maximum, so they're dropped before the
# strongest N are kept.
WIND_BLOB_PERCENTILE = 95
WIND_BLOB_MIN_CELLS = 5
N_WIND_BLOBS = 6
# Arrow length, and the width of its shaft, as fractions of the panel width:
# small enough to read as a direction tick on one grid cell rather than an
# annotation spanning the blob.
WIND_ARROW_LENGTH_FRAC = 1 / 14
WIND_ARROW_WIDTH_FRAC = 0.006
# Minimum spacing between the blobs that get an arrow, again as a fraction of
# the panel width. Deliberately wider than an arrow: it spreads the six arrows
# over the domain instead of letting several crowd into one corridor, so only
# the strongest peak of a cluster is kept and the next-strongest blob
# elsewhere takes the free slot.
WIND_BLOB_MIN_SEPARATION_FRAC = 1 / 11


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------


def to_lon360(lon):
    return lon + 360 if lon < 0 else lon


def _coarse_cell_edges(centers):
    """Cell boundaries implied by a 1D array of coarse grid-cell centers."""
    return np.concatenate([
        [centers[0] - 0.5 * (centers[1] - centers[0])],
        0.5 * (centers[:-1] + centers[1:]),
        [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
    ])


def _cell_index(edges, val):
    return int(np.searchsorted(edges, val, side="right") - 1)


def load_topography(lat_min, lat_max, lon_min, lon_max):
    """Static surface height (m) cropped to a lat/lon box, at the 3km
    zarr's native resolution -- much finer than the event grid, so contours
    drawn from it read as smooth terrain rather than following model cells."""
    ds = xr.open_zarr(TOPO_ZARR_PATH)
    hgt = ds.HGTsfc.sel(
        grid_yt=slice(lat_min, lat_max), grid_xt=slice(lon_min, lon_max)
    ).rename({"grid_yt": "lat", "grid_xt": "lon"})
    return hgt.compute()


def draw_topography_contours(ax, topo, *, interval_m=300, color="dimgray", linewidth=1.8):
    """Elevation contour lines (land only) at a fixed interval, layered over
    the wind-speed field."""
    max_elev = float(topo.max())
    levels = np.arange(interval_m, max_elev, interval_m)
    if len(levels) == 0:
        return
    ax.contour(
        topo.lon,
        topo.lat,
        topo.values,
        levels=levels,
        colors=color,
        linewidths=linewidth,
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


# Map panels that show an input field rather than a fine-grid output: they get
# the city labels instead of the dots, and no wind-blob arrows.
COARSE_SAMPLES = ("coarse", "bicubic")


def panel_var(event, var_name, sample):
    """`var_name` for one map panel, where `sample` is "coarse"/"bicubic"/
    "target"/an integer predicted-sample index."""
    if sample in ("coarse", "bicubic", "target"):
        return event[f"{var_name}_{sample}"]
    return event[f"{var_name}_predicted"].isel(sample=sample)


def wind_blob_threshold(event, percentile=WIND_BLOB_PERCENTILE):
    """Wind speed defining a "high wind" cell, taken from the target field so
    that every panel of a figure is thresholded at the same physical value."""
    return float(np.percentile(event["wind_speed_target"].values, percentile))


def high_wind_blob_peaks(
    speed,
    threshold,
    *,
    n_blobs=N_WIND_BLOBS,
    min_cells=WIND_BLOB_MIN_CELLS,
    min_separation_frac=WIND_BLOB_MIN_SEPARATION_FRAC,
):
    """Peak cell of each of the `n_blobs` strongest high-wind blobs in
    `speed`, as a list of (lat, lon, lat_index, lon_index) ordered by
    descending peak speed. A blob is an 8-connected region of cells at or
    above `threshold`; blobs smaller than `min_cells` are dropped, and blobs
    whose peak is within `min_separation_frac` of the domain width of an
    already-kept, stronger peak are skipped so their arrows can't collide."""
    speed = speed.transpose("lat", "lon")
    mask = speed.values >= threshold
    labels, n_found = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_found == 0:
        return []

    ids = np.arange(1, n_found + 1)
    sizes = ndimage.sum(mask, labels, ids)
    peaks = ndimage.maximum(speed.values, labels, ids)
    candidates = ids[sizes >= min_cells]
    if len(candidates) == 0:
        return []
    candidates = candidates[np.argsort(peaks[candidates - 1])[::-1]]

    lats, lons = speed.lat.values, speed.lon.values
    # The map panels are aspect-locked, so a degree of latitude and a degree
    # of longitude are the same on-page distance and one plain lat/lon
    # separation radius works in both directions.
    min_separation = min_separation_frac * (lons.max() - lons.min())

    positions = ndimage.maximum_position(speed.values, labels, list(candidates))
    kept = []
    for i, j in positions:
        lat, lon = float(lats[i]), float(lons[j])
        if any(
            np.hypot(lon - kept_lon, lat - kept_lat) < min_separation
            for kept_lat, kept_lon, _, _ in kept
        ):
            continue
        kept.append((lat, lon, i, j))
        if len(kept) == n_blobs:
            break
    return kept


def draw_wind_blob_arrows(
    ax,
    speed,
    eastward,
    northward,
    threshold,
    *,
    n_blobs=N_WIND_BLOBS,
    color="white",
    edge_color="black",
):
    """One wind-direction arrow per high-wind blob in `speed`, drawn at the
    blob's peak cell using that cell's wind components. Arrows are unit
    length -- the colormap already carries speed, so they only encode
    direction -- and start at that cell pointing downwind. They're white with
    a hairline dark outline: white alone drops to a luminance ratio of ~1.3
    over turbo's light green/yellow band, and the outline keeps the glyph
    separated (worst case ~4.7) wherever the fill matches the field. A peak
    sitting on the domain edge (its blob runs off the crop) has its arrow
    nudged inward by up to one arrow length so the glyph isn't clipped by the
    panel frame."""
    peaks = high_wind_blob_peaks(speed, threshold, n_blobs=n_blobs)
    if not peaks:
        return

    eastward = eastward.transpose("lat", "lon").values
    northward = northward.transpose("lat", "lon").values
    u = np.array([eastward[i, j] for _, _, i, j in peaks])
    v = np.array([northward[i, j] for _, _, i, j in peaks])
    magnitude = np.hypot(u, v)
    magnitude[magnitude == 0] = 1.0

    speed = speed.transpose("lat", "lon")
    lat_vals, lon_vals = speed.lat.values, speed.lon.values
    arrow_length = WIND_ARROW_LENGTH_FRAC * (lon_vals.max() - lon_vals.min())
    # Tail-pivoted, so the glyph can extend a full arrow length from the cell
    # in whichever direction the wind blows.
    inset = arrow_length
    lats = np.clip(
        [lat for lat, _, _, _ in peaks],
        lat_vals.min() + inset,
        lat_vals.max() - inset,
    )
    lons = np.clip(
        [lon for _, lon, _, _ in peaks],
        lon_vals.min() + inset,
        lon_vals.max() - inset,
    )

    ax.quiver(
        lons,
        lats,
        u / magnitude,
        v / magnitude,
        transform=ccrs.PlateCarree(),
        # scale_units="width" with scale=1/frac makes a unit vector that
        # fraction of the panel width, so arrows stay the same relative size
        # in the wide main-text panels and the small ensemble ones alike.
        scale=1 / WIND_ARROW_LENGTH_FRAC,
        scale_units="width",
        width=WIND_ARROW_WIDTH_FRAC,
        headwidth=4.0,
        headlength=4.5,
        headaxislength=4.0,
        pivot="tail",
        color=color,
        edgecolor=edge_color,
        linewidth=0.65,
        zorder=6,
    )


def add_outer_latlon_grid(ax, *, show_left, show_bottom, label_fontsize=13):
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels = False
    if show_left:
        gl.left_labels = True
    if show_bottom:
        gl.bottom_labels = True
    gl.xlabel_style = {"size": label_fontsize}
    gl.ylabel_style = {"size": label_fontsize}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_aspect("equal", adjustable="box")


def draw_histogram(ax, event, var_name):
    """Wind-speed distribution (target vs. all predicted samples) for the
    whole cropped region."""
    target_data = event[f"{var_name}_target"]
    predicted_data = event[f"{var_name}_predicted"]
    all_data = np.concatenate([target_data.values[None], predicted_data.values])

    bins = 50
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    sample_data = predicted_data.values
    num_samples = sample_data.shape[0]

    counts, _ = np.histogram(target_data.values, bins=bin_edges)
    ylim_min = 0.1
    ylim_max = 10 ** (np.log10(np.max(counts)) + 1)

    all_counts = []
    for i in range(num_samples):
        sample_counts, _ = np.histogram(sample_data[i].flatten(), bins=bin_edges)
        ax.step(
            bin_centers,
            sample_counts,
            where="mid",
            color="C0",
            alpha=HISTOGRAM_SAMPLE_ALPHA,
        )
        all_counts.append(sample_counts)
    all_counts = np.stack(all_counts)

    ax.step(bin_centers, counts, where="mid", color="black", linewidth=2, label="Target")
    ax.step(
        bin_centers,
        all_counts.mean(axis=0),
        where="mid",
        color="C0",
        linewidth=2,
        label="Prediction",
    )

    var_label = var_name.replace("_", " ").title()
    ax.set_xlabel(f"{var_label} [{pes.UNITS.get(var_name, '')}]", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale("log")
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Wind speed distribution", fontsize=22)
    ax.legend(fontsize=18, loc="lower left")


# ---------------------------------------------------------------------------
# Shared figure builder
# ---------------------------------------------------------------------------


def build_region_figure(
    event,
    suptitle,
    panels,
    *,
    coastline_linewidth=2.5,
    border_linewidth=1.5,
    title_fontsize=22,
    label_fontsize=13,
    decorate_coarse=None,
    decorate_other=None,
    topography=None,
    wind_arrows=True,
    n_cols=2,
):
    """An `n_cols`-column grid of `event`'s whole (already-cropped) domain: one map
    panel per `panels` entry, filled row by row, with the wind-speed
    histogram in the slot after the last map and a colorbar spanning all
    rows on the right. `panels` is [(sample_key, title), ...] with
    sample_key one of "coarse"/"bicubic"/"target"/an int predicted-sample index, in
    reading order; its length plus one (the histogram) must fill whole
    rows. `decorate_coarse`/`decorate_other` are optional callables(ax)
    for event-specific annotation (e.g. city markers) on the coarse panel
    and the remaining panels respectively. `topography`, if given, is a
    lat/lon-indexed elevation DataArray (from `load_topography`) contoured
    onto every map panel. `wind_arrows` adds a wind-direction arrow on each
    of the strongest high-wind blobs of the non-coarse panels. `suptitle` is
    just the event timestamp -- the rest of the figure's labelling
    (coarse/target/predicted, wind speed distribution) is carried by the
    panel titles.
    """
    if (len(panels) + 1) % n_cols != 0:
        raise ValueError("map panels plus the histogram must fill whole rows")
    n_rows = (len(panels) + 1) // n_cols

    def panel_field(sample):
        return panel_var(event, "wind_speed", sample)

    vmax = max(float(panel_field(sample).max()) for sample, _ in panels)
    blob_threshold = wind_blob_threshold(event) if wind_arrows else None

    panel_width_in = 8
    lat_span = float(event.lat.max() - event.lat.min())
    lon_span = float(event.lon.max() - event.lon.min())
    panel_height_in = panel_width_in * (lat_span / lon_span)
    fig_height = n_rows * panel_height_in + 1.0

    # n_cols map columns, a narrow spacer column sized to fit the colorbar
    # (which spans all rows).
    width_ratios = [1] * n_cols + [0.45]
    fig = plt.figure(figsize=(panel_width_in * (n_cols + 0.45), fig_height))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=width_ratios, wspace=0.28, hspace=0.05)
    map_axes = [
        fig.add_subplot(gs[divmod(i, n_cols)], projection=ccrs.PlateCarree())
        for i in range(len(panels))
    ]
    hist_ax = fig.add_subplot(gs[divmod(len(panels), n_cols)])

    img = None
    for i, (ax, (sample, title)) in enumerate(zip(map_axes, panels)):
        r, c = divmod(i, n_cols)
        ws = panel_field(sample)
        img = ws.plot(ax=ax, cmap="turbo", vmin=0, vmax=vmax, add_colorbar=False)
        ax.add_feature(pes.states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey", linewidth=0.6 * border_linewidth)
        ax.add_feature(cfeature.RIVERS, color="lightgrey", linewidth=0.4 * border_linewidth)
        ax.coastlines(color="lightgrey", linewidth=coastline_linewidth)
        if topography is not None:
            draw_topography_contours(ax, topography)
        ax.set_title(title, fontsize=title_fontsize)
        add_outer_latlon_grid(
            ax, show_left=(c == 0), show_bottom=(r == n_rows - 1), label_fontsize=label_fontsize
        )
        if sample in COARSE_SAMPLES:
            if decorate_coarse is not None:
                decorate_coarse(ax)
            continue
        if decorate_other is not None:
            decorate_other(ax)
        if wind_arrows:
            draw_wind_blob_arrows(
                ax,
                ws,
                panel_var(event, "eastward_wind_at_ten_meters", sample),
                panel_var(event, "northward_wind_at_ten_meters", sample),
                blob_threshold,
            )

    draw_histogram(hist_ax, event, "wind_speed")

    all_axes = map_axes + [hist_ax]
    for ax, label in zip(all_axes, "abcdefghij"):
        ax.set_title(f"{label})", loc="left", fontsize=title_fontsize, fontweight="bold")

    # Pull the axes up close under the suptitle instead of leaving the
    # default top margin.
    fig.subplots_adjust(top=0.96)

    # A plain fraction/pad colorbar doesn't track the geoaxes' actual
    # (aspect-locked) height, so it ends up taller than the panels. Draw
    # once to finalize their real positions, then size a colorbar axes to
    # match the full top-to-bottom span of the map column exactly, placed
    # within the spacer column's own nominal bounds (not offset from the
    # shrunk axes' horizontal position) so it can't encroach on the
    # histogram regardless of how much the geoaxes shrank.
    fig.canvas.draw()
    top_pos = map_axes[0].get_position()
    bottom_map_ax = map_axes[-1]
    bottom_pos = bottom_map_ax.get_position()
    _, _, lefts, rights = gs.get_grid_positions(fig)
    spacer_left, spacer_right = lefts[n_cols], rights[n_cols]
    cbar_x0 = spacer_left + 0.05 * (spacer_right - spacer_left)
    cbar_width = 0.15 * (spacer_right - spacer_left)
    cbar_ax = fig.add_axes(
        [cbar_x0, bottom_pos.y0, cbar_width, top_pos.y1 - bottom_pos.y0]
    )
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("10m wind speed [m/s]", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # The histogram axes isn't aspect-locked like the geoaxes, so it fills
    # its whole GridSpec cell and would end up taller than the (aspect-
    # shrunk) map panel it shares a row with. Force it to the same
    # y0/height as that panel.
    hist_pos = hist_ax.get_position()
    hist_ax.set_position([hist_pos.x0, bottom_pos.y0, hist_pos.width, bottom_pos.height])

    # Centered over the map columns only, not the narrower colorbar
    # column, so it sits over the middle of the actual subplots rather than
    # the whole figure.
    suptitle_x = 0.5 * (lefts[0] + rights[n_cols - 1])
    fig.suptitle(suptitle, fontsize=title_fontsize, x=suptitle_x, y=0.995, ha="center")
    return fig


def build_ensemble_figure(
    event,
    suptitle,
    *,
    n_cols=4,
    coastline_linewidth=1.5,
    border_linewidth=1.0,
    title_fontsize=14,
    label_fontsize=9,
    decorate_coarse=None,
    decorate_other=None,
    topography=None,
    wind_arrows=True,
):
    """One map panel per predicted sample, plus the coarse and target (fine)
    panels up front for reference, no histogram -- laid out in a grid as a
    supplement to `build_region_figure`'s single-sample main-text view, for
    eyeballing ensemble spread."""
    n_samples = event.sizes["sample"]
    panels = [("coarse", "Coarse"), ("target", "Target")] + [
        (i, f"Predicted {i}") for i in range(n_samples)
    ]
    n_panels = len(panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    def panel_field(sample):
        return panel_var(event, "wind_speed", sample)

    vmax = max(float(panel_field(sample).max()) for sample, _ in panels)
    blob_threshold = wind_blob_threshold(event) if wind_arrows else None

    panel_width_in = 4
    lat_span = float(event.lat.max() - event.lat.min())
    lon_span = float(event.lon.max() - event.lon.min())
    panel_height_in = panel_width_in * (lat_span / lon_span)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width_in * n_cols, panel_height_in * n_rows + 1.2),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = np.atleast_2d(axes)

    img = None
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i >= n_panels:
            ax.axis("off")
            continue
        sample, title = panels[i]
        ws = panel_field(sample)
        img = ws.plot(ax=ax, cmap="turbo", vmin=0, vmax=vmax, add_colorbar=False)
        ax.add_feature(pes.states_feature)
        ax.add_feature(cfeature.BORDERS, color="lightgrey", linewidth=border_linewidth)
        ax.coastlines(color="lightgrey", linewidth=coastline_linewidth)
        if topography is not None:
            draw_topography_contours(ax, topography)
        ax.set_title(title, fontsize=title_fontsize)
        add_outer_latlon_grid(
            ax, show_left=(c == 0), show_bottom=(r == n_rows - 1), label_fontsize=label_fontsize
        )
        if sample == "coarse":
            if decorate_coarse is not None:
                decorate_coarse(ax)
        else:
            if decorate_other is not None:
                decorate_other(ax)
            if wind_arrows:
                draw_wind_blob_arrows(
                    ax,
                    ws,
                    panel_var(event, "eastward_wind_at_ten_meters", sample),
                    panel_var(event, "northward_wind_at_ten_meters", sample),
                    blob_threshold,
                )

    fig.subplots_adjust(top=0.92, wspace=0.15, hspace=0.25)
    cbar = fig.colorbar(img, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label("10m wind speed [m/s]", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(suptitle, fontsize=title_fontsize + 4, y=0.98)
    return fig


def _load_cropped_event(nc_path, lat_min, lat_max, lon_min, lon_max, coarse):
    event = xr.open_dataset(nc_path)
    event = event.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    event_date = pes.load_event_date(nc_path)
    event = pes.merge_coarse(event, coarse, event_date)
    event = pes.add_wind_speed(event)
    event = add_bicubic_wind(event, coarse, event_date)
    return event


# Coarse cells kept around the event box before interpolating, so the bicubic
# kernel's outer neighbours exist at the box edges (the crop boxes here are
# whole coarse cells, so without a margin the edge rows would be missing).
BICUBIC_MARGIN_DEG = 2.0


def add_bicubic_wind(event, coarse, date):
    """Add the coarse 10m wind components bicubically interpolated onto the
    event grid (see bicubic_interp.py) as "<var>_bicubic", and the wind speed
    derived from those interpolated components as "wind_speed_bicubic"."""
    coarse_at_time = coarse.sel(time=date, method="nearest")
    box = coarse_at_time.sel(
        grid_yt=slice(
            float(event.lat.min()) - BICUBIC_MARGIN_DEG,
            float(event.lat.max()) + BICUBIC_MARGIN_DEG,
        ),
        grid_xt=slice(
            float(event.lon.min()) - BICUBIC_MARGIN_DEG,
            float(event.lon.max()) + BICUBIC_MARGIN_DEG,
        ),
    )
    for var in ("eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"):
        event[f"{var}_bicubic"] = xr.DataArray(
            bicubic_to_grid(
                box[var], "grid_yt", "grid_xt", event.lat.values, event.lon.values
            ),
            dims=["lat", "lon"],
            coords={"lat": event.lat, "lon": event.lon},
        )
    event["wind_speed_bicubic"] = np.sqrt(
        event.eastward_wind_at_ten_meters_bicubic**2
        + event.northward_wind_at_ten_meters_bicubic**2
    )
    return event


# Event fields the figures draw: wind speed for the maps and histogram, and
# the components behind the wind-blob arrows.
PLOT_VAR_PREFIXES = ("wind_speed_", "eastward_wind_at_ten_meters_", "northward_wind_at_ten_meters_")


def event_plot_data(event, topography):
    """The cropped wind fields plus the terrain, in one dataset. The terrain
    is on the (much finer) 3km grid, so it gets its own topo_lat/topo_lon
    dims rather than sharing the event's lat/lon."""
    fields = event[[v for v in event.data_vars if v.startswith(PLOT_VAR_PREFIXES)]]
    ds = fields.load()
    ds["topography"] = topography.rename({"lat": "topo_lat", "lon": "topo_lon"})
    return ds


def split_plot_data(ds):
    """Inverse of `event_plot_data`: (event, topography)."""
    topography = ds["topography"].rename({"topo_lat": "lat", "topo_lon": "lon"})
    return ds.drop_dims(["topo_lat", "topo_lon"]), topography


# ---------------------------------------------------------------------------
# Santa Ana event (Southern California, warm/foehn downslope windstorm)
# ---------------------------------------------------------------------------

SANTA_ANA_NC_PATH = EVENTS_DIR / "santa_ana_winds_20230212T1800.nc"
SANTA_ANA_OUT_PATH = OUT_DIR / "wind_event_santa_ana.png"

# label -> (lat, lon in -180/180); used only to derive the crop box below.
CORRIDORS = {
    "Cajon Pass → San Bernardino/Fontana": (34.30, -117.45),
    "Santa Ana Canyon → Anaheim/Corona": (33.87, -117.65),
    "San Gorgonio Pass → Banning/Beaumont": (33.93, -116.90),
    "Newhall Pass → San Fernando Valley": (34.32, -118.52),
}

# plain urban centers, for context (not corridor mouths)
CITIES = {
    "Los Angeles": (34.05, -118.24),
}


def corridor_coarse_bbox(coarse_lat, coarse_lon):
    """Bounding box, in coarse whole-cell units, of the rectangle of coarse
    grid cells needed to cover every corridor point's own cell (2 rows x 3
    columns for this set of corridors)."""
    lat_edges = _coarse_cell_edges(coarse_lat)
    lon_edges = _coarse_cell_edges(coarse_lon)
    rows, cols = [], []
    for lat, lon in CORRIDORS.values():
        rows.append(_cell_index(lat_edges, lat))
        cols.append(_cell_index(lon_edges, to_lon360(lon)))
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return lat_edges[r0], lat_edges[r1 + 1], lon_edges[c0], lon_edges[c1 + 1]


def draw_city_dots(ax, cities, color="white", size=120, edge_color="black"):
    for lat, lon in cities.values():
        lon360 = to_lon360(lon)
        ax.scatter(
            lon360,
            lat,
            s=size,
            color=color,
            edgecolors=edge_color or "none",
            linewidths=0.8,
            marker="o",
            zorder=5,
            transform=ccrs.PlateCarree(),
        )


def draw_city_labels(ax, cities, fontsize=15, color="white", dot_size=120):
    # White (or any non-black) label text gets a black outline stroke so it
    # stays legible over both the dark-blue and light-yellow ends of the
    # wind-speed colormap.
    stroke_color = "black" if color != "black" else None
    draw_city_dots(ax, cities, color=color, size=dot_size, edge_color=stroke_color)
    for label, (lat, lon) in cities.items():
        lon360 = to_lon360(lon)
        text = ax.annotate(
            label,
            xy=(lon360, lat),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            zorder=6,
            transform=ccrs.PlateCarree(),
        )
        if stroke_color:
            text.set_path_effects([pe.withStroke(linewidth=2.5, foreground=stroke_color)])


def santa_ana_plot_data():
    coarse = pes.get_coarse_data(pes.DEFAULT_COARSE_PATHS)
    lat_min, lat_max, lon_min, lon_max = corridor_coarse_bbox(
        coarse.grid_yt.values, coarse.grid_xt.values
    )
    event = _load_cropped_event(
        SANTA_ANA_NC_PATH, lat_min, lat_max, lon_min, lon_max, coarse
    )
    topography = load_topography(lat_min, lat_max, lon_min, lon_max)
    return event_plot_data(event, topography)


def make_santa_ana_figure(refresh=False):
    event, topography = split_plot_data(
        cached_dataset(f"{CACHE_SUBDIR}/santa_ana_event.nc", santa_ana_plot_data, refresh)
    )

    fig = build_region_figure(
        event,
        "2023-02-12T18:00",
        [("target", "Target"), (0, "Predicted"), ("coarse", "Coarse")],
        coastline_linewidth=5.0,
        border_linewidth=2.0,
        decorate_coarse=lambda ax: draw_city_labels(ax, CITIES),
        decorate_other=lambda ax: draw_city_dots(ax, CITIES),
        topography=topography,
    )
    fig.savefig(SANTA_ANA_OUT_PATH, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {SANTA_ANA_OUT_PATH}")

    ensemble_fig = build_ensemble_figure(
        event,
        "Santa Ana event (2023-02-12T18:00): predicted ensemble members",
        coastline_linewidth=3.5,
        border_linewidth=1.5,
        decorate_coarse=lambda ax: draw_city_labels(ax, CITIES),
        decorate_other=lambda ax: draw_city_dots(ax, CITIES),
        topography=topography,
    )
    ensemble_out_path = SUPPLEMENTS_DIR / f"{SANTA_ANA_OUT_PATH.stem}_ensemble.png"
    ensemble_fig.savefig(ensemble_out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(ensemble_fig)
    print(f"Saved: {ensemble_out_path}")


# ---------------------------------------------------------------------------
# Bora event (Adriatic, cold/gravity-driven katabatic outflow)
# ---------------------------------------------------------------------------

BORA_NC_PATH = EVENTS_DIR / "bora_cold_katabatic_20230126T1200.nc"
BORA_OUT_PATH = OUT_DIR / "wind_event_bora.png"
# The detected Bora event center, from bora_2023_full.csv (2023-01-26T12:00),
# per find_katabatic_cold_winds.py's Adriatic scan.
BORA_EVENT_LAT, BORA_EVENT_LON = 44.99354553222656, 14.917187690734863

# Senj, the classic Bora reference site (and the wind-rose site in
# figx-bora-wind-rose.py), right at the detected outflow.
BORA_CITIES = {
    "Senj": (44.99, 14.90),
}


def _centered_range(index, n, val, center):
    """Index range [lo, hi] of length n, centered on `index`. For even n
    there's no exact center, so it's grown toward whichever side of the
    cell's own center `val` actually falls on."""
    if n % 2 == 1:
        return index - n // 2, index + n // 2
    if val >= center:
        return index - (n - 1) // 2, index + n // 2
    return index - n // 2, index + (n - 1) // 2


def event_coarse_bbox(coarse_lat, coarse_lon, lat, lon, n_rows=2, n_cols=3):
    """Edges of the n_rows x n_cols block of coarse grid cells centered on
    the cell containing (lat, lon)."""
    lat_edges = _coarse_cell_edges(coarse_lat)
    lon_edges = _coarse_cell_edges(coarse_lon)
    r = _cell_index(lat_edges, lat)
    c = _cell_index(lon_edges, lon)

    r_lo, r_hi = _centered_range(r, n_rows, lat, coarse_lat[r])
    c_lo, c_hi = _centered_range(c, n_cols, lon, coarse_lon[c])

    return lat_edges[r_lo], lat_edges[r_hi + 1], lon_edges[c_lo], lon_edges[c_hi + 1]


def bora_plot_data():
    coarse = pes.get_coarse_data(pes.DEFAULT_COARSE_PATHS)
    lat_min, lat_max, lon_min, lon_max = event_coarse_bbox(
        coarse.grid_yt.values, coarse.grid_xt.values, BORA_EVENT_LAT, BORA_EVENT_LON
    )
    event = _load_cropped_event(BORA_NC_PATH, lat_min, lat_max, lon_min, lon_max, coarse)
    topography = load_topography(lat_min, lat_max, lon_min, lon_max)
    return event_plot_data(event, topography)


def make_bora_figure(refresh=False):
    event, topography = split_plot_data(
        cached_dataset(f"{CACHE_SUBDIR}/bora_event.nc", bora_plot_data, refresh)
    )

    fig = build_region_figure(
        event,
        "2023-01-26T12:00",
        [
            ("bicubic", "Bicubic"),
            ("target", "Target"),
            (0, "Prediction 0"),
            (1, "Prediction 1"),
            (2, "Prediction 2"),
        ],
        n_cols=3,
        decorate_coarse=lambda ax: draw_city_labels(ax, BORA_CITIES),
        decorate_other=lambda ax: draw_city_dots(ax, BORA_CITIES),
        topography=topography,
    )
    fig.savefig(BORA_OUT_PATH, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {BORA_OUT_PATH}")

    ensemble_fig = build_ensemble_figure(
        event,
        "Bora event (2023-01-26T12:00): predicted ensemble members",
        decorate_coarse=lambda ax: draw_city_labels(ax, BORA_CITIES),
        decorate_other=lambda ax: draw_city_dots(ax, BORA_CITIES),
        topography=topography,
    )
    ensemble_out_path = SUPPLEMENTS_DIR / f"{BORA_OUT_PATH.stem}_ensemble.png"
    ensemble_fig.savefig(ensemble_out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(ensemble_fig)
    print(f"Saved: {ensemble_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="recompute the cached plot data"
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    make_santa_ana_figure(args.refresh)
    make_bora_figure(args.refresh)


if __name__ == "__main__":
    main()
