import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import xarray as xr

from precip_scale import OCEAN_COLOR, TC_TOP_PERCENTILE, dry_masked, precip_norm

root = "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-07-16-ace2s-shieldplus-tuned-xshield-downscaled-tc-tracks-10yr"
ds=xr.open_dataset(f"{root}/north_atlantic_000_20210918T18.nc")
sample0 = ds.isel(sample=0)

TIMESTAMP = "2021-09-18 18:00 UTC"
# Half-width of the field of view, about the domain centre. The ensemble
# supplement below keeps its own, narrower one -- its panels are a third the
# width, so it does not carry the extra degrees as usefully.
HALF_WIDTH_DEG = 3.0

lat_center = float((ds.lat.min() + ds.lat.max()) / 2)
lon_center = float((ds.lon.min() + ds.lon.max()) / 2)
sample0 = sample0.sel(
    lat=slice(lat_center - HALF_WIDTH_DEG, lat_center + HALF_WIDTH_DEG),
    lon=slice(lon_center - HALF_WIDTH_DEG, lon_center + HALF_WIDTH_DEG),
)

lon_min, lon_max = float(sample0.lon.min()), float(sample0.lon.max())
lat_min, lat_max = float(sample0.lat.min()), float(sample0.lat.max())

wind_speed = (sample0["UGRD10m_predicted"]**2 + sample0["VGRD10m_predicted"]**2)**0.5
wind_speed.attrs["units"] = "m/s"

precip = (sample0["PRATEsfc_predicted"] * 86400).clip(min=0)
precip.attrs["units"] = "mm/day"

slp = sample0["PRMSL_predicted"] 
slp.attrs["units"] = "hPa"

plot_vars = {
    "Wind speed": wind_speed,
    "Sea level pressure": slp,
    "Surface precipitation rate": precip,
}

vmins = {"Wind speed": 0, "Surface precipitation rate": 0}


# These are plain axes, not a GeoAxes, so the ticks would otherwise read as
# bare numbers, and longitude on the store's 0-360 axis ("300" for 60W).
def _lon_label(lon, _pos=None):
    lon = ((lon + 180) % 360) - 180
    if np.isclose(lon, 0) or np.isclose(abs(lon), 180):
        return f"{abs(lon):g}°"
    return f"{abs(lon):g}°{'W' if lon < 0 else 'E'}"


def _lat_label(lat, _pos=None):
    if np.isclose(lat, 0):
        return "0°"
    return f"{abs(lat):g}°{'S' if lat < 0 else 'N'}"


def degree_ticks(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))


# Precipitation is drawn on the shared log-spaced band scale from precip_scale
# rather than a linear ramp: an eyewall is two orders of magnitude wetter than
# the rainbands around it, and on a linear scale everything but the core reads
# as one colour. The other two fields keep their linear ramps.
PRECIP_PANEL = "Surface precipitation rate"

fig, axes = plt.subplots(1, len(plot_vars), figsize=(5 * len(plot_vars), 5))
# wspace has to clear the widest colorbar tick label ("1010" on the pressure
# panel) and its "hPa" label, plus the next panel's own y tick labels; at
# 0.22 they touch, and at 0.32 "hPa" still crowds the precip panel.
fig.subplots_adjust(wspace=0.38, top=0.85)

for i, (ax, (name, da)) in enumerate(zip(axes, plot_vars.items())):
    units = da.attrs["units"]
    if name == PRECIP_PANEL:
        norm, cmap, precip_ticks = precip_norm(da.values, percentile=TC_TOP_PERCENTILE)
        plot_kwargs = {"cmap": cmap, "norm": norm}
        da = dry_masked(da)
        # The dry cells are coloured through the colormap's "bad" entry rather
        # than the axes background, which savefig(transparent=True) discards.
        # These axes are plain lat/lon, not a GeoAxes, so there are no land
        # polygons to draw either; the domain is open ocean bar a couple of
        # cells of Barbados, so every unshaded cell takes the sea colour.
        cmap = cmap.copy()
        cmap.set_bad(OCEAN_COLOR)
        plot_kwargs["cmap"] = cmap
    else:
        precip_ticks = None
        plot_kwargs = {"cmap": "turbo", "vmin": vmins.get(name)}
    im = da.plot(ax=ax, x="lon", y="lat", add_colorbar=False, **plot_kwargs)
    ax.set_title(name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(np.floor(lon_min), np.ceil(lon_max) + 1, 1))
    ax.set_yticks(np.arange(np.floor(lat_min), np.ceil(lat_max) + 1, 1))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    degree_ticks(ax)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    # The N/S, E/W tick labels already say which axis is which.
    ax.set_xlabel("")
    ax.set_ylabel("")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Unevenly spaced bands need their own edges as ticks; a locator choosing
    # round numbers would label places where the colour does not change.
    fig.colorbar(im, cax=cax, label=units, **({"ticks": precip_ticks} if precip_ticks else {}))

# The equal-aspect panels shrink inside their gridspec cells, so a fixed
# suptitle y leaves a wide gap above them; anchor it just above where the
# panel titles actually landed instead.
fig.canvas.draw()
titles_top = max(
    ax.title.get_window_extent().transformed(fig.transFigure.inverted()).y1
    for ax in axes
)
fig.suptitle(TIMESTAMP, fontsize=16, y=titles_top + 0.03, va="bottom")
plt.savefig('main_figures/HiRO-ACE-hurricane-example.png', dpi=400,
            transparent=True, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# Ensemble supplement: same variables, but every generated sample gets its own
# column (in place of the single sample=0 panel above) -- for eyeballing
# ensemble spread per variable, as in the perfect-prediction ensemble figures.
# ---------------------------------------------------------------------------

ENS_HALF_WIDTH_DEG = 2.0
ens = ds.sel(
    lat=slice(lat_center - ENS_HALF_WIDTH_DEG, lat_center + ENS_HALF_WIDTH_DEG),
    lon=slice(lon_center - ENS_HALF_WIDTH_DEG, lon_center + ENS_HALF_WIDTH_DEG),
)
ens_lon_min, ens_lon_max = float(ens.lon.min()), float(ens.lon.max())
ens_lat_min, ens_lat_max = float(ens.lat.min()), float(ens.lat.max())

# Row labels are wrapped (and the rows a little taller than the panels need)
# because a one-line "Surface precipitation rate" rotated alongside a 4-degree
# panel is taller than the panel and runs into the row above.
ens_vars = {
    "Wind\nspeed": (
        (ens["UGRD10m_predicted"] ** 2 + ens["VGRD10m_predicted"] ** 2) ** 0.5,
        "m/s",
    ),
    "Sea level\npressure": (ens["PRMSL_predicted"], "hPa"),
    "Surface precip.\nrate": (
        (ens["PRATEsfc_predicted"] * 86400).clip(min=0),
        "mm/day",
    ),
}

# Variables clipped at 0 in the single-sample figure above, by (unwrapped) name.
ENS_VMIN_ZERO = {"Wind speed", "Surface precip. rate"}
ENS_PRECIP_PANEL = "Surface precip.\nrate"

n_samples = ens.sizes["sample"]

fig_e, axes_e = plt.subplots(
    len(ens_vars),
    n_samples,
    figsize=(2.0 * n_samples, 2.4 * len(ens_vars)),
)

for row, (name, (da, units)) in enumerate(ens_vars.items()):
    # One color scale per variable, spanning every sample, so the panels in a
    # row are directly comparable.
    if name == ENS_PRECIP_PANEL:
        # One scale over every sample, as for the other rows, so the panels in
        # this row stay comparable.
        norm, cmap, precip_ticks = precip_norm(da.values, percentile=TC_TOP_PERCENTILE)
        cmap = cmap.copy()
        cmap.set_bad(OCEAN_COLOR)  # see the single-sample figure above
        plot_kwargs = {"cmap": cmap, "norm": norm}
        da = dry_masked(da)
    else:
        precip_ticks = None
        plot_kwargs = {
            "cmap": "turbo",
            "vmin": 0 if name.replace("\n", " ") in ENS_VMIN_ZERO else float(da.min()),
            "vmax": float(da.max()),
        }

    for col in range(n_samples):
        ax = axes_e[row, col]
        img = da.isel(sample=col).plot(
            ax=ax, x="lon", y="lat", add_colorbar=False, **plot_kwargs
        )
        ax.set_title(f"Sample {col}" if row == 0 else "", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(np.floor(ens_lon_min), np.ceil(ens_lon_max) + 1, 1))
        ax.set_yticks(np.arange(np.floor(ens_lat_min), np.ceil(ens_lat_max) + 1, 1))
        ax.set_xlim(ens_lon_min, ens_lon_max)
        ax.set_ylim(ens_lat_min, ens_lat_max)
        degree_ticks(ax)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.tick_params(labelsize=7)
        if col > 0:
            ax.set_yticklabels([])
        ax.set_ylabel("")
        if row < len(ens_vars) - 1:
            ax.set_xticklabels([])
        ax.set_xlabel("")

    cbar = fig_e.colorbar(
        img, ax=axes_e[row, :].tolist(), orientation="vertical", fraction=0.01,
        pad=0.008, aspect=30, **({"ticks": precip_ticks} if precip_ticks else {}),
    )
    cbar.set_label(units, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    axes_e[row, 0].text(
        -0.35,
        0.5,
        name,
        transform=axes_e[row, 0].transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        linespacing=1.2,
    )

fig_e.suptitle(
    "HiRO-ACEv2 North Atlantic hurricane 2021-09-18T18:00 -- all generated samples",
    fontsize=14,
)
fig_e.savefig(
    'supplements/HiRO-ACE-hurricane-example-ensemble.png',
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)
