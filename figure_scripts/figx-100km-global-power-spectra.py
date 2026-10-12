"""Global 10-year (2014-2023) spherical power spectra at 100 km resolution:
the X-SHiELD reference vs. the coarse ACE2S rollout, computed directly from
the underlying zarr stores (no precomputed spectrum files at this
resolution).

Same datasets, time window, and field set as figx-100km-global-histograms.py.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_harmonics
import xarray as xr

from plot_cache import SUPPLEMENTS_DIR, cached_arrays
from series_colors import HIRO_ACE_COLOR, REFERENCE_COLOR

HERE = Path(__file__).parent

REFERENCE_DATASET = "/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr"
ACE_DATASET = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-coarse-ace2s-10yr-inference/output_6hourly_predictions_ic0000.zarr"
)
SOURCES = {"reference": REFERENCE_DATASET, "HiRO-ACE": ACE_DATASET}
# the ACE2S rollout only covers this decade; the reference store spans a wider
# 2013-2024 window, so both are sliced to the same period before transforming
START, END = "2014-01-01", "2023-12-31"

COLORS = {"reference": REFERENCE_COLOR, "HiRO-ACE": HIRO_ACE_COLOR}
LABELS = {
    "reference": "X-SHiELD reference (100 km)",
    "HiRO-ACE": "HiRO-ACEv2 (100 km)",
}
TITLES = {
    "PRATEsfc": "Surface precipitation rate",
    "wind_speed": "Wind speed",
    "PRMSL": "Sea level pressure",
    "eastward_wind_at_ten_meters": "Eastward 10m wind",
    "northward_wind_at_ten_meters": "Northward 10m wind",
}
FIELDS = list(TITLES)
# the "legendre-gauss" grid matches fme's own default for lat/lon data (see
# fme.core.gridded_ops.LatLonOperations), and these stores' latitudes are a
# Gaussian grid (poles excluded, e.g. 89.24 rather than 90), not equiangular
SHT_GRID = "legendre-gauss"
# timesteps per read: bounds memory (chunk * 180 * 360 * 4 bytes ~= 130 MB)
# without materially affecting runtime, since the transform itself is cheap
# relative to zarr I/O
CHUNK_SIZE = 500
# computing these spectra means loading a decade of 6-hourly global fields
# through the spherical harmonic transform, so the result is cached to disk
# and reused on later runs (a name relative to plot_data_cache/)
CACHE = "global_100km_10yr_power_spectra.npz"
OUT_PATH = SUPPLEMENTS_DIR / "global-100km-10yr-power-spectra.png"

# the same variable is named differently across output sets; either name may
# be on disk, so treat the members of each group as interchangeable
VAR_ALIASES = {
    "UGRD10m": "eastward_wind_at_ten_meters",
    "eastward_wind_at_ten_meters": "UGRD10m",
    "VGRD10m": "northward_wind_at_ten_meters",
    "northward_wind_at_ten_meters": "VGRD10m",
}


def latlon_sizes(ds: xr.Dataset) -> tuple[int, int]:
    """(nlat, nlon) for ``ds``, whose latitude/longitude dims are named
    lat/lon in the ACE output but grid_yt/grid_xt in the X-SHiELD reference.
    """
    lat_dim = "lat" if "lat" in ds.sizes else "grid_yt"
    lon_dim = "lon" if "lon" in ds.sizes else "grid_xt"
    return ds.sizes[lat_dim], ds.sizes[lon_dim]


def resolve_var(ds: xr.Dataset, var: str) -> xr.DataArray | None:
    """The data variable in ``ds`` matching ``var``, under either naming
    convention, or None if neither is present."""
    for candidate in (var, VAR_ALIASES.get(var)):
        if candidate and candidate in ds.data_vars:
            return ds[candidate]
    return None


def iter_field_chunks(ds: xr.Dataset, var: str) -> Iterator[np.ndarray]:
    """Consecutive (time, lat, lon) chunks of ``var``, deriving ``wind_speed``
    from the two 10 m wind components (whichever naming convention the store
    uses) when needed. Yields nothing if the field isn't available.
    """
    n = ds.sizes["time"]
    if var == "wind_speed":
        east = resolve_var(ds, "eastward_wind_at_ten_meters")
        north = resolve_var(ds, "northward_wind_at_ten_meters")
        if east is None or north is None:
            return
        for start in range(0, n, CHUNK_SIZE):
            time_slice = slice(start, start + CHUNK_SIZE)
            yield np.hypot(
                east.isel(time=time_slice).values, north.isel(time=time_slice).values
            )
    else:
        da = resolve_var(ds, var)
        if da is None:
            return
        scale = 86400.0 if var == "PRATEsfc" else 1.0  # kg/m2/s -> mm/day
        for start in range(0, n, CHUNK_SIZE):
            yield da.isel(time=slice(start, start + CHUNK_SIZE)).values * scale


def mean_power_spectrum(
    ds: xr.Dataset, var: str, sht: torch_harmonics.RealSHT
) -> np.ndarray | None:
    """Spherical power spectrum of ``var``, averaged over all time samples.

    This is the mean of the per-timestep spectra, not the spectrum of the
    time-mean field: the two differ because the power spectrum is a nonlinear
    (quadratic) function of the field, so time-averaging must happen after
    the transform.
    """
    # Imported here, not at module level: only a recompute needs fme, so the
    # figure can be redrawn from the cache in an environment without it.
    from fme.core.metrics import spherical_power_spectrum

    total = None
    count = 0
    for chunk in iter_field_chunks(ds, var):
        tensor = torch.from_numpy(chunk).float()
        spectrum = spherical_power_spectrum(tensor, sht)
        chunk_sum = spectrum.sum(dim=0)
        total = chunk_sum if total is None else total + chunk_sum
        count += chunk.shape[0]
    if total is None:
        return None
    return (total / count).numpy()


def compute_spectra() -> dict[str, np.ndarray]:
    """Mean power spectrum for every (source, variable) pair found in
    SOURCES, plus the shared wavenumber axis under the "degree" key."""
    result = {}
    sht = None
    for source, path in SOURCES.items():
        print(f"computing power spectra for {source}: {path}")
        ds = xr.open_zarr(path).sel(time=slice(START, END))
        if sht is None:
            nlat, nlon = latlon_sizes(ds)
            sht = torch_harmonics.RealSHT(nlat, nlon, grid=SHT_GRID)
            result["degree"] = np.arange(sht.lmax)
        for var in FIELDS:
            spectrum = mean_power_spectrum(ds, var, sht)
            if spectrum is not None:
                result[f"{source}__{var}"] = spectrum
    return result


def load_spectra(refresh: bool = False) -> dict[str, np.ndarray]:
    """Cached spectra, computing and writing CACHE on first run."""
    return cached_arrays(CACHE, compute_spectra, refresh)


def plot_spectrum(ax, var, spectra, show_legend=False):
    degree = spectra["degree"]
    for source in SOURCES:
        key = f"{source}__{var}"
        if key not in spectra:
            continue
        # degree 0 (the global mean) is dropped so the log-log axes are valid
        ax.plot(
            degree[1:],
            spectra[key][1:],
            color=COLORS[source],
            label=LABELS[source],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    if var == "PRATEsfc":
        units = "mm/day"
    elif var == "PRMSL":
        units = "hPa"
    else:
        units = "m/s"
    ax.set_xlabel("spherical harmonic degree")
    ax.set_ylabel(f"power ({units}$^2$)")
    if show_legend:
        ax.legend()
    ax.set_title(TITLES[var])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="recompute and overwrite the cached spectra"
    )
    args = parser.parse_args()
    spectra = load_spectra(args.refresh)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 6)
    top_row = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]
    bottom_row = [gs[1, 1:3], gs[1, 3:5]]

    for i, (spec, var) in enumerate(
        zip(top_row, ["PRATEsfc", "wind_speed", "PRMSL"])
    ):
        plot_spectrum(fig.add_subplot(spec), var, spectra, show_legend=(i == 0))
    for spec, var in zip(
        bottom_row, ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]
    ):
        plot_spectrum(fig.add_subplot(spec), var, spectra)

    fig.suptitle(
        f"Global power spectra, {START[:4]}-{END[:4]}, 100 km resolution",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, transparent=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
