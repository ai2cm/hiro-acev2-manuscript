"""Ten-year (2014-2023) time-mean precipitation: X-SHiELD reference, coarse ACE2S, and bias.

Produces two figures.

``global_coarse_res_precip_bias.png`` -- three map columns: the reference
climatology, the ACE2S ensemble-mean climatology, and their difference. The
first two share one sequential colorbar so they can be read against each other;
the bias gets its own diverging scale.

``global_coarse_res_precip_bias_ensemble_std.png`` -- six panels: each of the
five initial-condition members' own time-mean bias on a shared diverging scale,
plus their per-pixel standard deviation. Because every member is differenced
against the same reference, that spread is identically the spread of the
members' climatologies; it measures how much of the bias field is reproducible
across initial conditions rather than an artifact of one rollout. Where the
spread is comparable to the bias itself, that bias is not robust.

``precip_time_means.nc`` -- the reference and per-member time-mean fields, keyed
by initial-condition tag. Written on first run and reused on later ones, since
averaging six 14600-step stores takes minutes while the result is a handful of
180x360 fields.

The two stores are on the identical 1-degree grid and Julian calendar and differ
only in coordinate naming (grid_yt/grid_xt vs lat/lon), so the reference is
renamed rather than regridded. Both are sliced to the same 2014-2023 window
before averaging -- the reference store spans 2013-2024, so taking its mean
without slicing would compare different periods.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from plot_cache import (
    CACHE_DIR as PLOT_DATA_CACHE,
    MAIN_FIGURES_DIR,
    SUPPLEMENTS_DIR,
    cached_dataset,
)

HERE = Path(__file__).parent
COARSE_ACE2S_DATASET_ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-coarse-ace2s-10yr-inference"
)
COARSENED_XSHIELD_REFERENCE_DATASET = (
    "/climate-default/2026-07-01-X-SHiELD-AMIP-1deg-8layer-11yr.zarr"
)

REFERENCE_LABEL = "X-SHiELD"
PREDICTION_LABEL = "ACE2S"
START, END = "2014-01-01", "2023-12-31"
# PRATEsfc is kg/m**2/s; 86400 converts to the conventional mm/day.
SECONDS_PER_DAY = 86400.0
CACHE = PLOT_DATA_CACHE / "precip_time_means.nc"
# The static land fraction used for the land/ocean statistics, so that drawing
# from the cache never opens the reference store.
LAND_FRACTION_CACHE = "figx-global-coarse-res-precip-bias/land_fraction.nc"

# Sequential (magnitude): one hue, light to dark. Diverging (bias): blue for a
# dry bias through a neutral midpoint to red for a wet one.
MEAN_CMAP = "Blues"
BIAS_CMAP = "RdBu_r"
SPREAD_CMAP = "Purples"


def time_mean_precip(path: Path | str, rename: dict | None = None) -> xr.DataArray:
    """Mean PRATEsfc over [START, END] from one store, in mm/day."""
    da = xr.open_zarr(path).PRATEsfc
    if rename:
        da = da.rename(rename)
    da = da.sel(time=slice(START, END))
    return (da.mean("time") * SECONDS_PER_DAY).compute()


def area_weighted(da: xr.DataArray, mask: xr.DataArray | None = None) -> float:
    """Area mean with cos(latitude) weights, optionally restricted by a fractional mask."""
    weights = np.cos(np.deg2rad(da.lat))
    if mask is not None:
        weights = weights * mask
    return float(da.weighted(weights).mean(("lat", "lon")))


def land_fraction(refresh: bool = False) -> xr.DataArray:
    """Static land fraction on the reference grid, renamed to lat/lon.

    Time-invariant in the reference store, so a single time slice is enough --
    no averaging or slicing to [START, END] is needed.
    """

    def compute() -> xr.Dataset:
        land_frac = (
            xr.open_zarr(COARSENED_XSHIELD_REFERENCE_DATASET)
            .land_fraction.isel(time=0)
            .rename({"grid_yt": "lat", "grid_xt": "lon"})
            .compute()
        )
        return land_frac.to_dataset(name="land_fraction")

    return cached_dataset(LAND_FRACTION_CACHE, compute, refresh)["land_fraction"]


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
    ax.set_title(title, fontsize=11)
    return mesh


def add_colorbar(fig, mesh, ax, label: str, extend: str) -> None:
    fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        extend=extend,
        label=label,
    )


def plot_climatologies(reference, prediction, bias, land_frac, n_members) -> Path:
    """Figure 1: reference and prediction on a shared scale, plus their bias."""
    vmax = float(
        np.percentile(
            np.concatenate([reference.values.ravel(), prediction.values.ravel()]), 99.9
        )
    )
    # Symmetric, so zero bias sits on the neutral midpoint of the diverging ramp.
    bias_vmax = float(np.percentile(np.abs(bias.values), 99.9))

    ref_mean = area_weighted(reference)
    pred_mean = area_weighted(prediction)
    rmse = np.sqrt(area_weighted(bias**2))

    land_bias = area_weighted(bias, mask=land_frac)
    land_rmsb = np.sqrt(area_weighted(bias**2, mask=land_frac))
    ocean_bias = area_weighted(bias, mask=1 - land_frac)
    ocean_rmsb = np.sqrt(area_weighted(bias**2, mask=1 - land_frac))

    fig, axes = plt.subplots(
        1, 3, figsize=(19.5, 4.0),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    # Sequential: one hue, light to dark, for magnitude.
    mesh = plot_map(
        axes[0], reference, f"{REFERENCE_LABEL}\n{ref_mean:.2f} mm/day global mean",
        cmap=MEAN_CMAP, vmin=0, vmax=vmax,
    )
    plot_map(
        axes[1],
        prediction,
        f"{PREDICTION_LABEL} ensemble mean ({n_members} members)\n"
        f"{pred_mean:.2f} mm/day global mean",
        cmap=MEAN_CMAP, vmin=0, vmax=vmax,
    )
    bias_mesh = plot_map(
        axes[2],
        bias,
        f"{PREDICTION_LABEL} minus {REFERENCE_LABEL}\n"
        f"Global: mean bias {pred_mean - ref_mean:+.2f}, RMSB {rmse:.2f} mm/day \n"
        f"Land: mean bias {land_bias:+.2f} , RMSB {land_rmsb:.2f} mm/day \n"
        f"Ocean: mean bias {ocean_bias:+.2f} , RMSB {ocean_rmsb:.2f} mm/day",
        cmap=BIAS_CMAP, vmin=-bias_vmax, vmax=bias_vmax,
    )

    add_colorbar(fig, mesh, axes[:2], "Mean precipitation (mm/day)", "max")
    add_colorbar(fig, bias_mesh, axes[2], "Bias (mm/day)", "both")
    fig.suptitle(f"Time-Mean Precipitation, {START[:4]}-{END[:4]}", fontsize=14, y=0.95)

    out_path = MAIN_FIGURES_DIR / "global_coarse_res_precip_bias.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def plot_ensemble_spread(member_biases, bias_std) -> Path:
    """Figure 2: each member's own time-mean bias, plus their per-pixel spread.

    Laid out 2x3: the five member biases on a shared diverging scale, then the
    spread map in the last cell on its own sequential scale. Reading them
    together is the point -- a bias feature that repeats across all five panels
    is systematic, one that moves is initial-condition noise, and the spread map
    summarises exactly that.
    """
    bias_vmax = float(np.percentile(np.abs(member_biases.values), 99.9))
    std_vmax = float(np.percentile(bias_std.values, 99.9))
    proj = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(19.5, 8.5))
    # Explicit margins: the two colorbars are placed by hand below the grid, so
    # the layout must not be resized by bbox_inches="tight" mid-construction.
    gs = fig.add_gridspec(2, 3, hspace=0.02, wspace=0.06, bottom=0.16, top=0.88)
    axes = [fig.add_subplot(gs[i, j], projection=proj) for i in range(2) for j in range(3)]

    for ax, (label, bias) in zip(axes, zip(member_biases.member.values, member_biases)):
        # Global-mean precipitation is energy-constrained, so every member's
        # global bias rounds to the same 2 dp; RMSE is what actually separates
        # them, and it is the honest summary of a field this spatially
        # structured -- the near-zero global mean is cancellation, not skill.
        mesh = plot_map(
            ax,
            bias,
            f"{label}\n{area_weighted(bias):+.3f} mm/day global bias, "
            f"{np.sqrt(area_weighted(bias**2)):.3f} RMSE",
            cmap=BIAS_CMAP, vmin=-bias_vmax, vmax=bias_vmax,
        )
    std_mesh = plot_map(
        axes[-1],
        bias_std,
        f"Std. dev. across {member_biases.sizes['member']} members\n"
        f"{area_weighted(bias_std):.3f} mm/day global mean",
        cmap=SPREAD_CMAP, vmin=0, vmax=std_vmax,
    )

    # Grid spans x in [0.125, 0.9] over 3 equal columns; the bias bar sits under
    # the left two, the spread bar under the third, matching what each labels.
    cax_bias = fig.add_axes([0.16, 0.09, 0.44, 0.022])
    cax_std = fig.add_axes([0.68, 0.09, 0.18, 0.022])
    fig.colorbar(mesh, cax=cax_bias, orientation="horizontal", extend="both",
                 label="Member time-mean bias (mm/day)")
    fig.colorbar(std_mesh, cax=cax_std, orientation="horizontal", extend="max",
                 label="Std. dev. of bias (mm/day)")
    fig.suptitle(
        f"{PREDICTION_LABEL} Per-Member Time-Mean Precipitation Bias vs "
        f"{REFERENCE_LABEL}, {START[:4]}-{END[:4]}",
        fontsize=14,
        y=0.95
    )

    out_path = SUPPLEMENTS_DIR / "global_coarse_res_precip_bias_ensemble_std.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    return out_path


def compute_time_means(refresh: bool = False) -> xr.Dataset:
    """Reference and per-member time means, cached to avoid recomputing.

    Averaging six 14600-step stores takes several minutes; the result is two
    small 180x360 fields, so caching them makes replotting instant. Delete
    CACHE (or pass --refresh) to force a recompute after changing START/END or
    the member stores.
    """
    if CACHE.exists() and not refresh:
        print(f"loading plot data from {CACHE}")
        return xr.open_dataset(CACHE).load()

    # ic???? rather than ic*: the root also holds output_6hourly_predictions_
    # ic0000_2023-onward.zarr, a 1-year continuation of ic0000, not a member.
    member_paths = sorted(
        Path(COARSE_ACE2S_DATASET_ROOT).glob("output_6hourly_predictions_ic????.zarr")
    )

    print(f"reference time mean: {COARSENED_XSHIELD_REFERENCE_DATASET}")
    reference = time_mean_precip(
        COARSENED_XSHIELD_REFERENCE_DATASET, rename={"grid_yt": "lat", "grid_xt": "lon"}
    )

    member_means = []
    for path in member_paths:
        print(f"prediction time mean: {path.name}")
        member_means.append(time_mean_precip(path))

    # Label members by their initial-condition tag (ic0000, ...) so the stored
    # file and the figure panels are self-describing.
    labels = [path.stem.split("_")[-1] for path in member_paths]
    members = xr.concat(member_means, dim="member").assign_coords(member=labels)

    cached = xr.Dataset({"reference": reference, "members": members})
    for name, da in cached.data_vars.items():
        da.attrs.update(units="mm/day", long_name=f"{name} time-mean precipitation")
    cached.attrs.update(
        start=START,
        end=END,
        reference_dataset=COARSENED_XSHIELD_REFERENCE_DATASET,
        prediction_root=COARSE_ACE2S_DATASET_ROOT,
    )
    CACHE.parent.mkdir(exist_ok=True)
    cached.to_netcdf(CACHE)
    print(f"wrote time means to {CACHE}")
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="recompute and overwrite the cached time means and land fraction",
    )
    args = parser.parse_args()
    ProgressBar(minimum=2.0).register()

    cached = compute_time_means(args.refresh)
    reference, members = cached["reference"], cached["members"]
    land_frac = land_fraction(args.refresh)

    prediction = members.mean("member")
    bias = prediction - reference
    member_biases = members - reference
    # Sample std (ddof=1) across the 5 members of each member's own bias.
    member_bias_std = member_biases.std("member", ddof=1)

    n_members = members.sizes["member"]
    print(f"Wrote {plot_climatologies(reference, prediction, bias, land_frac, n_members)}")
    print(f"Wrote {plot_ensemble_spread(member_biases, member_bias_std)}")


if __name__ == "__main__":
    main()
