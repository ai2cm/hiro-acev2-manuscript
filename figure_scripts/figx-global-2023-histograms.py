import os

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from series_colors import HIRO_ACE_COLOR, PERFECT_PRED_COLOR, REFERENCE_COLOR

ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-hiro-perfect-pred-global-2023"
)
HIRO_ACE_ROOT = (
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/"
    "2026-08-03-hiro-ace-global-2023"
)
PERCENTILES = (99.99, 99.9999)
COLORS = {
    "reference": REFERENCE_COLOR,
    "prediction": PERFECT_PRED_COLOR,
    "HiRO-ACE": HIRO_ACE_COLOR,
}
# reddish purple, the next unused color in the Okabe-Ito palette (see
# series_colors.py) -- specific to this script's ablation-vs-main comparison,
# so it lives here rather than in the shared palette.
ABLATION_COLOR = "#CC79A7"
COLORS["ablation"] = ABLATION_COLOR
# Okabe-Ito vermillion, matching figx-v1-to-v2-precip-histogram.py's v1 series
HIRO_V1_COLOR = "#D55E00"
COLORS["hiro_v1"] = HIRO_V1_COLOR
LABELS = {
    "reference": "X-SHiELD reference",
    "prediction": "Perfect prediction",
    "HiRO-ACE": "HiRO-ACEv2",
    "ablation": "Perfect prediction (ablation)",
    "hiro_v1": "HiROv1 (Perkins et al. 2026)",
}
# the ablation figure names the two perfect-prediction runs by model rather than
# by role, so they can be told apart from v1 and from the main figure's series
ABLATION_LABELS = {
    **LABELS,
    "prediction": "HiROv2",
    "ablation": "HiROv2-singleDenoiser",
}
TITLES = {
    "PRATEsfc": "Surface precipitation rate",
    "wind_speed": "Wind speed",
    "PRMSL": "Sea level pressure",
    "eastward_wind_at_ten_meters": "Eastward 10m wind",
    "northward_wind_at_ten_meters": "Northward 10m wind",
}
# one linestyle per percentile, so the two lines of a group are distinguishable
# from the other group as well as from each other by colour
PERCENTILE_STYLES = {99: "--",99.99: "--", 99.9999: "--"}
# the 10 m wind components are signed and roughly symmetric, so the lower tail
# is as interesting as the upper one; these mirror the upper percentiles above
LOWER_PERCENTILE_STYLES = {1: ":", 0.01: ":", 0.0001: ":"}


def is_wind_component(var: str) -> bool:
    """True for the eastward/northward 10 m wind, in either naming convention."""
    return var.endswith(("GRD10m", "_wind_at_ten_meters"))
# the same variable is named differently across output sets; either name may be
# on disk, so treat the members of each group as interchangeable when opening
VAR_ALIASES = {
    "UGRD10m": "eastward_wind_at_ten_meters",
    "eastward_wind_at_ten_meters": "UGRD10m",
    "VGRD10m": "northward_wind_at_ten_meters",
    "northward_wind_at_ten_meters": "VGRD10m",
}


def open_histogram(root: str, subdir: str, var: str) -> xr.Dataset:
    """Histogram dataset for ``var``, falling back to its alias name.

    ``VAR_ALIASES`` pairs the two conventions used for the 10 m wind
    components; whichever spelling exists on disk is opened.
    """
    for candidate in (var, VAR_ALIASES.get(var)):
        if candidate is None:
            continue
        path = f"{root}/{subdir}/{candidate}_histogram.nc"
        if os.path.exists(path):
            return xr.open_dataset(path)
    raise FileNotFoundError(
        f"no {var}_histogram.nc (or alias) in {root}/{subdir}"
    )


def trim_empty_bins(
    counts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``counts`` and ``edges`` without their all-zero leading and trailing bins.

    Only the contiguous runs of empty bins at the two ends are dropped; empty
    bins that have populated bins on both sides are kept, so gaps within the
    distribution stay visible. Returned unchanged if every bin is empty.
    """
    counts, edges = np.asarray(counts), np.asarray(edges)
    (populated,) = np.nonzero(counts)
    if len(populated) == 0:
        return counts, edges
    first, last = populated[0], populated[-1]
    # bin i spans edges[i] to edges[i + 1], so keeping bins first..last needs
    # one more edge than bins
    return counts[first : last + 1], edges[first : last + 2]


def histogram_percentile(
    counts: np.ndarray, edges: np.ndarray, percentile: float
) -> float:
    """Value at ``percentile`` of the distribution described by the histogram.

    The bin the percentile lands in is interpolated linearly, which assumes the
    samples are spread uniformly within a bin; resolution is therefore bounded
    by the bin width, and the far tail is only meaningful if the sample count is
    well above 100 / (100 - percentile).
    """
    counts = np.asarray(counts, dtype=float)
    cdf = np.cumsum(counts)
    target = cdf[-1] * percentile / 100
    i = min(int(np.searchsorted(cdf, target, side="left")), len(counts) - 1)
    below = cdf[i - 1] if i > 0 else 0.0
    within = (target - below) / counts[i] if counts[i] > 0 else 0.0
    return edges[i] + within * (edges[i + 1] - edges[i])


# (name, root, subdir[, vars]) for each series; the optional fourth element
# restricts the series to those variables, for a run that covers only some of
# them. The main figure's three series all cover every variable.
MAIN_SERIES = [
    ("reference", ROOT, "reference_histograms"),
    ("prediction", ROOT, "prediction_histograms"),
    ("HiRO-ACE", HIRO_ACE_ROOT, "prediction_histograms"),
]
# Same reference and perfect-prediction series as MAIN_SERIES, plus the
# perfect-prediction ablation run (same root, "_ablation" subdir) in place of
# HiRO-ACE, to compare the ablation directly against the main figure's
# perfect prediction. HiRO v1 only ran a precip-only perfect prediction, so it
# appears in the precipitation panel alone.
ABLATION_SERIES = [
    ("reference", ROOT, "reference_histograms"),
    ("prediction", ROOT, "prediction_histograms"),
    ("ablation", ROOT, "prediction_histograms_ablation"),
    (
        "hiro_v1",
        ROOT,
        "perfect_prediction_hiro_v1_precip_only_histograms",
        ("PRATEsfc",),
    ),
]


def plot_histogram(ax, var, series_specs, labels=LABELS):
    series = {}
    for name, root, subdir, *rest in series_specs:
        (only_vars,) = rest or (None,)
        if only_vars is not None and var not in only_vars:
            continue
        ds = open_histogram(root, subdir, var)
        counts, edges = trim_empty_bins(ds.counts.values, ds.edges.values)
        if var == "PRATEsfc":
            edges = edges * 86400  # kg/m2/s -> mm/day
        series[name] = (counts, edges)

    for name, (_counts, _edges) in series.items():
        # density, not mass per bin: prediction and reference have different
        # bin widths, so _counts / _counts.sum() is not comparable between them
        density = _counts / (_counts.sum() * np.diff(_edges))
        ax.step(
            _edges[:-1],
            density,
            where="post",
            color=COLORS[name],
            label=labels[name],
        )

    if var == "PRMSL":
        # low pressure is the extreme of interest (storms); high pressure isn't
        percentile_styles = dict(LOWER_PERCENTILE_STYLES)
    else:
        percentile_styles = dict(PERCENTILE_STYLES)
        if is_wind_component(var):
            percentile_styles.update(LOWER_PERCENTILE_STYLES)
    # offset as a fraction of the axis span, not of the value itself: PRMSL's
    # percentile values (~900) are far from zero, so a multiplicative offset
    # would shift the label far from the lines it's meant to label
    x_span = np.subtract(*ax.get_xlim()[::-1])
    for percentile, style in percentile_styles.items():
        values = [
            histogram_percentile(counts, edges, percentile)
            for counts, edges in series.values()
        ]
        for name, value in zip(series, values):
            ax.axvline(
                value, color=COLORS[name], linestyle=style, linewidth=1, alpha=0.8
            )
        # label the group once, just right of its rightmost line; x is in data
        # coordinates and y in axes fraction, so the text sits mid-plot
        ax.text(
            max(values) + 0.01 * x_span,
            0.5,
            f"{percentile:g}th",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            # two points under the base font, so it tracks any rc_context bump
            fontsize=plt.rcParams["font.size"] - 2,
            color="0.35",
        )
    #ax.set_xscale("log")
    ax.set_yscale("log")
    if var == "PRATEsfc":
        units="mm/day"
        ax.set_xlim(1, None)
    elif var == "PRMSL":
        units="hPa"
    elif var == "wind_speed" or is_wind_component(var):
        units="m/s"
    ax.set_xlabel(units)
    ax.set_ylabel(f"density ({units})$^{{-1}}$")
    ax.set_title(TITLES[var])



def save_figure(fig, out_path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)


# titles, axis/tick labels and legend at 16 pt; the base size stays at
# matplotlib's default 10 pt so the in-plot percentile tags (base - 2) stay 8 pt
FONTS = {
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}


def build_main_figure(series_specs, out_path, labels=LABELS):
    """Five histogram panels, three on top and two below.

    The bottom panels sit under the first two top columns and the legend fills
    the otherwise empty third slot of the bottom row.
    """
    with plt.rc_context(FONTS):
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3)

        axes = []
        for spec, var in zip(
            [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]],
            [
                "PRATEsfc",
                "wind_speed",
                "PRMSL",
                "eastward_wind_at_ten_meters",
                "northward_wind_at_ten_meters",
            ],
        ):
            ax = fig.add_subplot(spec)
            plot_histogram(ax, var, series_specs, labels=labels)
            axes.append(ax)

        legend_ax = fig.add_subplot(gs[1, 2])
        legend_ax.axis("off")
        # the precip panel carries every series (in the ablation figure HiRO v1
        # is precip-only), so take the handles from it
        legend_ax.legend(
            *axes[0].get_legend_handles_labels(), loc="center", frameon=False
        )

        save_figure(fig, out_path)


build_main_figure(MAIN_SERIES, "main_figures/figx-global-2023-histograms.png")
build_main_figure(
    ABLATION_SERIES,
    "main_figures/global-2023-histograms-pp-ablation.png",
    labels=ABLATION_LABELS,
)
