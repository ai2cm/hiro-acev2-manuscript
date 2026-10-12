"""One precipitation colour scale, shared by every snapshot figure that shades it.

Rain rate spans three orders of magnitude inside a single snapshot -- drizzle
ahead of a front, an orographic or eyewall core behind it -- so an evenly
spaced ramp puts almost the whole field in the bottom of the scale. These
figures instead shade discrete bands on a log-spaced ladder, which gives the
light and heavy parts of one panel comparable amounts of colour.

The ladder's top is chosen per figure, from the data, rather than fixed: a
weaker event drawn on a scale sized for a stronger one reads as a uniform
mid-band. It is the first rung at or above the upper percentile of the fields
being drawn, not their maximum -- an extreme is a handful of cells, and letting
them set the top costs the panel its contrast. They saturate instead, which is
what the bar's arrowhead means.

Every figure drawing precipitation imports from here, so a change to the ladder
moves them together and two figures never disagree about what a colour means.
Within one figure all panels share a scale, which is the comparison those
figures exist to make; across figures the tops may differ.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

# mm/day, log-spaced six to a decade (1, 1.5, 2, 3, 5, 7): each rung is roughly
# 1.5x the one below. It runs to 1000 because a tropical cyclone eyewall gets
# there -- the synoptic figures stop far short, and pick their own top. The
# 1-2-5 rungs of the coarser ladder this refines are all still on it, so a
# figure's top lands on the same round number as before.
PRECIP_LADDER = [
    0.5, 0.7,
    1, 1.5, 2, 3, 5, 7,
    10, 15, 20, 30, 50, 70,
    100, 150, 200, 300, 500, 700,
    1000,
]
# The bottom rung doubles as the floor: drier cells are left unshaded, so a dry
# panel shows whatever is behind it (a land/ocean basemap, or nothing) rather
# than a wash of the darkest band.
PRECIP_FLOOR = PRECIP_LADDER[0]
PRECIP_CMAP = "turbo"
PRECIP_TOP_PERCENTILE = 99.9
# Tropical cyclones use a higher one. An eyewall is a far smaller fraction of
# its panel than a frontal band is of a synoptic one, so the 99.9th percentile
# falls below the eyewall and saturates the very feature the figure is of.
TC_TOP_PERCENTILE = 99.99

# Basemap under the shading, where a figure has one: dry cells fall below the
# floor and are left unshaded, so what shows through them should read as
# geography rather than as a colour on the precipitation scale.
LAND_COLOR = "white"
OCEAN_COLOR = "#dceaf5"
# A floor under the top of the scale, so a very light event still gets a ramp
# rather than two bands. Expressed as a rain rate, not a number of rungs, so
# refining the ladder does not silently shrink it.
MIN_TOP = 5.0
# Label every Nth rung. At six rungs to a decade a label on each one is denser
# than a small colorbar can carry; every second one falls on the 1-2-5 series,
# which is what a reader looks for anyway, while every band keeps its colour.
TICK_STRIDE = 2


def precip_levels(*value_arrays, percentile=PRECIP_TOP_PERCENTILE):
    """The ladder truncated so its last rung is the top of this figure's scale.

    Every field that will share the scale is passed in; the rung is chosen from
    whichever of them reaches highest at ``percentile``. A coarse field is
    included on the same footing: it has few enough cells that its upper
    percentile is essentially its maximum, so a coarse panel does not saturate
    just because a high-resolution one alongside it has a longer tail.
    """
    needed = max(
        float(np.nanpercentile(np.asarray(values, dtype=float), percentile))
        for values in value_arrays
    )
    top = next(
        (i for i, level in enumerate(PRECIP_LADDER) if level >= max(needed, MIN_TOP)),
        len(PRECIP_LADDER) - 1,
    )
    return PRECIP_LADDER[: top + 1]


def precip_ticks(levels, stride=TICK_STRIDE):
    """The rungs to label: every ``stride``th, and always the top one.

    The top is forced on because it is the only rung that says where the scale
    saturates, and whether it happens to fall on the stride is an accident of
    how far the event reached.
    """
    ticks = list(levels[::stride])
    if ticks[-1] != levels[-1]:
        ticks.append(levels[-1])
    return ticks


def precip_norm(*value_arrays, percentile=PRECIP_TOP_PERCENTILE):
    """``(norm, cmap, ticks)`` for this figure's precipitation panels.

    ``ticks`` are the labelled subset of the band edges. They have to be given
    explicitly: the bands are unevenly spaced, so a tick locator picking its
    own round numbers would label positions where the colour does not change.
    """
    levels = precip_levels(*value_arrays, percentile=percentile)
    cmap = plt.get_cmap(PRECIP_CMAP)
    return BoundaryNorm(levels, cmap.N, extend="max"), cmap, precip_ticks(levels)


def dry_masked(da):
    """``da`` with sub-floor cells dropped, for a field plotted through xarray."""
    return da.where(da >= PRECIP_FLOOR)
