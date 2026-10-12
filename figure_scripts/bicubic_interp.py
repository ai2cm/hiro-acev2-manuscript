"""Bicubic interpolation of a coarse lat/lon field onto a finer lat/lon grid.

Uses the same kernel as the downscaling repo's bicubic baseline
(``torch.nn.functional.interpolate(mode="bicubic", align_corners=True)``), but
driven by coordinates via ``grid_sample``: each fine point is placed at its
fractional index within the coarse field, so the result lines up with the fine
grid regardless of the coarse field's offset or its slightly uneven latitude
spacing.
"""

import numpy as np
import torch
import xarray as xr


def bicubic_to_grid(
    field: xr.DataArray,
    lat_name: str,
    lon_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """``field`` (2D over ``lat_name``/``lon_name``) bicubically interpolated
    onto the ``lat`` x ``lon`` grid, as a ``(len(lat), len(lon))`` array.

    ``lon`` must use the same longitude convention as ``field``. ``field``
    should extend at least one coarse cell beyond the target grid on every
    side, so the kernel's outer neighbours exist at the edges.
    """
    field = field.transpose(lat_name, lon_name).sortby([lat_name, lon_name])
    src_lat = field[lat_name].values.astype(float)
    src_lon = field[lon_name].values.astype(float)

    # fractional (row, col) index of every target point, mapped to
    # grid_sample's [-1, 1] range (align_corners=True: -1 and 1 are the
    # centres of the first and last coarse cells)
    i_lat = np.interp(np.asarray(lat, dtype=float), src_lat, np.arange(len(src_lat)))
    i_lon = np.interp(np.asarray(lon, dtype=float), src_lon, np.arange(len(src_lon)))
    y = 2 * i_lat / (len(src_lat) - 1) - 1
    x = 2 * i_lon / (len(src_lon) - 1) - 1
    yy, xx = np.meshgrid(y, x, indexing="ij")
    grid = torch.from_numpy(np.stack([xx, yy], axis=-1)[None])
    src = torch.from_numpy(np.asarray(field.values, dtype=np.float64)[None, None])
    out = torch.nn.functional.grid_sample(
        src, grid, mode="bicubic", align_corners=True
    )
    return out[0, 0].numpy().astype(np.float32)
