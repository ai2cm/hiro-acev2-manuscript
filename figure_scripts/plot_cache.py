"""Plot-ready intermediate data for the figure scripts in this directory.

Each figure script reduces its source stores to the arrays it actually draws,
saves them under ``plot_data_cache/``, and on later runs draws straight from
the saved file without opening the source stores. Delete the file (or pass the
script's ``--refresh`` flag) to recompute it.

Caches that predate this directory (``time_mean_cache/``, ``wind_rose_cache/``,
the 100 km histogram/spectra npz files, ...) were copied into
``plot_data_cache/`` under their original names, and the scripts that wrote them
keep their own formats; the helpers here are for the figures that had none.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent
CACHE_DIR = HERE / "plot_data_cache"
MAIN_FIGURES_DIR = HERE / "main_figures"
SUPPLEMENTS_DIR = HERE / "supplements"


def cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _partial(path: Path) -> Path:
    # Written beside the target and renamed into place, so an interrupted
    # compute never leaves a truncated file that a later run would trust.
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def cached_dataset(
    name: str,
    compute: Callable[[], xr.Dataset],
    refresh: bool = False,
    **open_kwargs,
) -> xr.Dataset:
    """``plot_data_cache/<name>`` (netCDF) if present, else ``compute()`` saved there."""
    path = cache_path(name)
    if path.exists() and not refresh:
        print(f"loading plot data from {path}", flush=True)
        with xr.open_dataset(path, **open_kwargs) as ds:
            return ds.load()
    ds = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(path)
    ds.to_netcdf(tmp)
    tmp.replace(path)
    print(f"wrote plot data to {path}", flush=True)
    return ds


def cached_arrays(
    name: str,
    compute: Callable[[], dict[str, np.ndarray]],
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    """``plot_data_cache/<name>`` (npz) if present, else ``compute()`` saved there."""
    path = cache_path(name)
    if path.exists() and not refresh:
        print(f"loading plot data from {path}", flush=True)
        with np.load(path, allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files}
    arrays = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(path)
    np.savez(tmp, **arrays)
    tmp.replace(path)
    print(f"wrote plot data to {path}", flush=True)
    return arrays


def cached_frame(
    name: str,
    compute: Callable[[], pd.DataFrame],
    refresh: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """``plot_data_cache/<name>`` (csv) if present, else ``compute()`` saved there."""
    path = cache_path(name)
    if path.exists() and not refresh:
        print(f"loading plot data from {path}", flush=True)
        return pd.read_csv(path, **read_kwargs)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(path)
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"wrote plot data to {path}", flush=True)
    return df
