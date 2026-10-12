"""Fill the 100 km histogram cache for the ACE2S ensemble members.

``figx-100km-global-histograms.py`` caches the reference and ic0000, the two
series that figure draws. This script caches the rest of the ensemble, so that a
figure pairing a distribution with an ensemble-mean map can show the ensemble
rather than one initial condition.

Each member is a decade of 6-hourly global fields -- five variables of about
3.8 GB each, on a latency-bound network filesystem -- so the work is per-member
and resumable: one npz per member, written atomically, and a member whose file
already exists is skipped. Killing this job costs at most the members still in
flight, and rerunning it picks up where it stopped.

Members run in separate processes because the pass is bound by read latency
rather than by CPU or memory, and each still uses its own thread pool inside
``load_arrays``.

Every member is binned on ic0000's edges rather than its own, so the five are on
one grid and their counts can be summed into an ensemble distribution or read
bin by bin against each other. What a shared grid cannot hold it drops, so each
member also records how many values fell outside it and the range it actually
spanned; this script prints both, and a member whose tail reaches past ic0000's
is visible rather than quietly clipped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from global_histogram_common import (
    cached_edges,
    histogram_source,
    is_source_cached,
    member_source_key,
    write_source_cache,
)

# The member whose bin edges the others are binned on. It is already cached, and
# being the first initial condition it is no more arbitrary a choice than any
# other -- what matters is that all five share one grid.
EDGE_SOURCE_MEMBER = "ic0000"

COARSE_ACE2S_DATASET_ROOT = Path(
    "/climate-default/2026-08-03-hiro-ace-v2-outputs/2026-08-03-coarse-ace2s-10yr-inference"
)


def member_stores() -> dict[str, str]:
    """Cache key -> store path for every ACE2S member on disk."""
    # ic???? rather than ic*: the root also holds output_6hourly_predictions_
    # ic0000_2023-onward.zarr, a 1-year continuation of ic0000, not a member.
    stores = sorted(COARSE_ACE2S_DATASET_ROOT.glob("output_6hourly_predictions_ic????.zarr"))
    return {member_source_key(p.stem.split("_")[-1]): str(p) for p in stores}


def _task(item: tuple[str, str, dict]) -> str:
    source, path, edges = item
    print(f"computing histograms for {source}: {path}", flush=True)
    result = histogram_source(source, path, edges=edges)
    out = write_source_cache(source, result)
    # Say what the shared grid could not hold, per variable, while the numbers
    # are in hand: a clipped tail is the one way this binning can mislead.
    for var in sorted(edges):
        below, above = result[f"{source}__{var}__outside"]
        if below or above:
            lo, hi = result[f"{source}__{var}__range"]
            grid = edges[var]
            print(
                f"  {source} {var}: {below} below / {above} above ic0000's "
                f"[{grid[0]:.4g}, {grid[-1]:.4g}]; spans [{lo:.4g}, {hi:.4g}]",
                flush=True,
            )
    return f"{source} -> {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="how many members to read concurrently",
    )
    args = parser.parse_args()

    stores = member_stores()
    edge_source = member_source_key(EDGE_SOURCE_MEMBER)
    if not is_source_cached(edge_source):
        raise SystemExit(
            f"{EDGE_SOURCE_MEMBER} ({edge_source}) supplies the bin edges but is not "
            "cached; run figx-100km-global-histograms.py first"
        )
    edges = cached_edges(edge_source)
    print(f"binning on {EDGE_SOURCE_MEMBER}'s edges for: {', '.join(sorted(edges))}")

    pending = [
        (source, path, edges)
        for source, path in stores.items()
        if not is_source_cached(source)
    ]
    cached = [source for source in stores if is_source_cached(source)]
    print(f"{len(stores)} members found; {len(cached)} already cached ({', '.join(cached)})")
    if not pending:
        print("nothing to do")
        return

    print(f"computing {len(pending)}: {', '.join(source for source, _, _ in pending)}", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        for done in pool.map(_task, pending):
            print(f"done {done}", flush=True)
    print("all members cached", flush=True)


if __name__ == "__main__":
    main()
