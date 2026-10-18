#!/usr/bin/env python3
"""
bin_collapse.py
===============

Why effective_m falls short of the requested m, per dataset, family and m.

    cd ~/graph-summaries-for-medical-temporal-kg
    python bin_collapse.py --root ~/probabilistic_graphs/datasets/out \
        --out bin_collapse.csv

Each family loses bins for its own reason, and the reason is not visible in
effective_m alone:

  equal_width      never loses a bin, since the boundaries are j/m regardless
                   of the data.  Instead its bins sit empty.  Because
                   pi(s,p,o) = 1/d+(s), the attainable values are reciprocals,
                   which crowd near 0 and are isolated above it, so the number
                   of bins that CAN be occupied is fixed by m and the realised
                   out-degree range alone.  Reported as attainable vs occupied.

  equal_frequency  loses a boundary when two quantiles land on the same
                   repeated value, or when a quantile falls at or above
                   max(X), where the bin above it could never receive a value.
                   Reported as proposed / distinct / at-or-above-xmax.

  clustering       loses a boundary when k-means returns an empty cluster,
                   which happens when m approaches the number of distinct
                   values.  Each empty cluster is one gap fewer to cut at.
                   Reported as non-empty clusters out of m.

The authoritative effective_m comes from make_binning, the same entry point
the summarisation runs use, so the eff column here must agree with the
effective_m column of master.csv.  A mismatch means the recorded boundaries
and these refitted ones differ -- expected for clustering, which goes through
k-means, and a bug anywhere else.

Only dependencies: numpy, scipy (for clustering), and binning.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from array import array

import numpy as np

from binning import _dedupe_boundaries, _kmeans_1d, make_binning


def load_weights_fast(path: str) -> array:
    """The non-zero weights, as an array('d') rather than a list of floats.

    The larger graphs carry ~90 million weights; a Python list of floats
    costs about ten times what the packed array does.
    """
    X = array("d")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            w = float(parts[2].strip("[]"))
            if w > 0.0:
                X.append(w)
    return X


def bin_index(x: float, m: int) -> int:
    """Bin of x under the equal-width binning with m non-zero bins."""
    if x <= 0.0:
        return 0
    j = int(np.ceil(x * m - 1e-12))
    return max(1, min(j, m))


def ew_attainable(m: int, d_min: int, d_max: int) -> int:
    """How many equal-width bins a reciprocal weight could ever occupy.

    Only d <= m need enumerating: for d > m we have 1/d < 1/m, which falls in
    bin 1 whatever d is.
    """
    bins = set()
    for d in range(max(1, d_min), min(d_max, m) + 1):
        bins.add(bin_index(1.0 / d, m))
    if d_max > m:
        bins.add(1)
    return len(bins)


def diagnose(path: str, ms, seed: int = 0):
    """One row per (family, m) for the graph at path."""
    ds = os.path.basename(os.path.dirname(path))
    X = load_weights_fast(path)
    arr = np.frombuffer(X, dtype=float)
    distinct = np.unique(arr)
    xmin, xmax = float(distinct[0]), float(distinct[-1])
    d_min, d_max = int(round(1.0 / xmax)), int(round(1.0 / xmin))

    print(f"\n=== {ds}: {len(arr):,} weights, {len(distinct):,} distinct, "
          f"pi in [{xmin:g}, {xmax:g}], d+ in [{d_min}, {d_max}]")
    print(f"{'family':<17}{'m':>4}{'eff':>5}  {'why':<52}")
    print("-" * 80)

    rows = []
    for m in ms:
        # ---- equal_width ------------------------------------------------- #
        b = make_binning("equal_width", m, arr)
        occ = b.occupancy(arr)
        occupied = sum(1 for j in range(1, b.m + 1) if occ.get(j, 0) > 0)
        att = ew_attainable(m, d_min, d_max)
        why = f"attainable {att}/{m}, occupied {occupied}/{m}"
        rows.append(dict(dataset=ds, family="equal_width", m=m, effective_m=b.m,
                         occupied=occupied, attainable=att, note=why))
        print(f"{'equal_width':<17}{m:>4}{b.m:>5}  {why:<52}")

        # ---- equal_frequency --------------------------------------------- #
        qs = [j / m for j in range(1, m)]
        proposed = list(np.quantile(arr, qs)) if qs else []
        n_distinct = len({round(float(t), 12) for t in proposed})
        n_at_max = sum(1 for t in proposed if t >= xmax)
        b = make_binning("equal_frequency", m, arr)
        occ = b.occupancy(arr)
        occupied = sum(1 for j in range(1, b.m + 1) if occ.get(j, 0) > 0)
        why = (f"{len(proposed)} quantiles -> {n_distinct} distinct, "
               f"{n_at_max} at/above max(X)")
        rows.append(dict(dataset=ds, family="equal_frequency", m=m,
                         effective_m=b.m, occupied=occupied, attainable="",
                         note=why))
        print(f"{'equal_frequency':<17}{m:>4}{b.m:>5}  {why:<52}")

        # ---- clustering --------------------------------------------------- #
        if len(distinct) <= m:
            why = f"only {len(distinct)} distinct values, one cluster each"
            b = make_binning("clustering", m, arr, seed=seed)
            nonempty = len(distinct)
        else:
            labels = _kmeans_1d(arr, distinct, m, seed)
            groups = []
            for c in range(labels.max() + 1):
                vals = distinct[labels == c]
                if len(vals):
                    groups.append((vals.min(), vals.max()))
            groups.sort()
            interior = [(groups[i][1] + groups[i + 1][0]) / 2.0
                        for i in range(len(groups) - 1)]
            after = len(_dedupe_boundaries(interior)) - 2
            nonempty = len(groups)
            b = make_binning("clustering", m, arr, seed=seed)
            lost_dedupe = len(interior) - after
            why = (f"{nonempty}/{m} clusters non-empty"
                   + (f", {lost_dedupe} boundaries deduped" if lost_dedupe
                      else ""))
        occ = b.occupancy(arr)
        occupied = sum(1 for j in range(1, b.m + 1) if occ.get(j, 0) > 0)
        rows.append(dict(dataset=ds, family="clustering", m=m,
                         effective_m=b.m, occupied=occupied, attainable="",
                         note=why))
        print(f"{'clustering':<17}{m:>4}{b.m:>5}  {why:<52}")

    for r in rows:
        r["weights"] = len(arr)
        r["distinct_pi"] = len(distinct)
        r["d_min"] = d_min
        r["d_max"] = d_max
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/probabilistic_graphs/datasets/out")
    ap.add_argument("--graph-name", default="graph_weighted.wnt")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None,
                    help="dataset directory names to restrict to")
    ap.add_argument("--out", default=None, help="CSV to write")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    paths = sorted(glob.glob(f"{root}/*/{args.graph_name}"))
    if args.only:
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(p)) in args.only]
    if not paths:
        sys.exit(f"no {args.graph_name} under {root}")

    rows = []
    for p in paths:
        rows.extend(diagnose(p, args.m, args.seed))

    if args.out:
        cols = ["dataset", "weights", "distinct_pi", "d_min", "d_max",
                "family", "m", "effective_m", "occupied", "attainable", "note"]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[ok] wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())