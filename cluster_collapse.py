#!/usr/bin/env python3
"""
cluster_collapse.py
===================

Why clustering-based binning realises fewer bins than m requests, step by step.

    cd ~/graph-summaries-for-medical-temporal-kg
    python cluster_collapse.py --dataset bsbm1000 --m 25

Mirrors ``binning._kmeans_1d`` exactly, printing the state after each stage:

  1. the distinct weights and how many triples carry each
  2. the weighted-quantile seeds, and how many survive deduplication
  3. the padding, if the seeds collided -- note that the union of real and
     filler seeds is sorted and truncated to the m SMALLEST, so padding
     does not simply append
  4. the k-means assignment: which distinct values each centroid claimed,
     and which centroids claimed nothing
  5. the surviving boundaries and the resulting effective m

The effective m printed at the end is taken from ``make_binning``, the same
entry point the summarisation runs use, so it must agree with the
effective_m column of master.csv.

Only dependencies: numpy, scipy, and binning.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from binning import _dedupe_boundaries, make_binning


def load_fast(path: str) -> np.ndarray:
    a = np.fromiter((float(l.split()[2].strip("[]"))
                     for l in open(path, encoding="utf-8")
                     if len(l.split()) >= 4), dtype=float)
    return a[a > 0]


def recip(x: float) -> str:
    """Render a weight as 1/d when it is close to a reciprocal."""
    d = 1.0 / x
    return f"1/{round(d)}" if abs(d - round(d)) < 1e-6 else f"{x:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bsbm1000")
    ap.add_argument("--root", default="~/probabilistic_graphs/datasets/out")
    ap.add_argument("--graph-name", default="graph_weighted.wnt")
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=60,
                    help="cap on the per-cluster listing")
    args = ap.parse_args()

    path = os.path.join(os.path.expanduser(args.root),
                        args.dataset, args.graph_name)
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")

    arr = load_fast(path)
    distinct, counts = np.unique(arr, return_counts=True)
    m = args.m

    print(f"=== {args.dataset}: {len(arr):,} weights, {len(distinct)} distinct, "
          f"range [{distinct[0]:.6f}, {distinct[-1]:.6f}]\n")

    if len(distinct) <= m:
        print(f"[stage 0] only {len(distinct)} distinct values for m={m}; "
              f"the short-circuit applies, one cluster per value")
        return 0

    # ---- stage 1: weighted-quantile seeds ------------------------------- #
    cum = np.cumsum(counts) / counts.sum()
    raw = np.array([distinct[np.searchsorted(cum, (j + 0.5) / m)]
                    for j in range(m)], dtype=float)
    seeds = np.unique(raw)
    print(f"[stage 1] weighted-quantile seeds: {m} requested -> "
          f"{len(seeds)} distinct")
    for v in seeds:
        hits = int((raw == v).sum())
        n = int(counts[np.searchsorted(distinct, v)])
        print(f"           {v:.6f}  {recip(v):>8}  "
              f"chosen by {hits:>2} of {m} quantiles, {n:,} triples")

    # ---- stage 2: padding ------------------------------------------------ #
    padded = []
    if len(seeds) < m:
        extra = np.linspace(distinct.min(), distinct.max(), m)
        before = set(np.round(seeds, 12))
        seeds = np.unique(np.concatenate([seeds, extra]))[:m]
        padded = [v for v in seeds if round(float(v), 12) not in before]
        print(f"\n[stage 2] PADDING: {m - len(before)} seeds short, filled "
              f"from linspace({distinct.min():.6f}, {distinct.max():.6f}, {m})")
        print(f"           union sorted and truncated to the {m} smallest")
        print(f"           {len(padded)} of the final {len(seeds)} seeds are "
              f"filler, not data points")
        on_data = sum(1 for v in padded
                      if np.min(np.abs(distinct - v)) < 1e-12)
        print(f"           of those, {on_data} coincide with a real value")
    else:
        print("\n[stage 2] no padding needed")

    # ---- stage 3: k-means ------------------------------------------------ #
    from scipy.cluster.vq import kmeans2
    centroids, labels = kmeans2(distinct.reshape(-1, 1),
                                seeds.reshape(-1, 1),
                                minit="matrix", seed=args.seed)

    print(f"\n[stage 3] k-means over the {len(distinct)} distinct values\n")
    print(f"{'slot':>5}{'centroid':>12}{'values':>8}{'triples':>12}  members")
    print("-" * 78)
    nonempty, empty = [], []
    for c in range(len(seeds)):
        idx = np.where(labels == c)[0]
        cpos = float(centroids[c][0]) if c < len(centroids) else float("nan")
        if len(idx) == 0:
            empty.append((c, cpos))
            print(f"{c:>5}{cpos:>12.6f}{0:>8}{0:>12}  -- EMPTY --")
            continue
        vals = distinct[idx]
        trip = int(counts[idx].sum())
        nonempty.append((vals.min(), vals.max()))
        shown = ", ".join(recip(v) for v in vals[:6])
        if len(vals) > 6:
            shown += f", ... (+{len(vals) - 6})"
        print(f"{c:>5}{cpos:>12.6f}{len(idx):>8}{trip:>12,}  {shown}")

    print("-" * 78)
    print(f"{len(nonempty)} of {m} clusters non-empty, {len(empty)} empty")

    if empty:
        print("\n[stage 3b] the empty centroids, and the nearest real value "
              "to each")
        for c, cpos in empty:
            j = int(np.argmin(np.abs(distinct - cpos)))
            gap = abs(distinct[j] - cpos)
            print(f"           slot {c:>2} at {cpos:.6f}  nearest "
                  f"{recip(distinct[j]):>8} ({distinct[j]:.6f}), "
                  f"gap {gap:.6f}")

    # ---- stage 4: boundaries --------------------------------------------- #
    nonempty.sort()
    interior = [(nonempty[i][1] + nonempty[i + 1][0]) / 2.0
                for i in range(len(nonempty) - 1)]
    bounds = _dedupe_boundaries(interior)
    print(f"\n[stage 4] {len(interior)} midpoints proposed, "
          f"{len(bounds) - 2} survive deduplication")
    print("           boundaries: "
          + ", ".join(f"{b:.4f}" for b in bounds))

    b = make_binning("clustering", m, arr, seed=args.seed)
    print(f"\n[result] effective m = {b.m}  (requested {m})")
    if b.m != len(bounds) - 1:
        print(f"[warn] make_binning says {b.m}, this trace says "
              f"{len(bounds) - 1}; k-means did not reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())