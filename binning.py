#!/usr/bin/env python3
"""
binning.py
==========

Binning functions for probabilistic graph summarisation, implementing
Definitions 7-10 of the thesis.

    from binning import make_binning, load_weights

    X = load_weights("graph_weighted.wnt")   # non-zero weights of the graph
    B = make_binning("equal_width", m=5, X=X)
    B(0.78)     # -> 4
    B.report()  # occupancy, boundaries, collapse diagnostics

Model
-----
A *binning function* (Definition 7) is a map B : [0,1] -> B into a finite set
of bins whose preimages partition [0,1].  B is *zero-separating* if
B^-1(B(0)) = {0}.  Every binning here is zero-separating by construction:

    B(x) = 0                      if x == 0
    B(x) = j   where t_{j-1} < x <= t_j

with 0 = t_0 < t_1 < ... < t_m = 1.  Such a B has m+1 bins.  The three
functions below differ only in how t_1, ..., t_{m-1} are chosen.

    equal_width      (Def. 8)  t_j = j/m.  Independent of the data.
    equal_frequency  (Def. 9)  t_j at the (j/m)-quantile of X.
    clustering       (Def. 10) X is partitioned into m clusters by
                               one-dimensional clustering; t_j is placed at the
                               midpoint between the largest value of cluster j
                               and the smallest value of cluster j+1.

The pair (f, m) -- the binning function and the number of non-zero bins -- is
the model parameter Phi_p.  Applying it to a graph yields a concrete B, which
is computed ONCE before summarisation begins and never recomputed during
refinement, so the equivalence relation being refined does not change between
iterations.

Equal-frequency and clustering-based boundaries are derived from X and are
therefore graph-dependent: they must be reported alongside any summary for it
to be reproducible.  Use ``Binning.to_json`` / ``Binning.from_json`` for that.

Only dependencies: numpy (scipy optional, for k-means).
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np


FAMILIES = ("equal_width", "equal_frequency", "clustering")


# --------------------------------------------------------------------------- #
# the binning function itself
# --------------------------------------------------------------------------- #

class Binning:
    """A zero-separating binning function with fixed boundaries.

    Attributes
      family      one of FAMILIES, recorded for reproducibility
      m           number of non-zero bins
      boundaries  the interior and outer boundaries t_0 .. t_m, length m+1
    """

    def __init__(self, family: str, m: int, boundaries: Sequence[float]):
        if m < 1:
            raise ValueError("m must be at least 1")
        if len(boundaries) != m + 1:
            raise ValueError(
                f"expected {m + 1} boundaries for m={m}, got {len(boundaries)}"
            )
        if boundaries[0] != 0.0 or boundaries[-1] != 1.0:
            raise ValueError("boundaries must run from 0.0 to 1.0")
        if any(b >= c for b, c in zip(boundaries, boundaries[1:])):
            raise ValueError(f"boundaries must be strictly increasing: {boundaries}")

        self.family = family
        self.m = m
        self.boundaries = [float(b) for b in boundaries]
        # interior boundaries t_1 .. t_m, used for the bisect lookup
        self._cuts = self.boundaries[1:]

    # -- the map ----------------------------------------------------------- #

    def __call__(self, x: float) -> int:
        """Return the bin of x.  Bin 0 is reserved for x == 0 exactly."""
        if x < 0.0 or x > 1.0:
            raise ValueError(f"x={x} outside [0,1]")
        if x == 0.0:
            return 0
        # t_{j-1} < x <= t_j, so the interval is left-open: bisect_left finds
        # the first cut >= x, which is exactly the bin index.
        return bisect_left(self._cuts, x) + 1

    def bin_of_many(self, xs: Iterable[float]) -> List[int]:
        return [self(x) for x in xs]

    def interval(self, j: int) -> str:
        """Human-readable interval of bin j, for figures and captions."""
        if j == 0:
            return "{0}"
        return f"({self.boundaries[j - 1]:g}, {self.boundaries[j]:g}]"

    # -- properties -------------------------------------------------------- #

    @property
    def n_bins(self) -> int:
        """Total number of bins, including the zero bin."""
        return self.m + 1

    def is_zero_separating(self) -> bool:
        """True by construction; kept explicit because the model depends on it."""
        return True

    # -- reproducibility --------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(
            {"family": self.family, "m": self.m, "boundaries": self.boundaries},
            indent=2,
        )

    @classmethod
    def from_json(cls, s: str) -> "Binning":
        d = json.loads(s)
        return cls(d["family"], d["m"], d["boundaries"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Binning":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    # -- diagnostics ------------------------------------------------------- #

    def occupancy(self, X: Sequence[float]) -> Counter:
        """Count how many values of X land in each non-zero bin."""
        return Counter(self(x) for x in X)

    def report(self, X: Optional[Sequence[float]] = None) -> None:
        print(f"binning: {self.family}, m={self.m} ({self.n_bins} bins with zero)")
        print("  boundaries: " + ", ".join(f"{b:g}" for b in self.boundaries))
        if X is None:
            return
        occ = self.occupancy(X)
        used = sum(1 for j in range(1, self.m + 1) if occ.get(j, 0) > 0)
        print(f"  occupied: {used}/{self.m} non-zero bins, {len(X)} values")
        for j in range(1, self.m + 1):
            c = occ.get(j, 0)
            if c:
                print(f"    bin {j:<3} {self.interval(j):<18} {c:>10}  "
                      f"({c / len(X):.1%})")
        if used < self.m:
            print(f"  note: {self.m - used} non-zero bins are empty; the "
                  f"binning is coarser in effect than m suggests")

    def __repr__(self) -> str:
        return f"Binning({self.family!r}, m={self.m}, bins={self.n_bins})"


# --------------------------------------------------------------------------- #
# the three families (Definitions 8-10)
# --------------------------------------------------------------------------- #

def equal_width(m: int, X: Optional[Sequence[float]] = None) -> Binning:
    """Definition 8.  Boundaries at t_j = j/m, independent of X."""
    boundaries = [j / m for j in range(m + 1)]
    boundaries[0], boundaries[-1] = 0.0, 1.0
    return Binning("equal_width", m, boundaries)


def equal_frequency(m: int, X: Sequence[float]) -> Binning:
    """Definition 9.  t_j at the (j/m)-quantile of X, so each of the m non-zero
    bins receives approximately |X|/m of the observed values.

    Repeated values can make two quantiles coincide -- BSBM, for instance, has
    38 distinct weights among hundreds of thousands of edges.  Duplicate
    boundaries are dropped, which reduces m; the resulting m is recorded on the
    returned object so that Phi_p reflects what was actually used.
    """
    if X is None or len(X) == 0:
        raise ValueError("equal_frequency needs a non-empty X")
    arr = np.asarray(X, dtype=float)
    qs = [j / m for j in range(1, m)]
    interior = list(np.quantile(arr, qs)) if qs else []

    boundaries = _dedupe_boundaries(interior)
    m_eff = len(boundaries) - 1
    if m_eff < m:
        print(f"[binning] equal_frequency: {m - m_eff} boundary/boundaries "
              f"collapsed on repeated values, m {m} -> {m_eff}")
    return Binning("equal_frequency", m_eff, boundaries)


def clustering(m: int, X: Sequence[float], seed: int = 0) -> Binning:
    """Definition 10.  Partition X into m clusters by one-dimensional
    clustering and place t_j at the midpoint between the largest value of
    cluster j and the smallest value of cluster j+1.

    Uses scipy's k-means on the distinct values of X, weighted by frequency
    where available; falls back to equal-frequency over distinct values if
    scipy is not installed.
    """
    if X is None or len(X) == 0:
        raise ValueError("clustering needs a non-empty X")
    arr = np.asarray(X, dtype=float)
    distinct = np.unique(arr)

    if len(distinct) <= m:
        # Fewer distinct values than clusters: every value is its own cluster,
        # so put a boundary between each consecutive pair.
        interior = [
            (a + b) / 2.0 for a, b in zip(distinct, distinct[1:])
        ]
        boundaries = _dedupe_boundaries(interior)
        m_eff = len(boundaries) - 1
        print(f"[binning] clustering: only {len(distinct)} distinct values, "
              f"m {m} -> {m_eff}")
        return Binning("clustering", m_eff, boundaries)

    labels = _kmeans_1d(arr, distinct, m, seed)

    # order clusters by their value range and cut at the midpoints
    groups = []
    for c in range(labels.max() + 1):
        vals = distinct[labels == c]
        if len(vals):
            groups.append((vals.min(), vals.max()))
    groups.sort()
    interior = [
        (groups[i][1] + groups[i + 1][0]) / 2.0 for i in range(len(groups) - 1)
    ]

    boundaries = _dedupe_boundaries(interior)
    m_eff = len(boundaries) - 1
    if m_eff < m:
        print(f"[binning] clustering: m {m} -> {m_eff} after merging "
              f"coincident cluster boundaries")
    return Binning("clustering", m_eff, boundaries)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

def _dedupe_boundaries(interior: Sequence[float]) -> List[float]:
    """Assemble 0 < t_1 < ... < 1, dropping duplicates and out-of-range cuts."""
    seen = []
    for t in sorted(float(t) for t in interior):
        if t <= 0.0 or t >= 1.0:
            continue
        if seen and abs(t - seen[-1]) < 1e-12:
            continue
        seen.append(t)
    return [0.0] + seen + [1.0]


def _kmeans_1d(
    arr: np.ndarray, distinct: np.ndarray, m: int, seed: int
) -> np.ndarray:
    """Cluster the distinct values of arr into m groups, weighted by frequency."""
    try:
        from scipy.cluster.vq import kmeans2
    except ImportError:
        print("[binning] scipy unavailable; falling back to quantile cuts "
              "over distinct values")
        qs = [j / m for j in range(1, m)]
        cuts = np.quantile(distinct, qs) if qs else np.array([])
        return np.searchsorted(cuts, distinct)

    counts = Counter(float(x) for x in arr)
    weights = np.array([counts[float(v)] for v in distinct], dtype=float)

    # Replicating by weight would be prohibitive on large graphs, so seed the
    # centroids at weighted quantiles and let k-means refine from there.
    cum = np.cumsum(weights) / weights.sum()
    seeds = np.array([
        distinct[np.searchsorted(cum, (j + 0.5) / m)]
        for j in range(m)
    ], dtype=float)
    seeds = np.unique(seeds)
    if len(seeds) < m:  # pad if weighted quantiles collided
        extra = np.linspace(distinct.min(), distinct.max(), m)
        seeds = np.unique(np.concatenate([seeds, extra]))[:m]

    _, labels = kmeans2(
        distinct.reshape(-1, 1),
        seeds.reshape(-1, 1),
        minit="matrix",
        seed=seed,
    )
    return labels


# --------------------------------------------------------------------------- #
# factory and input
# --------------------------------------------------------------------------- #

def make_binning(
    family: str, m: int, X: Optional[Sequence[float]] = None, seed: int = 0
) -> Binning:
    """Build the binning determined by Phi_p = (family, m) and the graph.

    equal_width ignores X; the other two require it.
    """
    if family == "equal_width":
        return equal_width(m)
    if family == "equal_frequency":
        return equal_frequency(m, X)
    if family == "clustering":
        return clustering(m, X, seed=seed)
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def load_weights(path: str) -> List[float]:
    """Read the non-zero weights X from a weighted-triple file.

    Expects the format written by edge_weights.save_weighted_triples:

        <subject> <predicate> [probability] <object>
    """
    X = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            w = float(parts[2].strip("[]"))
            if w > 0.0:
                X.append(w)
    return X


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("weights", help="weighted-triple file (.wnt)")
    ap.add_argument("--family", choices=list(FAMILIES) + ["all"], default="all")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25],
                    help="numbers of non-zero bins to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None,
                    help="if given, save each binning as JSON for reproducibility")
    args = ap.parse_args()

    X = load_weights(args.weights)
    distinct = sorted(set(X))
    print(f"[info] {len(X)} non-zero weights, {len(distinct)} distinct")
    print(f"[info] range [{min(X):g}, {max(X):g}]")
    print()

    families = list(FAMILIES) if args.family == "all" else [args.family]
    for family in families:
        for m in args.m:
            B = make_binning(family, m, X, seed=args.seed)
            B.report(X)
            if args.outdir:
                os.makedirs(args.outdir, exist_ok=True)
                path = os.path.join(args.outdir, f"{family}_m{m}.json")
                B.save(path)
                print(f"  saved {path}")
            print()