#!/usr/bin/env python3
"""
binning.py
==========

Binning functions for probabilistic graph summarisation, implementing
Definitions 7-10 of the thesis.

    from binning import make_binning
    from Collector import SchemaCollector, mass_values

    summaries = SchemaCollector(model).collect_masses(G)
    X = mass_values(summaries)               # non-zero cell masses mu(s,P,O)
    B = make_binning("equal_frequency", m=5, X=X)
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

What X is
---------
X is the multiset of values the boundaries are fitted to. The binning is
applied to the cell masses mu(s,P,O), so X is those masses: the non-zero
cell masses of every vertex, collected for the summary model before any
partition is computed (Collector.mass_values). Fitting to the values the
binning is applied to is what makes equal-frequency equal-frequency in the
masses being compared.

Earlier versions fitted X to the edge weights pi(s,p,o) instead and applied
the result to mu. The two distributions coincide only where a cell holds a
single triple. load_weights still reads the edge weights, for comparison
runs (run_summary.py --fit-on pi) and for measures.py fidelity.

The pair (f, m) -- the binning function and the number of non-zero bins -- is
the model parameter Phi_p.  Applying it to the masses yields a concrete B,
which is computed ONCE per model, after the masses are collected and before
the partition is computed, and never recomputed during refinement, so the
equivalence relation being refined does not change between iterations.

Equal-frequency and clustering-based boundaries are derived from X and are
therefore graph- and model-dependent: they must be reported alongside any
summary for it to be reproducible.  Use ``Binning.to_json`` /
``Binning.from_json`` for that.

Floating-point noise
--------------------
A cell mass is a sum of edge weights, and two sums of the same rational can
differ in the last bits: 1/15 added three times is 0.20000000000000004, not
0.2. Left alone, a boundary at 0.2 would put the two in different bins.
Values and boundaries are therefore rounded to SNAP_DECIMALS decimal places
before they are compared, which makes equal rationals equal again without
merging any two genuinely different masses: under the reciprocal weights,
two distinct masses k/d and k'/d' differ by at least 1/(d d'), which is
above 1e-9 even at the largest outdegrees in the datasets.

Only dependencies: numpy (scipy optional, for k-means).
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np


FAMILIES = ("equal_width", "equal_frequency", "clustering")

# Decimal places to which values and boundaries are rounded before they are
# compared. None switches the rounding off and reproduces the old behaviour.
SNAP_DECIMALS: Optional[int] = 12


def snap(x: float) -> float:
    """Round x to SNAP_DECIMALS places, removing summation noise."""
    if SNAP_DECIMALS is None:
        return float(x)
    return round(float(x), SNAP_DECIMALS)


def _snap_array(arr: np.ndarray) -> np.ndarray:
    if SNAP_DECIMALS is None:
        return arr
    return np.round(arr, SNAP_DECIMALS)


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
        # value -> bin, filled as values are binned. The masses take few
        # distinct values compared with the number of cells, so most lookups
        # are answered here. Bounded so that a pathological input cannot
        # exhaust memory.
        self._memo: dict = {}

    # -- the map ----------------------------------------------------------- #

    def __call__(self, x: float, tol: float = 1e-6) -> int:
        """Return the bin of x.  Bin 0 is reserved for x == 0 exactly.

        Accumulated mass can exceed 1 by a small amount: summing 1/9 nine
        times gives 1.0000000002. Values within tol of the interval are
        clamped rather than rejected, since the excess is arithmetic rather
        than a genuine violation of Definition 1.

        x is rounded to SNAP_DECIMALS places before the lookup, so that two
        sums of the same rational land in the same bin. Any x > 0 lands in a
        non-zero bin, however small, so the binning stays zero-separating.
        """
        j = self._memo.get(x)
        if j is not None:
            return j
        if x < -tol or x > 1.0 + tol:
            raise ValueError(f"x={x} outside [0,1]")
        if x <= 0.0:
            j = 0
        else:
            j = bisect_left(self._cuts, snap(min(x, 1.0))) + 1
        if len(self._memo) < 1_000_000:
            self._memo[x] = j
        return j

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

    def to_dict(self) -> dict:
        """Plain-dict form, for collecting several binnings into one file."""
        return {
            "family": self.family,
            "m": self.m,
            "boundaries": self.boundaries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Binning":
        return cls(d["family"], d["m"], d["boundaries"])

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
    boundaries = [snap(j / m) for j in range(m + 1)]
    boundaries[0], boundaries[-1] = 0.0, 1.0
    return Binning("equal_width", m, boundaries)


def equal_frequency(m: int, X: Sequence[float]) -> Binning:
    """Definition 9.  t_j at the (j/m)-quantile of X, so each of the m non-zero
    bins receives approximately |X|/m of the observed values.

    Repeated values can make two quantiles coincide -- BSBM, for instance, has
    few distinct masses among millions of cells.  Duplicate boundaries are
    dropped, which reduces m; the resulting m is recorded on the returned
    object so that Phi_p reflects what was actually used.
    """
    if X is None or len(X) == 0:
        raise ValueError("equal_frequency needs a non-empty X")
    arr = _snap_array(np.asarray(X, dtype=float))
    qs = [j / m for j in range(1, m)]
    interior = list(np.quantile(arr, qs)) if qs else []

    boundaries = _dedupe_boundaries(interior, xmax=float(arr.max()))
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
    arr = _snap_array(np.asarray(X, dtype=float))
    distinct, counts = np.unique(arr, return_counts=True)

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

    labels = _kmeans_1d(arr, distinct, m, seed, counts=counts)

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
        print(f"[binning] clustering: m {m} -> {m_eff}; "
              f"{len(groups)} of {m} clusters non-empty")
    return Binning("clustering", m_eff, boundaries)


def save_binnings(
    binnings: dict, path: str, X: Optional[Sequence[float]] = None,
    source: Optional[str] = None, fitted_on: Optional[str] = None,
) -> None:
    """Write every binning of a run to one JSON file.

    ``binnings`` maps a key such as "equal_width_m5" to a Binning.  The
    distribution the boundaries were fitted to is summarised alongside them,
    since equal-frequency and clustering boundaries are only reproducible
    with respect to it.
    """
    doc = {"binnings": {}}
    if source is not None:
        doc["source"] = source
    if fitted_on is not None:
        doc["fitted_on"] = fitted_on
    if X is not None and len(X):
        arr = np.asarray(X, dtype=float)
        doc["values"] = {
            "count": int(arr.size),
            "distinct": int(np.unique(_snap_array(arr)).size),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    for key, b in binnings.items():
        entry = b.to_dict()
        if X is not None:
            occ = b.occupancy(X)
            entry["occupied"] = sum(
                1 for j in range(1, b.m + 1) if occ.get(j, 0) > 0
            )
            entry["counts"] = {str(j): occ.get(j, 0) for j in range(1, b.m + 1)}
        doc["binnings"][key] = entry
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def load_binnings(path: str) -> dict:
    """Read the file written by ``save_binnings`` back into Binning objects."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {k: Binning.from_dict(v) for k, v in doc["binnings"].items()}


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

def _dedupe_boundaries(
    interior: Sequence[float], xmax: Optional[float] = None
) -> List[float]:
    """Assemble 0 < t_1 < ... < 1, dropping duplicates and out-of-range cuts.

    A boundary at or above max(X) is also dropped: since bins are left-open,
    the bin above such a boundary can never receive a value, so it would be
    empty by construction rather than by accident.

    Boundaries are rounded like the values they are compared with, so a
    boundary and a value that differ only by summation noise count as equal.
    """
    if xmax is not None:
        xmax = snap(xmax)
    seen = []
    for t in sorted(snap(t) for t in interior):
        if t <= 0.0 or t >= 1.0:
            continue
        if xmax is not None and t >= xmax:
            continue
        if seen and abs(t - seen[-1]) < 1e-12:
            continue
        seen.append(t)
    return [0.0] + seen + [1.0]


def _kmeans_1d(
    arr: np.ndarray, distinct: np.ndarray, m: int, seed: int,
    counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cluster the distinct values of arr into m groups, weighted by frequency.

    counts, if given, are the frequencies of the distinct values in the same
    order (np.unique(arr, return_counts=True)); otherwise they are counted
    here, which is slow on tens of millions of values.
    """
    try:
        from scipy.cluster.vq import kmeans2
    except ImportError:
        print("[binning] scipy unavailable; falling back to quantile cuts "
              "over distinct values")
        qs = [j / m for j in range(1, m)]
        cuts = np.quantile(distinct, qs) if qs else np.array([])
        return np.searchsorted(cuts, distinct)

    if counts is None:
        c = Counter(float(x) for x in arr)
        weights = np.array([c[float(v)] for v in distinct], dtype=float)
    else:
        weights = np.asarray(counts, dtype=float)

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
    """Build the binning determined by Phi_p = (family, m) and the values X.

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
    """Read the non-zero edge weights pi from a weighted-triple file.

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


def load_cell_masses(path: str, progress_every: int = 0) -> np.ndarray:
    """The non-zero cell masses mu(s,p) of a weighted-triple file.

    One cell per (subject, predicate), which is the cell of AC, ACC and
    SchemEX under the union reading, so this is the X those models fit their
    boundaries to, read straight from the file without building the graph.
    The weights are added in file order, as the collector adds them, so the
    sums agree bit for bit.

    Cells are keyed by a hash of (subject, predicate) rather than the strings
    themselves, to bound memory on the larger graphs.
    """
    from collections import defaultdict
    acc: dict = defaultdict(float)
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            acc[hash((parts[0], parts[1]))] += float(parts[2].strip("[]"))
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[mu] {n:,} triples, {len(acc):,} cells")
    out = np.fromiter(acc.values(), dtype=float, count=len(acc))
    return out[out > 0.0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("weights", help="weighted-triple file (.wnt)")
    ap.add_argument("--values", choices=("mu", "pi"), default="mu",
                    help="fit to the cell masses mu(s,p) (default, the "
                         "distribution AC, ACC and SchemEX use) or to the "
                         "edge weights pi")
    ap.add_argument("--family", choices=list(FAMILIES) + ["all"], default="all")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25],
                    help="numbers of non-zero bins to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write all binnings to this single JSON file")
    args = ap.parse_args()

    if args.values == "mu":
        X = load_cell_masses(args.weights)
    else:
        X = np.asarray(load_weights(args.weights), dtype=float)
    print(f"[info] {len(X)} non-zero values ({args.values}), "
          f"{np.unique(_snap_array(X)).size} distinct")
    print(f"[info] range [{X.min():g}, {X.max():g}]")
    print()

    families = list(FAMILIES) if args.family == "all" else [args.family]
    results = {}
    for family in families:
        for m in args.m:
            b = make_binning(family, m, X, seed=args.seed)
            b.report(X)
            results[f"{family}_m{m}"] = b
            print()

    if args.out:
        save_binnings(results, args.out, X=X, source=args.weights,
                      fitted_on=args.values)
        print(f"[ok] wrote {args.out} ({len(results)} binnings)")
