#!/usr/bin/env python3
"""
checks.py
=========

Checks and diagnostics behind statements in the thesis.

    python checks.py b0               <graph.wnt> [<types.tsv>] [--verbose]
    python checks.py typesets         <graph.wnt> <types.tsv>
    python checks.py bin-collapse     --root out [--values mu|pi] [--out file.csv]
    python checks.py cluster-collapse --dataset bsbm1000 --m 25 [--values mu|pi]
    python checks.py classes          <graph.wnt> <types.tsv>

Each command takes the options of the script it replaces; run
``python checks.py <command> --help`` for them.

  b0                (was check_b0.py)          the m = 1 binning reproduces the
                    deterministic partition (Remark 17)
  typesets          (was check_typesets.py)    the class count of CC against
                    the number of distinct type sets
  bin-collapse      (was bin_collapse.py)      why the realised m falls short
                    of the requested m, per dataset, family and m
  cluster-collapse  (was cluster_collapse.py)  the same for clustering, step
                    by step through k-means
  classes           (was test.py)              the members and signature of
                    the largest equivalence classes

Only dependencies: numpy, scipy (for the clustering diagnostics), and the
project modules.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from binning import (_dedupe_boundaries, _kmeans_1d, _snap_array,
                     load_cell_masses, make_binning)


def _models():
    from run_summary import MODELS
    return MODELS


# =========================================================================== #
# 1. b0: the coarsest binning reproduces the deterministic partition
#    (was check_b0.py)
# =========================================================================== #

B0_HELP = """\
Verify that the coarsest zero-separating binning reproduces the deterministic
partition, for each summary model.

    python checks.py b0 out/bsbm1000/graph_weighted.wnt out/bsbm1000/vertex_types.tsv

Remark 17 states that under b_0, which distinguishes only zero from non-zero
mass, the probabilistic schema element coincides with the complex schema
element it extends. If that holds, running a model with no binning and
running it with b_0 must produce the same partition of the vertices.

The signature ids themselves differ between the two runs, since the binned
block changes the hash, so the partitions are compared rather than the ids.
Every family gives the same b_0 at m = 1, whatever it is fitted to, so the
check does not depend on the family.

AC, ACC and SchemEX are expected to agree. CC is expected to differ: type
triples carry no mass, so b_0 separates a typed dead-end from a vertex of the
same type set that has triples, which the deterministic class collection,
comparing type sets only, does not (Appendix on CC).
"""


def partition(ids: dict) -> set:
    """The partition induced by a node -> signature id mapping."""
    groups = defaultdict(set)
    for node, sid in ids.items():
        groups[sid].add(node)
    return {frozenset(v) for v in groups.values()}


def describe_difference(det: set, prob: set, limit: int = 3) -> None:
    """Show a few classes that the two partitions disagree about."""
    only_det = det - prob
    only_prob = prob - det
    print(f"      {len(only_det)} classes only in the deterministic run, "
          f"{len(only_prob)} only under b_0")
    for cls in list(only_det)[:limit]:
        members = sorted(str(m) for m in cls)[:4]
        print(f"        det : {len(cls)} members, e.g. {members}")
    for cls in list(only_prob)[:limit]:
        members = sorted(str(m) for m in cls)[:4]
        print(f"        b_0 : {len(cls)} members, e.g. {members}")


def cmd_b0(argv) -> int:
    from pgraph_io import load_weighted_graph
    from Collector import SchemaCollector
    MODELS = _models()

    ap = argparse.ArgumentParser(
        prog="checks.py b0", description=B0_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wnt", help="graph_weighted.wnt")
    ap.add_argument("types", nargs="?", default=None, help="vertex_types.tsv")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS),
                    help="which summary models to check")
    ap.add_argument("--verbose", action="store_true",
                    help="describe the classes the two partitions disagree about")
    args = ap.parse_args(argv)

    G = load_weighted_graph(args.wnt, args.types)
    b0 = make_binning("equal_width", 1)
    print(f"b_0 boundaries: {b0.boundaries}, {b0.n_bins} bins including zero")
    print()

    failures = 0
    for name in args.models:
        # one collection, binned two ways
        summaries = SchemaCollector(MODELS[name]()).collect_masses(G)
        det = partition(SchemaCollector.signatures(summaries, None))
        prob = partition(SchemaCollector.signatures(summaries, b0))
        agree = det == prob
        if not agree:
            failures += 1
        print(f"{'OK  ' if agree else 'DIFF'} {name:<8} "
              f"deterministic {len(det):>8} classes, "
              f"b_0 {len(prob):>8} classes")
        if args.verbose and not agree:
            describe_difference(det, prob)
        del summaries

    print()
    if failures:
        print(f"{failures} model(s) differ. For CC this is expected; for the "
              f"others it indicates that the binned mass is adding "
              f"distinctions b_0 should not make.")
    else:
        print("All checked models reproduce the deterministic partition "
              "under b_0.")
    return 0


# =========================================================================== #
# 2. typesets: CC's class count against the distinct type sets
#    (was check_typesets.py)
# =========================================================================== #

TYPESETS_HELP = """\
Reconcile the class collection's class count with the distinct_type_sets
figure reported by tables.py dataset-stats.

    python checks.py typesets out/watdiv10m/graph_weighted.wnt \\
                              out/watdiv10m/vertex_types.tsv

CC signs a vertex by its type set alone, so its class count is the number of
distinct type sets over the vertices the summariser sees, plus one for the
vertices carrying no type at all. The summariser sees every vertex of the
weighted-triple file *and* every vertex named in vertex_types.tsv, since
pgraph_io adds typed vertices that appear in no triple.

A statistic computed over a narrower population -- only subjects, say, or only
vertices that occur in the triple file -- will report fewer distinct type sets
than CC produces classes. This prints the count under each population so the
difference can be attributed rather than guessed at.
"""


def cmd_typesets(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="checks.py typesets", description=TYPESETS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wnt", help="graph_weighted.wnt")
    ap.add_argument("types", help="vertex_types.tsv")
    args = ap.parse_args(argv)

    # ---- the types file --------------------------------------------------- #
    type_sets = {}
    with open(args.types, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            type_sets[parts[0]] = frozenset(p for p in parts[1:] if p)

    # ---- the weighted-triple file ----------------------------------------- #
    subjects, objects = set(), set()
    with open(args.wnt, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            subjects.add(parts[0])
            objects.add(parts[3])

    graph_vertices = subjects | objects
    all_vertices = graph_vertices | set(type_sets)        # what pgraph_io builds

    def distinct_sets(population) -> int:
        return len({type_sets[v] for v in population if v in type_sets})

    def typed_count(population) -> int:
        return sum(1 for v in population if v in type_sets)

    print(f"{'entries in vertex_types.tsv':<42}{len(type_sets):>12,}")
    print(f"{'distinct type sets in that file':<42}{len(set(type_sets.values())):>12,}")
    print()
    print(f"{'vertices in the .wnt (subjects + objects)':<42}{len(graph_vertices):>12,}")
    print(f"{'vertices the summariser sees':<42}{len(all_vertices):>12,}")
    print(f"{'typed vertices in no triple':<42}"
          f"{len(set(type_sets) - graph_vertices):>12,}")
    print()
    print("distinct type sets, by population")
    for name, pop in (("subjects only", subjects),
                      ("objects only", objects),
                      ("vertices in the .wnt", graph_vertices),
                      ("all vertices the summariser sees", all_vertices)):
        print(f"  {name:<40}{distinct_sets(pop):>12,}   "
              f"({typed_count(pop):,} typed)")

    untyped = len(all_vertices) - typed_count(all_vertices)
    expected_cc = distinct_sets(all_vertices) + (1 if untyped else 0)
    print()
    print(f"{'vertices with no type at all':<42}{untyped:>12,}")
    print(f"{'=> CC should produce':<42}{expected_cc:>12,}   classes")
    print()

    # A type set occurring on exactly one vertex is worth seeing: if the gap is
    # made of singletons, it is likely a handful of unusual vertices rather than
    # a systematic difference in how the two scripts select vertices.
    counts = Counter(type_sets[v] for v in all_vertices if v in type_sets)
    singletons = [ts for ts, c in counts.items() if c == 1]
    print(f"type sets carried by exactly one vertex: {len(singletons)}")
    for ts in singletons[:10]:
        print("   " + ", ".join(sorted(ts)))
    return 0


# =========================================================================== #
# 3. bin-collapse: why the realised m falls short of the requested m
#    (was bin_collapse.py)
# =========================================================================== #

BIN_COLLAPSE_HELP = """\
Why effective_m falls short of the requested m, per dataset, family and m.

    python checks.py bin-collapse --root ~/probabilistic_graphs/datasets/out \\
        --out bin_collapse.csv

The values examined are the cell masses mu(s,p), which the boundaries are now
fitted to; --values pi examines the edge weights, as for a sweep made with
run_summary.py --fit-on pi. Each family loses bins for its own reason, and the
reason is not visible in effective_m alone:

  equal_width      never loses a bin, since the boundaries are j/m regardless
                   of the data.  Instead its bins can sit empty.  For pi, whose
                   values are the reciprocals 1/d+(s), the number of bins that
                   CAN be occupied is fixed by m and the realised out-degree
                   range alone, and is reported as attainable.  For mu, a sum
                   of such reciprocals, only occupancy is reported.

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
effective_m column of master.csv for AC, ACC and SchemEX.  A mismatch means
the recorded boundaries and these refitted ones differ -- possible for
clustering, which goes through k-means, and a bug anywhere else.
"""


def load_weights_fast(path: str) -> np.ndarray:
    """The non-zero edge weights, as a packed array rather than a list."""
    a = np.fromiter((float(l.split()[2].strip("[]"))
                     for l in open(path, encoding="utf-8")
                     if len(l.split()) >= 4), dtype=float)
    return a[a > 0]


def load_values(path: str, values: str) -> np.ndarray:
    """mu(s,p) or pi, rounded as binning.py rounds them before fitting."""
    arr = load_cell_masses(path) if values == "mu" else load_weights_fast(path)
    return _snap_array(np.asarray(arr, dtype=float))


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


def diagnose(path: str, ms, seed: int = 0, values: str = "mu"):
    """One row per (family, m) for the graph at path."""
    ds = os.path.basename(os.path.dirname(path))
    arr = load_values(path, values)
    distinct, counts = np.unique(arr, return_counts=True)
    xmin, xmax = float(distinct[0]), float(distinct[-1])
    if values == "pi":
        d_min, d_max = int(round(1.0 / xmax)), int(round(1.0 / xmin))
        rng = f"pi in [{xmin:g}, {xmax:g}], d+ in [{d_min}, {d_max}]"
    else:
        d_min = d_max = None
        rng = f"mu in [{xmin:g}, {xmax:g}]"

    print(f"\n=== {ds}: {len(arr):,} values ({values}), {len(distinct):,} "
          f"distinct, {rng}")
    print(f"{'family':<17}{'m':>4}{'eff':>5}  {'why':<52}")
    print("-" * 80)

    rows = []
    for m in ms:
        # ---- equal_width ------------------------------------------------- #
        b = make_binning("equal_width", m, arr)
        occ = b.occupancy(arr)
        occupied = sum(1 for j in range(1, b.m + 1) if occ.get(j, 0) > 0)
        if values == "pi":
            att = ew_attainable(m, d_min, d_max)
            why = f"attainable {att}/{m}, occupied {occupied}/{m}"
        else:
            att = ""
            why = f"occupied {occupied}/{m}"
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
        else:
            labels = _kmeans_1d(arr, distinct, m, seed, counts=counts)
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
        r["values"] = values
        r["n_values"] = len(arr)
        r["distinct"] = len(distinct)
        r["d_min"] = d_min
        r["d_max"] = d_max
    return rows


def cmd_bin_collapse(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="checks.py bin-collapse", description=BIN_COLLAPSE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/probabilistic_graphs/datasets/out")
    ap.add_argument("--graph-name", default="graph_weighted.wnt")
    ap.add_argument("--values", choices=("mu", "pi"), default="mu",
                    help="examine the cell masses (default) or the edge weights")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None,
                    help="dataset directory names to restrict to")
    ap.add_argument("--out", default=None, help="CSV to write")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    paths = sorted(glob.glob(f"{root}/*/{args.graph_name}"))
    if args.only:
        paths = [p for p in paths
                 if os.path.basename(os.path.dirname(p)) in args.only]
    if not paths:
        sys.exit(f"no {args.graph_name} under {root}")

    rows = []
    for p in paths:
        rows.extend(diagnose(p, args.m, args.seed, args.values))

    if args.out:
        cols = ["dataset", "values", "n_values", "distinct", "d_min", "d_max",
                "family", "m", "effective_m", "occupied", "attainable", "note"]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[ok] wrote {args.out} ({len(rows)} rows)")
    return 0


# =========================================================================== #
# 4. cluster-collapse: clustering-based binning step by step
#    (was cluster_collapse.py)
# =========================================================================== #

CLUSTER_COLLAPSE_HELP = """\
Why clustering-based binning realises fewer bins than m requests, step by step.

    python checks.py cluster-collapse --dataset bsbm1000 --m 25

Mirrors ``binning._kmeans_1d`` exactly, printing the state after each stage:

  1. the distinct values and how many cells (or triples, for --values pi)
     carry each
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
"""


def recip(x: float) -> str:
    """Render a value as 1/d when it is close to a reciprocal."""
    d = 1.0 / x
    return f"1/{round(d)}" if abs(d - round(d)) < 1e-6 else f"{x:.6f}"


def cmd_cluster_collapse(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="checks.py cluster-collapse", description=CLUSTER_COLLAPSE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bsbm1000")
    ap.add_argument("--root", default="~/probabilistic_graphs/datasets/out")
    ap.add_argument("--graph-name", default="graph_weighted.wnt")
    ap.add_argument("--values", choices=("mu", "pi"), default="mu",
                    help="examine the cell masses (default) or the edge weights")
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=60,
                    help="cap on the per-cluster listing")
    args = ap.parse_args(argv)

    path = os.path.join(os.path.expanduser(args.root),
                        args.dataset, args.graph_name)
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")

    arr = load_values(path, args.values)
    distinct, counts = np.unique(arr, return_counts=True)
    m = args.m
    unit = "cells" if args.values == "mu" else "triples"

    print(f"=== {args.dataset}: {len(arr):,} values ({args.values}), "
          f"{len(distinct)} distinct, "
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
              f"chosen by {hits:>2} of {m} quantiles, {n:,} {unit}")

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
    print(f"{'slot':>5}{'centroid':>12}{'values':>8}{unit:>12}  members")
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
        n_in = int(counts[idx].sum())
        nonempty.append((vals.min(), vals.max()))
        shown = ", ".join(recip(v) for v in vals[:6])
        if len(vals) > 6:
            shown += f", ... (+{len(vals) - 6})"
        print(f"{c:>5}{cpos:>12.6f}{len(idx):>8}{n_in:>12,}  {shown}")

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


# =========================================================================== #
# 5. classes: the members and signature of the largest classes
#    (was test.py)
# =========================================================================== #

def cmd_classes(argv) -> int:
    from pgraph_io import load_weighted_graph
    from Collector import SchemaCollector, mass_values
    MODELS = _models()

    ap = argparse.ArgumentParser(
        prog="checks.py classes",
        description="Show the members and signature of the largest equivalence "
                    "classes, and check that every member agrees on the bins.")
    ap.add_argument("wnt", help="graph_weighted.wnt")
    ap.add_argument("types", help="vertex_types.tsv")
    ap.add_argument("--model", default="SchemEX", choices=list(MODELS))
    ap.add_argument("--family", default="equal_width",
                    choices=("equal_width", "equal_frequency", "clustering"))
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--top", type=int, default=3,
                    help="how many of the largest classes to show")
    args = ap.parse_args(argv)

    def short(x):
        if isinstance(x, frozenset):
            return "{" + ", ".join(sorted(short(t) for t in x)) + "}"
        s = str(x)
        return s.rstrip(">").rsplit("/", 1)[-1].rsplit("#", 1)[-1] or s

    G = load_weighted_graph(args.wnt, args.types)
    summaries = SchemaCollector(MODELS[args.model]()).collect_masses(G)
    X = mass_values(summaries) if args.family != "equal_width" else None
    b = make_binning(args.family, args.m, X)
    ids = SchemaCollector.signatures(summaries, b)

    groups = defaultdict(list)
    for node, sid in ids.items():
        groups[sid].append(node)

    print(f"{args.model}, {args.family} m={args.m} "
          f"(boundaries {', '.join(f'{t:g}' for t in b.boundaries)}): "
          f"{len(groups):,} classes")
    for sid, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:args.top]:
        print(f"\n=== class {sid} : {len(members)} members ===")
        print("  sample:", members[:3])
        vs = summaries[members[0]]
        print("  types:", sorted(short(t) for t in vs.types))
        print("  predicates:", sorted(short(p) for p in vs.neighbors))
        print("  mass:", {f"{short(p)}|{short(o)}": round(v, 4) for (p, o), v in vs.mass.items()})
        print("  bins:", {f"{short(p)}|{short(o)}": v for (p, o), v in (vs.bins or {}).items()})
        # every member must agree on the binned profile
        print("  all members agree on bins:",
              all(summaries[m].bins == vs.bins for m in members))
    return 0


# =========================================================================== #
# command line
# =========================================================================== #

COMMANDS = {
    "b0": cmd_b0,
    "typesets": cmd_typesets,
    "bin-collapse": cmd_bin_collapse,
    "cluster-collapse": cmd_cluster_collapse,
    "classes": cmd_classes,
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help"):
            print(f"unknown command {argv[0]!r}; "
                  f"choose one of {', '.join(COMMANDS)}")
            return 2
        return 0
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
