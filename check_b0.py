#!/usr/bin/env python3
"""
check_b0.py
===========

Verify that the coarsest zero-separating binning reproduces the deterministic
partition, for each summary model.

    python check_b0.py out/bsbm1000/graph_weighted.wnt out/bsbm1000/vertex_types.tsv

Why this check exists
---------------------
Remark 17 states that under b_0, which distinguishes only zero from non-zero
mass, the probabilistic schema element coincides with the complex schema
element it extends. The abstract makes the same claim. If that holds, running
a model with no binning and running it with b_0 must produce the same
partition of the vertices.

The signature ids themselves differ between the two runs, since the binned
block changes the hash, so the partitions are compared rather than the ids.

Reading the output
------------------
AC, ACC and SchemEX are expected to agree. CC is expected to differ: its
implementation records only the source types, so a vertex with no outgoing
triples merges with one that has many, whereas the "and vice versa" clause of
Definition 4 separates them. That difference is in the deterministic
implementation, not in the binning.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from binning import make_binning
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector

from AttCollection import SEAttributeCollection
from ClassCollection import SEClassCollection
from AttClassCollection import SEAttributeClassCollection
from CompAttClassCollection import SEComplexAttributeClassCollection


MODELS = {
    "AC": SEAttributeCollection,
    "CC": SEClassCollection,
    "ACC": SEAttributeClassCollection,
    "SchemEX": SEComplexAttributeClassCollection,
}


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wnt", help="graph_weighted.wnt")
    ap.add_argument("types", nargs="?", default=None, help="vertex_types.tsv")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS),
                    help="which summary models to check")
    ap.add_argument("--verbose", action="store_true",
                    help="describe the classes the two partitions disagree about")
    args = ap.parse_args()

    G = load_weighted_graph(args.wnt, args.types)
    b0 = make_binning("equal_width", 1)
    print(f"b_0 boundaries: {b0.boundaries}, {b0.n_bins} bins including zero")
    print()

    failures = 0
    for name in args.models:
        collector = SchemaCollector(MODELS[name]())
        _, det_ids = collector.collect(G)
        _, b0_ids = collector.collect(G, binning=b0)

        det, prob = partition(det_ids), partition(b0_ids)
        agree = det == prob
        if not agree:
            failures += 1
        print(f"{'OK  ' if agree else 'DIFF'} {name:<8} "
              f"deterministic {len(det):>8} classes, "
              f"b_0 {len(prob):>8} classes")
        if args.verbose and not agree:
            describe_difference(det, prob)

    print()
    if failures:
        print(f"{failures} model(s) differ. For CC this is expected; for the "
              f"others it indicates that the binned mass is adding "
              f"distinctions b_0 should not make.")
    else:
        print("All checked models reproduce the deterministic partition "
              "under b_0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())