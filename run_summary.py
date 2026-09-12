#!/usr/bin/env python3
"""
run_summary.py
==============

Compute one probabilistic summary and its measures.

    python run_summary.py \
        --graph out/dyldo/graph_weighted.wnt \
        --types out/dyldo/vertex_types.tsv \
        --binnings out/dyldo/binnings.json \
        --config equal_frequency_m10 \
        --model SchemEX \
        --out out/dyldo/summaries

Or sweep every binning in the file against several models:

    python run_summary.py --graph ... --types ... --binnings ... \
        --config all --model AC ACC SchemEX --out out/dyldo/summaries

Two phases
----------
The first phase determines the binning from the model parameters and the input
graph, and replaces each vertex's outgoing mass by the bin it falls into. This
is what the model adds. The second computes the quotient under EQR_p, for
which any partition refinement is adequate: once the mass is binned, the
condition of Definition 12 is an ordinary equivalence over discrete labels.

Passing --config deterministic runs the second phase alone, with no binning,
which reproduces the deterministic FLUID summary. Together with the b_0
configuration this is the check of Remark 17.

Measures written
----------------
Per Section 4.5: the number of equivalence classes, the average and the
distribution of class sizes, the number of summary edges, runtime, peak
memory, the dead-end share, and the representative error, which is the
difference between a summarised vertex's exact outgoing masses and those the
payload keeps for the chosen representative.

Only dependencies: networkx, plus the project modules.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections import Counter, defaultdict
from statistics import mean, median

from binning import Binning, load_binnings
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector
from Summarizer import group_by_signature, build_summary_graph

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


def peak_memory_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def class_size_distribution(groups: dict) -> dict:
    """Summarise how the vertices are spread over the equivalence classes."""
    sizes = [len(v) for v in groups.values()]
    sizes.sort()
    hist = Counter()
    for s in sizes:
        if s == 1:
            hist["1"] += 1
        elif s <= 10:
            hist["2-10"] += 1
        elif s <= 100:
            hist["11-100"] += 1
        elif s <= 1000:
            hist["101-1000"] += 1
        else:
            hist[">1000"] += 1
    return {
        "classes": len(sizes),
        "mean_size": mean(sizes) if sizes else 0.0,
        "median_size": median(sizes) if sizes else 0,
        "max_size": sizes[-1] if sizes else 0,
        "singletons": hist.get("1", 0),
        "histogram": dict(hist),
    }


def representative_error(summaries: dict, groups: dict) -> dict:
    """How much the payload's representative misstates the other members.

    Section 3.8 keeps the exact outgoing masses of one designated member of
    each class. Members agree on the binned mass but not on the mass itself,
    so the payload is well defined only relative to that choice, and the
    discrepancy is worth reporting rather than assuming away.
    """
    errors = []
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = summaries[members[0]].mass
        for m in members[1:]:
            other = summaries[m].mass
            for cell in set(rep) | set(other):
                d = abs(rep.get(cell, 0.0) - other.get(cell, 0.0))
                if d > 0.0:
                    errors.append(d)
    if not errors:
        return {"cells_compared": 0, "mean": 0.0, "max": 0.0}
    return {
        "cells_compared": len(errors),
        "mean": mean(errors),
        "median": median(errors),
        "max": max(errors),
    }


def run_one(G, model_name: str, binning, label: str) -> dict:
    """Summarise G under one model and one binning, returning the measures."""
    t0 = time.perf_counter()
    collector = SchemaCollector(MODELS[model_name]())
    summaries, ids = collector.collect(G, binning=binning)
    t_collect = time.perf_counter() - t0

    groups = group_by_signature(ids)

    t1 = time.perf_counter()
    super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)
    t_graph = time.perf_counter() - t1

    n_super_edges = sum(len(d) for d in super_counts.values())
    n_super_pred_edges = sum(
        len(preds) for dsts in super_pred.values() for preds in dsts.values()
    )
    dead_ends = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

    result = {
        "model": model_name,
        "binning": label,
        "input_vertices": G.number_of_nodes(),
        "input_triples": G.number_of_edges(),
        "dead_ends": dead_ends,
        "dead_end_share": dead_ends / max(1, G.number_of_nodes()),
        "summary_edges": n_super_edges,
        "summary_labelled_edges": n_super_pred_edges,
        "seconds_collect": round(t_collect, 3),
        "seconds_summary_graph": round(t_graph, 3),
        "peak_memory_mb": round(peak_memory_mb(), 1),
    }
    result.update(class_size_distribution(groups))
    result["compression"] = (
        result["classes"] / G.number_of_nodes() if G.number_of_nodes() else 0.0
    )
    result["representative_error"] = representative_error(summaries, groups)

    if binning is not None:
        result["binning_family"] = binning.family
        result["binning_effective_m"] = binning.m
        result["binning_boundaries"] = binning.boundaries

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--graph", required=True, help="graph_weighted.wnt")
    ap.add_argument("--types", default=None, help="vertex_types.tsv")
    ap.add_argument("--binnings", default=None,
                    help="binnings.json written by binning.py")
    ap.add_argument("--config", nargs="+", default=["deterministic"],
                    help="binning keys from that file, or 'all', or "
                         "'deterministic' for no binning")
    ap.add_argument("--model", nargs="+", default=["SchemEX"],
                    choices=list(MODELS))
    ap.add_argument("--out", default=None,
                    help="directory for the measures file")
    args = ap.parse_args()

    G = load_weighted_graph(args.graph, args.types)

    binnings: dict = {}
    if args.binnings:
        binnings = load_binnings(args.binnings)

    configs: list = []
    for c in args.config:
        if c == "deterministic":
            configs.append(("deterministic", None))
        elif c == "all":
            configs.extend(sorted(binnings.items()))
        elif c in binnings:
            configs.append((c, binnings[c]))
        else:
            raise SystemExit(
                f"unknown config {c!r}; available: "
                f"{', '.join(sorted(binnings))} or 'deterministic'"
            )

    results = []
    for model_name in args.model:
        for label, b in configs:
            r = run_one(G, model_name, b, label)
            results.append(r)
            print(f"{model_name:<8} {label:<24} "
                  f"{r['classes']:>9,} classes  "
                  f"mean size {r['mean_size']:>8.2f}  "
                  f"edges {r['summary_edges']:>9,}  "
                  f"{r['seconds_collect']:>7.1f}s")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(os.path.dirname(args.graph)) or "summary"
        path = os.path.join(args.out, f"measures_{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"graph": args.graph, "results": results}, fh, indent=2)
        print(f"\n[ok] wrote {path} ({len(results)} runs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())