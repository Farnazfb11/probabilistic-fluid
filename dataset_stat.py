#!/usr/bin/env python3
"""
dataset_stats.py
================

Descriptive statistics for a preprocessed dataset, for the table in Section 4.1.

    python dataset_stats.py out/dyldo/graph_weighted.wnt --types out/dyldo/vertex_types.tsv
    python dataset_stats.py out/*/graph_weighted.wnt --markdown   # one row each

Reads the weighted-triple file directly rather than building a graph, so it
runs in a few minutes on the largest datasets and needs little memory.

The label ratio
---------------
For a vertex s, the label ratio is

    |{p : (s,p,o) in G}| / d+(s)

the number of distinct predicates on its outgoing triples divided by its
outdegree.  A ratio of 1 means every outgoing triple carries a different
predicate, so under ~_p = id no two triples share a cell and no aggregation
can occur.  A ratio below 1 means some predicate repeats, so the choice of how
to combine the weights of those triples -- summing them, comparing them
individually, taking the maximum -- changes the equivalence.

The mean ratio therefore measures how much that choice matters on a given
dataset, and the share of vertices with ratio 1 says for how many vertices it
does not matter at all.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean, median


def collect(wnt_path: str, types_path: str | None = None) -> dict:
    outdeg: dict = defaultdict(int)
    preds_per_subject: dict = defaultdict(set)
    cell_count: dict = defaultdict(int)      # (s, p) -> number of triples
    predicates: set = set()
    objects: set = set()
    weights: set = set()
    n_triples = 0

    with open(wnt_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            s, p, w, o = parts[0], parts[1], parts[2], parts[3]
            n_triples += 1
            outdeg[s] += 1
            preds_per_subject[s].add(p)
            cell_count[(s, p)] += 1
            predicates.add(p)
            objects.add(o)
            weights.add(float(w.strip("[]")))

    vertices = set(outdeg) | objects
    dead_ends = len(objects - set(outdeg))

    ratios = [len(preds_per_subject[s]) / outdeg[s] for s in outdeg]
    no_repeat = sum(1 for r in ratios if r == 1.0)

    multi_cells = sum(1 for c in cell_count.values() if c > 1)
    edges_in_multi = sum(c for c in cell_count.values() if c > 1)

    stats = {
        "triples": n_triples,
        "vertices": len(vertices),
        "weighted_subjects": len(outdeg),
        "dead_ends": dead_ends,
        "dead_end_share": dead_ends / max(1, len(vertices)),
        "distinct_predicates": len(predicates),
        "distinct_weights": len(weights),
        "min_weight": min(weights) if weights else None,
        "max_weight": max(weights) if weights else None,
        "mean_outdegree": mean(outdeg.values()) if outdeg else 0.0,
        "median_outdegree": median(outdeg.values()) if outdeg else 0,
        "max_outdegree": max(outdeg.values()) if outdeg else 0,
        "mean_label_ratio": mean(ratios) if ratios else 0.0,
        "median_label_ratio": median(ratios) if ratios else 0.0,
        "subjects_ratio_one": no_repeat,
        "subjects_ratio_one_share": no_repeat / max(1, len(outdeg)),
        "cells": len(cell_count),
        "cells_with_repeats": multi_cells,
        "edges_in_repeated_cells": edges_in_multi,
        "edges_in_repeated_cells_share": edges_in_multi / max(1, n_triples),
    }

    if types_path:
        typed = 0
        all_types: set = set()
        type_sets: set = set()
        with open(types_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                typed += 1
                all_types.update(parts[1:])
                type_sets.add(frozenset(parts[1:]))
        stats["typed_vertices"] = typed
        stats["distinct_types"] = len(all_types)
        stats["distinct_type_sets"] = len(type_sets)

    return stats


def print_report(name: str, s: dict) -> None:
    print(f"===== {name} =====")
    print(f"  triples                     {s['triples']:>12,}")
    print(f"  vertices                    {s['vertices']:>12,}")
    print(f"  weighted subjects           {s['weighted_subjects']:>12,}")
    print(f"  dead ends                   {s['dead_ends']:>12,} "
          f"({s['dead_end_share']:.1%})")
    print(f"  distinct predicates         {s['distinct_predicates']:>12,}")
    if "typed_vertices" in s:
        print(f"  typed vertices              {s['typed_vertices']:>12,}")
        print(f"  distinct types              {s['distinct_types']:>12,}")
        print(f"  distinct type sets          {s['distinct_type_sets']:>12,}")
    print(f"  distinct weights            {s['distinct_weights']:>12,}")
    print(f"  weight range                {s['min_weight']:.3e} .. {s['max_weight']:.3f}")
    print(f"  outdegree mean / median / max  "
          f"{s['mean_outdegree']:.2f} / {s['median_outdegree']} / {s['max_outdegree']:,}")
    print(f"  label ratio mean / median   "
          f"{s['mean_label_ratio']:.3f} / {s['median_label_ratio']:.3f}")
    print(f"  subjects with ratio 1       {s['subjects_ratio_one']:>12,} "
          f"({s['subjects_ratio_one_share']:.1%})")
    print(f"  cells with repeated predicate {s['cells_with_repeats']:>10,} "
          f"of {s['cells']:,}")
    print(f"  edges in those cells        {s['edges_in_repeated_cells']:>12,} "
          f"({s['edges_in_repeated_cells_share']:.1%})")
    print()


def markdown_row(name: str, s: dict) -> str:
    return (f"| {name} | {s['triples']:,} | {s['vertices']:,} | "
            f"{s['dead_end_share']:.1%} | {s['distinct_predicates']:,} | "
            f"{s['distinct_weights']:,} | {s['mean_outdegree']:.1f} | "
            f"{s['mean_label_ratio']:.3f} | "
            f"{s['edges_in_repeated_cells_share']:.1%} |")


MARKDOWN_HEADER = (
    "| dataset | triples | vertices | dead ends | predicates | "
    "distinct pi | mean outdeg | mean label ratio | edges aggregating |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wnt", nargs="+", help="one or more graph_weighted.wnt files")
    ap.add_argument("--types", default=None,
                    help="vertex_types.tsv, when a single wnt is given")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table row per dataset")
    ap.add_argument("--json", default=None, help="also write the statistics here")
    args = ap.parse_args()

    all_stats = {}
    if args.markdown:
        print(MARKDOWN_HEADER)

    for path in args.wnt:
        name = os.path.basename(os.path.dirname(path)) or path
        types = args.types
        if types is None:
            guess = os.path.join(os.path.dirname(path), "vertex_types.tsv")
            types = guess if os.path.exists(guess) else None
        s = collect(path, types)
        all_stats[name] = s
        if args.markdown:
            print(markdown_row(name, s))
        else:
            print_report(name, s)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(all_stats, fh, indent=2)
        print(f"[ok] wrote {args.json}")