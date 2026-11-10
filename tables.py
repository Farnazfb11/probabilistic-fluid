#!/usr/bin/env python3
"""
tables.py
=========

The numbers behind the tables of the thesis.

    python tables.py build          --root out --out out/tables
    python tables.py dataset-stats  out/*/graph_weighted.wnt --markdown
    python tables.py collect-stats  [--parse] [--only KEY ...]
    python tables.py parse-counts   <source file> [--keep-lang]

Each command takes the options of the script it replaces; run
``python tables.py <command> --help`` for them.

  build          (was build_tables.py)  every measure of every sweep in one
                 long table, plus pivots and a workbook
  dataset-stats  (was dataset_stats.py) descriptive statistics of a
                 preprocessed dataset, read from its weighted-triple file
  collect-stats  (was collect_stats.py) the dataset statistics of all six
                 datasets with provenance, and the LaTeX rows of the dataset
                 tables
  parse-counts   (was parse_counts.py)  the parser's deduplication counters
                 for one source file

Only dependencies: the standard library (pandas optional, for the workbook).
parse-counts and collect-stats --parse also need RDFparser.py.
"""


from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

# =========================================================================== #
# 1. build: the master table and pivots   (was build_tables.py)
# =========================================================================== #

# Highest priority first. A row from an earlier directory wins.
SOURCES = ("final", "summaries_schemex", "summaries_reperr", "summaries")

FAMILY_ORDER = {"deterministic": 0, "equal_width": 1, "equal_frequency": 2,
                "clustering": 3}
MODEL_ORDER = {"AC": 0, "CC": 1, "ACC": 2, "SchemEX": 3}

# Models whose cells are (subject, predicate), the cells measures.py fidelity uses.
PER_PREDICATE_MODELS = ("AC", "ACC", "SchemEX")

Key = Tuple[str, str, str]      # dataset, model, binning


def binning_key(label: str):
    if label == "deterministic":
        return (0, 0, 0)
    fam, _, m = label.rpartition("_m")
    return (1, FAMILY_ORDER.get(fam, 98), int(m) if m.isdigit() else 0)


def row_key(r: Dict[str, Any]):
    return (r["dataset"], MODEL_ORDER.get(r["model"], 9),
            binning_key(r["binning"]))


def phase(r: Dict[str, Any], name: str, field: str, default=None):
    return ((r.get("phases") or {}).get(name) or {}).get(field, default)


def _add(*xs):
    return None if any(x is None for x in xs) else sum(xs)


def flatten(dataset: str, source: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """One measures row into one flat record."""
    err = r.get("representative_error") or {}
    disp = r.get("dispersion") or {}
    label = r["binning"]
    fam, _, m = label.rpartition("_m")
    out = {
        "dataset": dataset,
        "model": r["model"],
        "binning": label,
        "family": r.get("binning_family") or (fam if label != "deterministic"
                                              else "deterministic"),
        "requested_m": int(m) if m.isdigit() else None,
        "effective_m": r.get("binning_effective_m"),
        "fitted_on": r.get("binning_fitted_on"),

        "input_vertices": r.get("input_vertices"),
        "input_triples": r.get("input_triples"),
        "dead_end_share": r.get("dead_end_share"),

        "classes": r.get("classes"),
        "compression": r.get("compression"),
        "mean_size": r.get("mean_size"),
        "median_size": r.get("median_size"),
        "max_size": r.get("max_size"),
        "singletons": r.get("singletons"),
        "summary_edges": r.get("summary_edges"),
        "summary_labelled_edges": r.get("summary_labelled_edges"),

        "rep_err_mean": err.get("mean"),
        "rep_err_median": err.get("median"),
        "rep_err_max": err.get("max"),
        "rep_err_cells": err.get("cells_compared"),

        "disp_mean": disp.get("mean"),
        "disp_median": disp.get("median"),
        "disp_max": disp.get("max"),
        "disp_cells": disp.get("cells"),

        "sec_collect": phase(r, "collect", "seconds"),
        "collect_shared": phase(r, "collect", "shared"),
        "sec_mass_values": phase(r, "mass_values", "seconds"),
        "sec_binning": phase(r, "binning", "seconds"),
        "sec_partition": phase(r, "partition", "seconds"),
        "sec_summary_graph": phase(r, "summary_graph", "seconds"),
        "sec_fidelity": phase(r, "fidelity", "seconds"),
        "sec_rep_error": phase(r, "representative_error", "seconds"),
        "sec_export": phase(r, "export", "seconds"),
        "sec_total": r.get("seconds_total"),

        "rss_delta_collect_mb": phase(r, "collect", "rss_delta_mb"),
        "hwm_increase_collect_mb": phase(r, "collect", "hwm_increase_mb"),
        "rss_delta_partition_mb": phase(r, "partition", "rss_delta_mb"),
        "process_peak_rss_mb": r.get("process_peak_rss_mb"),

        "source": source,
    }
    c, p, g = out["sec_collect"], out["sec_partition"], out["sec_summary_graph"]
    if p is not None:
        # fitted after collection: phase 2 is partition + summary graph
        out["sec_phase2"] = _add(p, g)
        out["sec_summary"] = _add(c, p, g)
    else:
        # older sweep: the collect step did the binning and the signatures,
        # so it was phase 2 together with the summary graph
        out["sec_phase2"] = _add(c, g)
        out["sec_summary"] = out["sec_phase2"]
    return out


def read_measures(root: str) -> Dict[Key, Dict[str, Any]]:
    """Resolve the measures rows across the source directories."""
    by_key: Dict[Key, Dict[str, Any]] = {}
    rank = {s: i for i, s in enumerate(SOURCES)}
    found: Dict[str, int] = defaultdict(int)
    err_fields = ("rep_err_mean", "rep_err_median", "rep_err_max",
                  "rep_err_cells", "disp_mean", "disp_median", "disp_max",
                  "disp_cells")

    for src in SOURCES:
        for path in sorted(glob.glob(f"{root}/*/{src}/measures_*.json")):
            dataset = path.split(os.sep)[-3]
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            for r in doc.get("results", []):
                rec = flatten(dataset, src, r)
                key = (dataset, rec["model"], rec["binning"])
                found[src] += 1
                prev = by_key.get(key)
                if prev is None or rank[src] < rank[prev["source"]]:
                    # keep any representative error the superseded row had
                    if prev and rec["rep_err_mean"] is None:
                        for f in err_fields:
                            rec[f] = prev.get(f)
                        rec["rep_err_from"] = prev["source"]
                    by_key[key] = rec
                elif prev["rep_err_mean"] is None and rec["rep_err_mean"] is not None:
                    for f in err_fields:
                        prev[f] = rec.get(f)
                    prev["rep_err_from"] = src

    for s in SOURCES:
        if found[s]:
            print(f"[measures] {s:<20} {found[s]:>5} rows")
    print(f"[measures] {len(by_key)} resolved (dataset, model, binning) rows")
    return by_key


def add_bin_fidelity(root: str, by_key: Dict[Key, Dict[str, Any]]) -> None:
    """Quantisation error per (dataset, binning).

    measures.py fidelity measures the (subject, predicate) cells, which are the
    cells of AC, ACC and SchemEX, and those three share their boundaries when
    the boundaries are fitted to mu. CC has its own cell and its own fit, so
    its rows are not annotated.
    """
    n = 0
    for path in sorted(glob.glob(f"{root}/*/bin_fidelity.json")):
        dataset = path.split(os.sep)[-2]
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for r in doc.get("results", []):
            fields = {
                "quant_err_pi": r["pi"]["quantisation_error"],
                "quant_err_mu": r["mu"]["quantisation_error"],
                "bins_occupied_pi": r["pi"]["bins_occupied"],
                "bins_occupied_mu": r["mu"]["bins_occupied"],
            }
            for key, rec in by_key.items():
                if (key[0] == dataset and key[2] == r["binning"]
                        and key[1] in PER_PREDICATE_MODELS):
                    rec.update(fields)
                    n += 1
    if n:
        print(f"[bin_fidelity] annotated {n} rows")


def add_unary(root: str, by_key: Dict[Key, Dict[str, Any]]) -> None:
    path = os.path.join(root, "unary", "rows.jsonl")
    if not os.path.exists(path):
        return
    wanted = ("eqc", "avg_eqc_size", "std_eqc_size", "total_payload",
              "edges_unweighted", "edges_weighted", "avg_out_degree_w",
              "avg_in_degree_w", "avg_degree_w", "tmh_w", "tmh_u",
              "comp_weak", "comp_strong")
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                u = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = by_key.get((u["dataset"], u["model"], u["binning"]))
            if rec is None:
                continue
            for f in wanted:
                if f in u:
                    rec["unary_" + f] = u[f]
            n += 1
    print(f"[unary] annotated {n} rows")


def write_csv(rows: List[Dict[str, Any]], path: str,
              fields: Optional[List[str]] = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {path}  ({len(rows)} rows)")


def pivot(rows: List[Dict[str, Any]], value: str) -> List[Dict[str, Any]]:
    """Rows: dataset x model. Columns: binning. Cells: one measure."""
    labels = sorted({r["binning"] for r in rows}, key=binning_key)
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        g = grouped.setdefault((r["dataset"], r["model"]),
                               {"dataset": r["dataset"], "model": r["model"]})
        g[r["binning"]] = r.get(value)
    out = [grouped[k] for k in sorted(
        grouped, key=lambda k: (k[0], MODEL_ORDER.get(k[1], 9)))]
    return out, ["dataset", "model"] + labels


def efficiency_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per dataset and model: runtime collapsed over the binnings.

    Runtime barely varies with the binning, so a mean and a range say what
    thirteen rows would, and the narrowness of the range is itself the point.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["model"])].append(r)

    out = []
    for (dataset, model), rs in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], MODEL_ORDER.get(kv[0][1], 9))):
        def col(f):
            v = [r[f] for r in rs if r.get(f) is not None]
            return v or None
        p2, summ = col("sec_phase2"), col("sec_summary")
        bin_s, mem = col("sec_binning"), col("rss_delta_collect_mb")
        coll = col("sec_collect")
        out.append({
            "dataset": dataset,
            "model": model,
            "runs": len(rs),
            "collect_s": round(mean(coll), 1) if coll else None,
            "phase2_s_mean": round(mean(p2), 2) if p2 else None,
            "phase2_s_min": round(min(p2), 2) if p2 else None,
            "phase2_s_max": round(max(p2), 2) if p2 else None,
            "summary_s_mean": round(mean(summ), 1) if summ else None,
            "binning_s_mean": round(mean(bin_s), 2) if bin_s else None,
            "rss_delta_mb_mean": round(mean(mem), 1) if mem else None,
            "peak_rss_mb": max((r["process_peak_rss_mb"] for r in rs
                                if r.get("process_peak_rss_mb") is not None),
                               default=None),
        })
    return out


BUILD_HELP = """\
Collect every measure into one long table, plus a few pivots to look at.

    python tables.py build --root ~/probabilistic_graphs/datasets/out \
                             --out  ~/probabilistic_graphs/datasets/out/tables

Sources, each optional
----------------------
<root>/<dataset>/{final,summaries_schemex,summaries_reperr,summaries}/measures_*.json
    the sweep measures. Several directories hold rows for the same
    (dataset, model, binning): the earlier a directory appears in SOURCES
    below, the higher its priority, so a rerun supersedes the run it
    replaced. Representative error is additionally back-filled from any
    source that has it, since a run made with --no-rep-error is otherwise
    complete.

<root>/<dataset>/bin_fidelity.json
    quantisation error of each binning, over pi and over mu
    (measures.py fidelity).

<root>/unary/rows.jsonl
    the unary measures over the exported summary packages (measures.py unary).

Output
------
master.csv          one row per (dataset, model, binning), every column
pivot_*.csv         one measure, datasets x models as rows, binnings as columns
tables.xlsx         the same sheets in one workbook, if pandas is installed

The master table is the thing to read first and decide from; the pivots are
there because a 312-row sheet is hard to see a pattern in.

Runtime columns
---------------
Since the boundaries are fitted to the collected masses, a run consists of
the collect step (shared by all configurations of a model unless the sweep
used --recollect), phase 1 (fitting) and phase 2 (partition and summary
graph). sec_phase2 is phase 2 alone; sec_summary is collect + phase 2, which
is what sec_phase2 meant for sweeps made before the change, whose rows have
no partition step and are read accordingly.
"""


def cmd_build(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="tables.py build", description=BUILD_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the out/ directory")
    ap.add_argument("--out", required=True, help="directory for the tables")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out)

    by_key = read_measures(root)
    if not by_key:
        print("[info] no measures files found")
        return 1

    # Rows fitted to mu and rows from sweeps made before the change (no
    # fitted_on recorded, boundaries fitted to pi) must not end up in one
    # table unnoticed: the higher-priority source wins per row, so a stale
    # directory can silently replace new results.
    binned = [r for r in by_key.values() if r["binning"] != "deterministic"
              and r.get("effective_m") is not None]
    kinds = defaultdict(set)
    for r in binned:
        kinds[r["dataset"]].add(r.get("fitted_on") or "pi (old sweep)")
    for dataset, k in sorted(kinds.items()):
        if len(k) > 1:
            print(f"[warn] {dataset}: rows fitted in different ways are "
                  f"mixed ({', '.join(sorted(k))}); move the old measures "
                  f"files out of {root}/{dataset}/ before building the tables")
    add_bin_fidelity(root, by_key)
    add_unary(root, by_key)

    rows = sorted(by_key.values(), key=row_key)
    write_csv(rows, os.path.join(out, "master.csv"))

    pivots = {}
    for value in ("classes", "compression", "mean_size", "max_size",
                  "summary_edges", "rep_err_mean", "rep_err_max",
                  "disp_mean", "disp_max",
                  "sec_phase2", "effective_m", "quant_err_mu",
                  "unary_tmh_w", "unary_comp_weak"):
        if not any(r.get(value) is not None for r in rows):
            continue
        prows, fields = pivot(rows, value)
        write_csv(prows, os.path.join(out, f"pivot_{value}.csv"), fields)
        pivots[value] = (prows, fields)

    eff = efficiency_summary(rows)
    write_csv(eff, os.path.join(out, "efficiency.csv"))

    # one workbook, if pandas is around
    try:
        import pandas as pd
    except ImportError:
        print("[info] pandas not installed; CSVs only")
        return 0

    xlsx = os.path.join(out, "tables.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="master", index=False)
        pd.DataFrame(eff).to_excel(xw, sheet_name="efficiency", index=False)
        for value, (prows, fields) in pivots.items():
            pd.DataFrame(prows, columns=fields).to_excel(
                xw, sheet_name=value[:31], index=False)
    print(f"[ok] {xlsx}")
    return 0



# =========================================================================== #
# 2. dataset-stats: descriptive statistics   (was dataset_stats.py)
# =========================================================================== #

DATASET_STATS_HELP = """\
Descriptive statistics for a preprocessed dataset, for the table in Section 4.1.

    python tables.py dataset-stats out/dyldo/graph_weighted.wnt --types out/dyldo/vertex_types.tsv
    python tables.py dataset-stats out/*/graph_weighted.wnt --markdown   # one row each

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


def dataset_stats(wnt_path: str, types_path: str | None = None) -> dict:
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


def cmd_dataset_stats(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="tables.py dataset-stats", description=DATASET_STATS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wnt", nargs="+", help="one or more graph_weighted.wnt files")
    ap.add_argument("--types", default=None,
                    help="vertex_types.tsv, when a single wnt is given")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table row per dataset")
    ap.add_argument("--json", default=None, help="also write the statistics here")
    args = ap.parse_args(argv)

    all_stats = {}
    if args.markdown:
        print(MARKDOWN_HEADER)

    for path in args.wnt:
        name = os.path.basename(os.path.dirname(path)) or path
        types = args.types
        if types is None:
            guess = os.path.join(os.path.dirname(path), "vertex_types.tsv")
            types = guess if os.path.exists(guess) else None
        s = dataset_stats(path, types)
        all_stats[name] = s
        if args.markdown:
            print(markdown_row(name, s))
        else:
            print_report(name, s)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(all_stats, fh, indent=2)
        print(f"[ok] wrote {args.json}")
    return 0


# =========================================================================== #
# 3. collect-stats: every dataset statistic with provenance
#    (was collect_stats.py)
# =========================================================================== #

COLLECT_STATS_HELP = """\
Collect every descriptive statistic reported for the datasets into one place,
with provenance, so the figures in the thesis can be traced to a recorded run.

    python tables.py collect-stats                 # reuse cached parse counts
    python tables.py collect-stats --parse         # re-run the parser on every dataset
    python tables.py collect-stats --only btc150m dyldo_2015-10-11

Two kinds of statistic are collected per dataset.

1. Deduplication counters, from RDFParser: the number of input lines, the
   number of distinct triples remaining after deduplication on the normalised
   (s, p, o), the number of duplicates dropped, and the number of lines that
   did not match the N-Triples/N-Quads pattern. A non-zero count in the last
   field means the reduction is partly parse failure rather than duplication,
   so it is always recorded.

   These require a full pass over the source file, which takes minutes to hours.
   The result is cached in <outdir>/parse_counts.json and reused unless --parse
   is given.

2. Graph statistics, from the dataset-stats command: written to
   <outdir>/dataset_stats.json when that command was run with --json. This
   command reads that file rather than recomputing, so run dataset-stats first
   for any dataset whose graph has changed.

Outputs, all under stats/ in the repository so they can be committed:

    stats/all_datasets.json   every figure, plus provenance
    stats/all_datasets.md     a readable table
    stats/tables.tex          the two LaTeX table bodies used in the thesis

Emitting the LaTeX rows here is deliberate: the thesis tables are then a copy
of recorded output rather than transcribed by hand.
"""

# The datasets. Edit the paths here if anything moves.
HOME = os.path.expanduser("~")
DATA = os.path.join(HOME, "probabilistic_graphs", "datasets")
OUT = os.path.join(DATA, "out")

DATASETS = [
    # key                 label          source file                                        outdir
    ("bsbm1000",          "BSBM 1000",   f"{DATA}/bsbm/bsbm_1000.nt",                        f"{OUT}/bsbm1000"),
    ("bsbm100m",          "BSBM 100M",   f"{DATA}/bsbm100m/BSBM100M.nt.gz",                  f"{OUT}/bsbm100m"),
    ("watdiv10m",         "WatDiv 10M",  f"{DATA}/watdiv/watdiv.10M.nt",                     f"{OUT}/watdiv10m"),
    ("dbpedia39",         "DBpedia 3.9", f"{DATA}/dbpedia39/dbpedia39_combined.nt",          f"{OUT}/dbpedia39"),
    ("dyldo_2015-10-11",  "DyLDO",       "/media/nvme7n1/fbtelki/dyldo_full/data/2015-10-11/data.nq.gz",
                                                                                             f"{OUT}/dyldo_2015-10-11"),
    ("btc150m",           "BTC150M",     f"{DATA}/btc150m/BTC150M-btc2019-wikidata.org_00001.nq.gz",
                                                                                             f"{OUT}/btc150m"),
]

REPO = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(REPO, "stats")


def _run(cmd: list[str]) -> str | None:
    import subprocess
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def provenance() -> dict:
    """Everything needed to say which code produced these numbers."""
    import platform
    import socket
    from datetime import datetime, timezone
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def file_facts(path: str) -> dict:
    from datetime import datetime, timezone
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                 .isoformat(timespec="seconds")}
    except FileNotFoundError:
        return {"path": path, "exists": False}


def parser_counts(source: str) -> dict:
    """Run RDFParser over a source file and return its counters."""
    from RDFparser import RDFParser          # imported late: only needed when parsing
    t0 = time.time()
    p = RDFParser(source)
    for _ in p.parse():
        pass
    return {
        "lines": p.counter_lines,
        "distinct": p.counter_yielded,
        "duplicates": p.counter_duplicates,
        "non_matching": p.counter_NM,
        "seconds": round(time.time() - t0, 1),
        "source": source,
    }


def parse_counts(source: str, outdir: str, force: bool) -> dict | None:
    """RDFParser's deduplication counters, cached in <outdir>/parse_counts.json."""
    cache = os.path.join(outdir, "parse_counts.json")

    if not force and os.path.exists(cache):
        with open(cache) as fh:
            d = json.load(fh)
        d["from_cache"] = True
        return d

    if not os.path.exists(source):
        return None

    d = parser_counts(source)
    d["from_cache"] = False
    os.makedirs(outdir, exist_ok=True)
    with open(cache, "w") as fh:
        json.dump(d, fh, indent=2)
    return d


def graph_stats(outdir: str) -> dict | None:
    """Whatever the dataset-stats command wrote for this dataset."""
    path = os.path.join(outdir, "dataset_stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    # dataset-stats may write either a bare dict or one keyed by dataset name
    if len(d) == 1 and isinstance(next(iter(d.values())), dict):
        d = next(iter(d.values()))
    return d


def derived(pc: dict | None, gs: dict | None) -> dict:
    """Figures computed from the two sources, rather than measured separately."""
    out = {}
    if pc and pc.get("lines"):
        out["duplicate_share"] = pc["duplicates"] / pc["lines"]
    if pc and gs and pc.get("distinct") and gs.get("triples"):
        out["type_triples"] = pc["distinct"] - gs["triples"]
    if gs:
        if gs.get("distinct_type_sets"):
            out["vertices_per_type_set"] = gs["typed_vertices"] / gs["distinct_type_sets"]
        if gs.get("max_outdegree") and gs.get("min_weight") is not None:
            # min pi should equal 1 / max outdegree exactly; record the check
            expected = 1.0 / gs["max_outdegree"]
            out["min_weight_matches_max_outdegree"] = abs(expected - gs["min_weight"]) < 1e-12
        if gs.get("vertices") and gs.get("weighted_subjects") is not None:
            out["dead_ends_equals_vertices_minus_subjects"] = (
                gs["vertices"] - gs["weighted_subjects"] == gs.get("dead_ends")
            )
    return out


def tex(n) -> str:
    """Thousands separators LaTeX will not treat as maths."""
    if n is None:
        return "---"
    return f"{int(n):,}".replace(",", "{,}")


def write_markdown(rows: list[dict], path: str, prov: dict) -> None:
    L = []
    L.append(f"# Dataset statistics\n")
    L.append(f"Generated {prov['generated_at']} on {prov['host']}, "
             f"commit `{prov['git_commit']}`"
             f"{' (working tree dirty)' if prov['git_dirty'] else ''}.\n")
    L.append("## Size and deduplication\n")
    L.append("| Dataset | Input | Distinct | Duplicates | Non-matching | Mass-carrying | Type triples | Vertices | Dead ends |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        pc, gs, dv = r["parse_counts"] or {}, r["graph_stats"] or {}, r["derived"]
        L.append(
            f"| {r['label']} | {pc.get('lines','—'):,} | {pc.get('distinct','—'):,} "
            f"| {pc.get('duplicates','—'):,} | {pc.get('non_matching','—')} "
            f"| {gs.get('triples','—'):,} | {dv.get('type_triples','—'):,} "
            f"| {gs.get('vertices','—'):,} | {100*gs.get('dead_end_share',0):.1f}% |"
            if pc and gs else f"| {r['label']} | incomplete |")
    L.append("\n## Types, probabilities and aggregation\n")
    L.append("| Dataset | Typed | Types | Type sets | Vertices/type set | \\|π\\| | Label ratio | ratio=1 | Cells | Rep. cells | Rep. triples |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        gs, dv = r["graph_stats"] or {}, r["derived"]
        if not gs:
            L.append(f"| {r['label']} | incomplete |")
            continue
        L.append(
            f"| {r['label']} | {gs['typed_vertices']:,} | {gs['distinct_types']:,} "
            f"| {gs['distinct_type_sets']:,} | {dv.get('vertices_per_type_set',0):,.0f} "
            f"| {gs['distinct_weights']:,} | {gs['mean_label_ratio']:.3f} "
            f"| {100*gs['subjects_ratio_one_share']:.1f}% | {gs['cells']:,} "
            f"| {gs['cells_with_repeats']:,} | {100*gs['edges_in_repeated_cells_share']:.1f}% |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_tex(rows: list[dict], path: str, prov: dict) -> None:
    L = [f"% generated by tables.py collect-stats on {prov['generated_at']}",
         f"% commit {prov['git_commit']}",
         "", "% ---- tab:datasets-size"]
    for r in rows:
        pc, gs = r["parse_counts"] or {}, r["graph_stats"] or {}
        if not gs:
            continue
        L.append(
            f"{r['label']:<12}& {tex(pc.get('lines')):>17} & {tex(pc.get('distinct')):>16} "
            f"& {tex(gs['triples']):>16} & {tex(gs['vertices']):>15} "
            f"& {100*gs['dead_end_share']:>4.1f}\\% & {gs['mean_outdegree']:>5.2f} "
            f"& {int(gs['median_outdegree']):>2} & {tex(gs['max_outdegree']):>9} "
            f"& {tex(gs['distinct_predicates']):>9} \\\\")
    L += ["", "% ---- tab:datasets-prob"]
    for r in rows:
        gs = r["graph_stats"] or {}
        if not gs:
            continue
        L.append(
            f"{r['label']:<12}& {tex(gs['typed_vertices']):>15} & {tex(gs['distinct_types']):>9} "
            f"& {tex(gs['distinct_type_sets']):>9} & {tex(gs['distinct_weights']):>7} "
            f"& {gs['mean_label_ratio']:.3f} & {100*gs['subjects_ratio_one_share']:>5.1f}\\% "
            f"& {tex(gs['cells']):>15} & {tex(gs['cells_with_repeats']):>12} "
            f"& {100*gs['edges_in_repeated_cells_share']:>5.1f}\\% \\\\")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def cmd_collect_stats(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="tables.py collect-stats", description=COLLECT_STATS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse", action="store_true",
                    help="re-run RDFParser even when cached counts exist (slow)")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="restrict to these dataset keys")
    ap.add_argument("--out", default=STATS_DIR, help=f"output directory (default {STATS_DIR})")
    args = ap.parse_args(argv)

    selected = [d for d in DATASETS if not args.only or d[0] in args.only]
    if args.only:
        unknown = set(args.only) - {d[0] for d in DATASETS}
        if unknown:
            sys.exit(f"unknown dataset key(s): {', '.join(sorted(unknown))}")

    prov = provenance()
    rows = []

    for key, label, source, outdir in selected:
        print(f"[{key}] ", end="", flush=True)
        pc = parse_counts(source, outdir, args.parse)
        if pc is None:
            print("no source file and no cached parse counts; ", end="")
        elif pc.get("from_cache"):
            print("parse counts cached; ", end="")
        else:
            print(f"parsed in {pc['seconds']}s; ", end="")

        gs = graph_stats(outdir)
        if gs is None:
            print("no dataset_stats.json -- run tables.py dataset-stats "
                  "--json for this dataset")
        else:
            print("graph stats read")

        rows.append({
            "key": key, "label": label,
            "source": file_facts(source),
            "outdir": outdir,
            "parse_counts": pc,
            "graph_stats": gs,
            "derived": derived(pc, gs),
        })

    os.makedirs(args.out, exist_ok=True)
    doc = {"provenance": prov, "datasets": rows}

    with open(os.path.join(args.out, "all_datasets.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    write_markdown(rows, os.path.join(args.out, "all_datasets.md"), prov)
    write_tex(rows, os.path.join(args.out, "tables.tex"), prov)

    print(f"\nwrote {args.out}/all_datasets.json, all_datasets.md, tables.tex")

    bad = [r["label"] for r in rows
           if (r["parse_counts"] or {}).get("non_matching")]
    if bad:
        print(f"WARNING: unparseable lines in {', '.join(bad)} -- "
              f"the reduction there is not deduplication alone")
    for r in rows:
        d = r["derived"]
        for check in ("min_weight_matches_max_outdegree",
                      "dead_ends_equals_vertices_minus_subjects"):
            if d.get(check) is False:
                print(f"WARNING: {r['label']} fails check {check}")
    return 0


# =========================================================================== #
# 4. parse-counts: the parser's counters for one file   (was parse_counts.py)
# =========================================================================== #

def cmd_parse_counts(argv) -> int:
    import re
    from RDFparser import RDFParser

    ap = argparse.ArgumentParser(
        prog="tables.py parse-counts",
        description="Report RDFParser's deduplication counters for one file.")
    ap.add_argument("path", help="source file (.nt, .nq, optionally .gz)")
    ap.add_argument("--keep-lang", action="store_true",
                    help="keep language tags on literals instead of "
                         "stripping them")
    args = ap.parse_args(argv)

    p = RDFParser(args.path)
    if args.keep_lang:
        p._strip_lang_tag = re.compile(r"(?!x)(x)(x)")   # never matches, two groups
    for _ in p.parse():
        pass
    print(f"{args.path}")
    print(f"  lines        {p.counter_lines:,}")
    print(f"  distinct     {p.counter_yielded:,}")
    print(f"  duplicates   {p.counter_duplicates:,}")
    print(f"  non-matching {p.counter_NM:,}")
    return 0


# =========================================================================== #
# command line
# =========================================================================== #

COMMANDS = {
    "build": cmd_build,
    "dataset-stats": cmd_dataset_stats,
    "collect-stats": cmd_collect_stats,
    "parse-counts": cmd_parse_counts,
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
