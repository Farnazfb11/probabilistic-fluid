#!/usr/bin/env python3
"""
build_tables.py
===============

Collect every measure into one long table, plus a few pivots to look at.

    python build_tables.py --root ~/probabilistic_graphs/datasets/out \\
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
    quantisation error of each binning, over pi and over mu.

<root>/unary/rows.jsonl
    the unary measures over the exported summary packages.

Output
------
master.csv          one row per (dataset, model, binning), every column
pivot_*.csv         one measure, datasets x models as rows, binnings as columns
tables.xlsx         the same sheets in one workbook, if pandas is installed

The master table is the thing to read first and decide from; the pivots are
there because a 312-row sheet is hard to see a pattern in.

Only dependencies: the standard library (pandas optional, for the workbook).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Highest priority first. A row from an earlier directory wins.
SOURCES = ("final", "summaries_schemex", "summaries_reperr", "summaries")

FAMILY_ORDER = {"deterministic": 0, "equal_width": 1, "equal_frequency": 2,
                "clustering": 3}
MODEL_ORDER = {"AC": 0, "CC": 1, "ACC": 2, "SchemEX": 3}

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


def flatten(dataset: str, source: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """One measures row into one flat record."""
    err = r.get("representative_error") or {}
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

        "sec_binning": phase(r, "binning", "seconds"),
        "sec_collect": phase(r, "collect", "seconds"),
        "sec_summary_graph": phase(r, "summary_graph", "seconds"),
        "sec_rep_error": phase(r, "representative_error", "seconds"),
        "sec_export": phase(r, "export", "seconds"),
        "sec_total": r.get("seconds_total"),

        "rss_delta_collect_mb": phase(r, "collect", "rss_delta_mb"),
        "hwm_increase_collect_mb": phase(r, "collect", "hwm_increase_mb"),
        "process_peak_rss_mb": r.get("process_peak_rss_mb"),

        "source": source,
    }
    # seconds_total excludes the export by design; phase 2 is the model's own work
    c, g = out["sec_collect"], out["sec_summary_graph"]
    out["sec_phase2"] = (c + g) if (c is not None and g is not None) else None
    return out


def read_measures(root: str) -> Dict[Key, Dict[str, Any]]:
    """Resolve the measures rows across the source directories."""
    by_key: Dict[Key, Dict[str, Any]] = {}
    rank = {s: i for i, s in enumerate(SOURCES)}
    found: Dict[str, int] = defaultdict(int)

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
                        for f in ("rep_err_mean", "rep_err_median",
                                  "rep_err_max", "rep_err_cells"):
                            rec[f] = prev[f]
                        rec["rep_err_from"] = prev["source"]
                    by_key[key] = rec
                elif prev["rep_err_mean"] is None and rec["rep_err_mean"] is not None:
                    for f in ("rep_err_mean", "rep_err_median",
                              "rep_err_max", "rep_err_cells"):
                        prev[f] = rec[f]
                    prev["rep_err_from"] = src

    for s in SOURCES:
        if found[s]:
            print(f"[measures] {s:<20} {found[s]:>5} rows")
    print(f"[measures] {len(by_key)} resolved (dataset, model, binning) rows")
    return by_key


def add_bin_fidelity(root: str, by_key: Dict[Key, Dict[str, Any]]) -> None:
    """Quantisation error is per (dataset, binning): the same for every model."""
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
                if key[0] == dataset and key[2] == r["binning"]:
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
        p2, bin_s, mem = col("sec_phase2"), col("sec_binning"), col("rss_delta_collect_mb")
        out.append({
            "dataset": dataset,
            "model": model,
            "runs": len(rs),
            "phase2_s_mean": round(mean(p2), 1) if p2 else None,
            "phase2_s_min": round(min(p2), 1) if p2 else None,
            "phase2_s_max": round(max(p2), 1) if p2 else None,
            "binning_s_mean": round(mean(bin_s), 2) if bin_s else None,
            "rss_delta_mb_mean": round(mean(mem), 1) if mem else None,
            "peak_rss_mb": max((r["process_peak_rss_mb"] for r in rs
                                if r.get("process_peak_rss_mb") is not None),
                               default=None),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the out/ directory")
    ap.add_argument("--out", required=True, help="directory for the tables")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out)

    by_key = read_measures(root)
    if not by_key:
        print("[info] no measures files found")
        return 1
    add_bin_fidelity(root, by_key)
    add_unary(root, by_key)

    rows = sorted(by_key.values(), key=row_key)
    write_csv(rows, os.path.join(out, "master.csv"))

    pivots = {}
    for value in ("classes", "compression", "mean_size", "max_size",
                  "summary_edges", "rep_err_mean", "rep_err_max",
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


if __name__ == "__main__":
    raise SystemExit(main())