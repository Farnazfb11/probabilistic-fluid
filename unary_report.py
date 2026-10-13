#!/usr/bin/env python3
"""
unary_report.py
===============

Run the unary measures over the exported summary packages and write the
numbers to disk.

    python unary_report.py \
        --packages ~/probabilistic_graphs/datasets/out/*/packages \
        --out out/unary

This is the counterpart of the plotting script from the temporal project.
There the x-axis was time, so one row per weekly snapshot made sense. Here a
dataset has 52 packages, one per (model, binning) pair, so the rows are keyed
by those instead and nothing is plotted: the numbers go to JSON, CSV and a
readable table, and the thesis decides later which of them become figures.

Resumability
------------
Each row is appended to <out>/rows.jsonl as soon as it is computed, and a
re-run skips any (dataset, model, binning) already in that file. A run of a
few hundred packages that dies halfway therefore loses only the package it
was working on. Pass --force to recompute everything.

Cost
----
The expensive part is the component counts, which walk the summary graph.
By default only the weak and strong counts are computed. --extra-components
adds the node-threshold and payload-filtered variants of the old script,
which multiply the work by roughly six, so turn them on once you know the
basic numbers are worth refining.

Only dependencies: the project modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from save import load_summary_package
from Unary_measure import (
    average_degree,
    average_eqc_size,
    average_in_degree,
    average_out_degree,
    connected_components_count,
    connected_components_count_no_isolated_out,
    connected_components_count_payload_filtered,
    count_equivalence_classes,
    edge_set,
    std_eqc_size,
    tmh_tendency_to_make_hubs,
    total_payload,
)

# summary_<dataset>_<model>_<binning>.json.gz
# The dataset name may contain underscores (dyldo_2015-10-11) and so may the
# binning label (equal_width_m5), so the model names are listed explicitly
# rather than split on "_". ACC and SchemEX precede AC and CC so that the
# longer names win.
FILENAME = re.compile(
    r"^summary_(?P<dataset>.+)_(?P<model>SchemEX|ACC|AC|CC)_(?P<binning>.+)\.json\.gz$"
)

MODEL_ORDER = {"AC": 0, "CC": 1, "ACC": 2, "SchemEX": 3}
FAMILY_ORDER = {"deterministic": 0, "equal_width": 1, "equal_frequency": 2,
                "clustering": 3}


def parse_name(path: Path) -> Optional[Dict[str, str]]:
    m = FILENAME.match(path.name)
    return m.groupdict() if m else None


def binning_sort_key(label: str):
    """Order the binnings as a reader expects: deterministic, then by family and m."""
    if label == "deterministic":
        return (0, 0, 0)
    m = re.match(r"^(?P<family>.+)_m(?P<m>\d+)$", label)
    if not m:
        return (1, 99, 0)
    return (1, FAMILY_ORDER.get(m.group("family"), 98), int(m.group("m")))


def row_sort_key(r: Dict[str, Any]):
    return (r["dataset"], MODEL_ORDER.get(r["model"], 9),
            binning_sort_key(r["binning"]))


def measure(pkg: Dict[str, Any], *, extra_components: bool,
            min_payload: int) -> Dict[str, Any]:
    """The unary measures for one summary package."""
    row: Dict[str, Any] = {
        "eqc": int(count_equivalence_classes(pkg)),
        "total_payload": int(total_payload(pkg)),
        "avg_eqc_size": float(average_eqc_size(pkg)),
        "std_eqc_size": float(std_eqc_size(pkg, population=False)),

        "edges_unweighted": len(edge_set(pkg, include_weights=False)),
        "edges_weighted": sum(w for *_, w in edge_set(pkg, include_weights=True)),

        "avg_out_degree_w": float(average_out_degree(pkg, weighted=True)),
        "avg_out_degree_u": float(average_out_degree(pkg, weighted=False)),
        "avg_in_degree_w": float(average_in_degree(pkg, weighted=True)),
        "avg_in_degree_u": float(average_in_degree(pkg, weighted=False)),
        "avg_degree_w": float(average_degree(pkg, weighted=True)),
        "avg_degree_u": float(average_degree(pkg, weighted=False)),

        "tmh_w": float(tmh_tendency_to_make_hubs(pkg, weighted=True)),
        "tmh_u": float(tmh_tendency_to_make_hubs(pkg, weighted=False)),

        "comp_weak": int(connected_components_count(pkg, weak=True)),
        "comp_strong": int(connected_components_count(pkg, weak=False)),
    }

    if extra_components:
        for n in (3, 5, 7):
            row[f"comp_weak_ge_{n}"] = int(connected_components_count(
                pkg, weak=True, min_entity_nodes=n, min_payload=1))
            row[f"comp_strong_ge_{n}"] = int(connected_components_count(
                pkg, weak=False, min_entity_nodes=n, min_payload=1))
        row["comp_weak_payload_filtered"] = int(
            connected_components_count_payload_filtered(
                pkg, weak=True, min_payload=min_payload))
        row["comp_strong_payload_filtered"] = int(
            connected_components_count_payload_filtered(
                pkg, weak=False, min_payload=min_payload))
        row["comp_weak_no_isolated_out"] = int(
            connected_components_count_no_isolated_out(pkg, weak=True))
        row["comp_strong_no_isolated_out"] = int(
            connected_components_count_no_isolated_out(pkg, weak=False))

    return row


def load_done(path: Path) -> Dict[tuple, Dict[str, Any]]:
    """Rows already computed, keyed by (dataset, model, binning)."""
    done: Dict[tuple, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated last line from a killed run
            done[(r["dataset"], r["model"], r["binning"])] = r
    return done


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=row_sort_key)

    # one JSON per dataset, so each sits beside that dataset's measures file
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)
    for dataset, rs in by_dataset.items():
        p = out_dir / f"unary_{dataset}.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump({"dataset": dataset, "results": rs}, fh, indent=2)
        print(f"[ok] {p}  ({len(rs)} packages)")

    # one CSV over everything, for a spreadsheet or a plotting script
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    p = out_dir / "unary_all.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {p}")

    # and a table to read
    p = out_dir / "unary_all.md"
    cols = [("model", "model", "{}"), ("binning", "binning", "{}"),
            ("eqc", "classes", "{:,}"), ("avg_eqc_size", "mean size", "{:.2f}"),
            ("std_eqc_size", "sd size", "{:.2f}"),
            ("edges_unweighted", "edges", "{:,}"),
            ("avg_out_degree_w", "out deg", "{:.2f}"),
            ("tmh_w", "TMH", "{:.1f}"),
            ("comp_weak", "weak", "{:,}"), ("comp_strong", "strong", "{:,}")]
    with p.open("w", encoding="utf-8") as fh:
        for dataset, rs in by_dataset.items():
            fh.write(f"\n## {dataset}\n\n")
            fh.write("| " + " | ".join(c[1] for c in cols) + " |\n")
            fh.write("|" + "|".join("---" for _ in cols) + "|\n")
            for r in rs:
                cells = []
                for key, _, fmt in cols:
                    v = r.get(key)
                    cells.append("" if v is None else fmt.format(v))
                fh.write("| " + " | ".join(cells) + " |\n")
    print(f"[ok] {p}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packages", nargs="+", required=True,
                    help="one or more directories holding summary_*.json.gz")
    ap.add_argument("--out", required=True,
                    help="directory for rows.jsonl and the reports")
    ap.add_argument("--extra-components", action="store_true",
                    help="also compute the threshold and payload-filtered "
                         "component counts (roughly six times the work)")
    ap.add_argument("--min-payload", type=int, default=40,
                    help="vcp threshold for the payload-filtered counts")
    ap.add_argument("--force", action="store_true",
                    help="recompute packages already present in rows.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "rows.jsonl"

    done = {} if args.force else load_done(jsonl)
    if done:
        print(f"[setup] {len(done)} packages already done, skipping those")

    files: List[Path] = []
    for d in args.packages:
        files.extend(sorted(Path(d).glob("summary_*.json.gz")))
    if not files:
        print("[info] no packages found", file=sys.stderr)
        return 1
    print(f"[setup] {len(files)} packages found")

    fh = jsonl.open("a", encoding="utf-8")
    rows = list(done.values())

    for i, path in enumerate(files, 1):
        meta = parse_name(path)
        if meta is None:
            print(f"[warn] skip {path.name}: unexpected filename")
            continue
        key = (meta["dataset"], meta["model"], meta["binning"])
        if key in done:
            continue

        t0 = time.perf_counter()
        try:
            pkg = load_summary_package(str(path))
            t_load = time.perf_counter() - t0
            t1 = time.perf_counter()
            row = measure(pkg, extra_components=args.extra_components,
                          min_payload=args.min_payload)
            t_measure = time.perf_counter() - t1
            del pkg  # the graph cache lives on the package; drop both together
        except Exception as e:
            print(f"[warn] skip {path.name}: {type(e).__name__}: {e}")
            continue

        row.update({
            "dataset": meta["dataset"],
            "model": meta["model"],
            "binning": meta["binning"],
            "file": str(path),
            "bytes": path.stat().st_size,
            "seconds_load": round(t_load, 2),
            "seconds_measure": round(t_measure, 2),
        })
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        rows.append(row)

        print(f"[{i:>4}/{len(files)}] {meta['dataset']:<18} {meta['model']:<8} "
              f"{meta['binning']:<22} {row['eqc']:>10,} classes  "
              f"load {t_load:>6.1f}s  measure {t_measure:>7.1f}s")

    fh.close()
    write_outputs(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())