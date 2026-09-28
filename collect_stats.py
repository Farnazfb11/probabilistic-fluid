#!/usr/bin/env python3
"""
collect_stats.py
================

Collect every descriptive statistic reported for the datasets into one place,
with provenance, so the figures in the thesis can be traced to a recorded run.

    python collect_stats.py                 # reuse cached parse counts
    python collect_stats.py --parse         # re-run the parser on every dataset
    python collect_stats.py --only btc150m dyldo_2015-10-11

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

2. Graph statistics, from dataset_stats.py: written to <outdir>/dataset_stats.json
   when that script was run. This script reads that file rather than recomputing,
   so run dataset_stats.py first for any dataset whose graph has changed.

Outputs, all under stats/ in the repository so they can be committed:

    stats/all_datasets.json   every figure, plus provenance
    stats/all_datasets.md     a readable table
    stats/tables.tex          the two LaTeX table bodies used in the thesis

Emitting the LaTeX rows here is deliberate: the thesis tables are then a copy
of recorded output rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# The datasets. Edit the paths here if anything moves.
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def provenance() -> dict:
    """Everything needed to say which code produced these numbers."""
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
    try:
        st = os.stat(path)
        return {"path": path, "exists": True, "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                 .isoformat(timespec="seconds")}
    except FileNotFoundError:
        return {"path": path, "exists": False}


# --------------------------------------------------------------------------- #
# the two statistic sources
# --------------------------------------------------------------------------- #

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

    from RDFparser import RDFParser          # imported late: only needed when parsing

    t0 = time.time()
    p = RDFParser(source)
    for _ in p.parse():
        pass
    d = {
        "lines": p.counter_lines,
        "distinct": p.counter_yielded,
        "duplicates": p.counter_duplicates,
        "non_matching": p.counter_NM,
        "seconds": round(time.time() - t0, 1),
        "source": source,
        "from_cache": False,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(cache, "w") as fh:
        json.dump(d, fh, indent=2)
    return d


def graph_stats(outdir: str) -> dict | None:
    """Whatever dataset_stats.py wrote for this dataset."""
    path = os.path.join(outdir, "dataset_stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    # dataset_stats.py may write either a bare dict or one keyed by dataset name
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


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

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
    L = [f"% generated by collect_stats.py on {prov['generated_at']}",
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


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse", action="store_true",
                    help="re-run RDFParser even when cached counts exist (slow)")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="restrict to these dataset keys")
    ap.add_argument("--out", default=STATS_DIR, help=f"output directory (default {STATS_DIR})")
    args = ap.parse_args()

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
            print("no dataset_stats.json -- run dataset_stats.py for this dataset")
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


if __name__ == "__main__":
    main()