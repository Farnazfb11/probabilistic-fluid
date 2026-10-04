#!/usr/bin/env python3
"""
run_summary.py
==============

Compute probabilistic summaries and record their measures, with the cost of
both phases.

    python run_summary.py \
        --graph out/dyldo/graph_weighted.wnt \
        --types out/dyldo/vertex_types.tsv \
        --family all --m 1 5 10 25 \
        --model AC ACC SchemEX \
        --out out/dyldo/summaries \
        --save-summaries out/dyldo/summaries/packages

Two phases
----------
The first phase determines the binning from the model parameters and the input
graph, and replaces each vertex's outgoing mass by the bin it falls into. This
is what the model adds. The second computes the quotient under EQR_p, for
which any partition refinement is adequate: once the mass is binned, the
condition of Definition 12 is an ordinary equivalence over discrete labels.

Passing --deterministic runs the second phase alone, with no binning,
which reproduces the deterministic FLUID summary. Together with the b_0
configuration this is the check of Remark 17.

Both phases are measured. Earlier versions loaded a binning precomputed by
binning.py, so phase 1 appeared to be free; it is not. Reading the weights and
placing the boundaries is work the model requires, and on the larger graphs it
is a visible share of the total.

Binning may be either computed here, from --family and --m, or loaded from a
binnings.json with --binnings. Loading is right for reproducing a recorded
summary, but a loaded binning has no construction cost to report, so the
measures mark it as such rather than recording zero.

Measuring memory
----------------
getrusage(RUSAGE_SELF).ru_maxrss is the peak resident size since the process
started and never decreases, so in a sweep every run after the first reports
the high-water mark of its predecessors. It is recorded once per sweep, as the
requirement of the whole job, and is not used per run. Per run we record

  rss_delta_mb      resident size after minus before, from /proc/self/status
  hwm_increase_mb   how much this run raised the process high-water mark;
                    zero means the run stayed below an earlier peak
  python_peak_mb    tracemalloc's peak for this run, with --tracemalloc only,
                    since tracing costs a substantial share of the runtime

None of the three is the whole truth on its own: rss_delta understates because
the allocator does not always return freed pages, and python_peak misses
buffers allocated outside Python. Reported together they bound the answer.

Only dependencies: networkx, numpy, plus the project modules.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import tracemalloc
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, median

from binning import FAMILIES, load_binnings, load_weights, make_binning
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector
from Summarizer import group_by_signature, build_summary_graph
from save import build_package, write_package

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

# Families whose boundaries are derived from the observed weights, and which
# therefore require a pass over the graph before the binning can be built.
DATA_DEPENDENT = ("equal_frequency", "clustering")


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def _proc_status_kb(field: str):
    """One VmXXX field of /proc/self/status, in KiB, or None off Linux."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def rss_mb() -> float:
    kb = _proc_status_kb("VmRSS:")
    return kb / 1024.0 if kb is not None else float("nan")


def hwm_mb() -> float:
    kb = _proc_status_kb("VmHWM:")
    return kb / 1024.0 if kb is not None else float("nan")


def maxrss_mb() -> float:
    """Process peak resident size since start. Monotone: a sweep-level figure."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Phase:
    """Time and memory for one phase of work.

    Used as a context manager; the measurements are available afterwards via
    as_dict(). Phases are not nested, so the single global tracemalloc state
    is safe to start and stop here.
    """

    def __init__(self, name: str, trace: bool = False):
        self.name = name
        self.trace = trace
        self.seconds = 0.0
        self.python_peak_mb = None

    def __enter__(self) -> "Phase":
        self._rss0 = rss_mb()
        self._hwm0 = hwm_mb()
        if self.trace:
            tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.seconds = time.perf_counter() - self._t0
        if self.trace:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.python_peak_mb = peak / (1024.0 * 1024.0)
        self._rss1 = rss_mb()
        self._hwm1 = hwm_mb()
        return False

    def as_dict(self) -> dict:
        d = {
            "seconds": round(self.seconds, 3),
            "rss_before_mb": round(self._rss0, 1),
            "rss_after_mb": round(self._rss1, 1),
            "rss_delta_mb": round(self._rss1 - self._rss0, 1),
            "hwm_increase_mb": round(self._hwm1 - self._hwm0, 1),
        }
        if self.python_peak_mb is not None:
            d["python_peak_mb"] = round(self.python_peak_mb, 1)
        return d


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _git(*args: str):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def provenance() -> dict:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
    }


# --------------------------------------------------------------------------- #
# measures
# --------------------------------------------------------------------------- #

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

    Linear in the number of vertices, but with a large constant: on the bigger
    graphs it can dominate the run, which is why --no-rep-error exists.
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


def run_one(
    G,
    model_name: str,
    binning,
    label: str,
    binning_phase: dict,
    trace: bool,
    want_rep_error: bool,
    save_to: str | None = None,
) -> dict:
    """Summarise G under one model and one binning, returning the measures.

    binning_phase describes how the binning for this configuration was
    obtained and what it cost; it is passed in because a binning is built once
    per configuration and then used by every model, so charging its
    construction to a single model would misreport it.
    """
    with Phase("collect", trace) as ph_collect:
        collector = SchemaCollector(MODELS[model_name]())
        summaries, ids = collector.collect(G, binning=binning)

    groups = group_by_signature(ids)

    with Phase("summary_graph", trace) as ph_graph:
        super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)

    n_super_edges = sum(len(d) for d in super_counts.values())
    n_super_pred_edges = sum(
        len(preds) for dsts in super_pred.values() for preds in dsts.values()
    )
    dead_ends = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

    phases = {
        "binning": binning_phase,
        "collect": ph_collect.as_dict(),
        "summary_graph": ph_graph.as_dict(),
    }
    seconds_total = (
        binning_phase.get("seconds", 0.0)
        + ph_collect.seconds
        + ph_graph.seconds
    )

    result = {
        "model": model_name,
        "binning": label,
        "input_vertices": G.number_of_nodes(),
        "input_triples": G.number_of_edges(),
        "dead_ends": dead_ends,
        "dead_end_share": dead_ends / max(1, G.number_of_nodes()),
        "summary_edges": n_super_edges,
        "summary_labelled_edges": n_super_pred_edges,
        "phases": phases,
        "seconds_total": round(seconds_total, 3),
        "process_peak_rss_mb": round(maxrss_mb(), 1),
    }
    result.update(class_size_distribution(groups))
    result["compression"] = (
        result["classes"] / G.number_of_nodes() if G.number_of_nodes() else 0.0
    )

    if want_rep_error:
        with Phase("representative_error", trace) as ph_rep:
            result["representative_error"] = representative_error(summaries, groups)
        phases["representative_error"] = ph_rep.as_dict()
    else:
        result["representative_error"] = None

    if save_to:
        with Phase("export", trace) as ph_exp:
            pkg = build_package(
                G, summaries, ids,
                strategy_name=model_name,
                include_pred_counts=False,
                meta={
                    "binning": label,
                    "binning_family": None if binning is None else binning.family,
                    "binning_effective_m": None if binning is None else binning.m,
                    "binning_boundaries": None if binning is None else binning.boundaries,
                    "classes": result["classes"],
                    "input_vertices": result["input_vertices"],
                    "input_triples": result["input_triples"],
                },
            )
            write_package(pkg, save_to)
        phases["export"] = ph_exp.as_dict()
        result["summary_path"] = save_to

    if binning is not None:
        result["binning_family"] = binning.family
        result["binning_effective_m"] = binning.m
        result["binning_boundaries"] = binning.boundaries

    return result


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #

class RunLog:
    """Append-only text log, written as the sweep proceeds.

    Written line by line and flushed, so a sweep killed by the scheduler or by
    memory pressure still leaves a record of everything that completed. The
    JSON file is only written at the end and would be lost in that case.
    """

    def __init__(self, path: str | None):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def write(self, line: str = "") -> None:
        print(line)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--graph", required=True, help="graph_weighted.wnt")
    ap.add_argument("--types", default=None, help="vertex_types.tsv")

    ap.add_argument("--family", nargs="+", default=None,
                    choices=list(FAMILIES) + ["all"],
                    help="binning families to build here, so that phase 1 is "
                         "measured; combine with --m")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25],
                    help="numbers of non-zero bins to request")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--binnings", default=None,
                    help="binnings.json written by binning.py; a binning "
                         "loaded this way has no construction cost to report")
    ap.add_argument("--config", nargs="+", default=None,
                    help="keys from that file, or 'all'")

    ap.add_argument("--deterministic", action="store_true",
                    help="also run with no binning at all")
    ap.add_argument("--model", nargs="+", default=["SchemEX"], choices=list(MODELS))
    ap.add_argument("--out", default=None, help="directory for the measures and log")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="measure Python allocation peaks per phase; accurate "
                         "but slows the run substantially")
    ap.add_argument("--no-rep-error", action="store_true",
                    help="skip the representative error, which is the most "
                         "expensive measure on large graphs")
    ap.add_argument("--save-summaries", default=None,
                    help="directory for the summary packages, one per model "
                         "and configuration")
    args = ap.parse_args()

    if not args.family and not args.binnings and not args.deterministic:
        ap.error("give --family, or --binnings, or --deterministic")

    name = os.path.basename(os.path.dirname(args.graph)) or "summary"
    log_path = os.path.join(args.out, f"run_{name}.log") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    log = RunLog(log_path)

    prov = provenance()
    log.write("=" * 78)
    log.write(f"run_summary  {prov['started_at']}  {prov['host']}")
    log.write(f"graph   {args.graph}")
    log.write(f"commit  {prov['git_commit']}"
              f"{'  (dirty)' if prov['git_dirty'] else ''}")
    log.write(f"models  {' '.join(args.model)}")
    log.write("=" * 78)

    # ---- shared setup: load the graph once ------------------------------- #
    with Phase("load_graph", args.tracemalloc) as ph_load:
        G = load_weighted_graph(args.graph, args.types)
    log.write(f"[setup] graph loaded in {ph_load.seconds:.1f}s, "
              f"rss {ph_load._rss1:.0f} MiB "
              f"({G.number_of_nodes():,} vertices, {G.number_of_edges():,} triples)")

    # ---- the weights, needed only by data-dependent families ------------- #
    families = []
    if args.family:
        families = list(FAMILIES) if "all" in args.family else list(args.family)

    X = None
    ph_weights = None
    if any(f in DATA_DEPENDENT for f in families):
        with Phase("load_weights", args.tracemalloc) as ph_weights:
            X = load_weights(args.graph)
        log.write(f"[setup] {len(X):,} weights read in {ph_weights.seconds:.1f}s "
                  f"for the data-dependent families")

    # ---- assemble the configurations ------------------------------------- #
    # Each entry is (label, binning, binning_phase_dict). The binning is built
    # once per configuration and shared by every model, so its cost is recorded
    # against the configuration rather than against one model's run.
    configs: list = []

    if args.deterministic:
        configs.append(("deterministic", None,
                        {"seconds": 0.0, "source": "none"}))

    for family in families:
        for m in args.m:
            with Phase("binning", args.tracemalloc) as ph_bin:
                b = make_binning(family, m, X, seed=args.seed)
            d = ph_bin.as_dict()
            d["source"] = "computed"
            d["requested_m"] = m
            d["effective_m"] = b.m
            # The weight pass is shared by every data-dependent configuration,
            # so it is reported separately rather than added to each of them.
            if family in DATA_DEPENDENT and ph_weights is not None:
                d["shared_weight_load_seconds"] = round(ph_weights.seconds, 3)
            configs.append((f"{family}_m{m}", b, d))

    if args.binnings:
        loaded = load_binnings(args.binnings)
        keys = args.config or ["all"]
        wanted = sorted(loaded) if "all" in keys else keys
        for k in wanted:
            if k not in loaded:
                raise SystemExit(f"unknown config {k!r}; available: "
                                 f"{', '.join(sorted(loaded))}")
            configs.append((k, loaded[k],
                            {"seconds": 0.0, "source": "loaded",
                             "note": "built by binning.py; construction not measured here"}))

    log.write(f"[setup] {len(configs)} configurations x {len(args.model)} models "
              f"= {len(configs) * len(args.model)} runs")
    log.write("")
    log.write(f"{'model':<8} {'binning':<24} {'classes':>11} {'mean size':>10} "
              f"{'edges':>11} {'bin s':>7} {'coll s':>8} {'total s':>8} {'dRSS MiB':>9}")
    log.write("-" * 110)

    # ---- the sweep -------------------------------------------------------- #
    results = []
    for label, b, binning_phase in configs:
        for model_name in args.model:
            save_to = (
                os.path.join(args.save_summaries,
                             f"summary_{name}_{model_name}_{label}.json.gz")
                if args.save_summaries else None
            )
            r = run_one(G, model_name, b, label, binning_phase,
                        args.tracemalloc, not args.no_rep_error, save_to)
            results.append(r)
            p = r["phases"]
            log.write(
                f"{model_name:<8} {label:<24} {r['classes']:>11,} "
                f"{r['mean_size']:>10.2f} {r['summary_edges']:>11,} "
                f"{p['binning']['seconds']:>7.1f} {p['collect']['seconds']:>8.1f} "
                f"{r['seconds_total']:>8.1f} "
                f"{p['collect']['rss_delta_mb']:>9.1f}"
            )

    # ---- totals ----------------------------------------------------------- #
    setup = {
        "load_graph": ph_load.as_dict(),
        "load_weights": ph_weights.as_dict() if ph_weights is not None else None,
    }
    # Each run's seconds_total includes its configuration's binning cost, which
    # is correct for "what one summarisation costs" but would double-count the
    # binning across the models sharing it. The sweep total therefore adds the
    # phase-2 work of every run to the phase-1 work of every configuration once.
    phase2_seconds = sum(
        r["phases"]["collect"]["seconds"] + r["phases"]["summary_graph"]["seconds"]
        for r in results
    )
    binning_seconds = sum(c[2].get("seconds", 0.0) for c in configs)
    total_run_seconds = phase2_seconds + binning_seconds
    shared_seconds = ph_load.seconds + (ph_weights.seconds if ph_weights else 0.0)

    log.write("-" * 110)
    log.write(f"[total] {len(results)} runs in {total_run_seconds:.1f}s, "
              f"shared setup {shared_seconds:.1f}s, "
              f"process peak rss {maxrss_mb():.0f} MiB")
    log.write("")

    if args.out:
        path = os.path.join(args.out, f"measures_{name}.json")
        doc = {
            "provenance": prov,
            "graph": args.graph,
            "types": args.types,
            "setup": setup,
            "seconds_shared_setup": round(shared_seconds, 3),
            "seconds_all_runs": round(total_run_seconds, 3),
            "seconds_phase1_binning": round(binning_seconds, 3),
            "seconds_phase2_summarisation": round(phase2_seconds, 3),
            "process_peak_rss_mb": round(maxrss_mb(), 1),
            "results": results,
            "save_summaries": bool(args.save_summaries),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        log.write(f"[ok] wrote {path} ({len(results)} runs)")
        if log_path:
            log.write(f"[ok] log appended to {log_path}")

    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())