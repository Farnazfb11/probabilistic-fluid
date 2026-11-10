#!/usr/bin/env python3
"""
run_summary.py
==============

Compute probabilistic summaries and record their measures, with the cost of
every step.

    python run_summary.py \
        --graph out/dyldo/graph_weighted.wnt \
        --types out/dyldo/vertex_types.tsv \
        --deterministic --family all --m 1 5 10 25 \
        --model AC CC ACC SchemEX \
        --out out/dyldo/summaries \
        --save-summaries out/dyldo/summaries/packages

Steps
-----
For each summary model:

  collect   send and merge the messages of every triple, giving each vertex
            its summary with the exact cell masses mu(s,P,O). This does not
            depend on the binning, so it is done once per model and shared by
            all its configurations (pass --recollect to repeat it per run).

Then for each configuration of that model:

  phase 1   fit the bin boundaries to the masses just collected. Only the
            data-dependent families need the masses themselves; their
            extraction into one array is recorded as "mass_values", once per
            model. equal_width is closed-form and costs nothing here.
  phase 2   bin every vertex's masses ("partition": seal, signature, grouping)
            and build the summary graph ("summary_graph").

Fitting after collection is the change from earlier versions, which fitted
the boundaries to the edge weights pi before collection and applied them to
the masses mu. --fit-on pi restores that behaviour, for comparison.

Passing --deterministic runs phase 2 with no binning, which reproduces the
deterministic FLUID summary. Together with the m=1 configurations this is the
check of Remark 17.

Binning may also be loaded from a binnings.json with --binnings, for
reproducing a recorded summary; a loaded binning has no construction cost to
report, so the measures mark it as such rather than recording zero.

Fidelity
--------
Unless --no-rep-error is given, each run also records the representative
error and the within-class dispersion, both from measures.both_measures over
the summaries and partition the run has just built, so no second pass over the
graph is needed (measures.py dispersion recomputes them for a sweep run with
--no-rep-error).

Measuring memory
----------------
getrusage(RUSAGE_SELF).ru_maxrss is the peak resident size since the process
started and never decreases, so in a sweep every run after the first reports
the high-water mark of its predecessors. It is recorded once per sweep, as the
requirement of the whole job, and is not used per run. Per step we record

  rss_delta_mb      resident size after minus before, from /proc/self/status
  hwm_increase_mb   how much this step raised the process high-water mark;
                    zero means the step stayed below an earlier peak
  python_peak_mb    tracemalloc's peak for this step, with --tracemalloc only,
                    since tracing costs a substantial share of the runtime

None of the three is the whole truth on its own: rss_delta understates because
the allocator does not always return freed pages, and python_peak misses
buffers allocated outside Python. Reported together they bound the answer.
The memory the summary needs is that of the collect step; the shared collect
step is measured once per model and repeated in each of its runs.

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

import numpy as np

from binning import FAMILIES, load_binnings, load_weights, make_binning
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector, mass_values
from Summarizer import group_by_signature, build_summary_graph
from measures import both_measures
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

# Families whose boundaries are derived from the observed values, and which
# therefore need those values before the binning can be built.
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
    """Time and memory for one step of work.

    Used as a context manager; the measurements are available afterwards via
    as_dict(). Steps are not nested, so the single global tracemalloc state
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


def collect_model(G, model_name: str, trace: bool):
    """The collect step: every vertex's summary with its exact masses."""
    with Phase("collect", trace) as ph:
        summaries = SchemaCollector(MODELS[model_name]()).collect_masses(G)
    return summaries, ph.as_dict()


def run_one(
    G,
    model_name: str,
    summaries: dict,
    binning,
    label: str,
    collect_phase: dict,
    values_phase,
    binning_phase: dict,
    trace: bool,
    want_fidelity: bool,
    save_to: str | None = None,
) -> dict:
    """Bin the collected summaries under one configuration and measure.

    collect_phase, values_phase and binning_phase describe the steps before
    phase 2 and what they cost; they are passed in because the collect step
    and the mass extraction are shared by every configuration of a model, and
    charging them to a single run would misreport them.
    """
    with Phase("partition", trace) as ph_part:
        ids = SchemaCollector.signatures(summaries, binning)
        groups = group_by_signature(ids)

    with Phase("summary_graph", trace) as ph_graph:
        super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)

    n_super_edges = sum(len(d) for d in super_counts.values())
    n_super_pred_edges = sum(
        len(preds) for dsts in super_pred.values() for preds in dsts.values()
    )
    del super_counts, super_pred
    dead_ends = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

    phases = {
        "collect": collect_phase,
        "mass_values": values_phase,
        "binning": binning_phase,
        "partition": ph_part.as_dict(),
        "summary_graph": ph_graph.as_dict(),
    }
    seconds_phase1 = binning_phase.get("seconds", 0.0)
    seconds_phase2 = ph_part.seconds + ph_graph.seconds
    seconds_total = (
        collect_phase.get("seconds", 0.0)
        + (values_phase or {}).get("seconds", 0.0)
        + seconds_phase1
        + seconds_phase2
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
        "seconds_phase1": round(seconds_phase1, 3),
        "seconds_phase2": round(seconds_phase2, 3),
        "seconds_total": round(seconds_total, 3),
        "process_peak_rss_mb": round(maxrss_mb(), 1),
    }
    result.update(class_size_distribution(groups))
    result["compression"] = (
        result["classes"] / G.number_of_nodes() if G.number_of_nodes() else 0.0
    )

    if want_fidelity:
        with Phase("fidelity", trace) as ph_fid:
            fid = both_measures(summaries, groups)
        rep = fid["representative_error"]
        # same keys as before, so tables.py reads it unchanged
        result["representative_error"] = {
            "cells_compared": rep["cells"],
            "pairs": rep["pairs"],
            "mean": rep["mean"],
            "median": rep["median"],
            "max": rep["max"],
        }
        result["dispersion"] = fid["dispersion"]
        result["classes_with_differing_cells"] = fid["classes_with_differing_cells"]
        phases["fidelity"] = ph_fid.as_dict()
    else:
        result["representative_error"] = None
        result["dispersion"] = None

    if save_to:
        with Phase("export", trace) as ph_exp:
            pkg = build_package(
                G, summaries, ids,
                strategy_name=model_name,
                meta={
                    "binning": label,
                    "binning_family": None if binning is None else binning.family,
                    "binning_effective_m": None if binning is None else binning.m,
                    "binning_boundaries": None if binning is None else binning.boundaries,
                    "binning_fitted_on": binning_phase.get("fitted_on"),
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
        result["binning_fitted_on"] = binning_phase.get("fitted_on")

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
                    help="binning families to fit here, so that phase 1 is "
                         "measured; combine with --m")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25],
                    help="numbers of non-zero bins to request")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-on", choices=("mu", "pi"), default="mu",
                    help="fit the boundaries to each model's cell masses mu "
                         "(default) or, as before, to the edge weights pi")
    ap.add_argument("--recollect", action="store_true",
                    help="repeat the collect step for every configuration "
                         "instead of sharing it within a model, so that each "
                         "run is measured on its own")

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
                    help="measure Python allocation peaks per step; accurate "
                         "but slows the run substantially")
    ap.add_argument("--no-rep-error", action="store_true",
                    help="skip the representative error and the dispersion, "
                         "the most expensive measures on large graphs")
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
    log.write(f"fit on  {args.fit_on}"
              f"{'  (collect repeated per run)' if args.recollect else ''}")
    log.write("=" * 78)

    # ---- shared setup: load the graph once ------------------------------- #
    with Phase("load_graph", args.tracemalloc) as ph_load:
        G = load_weighted_graph(args.graph, args.types)
    log.write(f"[setup] graph loaded in {ph_load.seconds:.1f}s, "
              f"rss {ph_load._rss1:.0f} MiB "
              f"({G.number_of_nodes():,} vertices, {G.number_of_edges():,} triples)")

    families = []
    if args.family:
        families = list(FAMILIES) if "all" in args.family else list(args.family)
    any_data_dependent = any(f in DATA_DEPENDENT for f in families)

    # ---- the edge weights, only for --fit-on pi -------------------------- #
    X_pi = None
    ph_weights = None
    if args.fit_on == "pi" and any_data_dependent:
        with Phase("load_weights", args.tracemalloc) as ph_weights:
            X_pi = np.asarray(load_weights(args.graph), dtype=float)
        log.write(f"[setup] {len(X_pi):,} edge weights read in "
                  f"{ph_weights.seconds:.1f}s for the data-dependent families")

    # ---- the configurations ---------------------------------------------- #
    # Each entry is (label, family, m, loaded binning or None). The binning
    # itself is fitted per model, after that model's masses are collected.
    configs: list = []
    if args.deterministic:
        configs.append(("deterministic", None, None, None))
    for family in families:
        for m in args.m:
            configs.append((f"{family}_m{m}", family, m, None))
    if args.binnings:
        loaded = load_binnings(args.binnings)
        keys = args.config or ["all"]
        wanted = sorted(loaded) if "all" in keys else keys
        for k in wanted:
            if k not in loaded:
                raise SystemExit(f"unknown config {k!r}; available: "
                                 f"{', '.join(sorted(loaded))}")
            configs.append((k, loaded[k].family, loaded[k].m, loaded[k]))

    log.write(f"[setup] {len(configs)} configurations x {len(args.model)} models "
              f"= {len(configs) * len(args.model)} runs")
    log.write("")
    log.write(f"{'model':<8} {'binning':<24} {'classes':>11} {'mean size':>10} "
              f"{'edges':>11} {'coll s':>8} {'fit s':>7} {'part s':>8} "
              f"{'total s':>8} {'dRSS MiB':>9}")
    log.write("-" * 114)

    # ---- the sweep -------------------------------------------------------- #
    results = []
    binnings_used: dict = {}
    seconds_collect_all = 0.0
    seconds_values_all = 0.0
    seconds_fit_all = 0.0

    for model_name in args.model:
        summaries = None
        collect_phase = None
        X_mu = None
        values_phase = None

        for label, family, m, fixed in configs:
            # ---- collect: once per model, or per run with --recollect ---- #
            if summaries is None or args.recollect:
                summaries = None          # release the previous run's first
                X_mu = None
                values_phase = None
                summaries, collect_phase = collect_model(G, model_name,
                                                         args.tracemalloc)
                collect_phase["shared"] = not args.recollect
                seconds_collect_all += collect_phase["seconds"]

            # ---- phase 1: the binning --------------------------------- #
            if family is None:
                b = None
                binning_phase = {"seconds": 0.0, "source": "none"}
            elif fixed is not None:
                b = fixed
                binning_phase = {"seconds": 0.0, "source": "loaded",
                                 "fitted_on": "loaded",
                                 "note": "built by binning.py; construction "
                                         "not measured here"}
            else:
                X = None
                if family in DATA_DEPENDENT:
                    if args.fit_on == "pi":
                        X = X_pi
                    else:
                        if X_mu is None:
                            with Phase("mass_values", args.tracemalloc) as ph_val:
                                X_mu = mass_values(summaries)
                            values_phase = ph_val.as_dict()
                            values_phase["values"] = int(X_mu.size)
                            values_phase["shared"] = not args.recollect
                            seconds_values_all += values_phase["seconds"]
                        X = X_mu
                with Phase("binning", args.tracemalloc) as ph_bin:
                    b = make_binning(family, m, X, seed=args.seed)
                binning_phase = ph_bin.as_dict()
                binning_phase.update({
                    "source": "computed",
                    # the sweep's setting; equal_width ignores the values
                    "fitted_on": args.fit_on,
                    "closed_form": family not in DATA_DEPENDENT,
                    "requested_m": m,
                    "effective_m": b.m,
                    "values": None if X is None else int(len(X)),
                })
                seconds_fit_all += binning_phase["seconds"]
            if b is not None:
                binnings_used.setdefault(model_name, {})[label] = b.to_dict()

            # values_phase is reported only for runs that used the masses
            vp = values_phase if (family in DATA_DEPENDENT
                                  and args.fit_on == "mu") else None

            save_to = (
                os.path.join(args.save_summaries,
                             f"summary_{name}_{model_name}_{label}.json.gz")
                if args.save_summaries else None
            )
            r = run_one(G, model_name, summaries, b, label, collect_phase, vp,
                        binning_phase, args.tracemalloc,
                        not args.no_rep_error, save_to)
            results.append(r)
            p = r["phases"]
            log.write(
                f"{model_name:<8} {label:<24} {r['classes']:>11,} "
                f"{r['mean_size']:>10.2f} {r['summary_edges']:>11,} "
                f"{p['collect']['seconds']:>8.1f} "
                f"{p['binning']['seconds']:>7.2f} "
                f"{p['partition']['seconds']:>8.1f} "
                f"{r['seconds_total']:>8.1f} "
                f"{p['collect']['rss_delta_mb']:>9.1f}"
            )

        summaries = None
        X_mu = None

    # ---- totals ----------------------------------------------------------- #
    setup = {
        "load_graph": ph_load.as_dict(),
        "load_weights": ph_weights.as_dict() if ph_weights is not None else None,
    }
    # The collect step and the mass extraction are counted once per time they
    # actually ran, the fits once per configuration, and phase 2 once per run.
    phase2_seconds = sum(r["seconds_phase2"] for r in results)
    total_run_seconds = (seconds_collect_all + seconds_values_all
                         + seconds_fit_all + phase2_seconds)
    shared_seconds = ph_load.seconds + (ph_weights.seconds if ph_weights else 0.0)

    log.write("-" * 114)
    log.write(f"[total] {len(results)} runs in {total_run_seconds:.1f}s "
              f"(collect {seconds_collect_all:.1f}s, mass values "
              f"{seconds_values_all:.1f}s, fitting {seconds_fit_all:.1f}s, "
              f"phase 2 {phase2_seconds:.1f}s), "
              f"shared setup {shared_seconds:.1f}s, "
              f"process peak rss {maxrss_mb():.0f} MiB")
    log.write("")

    if args.out:
        path = os.path.join(args.out, f"measures_{name}.json")
        doc = {
            "provenance": prov,
            "graph": args.graph,
            "types": args.types,
            "fit_on": args.fit_on,
            "recollect": args.recollect,
            "setup": setup,
            "seconds_shared_setup": round(shared_seconds, 3),
            "seconds_all_runs": round(total_run_seconds, 3),
            "seconds_collect": round(seconds_collect_all, 3),
            "seconds_mass_values": round(seconds_values_all, 3),
            "seconds_phase1_binning": round(seconds_fit_all, 3),
            "seconds_phase2_summarisation": round(phase2_seconds, 3),
            "process_peak_rss_mb": round(maxrss_mb(), 1),
            "results": results,
            "save_summaries": bool(args.save_summaries),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        log.write(f"[ok] wrote {path} ({len(results)} runs)")

        # The boundaries every run used, per model, since the masses they are
        # fitted to depend on the model's cells. Not in the binnings.json
        # format of binning.py, which holds one set for all models.
        bpath = os.path.join(args.out, f"binnings_used_{name}.json")
        with open(bpath, "w", encoding="utf-8") as fh:
            json.dump({"graph": args.graph, "fit_on": args.fit_on,
                       "by_model": binnings_used}, fh, indent=2)
        log.write(f"[ok] wrote {bpath}")
        if log_path:
            log.write(f"[ok] log appended to {log_path}")

    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
