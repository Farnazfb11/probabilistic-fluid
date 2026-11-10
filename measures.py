#!/usr/bin/env python3
"""
measures.py
===========

The measures of the thesis that are computed apart from the summarisation run.

    python measures.py fidelity   --graph ... --measures ... --out ...
    python measures.py dispersion --graph ... --types ... --family all ...
    python measures.py unary      --packages ... --out ...

Each command takes the options of the script it replaces; run
``python measures.py <command> --help`` for them.

  fidelity    (was bin_fidelity.py)  quantisation error and non-empty bins
              of each binning, over the cell masses mu and the edge weights pi
  dispersion  (was dispersion.py)    within-class dispersion and
              representative error, recomputed from the graph
  unary       (was unary_report.py)  the unary measures over the exported
              summary packages

run_summary.py imports both_measures from here and records the representative
error and the dispersion of every run itself, so the dispersion command is
only needed for a sweep run with --no-rep-error, or to check a recorded one.

Only dependencies: numpy and the project modules. The unary command also
needs Unary_measure.py and save.py from the summarisation project.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from statistics import mean as _mean
from typing import Any, Dict, List, Optional

import numpy as np

from binning import (FAMILIES, Binning, load_binnings, load_cell_masses,
                     load_weights, make_binning, snap)

# Families whose boundaries are fitted to the observed values.
DATA_DEPENDENT = ("equal_frequency", "clustering")

# Models whose cells are (subject, predicate) under the union reading. Their
# masses are the mu computed by the fidelity command, and they share their
# boundaries when the boundaries are fitted to mu. CC has a single cell per
# subject, with mass one for every vertex that has triples.
PER_PREDICATE_MODELS = ("AC", "ACC", "SchemEX")


# =========================================================================== #
# 1. Within-class measures: representative error and dispersion
# =========================================================================== #
#
# The representative error measures the summary's output: the payload keeps
# the exact masses of one designated member, so a query against the summary is
# answered with that member's numbers, and the error is the distance to them.
# Its reference point is fixed by Definition (mass payload) and not chosen
# freely.
#
# It has one weakness. The representative is the first member encountered
# during collection, so it depends on the order the input is read rather than
# on any property of the class. Dispersion removes that dependence by taking
# each class's own mean as the reference. It answers a different question ---
# how tightly does the binning hold a class together --- and, being measured
# about the mean, is a property of the partition rather than of one member.
#
# Both are computed over the same population of (member, cell) pairs, so they
# are directly comparable. One asymmetry is deliberate: the representative
# error compares members[1:] against members[0], whereas dispersion includes
# every member, the representative having no privileged status about a mean.
#
# Deviations are accumulated into a histogram quantised to 1e-6 rather than a
# list, so the median is exact to that resolution while memory stays bounded
# by the number of distinct deviations. The mean and the maximum are exact.

PLACES = 6


class DevStats:
    """Streaming mean, median and maximum of a stream of deviations.

    The median comes from a histogram quantised to 1e-6; mean and maximum are
    exact. Zero deviations are counted but excluded from the summary
    statistics, so that a difference is recorded only when it is non-zero.
    """

    def __init__(self) -> None:
        self.pairs = 0          # every (member, cell) pair examined
        self.nonzero = 0        # those that actually disagree
        self.total = 0.0
        self.maximum = 0.0
        self.hist: Counter = Counter()
        self._q = 10 ** PLACES

    def add(self, d: float) -> None:
        self.pairs += 1
        if d <= 0.0:
            return
        self.nonzero += 1
        self.total += d
        if d > self.maximum:
            self.maximum = d
        self.hist[round(d * self._q)] += 1

    def summary(self) -> dict:
        if not self.nonzero:
            return {"pairs": self.pairs, "cells": 0,
                    "mean": 0.0, "median": 0.0, "max": 0.0}
        half = self.nonzero / 2.0
        cum = 0
        median = 0.0
        for key in sorted(self.hist):
            cum += self.hist[key]
            if cum >= half:
                median = key / self._q
                break
        return {
            "pairs": self.pairs,
            "cells": self.nonzero,
            "mean": self.total / self.nonzero,
            "median": median,
            "max": self.maximum,
        }


def both_measures(summaries: dict, groups: dict) -> dict:
    """Representative error and within-class dispersion, in one pass.

    For each class of at least two members, the cells considered are the union
    of the members' cells, a cell absent from a member counting as mass zero.
    Members of a binned class share their cell keys by construction, since the
    bins enter the signature; in the deterministic run they need not, and the
    union is then what makes the comparison well defined.
    """
    rep = DevStats()
    dis = DevStats()
    key_mismatch = 0
    classes_used = 0

    for members in groups.values():
        if len(members) < 2:
            continue
        classes_used += 1

        masses = [summaries[m].mass for m in members]
        keys: set = set()
        for mm in masses:
            keys |= set(mm)
        if any(len(mm) != len(keys) for mm in masses):
            key_mismatch += 1

        reference = masses[0]
        for cell in keys:
            values = [mm.get(cell, 0.0) for mm in masses]

            # representative error: members[1:] against members[0]
            r = reference.get(cell, 0.0)
            for v in values[1:]:
                rep.add(abs(v - r))

            # dispersion: every member against the class mean
            mu = _mean(values)
            for v in values:
                dis.add(abs(v - mu))

    return {
        "classes_compared": classes_used,
        "classes_with_differing_cells": key_mismatch,
        "representative_error": rep.summary(),
        "dispersion": dis.summary(),
    }


def load_expected(paths) -> dict:
    """Recorded measures, keyed by (model, binning).

    Several measures files may cover one dataset, so the first file supplying
    a value for a given (model, binning) wins, as in tables.py. Each entry
    holds the recorded representative error, if any, and the recorded
    boundaries.
    """
    out = {}
    for path in (paths or []):
        if not os.path.exists(path):
            print(f"[warn] no such measures file: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for r in doc.get("results", []):
            key = (r["model"], r["binning"])
            entry = out.setdefault(key, {})
            err = r.get("representative_error") or {}
            if "error" not in entry and err.get("cells_compared"):
                entry["error"] = err
            if "binning" not in entry and r.get("binning_boundaries"):
                entry["binning"] = Binning.from_dict({
                    "family": r["binning_family"],
                    "m": r["binning_effective_m"],
                    "boundaries": [float(x) for x in r["binning_boundaries"]],
                })
    return out


DISPERSION_HELP = """\
Within-class dispersion and representative error, recomputed from the graph.

    python measures.py dispersion \\
        --graph  out/bsbm1000/graph_weighted.wnt \\
        --types  out/bsbm1000/vertex_types.tsv \\
        --deterministic --family all --m 1 5 10 25 \\
        --model AC CC ACC SchemEX \\
        --expect out/bsbm1000/final/measures_bsbm1000.json \\
        --out out/bsbm1000/dispersion

The boundaries are fitted to the cell masses mu of each model, as in
run_summary.py: the masses are collected once per model, the binnings are
fitted to them, and every configuration reuses the same summaries.
--binnings loads a fixed set instead, for reproducing an old sweep.

With --expect, the boundaries recorded in that measures file are used where
present, so the partition is the one the summary was built on, and the
boundaries refitted here are compared with them. The recomputed
representative error is printed beside the recorded one. They must agree: a
mismatch means the partition differs from the one the summary was built on,
and the dispersion figure cannot be trusted either.
"""


def cmd_dispersion(argv) -> int:
    from pgraph_io import load_weighted_graph
    from Collector import SchemaCollector, mass_values
    from Summarizer import group_by_signature
    from run_summary import MODELS

    ap = argparse.ArgumentParser(
        prog="measures.py dispersion", description=DISPERSION_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--types", default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="include the run with no binning")
    ap.add_argument("--family", nargs="+", default=None,
                    choices=list(FAMILIES) + ["all"],
                    help="binning families to fit to each model's masses")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--binnings", default=None,
                    help="binnings.json to load instead of fitting; for "
                         "reproducing a sweep made before the boundaries were "
                         "fitted to mu")
    ap.add_argument("--config", nargs="+", default=None,
                    help="with --binnings: keys from that file, or 'all'")
    ap.add_argument("--model", nargs="+", default=["SchemEX"],
                    choices=list(MODELS))
    ap.add_argument("--expect", nargs="+", default=None,
                    help="measures_<dataset>.json files to validate against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not (args.deterministic or args.family or args.binnings):
        ap.error("give --deterministic, --family or --binnings")

    G = load_weighted_graph(args.graph, args.types)
    print(f"[graph] {G.number_of_nodes():,} vertices, "
          f"{G.number_of_edges():,} triples")

    # configurations: (label, family, m, fixed binning or None)
    configs: list = []
    if args.deterministic:
        configs.append(("deterministic", None, None, None))
    if args.family:
        families = list(FAMILIES) if "all" in args.family else list(args.family)
        for f in families:
            for m in args.m:
                configs.append((f"{f}_m{m}", f, m, None))
    if args.binnings:
        loaded = load_binnings(args.binnings)
        keys = args.config or ["all"]
        wanted = sorted(loaded) if "all" in keys else keys
        for k in wanted:
            if k not in loaded:
                raise SystemExit(f"unknown config {k!r}; available: "
                                 f"{', '.join(sorted(loaded))}")
            configs.append((k, loaded[k].family, loaded[k].m, loaded[k]))

    expected = load_expected(args.expect)

    print(f"\n{'model':<8}{'binning':<22}"
          f"{'rep mean':>11}{'disp mean':>11}{'ratio':>7}"
          f"{'rep max':>11}{'disp max':>11}  check")
    print("-" * 100)

    results = []
    for model_name in args.model:
        t0 = time.perf_counter()
        collector = SchemaCollector(MODELS[model_name]())
        summaries = collector.collect_masses(G)
        t_collect = time.perf_counter() - t0
        X = None

        for label, family, m, fixed in configs:
            t0 = time.perf_counter()
            exp = expected.get((model_name, label), {})
            boundaries_check = "-"

            if family is None:
                b = None
            elif fixed is not None:
                b = fixed
            else:
                if family in DATA_DEPENDENT and X is None:
                    X = mass_values(summaries)
                b = make_binning(family, m, X, seed=args.seed)
                recorded = exp.get("binning")
                if recorded is not None:
                    same = (len(recorded.boundaries) == len(b.boundaries) and
                            all(abs(x - y) <= 1e-12 for x, y in
                                zip(recorded.boundaries, b.boundaries)))
                    boundaries_check = "bounds OK" if same else "bounds DIFFER"
                    # the recorded boundaries are the ones the summary used
                    b = recorded

            ids = SchemaCollector.signatures(summaries, b)
            groups = group_by_signature(ids)
            res = both_measures(summaries, groups)
            res["model"] = model_name
            res["binning"] = label
            res["classes"] = len(groups)
            res["seconds"] = round(time.perf_counter() - t0, 1)
            res["seconds_collect_shared"] = round(t_collect, 1)
            if b is not None:
                res["binning_boundaries"] = b.boundaries
                res["binning_effective_m"] = b.m
            res["boundaries_check"] = boundaries_check

            check = "-"
            err = exp.get("error")
            if err and err.get("cells_compared"):
                got = res["representative_error"]
                same_n = got["cells"] == err["cells_compared"]
                same_mean = abs(got["mean"] - err["mean"]) <= 1e-9 * max(
                    1.0, abs(err["mean"]))
                check = "OK" if (same_n and same_mean) else "MISMATCH"
                res["expected_representative_error"] = err
            res["validation"] = check

            r, d = res["representative_error"], res["dispersion"]
            ratio = (d["mean"] / r["mean"]) if r["mean"] else 0.0
            print(f"{model_name:<8}{label:<22}"
                  f"{r['mean']:>11.3e}{d['mean']:>11.3e}{ratio:>7.2f}"
                  f"{r['max']:>11.3e}{d['max']:>11.3e}  {check}"
                  f"{'' if boundaries_check == '-' else '  ' + boundaries_check}")

            results.append(res)
            del ids, groups

        del summaries, X

    mismatches = [r for r in results if r["validation"] == "MISMATCH"]
    if mismatches:
        print(f"\n[warn] {len(mismatches)} run(s) did not reproduce the "
              f"recorded representative error. The partition computed here "
              f"differs from the one the summary was built on; do not report "
              f"the dispersion for those rows until that is resolved.")

    differ = [r for r in results if r["boundaries_check"] == "bounds DIFFER"]
    if differ:
        print(f"\n[note] {len(differ)} run(s) refitted boundaries that differ "
              f"from the recorded ones; the recorded ones were used.")

    odd = [r for r in results if r["classes_with_differing_cells"]
           and r["binning"] != "deterministic"]
    if odd:
        print(f"\n[note] {len(odd)} binned run(s) contain classes whose members "
              f"hold different cell keys. Under the union reading every member "
              f"of a binned class should hold the same keys, so this indicates "
              f"add_mass is not aggregating per predicate class.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(os.path.dirname(args.graph)) or "graph"
        path = os.path.join(args.out, f"dispersion_{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"graph": args.graph, "fitted_on": "mu",
                       "results": results}, fh, indent=2)
        print(f"\n[ok] wrote {path} ({len(results)} runs)")
    return 0


# =========================================================================== #
# 2. Binning: quantisation error and non-empty bins
# =========================================================================== #
#
# The binning is applied to the cell masses mu(s,P,O), and the boundaries of
# the data-dependent families are fitted to those same masses (run_summary.py,
# --fit-on mu). Earlier sweeps fitted them to the edge weights pi instead; the
# two coincide only where a cell holds a single triple. The fidelity command
# reports, for each binning, where its boundaries fall against each
# distribution. mu is the one that matters for the summary; pi shows how the
# same boundaries cut the weights the masses are built from.
#
# mu is computed for sim_p the identity and sim_o the total relation, one cell
# per (subject, predicate): the cell of AC, ACC and SchemEX under the union
# reading. CC rows are therefore skipped when reading --measures.
#
# The quantisation error of a bin is the mean absolute deviation of the values
# in it from their own mean -- how far a value sits from the other values it
# has been made indistinguishable from. Averaged over values, it is the
# per-value quantisation error of the binning. It needs no summarisation, so
# it is cheap on every dataset.


def cell_masses(wnt_path: str, progress_every: int = 0) -> np.ndarray:
    """The multiset of mu(s,p) over the weighted-triple file."""
    out = load_cell_masses(wnt_path, progress_every)
    print(f"[mu] {len(out):,} cells")
    return out


def describe(values, b, label: str) -> dict:
    """Occupancy and quantisation error of one binning over one distribution."""
    per_bin: dict = defaultdict(list)
    for x in values:
        per_bin[b(x)].append(float(x))

    bins = []
    total_abs_dev = 0.0
    for key in sorted(per_bin):
        v = per_bin[key]
        mu_bin = mean(v)
        dev = sum(abs(x - mu_bin) for x in v)
        total_abs_dev += dev
        bins.append({
            "bin": key,
            "count": len(v),
            "min": min(v),
            "max": max(v),
            "mean": mu_bin,
            "width_realised": max(v) - min(v),
            "mean_abs_dev": dev / len(v),
        })

    n = len(values)
    return {
        "distribution": label,
        "values": n,
        "bins_occupied": len(per_bin),
        "quantisation_error": total_abs_dev / n if n else 0.0,
        "per_bin": bins,
    }


def _n_distinct(values) -> int:
    return len({snap(x) for x in values})


FIDELITY_HELP = """\
What a binning discards, measured on the values it is actually applied to.

    python measures.py fidelity \\
        --graph out/bsbm100m/graph_weighted.wnt \\
        --measures out/bsbm100m/final/measures_bsbm100m.json \\
        --out out/bsbm100m/bin_fidelity.json --no-per-bin

Pass --measures to describe the binnings a completed sweep actually used,
reconstructed from the boundaries it recorded (AC, ACC and SchemEX rows; CC
has a different cell). Pass --family and --m instead to fit fresh ones, which
is only appropriate when no summary depends on them.
"""


def cmd_fidelity(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="measures.py fidelity", description=FIDELITY_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="graph_weighted.wnt")
    ap.add_argument("--family", nargs="+", default=["all"],
                    choices=list(FAMILIES) + ["all"])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-on", choices=("mu", "pi"), default="mu",
                    help="with --family/--m: the distribution to fit to")
    ap.add_argument("--binnings", default=None,
                    help="binnings.json to use instead of --family/--m")
    ap.add_argument("--measures", nargs="+", default=None,
                    help="measures_*.json files; measure the boundaries those "
                         "runs actually recorded instead of refitting. Use "
                         "this to describe summaries that already exist: "
                         "equal_width and equal_frequency refit identically, "
                         "but clustering goes through k-means and need not.")
    ap.add_argument("--out", default=None, help="output JSON")
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--no-per-bin", action="store_true",
                    help="omit the per-bin detail from the JSON")
    args = ap.parse_args(argv)

    X = load_weights(args.graph)
    print(f"[pi] {len(X):,} edge weights, {_n_distinct(X):,} distinct")
    M = cell_masses(args.graph, args.progress)
    print(f"[mu] {_n_distinct(M):,} distinct cell masses")

    fitted_on = None
    if args.measures:
        seen: dict = {}
        skipped = 0
        for path in args.measures:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            fitted_on = doc.get("fit_on", fitted_on or "pi")
            for r in doc["results"]:
                if not r.get("binning_boundaries"):
                    continue          # the deterministic row has no binning
                if r.get("model") not in PER_PREDICATE_MODELS:
                    skipped += 1      # CC: a different cell
                    continue
                label = r["binning"]
                bounds = [float(x) for x in r["binning_boundaries"]]
                if label in seen:
                    if seen[label].boundaries != bounds:
                        print(f"[warn] {label} has different boundaries for "
                              f"{r['model']} in {path}; keeping the first")
                    continue
                seen[label] = Binning.from_dict({
                    "family": r["binning_family"],
                    "m": r["binning_effective_m"],
                    "boundaries": bounds,
                })
        configs = sorted(seen.items())
        print(f"[setup] {len(configs)} binnings reconstructed from measures"
              f"{f', {skipped} CC rows skipped' if skipped else ''}")
    elif args.binnings:
        configs = sorted(load_binnings(args.binnings).items())
    else:
        fitted_on = args.fit_on
        values = M if args.fit_on == "mu" else X
        families = list(FAMILIES) if "all" in args.family else args.family
        configs = [(f"{f}_m{m}", make_binning(f, m, values, seed=args.seed))
                   for f in families for m in args.m]

    results = []
    print(f"\n{'binning':<24} {'req m':>5} {'eff m':>5} "
          f"{'pi bins':>8} {'mu bins':>8} {'pi err':>10} {'mu err':>10}")
    print("-" * 76)
    for label, b in configs:
        d_pi = describe(X, b, "pi")
        d_mu = describe(M, b, "mu")
        req = int(label.rsplit("_m", 1)[1]) if "_m" in label else None
        row = {
            "binning": label,
            "family": getattr(b, "family", None),
            "requested_m": req,
            "effective_m": getattr(b, "m", None),
            "boundaries": getattr(b, "boundaries", None),
            "pi": d_pi,
            "mu": d_mu,
        }
        if args.no_per_bin:
            row["pi"].pop("per_bin")
            row["mu"].pop("per_bin")
        results.append(row)
        print(f"{label:<24} {str(req):>5} {str(b.m):>5} "
              f"{d_pi['bins_occupied']:>8} {d_mu['bins_occupied']:>8} "
              f"{d_pi['quantisation_error']:>10.6f} "
              f"{d_mu['quantisation_error']:>10.6f}")

    doc = {
        "graph": args.graph,
        "fitted_on": fitted_on,
        "pi_values": len(X),
        "pi_distinct": _n_distinct(X),
        "mu_values": len(M),
        "mu_distinct": _n_distinct(M),
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\n[ok] wrote {args.out}")
    return 0


# =========================================================================== #
# 3. Unary measures over the exported summary packages
# =========================================================================== #
#
# The counterpart of the plotting script from the temporal project. There the
# x-axis was time, so one row per weekly snapshot made sense. Here a dataset
# has 52 packages, one per (model, binning) pair, so the rows are keyed by
# those instead and nothing is plotted: the numbers go to JSON, CSV and a
# readable table.
#
# Each row is appended to <out>/rows.jsonl as soon as it is computed, and a
# re-run skips any (dataset, model, binning) already in that file, so a run
# that dies halfway loses only the package it was working on. --force
# recomputes everything.
#
# The expensive part is the component counts, which walk the summary graph.
# By default only the weak and strong counts are computed. --extra-components
# adds the node-threshold and payload-filtered variants, which multiply the
# work by roughly six.

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


def unary_row(pkg: Dict[str, Any], *, extra_components: bool,
              min_payload: int) -> Dict[str, Any]:
    """The unary measures for one summary package."""
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


def write_unary_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
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


UNARY_HELP = """\
Run the unary measures over the exported summary packages and write the
numbers to disk.

    python measures.py unary \\
        --packages ~/probabilistic_graphs/datasets/out/*/final/packages \\
        --out out/unary
"""


def cmd_unary(argv) -> int:
    from save import load_summary_package

    ap = argparse.ArgumentParser(
        prog="measures.py unary", description=UNARY_HELP,
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
    args = ap.parse_args(argv)

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
            row = unary_row(pkg, extra_components=args.extra_components,
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
    write_unary_outputs(rows, out_dir)
    return 0


# =========================================================================== #
# command line
# =========================================================================== #

COMMANDS = {
    "fidelity": cmd_fidelity,
    "dispersion": cmd_dispersion,
    "unary": cmd_unary,
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
