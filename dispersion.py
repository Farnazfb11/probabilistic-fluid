#!/usr/bin/env python3
"""
dispersion.py
=============

Within-class dispersion: how far the members of an equivalence class sit from
their own mean, as opposed to from a designated representative.

    python dispersion.py \
        --graph  out/bsbm1000/graph_weighted.wnt \
        --types  out/bsbm1000/vertex_types.tsv \
        --binnings out/bsbm1000/binnings.json \
        --config all --model AC CC ACC SchemEX \
        --expect out/bsbm1000/summaries/measures_bsbm1000.json \
        --out out/bsbm1000/dispersion

Why
---
The representative error of run_summary.py measures the summary's output: the
payload keeps the exact masses of one designated member, so a query against
the summary is answered with that member's numbers, and the error is the
distance to them. Its reference point is therefore fixed by Definition
(mass payload) and not chosen freely.

It has one weakness. The representative is the first member encountered during
collection, so it depends on the order the input is read rather than on any
property of the class. Dispersion removes that dependence by taking each
class's own mean as the reference. It answers a different question --- how
tightly does the binning hold a class together --- and, being measured about
the mean, is a property of the partition rather than of one member.

Both are computed here over the same population of (member, cell) pairs, so
they are directly comparable. Note one asymmetry, kept deliberately so that
the figure matches run_summary.py: the representative error compares
members[1:] against members[0], whereas dispersion includes every member,
the representative having no privileged status about a mean.

Validation
----------
Pass --expect with the measures file of the corresponding run. The script then
prints the recomputed representative error beside the recorded one. They must
agree: the partition is recomputed here from the recorded binning boundaries
through the same collector, so a mismatch means the partition differs from the
one the summary was built on, and the dispersion figure cannot be trusted
either. This is the check to look at first.

Memory
------
Deviations are accumulated into a histogram quantised to 1e-6 rather than a
list, so the median is exact to that resolution while memory stays bounded by
the number of distinct deviations. The mean and the maximum are exact.

Only dependencies: the project modules (networkx comes in through pgraph_io).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from statistics import mean as _mean

from binning import load_binnings
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector
from Summarizer import group_by_signature

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

PLACES = 6


class DevStats:
    """Streaming mean, median and maximum of a stream of deviations.

    The median comes from a histogram quantised to 1e-6; mean and maximum are
    exact. Zero deviations are counted but excluded from the summary
    statistics, matching representative_error of run_summary.py, which records
    a difference only when it is non-zero.
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
    bins enter the signature; under --config deterministic they need not, and
    the union is then what makes the comparison well defined.
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


def load_expected(path: str | None) -> dict:
    """Recorded representative errors, keyed by (model, binning)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out = {}
    for r in doc.get("results", []):
        err = r.get("representative_error") or {}
        out[(r["model"], r["binning"])] = err
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--types", default=None)
    ap.add_argument("--binnings", default=None)
    ap.add_argument("--config", nargs="+", default=["deterministic"])
    ap.add_argument("--model", nargs="+", default=["SchemEX"],
                    choices=list(MODELS))
    ap.add_argument("--expect", default=None,
                    help="measures_<dataset>.json to validate against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    G = load_weighted_graph(args.graph, args.types)
    print(f"[graph] {G.number_of_nodes():,} vertices, "
          f"{G.number_of_edges():,} triples")

    binnings = load_binnings(args.binnings) if args.binnings else {}
    configs: list = []
    for c in args.config:
        if c == "deterministic":
            configs.append(("deterministic", None))
        elif c == "all":
            configs.extend(sorted(binnings.items()))
        elif c in binnings:
            configs.append((c, binnings[c]))
        else:
            raise SystemExit(f"unknown config {c!r}; available: "
                             f"{', '.join(sorted(binnings))} or deterministic")

    expected = load_expected(args.expect)

    print(f"\n{'model':<8}{'binning':<22}"
          f"{'rep mean':>11}{'disp mean':>11}{'ratio':>7}"
          f"{'rep max':>11}{'disp max':>11}  check")
    print("-" * 94)

    results = []
    for model_name in args.model:
        for label, b in configs:
            t0 = time.perf_counter()
            collector = SchemaCollector(MODELS[model_name]())
            summaries, ids = collector.collect(G, binning=b)
            groups = group_by_signature(ids)
            m = both_measures(summaries, groups)
            m["model"] = model_name
            m["binning"] = label
            m["classes"] = len(groups)
            m["seconds"] = round(time.perf_counter() - t0, 1)

            exp = expected.get((model_name, label))
            check = "-"
            if exp and exp.get("cells_compared"):
                got = m["representative_error"]
                same_n = got["cells"] == exp["cells_compared"]
                same_mean = abs(got["mean"] - exp["mean"]) <= 1e-9 * max(
                    1.0, abs(exp["mean"]))
                check = "OK" if (same_n and same_mean) else "MISMATCH"
                m["expected_representative_error"] = exp
            m["validation"] = check

            r, d = m["representative_error"], m["dispersion"]
            ratio = (d["mean"] / r["mean"]) if r["mean"] else 0.0
            print(f"{model_name:<8}{label:<22}"
                  f"{r['mean']:>11.3e}{d['mean']:>11.3e}{ratio:>7.2f}"
                  f"{r['max']:>11.3e}{d['max']:>11.3e}  {check}")

            results.append(m)
            del summaries, ids, groups

    mismatches = [r for r in results if r["validation"] == "MISMATCH"]
    if mismatches:
        print(f"\n[warn] {len(mismatches)} run(s) did not reproduce the "
              f"recorded representative error. The partition computed here "
              f"differs from the one the summary was built on; do not report "
              f"the dispersion for those rows until that is resolved.")

    odd = [r for r in results if r["classes_with_differing_cells"]]
    if odd:
        print(f"\n[note] {len(odd)} run(s) contain classes whose members hold "
              f"different cell keys. Under the union reading every member of a "
              f"class should hold the same keys, so this indicates add_mass "
              f"is not aggregating per predicate class.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        name = os.path.basename(os.path.dirname(args.graph)) or "graph"
        path = os.path.join(args.out, f"dispersion_{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"graph": args.graph, "results": results}, fh, indent=2)
        print(f"\n[ok] wrote {path} ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())