#!/usr/bin/env python3
"""
bin_fidelity.py
===============

What a binning discards, measured on the values it is actually applied to.

    python bin_fidelity.py \
        --graph out/bsbm100m/graph_weighted.wnt \
        --measures out/bsbm100m/summaries/measures_bsbm100m.json \
                   out/bsbm100m/summaries_schemex/measures_bsbm100m.json \
        --out out/bsbm100m/bin_fidelity.json --no-per-bin

Pass --measures to describe the binnings a completed sweep actually used,
reconstructed from the boundaries it recorded. Pass --family and --m instead
to fit fresh ones, which is only appropriate when no summary depends on them.

Two distributions
-----------------
Section 3.4 derives the boundaries of the data-dependent families from X, the
multiset of non-zero edge weights pi(s,p,o). The binning is then applied not
to those weights but to the cell masses mu(s,P,O), which are sums of them.
The two coincide only where a cell holds a single triple.

This script computes both and reports, for each binning, where its boundaries
fall against each distribution. Where they diverge, a family that is
equal-frequency in pi is not equal-frequency in mu, and resolution is spent
where the values being compared are not.

mu is computed for sim_p the identity and sim_o the total relation, that is
one cell per (subject, predicate). That is the coarsest instantiation, and the
one every model in the evaluation uses under the union reading, so it is the
distribution the binning actually meets.

Quantisation error
------------------
For each bin, the mean absolute deviation of the values in it from their own
mean -- how far a value sits from the other values it has been made
indistinguishable from. Averaged over values, this is the per-value
quantisation error of the binning, and unlike the representative error of
run_summary.py it needs no summarisation, so it is cheap on every dataset.

Only dependencies: the project modules.
"""

from __future__ import annotations

import argparse
import json
import os
from array import array
from collections import defaultdict
from statistics import mean

from binning import FAMILIES, Binning, load_binnings, load_weights, make_binning


def cell_masses(wnt_path: str, progress_every: int = 0) -> array:
    """The multiset of mu(s,p) over the weighted-triple file.

    Cells are keyed by a hash of (subject, predicate) rather than the strings
    themselves: the dictionary holds one entry per cell, tens of millions on
    the larger graphs, and the keys would otherwise dominate memory. A
    collision would merge two cells, which at 64 bits is not worth guarding
    against for a descriptive statistic.
    """
    acc: dict = defaultdict(float)
    n = 0
    with open(wnt_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            s, p, w = parts[0], parts[1], parts[2]
            acc[hash((s, p))] += float(w.strip("[]"))
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[mu] {n:,} triples, {len(acc):,} cells")
    out = array("d", acc.values())
    print(f"[mu] {n:,} triples -> {len(out):,} cells")
    return out


def describe(values, b, label: str) -> dict:
    """Occupancy and quantisation error of one binning over one distribution."""
    per_bin: dict = defaultdict(list)
    for x in values:
        per_bin[b(x)].append(x)

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

    return {
        "distribution": label,
        "values": len(values),
        "bins_occupied": len(per_bin),
        "quantisation_error": total_abs_dev / len(values) if values else 0.0,
        "per_bin": bins,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="graph_weighted.wnt")
    ap.add_argument("--family", nargs="+", default=["all"],
                    choices=list(FAMILIES) + ["all"])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--seed", type=int, default=0)
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
    args = ap.parse_args()

    X = load_weights(args.graph)
    print(f"[pi] {len(X):,} edge weights, {len(set(X)):,} distinct")
    M = cell_masses(args.graph, args.progress)
    print(f"[mu] {len(set(M)):,} distinct cell masses")

    if args.measures:
        seen: dict = {}
        for path in args.measures:
            with open(path, encoding="utf-8") as fh:
                for r in json.load(fh)["results"]:
                    if not r.get("binning_boundaries"):
                        continue          # the deterministic row has no binning
                    label = r["binning"]
                    bounds = [float(x) for x in r["binning_boundaries"]]
                    if label in seen:
                        if seen[label].boundaries != bounds:
                            print(f"[warn] {label} has different boundaries in "
                                  f"{path}; keeping the first")
                        continue
                    seen[label] = Binning.from_dict({
                        "family": r["binning_family"],
                        "m": r["binning_effective_m"],
                        "boundaries": bounds,
                    })
        configs = sorted(seen.items())
        print(f"[setup] {len(configs)} binnings reconstructed from measures")
    elif args.binnings:
        configs = sorted(load_binnings(args.binnings).items())
    else:
        families = list(FAMILIES) if "all" in args.family else args.family
        configs = [(f"{f}_m{m}", make_binning(f, m, X, seed=args.seed))
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
        "pi_values": len(X),
        "pi_distinct": len(set(X)),
        "mu_values": len(M),
        "mu_distinct": len(set(M)),
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\n[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())