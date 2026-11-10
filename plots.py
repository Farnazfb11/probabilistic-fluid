#!/usr/bin/env python3
"""
plots.py
========

The figures of the thesis that are drawn from the recorded measures.

    python plots.py bins --graph ... --measures ... --out ...

Each command takes the options of the script it replaces; run
``python plots.py <command> --help`` for them.

  bins   (was plot_bins.py)  where the bin boundaries fall against the
         edge weights pi and the cell masses mu, one figure per binning

Dependencies: matplotlib, seaborn, numpy, plus binning.py and measures.py.
"""


from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from binning import Binning, load_weights
from measures import PER_PREDICATE_MODELS, cell_masses

sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BLUE = "#1f77b4"
ORANGE_DARK = "#d95f02"


# =========================================================================== #
# bins: boundaries against the distributions   (was plot_bins.py)
# =========================================================================== #

BINS_HELP = """\
Where the bin boundaries fall against the distributions they are applied to.

    python plots.py bins \
        --graph out/bsbm100m/graph_weighted.wnt \
        --measures out/bsbm100m/summaries/measures_bsbm100m.json \
        --out out/bsbm100m/figures

One figure per binning, two panels. The upper panel is pi, the per-triple
edge weights; the lower is mu, the per-cell masses the binning is applied to
and, since the change to run_summary.py, also fitted to. Boundaries are drawn
over both, and each bin is annotated with the share of values it receives.

An equal-frequency binning should now give equal shares on mu, since that is
what it was fitted to, up to the bins lost on repeated values. The pi panel
shows how the same boundaries cut the weights the masses are built from. For
a sweep made with --fit-on pi the roles are reversed.

The boundaries are taken from the AC, ACC or SchemEX rows of the measures
file, whose (subject, predicate) cells are the mu plotted here. CC has a
single cell per subject and its own boundaries, so its rows are skipped.
"""


def binnings_from_measures(paths) -> dict:
    seen = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for r in json.load(fh)["results"]:
                if not r.get("binning_boundaries") or r["binning"] in seen:
                    continue
                if r.get("model") not in PER_PREDICATE_MODELS:
                    continue
                seen[r["binning"]] = Binning.from_dict({
                    "family": r["binning_family"],
                    "m": r["binning_effective_m"],
                    "boundaries": [float(x) for x in r["binning_boundaries"]],
                })
    return seen


def panel(ax, values, b, colour, ylabel, bins=200, log=True):
    """Histogram of one distribution with the bin boundaries drawn over it."""
    ax.hist(values, bins=bins, range=(0.0, 1.0), color=colour,
            edgecolor="none", alpha=0.85)
    if log:
        ax.set_yscale("log")

    for t in b.boundaries[1:-1]:
        ax.axvline(t, color="black", linestyle="--", linewidth=0.7)

    # share of values per bin, printed at the centre of each interval
    counts = b.occupancy(values)
    n = len(values)
    for j in range(1, b.m + 1):
        lo, hi = b.boundaries[j - 1], b.boundaries[j]
        share = counts.get(j, 0) / n if n else 0.0
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.45,
                f"{share:.0%}" if share >= 0.005 else "<1%",
                ha="center", va="center", fontsize=8, color="black")

    ax.set_xlim(0, 1)
    ax.set_ylabel(ylabel)


def cmd_bins(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="plots.py bins", description=BINS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--measures", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", default=None,
                    help="binning labels to plot (default: all)")
    ap.add_argument("--hist-bins", type=int, default=200)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(args.graph)) or "graph"

    X = np.asarray(load_weights(args.graph), dtype=float)
    M = np.asarray(cell_masses(args.graph), dtype=float)
    print(f"[data] {len(X):,} weights, {len(M):,} cell masses")

    for label, b in sorted(binnings_from_measures(args.measures).items()):
        if args.only and label not in args.only:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        panel(ax1, X, b, BLUE, r"$\pi$ values", args.hist_bins)
        panel(ax2, M, b, ORANGE_DARK, r"$\mu$ values", args.hist_bins)
        ax2.set_xlabel("probability mass")
        fig.tight_layout()
        path = os.path.join(args.out, f"bins_{name}_{label}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {path}")

    return 0


# =========================================================================== #
# command line
# =========================================================================== #

COMMANDS = {
    "bins": cmd_bins,
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
