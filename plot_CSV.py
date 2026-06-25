#!/usr/bin/env python3
"""
plot_from_csv.py

Regenerate line plots directly from an existing metrics.csv, WITHOUT titles.

Design:
  * SINGLE_METRICS maps a CSV column name -> (ylabel, output filename).
    Each known column is drawn to the SAME filename the original scripts used,
    so your LaTeX \\includegraphics paths keep working.
  * DUAL_AXIS_PAIRS handles twin-axis plots (e.g. weak vs strong components).
  * Any numeric column not in the config is still plotted generically (using
    the column name as the label) unless you pass --no-extra, so nothing is
    silently dropped.
  * No ax.set_title is ever called -> no caption on top of any plot.

Usage:
  python plot_CSV.py -i *.csv \
      --out-dir results_binary_attribute

  python plot_CSV.py -i results_unary_attribute/metrics.csv \
      --out-dir results_unary_attribute
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Same visual style as the original plotting scripts.
sns.set_theme(style="white")


# ----------------------------------------------------------------------
# CONFIG: match original CSV columns -> plots
# Edit / extend these to match YOUR exact column headers.
# Anything not listed here is still plotted generically (see --no-extra).
# ----------------------------------------------------------------------

# col_name : (y-axis label, output pdf filename)
SINGLE_METRICS: Dict[str, Tuple[str, str]] = {
    # ---- Binary_plot.py columns (matched to original labels/filenames) ----
    "jac_v":          ("δ²_JacV (vertex Jaccard distance)", "jaccard_vertices.pdf"),
    "jac_e":          ("δ²_JacE (edge Jaccard distance)",   "jaccard_edges.pdf"),
    "ged":            ("δ²_GED (edit operations)",          "ged_structural.pdf"),
    "kld":            ("δ²_KLD (symmetric KL, bits)",       "kld_symmetric.pdf"),

    # ---- all EMD variants (whichever --emd-mode produced the CSV) ----
    "emd_sid_degree": ("δ²_EMD (exact SID + log-weighted out-degree fallback)",
                       "emd_sid_degree.pdf"),
    "emd_degree":     ("δ²_EMD (degree-based cost)",        "emd_degree.pdf"),
    "emd_signature":  ("δ²_EMD (signature-field cost)",     "emd_signature.pdf"),

    # ---- SID-overlap diagnostics (present only in sid_degree runs that
    #      write them; harmlessly skipped if absent) ----
    "shared_sids":       ("Shared SIDs",        "sid_shared.pdf"),
    "unchanged_payload": ("Unchanged payload",  "sid_unchanged_payload.pdf"),
    "residual_supply":   ("Residual supply",    "sid_residual_supply.pdf"),
    "residual_demand":   ("Residual demand",    "sid_residual_demand.pdf"),

    # ---- eqc_plot.py / unary columns ----
    # The exact header names below are best-guesses. Check your unary
    # metrics.csv header row and correct any that differ; the generic
    # fallback will catch anything not listed here so you can see the name.
    # "avg_eqc_size":   ("Average EQC size",            "avg_eqc_size.pdf"),
    # "num_eqc":        ("Number of EQCs",              "num_eqc.pdf"),
    # "std_eqc":        ("Std. dev. of EQC size",       "std_eqc.pdf"),
    # "tmh_weighted":   ("TMH (weighted)",              "tmh_weighted.pdf"),
    # "tmh_unweighted": ("TMH (unweighted)",            "tmh_unweighted.pdf"),
}

# Dual y-axis plots: (left column, right column) drawn on twin axes.
# Used for the weak-vs-strong connected-components figure. Fill in the
# real column names from your unary CSV header.
DUAL_AXIS_PAIRS: List[Dict[str, str]] = [
    # {
    #     "left_col":    "wcc_payload40",
    #     "right_col":   "scc_payload40",
    #     "left_label":  "Weakly connected components (payload ≥ 40)",
    #     "right_label": "Strongly connected components (payload ≥ 40)",
    #     "filename":    "components_payload_filtered.pdf",
    # },
]

LABEL_COL_CANDIDATES = ["label", "week", "date", "timestep"]


# ----------------------------------------------------------------------
# Plot helpers (styling copied from the original scripts; NO set_title)
# ----------------------------------------------------------------------

def _year_ticks(labels: List[str], max_ticks: int) -> Tuple[List[int], List[str]]:
    if not labels:
        return [], []
    years_seen: List[str] = []
    for lbl in labels:
        y = str(lbl)[:4]
        if not years_seen or years_seen[-1] != y:
            years_seen.append(y)
    if len(years_seen) > max_ticks:
        step = max(1, len(years_seen) // max_ticks)
        years_used = years_seen[::step]
    else:
        years_used = years_seen
    n = len(labels)
    m = len(years_used)
    if m == 0:
        return [], []
    if m == 1:
        positions = [0]
    else:
        positions = [int(round(i * (n - 1) / (m - 1))) for i in range(m)]
    return positions, years_used


def _resolve_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fall back to the first column
    return df.columns[0]


def plot_single(df, label_col, col, ylabel, out_pdf, max_xticks):
    labels = [str(x) for x in df[label_col].tolist()]
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    sns.lineplot(x=list(range(len(df))), y=df[col], ax=ax, linewidth=1.2)
    # NOTE: deliberately no ax.set_title(...) -> no caption on top.
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out_pdf}")


def plot_dual(df, label_col, spec, out_pdf, max_xticks):
    labels = [str(x) for x in df[label_col].tolist()]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    x = list(range(len(df)))
    l1, = ax1.plot(x, df[spec["left_col"]], color="#1f77b4",
                   linewidth=1.2, label=spec["left_label"])
    l2, = ax2.plot(x, df[spec["right_col"]], color="#ff7f0e",
                   linewidth=1.2, linestyle=":", label=spec["right_label"])
    # NOTE: deliberately no title.
    ax1.set_xlabel("Timestep", fontsize=12)
    ax1.set_ylabel(spec["left_label"], fontsize=12, color="#1f77b4")
    ax2.set_ylabel(spec["right_label"], fontsize=12, color="#ff7f0e")
    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(handles=[l1, l2], loc="best")
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out_pdf}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Regenerate plots from an existing metrics.csv, without titles."
    )
    ap.add_argument("-i", "--input", required=True, help="Path to metrics.csv")
    ap.add_argument("--out-dir", required=True, help="Directory for output PDFs")
    ap.add_argument("--label-col", default=None,
                    help="Label/timestep column (auto-detected if omitted)")
    ap.add_argument("--max-xticks", type=int, default=30)
    ap.add_argument("--no-extra", action="store_true",
                    help="Do NOT plot numeric columns that are not in the config.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.out_dir)
    label_col = args.label_col or _resolve_label_col(df)

    print(f"[info] columns found: {list(df.columns)}")
    print(f"[info] using label column: {label_col}")

    plotted_cols = {label_col}

    # 1) dual-axis pairs first (so their columns aren't re-plotted singly)
    for spec in DUAL_AXIS_PAIRS:
        lc, rc = spec["left_col"], spec["right_col"]
        if lc in df.columns and rc in df.columns:
            plot_dual(df, label_col, spec, out_dir / spec["filename"], args.max_xticks)
            plotted_cols.update({lc, rc})
        else:
            missing = [c for c in (lc, rc) if c not in df.columns]
            print(f"[skip] dual plot {spec['filename']}: missing {missing}")

    # 2) configured single metrics
    for col, (ylabel, fname) in SINGLE_METRICS.items():
        if col in df.columns and col not in plotted_cols:
            plot_single(df, label_col, col, ylabel, out_dir / fname, args.max_xticks)
            plotted_cols.add(col)

    # 3) generic fallback for any remaining numeric column
    if not args.no_extra:
        for col in df.columns:
            if col in plotted_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            print(f"[extra] plotting unconfigured column '{col}' generically")
            plot_single(df, label_col, col, col, out_dir / f"{col}.pdf", args.max_xticks)
            plotted_cols.add(col)

    print("[done]")


if __name__ == "__main__":
    main()