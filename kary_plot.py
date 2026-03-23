from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# CHANGED: same plotting style as your existing binary plotting code
# WHY: keeps the look of the figures consistent across the paper
sns.set_theme(style="white")


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
        positions = [
            int(round(i * (n - 1) / (m - 1)))
            for i in range(m)
        ]

    return positions, years_used


def _read_table(path: Path) -> pd.DataFrame:
    """
    CHANGED: supports both Excel and CSV
    WHY: your scores may already be stored in either format
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def add_kary_columns(
    df: pd.DataFrame,
    *,
    score_col: str,
    k_soft: int = 4,
    k_trend: int = 12,
) -> pd.DataFrame:
    """
    CHANGED: derive k-ary columns from existing binary scores
    WHY: your Excel already has pairwise Jaccard scores, so no need to recompute from JSONs

    For k snapshots, we aggregate over k-1 consecutive binary distances.
    """
    out = df.copy()

    soft_window = k_soft - 1
    trend_window = k_trend - 1

    if soft_window < 1 or trend_window < 1:
        raise ValueError("k values must be at least 2")

    # CHANGED: rolling mean for softening
    # WHY: k=4 means average over 3 consecutive binary scores
    out[f"{score_col}_k{k_soft}_soft"] = (
        out[score_col].rolling(window=soft_window, min_periods=soft_window).mean()
    )

    # CHANGED: rolling mean for long-window trend level
    # WHY: k=12 means average over 11 consecutive binary scores
    out[f"{score_col}_k{k_trend}_trend"] = (
        out[score_col].rolling(window=trend_window, min_periods=trend_window).mean()
    )

    return out


def _plot_kary_metric(
    df: pd.DataFrame,
    *,
    label_col: str,
    raw_col: str,
    soft_col: str,
    trend_col: str,
    title: str,
    ylabel: str,
    out_pdf: Path,
    max_xticks: int,
):
    labels = [str(x) for x in df[label_col].tolist()]

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    # CHANGED: plot raw binary metric and both k-ary derived series
    # WHY: this lets you compare short-term noise against smoothed and long-window behavior
    sns.lineplot(
        x=list(range(len(df))),
        y=df[raw_col],
        ax=ax,
        linewidth=1.2,
        label="Binary",
        color = "#ffc508",
    )

    sns.lineplot(
        x=list(range(len(df))),
        y=df[soft_col],
        ax=ax,
        linewidth=1.4,
        label="k=4 softening",
        color = "#37c71a",
    )

    sns.lineplot(
        x=list(range(len(df))),
        y=df[trend_col],
        ax=ax,
        linewidth=1.4,
        label="k=12 trend",
        color = "#091be0",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")


def _write_table(df: pd.DataFrame, out_path: Path):
    """
    CHANGED: save enriched table back to Excel or CSV
    WHY: you will likely want both the plot and the new k-ary values for later analysis
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {out_path.suffix}")

    print(f"[ok] Table: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Create k-ary softening/trend curves from an existing Excel/CSV table of binary scores."
    )
    ap.add_argument(
        "-i", "--input",
        required=True,
        help="Excel or CSV file containing at least a label column and a binary score column."
    )
    ap.add_argument(
        "--label-col",
        default="label",
        help="Column containing timestep labels, e.g. dates."
    )
    ap.add_argument(
        "--score-col",
        default="jac_v",
        help="Column containing the binary score to aggregate, e.g. jac_v."
    )
    ap.add_argument(
        "--k-soft",
        type=int,
        default=4,
        help="Window length k for softening. Uses k-1 binary scores."
    )
    ap.add_argument(
        "--k-trend",
        type=int,
        default=12,
        help="Window length k for trend. Uses k-1 binary scores."
    )
    ap.add_argument(
        "--out-table",
        default="results_kary/jaccard_with_kary.xlsx",
        help="Output Excel/CSV file with added k-ary columns."
    )
    ap.add_argument(
        "--out-pdf",
        default="results_kary/jaccard_kary.pdf",
        help="Output PDF plot."
    )
    ap.add_argument(
        "--title",
        default="Binary and k-ary Jaccard Distance over Entity Vertices",
        help="Plot title."
    )
    ap.add_argument(
        "--ylabel",
        default="δ_JacV",
        help="Y-axis label."
    )
    ap.add_argument(
        "--max-xticks",
        type=int,
        default=30,
        help="Maximum number of year tick labels."
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    out_table = Path(args.out_table)
    out_pdf = Path(args.out_pdf)

    df = _read_table(in_path)

    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    if args.score_col not in df.columns:
        raise ValueError(f"Missing score column: {args.score_col}")

    df2 = add_kary_columns(
        df,
        score_col=args.score_col,
        k_soft=args.k_soft,
        k_trend=args.k_trend,
    )

    soft_col = f"{args.score_col}_k{args.k_soft}_soft"
    trend_col = f"{args.score_col}_k{args.k_trend}_trend"

    _write_table(df2, out_table)

    _plot_kary_metric(
        df2,
        label_col=args.label_col,
        raw_col=args.score_col,
        soft_col=soft_col,
        trend_col=trend_col,
        title=args.title,
        ylabel=args.ylabel,
        out_pdf=out_pdf,
        max_xticks=args.max_xticks,
    )

    print("[done] k-ary table and plot created.")


if __name__ == "__main__":
    main()