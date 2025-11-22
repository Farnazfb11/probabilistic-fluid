from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
from save import load_summary_package
# CHANGED: also import count_equivalence_classes
# WHY: we need the number of EQCs for the orange curve (right y-axis).
from Unary_comp import (
    average_eqc_size,
    std_eqc_size,
    tmh_tendency_to_make_hubs,
    count_equivalence_classes,        # CHANGED
)

def _label(p: Path) -> str:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", p.stem)
    return m.group(1) if m else p.stem

def _sort_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", p.stem)
    return (int(f"{m.group(1)}{m.group(2)}{m.group(3)}"), p.stem) if m else (10**10, p.stem)

def collect(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(input_dir.rglob("*.json"), key=_sort_key):
        try:
            pkg = load_summary_package(fp)
            rows.append({
                "file": str(fp),
                "label": _label(fp),
                # CHANGED: add eqc count for second y-axis
                # WHY: Figure 10 plots avg EQC size vs. number of EQCs.
                "eqc": int(count_equivalence_classes(pkg)),          # CHANGED
                "avg": float(average_eqc_size(pkg)),
                "std": float(std_eqc_size(pkg, population=False)),
                "tmh_w": float(tmh_tendency_to_make_hubs(pkg, weighted=True)),
                "tmh_u": float(tmh_tendency_to_make_hubs(pkg, weighted=False)),
            })
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
    return rows

def maybe_write_csv(rows: List[Dict[str, Any]], csv_path: Optional[Path]):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        # CHANGED: keep CSV consistent with collected fields (eqc, avg, std, tmh_w, tmh_u)
        # WHY: previous code wrote a non-existent 'tmh' column.
        w = csv.DictWriter(
            f,
            fieldnames=["label", "file", "eqc", "avg", "std", "tmh_w", "tmh_u"],  # CHANGED
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] CSV: {csv_path}")

def _thin_indices(n: int, max_ticks: int) -> List[int]:
    if n <= max_ticks:
        return list(range(n))
    step = max(1, n // max_ticks)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx

def _year_ticks(labels: List[str], max_ticks: int) -> Tuple[List[int], List[str]]:
    # Convert date labels like '2012-05-06' into year-based ticks.
    # Only show one tick per year.
    years = [lbl[:4] for lbl in labels]  # extract year
    keep_indices = []
    keep_labels  = []

    last_year = None
    for i, y in enumerate(years):
        if y != last_year:
            keep_indices.append(i)
            keep_labels.append(y)
            last_year = y

    # optionally thin further if too many years
    if len(keep_indices) > max_ticks:
        step = max(1, len(keep_indices) // max_ticks)
        keep_indices = keep_indices[::step]
        keep_labels  = keep_labels[::step]

    return keep_indices, keep_labels

# CHANGED: new plot function to mimic Figure 10
#          left y-axis: average EQC size (blue)
#          right y-axis: number of EQCs (orange, dotted)
def plot_avg_and_eqc_count(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    avgs   = [r["avg"]   for r in rows]
    eqcs   = [r["eqc"]   for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # left axis: average EQC size (blue)
    ax1.plot(labels, avgs, color="blue", linewidth=1.0, label="Average size of EQC")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Average size of EQC", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # right axis: number of EQCs (orange, dotted)
    ax2 = ax1.twinx()
    ax2.plot(labels, eqcs, color="orange", linestyle=":", linewidth=1.0, label="Number of EQC")
    ax2.set_ylabel("Number of EQC", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.suptitle(title)

    # thin x-axis labels
    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=0)
    
    # build combined legend (blue + orange) in one place
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

# CHANGED: new function — standard deviation on its own plot
# WHY: the number was too big it dawrfed the avg vs eqc count plot
def plot_std_only(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    stds   = [r["std"]   for r in rows]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(labels, stds, color="blue", linewidth=1.0, label="Std. dev. of EQC size")
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Std. dev. of EQC size")
    ax.legend()

    xticks, xlabels = _year_ticks(labels, max_xticks)
    plt.xticks(xticks, xlabels)
    
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")

def plot_tmh(rows: List[Dict[str, Any]], out_pdf: Path, title: str, key: str, max_xticks: int):
    """key is 'tmh_w' or 'tmh_u'."""
    labels = [r["label"] for r in rows]
    tmh    = [r[key]      for r in rows]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(labels, tmh, marker="^", linestyle="--", label="TMH (tendency to make hubs)")
    ax.set_title(title)
    ax.set_xlabel("Snapshot / Summary")
    ax.set_ylabel("TMH")
    ax.legend()

    xticks, xlabels = _year_ticks(labels, max_xticks)
    plt.xticks(xticks, xlabels)
    
    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")


def main():
    # CHANGED: argument text updated to mention the new plots
    ap = argparse.ArgumentParser(
        description="Create PDFs: avg+EQC count, SD of EQC size, TMH weighted, TMH unweighted."
    )
    ap.add_argument("-i", "--input-dir", required=True, help="Directory with summary JSONs (recursive).")
    ap.add_argument("--out-dir", default="results", help="Output directory for PDFs (default: results).")

    # CHANGED: eqc-pdf now means combined avg+EQC-count plot
    ap.add_argument(
        "--avg-eqc-pdf",
        default="avg_and_eqc_count.pdf",
        help="Filename for combined plot: average EQC size vs number of EQCs.",
    )
    # CHANGED: new argument for SD-only plot
    ap.add_argument(
        "--std-eqc-pdf",
        default="eqc_std.pdf",
        help="Filename for standard deviation of EQC sizes PDF.",
    )

    ap.add_argument("--tmh-weighted-pdf", default="tmh_weighted.pdf", help="Filename for weighted TMH PDF.")
    ap.add_argument("--tmh-unweighted-pdf", default="tmh_unweighted.pdf", help="Filename for unweighted TMH PDF.")

    # CHANGED: titles slightly tweaked
    ap.add_argument(
        "--title-avg-eqc",
        default="Average size of EQC vs Number of EQC",
        help="Title for avg+EQC-count plot.",
    )
    ap.add_argument(
        "--title-std-eqc",
        default="Std. deviation of EQC size",
        help="Title for SD-only plot.",
    )
    ap.add_argument("--title-tmh-w", default="TMH (Weighted Degrees)", help="Title for weighted TMH plot.")
    ap.add_argument("--title-tmh-u", default="TMH (Unweighted Degrees)", help="Title for unweighted TMH plot.")
    ap.add_argument("--max-xticks", type=int, default=40, help="Max x-axis tick labels (default: 40).")
    ap.add_argument("--csv", help="(Optional) write a CSV with eqc, avg, std, tmh_w, tmh_u.")
    # optional switches to skip any of the PDFs
    ap.add_argument("--no-avg-eqc", action="store_true", help="Skip avg+EQC-count PDF.")   # CHANGED
    ap.add_argument("--no-std-eqc", action="store_true", help="Skip SD-only EQC PDF.")     # CHANGED
    ap.add_argument("--no-tmh-weighted", action="store_true", help="Skip weighted TMH PDF.")
    ap.add_argument("--no-tmh-unweighted", action="store_true", help="Skip unweighted TMH PDF.")
    args = ap.parse_args()

    rows = collect(Path(args.input_dir))
    if not rows:
        print("[info] No summaries found or all failed to parse.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_write_csv(rows, Path(args.csv) if args.csv else None)

    if not args.no-avg_eqc if False else False:
        pass  # dummy to avoid syntax highlight confusion
    # proper conditionals:
    if not args.no_avg_eqc:  # CHANGED
        plot_avg_and_eqc_count(
            rows,
            out_dir / args.avg_eqc_pdf,
            args.title_avg_eqc,
            args.max_xticks,
        )
    if not args.no_std_eqc:  # CHANGED
        plot_std_only(
            rows,
            out_dir / args.std_eqc_pdf,
            args.title_std_eqc,
            args.max_xticks,
        )
    if not args.no_tmh_weighted:
        plot_tmh(rows, out_dir / args.tmh_weighted_pdf, args.title_tmh_w, key="tmh_w", max_xticks=args.max_xticks)
    if not args.no_tmh_unweighted:
        plot_tmh(rows, out_dir / args.tmh_unweighted_pdf, args.title_tmh_u, key="tmh_u", max_xticks=args.max_xticks)

if __name__ == "__main__":
    main()
