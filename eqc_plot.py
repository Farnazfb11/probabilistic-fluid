from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
from save import load_summary_package
from Unary_comp import average_eqc_size, std_eqc_size
from Unary_comp import tmh_tendency_to_make_hubs

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
                "avg": float(average_eqc_size(pkg)),
                "std": float(std_eqc_size(pkg, population=False)),
            })
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
    return rows

def maybe_write_csv(rows: List[Dict[str, Any]], csv_path: Optional[Path]):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        # CHANGED: add 'tmh' column
        # WHY: keep CSV (if requested) consistent with what we plot
        w = csv.DictWriter(f, fieldnames=["label", "file", "avg", "std", "tmh"])
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

def plot_pdf(rows: List[Dict[str, Any]], out_pdf: Path, title: str, max_xticks: int, tmh_on_secondary_axis: bool = True):
    if not rows:
        print("[info] nothing to plot")
        return

    labels = [r["label"] for r in rows]
    avgs   = [r["avg"]   for r in rows]
    stds   = [r["std"]   for r in rows]
    tmhs   = [r["tmh"]   for r in rows]  # CHANGED: series for TMH
    # WHY: we’ll plot TMH alongside avg/std

    plt.figure(figsize=(12, 6))  # single figure, clean

    # Primary axis: EQC size stats
    ax1 = plt.gca()
    l1, = ax1.plot(labels, avgs, marker="o", label="Average EQC size")
    l2, = ax1.plot(labels, stds, marker="s", label="Std. dev. EQC size")
    ax1.set_xlabel("Snapshot / Summary")
    ax1.set_ylabel("EQC size (count)")
    ax1.set_title(title)

    # CHANGED: plot TMH, optionally on a secondary y-axis
    # WHY: TMH may have a different scale than counts; secondary axis keeps both readable.
    if tmh_on_secondary_axis:
        ax2 = ax1.twinx()
        l3, = ax2.plot(labels, tmhs, marker="^", linestyle="--", label="TMH (hub tendency)")
        ax2.set_ylabel("TMH")
        lines = [l1, l2, l3]
        labels_for_legend = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels_for_legend, loc="best")
    else:
        l3, = ax1.plot(labels, tmhs, marker="^", linestyle="--", label="TMH (hub tendency)")
        ax1.legend(loc="best")

    # Thin x-ticks if there are too many labels (prevents overlap in PDF)
    n = len(labels)
    keep = _thin_indices(n, max_xticks)
    plt.xticks(keep, [labels[i] for i in keep], rotation=60, ha="right")

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")  # vector PDF
    print(f"[ok] PDF plot: {out_pdf}")

def main():
    ap = argparse.ArgumentParser(description="Plot avg & std of EQC sizes plus TMH to PDF.")
    ap.add_argument("-i", "--input-dir", required=True, help="Directory with summary JSONs (recursive).")
    ap.add_argument("--out-pdf", default="results/eqc_stats.pdf", help="Output PDF path (vector graphics).")
    ap.add_argument("--title", default="EQC Size & TMH", help="Plot title.")
    ap.add_argument("--csv", help="(Optional) also write a CSV with the values.")
    ap.add_argument("--max-xticks", type=int, default=40, help="Max number of x-axis tick labels (default: 40).")
    ap.add_argument("--tmh-unweighted", action="store_true",
                    help="Use unweighted degrees for TMH (default: weighted).")  # CHANGED: CLI switch
    # WHY: sometimes you might want structural TMH ignoring rolled edge weights
    ap.add_argument("--tmh-secondary-axis", action="store_true",
                    help="Plot TMH on a secondary y-axis (default). Use --no-tmh-secondary-axis to disable.")  # CHANGED: hint
    ap.add_argument("--no-tmh-secondary-axis", dest="tmh_secondary_axis", action="store_false")  # CHANGED: negation flag
    ap.set_defaults(tmh_secondary_axis=True)  # CHANGED: default True

    args = ap.parse_args()

    rows = collect(Path(args.input_dir), tmh_weighted=not args.tmh_unweighted)  # CHANGED: pass weighted flag
    if not rows:
        print("[info] No summaries found or all failed to parse.")
        return

    maybe_write_csv(rows, Path(args.csv) if args.csv else None)
    plot_pdf(rows, Path(args.out_pdf), args.title, args.max_xticks, tmh_on_secondary_axis=args.tmh_secondary_axis)

if __name__ == "__main__":
    main()