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

def plot_eqc_sizes(rows: List[Dict[str, Any]], out_pdf: Path, title: str, max_xticks: int):
    labels = [r["label"] for r in rows]
    avgs   = [r["avg"]   for r in rows]
    stds   = [r["std"]   for r in rows]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(labels, avgs, marker="o", label="Average EQC size")
    ax.plot(labels, stds, marker="s", label="Std. dev. EQC size")
    ax.set_title(title)
    ax.set_xlabel("Snapshot / Summary")
    ax.set_ylabel("EQC size (count)")
    ax.legend()

    keep = _thin_indices(len(labels), max_xticks)
    plt.xticks(keep, [labels[i] for i in keep], rotation=60, ha="right")

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

    keep = _thin_indices(len(labels), max_xticks)
    plt.xticks(keep, [labels[i] for i in keep], rotation=60, ha="right")

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description="Create three PDFs: EQC sizes, TMH weighted, TMH unweighted.")
    ap.add_argument("-i", "--input-dir", required=True, help="Directory with summary JSONs (recursive).")
    ap.add_argument("--out-dir", default="results", help="Output directory for PDFs (default: results).")
    ap.add_argument("--eqc-pdf", default="eqc_sizes.pdf", help="Filename for EQC sizes PDF.")
    ap.add_argument("--tmh-weighted-pdf", default="tmh_weighted.pdf", help="Filename for weighted TMH PDF.")
    ap.add_argument("--tmh-unweighted-pdf", default="tmh_unweighted.pdf", help="Filename for unweighted TMH PDF.")
    ap.add_argument("--title-eqc", default="EQC Size: Average & Standard Deviation", help="Title for EQC sizes plot.")
    ap.add_argument("--title-tmh-w", default="TMH (Weighted Degrees)", help="Title for weighted TMH plot.")
    ap.add_argument("--title-tmh-u", default="TMH (Unweighted Degrees)", help="Title for unweighted TMH plot.")
    ap.add_argument("--max-xticks", type=int, default=40, help="Max x-axis tick labels (default: 40).")
    ap.add_argument("--csv", help="(Optional) write a CSV with avg, std, tmh_w, tmh_u.")
    # optional switches to skip any of the PDFs
    ap.add_argument("--no-eqc", action="store_true", help="Skip EQC sizes PDF.")
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

    if not args.no_eqc:
        plot_eqc_sizes(rows, out_dir / args.eqc_pdf, args.title_eqc, args.max_xticks)
    if not args.no_tmh_weighted:
        plot_tmh(rows, out_dir / args.tmh_weighted_pdf, args.title_tmh_w, key="tmh_w", max_xticks=args.max_xticks)
    if not args.no_tmh_unweighted:
        plot_tmh(rows, out_dir / args.tmh_unweighted_pdf, args.title_tmh_u, key="tmh_u", max_xticks=args.max_xticks)

if __name__ == "__main__":
    main()