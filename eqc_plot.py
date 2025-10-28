from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
from save import load_summary_package
from Unary_comp import average_eqc_size, std_eqc_size

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
        w = csv.DictWriter(f, fieldnames=["label", "file", "avg", "std"])
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

def plot_pdf(rows: List[Dict[str, Any]], out_pdf: Path, title: str, max_xticks: int):
    if not rows:
        print("[info] nothing to plot")
        return

    labels = [r["label"] for r in rows]
    avgs   = [r["avg"]   for r in rows]
    stds   = [r["std"]   for r in rows]

    plt.figure(figsize=(12, 6))  # single figure, clean
    plt.plot(labels, avgs, marker="o", label="Average EQC size")
    plt.plot(labels, stds, marker="s", label="Std. dev. EQC size")
    plt.title(title)
    plt.xlabel("Snapshot / Summary")
    plt.ylabel("Count")
    plt.legend()

    # Thin x-ticks if there are too many labels (prevents overlap in PDF)
    n = len(labels)
    keep = _thin_indices(n, max_xticks)
    plt.xticks(keep, [labels[i] for i in keep], rotation=60, ha="right")

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")  # vector PDF
    print(f"[ok] PDF plot: {out_pdf}")