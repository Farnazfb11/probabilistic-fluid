from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from save import load_summary_package
from Binary_comp import (   # binary-metric functions in Binary_comp.py
    jaccard_vertices,
    jaccard_edges,
    graph_edit_distance_structural,
    symmetric_kl_divergence,
)

# nice default style, same as EQC plots
sns.set_theme(style="white")


# helpers to find / label snapshots

def _label(p: Path) -> str:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", p.stem)
    return m.group(1) if m else p.stem


def _sort_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", p.stem)
    return (int(f"{m.group(1)}{m.group(2)}{m.group(3)}"), p.stem) if m else (10**10, p.stem)


def _year_ticks(labels: List[str], max_ticks: int) -> Tuple[List[int], List[str]]:
    if not labels:
        return [], []

    years_seen: List[str] = []
    for lbl in labels:
        y = lbl[:4]
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


# collect pairwise (t, t+1) metrics

def collect_pairs(input_dir: Path) -> List[Dict[str, Any]]:
    # Stream through summaries in date order and compute binary metrics
    # only for consecutive pairs S_t, S_{t+1}, keeping at most two
    # packages in memory at any time.
    files = sorted(input_dir.rglob("*.json"), key=_sort_key)
    if len(files) < 2:
        return []

    rows: List[Dict[str, Any]] = []

    prev_pkg: Optional[Dict[str, Any]] = None
    prev_label: Optional[str] = None

    for fp in files:
        try:
            pkg = load_summary_package(fp)
            label = _label(fp)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
            continue

        if prev_pkg is not None:
            # S_t = prev_pkg, S_{t+1} = pkg
            try:
                jac_v = float(jaccard_vertices(prev_pkg, pkg))
                jac_e = float(jaccard_edges(prev_pkg, pkg))
                ged   = float(graph_edit_distance_structural(prev_pkg, pkg))
                kld   = float(symmetric_kl_divergence(prev_pkg, pkg))
            except Exception as e:
                print(f"[warn] metrics failed for pair {prev_label}→{label}: {e}")
                # still advance window
            else:
                rows.append({
                    "label": label,   # change leading to this snapshot
                    "jac_v": jac_v,
                    "jac_e": jac_e,
                    "ged":   ged,
                    "kld":   kld,
                })

        # slide window
        prev_pkg = pkg
        prev_label = label

    return rows


def maybe_write_csv(rows: List[Dict[str, Any]], csv_path: Optional[Path]):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["label", "jac_v", "jac_e", "ged", "kld"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] CSV: {csv_path}")


# generic “one-curve” plot helper

def _plot_metric(
    rows: List[Dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
    out_pdf: Path,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    values = [r[key] for r in rows]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.lineplot(
        x=labels,
        y=values,
        ax=ax,
        linewidth=1.0,
        label=ylabel,
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.legend()

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")

# main CLI 

def main():
    ap = argparse.ArgumentParser(
        description="Binary metrics over consecutive summaries: Jaccard (V/E), GED, KLD."
    )
    ap.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Directory with summary JSONs (recursive).",
    )
    ap.add_argument(
        "--out-dir",
        default="results_binary",
        help="Output directory for PDFs (default: results_binary).",
    )

    # output filenames
    ap.add_argument(
        "--jac-v-pdf",
        default="jaccard_vertices.pdf",
        help="PDF filename for Jaccard over vertices.",
    )
    ap.add_argument(
        "--jac-e-pdf",
        default="jaccard_edges.pdf",
        help="PDF filename for Jaccard over edges.",
    )
    ap.add_argument(
        "--ged-pdf",
        default="ged_structural.pdf",
        help="PDF filename for structural GED approximation.",
    )
    ap.add_argument(
        "--kld-pdf",
        default="kld_symmetric.pdf",
        help="PDF filename for symmetric KL divergence.",
    )

    # titles
    ap.add_argument(
        "--title-jac-v",
        default="Jaccard over entity vertices (δ²_JacV)",
        help="Title for vertex Jaccard plot.",
    )
    ap.add_argument(
        "--title-jac-e",
        default="Jaccard over superedges (δ²_JacE)",
        help="Title for edge Jaccard plot.",
    )
    ap.add_argument(
        "--title-ged",
        default="Structural graph edit distance (δ²_GED)",
        help="Title for GED-like plot.",
    )
    ap.add_argument(
        "--title-kld",
        default="Symmetric KL divergence over EQC-size distributions (δ²_KLD)",
        help="Title for KLD plot.",
    )

    ap.add_argument(
        "--max-xticks",
        type=int,
        default=30,
        help="Max number of x-axis tick labels (years) to show.",
    )

    ap.add_argument(
        "--csv",
        help="(Optional) write a CSV with label, jac_v, jac_e, ged, kld.",
    )

    # optional skip flags
    ap.add_argument("--no-jac-v", action="store_true", help="Skip Jaccard-vertices PDF.")
    ap.add_argument("--no-jac-e", action="store_true", help="Skip Jaccard-edges PDF.")
    ap.add_argument("--no-ged",   action="store_true", help="Skip GED PDF.")
    ap.add_argument("--no-kld",   action="store_true", help="Skip KLD PDF.")

    args = ap.parse_args()

    rows = collect_pairs(Path(args.input_dir))
    if not rows:
        print("[info] Not enough summaries or all pairs failed to compute.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_write_csv(rows, Path(args.csv) if args.csv else None)

    if not args.no_jac_v:
        _plot_metric(
            rows,
            key="jac_v",
            ylabel="δ²_JacV (vertex Jaccard distance)",
            title=args.title_jac_v,
            out_pdf=out_dir / args.jac_v_pdf,
            max_xticks=args.max_xticks,
        )

    if not args.no_jac_e:
        _plot_metric(
            rows,
            key="jac_e",
            ylabel="δ²_JacE (edge Jaccard distance)",
            title=args.title_jac_e,
            out_pdf=out_dir / args.jac_e_pdf,
            max_xticks=args.max_xticks,
        )

    if not args.no_ged:
        _plot_metric(
            rows,
            key="ged",
            ylabel="δ²_GED (edit operations)",
            title=args.title_ged,
            out_pdf=out_dir / args.ged_pdf,
            max_xticks=args.max_xticks,
        )

    if not args.no_kld:
        _plot_metric(
            rows,
            key="kld",
            ylabel="δ²_KLD (symmetric KL)",
            title=args.title_kld,
            out_pdf=out_dir / args.kld_pdf,
            max_xticks=args.max_xticks,
        )


if __name__ == "__main__":
    main()
