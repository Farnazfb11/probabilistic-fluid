from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import time

from save import load_summary_package
from Binary_measure import (
    jaccard_vertices,
    jaccard_edges,
    graph_edit_distance_structural,
    symmetric_kl_divergence,
    earth_mover_distance_degree,  # ADDED: Import EMD
    _degree_map_weighted_from_pkg
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

def collect_pairs(input_dir: Path, *, compute_emd: bool = True) -> List[Dict[str, Any]]:
    files = sorted(input_dir.rglob("*.json"), key=_sort_key)
    """
    Stream through summaries in date order and compute binary metrics
    only for consecutive pairs S_t, S_{t+1}, keeping at most two
    packages in memory at any time.
    """
    rows: List[Dict[str, Any]] = []
    prev_pkg: Optional[Dict[str, Any]] = None
    prev_label: Optional[str] = None

    # CHANGED: cache degree maps so we do not recompute per pair
    # WHY: you want degree based EMD without reconstructing graphs repeatedly
    prev_deg: Optional[Dict[int, int]] = None

    # CHANGED: counter for consecutive pairs
    pair_idx = 0

    for fp in files:
        label = _label(fp)

        try:
            pkg = load_summary_package(fp)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
            continue

        # CHANGED: compute degree map once for this week, only if EMD enabled
        curr_deg: Optional[Dict[int, int]] = None
        if compute_emd:
            try:
                curr_deg = _degree_map_weighted_from_pkg(pkg)
            except Exception as e:
                curr_deg = None
                print(f"[warn] degree map failed for {label}: {e}")

        if prev_pkg is not None:
            pair_idx += 1
            try:
                row = {
                    "label": label,
                    "jac_v": float(jaccard_vertices(prev_pkg, pkg)),
                    "jac_e": float(jaccard_edges(prev_pkg, pkg)),
                    "ged":   float(graph_edit_distance_structural(prev_pkg, pkg)),
                    "kld":   float(symmetric_kl_divergence(prev_pkg, pkg)),
                }

                if compute_emd:
                    # CHANGED: pass cached degrees
                    # WHY: avoid rebuilding or recomputing anything per pair
                    print(f"[emd] START pair {pair_idx}: {prev_label} -> {label}", flush=True)
                    t0 = time.perf_counter()
                    row["emd"] = float(earth_mover_distance_degree(prev_pkg, pkg, deg_t=prev_deg, deg_tp1=curr_deg))
                    dt = time.perf_counter() - t0
                    print(f"[emd] DONE  pair {pair_idx}: {prev_label} -> {label} | emd={row['emd']} | {dt:.1f}s", flush=True)

                rows.append(row)

            except Exception as e:
                print(f"[warn] metrics failed for pair {prev_label}→{label}: {e}")

        prev_pkg = pkg
        prev_label = label
        prev_deg = curr_deg  # CHANGED: carry forward cached degree map

    return rows




def maybe_write_csv(rows: List[Dict[str, Any]], csv_path: Optional[Path]):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["label", "jac_v", "jac_e", "ged", "kld"]
    if rows and "emd" in rows[0]:
        fieldnames.append("emd")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] CSV: {csv_path}")


# generic "one-curve" plot helper

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

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    sns.lineplot(
        x=list(range(len(labels))),
        y=values,
        ax=ax,
        linewidth=1.2,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-axis ticks to show years
    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")

# main CLI 

def main():
    ap = argparse.ArgumentParser(
        description="Binary metrics over consecutive summaries: Jaccard (V/E), GED, KLD, EMD."
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
    ap.add_argument(
        "--emd-pdf",
        default="emd_payload.pdf",
        help="PDF filename for Earth Mover's Distance.",
    )

    # titles
    ap.add_argument(
        "--title-jac-v",
        default="Jaccard Distance over Entity Vertices (δ²_JacV)",
        help="Title for vertex Jaccard plot.",
    )
    ap.add_argument(
        "--title-jac-e",
        default="Jaccard Distance over Superedges (δ²_JacE)",
        help="Title for edge Jaccard plot.",
    )
    ap.add_argument(
        "--title-ged",
        default="Structural Graph Edit Distance (δ²_GED)",
        help="Title for GED-like plot.",
    )
    ap.add_argument(
        "--title-kld",
        default="Symmetric KL Divergence over EQC Distributions (δ²_KLD)",
        help="Title for KLD plot.",
    )
    ap.add_argument(
        "--title-emd",
        default="Earth Mover's Distance over EQC Payloads (δ²_EMD)",
        help="Title for EMD plot.",
    )

    ap.add_argument(
        "--max-xticks",
        type=int,
        default=30,
        help="Max number of x-axis tick labels (years) to show.",
    )

    ap.add_argument(
        "--csv",
        help="(Optional) write a CSV with label, jac_v, jac_e, ged, kld, emd.",
    )

    # optional skip flags
    ap.add_argument("--no-jac-v", action="store_true", help="Skip Jaccard-vertices PDF.")
    ap.add_argument("--no-jac-e", action="store_true", help="Skip Jaccard-edges PDF.")
    ap.add_argument("--no-ged",   action="store_true", help="Skip GED PDF.")
    ap.add_argument("--no-kld",   action="store_true", help="Skip KLD PDF.")
    ap.add_argument("--no-emd",   action="store_true", help="Skip EMD PDF.")

    args = ap.parse_args()

    rows = collect_pairs(Path(args.input_dir), compute_emd=(not args.no_emd))
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
            ylabel="δ²_KLD (symmetric KL, bits)",
            title=args.title_kld,
            out_pdf=out_dir / args.kld_pdf,
            max_xticks=args.max_xticks,
        )

    if not args.no_emd:
        _plot_metric(
            rows,
            key="emd",
            ylabel="δ²_EMD (entity movement cost)",
            title=args.title_emd,
            out_pdf=out_dir / args.emd_pdf,
            max_xticks=args.max_xticks,
        )

    print(f"\n[done] All metrics computed for {len(rows)} consecutive pairs.")


if __name__ == "__main__":
    main()