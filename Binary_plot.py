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
    earth_mover_distance_degree,
    earth_mover_distance_signature,
    earth_mover_distance_sid_then_degree,   # CHANGED
    sid_overlap_stats,                      # CHANGED
    _degree_map_weighted_from_pkg,
)


# nice default style, same as EQC plots
sns.set_theme(style="white")


# helpers to find / label snapshots

def _sid_degree_label(degree_mode: str) -> str:
    mapping = {
        "log_weighted_out": "Degree-based EMD",
        "log_weighted_total": "Degree-based EMD",
        "weighted_out": "Degree-based EMD",
        "weighted_total": "Degree-based EMD",
        "unweighted_out": "Degree-based EMD",
    }
    return mapping.get(degree_mode, "Degree-based EMD")

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

def collect_pairs(
    input_dir: Path,
    *,
    compute_jac_v: bool = True,
    compute_jac_e: bool = True,
    compute_ged: bool = True,
    compute_kld: bool = True,
    emd_mode: str = "degree",   # "degree", "signature", "both", "sid_degree", "none"
    degree_mode: str = "log_weighted_out",
) -> List[Dict[str, Any]]:
    files = sorted(input_dir.rglob("*.json"), key=_sort_key)

    rows: List[Dict[str, Any]] = []
    prev_pkg: Optional[Dict[str, Any]] = None
    prev_label: Optional[str] = None
    prev_deg: Optional[Dict[int, float]] = None
    pair_idx = 0

    for fp in files:
        label = _label(fp)

        try:
            pkg = load_summary_package(fp)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}")
            continue

        curr_deg: Optional[Dict[int, float]] = None
        if emd_mode in ("degree", "both", "sid_degree"):
            try:
                                # ---------------- CHANGED ----------------
                # WHY CHANGED:
                # Degree proxy is now configurable from the CLI.
                curr_deg = _degree_map_weighted_from_pkg(pkg, mode=degree_mode)
            except Exception as e:
                curr_deg = None
                print(f"[warn] degree map failed for {label}: {e}")

        if prev_pkg is not None:
            pair_idx += 1
            try:
                row = {
                    "label": label,
                }

                if compute_jac_v:
                    row["jac_v"] = float(jaccard_vertices(prev_pkg, pkg))

                if compute_jac_e:
                    row["jac_e"] = float(jaccard_edges(prev_pkg, pkg))

                if compute_ged:
                    row["ged"] = float(graph_edit_distance_structural(prev_pkg, pkg))

                if compute_kld:
                    row["kld"] = float(symmetric_kl_divergence(prev_pkg, pkg))

                if emd_mode == "sid_degree":
                    stats = sid_overlap_stats(prev_pkg, pkg)

                    print(
                        f"[sid-check] pair {pair_idx}: {prev_label} -> {label} "
                        f"| shared_sids={stats['shared_sids']} "
                        f"| unchanged_payload={stats['unchanged_payload']} "
                        f"| residual_supply={stats['residual_supply']} "
                        f"| residual_demand={stats['residual_demand']}",
                        flush=True
                    )

                    print(
                        f"[emd-sid-degree] START pair {pair_idx}: {prev_label} -> {label} "
                        f"| degree_mode={degree_mode}",
                        flush=True
                    )
                    t0 = time.perf_counter()

                    row["emd_sid_degree"] = float(
                        earth_mover_distance_sid_then_degree(
                            prev_pkg, pkg,
                            deg_t=prev_deg,
                            deg_tp1=curr_deg
                        )
                    )

                    dt = time.perf_counter() - t0

                    row["shared_sids"] = stats["shared_sids"]
                    row["unchanged_payload"] = stats["unchanged_payload"]
                    row["residual_supply"] = stats["residual_supply"]
                    row["residual_demand"] = stats["residual_demand"]

                    print(
                        f"[emd-sid-degree] DONE  pair {pair_idx}: {prev_label} -> {label} "
                        f"| emd_sid_degree={row['emd_sid_degree']} | {dt:.1f}s",
                        flush=True
                    )

                if emd_mode in ("degree", "both"):
                    print(
                        f"[emd-degree] START pair {pair_idx}: {prev_label} -> {label} "
                        f"| degree_mode={degree_mode}",
                        flush=True
                    )
                    t0 = time.perf_counter()
                    row["emd_degree"] = float(
                        earth_mover_distance_degree(
                            prev_pkg, pkg,
                            deg_t=prev_deg,
                            deg_tp1=curr_deg
                        )
                    )
                    dt = time.perf_counter() - t0
                    print(
                        f"[emd-degree] DONE  pair {pair_idx}: {prev_label} -> {label} "
                        f"| emd_degree={row['emd_degree']} | {dt:.1f}s",
                        flush=True
                    )

                if emd_mode in ("signature", "both"):
                    print(f"[emd-signature] START pair {pair_idx}: {prev_label} -> {label}", flush=True)
                    t0 = time.perf_counter()
                    row["emd_signature"] = float(
                        earth_mover_distance_signature(prev_pkg, pkg)
                    )
                    dt = time.perf_counter() - t0
                    print(
                        f"[emd-signature] DONE  pair {pair_idx}: {prev_label} -> {label} "
                        f"| emd_signature={row['emd_signature']} | {dt:.1f}s",
                        flush=True
                    )

                rows.append(row)

            except Exception as e:
                print(f"[warn] metrics failed for pair {prev_label}→{label}: {e}")

        prev_pkg = pkg
        prev_label = label
        prev_deg = curr_deg

    return rows




def maybe_write_csv(rows: List[Dict[str, Any]], csv_path: Optional[Path]):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["label"]

    if any("jac_v" in r for r in rows):
        fieldnames.append("jac_v")
    if any("jac_e" in r for r in rows):
        fieldnames.append("jac_e")
    if any("ged" in r for r in rows):
        fieldnames.append("ged")
    if any("kld" in r for r in rows):
        fieldnames.append("kld")
    if any("emd_degree" in r for r in rows):
        fieldnames.append("emd_degree")
    if any("emd_signature" in r for r in rows):
        fieldnames.append("emd_signature")
    if any("emd_sid_degree" in r for r in rows):
        fieldnames.append("emd_sid_degree")
    if any("shared_sids" in r for r in rows):
        fieldnames.append("shared_sids")
    if any("unchanged_payload" in r for r in rows):
        fieldnames.append("unchanged_payload")
    if any("residual_supply" in r for r in rows):
        fieldnames.append("residual_supply")
    if any("residual_demand" in r for r in rows):
        fieldnames.append("residual_demand")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[ok] CSV: {csv_path}")

def load_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load previously saved rows from a CSV instead of re-scanning JSON pairs."""
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for float_field in [
                "jac_v", "jac_e", "ged", "kld",
                "emd_degree", "emd_signature", "emd_sid_degree",
                "shared_sids", "unchanged_payload",
                "residual_supply", "residual_demand",
            ]:
                if float_field in row and row[float_field] != "":
                    row[float_field] = float(row[float_field])
            rows.append(row)
    print(f"[ok] Loaded {len(rows)} rows from {csv_path}")
    return rows

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
        "--emd-degree-pdf",
        default="emd_degree.pdf",
        help="PDF filename for degree-based EMD.",
    )
    ap.add_argument(
        "--emd-signature-pdf",
        default="emd_signature.pdf",
        help="PDF filename for signature-based EMD.",
    )
    ap.add_argument(
    "--emd-sid-degree-pdf",
    default="emd_sid_degree.pdf",
    help="PDF filename for exact-SID-plus-degree-fallback EMD.",
    )

    ap.add_argument(
    "--emd-mode",
    choices=["none", "degree", "signature", "both", "sid_degree"],
    default="sid_degree",
    help="Which EMD variant to compute.",
    )
    ap.add_argument(
        "--degree-mode",
        choices=[
            "weighted_total",
            "weighted_out",
            "unweighted_out",
            "log_weighted_out",
            "log_weighted_total",
        ],
        default="log_weighted_out",
        help="Degree proxy for degree-based EMD.",
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
        "--title-emd-degree",
        default="Earth Mover's Distance over EQC Payloads (Degree-based)",
        help="Title for degree-based EMD plot.",
    )
    ap.add_argument(
        "--title-emd-signature",
        default="Earth Mover's Distance over EQC Payloads (Signature-based)",
        help="Title for signature-based EMD plot.",
    )
    ap.add_argument(
        "--title-emd-sid-degree",
        default="Degree-based EMD",
        help="Title for exact-SID-plus-degree-fallback EMD plot.",
    )

    ap.add_argument(
        "--max-xticks",
        type=int,
        default=30,
        help="Max number of x-axis tick labels (years) to show.",
    )

    ap.add_argument(
        "--csv",
        help="(Optional) write a CSV with label, jac_v, jac_e, ged, kld, emd_degree, emd_signature.",
    )

    # optional skip flags
    ap.add_argument("--no-jac-v", action="store_true", help="Skip Jaccard-vertices PDF.")
    ap.add_argument("--no-jac-e", action="store_true", help="Skip Jaccard-edges PDF.")
    ap.add_argument("--no-ged",   action="store_true", help="Skip GED PDF.")
    ap.add_argument("--no-kld",   action="store_true", help="Skip KLD PDF.")
    ap.add_argument("--no-emd",   action="store_true", help="Skip all EMD PDFs.")
    ap.add_argument("-i", "--input-dir", required=False, help="Directory with summary JSONs (recursive).")
    ap.add_argument("--from-csv", help="Load rows from a previously saved CSV instead of scanning JSONs.")

    args = ap.parse_args()

    emd_mode = "none" if args.no_emd else args.emd_mode

    if args.from_csv:
        rows = load_from_csv(Path(args.from_csv))
    elif args.input_dir:
        emd_mode = "none" if args.no_emd else args.emd_mode
        rows = collect_pairs(
            Path(args.input_dir),
            compute_jac_v=(not args.no_jac_v),
            compute_jac_e=(not args.no_jac_e),
            compute_ged=(not args.no_ged),
            compute_kld=(not args.no_kld),
            emd_mode=emd_mode,
            degree_mode=args.degree_mode,
        )
    else:
        print("[error] Provide either -i/--input-dir or --from-csv.")
        return

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

    has_emd_degree = any("emd_degree" in r for r in rows)
    has_emd_signature = any("emd_signature" in r for r in rows)
    has_emd_sid_degree = any("emd_sid_degree" in r for r in rows)

    if not args.no_emd and rows:
        if has_emd_degree:
            _plot_metric(
                rows,
                key="emd_degree",
                ylabel="δ²_EMD (degree-based cost)",
                title=args.title_emd_degree,
                out_pdf=out_dir / args.emd_degree_pdf,
                max_xticks=args.max_xticks,
            )

        if has_emd_signature:
            _plot_metric(
                rows,
                key="emd_signature",
                ylabel="δ²_EMD (signature-field cost)",
                title=args.title_emd_signature,
                out_pdf=out_dir / args.emd_signature_pdf,
                max_xticks=args.max_xticks,
            )

        if has_emd_sid_degree:
            _plot_metric(
                rows,
                key="emd_sid_degree",
                ylabel=_sid_degree_label(args.degree_mode),
                title=args.title_emd_sid_degree,
                out_pdf=out_dir / args.emd_sid_degree_pdf,
                max_xticks=args.max_xticks,
            )

    print(f"\n[done] All metrics computed for {len(rows)} consecutive pairs.")


if __name__ == "__main__":
    main()