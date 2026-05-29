from __future__ import annotations

import argparse, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


import matplotlib.pyplot as plt
import seaborn as sns  # NEW: seaborn for nicer plots
import math


from save import load_summary_package
from Unary_measure import (
    average_eqc_size,
    std_eqc_size,
    tmh_tendency_to_make_hubs,
    count_equivalence_classes,
    edge_set,
    connected_components_count,
    connected_components_count_payload_filtered,
    connected_components_count_no_isolated_out,
    total_payload,
)

# NEW: set a global seaborn theme ready for paper
sns.set_theme(style="white", context="paper")

plt.rcParams.update({
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BLUE = "#1f77b4"          
ORANGE_DARK = "#d95f02"  


def _label(p: Path) -> str:
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", p.stem)
    return m.group(1) if m else p.stem

def _sort_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", p.stem)
    return (int(f"{m.group(1)}{m.group(2)}{m.group(3)}"), p.stem) if m else (10**10, p.stem)

def collect(input_dir: Path, *, component_min_payload: int = 40) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(input_dir.rglob("*.json"), key=_sort_key):
        try:
            pkg = load_summary_package(fp)
            rows.append({
                "file": str(fp),
                "label": _label(fp),

                "eqc": int(count_equivalence_classes(pkg)),
                "avg": float(average_eqc_size(pkg)),
                "std": float(std_eqc_size(pkg, population=False)),
                "tmh_w": float(tmh_tendency_to_make_hubs(pkg, weighted=True)),
                "tmh_u": float(tmh_tendency_to_make_hubs(pkg, weighted=False)),

                "edges_unweighted": len(edge_set(pkg, include_weights=False)),
                "edges_weighted": sum(
                    w for *_, w in edge_set(pkg, include_weights=True)
                ),

                "comp_weak": int(
                    connected_components_count(pkg, weak=True)
                ),
                "comp_strong": int(
                    connected_components_count(pkg, weak=False)
                ),

                # NEW: payload-filtered components
                "comp_weak_payload_filtered": int(
                    connected_components_count_payload_filtered(
                        pkg, weak=True, min_payload=component_min_payload
                    )
                ),
                "comp_strong_payload_filtered": int(
                    connected_components_count_payload_filtered(
                        pkg, weak=False, min_payload=component_min_payload
                    )
                ),
                # CHANGED: thresholded weak components
                # WHY CHANGED: lets you compare how many weak components remain
                # when tiny 1-node / 2-node fragments are ignored
                "comp_weak_ge_3": int(
                    connected_components_count(pkg, weak=True, min_entity_nodes=3, min_payload=1)
                ),
                "comp_weak_ge_5": int(
                    connected_components_count(pkg, weak=True, min_entity_nodes=5, min_payload=1)
                ),
                "comp_weak_ge_7": int(
                    connected_components_count(pkg, weak=True, min_entity_nodes=7, min_payload=1)
                ),

                # CHANGED: thresholded strong components
                # WHY CHANGED: same idea, but for strongly connected components
                "comp_strong_ge_3": int(
                    connected_components_count(pkg, weak=False, min_entity_nodes=3, min_payload=1)
                ),
                "comp_strong_ge_5": int(
                    connected_components_count(pkg, weak=False, min_entity_nodes=5, min_payload=1)
                ),
                "comp_strong_ge_7": int(
                    connected_components_count(pkg, weak=False, min_entity_nodes=7, min_payload=1)
                ),

                "comp_weak": int(
                    connected_components_count(pkg, weak=True)
                ),
                "comp_strong": int(
                    connected_components_count(pkg, weak=False)
                ),

                # CHANGED: components after removing EQCs with out-degree 0
                # WHY CHANGED: this matches your new metric definition
                "comp_weak_no_isolated_out": int(
                    connected_components_count_no_isolated_out(pkg, weak=True)
                ),
                "comp_strong_no_isolated_out": int(
                    connected_components_count_no_isolated_out(pkg, weak=False)
                ),

                "total_payload": int(total_payload(pkg)),
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
            fieldnames=[
                "label",
                "file",
                "eqc",
                "avg",
                "std",
                "tmh_w",
                "tmh_u",
                "edges_unweighted",
                "edges_weighted",
                "comp_weak",
                "comp_strong",
                "comp_weak_no_isolated_out",      # CHANGED
                "comp_strong_no_isolated_out",    # CHANGED
                "comp_weak_payload_filtered",
                "comp_strong_payload_filtered",
                "comp_weak_ge_3",
                "comp_weak_ge_5",
                "comp_weak_ge_7",
                "comp_strong_ge_3",
                "comp_strong_ge_5",
                "comp_strong_ge_7",
            ]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] CSV: {csv_path}")

def _year_ticks(labels: List[str], max_ticks: int) -> Tuple[List[int], List[str]]:
    # Build x-ticks that show (at most) one label per year,
    # but space the year labels *evenly* across the full x-range.

    # We keep all weekly data points; only the tick positions are "normalized"
    # so years don't bunch up visually when some years have more weeks.
    if not labels:
        return [], []

    # Extract years in order of appearance (they're already chronological)
    years_seen: List[str] = []
    for lbl in labels:
        y = lbl[:4]
        if not years_seen or years_seen[-1] != y:
            years_seen.append(y)

    # Optionally thin years further if there are too many distinct years
    if len(years_seen) > max_ticks:
        step = max(1, len(years_seen) // max_ticks)
        years_used = years_seen[::step]
    else:
        years_used = years_seen

    n = len(labels)
    m = len(years_used)
    if m == 0:
        return [], []

    # Evenly space tick positions from 0 to n-1 for the chosen years
    if m == 1:
        positions = [0]
    else:
        positions = [
            int(round(i * (n - 1) / (m - 1)))
            for i in range(m)
        ]

    return positions, years_used

#the 5 year filter
def _filter_rows_by_year_range(
    rows: List[Dict[str, Any]],
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    subset: List[Dict[str, Any]] = []
    for r in rows:
        try:
            y = int(r["label"][:4])
        except Exception:
            continue
        if start_year <= y <= end_year:
            subset.append(r)
    return subset

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
    xs     = list(range(len(labels)))
    avgs   = [r["avg"]   for r in rows]
    eqcs   = [r["eqc"]   for r in rows]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis line (no seaborn legend)
    sns.lineplot(
        x=xs,
        y=avgs,
        ax=ax1,
        color=BLUE,
        linewidth=1.2,
        label="Average size of EQC",
        legend=False,
    )
    ax1.set_ylabel("Average size of EQC", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(False)

    # Right axis line (no seaborn legend)
    ax2 = ax1.twinx()
    sns.lineplot(
        x=xs,
        y=eqcs,
        ax=ax2,
        color=ORANGE_DARK,
        linestyle=":",
        linewidth=1.2,
        label="Number of EQC",
        legend=False,
    )
    ax2.set_ylabel("Number of EQC", color=ORANGE_DARK)
    ax2.tick_params(axis="y", labelcolor=ORANGE_DARK)
    ax2.grid(False)

    fig.suptitle(title)

    # x axis ticks
    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=0)

    # Single combined legend (one box)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
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
    xs = list(range(len(labels)))
    stds = [r["std"] for r in rows]

    

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    sns.lineplot(
        x=xs,
        y=stds,
        ax=ax,
        color=BLUE,
        linewidth=1.2,
    )
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Std. dev. of EQC size")
    ax.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    plt.xticks(xticks, xlabels)

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[ok] PDF: {out_pdf}")

def plot_tmh(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    key: str,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))
    tmh = [r[key] for r in rows]

    if key == "tmh_w":
        raw_edges   = [r["edges_weighted"]   for r in rows]
        edges_plot  = [0 if v <= 0 else math.log10(v) for v in raw_edges]
        edge_label  = "Weighted superedges (log10 count)"
    else:
        raw_edges   = [r["edges_unweighted"] for r in rows]
        edges_plot  = raw_edges
        edge_label  = "Number of superedges"

    fig, ax1 = plt.subplots(figsize=(12, 5))


    sns.lineplot(
        x=xs,
        y=tmh,
        ax=ax1,
        color=BLUE,
        linewidth=1.2,
    )
    ax1.set_ylabel("TMH", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(False)

    ax2 = ax1.twinx()
    sns.lineplot(
        x=xs,
        y=edges_plot,
        ax=ax2,
        color=ORANGE_DARK,
        linestyle=":",
        linewidth=1.2,
    )
    ax2.set_ylabel(edge_label, color=ORANGE_DARK)
    ax2.tick_params(axis="y", labelcolor=ORANGE_DARK)
    ax2.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=0)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

def plot_components_unfiltered(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))
    weak_vals = [r["comp_weak"] for r in rows]
    strong_vals = [r["comp_strong"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        x=xs,
        y=weak_vals,
        ax=ax,
        color=BLUE,
        linewidth=1.2,
        label="Weakly connected",
    )

    sns.lineplot(
        x=xs,
        y=strong_vals,
        ax=ax,
        color=ORANGE_DARK,
        linestyle=":",
        linewidth=1.2,
        label="Strongly connected",
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of components")
    ax.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.legend(loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

def plot_components_payload_filtered(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
    min_payload: int,
):
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))
    weak_vals = [r["comp_weak_payload_filtered"] for r in rows]
    strong_vals = [r["comp_strong_payload_filtered"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        x=xs,
        y=weak_vals,
        ax=ax,
        color=BLUE,
        linewidth=1.2,
        label=f"Weakly connected (payload ≥ {min_payload})",
    )

    sns.lineplot(
        x=xs,
        y=strong_vals,
        ax=ax,
        color=ORANGE_DARK,
        linestyle=":",
        linewidth=1.2,
        label=f"Strongly connected (payload ≥ {min_payload})",
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of components")
    ax.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.legend(loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

def plot_components_thresholded(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
    *,
    prefix: str,
    ylabel: str,
):
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))

    raw_vals = [r[prefix] for r in rows]
    ge3_vals = [r[f"{prefix}_ge_3"] for r in rows]
    ge5_vals = [r[f"{prefix}_ge_5"] for r in rows]
    ge7_vals = [r[f"{prefix}_ge_7"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))

    # CHANGED: four curves on one plot, similar idea to your k-ary multi-line view
    # WHY CHANGED: easier comparison between raw and thresholded versions
    sns.lineplot(
        x=xs,
        y=raw_vals,
        ax=ax,
        linewidth=1.2,
        label="Unfiltered",
        color="#ffc508",
    )

    sns.lineplot(
        x=xs,
        y=ge3_vals,
        ax=ax,
        linewidth=1.2,
        label="≥ 3 nodes",
        color="#37c71a",
    )

    sns.lineplot(
        x=xs,
        y=ge5_vals,
        ax=ax,
        linewidth=1.2,
        label="≥ 5 nodes",
        color="#091be0",
    )

    sns.lineplot(
        x=xs,
        y=ge7_vals,
        ax=ax,
        linewidth=1.2,
        label="≥ 7 nodes",
        color="#d95f02",
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.legend(loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

def plot_components_no_isolated_out(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int,
):
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))
    weak_vals = [r["comp_weak_no_isolated_out"] for r in rows]
    strong_vals = [r["comp_strong_no_isolated_out"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        x=xs,
        y=weak_vals,
        ax=ax,
        color=BLUE,
        linewidth=1.2,
        label="Weakly connected (out-degree > 0)",
    )

    sns.lineplot(
        x=xs,
        y=strong_vals,
        ax=ax,
        color=ORANGE_DARK,
        linestyle=":",
        linewidth=1.2,
        label="Strongly connected (out-degree > 0)",
    )

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of components")
    ax.grid(False)

    xticks, xlabels = _year_ticks(labels, max_xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.legend(loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

def plot_total_payload(
    rows: List[Dict[str, Any]],
    out_pdf: Path,
    title: str,
    max_xticks: int = 40,
) -> None:
    labels = [r["label"] for r in rows]
    values = [r["total_payload"] for r in rows]
    xs = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, values, color=BLUE, linewidth=0.8, label="Total payload (sum of vcp)")
    ax.set_title(title)
    ax.set_ylabel("Total payload (entity count)")
    ax.set_xlabel("Timestep")

    ticks, tick_labels = _year_ticks(labels, max_xticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] PDF: {out_pdf}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Create PDFs: avg+EQC count, SD of EQC size, TMH weighted, TMH unweighted."
    )
    ap.add_argument("-i", "--input-dir", required=True, help="Directory with summary JSONs (recursive).")
    ap.add_argument("--out-dir", default="results", help="Output directory for PDFs (default: results).")

    ap.add_argument(
        "--avg-eqc-pdf",
        default="avg_and_eqc_count.pdf",
        help="Filename for combined plot: average EQC size vs number of EQCs.",
    )
    ap.add_argument(
        "--std-eqc-pdf",
        default="eqc_std.pdf",
        help="Filename for standard deviation of EQC sizes PDF.",
    )

    ap.add_argument("--tmh-weighted-pdf", default="tmh_weighted.pdf", help="Filename for weighted TMH PDF.")
    ap.add_argument("--tmh-unweighted-pdf", default="tmh_unweighted.pdf", help="Filename for unweighted TMH PDF.")

    # NEW: extra TMH-weighted plot for 2015–2020
    ap.add_argument(
        "--tmh-weighted-2015-2020-pdf",
        default="tmh_weighted_2015_2020.pdf",
        help="Filename for weighted TMH PDF, restricted to years 2015–2020.",
    )

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

    # NEW: separate title for the 2015–2020 TMH plot
    ap.add_argument(
        "--title-tmh-w-2015-2020",
        default="TMH (Weighted Degrees), 2015–2020",
        help="Title for weighted TMH plot restricted to 2015–2020.",
    )

    ap.add_argument("--max-xticks", type=int, default=40, help="Max x-axis tick labels (default: 40).")
    ap.add_argument(
        "--csv",
        help="(Optional) write a CSV with eqc, avg, std, tmh_w, tmh_u, edges_unweighted, edges_weighted.",
    )

    ap.add_argument("--no-avg-eqc", action="store_true", help="Skip avg+EQC-count PDF.")
    ap.add_argument("--no-std-eqc", action="store_true", help="Skip SD-only EQC PDF.")
    ap.add_argument("--no-tmh-weighted", action="store_true", help="Skip weighted TMH PDF.")
    ap.add_argument("--no-tmh-unweighted", action="store_true", help="Skip unweighted TMH PDF.")
    # NEW: skip-flag just for the 2015–2020 plot
    ap.add_argument(
        "--no-tmh-weighted-2015-2020",
        action="store_true",
        help="Skip weighted TMH PDF for 2015–2020.",
    )
    ap.add_argument(
    "--components-payload-filtered-pdf",
    default="components_payload_filtered.pdf",
    help="Filename for weak/strong connected components plot after payload filtering.",
    )

    ap.add_argument(
        "--title-components-payload-filtered",
        default="Weakly vs Strongly Connected Components (payload filtered)",
        help="Title for payload-filtered components plot.",
    )

    ap.add_argument(
        "--component-min-payload",
        type=int,
        default=40,
        help="Minimum payload (vcp) required to keep a vertex for the component plot.",
    )

    ap.add_argument(
        "--no-components-payload-filtered",
        action="store_true",
        help="Skip payload-filtered weak/strong components PDF.",
    )
    ap.add_argument(
        "--components-unfiltered-pdf",
        default="components_unfiltered.pdf",
        help="Filename for weak/strong connected components plot.",
    )

    ap.add_argument(
        "--title-components-unfiltered",
        default="Weakly vs Strongly Connected Components",
        help="Title for unfiltered components plot.",
    )

    ap.add_argument(
        "--no-components-unfiltered",
        action="store_true",
        help="Skip unfiltered weak/strong components PDF.",
    )

    ap.add_argument(
    "--components-weak-thresholded-pdf",
    default="components_weak_thresholded.pdf",
    help="Filename for weakly connected components plot with node-count thresholds.",
    )

    ap.add_argument(
        "--components-strong-thresholded-pdf",
        default="components_strong_thresholded.pdf",
        help="Filename for strongly connected components plot with node-count thresholds.",
    )

    ap.add_argument(
        "--title-components-weak-thresholded",
        default="Weakly Connected Components with Node Thresholds",
        help="Title for thresholded weak components plot.",
    )

    ap.add_argument(
        "--title-components-strong-thresholded",
        default="Strongly Connected Components with Node Thresholds",
        help="Title for thresholded strong components plot.",
    )

    ap.add_argument(
        "--no-components-weak-thresholded",
        action="store_true",
        help="Skip weakly connected components threshold plot.",
    )

    ap.add_argument(
        "--no-components-strong-thresholded",
        action="store_true",
        help="Skip strongly connected components threshold plot.",
    )

    ap.add_argument(
        "--components-no-isolated-out-pdf",
        default="components_no_isolated_out.pdf",
        help="Filename for weak/strong connected components plot after removing EQCs with out-degree 0.",
    )

    ap.add_argument(
        "--title-components-no-isolated-out",
        default="Weakly vs Strongly Connected Components (out-degree > 0)",
        help="Title for components plot after removing EQCs with out-degree 0.",
    )

    ap.add_argument(
        "--no-components-no-isolated-out",
        action="store_true",
        help="Skip weak/strong components plot after removing EQCs with out-degree 0.",
    )
    ap.add_argument("--no-total-payload", action="store_true", help="Skip total payload PDF.")
    ap.add_argument("--total-payload-pdf", default="total_payload.pdf")
    ap.add_argument("--title-total-payload", default="Total Payload over Time")

    args = ap.parse_args()

    rows = collect(
        Path(args.input_dir),
        component_min_payload=args.component_min_payload,
    )
    if not rows:
        print("[info] No summaries found or all failed to parse.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_write_csv(rows, Path(args.csv) if args.csv else None)

    if not args.no_avg_eqc:
        plot_avg_and_eqc_count(
            rows,
            out_dir / args.avg_eqc_pdf,
            args.title_avg_eqc,
            args.max_xticks,
        )
    if not args.no_std_eqc:
        plot_std_only(
            rows,
            out_dir / args.std_eqc_pdf,
            args.title_std_eqc,
            args.max_xticks,
        )
    if not args.no_tmh_weighted:
        plot_tmh(
            rows,
            out_dir / args.tmh_weighted_pdf,
            args.title_tmh_w,
            key="tmh_w",
            max_xticks=args.max_xticks,
        )
    if not args.no_tmh_unweighted:
        plot_tmh(
            rows,
            out_dir / args.tmh_unweighted_pdf,
            args.title_tmh_u,
            key="tmh_u",
            max_xticks=args.max_xticks,
        )
    if not args.no_components_payload_filtered:
        plot_components_payload_filtered(
            rows,
            out_dir / args.components_payload_filtered_pdf,
            args.title_components_payload_filtered,
            args.max_xticks,
            args.component_min_payload,
        )
    if not args.no_components_unfiltered:
        plot_components_unfiltered(
            rows,
            out_dir / args.components_unfiltered_pdf,
            args.title_components_unfiltered,
            args.max_xticks,
        )
    if not args.no_components_weak_thresholded:
        plot_components_thresholded(
            rows,
            out_dir / args.components_weak_thresholded_pdf,
            args.title_components_weak_thresholded,
            args.max_xticks,
            prefix="comp_weak",
            ylabel="Number of weak components",
        )

    if not args.no_components_strong_thresholded:
        plot_components_thresholded(
            rows,
            out_dir / args.components_strong_thresholded_pdf,
            args.title_components_strong_thresholded,
            args.max_xticks,
            prefix="comp_strong",
            ylabel="Number of strong components",
        )

    if not args.no_components_no_isolated_out:
        plot_components_no_isolated_out(
            rows,
            out_dir / args.components_no_isolated_out_pdf,
            args.title_components_no_isolated_out,
            args.max_xticks,
        )

    # NEW: extra plot for weighted TMH in 2015–2020 only
    if not args.no_tmh_weighted_2015_2020:
        rows_2015_2020 = _filter_rows_by_year_range(rows, 2015, 2020)
        if not rows_2015_2020:
            print("[info] No summaries with labels in years 2015–2020; skipping extra TMH plot.")
        else:
            plot_tmh(
                rows_2015_2020,
                out_dir / args.tmh_weighted_2015_2020_pdf,
                args.title_tmh_w_2015_2020,
                key="tmh_w",
                max_xticks=args.max_xticks,
            )
    
    if not args.no_total_payload:
        plot_total_payload(
            rows,
            out_dir / args.total_payload_pdf,
            args.title_total_payload,
            args.max_xticks,
        )

if __name__ == "__main__":
    main()