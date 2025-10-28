from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse canonical loader from your save.py
from save import load_summary_package as load_summary

# One-shot unary metrics aggregator (functions 1–8)
from Unary_comp import summarize_unary_metrics


def main():
    ap = argparse.ArgumentParser(
        description="One-shot unary summary of a FLUID-style summary package."
    )
    ap.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a saved summary JSON (output of save_summary_for_strategy)."
    )
    ap.add_argument(
        "--out-json",
        help="Optional: path to write metrics as JSON (e.g., results/metrics.json)."
    )
    args = ap.parse_args()

    path = Path(args.input)

    # 1) Load the summary using the canonical loader
    pkg = load_summary(path)

    # soft notice when the package is edge-free
    # Help the user understand why edge-based metrics will be 0.
    if not pkg.get("edges"):
        print("[info] No 'edges' section found (edge-free export). Edge-based metrics will be 0.")

    # 2) Compute one-shot unary metrics (counts, averages, TMH, components, etc.)
    metrics = summarize_unary_metrics(pkg)

    # 3) Pretty print to console
    print("\n=== One-shot Unary Summary ===")
    print(f"File: {path}\n")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")
    print("==============================\n")

    # 4) Optional: save JSON to file
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[main_compare] Metrics written to: {outp}")


if __name__ == "__main__":
    main()
