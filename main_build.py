from itertools import islice
import argparse
import os
import re
from pathlib import Path

from RDFparser import RDFParser
from RDFGraphBuilder import RDFGraphBuilder
from save import save_summary_for_strategy, load_summary_package


# To get both files or a list and treat them as one
def infer_week_tag(input_paths: list[str]) -> str:
    # If multiple files are given, try to infer from parent directory or first matching YYYY-MM-DD
    for path in input_paths:
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", path)
        if m:
            return m.group(1)
    # fallback: use parent dir name of first file, else stem of first file
    p = Path(input_paths[0])
    return p.parent.name or p.stem

def ensure_out_path(out_arg: str, input_paths: list[str], strategy: str) -> Path:
    # Resolve the final output file path.
    week = infer_week_tag(input_paths)
    out = Path(out_arg)

    # Treat existing directory or trailing slash as a directory target
    is_dir_target = out_arg.endswith(os.sep) or (out.exists() and out.is_dir())

    if is_dir_target:
        out_dir = out if out_arg else Path("summaries")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{week}_{strategy}.json"
    else:
        # If a filename was given, make sure the directory exists
        out.parent.mkdir(parents=True, exist_ok=True)
    return out

def build_graph(paths: list[str], limit: int, include_literals: bool, use_numeric_ids: bool, add_reverse_edges: bool):
    #one graph for multiple files
    builder = RDFGraphBuilder(
        include_literals=include_literals,
        use_numeric_ids=use_numeric_ids,
        add_reverse_edges=add_reverse_edges,
    )
    #Parse N-Quads(.gz) and build a NetworkX MultiDiGraph with builder flags.
    count = 0
    for path in paths:
        try:
            parser = RDFParser(path)
            print(f"[INFO] Processing file: {path}")
            triples_iter = parser.parse()
            if limit and limit > 0:
                triples_iter = islice(triples_iter, max(0, limit - count))
            for triple in triples_iter:
                builder.add_statement(*triple)
                count += 1
                if limit and limit > 0 and count >= limit:
                    break
            if limit and limit > 0 and count >= limit:
                break
        except FileNotFoundError:
            print(f"[WARN] File not found, skipping: {path}")
            continue
    
    return builder.G, count

def main():
    ap = argparse.ArgumentParser(
        description="Build RDF graph, summarize with a chosen strategy, save JSON package."
    )
    ap.add_argument("--input", "-i", nargs="+", required=True, help="One or more paths to .nq or .nq.gz files")
    ap.add_argument("--limit", "-n", type=int, default=10000, help="Max triples to read (0=all)")
    ap.add_argument("--include-literals", action="store_true", help="Include literals as nodes")
    ap.add_argument("--use-numeric-ids", action="store_true", help="Hash IDs to numeric ints")
    ap.add_argument("--reverse-edges", action="store_true", help="Add reverse edges")
    ap.add_argument(
        "--strategy", "-s",
        choices=["attribute", "class", "attributeclass"],
        default="attributeclass",
        help="Summary strategy to use"
    )
    ap.add_argument(
        "--out", "-o",
        default="summaries/",
        help="Output file OR directory. If a directory or ends with '/', file will be named <WEEK>_<STRATEGY>.json in that dir."
    )
    ap.add_argument(
        "--include-membership",
        action="store_true",
        help="If set, include the (large) node→supernode membership map in the JSON (default: off)."
    )
    ap.add_argument(
        "--load-test",
        action="store_true",
        help="If set, reload the saved summary for a smoke test (default: off)."
    )
    args = ap.parse_args()

    # --- Build RDF graph from multiple files ---
    G, count = build_graph(
        paths=args.input,
        limit=args.limit,
        include_literals=args.include_literals,
        use_numeric_ids=args.use_numeric_ids,
        add_reverse_edges=args.reverse_edges,
    )
    print(f"[main] Graph built. Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}  Triples={count}")

    if G.number_of_nodes() and G.number_of_edges():
        u, v, d = next(iter(G.edges(data=True)))
        print("[main] Example edge:", G.nodes[u].get("label"), "--", d.get("predicate"), "-->", G.nodes[v].get("label"))

    # --- Resolve output path based on week + strategy ---
    out_path = ensure_out_path(args.out, args.input, args.strategy)
    print(f"[main] Saving summary to: {out_path}")

    # --- Save summary automatically ---
    meta = {
        "datasets": [os.path.basename(p) for p in args.input],
        "include_literals": args.include_literals,
        "use_numeric_ids": args.use_numeric_ids,
        "add_reverse_edges": args.reverse_edges,
        "limit": args.limit,
        "include_membership": args.include_membership,  # record how we saved
    }
    pkg = save_summary_for_strategy(
        G,
        strategy=args.strategy,
        out_path=str(out_path),
        meta=meta,
        include_membership=args.include_membership,
    )

    # --- Print quick summary from pkg instead of reloading file ---
    print(
        f"[summary] Saved summary with strategy={pkg['meta'].get('strategy')} "
        f"(supernodes={len(pkg['nodes'])}, superedges={len(pkg['edges'])})"
    )

    #--- Load back as a smoke test ---
    if args.load_test:  
        loaded = load_summary_package(str(out_path))
        print(
            f"[load-test] Loaded summary: strategy={loaded.get('meta', {}).get('strategy')}, "
            f"supernodes={len(loaded.get('nodes', {}))}, superedges={len(loaded.get('edges', []))}"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user. Exiting cleanly.")
