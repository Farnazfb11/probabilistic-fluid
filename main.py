from itertools import islice
import argparse
import os
import re
from pathlib import Path

from RDFparser import RDFParser
from RDFGraphBuilder import RDFGraphBuilder
from save import save_summary_for_strategy, load_summary_package

def infer_week_tag(input_path: str) -> str:

    # Try to infer a YYYY-MM-DD tag (the 'week') from the input path.
    # Falls back to the input file stem if no date is found.

    # Search anywhere in the path for YYYY-MM-DD
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", input_path)
    if m:
        return m.group(1)
    # else: use parent directory name or filename stem
    p = Path(input_path)
    if p.parent.name:
        return p.parent.name
    return p.stem

def ensure_out_path(out_arg: str, input_path: str, strategy: str) -> Path:
    
    # Resolve the final output file path.
    # If out_arg is a directory or endswith '/', create
    # <out_dir>/<week>_<strategy>.json.
    # If out_arg looks like a file, we keep it as-is.
    
    week = infer_week_tag(input_path)
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

def build_graph(path: str, limit: int, include_literals: bool, use_numeric_ids: bool, add_reverse_edges: bool):
    #Parse N-Quads(.gz) and build a NetworkX MultiDiGraph with builder flags.
    parser = RDFParser(path)
    triples_iter = parser.parse()
    if limit and limit > 0:
        triples_iter = islice(triples_iter, limit)

    builder = RDFGraphBuilder(
        include_literals=include_literals,
        use_numeric_ids=use_numeric_ids,
        add_reverse_edges=add_reverse_edges,
    )
    return builder.build(triples_iter)

def main():
    ap = argparse.ArgumentParser(
        description="Build RDF graph, summarize with a chosen strategy, save JSON package."
    )
    ap.add_argument("--input", "-i", required=True, help="Path to .nq or .nq.gz")
    ap.add_argument("--limit", "-n", type=int, default=10000, help="Max triples to read (0=all)")
    ap.add_argument("--include-literals", action="store_true", help="Include literals as nodes")
    ap.add_argument("--use-numeric-ids", action="store_true", help="Hash IDs to numeric ints")
    ap.add_argument("--reverse-edges", action="store_true", help="Add reverse edges")
    ap.add_argument(
        "--strategy", "-s",
        choices=["attribute", "class", "complex"],
        default="complex",
        help="Summary strategy to use"
    )
    ap.add_argument(
        "--out", "-o",
        default="summaries/",
        help="Output file OR directory. If a directory or ends with '/', file will be named <WEEK>_<STRATEGY>.json in that dir."
    )
    ap.add_argument("--no-load-test", action="store_true", help="Skip reloading the saved summary for a smoke test")

    args = ap.parse_args()

    # --- Build RDF graph ---
    G = build_graph(
        path=args.input,
        limit=args.limit,
        include_literals=args.include_literals,
        use_numeric_ids=args.use_numeric_ids,
        add_reverse_edges=args.reverse_edges,
    )
    print(f"[main] Graph built. Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")

    if G.number_of_nodes() and G.number_of_edges():
        u, v, d = next(iter(G.edges(data=True)))
        print("[main] Example edge:", G.nodes[u].get("label"), "--", d.get("predicate"), "-->", G.nodes[v].get("label"))

    # --- Resolve output path based on week + strategy ---
    out_path = ensure_out_path(args.out, args.input, args.strategy)
    print(f"[main] Saving summary to: {out_path}")

    # --- Save summary automatically ---
    meta = {
        "dataset": os.path.basename(args.input),
        "include_literals": args.include_literals,
        "use_numeric_ids": args.use_numeric_ids,
        "add_reverse_edges": args.reverse_edges,
        "limit": args.limit,
    }
    pkg = save_summary_for_strategy(
        G,
        strategy=args.strategy,   # string key, handled inside save.py
        out_path=str(out_path),
        meta=meta,
    )

    # --- Load back as a smoke test ---
    if not args.no_load_test:
        loaded = load_summary_package(str(out_path))
        print(
            f"[load-test] Loaded summary: strategy={loaded.get('meta', {}).get('strategy')}, "
            f"supernodes={len(loaded.get('nodes', {}))}, superedges={len(loaded.get('edges', []))}"
        )

if __name__ == "__main__":
    main()
