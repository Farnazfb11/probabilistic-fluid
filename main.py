# main.py – argparse version, with auto save+load

from itertools import islice
import argparse
import os

from RDFparser import RDFParser
from RDFGraphBuilder import RDFGraphBuilder
from save import save_summary_for_strategy, load_summary_package

def build_graph(path: str, limit: int, include_literals: bool, use_numeric_ids: bool, add_reverse_edges: bool):
    """Parse N-Quads(.gz) and build a NetworkX MultiDiGraph with builder flags."""
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
        description="Build RDF graph, summarize with a chosen strategy, save JSON, then load as a test."
    )
    ap.add_argument("--input", "-i", required=True, help="Path to .nq or .nq.gz file")
    ap.add_argument("--limit", "-n", type=int, default=10000, help="Max triples to process (0 = all)")
    ap.add_argument("--include-literals", action="store_true", help="Include literal objects as nodes/edges")
    ap.add_argument("--use-numeric-ids", action="store_true", help="Hash IDs to integers instead of keeping strings")
    ap.add_argument("--reverse-edges", action="store_true", help="Add reverse edges for non-literals")
    ap.add_argument("--strategy", "-s",
                    choices=["attribute", "class", "complex"],
                    default="complex",
                    help="Summary strategy to use")
    ap.add_argument("--out", "-o", default="summaries/summary.json", help="Output JSON path for the saved summary")
    ap.add_argument("--no-load-test", action="store_true", help="Skip reloading the saved summary as a test")

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

    # Optional peek at one node and edge
    if G.number_of_nodes() and G.number_of_edges():
        u, v, d = next(iter(G.edges(data=True)))
        print("[main] Example edge:", G.nodes[u].get("label"), "--", d.get("predicate"), "-->", G.nodes[v].get("label"))

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
        out_path=args.out,
        meta=meta,
    )

    # --- Load back as a smoke test ---
    if not args.no_load_test:
        loaded = load_summary_package(args.out)
        print(f"[load-test] Loaded summary: strategy={loaded.get('meta', {}).get('strategy')}, "
              f"supernodes={len(loaded.get('nodes', {}))}, superedges={len(loaded.get('edges', {}))}")

if __name__ == "__main__":
    main()
