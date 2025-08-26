# main.py (no argparse version)

from itertools import islice

from Collector import SchemaCollector
from RDFparser import RDFParser
from RDFGraphBuilder import RDFGraphBuilder

from AttCollection import SEAttributeCollection
from Collector import SchemaCollector
from Summarizer import group_by_signature, build_summary_graph

# =========================
# EDIT THESE SETTINGS
# =========================
PATH = r"C:\Users\Surface\HiwiProject\2012-05-06\data-seedlist.nq.gz"  
LIMIT = 10000            # number of triples to process 0 = all
PRINT_Debug= False       # True = only print first LIMIT triples, don't build graph
INCLUDE_LITERALS = False # include literal objects as nodes/edges
STRING_IDS = True        # True = keep original string IDs; False = hash to ints
REVERSE_EDGES = False    # add reversed edges (object->subject) for non-literals
# =========================

def main():
    # 1) Parse triples (the parser itself has no internal limit)
    parser = RDFParser(PATH)
    triples_iter = parser.parse()

    # Optional external limit
    if LIMIT and LIMIT > 0:
        triples_iter = islice(triples_iter, LIMIT)

    # 2) Print-only preview
    if PRINT_Debug:
        count = 0
        for s, p, o, g in triples_iter:
            print(f"S: {s}\nP: {p}\nO: {o}\nG: {g}\n---")
            count += 1
        if LIMIT and LIMIT > 0:
            print(f"[main] Printed {count} triple(s) (LIMIT={LIMIT}).")
        else:
            print(f"[main] Printed {count} triple(s).")
        return

    # 3) Build graph
    builder = RDFGraphBuilder(
        include_literals=INCLUDE_LITERALS,
        use_numeric_ids=not STRING_IDS,
        add_reverse_edges=REVERSE_EDGES,
    )
    G = builder.build(triples_iter)

    # 4) Quick stats + small peek
    print(f"[main] Graph built.")
    print(f"[main] Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        try:
            some_node = next(iter(G.nodes))
            print("[main] Sample node label:", G.nodes[some_node].get("label"))
            print("[main] Sample node types:", G.nodes[some_node].get("types"))
            u, v, d = next(iter(G.edges(data=True)))
            print("[main] Sample edge:", G.nodes[u].get("label"), "--", d.get("predicate"), "-->", G.nodes[v].get("label"))
        except StopIteration:
            pass
    
    # 4) Run AttributeCollection via SchemaCollector (GraphX-like aggregate)
    strategy = SEAttributeCollection()
    collector = SchemaCollector(strategy)
    per_v_summary, per_v_id = collector.collect(G)

    # 5) Group by signature (FLUID supernodes)
    groups = group_by_signature(per_v_id)
    print(f"[summary] Signature groups (supernodes): {len(groups)}")

    # Show a few groups (id, size, sample members)
    for sid, members in list(groups.items())[:10]:
        sample = members[:3] if isinstance(members, list) else []
        print(f"  Supernode {sid} with {len(members)} members -> sample: {sample}")

    # 6) Build summary graph (superedges) with per-predicate counts
    super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)

    # Quick stats
    total_superedges = sum(len(dst) for dst in super_counts.values())
    print(f"[summary] Summary graph supernodes: {len(super_counts)}")
    print(f"[summary] Summary graph superedges: {total_superedges}")

    # Show top-10 heaviest superedges
    #if edge count append
    #(src Id)-> (dst Id) : edge_count(predicate)
    heavy = []
    for su, row in super_counts.items():
        for sv, c in row.items():
            heavy.append((c, su, sv))
    for c, su, sv in sorted(heavy, reverse=True)[:10]:
        preds = dict(sorted(super_pred.get(su, {}).get(sv, {}).items(), key=lambda kv: kv[1], reverse=True))
        print(f"  {su} -> {sv} : {c} edges {('(' + ', '.join(f'{k}: {v}' for k, v in list(preds.items())[:3]) + ')') if preds else ''}")

if __name__ == "__main__":
    main()