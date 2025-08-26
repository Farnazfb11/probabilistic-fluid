from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any
import networkx as nx


def group_by_signature(per_vertex_id: Dict[Any, int]) -> Dict[int, List[Any]]:
    # Group raw graph nodes into FLUID supernodes by their summary signature id.
    # Parameters
    # per_vertex_id : dict[node] -> signature_id

    # Returns
    # groups : dict[signature_id] -> list[node]
    groups: Dict[int, List[Any]] = defaultdict(list)
    for node, sid in per_vertex_id.items():
        groups[sid].append(node)
    return groups


def build_summary_graph(
    G: nx.MultiDiGraph,
    groups: Dict[int, List[Any]],
    *,
    by_predicate: bool = True,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, Dict[int, Dict[str, int]]]]:
    # Build a summary (supernode) edge view from the raw graph.

    # Parameters
    # G: NetworkX MultiDiGraph
    # groups: dict[signature_id] -> list[node]
    # by_predicate: if True, also count edges per predicate label

    # Returns
    # super_counts: dict[src_sid][dst_sid] = total edge count between supernodes
    # super_pred: dict[src_sid][dst_sid][predicate] = count per predicate (if by_predicate=True)

    # Invert groups: raw node -> supernode id
    node2sid: Dict[Any, int] = {}
    for sid, members in groups.items():
        for n in members:
            node2sid[n] = sid

    super_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    super_pred: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Walk raw edges and accumulate into superedges
    for u, v, data in G.edges(data=True):
        su = node2sid.get(u)
        sv = node2sid.get(v)
        if su is None or sv is None:
            continue
        super_counts[su][sv] += 1
        if by_predicate:
            p = data.get("predicate")
            if p is not None:
                super_pred[su][sv][p] += 1

    return super_counts, super_pred
