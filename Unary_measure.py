from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, pstdev
from typing import Dict, Any, List, Optional, Set, Tuple

import networkx as nx

def _eqc_sizes_from_nodes(pkg: Dict[str, Any]) -> Optional[List[int]]:

    # extract EQC sizes directly from the summary nodes:
    #   nodes[<sid>]["vcp"] = vertex count payload (class size).
    # returns: list[int]  if ALL nodes carry a `vcp` value, None -> if any node lacks `vcp` (caller should try a fallback).

    nodes = pkg.get("nodes") or {}
    sizes: List[int] = []
    for _, node in nodes.items():
        v = node.get("vcp")
        if v is None:
            return None  # Incomplete info → let caller fall back to "membership" if it is included
        sizes.append(int(v))
    return sizes


def _eqc_sizes_from_membership(pkg: Dict[str, Any]) -> Optional[List[int]]: #optional

    # Fallback path if `vcp` is absent and include_membership: derive EQC sizes by *inverting*
    # the membership map (label/id → sid) into counts per sid.
    # Returns: list[int] of per-class sizes, orNone if membership is unavailable.
    membership = pkg.get("membership")
    if not membership:
        return None

    counts: Dict[str, int] = defaultdict(int)
    for _, sid in membership.items():
        counts[str(sid)] += 1
    return list(counts.values())


def _require_edges(pkg: Dict[str, Any]) -> Dict[str, Any]:

    # Fetch the edges dictionary safely. Returns {} if not present.
    # to check edges exist
    edges = pkg.get("edges")
    return edges if edges else {}


def _build_digraph(pkg: Dict[str, Any]) -> nx.DiGraph:
    # because several metrics are naturally computed on a graph object,
    # Construct a directed graph of supernodes from the summary, where:
    # Node set = all supernode ids present in `nodes` (ensures isolated nodes).
    # Edge (u→v) has attribute 'weight' = rolled `count` between u and v.

    G = nx.DiGraph()

    # Add weighted edges from the edges section.
    for _, e in _require_edges(pkg).items():
        u = int(e["src"])
        v = int(e["dst"])
        w = int(e.get("count", 1))
        # default to 1 if count is missing to make robust if summary is a different source
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

        # Ensure explicit node creation
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)

    # Ensure isolated nodes (no in/out edges) are present by lifting from "nodes".
    for sid_str in (pkg.get("nodes") or {}).keys():
        sid = int(sid_str)
        if sid not in G:
            G.add_node(sid)
    
    # synthesize nodes from membership if nodes/edges are missing
    # component counts and degree stats to behave (zeros) over the correct V.
    if G.number_of_nodes() == 0 and pkg.get("membership"):
        uniq_sids = set(int(sid) for sid in pkg["membership"].values())
        for sid in uniq_sids:
            if sid not in G:
                G.add_node(sid)

    return G

# NEW: helper for V^E (entity EQCs) per Justin
def _entity_ids(pkg: Dict[str, Any]) -> Set[int]:
    """
    # CHANGED: define entity summary vertices V^E(S) as the keys in pkg['nodes'].
    # Justin counts metrics over entity EQCs; literal buckets are not part of V^E.
    """
    return {int(s) for s in (pkg.get("nodes") or {}).keys()}

def _entity_ids_min_payload(pkg: Dict[str, Any], min_payload: int = 1) -> Set[int]:
    """
    Return entity EQC ids whose payload size vcp is at least min_payload.
    """
    nodes = pkg.get("nodes") or {}
    out: Set[int] = set()

    for sid_str, node in nodes.items():
        vcp = int(node.get("vcp", 0))
        if vcp >= min_payload:
            out.add(int(sid_str))

    return out

#helpers for calculating out degree
def _weighted_out_degrees_over_entities(G: nx.DiGraph, V_E: Set[int]) -> List[int]:
    """
    # CHANGED: degrees are computed for sources in V^E only,
    # but we count edges to ANY target (entity or literal), matching Justin.
    """
    return [int(G.out_degree(n, weight="weight")) for n in V_E if n in G]


def _unweighted_out_degrees_over_entities(G: nx.DiGraph, V_E: Set[int]) -> List[int]:
    """
    # CHANGED: unweighted out-degrees for V^E only.
    """
    return [int(G.out_degree(n)) for n in V_E if n in G]

#helpers for calculating in degree
def _weighted_in_degrees_over_entities(G: nx.DiGraph, V_E: Set[int]) -> List[int]:
    return [int(G.in_degree(n, weight="weight")) for n in V_E if n in G]


def _unweighted_in_degrees_over_entities(G: nx.DiGraph, V_E: Set[int]) -> List[int]:
    return [int(G.in_degree(n)) for n in V_E if n in G]


# 1) count_equivalence_classes
def count_equivalence_classes(pkg: Dict[str, Any]) -> int:

    # Count the number of equivalence classes (FLUID supernodes) in the summary.
    # number of supernodes = eqc
    return len(pkg.get("nodes") or {})

# 2) eqc_size_list
def eqc_size_list(pkg: Dict[str, Any]) -> List[int]:

    # Extract the list of EQC sizes (payload cardinalities).
    # Priority:
    #   1) Use nodes[*]['vcp'] if present (works for full and only-id exports).
    #   2) Fall back to counting occurrences per sid in 'membership'.
    # Raises:
    #   ValueError if cannot compute sizes.
    # Returns:
    #   list[int]: e.g., [5, 12, 3, 7, ...]
    sizes = _eqc_sizes_from_nodes(pkg)
    if sizes is not None:
        return sizes

    sizes = _eqc_sizes_from_membership(pkg)
    if sizes is not None:
        return sizes

    raise ValueError(
        "Cannot derive EQC sizes: 'nodes[].vcp' and 'membership' both missing."
    )

# 3) average_eqc_size
def average_eqc_size(pkg: Dict[str, Any]) -> float:
    
    # Compute the arithmetic mean of EQC sizes.
    # Formula: mean = sum(sizes) / len(sizes), with 0.0 for empty input.
    # Re: float: average class size (e.g., 7.4)
    sizes = eqc_size_list(pkg)
    return float(mean(sizes)) if sizes else 0.0

# 4) std_eqc_size
def std_eqc_size(pkg: Dict[str, Any], *, population: bool = False) -> float:

    # Compute the standard deviation of EQC sizes.
    # Parameters:
    #   population (bool):
    #     - False (default): unbiased, n-1
    #     - True           : n
    # Edge cases: If there are 0 or 1 classes, the dispersion is 0.0 by definition.
    # Returns: float: standard deviation (e.g., 3.1)
    sizes = eqc_size_list(pkg)
    if len(sizes) < 2:
        return 0.0
    return float(stdev(sizes)) if not population else float(pstdev(sizes))

# 5) edge_set
def edge_set(
    pkg: Dict[str, Any],
    *,
    include_weights: bool = False
) -> Set[Tuple[int, int] | Tuple[int, int, int]]:
    """
    Materialize the directed superedge set from the summary.

    Parameters
    pkg:
        Summary package dictionary.

    include_weights:
        - False: return a structural edge set of pairs {(src, dst)}.
                 Parallel edges are collapsed automatically because a set of pairs
                 cannot contain duplicates.
        - True:  return a weighted edge set {(src, dst, total_count)} where
                 parallel edges with the same (src, dst) are MERGED by summing counts.
                 This is Option B.

    Returns
    Set of edges:
        - include_weights=False -> Set[(src, dst)]
        - include_weights=True  -> Set[(src, dst, total_count)]
    """
    edges = pkg.get("edges") or {}
    if not edges:
        return set()

    if not include_weights:
        # structural view
        return {(int(e["src"]), int(e["dst"])) for e in edges.values()}

    # Option B: merge parallel edges by summing their counts
    agg: Dict[Tuple[int, int], int] = defaultdict(int)
    for e in edges.values():
        u = int(e["src"])
        v = int(e["dst"])
        w = int(e.get("count", 1))
        agg[(u, v)] += w

    return {(u, v, w) for (u, v), w in agg.items()}

# 6.1 ) average_out_degree
def average_out_degree(pkg: Dict[str, Any], *, weighted: bool = True) -> float:
    """
        Average out-degree across supernodes.
        - weighted=True  : sum of per-node *weighted* out-degrees divided by |V|.
                        Equivalent to (total rolled edge weight) / |V|. volume of links
        - weighted=False : sum of per-node *unweighted* out-degrees divided by |V|.
                        Counts distinct outgoing neighbors per node. structural diversity
        Higher values indicate denser inter-class connectivity.
        Re: float: average out-degree
    """
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)
    n = len(V_E)
    if n == 0:
        return 0.0

    if weighted:
        degs = _weighted_out_degrees_over_entities(G, V_E)
    else:
        degs = _unweighted_out_degrees_over_entities(G, V_E)

    total = sum(degs)
    return float(total) / float(n) if n else 0.0

# 6.2) average_in_degree
def average_in_degree(pkg: Dict[str, Any], *, weighted: bool = True) -> float:
    """
        Average in-degree across supernodes.
        - weighted=True  : sum of per-node *weighted* in-degrees divided by |V|.
                        Equivalent to (total rolled edge weight) / |V|. volume of links
        - weighted=False : sum of per-node *unweighted* in-degrees divided by |V|.
                        Counts distinct ingoing neighbors per node. structural diversity
        Higher values indicate denser inter-class connectivity.
        Re: float: average in-degree
    """
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)
    n = len(V_E)
    if n == 0:
        return 0.0

    if weighted:
        degs = _weighted_in_degrees_over_entities(G, V_E)
    else:
        degs = _unweighted_in_degrees_over_entities(G, V_E)

    return float(sum(degs)) / float(n)

# 6.3) average degree

def average_degree(pkg: Dict[str, Any], *, weighted: bool = True) -> float:
    """
    Average total degree over entity EQCs.

    total_degree(eq) = in_degree(eq) + out_degree(eq)

    weighted=True uses rolled edge counts as weights.
    weighted=False uses number of distinct neighbors.

    Returns 0.0 if there are no entity EQCs.
    """
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)
    if not V_E:
        return 0.0

    total = 0.0
    for n in V_E:
        if weighted:
            total += float(G.in_degree(n, weight="weight") + G.out_degree(n, weight="weight"))
        else:
            total += float(G.in_degree(n) + G.out_degree(n))

    return total / float(len(V_E))

# 7) tmh_tendency_to_make_hubs
def tmh_tendency_to_make_hubs(pkg: Dict[str, Any], *, weighted: bool = True) -> float:
    """
    TMH as in the paper:

    TMH(S_t) = (sum_{eq in V^E} deg(eq)^2) / (sum_{eq in V^E} deg(eq))

    where deg(eq) counts incoming + outgoing edges.
    Uses weighted degrees if weighted=True.
    Returns 0.0 if the denominator is 0.
    """
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)
    if not V_E:
        return 0.0

    degs: List[float] = []
    for n in V_E:
        if weighted:
            d = float(G.in_degree(n, weight="weight") + G.out_degree(n, weight="weight"))
        else:
            d = float(G.in_degree(n) + G.out_degree(n))
        degs.append(d)

    s1 = sum(degs)
    if s1 == 0:
        return 0.0

    s2 = sum(d * d for d in degs)
    return s2 / s1

# 8) connected_components_count

def connected_components_count(
    pkg: Dict[str, Any],
    *,
    weak: bool = True,
    min_entity_nodes: int = 1,
    min_payload: int = 1,
) -> int:
    """
    Count connected components in the summary graph of supernodes.

    Parameters:
      weak (bool):
        - True  : weakly connected components (ignore edge directions)
        - False : strongly connected components (respect directions)

      min_entity_nodes (int):
        Minimum number of entity EQCs a component must contain to be counted.

      min_payload (int):
        Minimum payload size (vcp) an entity EQC must have to be included at all.
        Example:
          - 1  -> current behavior
          - 40 -> ignore EQCs with vcp < 40 before computing components
    """
    G = _build_digraph(pkg)

    # CHANGED: filter entity vertices by payload
    # WHY CHANGED: ignore tiny EQCs before computing components
    V_E = _entity_ids_min_payload(pkg, min_payload=min_payload)

    if G.number_of_nodes() == 0 or len(V_E) == 0:
        return 0

    # CHANGED: restrict graph to the surviving entity vertices only
    # WHY CHANGED: components should be computed after filtering vertices out
    H = G.subgraph(V_E).copy()

    if H.number_of_nodes() == 0:
        return 0

    if weak:
        UG = H.to_undirected(as_view=True)
        comps = nx.connected_components(UG)
    else:
        comps = nx.strongly_connected_components(H)

    return sum(1 for C in comps if len(C) >= min_entity_nodes)

def connected_components_count_payload_filtered(
    pkg: Dict[str, Any],
    *,
    weak: bool = True,
    min_payload: int = 40,
) -> int:
    """
    Count components after filtering out EQC vertices whose payload (vcp) is below min_payload.

    Parameters:
      weak:
        - True  -> weakly connected components
        - False -> strongly connected components
      min_payload:
        Keep only vertices with vcp >= min_payload.

    Returns:
      int: number of connected components in the induced subgraph of retained EQCs.
    """
    G = _build_digraph(pkg)
    nodes = pkg.get("nodes") or {}

    # keep only entity EQCs with enough payload
    keep: Set[int] = set()
    for sid_str, node in nodes.items():
        vcp = int(node.get("vcp", 0))
        if vcp >= min_payload:
            keep.add(int(sid_str))

    if not keep:
        return 0

    # induced subgraph on retained nodes only
    H = G.subgraph(keep).copy()

    if H.number_of_nodes() == 0:
        return 0

    if weak:
        return nx.number_connected_components(H.to_undirected(as_view=True))
    else:
        return nx.number_strongly_connected_components(H)

def connected_components_count_no_isolated_out(
    pkg: Dict[str, Any],
    *,
    weak: bool = True,
) -> int:
    """
    Count connected components after removing all EQCs with out-degree 0.
    """
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)

    if G.number_of_nodes() == 0 or len(V_E) == 0:
        return 0

    # keep only entity EQCs that have at least one outgoing edge
    keep = {n for n in V_E if n in G and G.out_degree(n) > 0}

    if not keep:
        return 0

    H = G.subgraph(keep).copy()

    if H.number_of_nodes() == 0:
        return 0

    if weak:
        return nx.number_connected_components(H.to_undirected(as_view=True))
    else:
        return nx.number_strongly_connected_components(H)

# Convenience: one-look summary
def summarize_unary_metrics(pkg: Dict[str, Any]) -> Dict[str, Any]:
    sizes = eqc_size_list(pkg)
    return {
        "eqc_count": count_equivalence_classes(pkg),
        "avg_eqc_size": average_eqc_size(pkg),
        "std_eqc_size": std_eqc_size(pkg),

        "edge_count_unweighted": len(edge_set(pkg, include_weights=False)),
        "edge_count_weighted": sum(w for *_, w in edge_set(pkg, include_weights=True)),

        "avg_out_degree_weighted": average_out_degree(pkg, weighted=True),
        "tmh_weighted": tmh_tendency_to_make_hubs(pkg, weighted=True),

        # original counts
        "components_weak": connected_components_count(pkg, weak=True, min_entity_nodes=1, min_payload=1),
        "components_strong": connected_components_count(pkg, weak=False, min_entity_nodes=1, min_payload=1),

        # original size-threshold counts
        "components_weak_ge_3": connected_components_count(pkg, weak=True, min_entity_nodes=3, min_payload=1),
        "components_weak_ge_5": connected_components_count(pkg, weak=True, min_entity_nodes=5, min_payload=1),
        "components_weak_ge_7": connected_components_count(pkg, weak=True, min_entity_nodes=7, min_payload=1),

        "components_strong_ge_3": connected_components_count(pkg, weak=False, min_entity_nodes=3, min_payload=1),
        "components_strong_ge_5": connected_components_count(pkg, weak=False, min_entity_nodes=5, min_payload=1),
        "components_strong_ge_7": connected_components_count(pkg, weak=False, min_entity_nodes=7, min_payload=1),

        # CHANGED: payload-filtered counts (vcp >= 40)
        # WHY CHANGED: ignore very small EQCs before connectivity analysis
        "components_weak_vcp_ge_40": connected_components_count(pkg, weak=True, min_entity_nodes=1, min_payload=40),
        "components_strong_vcp_ge_40": connected_components_count(pkg, weak=False, min_entity_nodes=1, min_payload=40),

        "components_weak_no_isolated_out": connected_components_count_no_isolated_out(pkg, weak=True),
        "components_strong_no_isolated_out": connected_components_count_no_isolated_out(pkg, weak=False),
        
        "components_weak_vcp_ge_40_ge_3": connected_components_count(pkg, weak=True, min_entity_nodes=3, min_payload=40),
        "components_weak_vcp_ge_40_ge_5": connected_components_count(pkg, weak=True, min_entity_nodes=5, min_payload=40),
        "components_weak_vcp_ge_40_ge_7": connected_components_count(pkg, weak=True, min_entity_nodes=7, min_payload=40),

        "components_strong_vcp_ge_40_ge_3": connected_components_count(pkg, weak=False, min_entity_nodes=3, min_payload=40),
        "components_strong_vcp_ge_40_ge_5": connected_components_count(pkg, weak=False, min_entity_nodes=5, min_payload=40),
        "components_strong_vcp_ge_40_ge_7": connected_components_count(pkg, weak=False, min_entity_nodes=7, min_payload=40),

        "n_sizes": len(sizes),
    }
