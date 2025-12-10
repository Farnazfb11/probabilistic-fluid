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

    # Materialize the directed superedge set from the summary.
    # Modes:
    #   - include_weights=False → {(src, dst)}
    #   - include_weights=True  → {(src, dst, count)}
    #   - structural view use counts
    #   - {} if no edge - robust default
    # Re: set of tuples as described above.
    E = _require_edges(pkg)
    if not E:
        return set()

    if include_weights:
        return {(int(e["src"]), int(e["dst"]), int(e.get("count", 1))) for e in E.values()}
    return {(int(e["src"]), int(e["dst"])) for e in E.values()}

# 6) average_out_degree
def average_out_degree(pkg: Dict[str, Any], *, weighted: bool = True) -> float:

    # Average out-degree across supernodes.
    # - weighted=True  : sum of per-node *weighted* out-degrees divided by |V|.
    #                    Equivalent to (total rolled edge weight) / |V|. volume of links
    # - weighted=False : sum of per-node *unweighted* out-degrees divided by |V|.
    #                    Counts distinct outgoing neighbors per node. structural diversity
    # Higher values indicate denser inter-class connectivity.
    # Re: float: average out-degree
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

# 7) tmh_tendency_to_make_hubs
def tmh_tendency_to_make_hubs(pkg: Dict[str, Any], *, weighted: bool = True) -> float:

    # Compute TMH (Tendency to Make Hubs) statistic from the out-degree sequence:
    #     TMH = (Σ d_i^2) / (2 * Σ d_i)
    # where d_i is the (weighted or unweighted) out-degree of node i.
    # meaning:(self note)
    #   - Higher TMH → degree mass is concentrated in few hubs (skewed).
    #   - Lower TMH → degrees are more evenly distributed.
    # Edge cases:
    #   - If Σ d_i == 0 (no edges), TMH is defined as 0.0 (no hub tendency).
    # Re: float: TMH value
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)

    degs = [G.out_degree(n, weight="weight") for n in V_E]
    s1 = sum(degs)
    if s1 == 0:
        return 0.0
    s2 = sum(d*d for d in degs)
    return s2 / s1

# 8) connected_components_count

def connected_components_count(pkg: Dict[str, Any], *, weak: bool = True) -> int:

    # Count connected components in the summary graph of supernodes.
    # Parameters:
    #   weak (bool):
    #     - True  (default): weakly connected components (ignore edge directions).
    #     - False          : strongly connected components (respect directions).
    # Re: int: number of components
    G = _build_digraph(pkg)
    V_E = _entity_ids(pkg)

    if G.number_of_nodes() == 0 or len(V_E) == 0:
        return 0

    if weak:
        UG = G.to_undirected(as_view=True)
        comps = nx.connected_components(UG)
    else:
        comps = nx.strongly_connected_components(G)

    # count only components that contain at least one entity EQC
    return sum(1 for C in comps if any((n in V_E) for n in C))

# Convenience: one-look summary
def summarize_unary_metrics(pkg: Dict[str, Any]) -> Dict[str, Any]:

    # Produce a compact dictionary with the core unary metrics (1–8),
    # using weighted degrees and weak connectivity by default.
    # This is handy for quick logging or tabular reporting.

    # Returns keys:
    #   - eqc_count
    #   - avg_eqc_size
    #   - std_eqc_size
    #   - edge_count_unweighted
    #   - edge_count_weighted
    #   - avg_out_degree_weighted
    #   - tmh_weighted
    #   - components_weak
    #   - components_strong (if needed, can be added)
    #   - n_sizes  (sanity: how many classes contributed to size stats)
    sizes = eqc_size_list(pkg)
    return {
        "eqc_count": count_equivalence_classes(pkg),
        "avg_eqc_size": average_eqc_size(pkg),
        "std_eqc_size": std_eqc_size(pkg),

        "edge_count_unweighted": len(edge_set(pkg, include_weights=False)),
        "edge_count_weighted": sum(w for *_, w in edge_set(pkg, include_weights=True)),

        # CHANGED: over V^E(S) only
        "avg_out_degree_weighted": average_out_degree(pkg, weighted=True),
        "tmh_weighted": tmh_tendency_to_make_hubs(pkg, weighted=True),

        # CHANGED: components that include at least one entity EQC
        "components_weak": connected_components_count(pkg, weak=True),
        "components_strong": connected_components_count(pkg, weak=False),

        "n_sizes": len(sizes),
    }
