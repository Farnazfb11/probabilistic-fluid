import math
from typing import Dict, Any, Set, Tuple, List

import networkx as nx

from Unary_measure import (
    _entity_ids,       # V^E(S)
    edge_set,          # structural edges
    eqc_size_list,     # EQC sizes (list)
    _build_digraph,    # graph construction from summary
)


# 1) Jaccard over vertices and edges (UNCHANGED - already correct)

def jaccard_vertices(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Binary metric δ^{(2)}_{JacV}(S_t, S_{t+1}) over entity vertices V^E.
    Paper Equation 12.
    """
    V_t: Set[int] = _entity_ids(pkg_t)
    V_tp1: Set[int] = _entity_ids(pkg_tp1)

    if not V_t and not V_tp1:
        return 0.0

    inter = V_t & V_tp1
    union = V_t | V_tp1

    return 1.0 - (len(inter) / float(len(union)))


def jaccard_edges(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Binary metric δ^{(2)}_{JacE}(S_t, S_{t+1}) over superedges E(S).
    Paper Equation 13.
    """
    E_t: Set[Tuple[int, int]] = edge_set(pkg_t, include_weights=False)
    E_tp1: Set[Tuple[int, int]] = edge_set(pkg_tp1, include_weights=False)

    if not E_t and not E_tp1:
        return 0.0

    inter = E_t & E_tp1
    union = E_t | E_tp1

    return 1.0 - (len(inter) / float(len(union)))


# 2) Graph Edit Distance - FIXED to match paper's per-class formulation

def graph_edit_distance_structural(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> int:
    """
    Graph Edit Distance as per Paper Equation 14.
    """
    G_t = _build_digraph(pkg_t)
    G_tp1 = _build_digraph(pkg_tp1)
    
    V_t = set(_entity_ids(pkg_t))
    V_tp1 = set(_entity_ids(pkg_tp1))
    S_GED = V_t | V_tp1
    
    total_edge_edits = 0
    total_class_edits = 0
    
    for sid in S_GED:
        # Get incident edges for this class
        edges_t = _get_incident_edges(G_t, sid)
        edges_tp1 = _get_incident_edges(G_tp1, sid)
        
        # Count edge operations (will be counted twice - once per endpoint)
        edge_additions = len(edges_tp1 - edges_t)
        edge_deletions = len(edges_t - edges_tp1)
        total_edge_edits += edge_additions + edge_deletions
        
        # Count class creation/deletion (only for isolated classes)
        if sid not in V_t and sid in V_tp1:
            if not edges_tp1:
                total_class_edits += 1
        elif sid in V_t and sid not in V_tp1:
            if not edges_t:
                total_class_edits += 1
    
    # Edge edits counted twice (from both endpoints), divide by 2
    # Class edits counted once, don't divide
    return (total_edge_edits // 2) + total_class_edits


def _get_incident_edges(graph: nx.DiGraph, sid: int) -> Set[Tuple[int, int]]:
    """Get all edges where sid is either source or target."""
    edges = set()
    if sid in graph:
        # Outgoing edges
        for succ in graph.successors(sid):
            edges.add((sid, succ))
        # Incoming edges  
        for pred in graph.predecessors(sid):
            edges.add((pred, sid))
    return edges


# 3) Symmetric KL divergence - FIXED support sets to match paper

def _eqc_size_map(pkg: Dict[str, Any]) -> Dict[int, int]:
    """
    Build sid -> size map.
    Priority: 1) nodes[sid]['vcp'], 2) membership count
    """
    nodes = pkg.get("nodes") or {}
    
    # Try via vcp
    all_have_vcp = all("vcp" in node for node in nodes.values())
    if all_have_vcp and nodes:
        return {int(sid): int(node["vcp"]) for sid, node in nodes.items()}

    # Fallback via membership
    membership = pkg.get("membership")
    if membership:
        counts: Dict[int, int] = {}
        for _, sid in membership.items():
            k = int(sid)
            counts[k] = counts.get(k, 0) + 1
        return counts

    raise ValueError(
        "Cannot derive EQC size map: 'nodes[].vcp' and 'membership' both missing."
    )


def _normalize_sizes(size_map: Dict[int, int]) -> Dict[int, float]:
    """Turn sid -> size into probability distribution P(eq)."""
    total = float(sum(size_map.values()))
    if total <= 0.0:
        return {sid: 0.0 for sid in size_map.keys()}
    return {sid: (sz / total) for sid, sz in size_map.items()}


def _kl_divergence(
    P: Dict[int, float],
    Q: Dict[int, float],
    support: Set[int],
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute D(P || Q) over given support set.
    
    As per paper Equation 17 and note in Equation 19:
    - If P(eq) = 0, summand is 0 (limit x log x → 0)
    - If Q(eq) = 0 and P(eq) > 0, clip Q to epsilon
    """
    acc = 0.0
    for sid in support:
        p = P.get(sid, 0.0)
        if p <= 0.0:
            continue  # 0 * log(0/·) = 0

        q = Q.get(sid, 0.0)
        if q <= 0.0:
            q = epsilon

        acc += p * math.log2(p / q)
    return acc


def symmetric_kl_divergence(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Symmetric KL divergence as per Paper Equations 17 and 20.
    
    FIXED: Support sets now match paper exactly:
    - D(S_t, S_{t+1}): sum over V^E(S_{t+1})  [Eq. 17]
    - D(S_{t+1}, S_t): sum over V^E(S_t)
    
    δ^{(2)}_{KLD}(S_t, S_{t+1}) = D(S_t, S_{t+1}) + D(S_{t+1}, S_t)
    """
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)

    P_t = _normalize_sizes(size_t)
    P_tp1 = _normalize_sizes(size_tp1)

    V_t = set(size_t.keys())
    V_tp1 = set(size_tp1.keys())

    # FIXED: Match paper's Equation 17
    # D(S_t, S_{t+1}): sum over V^E(S_{t+1})
    D_t_tp1 = _kl_divergence(P_t, P_tp1, support=V_tp1)

    # D(S_{t+1}, S_t): sum over V^E(S_t)
    D_tp1_t = _kl_divergence(P_tp1, P_t, support=V_t)

    return D_t_tp1 + D_tp1_t


# 4) Earth Mover's Distance - ENHANCED with proper flow network

def earth_mover_distance(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Earth Mover's Distance as per Paper Equation 15.
    
    Implements the full min-cost flow network from Figures 3-4:
    - V_d: classes with decreased payload (supply)
    - V_i: classes with increased payload (demand)
    - Distance d_{i,j} between classes (using edit distance)
    - Flow network with source, target, and balance vertices
    
    Falls back to simplified calculation if flow network fails.
    """
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)
    
    # Build V_d (decreased) and V_i (increased)
    V_d: Dict[int, int] = {}
    V_i: Dict[int, int] = {}
    
    all_sids = set(size_t.keys()) | set(size_tp1.keys())
    for sid in all_sids:
        s_t = size_t.get(sid, 0)
        s_tp1 = size_tp1.get(sid, 0)
        delta = s_tp1 - s_t
        
        if delta > 0:
            V_i[sid] = delta
        elif delta < 0:
            V_d[sid] = -delta
    
    # No change → zero cost
    if not V_d and not V_i:
        return 0.0
    
    # Try full flow network implementation
    try:
        return _emd_min_cost_flow(V_d, V_i, pkg_t, pkg_tp1)
    except (nx.NetworkXError, nx.NetworkXUnfeasible, Exception):
        # Fallback to simplified version (uniform distance)
        return _emd_simplified(V_d, V_i)


def _structural_edit_distance(sid_i: int, sid_j: int, 
                               pkg_t: Dict[str, Any], 
                               pkg_tp1: Dict[str, Any]) -> float:
    """
    Calculate structural edit distance d_{i,j} between two classes.
    """
    if sid_i == sid_j:
        return 0.0
    
    G_t = _build_digraph(pkg_t)
    G_tp1 = _build_digraph(pkg_tp1)
    
    def get_total_degree(graph: nx.DiGraph, sid: int) -> int:
        if sid not in graph:
            return 0
        # Get weighted degree correctly
        out_deg = graph.out_degree(sid, weight='weight')  # Already a number
        in_deg = graph.in_degree(sid, weight='weight')    # Already a number
        return int(out_deg + in_deg)
    
    deg_i = get_total_degree(G_t, sid_i)
    deg_j = get_total_degree(G_tp1, sid_j)
    
    return float(abs(deg_i - deg_j) + 1)


def _emd_min_cost_flow(V_d: Dict[int, int], V_i: Dict[int, int],
                       pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Full min-cost flow implementation as per paper Figures 3-4.
    """
    F = nx.DiGraph()
    
    source = "SOURCE"
    target = "TARGET"
    
    # Calculate totals
    total_supply = sum(V_d.values())
    total_demand = sum(V_i.values())
    
    # Add supply nodes (V_d) and edges from source
    for sid, supply in V_d.items():
        node_d = f"d_{sid}"
        F.add_edge(source, node_d, weight=0, capacity=supply)
    
    # Add demand nodes (V_i) and edges to target
    for sid, demand in V_i.items():
        node_i = f"i_{sid}"
        F.add_edge(node_i, target, weight=0, capacity=demand)
    
    # Add edges between all V_d and V_i pairs with cost = d_{i,j}
    for sid_i in V_d.keys():
        for sid_j in V_i.keys():
            node_i = f"d_{sid_i}"
            node_j = f"i_{sid_j}"
            dist = _structural_edit_distance(sid_i, sid_j, pkg_t, pkg_tp1)
            # Large capacity to allow any flow amount
            F.add_edge(node_i, node_j, weight=dist, capacity=max(total_supply, total_demand))
    
    # Handle supply/demand imbalance (paper Figures 3-4)
    if total_supply != total_demand:
        imbalance = abs(total_supply - total_demand)
        dummy = "DUMMY"
        
        if total_supply < total_demand:
            # Need extra supply (cost = 1)
            F.add_edge(source, dummy, weight=1, capacity=imbalance)
            for sid_j in V_i.keys():
                node_j = f"i_{sid_j}"
                F.add_edge(dummy, node_j, weight=0, capacity=imbalance)
        else:
            # Need to absorb surplus (cost = 1)
            for sid_i in V_d.keys():
                node_i = f"d_{sid_i}"
                F.add_edge(node_i, dummy, weight=0, capacity=imbalance)
            F.add_edge(dummy, target, weight=1, capacity=imbalance)
    
    # Set demands and solve
    F.nodes[source]['demand'] = -max(total_supply, total_demand)
    F.nodes[target]['demand'] = max(total_supply, total_demand)
    
    flow_cost, flow_dict = nx.network_simplex(F)
    return float(flow_cost)


def _emd_simplified(V_d: Dict[int, int], V_i: Dict[int, int]) -> float:
    """
    Simplified EMD with uniform distance d_{i,j} = 0 if i==j else 1.
    This is the closed-form solution for uniform distance metric.
    """
    total_moved = 0
    
    all_sids = set(V_d.keys()) | set(V_i.keys())
    for sid in all_sids:
        supply = V_d.get(sid, 0)
        demand = V_i.get(sid, 0)
        total_moved += abs(supply - demand)
    
    # Each movement counted twice (leaving source, entering destination)
    return total_moved / 2.0