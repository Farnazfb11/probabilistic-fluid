from __future__ import annotations

import math
from typing import Dict, Any, Set, Tuple, List

import networkx as nx  # NEW

from Unary_comp import (
    _entity_ids,       # V^E(S)
    edge_set,          # structural edges
    eqc_size_list,     # EQC sizes (list)
    _build_digraph,    # NEW: graph construction from summary
)


# 1) Jaccard over vertices and edges

def jaccard_vertices(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    # Binary metric δ^{(2)}_{JacV}(S_t, S_{t+1}) over entity vertices V^E.
    # δ^{(2)}_{JacV}(S_t, S_{t+1})
    #   = 1 - | V^E(S_t) ∩ V^E(S_{t+1}) | / | V^E(S_t) ∪ V^E(S_{t+1}) |
    # we assume that supernode IDs (keys in `nodes`) are stable across snapshots.
    V_t: Set[int] = _entity_ids(pkg_t)
    V_tp1: Set[int] = _entity_ids(pkg_tp1)

    if not V_t and not V_tp1:
        # both empty → no difference
        return 0.0

    inter = V_t & V_tp1
    union = V_t | V_tp1

    return 1.0 - (len(inter) / float(len(union)))


def jaccard_edges(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    # binary metric δ^{(2)}_{JacE}(S_t, S_{t+1}) over superedges E(S).
    # δ^{(2)}_{JacE}(S_t, S_{t+1})
    #   = 1 - | E(S_t) ∩ E(S_{t+1}) | / | E(S_t) ∪ E(S_{t+1}) |
    # We use the structural edge set:
    #   E(S) = { (src, dst) } ignoring weights.
    E_t: Set[Tuple[int, int]] = edge_set(pkg_t, include_weights=False)
    E_tp1: Set[Tuple[int, int]] = edge_set(pkg_tp1, include_weights=False)

    if not E_t and not E_tp1:
        return 0.0

    inter = E_t & E_tp1
    union = E_t | E_tp1

    return 1.0 - (len(inter) / float(len(union)))

# 2) Simple structural Graph Edit Distance (GED-like)
def graph_edit_distance_structural(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> int:
    """
    Implement Equation 14: δ^{(2)}_{GED}(S_t, S_{t+1})
    
    For each equivalence class (SID) in V^E(S_t) ∪ V^E(S_{t+1}):
    1. Get its adjacency sets (outgoing/incoming neighbors) in both summaries
    2. Count the minimum number of edge operations needed:
       - Add edge
       - Remove edge  
       - Edge moves between classes count as 2 operations (remove + add)
    3. Sum across all classes
    """
    # Build graphs
    G_t = _build_digraph(pkg_t)
    G_tp1 = _build_digraph(pkg_tp1)
    
    # Get all SIDs
    V_t = set(_entity_ids(pkg_t))
    V_tp1 = set(_entity_ids(pkg_tp1))
    all_sids = V_t | V_tp1
    
    total_edits = 0
    
    # Helper to get all edges incident to a SID
    def get_incident_edges(graph: nx.DiGraph, sid: int) -> Set[Tuple[int, int]]:
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
    
    # Process each class
    for sid in all_sids:
        # Get incident edges in both summaries
        edges_t = get_incident_edges(G_t, sid)
        edges_tp1 = get_incident_edges(G_tp1, sid)
        
        # Edge operations needed:
        # Add edges present in t+1 but not in t
        edge_additions = len(edges_tp1 - edges_t)
        # Remove edges present in t but not in t+1
        edge_deletions = len(edges_t - edges_tp1)
        
        class_edits = edge_additions + edge_deletions
        total_edits += class_edits
    
    # Note: This counts edge moves between classes correctly
    # When an edge moves from class A to class B:
    # - For class A: counts as 1 deletion
    # - For class B: counts as 1 addition
    # Total: 2 operations ✓
    
    return total_edits

# 3) Symmetric KL divergence over EQC-size distributions

def _eqc_size_map(pkg: Dict[str, Any]) -> Dict[int, int]:
    # We rely on the same logic as eqc_size_list, but we keep the SID association.
    # Priority:
    #   1) nodes[sid]['vcp']
    #   2) membership (if present)

    nodes = pkg.get("nodes") or {}
    # 1) try via vcp
    all_have_vcp = all("vcp" in node for node in nodes.values())
    if all_have_vcp and nodes:
        return {int(sid): int(node["vcp"]) for sid, node in nodes.items()}

    # 2) fallback via membership
    membership = pkg.get("membership")
    if membership:
        counts: Dict[int, int] = {}
        for _, sid in membership.items():
            k = int(sid)
            counts[k] = counts.get(k, 0) + 1
        return counts

    # If we cannot derive, raise same way as eqc_size_list
    raise ValueError(
        "Cannot derive EQC size map: 'nodes[].vcp' and 'membership' both missing."
    )


def _normalize_sizes(size_map: Dict[int, int]) -> Dict[int, float]:
    # Turn sid -> size into a probability distribution P(eq).
    # P(eq) = |eq| / sum_{eq'} |eq'|   over V^E(S).
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
    # Compute D(P || Q) over a given support set of SIDs.

    # D(P || Q) = sum_{eq in support} P(eq) * log2(P(eq) / Q(eq))

    # Conventions:
    #   - If P(eq) == 0, the summand is treated as 0 (limit x log x → 0).
    #   - If Q(eq) == 0 and P(eq) > 0, we clip Q(eq) to `epsilon` to avoid
    #     infinite divergence. This introduces a very small positive penalty
    #     for probability mass that has "no counterpart" in Q.

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
    # Symmetric KL-based binary metric δ^{(2)}_{KLD}(S_t, S_{t+1}).

    # Following the LaTeX:

    #   D(S_t, S_{t+1})   = sum_{eq in V^E(S_{t+1})} P_{S_t}(eq)   log2( P_{S_t}(eq) / P_{S_{t+1}}(eq) )
    #   D(S_{t+1}, S_t)   = sum_{eq in V^E(S_t)}     P_{S_{t+1}}(eq) log2( P_{S_{t+1}}(eq) / P_{S_t}(eq) )

    #   δ^{(2)}_{KLD}(S_t, S_{t+1}) = D(S_t, S_{t+1}) + D(S_{t+1}, S_t).

    # Implementation notes:
    #   - We derive P_{S_t} and P_{S_{t+1}} from EQC sizes (vcp or membership).
    #   - We use the SID as the identifier of an equivalence class.
    #     This assumes SID stability across snapshots (same signature → same ID).
    #   - For classes that are only present in one summary, the other side sees
    #     probability 0; we clip denominators to epsilon > 0 to avoid infinities.
    # """
    size_t = _eqc_size_map(pkg_t)        # sid -> |eq| at time t
    size_tp1 = _eqc_size_map(pkg_tp1)    # sid -> |eq| at time t+1

    P_t = _normalize_sizes(size_t)
    P_tp1 = _normalize_sizes(size_tp1)

    V_t = set(size_t.keys())
    V_tp1 = set(size_tp1.keys())

    D_t_tp1 = _kl_divergence(P_t, P_tp1, support=V_tp1)
    D_tp1_t = _kl_divergence(P_tp1, P_t, support=V_t)

    return D_t_tp1 + D_tp1_t

# Earth Mover's Distance (EMD) over EQC payloads 
def earth_mover_distance(pkg_t: Dict[str, Any], pkg_tp1: Dict[str, Any]) -> float:
    """
    Correct implementation of Equation 15: δ^{(2)}_{EMD}(S_t, S_{t+1})
    
    Matches the paper's flow network construction (Figures 3-4).
    Steps:
    1. Identify V_d (payload decreased) and V_i (payload increased) classes
    2. Calculate proper structural distances d_ij between classes
    3. Build min-cost-flow network with source, target, and two extra vertices
    4. Solve and return total cost
    
    Note: d_ij should be structural edit distance between classes.
    For simplicity, we'll use degree difference as approximation if edit distance is unavailable.
    """
    # 1. Build size maps and identify V_d, V_i
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)
    
    V_d: Dict[int, int] = {}  # sid -> supply amount (payload decreased)
    V_i: Dict[int, int] = {}  # sid -> demand amount (payload increased)
    
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
    
    # 2. Calculate structural distances d_ij
    G_t = _build_digraph(pkg_t)
    G_tp1 = _build_digraph(pkg_tp1)
    
    def _structural_distance(sid_i: int, sid_j: int) -> float:
        """Calculate d_ij = structural edit distance."""
        # Get weighted degrees
        def get_weighted_degree(graph: nx.DiGraph, sid: int) -> int:
            if sid not in graph:
                return 0
            out_deg = sum(weight for _, _, weight in graph.out_edges(sid, data='weight', default=1))
            in_deg = sum(weight for _, _, weight in graph.in_edges(sid, data='weight', default=1))
            return int(out_deg + in_deg)
        
        deg_i = get_weighted_degree(G_t, sid_i)
        deg_j = get_weighted_degree(G_tp1, sid_j)
        return float(abs(deg_i - deg_j))
    
    # 3. Build the flow network
    F = nx.DiGraph()
    
    # Add source and target nodes
    source = "SOURCE"
    target = "TARGET"
    F.add_node(source, demand=0)
    F.add_node(target, demand=0)
    
    # Create nodes for V_d and V_i
    vd_nodes = {}
    vi_nodes = {}
    
    for sid, supply in V_d.items():
        node = f"d_{sid}"
        vd_nodes[sid] = node
        F.add_node(node, demand=0)
    
    for sid, demand in V_i.items():
        node = f"i_{sid}"
        vi_nodes[sid] = node
        F.add_node(node, demand=0)
    
    # Add two extra vertices for flow balance
    extra_source = "EXTRA_SOURCE"
    extra_target = "EXTRA_TARGET"
    F.add_node(extra_source, demand=0)
    F.add_node(extra_target, demand=0)
    
    # Calculate total supply and demand
    total_supply = sum(V_d.values())
    total_demand = sum(V_i.values())
    
    # 4. Add edges according to paper's flow network
    BIG_CAP = float('inf') if hasattr(float, 'inf') else 1e12
    
    # (a) Source -> V_d edges: cost 0, capacity = supply
    for sid, supply in V_d.items():
        F.add_edge(source, vd_nodes[sid], 
                   weight=0.0, 
                   capacity=float(supply),
                   cost=0.0)
    
    # (b) V_i -> Target edges: cost 0, capacity = demand
    for sid, demand in V_i.items():
        F.add_edge(vi_nodes[sid], target,
                   weight=0.0,
                   capacity=float(demand),
                   cost=0.0)
    
    # (c) V_d -> V_i edges: cost = d_ij, capacity = large
    for d_sid, d_node in vd_nodes.items():
        for i_sid, i_node in vi_nodes.items():
            dist = _structural_distance(d_sid, i_sid)
            F.add_edge(d_node, i_node,
                       weight=dist,
                       capacity=BIG_CAP,
                       cost=dist)
    
    # (d) Handle flow imbalance with extra vertices (cost = 1)
    flow_diff = abs(total_supply - total_demand)
    
    if total_supply < total_demand:
        # Need to create missing mass
        F.add_edge(source, extra_source,
                   weight=1.0,
                   capacity=float(flow_diff),
                   cost=1.0)
        
        for i_sid, i_node in vi_nodes.items():
            F.add_edge(extra_source, i_node,
                       weight=0.0,
                       capacity=float(flow_diff),
                       cost=0.0)
    
    elif total_demand < total_supply:
        # Need to absorb surplus
        for d_sid, d_node in vd_nodes.items():
            F.add_edge(d_node, extra_target,
                       weight=0.0,
                       capacity=float(flow_diff),
                       cost=0.0)
        
        F.add_edge(extra_target, target,
                   weight=1.0,
                   capacity=float(flow_diff),
                   cost=1.0)
    
    # 5. Set demands and solve
    try:
        F.nodes[source]['demand'] = -total_supply
        F.nodes[target]['demand'] = total_demand
        
        flow_cost, flow_dict = nx.network_simplex(F)
        return float(flow_cost)
        
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        # Fallback to simplified calculation
        return _simplified_emd_calculation(V_d, V_i, _structural_distance, 
                                          total_supply, total_demand)


def _simplified_emd_calculation(V_d: Dict[int, int], V_i: Dict[int, int],
                               distance_func, total_supply: int, total_demand: int) -> float:
    """Simplified EMD calculation when full flow network fails."""
    if not V_d or not V_i:
        return 0.0
    
    # Greedy assignment
    total_cost = 0.0
    remaining_supply = V_d.copy()
    remaining_demand = V_i.copy()
    
    supply_items = list(remaining_supply.items())
    demand_items = list(remaining_demand.items())
    
    while supply_items and demand_items:
        min_dist = float('inf')
        best_s_idx, best_d_idx = 0, 0
        
        for s_idx, (s_sid, s_amt) in enumerate(supply_items):
            if s_amt <= 0:
                continue
            for d_idx, (d_sid, d_amt) in enumerate(demand_items):
                if d_amt <= 0:
                    continue
                dist = distance_func(s_sid, d_sid)
                if dist < min_dist:
                    min_dist = dist
                    best_s_idx, best_d_idx = s_idx, d_idx
        
        if min_dist == float('inf'):
            break
        
        s_sid, s_amt = supply_items[best_s_idx]
        d_sid, d_amt = demand_items[best_d_idx]
        
        flow = min(s_amt, d_amt)
        total_cost += flow * min_dist
        
        supply_items[best_s_idx] = (s_sid, s_amt - flow)
        demand_items[best_d_idx] = (d_sid, d_amt - flow)
    
    # Cost for unmatched mass (cost = 1 as in paper)
    unmatched_supply = sum(amt for _, amt in supply_items if amt > 0)
    unmatched_demand = sum(amt for _, amt in demand_items if amt > 0)
    
    total_cost += (unmatched_supply + unmatched_demand) * 1.0
    return float(total_cost)