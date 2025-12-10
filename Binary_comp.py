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
    # δ^{(2)}_{GED}(S_t, S_{t+1})

    # We approximate the graph edit distance over V^E and E(S) by counting:\
    #   - node insertions / deletions
    #   - edge insertions / deletions

    # GED_structural =
    #   |V^E(S_t) \ V^E(S_{t+1})|   +  # nodes deleted
    #   |V^E(S_{t+1}) \ V^E(S_t)|   +  # nodes inserted
    #   |E(S_t) \ E(S_{t+1})|       +  # edges deleted
    #   |E(S_{t+1}) \ E(S_t)|          # edges inserted

    V_t: Set[int] = _entity_ids(pkg_t)
    V_tp1: Set[int] = _entity_ids(pkg_tp1)

    E_t: Set[Tuple[int, int]] = edge_set(pkg_t, include_weights=False)
    E_tp1: Set[Tuple[int, int]] = edge_set(pkg_tp1, include_weights=False)

    node_deletions = len(V_t - V_tp1)
    node_insertions = len(V_tp1 - V_t)

    edge_deletions = len(E_t - E_tp1)
    edge_insertions = len(E_tp1 - E_t)

    return node_deletions + node_insertions + edge_deletions + edge_insertions

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
    # Symmetric Earth Mover's Distance δ^{(2)}_{EMD}(S_t, S_{t+1})
    # over EQC payloads, as in the paper.

    # Intuition:
    #   - We look at how EQC sizes (payloads) change between t and t+1.
    #   - Classes whose payload decreased form V_d (supply).
    #   - Classes whose payload increased form V_i (demand).
    #   - Moving |Δ| instances from eq_i at t to eq_j at t+1 costs d_ij,
    #     where d_ij is a structural distance between the two classes.
    #   - We then solve a min-cost-flow problem and return sum(d_ij * f_ij).

    # Implementation details:
    #   - Payloads come from vcp or membership.
    #   - Distance d_ij is |deg_t(eq_i) - deg_{t+1}(eq_j)| using weighted degrees.
    #   - If total supply != total demand, we introduce a dummy node with cost 1
    #     to absorb surplus / create missing mass (as in the paper’s “extra nodes”
    #     idea).
    
    # Build size maps at t and t+1 
    size_t = _eqc_size_map(pkg_t)      # sid -> |eq| at time t
    size_tp1 = _eqc_size_map(pkg_tp1)  # sid -> |eq| at time t+1

    # Compute deltas: which classes grew (V_i) or shrank (V_d)?
    V_d: Dict[int, int] = {}  # payload decreased here (supply)
    V_i: Dict[int, int] = {}  # payload increased here (demand)

    all_sids = set(size_t.keys()) | set(size_tp1.keys())
    for sid in all_sids:
        s_t = size_t.get(sid, 0)
        s_tp1 = size_tp1.get(sid, 0)
        delta = s_tp1 - s_t
        if delta > 0:
            V_i[sid] = delta
        elif delta < 0:
            V_d[sid] = -delta

    # No change in payload → zero cost.
    if not V_d and not V_i:
        return 0.0

    # 2. Structural distances via degrees in both summaries
    G_t = _build_digraph(pkg_t)
    G_tp1 = _build_digraph(pkg_tp1)

    def _deg_t(sid: int) -> int:
        if sid not in G_t:
            return 0
        return int(G_t.out_degree(sid, weight="weight") + G_t.in_degree(sid, weight="weight"))

    def _deg_tp1(sid: int) -> int:
        if sid not in G_tp1:
            return 0
        return int(G_tp1.out_degree(sid, weight="weight") + G_tp1.in_degree(sid, weight="weight"))

    # 3. Build min-cost-flow network 
    F = nx.DiGraph()

    # Supply nodes (V_d): negative demand
    for sid, supply in V_d.items():
        node = ("d", sid)
        F.add_node(node, demand=-int(supply))

    # Demand nodes (V_i): positive demand
    for sid, demand in V_i.items():
        node = ("i", sid)
        F.add_node(node, demand=int(demand))

    total_supply = sum(V_d.values())
    total_demand = sum(V_i.values())

    # Edges between shrinking and growing classes
    # capacity is set high enough; cost = structural distance d_ij
    BIG_CAP = 10**12
    for d_sid, supply in V_d.items():
        d_node = ("d", d_sid)
        deg_d = _deg_t(d_sid)
        for i_sid, demand in V_i.items():
            i_node = ("i", i_sid)
            deg_i = _deg_tp1(i_sid)
            dist = abs(deg_d - deg_i)
            F.add_edge(d_node, i_node, weight=float(dist), capacity=BIG_CAP)

    # 4. Balance total supply/demand with a dummy node (cost 1 per unit) 
    if total_supply > total_demand:
        # extra demand to absorb surplus at cost 1
        extra = ("dummy_demand", 0)
        F.add_node(extra, demand=int(total_supply - total_demand))
        for d_sid, supply in V_d.items():
            d_node = ("d", d_sid)
            F.add_edge(d_node, extra, weight=1.0, capacity=BIG_CAP)
    elif total_demand > total_supply:
        # extra supply to create missing mass at cost 1
        extra = ("dummy_supply", 0)
        F.add_node(extra, demand=-int(total_demand - total_supply))
        for i_sid, demand in V_i.items():
            i_node = ("i", i_sid)
            F.add_edge(extra, i_node, weight=1.0, capacity=BIG_CAP)

    # Sanity: after adding dummy, total demands should sum to 0
    # (network_simplex requires this).
    total_node_demand = sum(d for _, d in F.nodes(data="demand"))
    if total_node_demand != 0:
        # Fallback: if something is still off, just return 0 to avoid crashes.
        # This should not normally happen.
        return 0.0

    # 5. Solve min-cost-flow: δ^{(2)}_{EMD}(S_t, S_{t+1}) = Σ d_ij f_ij 
    cost, _ = nx.network_simplex(F)
    return float(cost)
