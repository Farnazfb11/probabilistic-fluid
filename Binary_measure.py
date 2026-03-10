import math
from typing import Dict, Any, Set, Tuple, List, Optional

import networkx as nx

from Unary_measure import (
    _entity_ids,       # V^E(S)
    edge_set,          # structural edges
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
    # entity vertex ids
    V_t = set(_entity_ids(pkg_t))
    V_tp1 = set(_entity_ids(pkg_tp1))

    # unweighted edge sets
    E_t = edge_set(pkg_t, include_weights=False)
    E_tp1 = edge_set(pkg_tp1, include_weights=False)

    # edge edits are just symmetric difference size
    edge_edits = len(E_t ^ E_tp1)

    # incident vertex sets (who appears in at least one edge)
    inc_t = set()
    for u, v in E_t:
        inc_t.add(u)
        inc_t.add(v)

    inc_tp1 = set()
    for u, v in E_tp1:
        inc_tp1.add(u)
        inc_tp1.add(v)

    # isolated class edits like your code
    class_edits = 0

    # created isolated classes: in tp1 but not in t, and has no incident edges in tp1
    for sid in (V_tp1 - V_t):
        if sid not in inc_tp1:
            class_edits += 1

    # deleted isolated classes: in t but not in tp1, and had no incident edges in t
    for sid in (V_t - V_tp1):
        if sid not in inc_t:
            class_edits += 1

    return edge_edits + class_edits


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


# 4) Earth Mover's Distance - ENHANCED with proper flow network - degree-based structural edit costs



def earth_mover_distance_degree(
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
    *,
    deg_t: Optional[Dict[int, int]] = None,
    deg_tp1: Optional[Dict[int, int]] = None,
) -> float:
    """
    CHANGED: accepts optional cached degree maps.
    WHY: avoid recomputing degrees and avoid NetworkX graph reconstruction.
    """
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)

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

    if not V_d and not V_i:
        return 0.0

    try:
        return _emd_min_cost_flow(V_d, V_i, pkg_t, pkg_tp1, deg_t=deg_t, deg_tp1=deg_tp1)
    except (nx.NetworkXError, nx.NetworkXUnfeasible, Exception):
        return _emd_simplified(V_d, V_i)

def _degree_map_weighted_from_pkg(pkg: Dict[str, Any]) -> Dict[int, int]:
    """
    Compute total weighted degree per sid directly from pkg['edges'].

    total degree = in(count) + out(count)
    Works with your JSON where edges is:
      "edges": { "src->dst": {"src":..., "dst":..., "count":..., ...}, ... }
    """
    deg: Dict[int, int] = {}

    # Prefill nodes so isolated nodes exist too
    nodes = pkg.get("nodes") or {}
    for sid in nodes.keys():
        deg[int(sid)] = 0

    edges = pkg.get("edges") or {}
    if not edges:
        raise ValueError(
            "EMD requires pkg['edges'], but it's missing/empty. "
            "Rebuild summaries with include_edges=True."
        )

    for e in edges.values():  # IMPORTANT: values()
        u = int(e["src"])
        v = int(e["dst"])
        w = int(e.get("count", 1))
        deg[u] = deg.get(u, 0) + w
        deg[v] = deg.get(v, 0) + w

    return deg


def _structural_edit_distance_fast(
    sid_i: int,
    sid_j: int,
    deg_t: Dict[int, int],
    deg_tp1: Dict[int, int],
) -> float:
    # same behavior as your old one:
    # abs(deg_i - deg_j) + 1 unless same sid
    if sid_i == sid_j:
        return 0.0
    di = deg_t.get(sid_i, 0)
    dj = deg_tp1.get(sid_j, 0)
    return float(abs(di - dj) + 1)


def _emd_min_cost_flow(
    V_d: Dict[int, int],
    V_i: Dict[int, int],
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
    *,
    deg_t: Optional[Dict[int, int]] = None,
    deg_tp1: Optional[Dict[int, int]] = None,
) -> float:
    """
    CHANGED: do not reconstruct NetworkX graphs when degrees can be computed from edges.
    WHY: speed.
    """

    if deg_t is None:
        deg_t = _degree_map_weighted_from_pkg(pkg_t)  # CHANGED
    if deg_tp1 is None:
        deg_tp1 = _degree_map_weighted_from_pkg(pkg_tp1)  # CHANGED

    total_supply = int(sum(V_d.values()))
    total_demand = int(sum(V_i.values()))

    F = nx.DiGraph()
    source = "SOURCE"
    target = "TARGET"

    F.add_node(source, demand=-total_supply)
    F.add_node(target, demand=total_demand)

    # if imbalance exists, we add one dummy node to balance
    # and keep total supply == total demand in the flow problem
    if total_supply < total_demand:
        # add dummy supply
        dummy_supply = "DUMMY_SUPPLY"
        missing = total_demand - total_supply
        F.add_node(dummy_supply, demand=-missing)
        # send dummy supply to target with cost 1 per unit (paper)
        F.add_edge(dummy_supply, target, weight=1, capacity=missing)
    elif total_demand < total_supply:
        # add dummy demand
        dummy_demand = "DUMMY_DEMAND"
        extra = total_supply - total_demand
        F.add_node(dummy_demand, demand=extra)
        # send supply to dummy demand with cost 1 per unit (paper)
        F.add_edge(source, dummy_demand, weight=1, capacity=extra)

    # add supply nodes and edges from source
    for sid, supply in V_d.items():
        node_d = f"d_{sid}"
        F.add_node(node_d, demand=-int(supply))
        F.add_edge(source, node_d, weight=0, capacity=int(supply))

    # add demand nodes and edges to target
    for sid, demand in V_i.items():
        node_i = f"i_{sid}"
        F.add_node(node_i, demand=int(demand))
        F.add_edge(node_i, target, weight=0, capacity=int(demand))

    # connect V_d to V_i with costs
    # capacity can be total_supply, enough for any distribution
    cap = max(1, total_supply)
    for sid_i in V_d.keys():
        node_i = f"d_{sid_i}"
        for sid_j in V_i.keys():
            node_j = f"i_{sid_j}"
            dist = _structural_edit_distance_fast(sid_i, sid_j, deg_t, deg_tp1)
            F.add_edge(node_i, node_j, weight=dist, capacity=cap)

    flow_cost, _ = nx.network_simplex(F)
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

# 5) signiture based EMD
def _vcp_map(pkg: Dict[str, Any]) -> Dict[int, int]:
    """
    sid -> vcp (payload size)
    """
    nodes = pkg.get("nodes") or {}
    return {int(sid): int(node.get("vcp", 0)) for sid, node in nodes.items()}


def _safe_set(x: Any) -> Set[str]:
    """
    Convert JSON field to a clean set of strings.
    """
    if not x:
        return set()
    if isinstance(x, set):
        return {str(v) for v in x}
    if isinstance(x, (list, tuple)):
        return {str(v) for v in x}
    return {str(x)}


def _pred_to_dst_types_set(node: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """
    Convert
      pred_to_dst_types = { pred: [dst1, dst2, ...], ... }
    into a set of pairs:
      { (pred, dst1), (pred, dst2), ... }

    This lets us compare pred_to_dst_types like a normal set.
    """
    p2d = node.get("pred_to_dst_types") or {}
    out: Set[Tuple[str, str]] = set()

    if isinstance(p2d, dict):
        for pred, dsts in p2d.items():
            for dst in _safe_set(dsts):
                out.add((str(pred), str(dst)))

    return out


def _node_field_sets(node: Dict[str, Any]) -> Dict[str, Set[Any]]:
    """
    Extract the 3 comparable set-valued fields from one summary node.
    """
    return {
        "predicates": _safe_set(node.get("predicates")),
        "types": _safe_set(node.get("types")),
        "pred_to_dst_types": _pred_to_dst_types_set(node),
    }


def _symdiff_count(A: Set[Any], B: Set[Any]) -> int:
    """
    Difference count between two sets.
    This is the size of the symmetric difference.
    """
    return len(A ^ B)


def summary_field_distance(
    node_i: Dict[str, Any],
    node_j: Dict[str, Any],
) -> float:
    """
    Your requested distance:

    - Compare the corresponding fields:
        predicates
        types
        pred_to_dst_types
    - For each field, distance = |set_i - set_j| interpreted as symmetric difference count
    - If only 1 field is active: d = that one difference count
    - If 2 fields are active: d = average of the 2 difference counts
    - If 3 fields are active: d = average of the 3 difference counts

    "Active" means the field is non-empty in at least one of the two nodes.
    If all 3 are empty on both sides, returns 0.
    """
    fi = _node_field_sets(node_i)
    fj = _node_field_sets(node_j)

    diffs = []

    for key in ("pred_to_dst_types", "predicates", "types"):
        A = fi[key]
        B = fj[key]

        # ignore fields that are empty on both sides
        if not A and not B:
            continue

        diffs.append(_symdiff_count(A, B))

    if not diffs:
        return 0.0

    return sum(diffs) / float(len(diffs))


def _summary_field_distance_scaled(
    node_i: Dict[str, Any],
    node_j: Dict[str, Any],
) -> int:
    """
    Integer scaled version of summary_field_distance for network_simplex.

    Since the average is over 1, 2, or 3 fields, multiplying by 6 makes
    the result integral in all cases.

    scaled_cost = 6 * d
    final EMD = flow_cost / 6
    """
    fi = _node_field_sets(node_i)
    fj = _node_field_sets(node_j)

    diffs = []

    for key in ("pred_to_dst_types", "predicates", "types"):
        A = fi[key]
        B = fj[key]

        if not A and not B:
            continue

        diffs.append(_symdiff_count(A, B))

    if not diffs:
        return 0

    k = len(diffs)         # 1, 2, or 3
    sumdiff = sum(diffs)

    # 6 * average = 6 * sumdiff / k
    return (6 * sumdiff) // k


def _node_map(pkg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    sid -> node_json
    """
    nodes = pkg.get("nodes") or {}
    return {int(sid): node for sid, node in nodes.items()}


def earth_mover_distance_signature(
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
) -> float:
    """
    EMD using summary fields as distance.

    Supply / demand:
      derived from vcp differences, just like your previous EMD

    Distance d(i,j):
      average difference count across active fields among
        predicates
        types
        pred_to_dst_types

    Flow:
      same min-cost-flow idea as before

    IMPORTANT:
      This requires FULL summaries (only_id=False),
      because only_id summaries do not store the needed fields.
    """
    size_t = _vcp_map(pkg_t)
    size_tp1 = _vcp_map(pkg_tp1)

    nodes_t = _node_map(pkg_t)
    nodes_tp1 = _node_map(pkg_tp1)

    # Build supply and demand sets from payload change
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

    if not V_d and not V_i:
        return 0.0

    total_supply = int(sum(V_d.values()))
    total_demand = int(sum(V_i.values()))

    F = nx.DiGraph()
    source = "SOURCE"
    target = "TARGET"

    # ONLY source and target carry node demands
    F.add_node(source, demand=-total_supply)
    F.add_node(target, demand=total_demand)

    # If imbalance exists, add dummy node exactly like before
    if total_supply < total_demand:
        dummy_supply = "DUMMY_SUPPLY"
        missing = total_demand - total_supply
        F.add_node(dummy_supply, demand=-missing)
        F.add_edge(dummy_supply, target, weight=1, capacity=missing)

    elif total_demand < total_supply:
        dummy_demand = "DUMMY_DEMAND"
        extra = total_supply - total_demand
        F.add_node(dummy_demand, demand=extra)
        F.add_edge(source, dummy_demand, weight=1, capacity=extra)

    # source -> supply nodes
    for sid, supply in V_d.items():
        nd = f"d_{sid}"
        F.add_node(nd)  # no demand here
        F.add_edge(source, nd, weight=0, capacity=int(supply))

    # demand nodes -> target
    for sid, demand in V_i.items():
        ni = f"i_{sid}"
        F.add_node(ni)  # no demand here
        F.add_edge(ni, target, weight=0, capacity=int(demand))

    # supply -> demand with summary-field distance cost
    cap = max(1, total_supply)

    for sid_i in V_d.keys():
        nd = f"d_{sid_i}"
        node_i = nodes_t.get(sid_i, {})

        for sid_j in V_i.keys():
            ni = f"i_{sid_j}"
            node_j = nodes_tp1.get(sid_j, {})

            scaled_cost = _summary_field_distance_scaled(node_i, node_j)
            F.add_edge(nd, ni, weight=scaled_cost, capacity=cap)

    flow_cost, _ = nx.network_simplex(F)

    # undo the x6 scaling
    return float(flow_cost) / 6.0