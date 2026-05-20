import math
from typing import Dict, Any, Set, Tuple, List, Optional
from typing import FrozenSet
import networkx as nx
from collections import defaultdict

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
    deg_t: Optional[Dict[int, float]] = None,
    deg_tp1: Optional[Dict[int, float]] = None,
) -> float:
    """
    CHANGED:
    Replaced the generic NetworkX min-cost-flow formulation with an exact
    1D transport solver over degree buckets.

    WHY:
    Your transport cost is degree-based:
        cost(i, j) = abs(deg_t[i] - deg_tp1[j]) + 1
    so the problem depends only on degree values, not on the full bipartite
    sid-to-sid graph. That means we can solve it exactly by sorting degree
    buckets and matching mass greedily, which is much faster and uses far less memory.
    """
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)

    # ---------------- CHANGED ----------------
    # WHY CHANGED:
    # Default to log-weighted out-degree to avoid scale explosion.
    if deg_t is None:
        deg_t = _degree_map_weighted_from_pkg(pkg_t, mode="log_weighted_out")
    if deg_tp1 is None:
        deg_tp1 = _degree_map_weighted_from_pkg(pkg_tp1, mode="log_weighted_out")

    # ---------------- CHANGED ----------------
    # WHY CHANGED:
    # Degree buckets may now be float values after log1p compression.
    supply_by_degree: Dict[float, int] = defaultdict(int)
    demand_by_degree: Dict[float, int] = defaultdict(int)

    all_sids = set(size_t.keys()) | set(size_tp1.keys())

    for sid in all_sids:
        s_t = size_t.get(sid, 0)
        s_tp1 = size_tp1.get(sid, 0)
        delta = s_tp1 - s_t

        # ---------------- CHANGED ----------------
        # WHY CHANGED:
        # Keep degree values as floats. Casting to int would throw away the
        # effect of log compression and make many classes collapse into the same bucket.
        if delta < 0:
            d = float(deg_t.get(sid, 0.0))
            supply_by_degree[d] += -delta

        elif delta > 0:
            d = float(deg_tp1.get(sid, 0.0))
            demand_by_degree[d] += delta

    if not supply_by_degree and not demand_by_degree:
        return 0.0

    return _emd_degree_1d_exact(supply_by_degree, demand_by_degree)


def _emd_degree_1d_exact(
    supply_by_degree: Dict[float, int],
    demand_by_degree: Dict[float, int],
) -> float:
    """
    CHANGED:
    Exact EMD solver specialized for degree-based cost.

    WHY:
    For cost = abs(d1 - d2) + 1, the optimal transport between weighted points
    on a line can be solved exactly by sorting degree buckets and matching them
    with a two-pointer sweep.

    NOTES:
    - The +1 is charged per unit of real transported mass.
    - If total supply != total demand, leftover mass is matched to an implicit
      dummy with cost 1 per unit, matching your previous imbalance handling.
    """
    supplies: List[List[int]] = [[deg, mass] for deg, mass in sorted(supply_by_degree.items()) if mass > 0]
    demands: List[List[int]] = [[deg, mass] for deg, mass in sorted(demand_by_degree.items()) if mass > 0]

    i = 0
    j = 0
    cost = 0.0

    while i < len(supplies) and j < len(demands):
        deg_s, mass_s = supplies[i]
        deg_d, mass_d = demands[j]

        moved = min(mass_s, mass_d)

        # CHANGED:
        # real transport cost for one unit is abs(deg_s - deg_d) + 1
        # WHY:
        # preserves your original structural edit cost model exactly
        cost += moved * (abs(deg_s - deg_d) + 1)

        supplies[i][1] -= moved
        demands[j][1] -= moved

        if supplies[i][1] == 0:
            i += 1
        if demands[j][1] == 0:
            j += 1

    # CHANGED:
    # leftover unmatched mass goes to dummy at cost 1 per unit
    # WHY:
    # matches the logic you used before for supply-demand imbalance
    leftover_supply = sum(mass for _, mass in supplies[i:])
    leftover_demand = sum(mass for _, mass in demands[j:])
    cost += abs(leftover_supply - leftover_demand) * 1.0

    return float(cost)

def _degree_map_weighted_from_pkg(
    pkg: Dict[str, Any],
    *,
    mode: str = "log_weighted_out",
) -> Dict[int, float]:
    """
    CHANGED:
    Degree map now supports multiple degree variants and defaults to
    log-weighted OUT-degree.

    WHY CHANGED:
    Your previous version used weighted in-degree + weighted out-degree with
    raw rolled edge counts. That can explode badly and dominate the EMD scale.
    The new default keeps edge weights, but compresses them with log1p and only
    uses outgoing superedges, which is much more stable and easier to interpret.

    Supported modes:
    - "weighted_total":
        raw weighted in-degree + out-degree
    - "weighted_out":
        raw weighted out-degree only
    - "unweighted_out":
        number of distinct outgoing superedges
    - "log_weighted_out":   [DEFAULT]
        log(1 + weighted out-degree)
    - "log_weighted_total":
        log(1 + weighted total degree)
    """
    nodes = pkg.get("nodes") or {}
    deg: Dict[int, float] = {int(sid): 0.0 for sid in nodes}

    edges = pkg.get("edges")
    if not edges:
        raise ValueError(
            "EMD requires pkg['edges'], but it's missing or empty. "
            "Rebuild summaries with include_edges=True."
        )

    # ---------------- CHANGED ----------------
    # WHY CHANGED:
    # Compute several degree notions from the same summary package so you can
    # compare them without rewriting the metric again.
    for e in edges.values():
        u = int(e["src"])
        v = int(e["dst"])
        w = float(e.get("count", 1))

        if mode == "weighted_total":
            deg[u] = deg.get(u, 0.0) + w
            deg[v] = deg.get(v, 0.0) + w

        elif mode == "weighted_out":
            deg[u] = deg.get(u, 0.0) + w

        elif mode == "unweighted_out":
            deg[u] = deg.get(u, 0.0) + 1.0

        elif mode == "log_weighted_out":
            # First accumulate weighted out-degree, then compress below
            deg[u] = deg.get(u, 0.0) + w

        elif mode == "log_weighted_total":
            # First accumulate weighted total degree, then compress below
            deg[u] = deg.get(u, 0.0) + w
            deg[v] = deg.get(v, 0.0) + w

        else:
            raise ValueError(
                f"Unknown degree mode '{mode}'. "
                f"Choose from weighted_total, weighted_out, unweighted_out, "
                f"log_weighted_out, log_weighted_total."
            )

    # ---------------- CHANGED ----------------
    # WHY CHANGED:
    # Apply logarithmic compression only after accumulation.
    if mode in ("log_weighted_out", "log_weighted_total"):
        deg = {sid: math.log1p(val) for sid, val in deg.items()}

    return deg

# 5) signiture based EMD
EMPTY_SIG = {
    "predicates": frozenset(),
    "types": frozenset(),
    "pred_to_dst_types": frozenset(),
}


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
    """
    p2d = node.get("pred_to_dst_types") or {}
    out: Set[Tuple[str, str]] = set()

    if isinstance(p2d, dict):
        for pred, dsts in p2d.items():
            for dst in _safe_set(dsts):
                out.add((str(pred), str(dst)))

    return out


def _build_signature_sets(
    node: Dict[str, Any],
) -> Dict[str, FrozenSet[Any]]:
    """
    CHANGED: build normalized set-valued signature fields once.
    WHY: avoid rebuilding the same sets in every pairwise comparison.
    """
    return {
        "predicates": frozenset(_safe_set(node.get("predicates"))),
        "types": frozenset(_safe_set(node.get("types"))),
        "pred_to_dst_types": frozenset(_pred_to_dst_types_set(node)),
    }


def _build_signature_key(
    sig_sets: Dict[str, FrozenSet[Any]],
) -> Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]]:
    """
    CHANGED: canonical immutable key for grouping equal signatures.
    WHY: lets us aggregate flow by signature instead of by sid.
    """
    return (
        sig_sets["predicates"],
        sig_sets["types"],
        sig_sets["pred_to_dst_types"],
    )


def _summary_field_distance_scaled_cached(
    sig_i: Dict[str, FrozenSet[Any]],
    sig_j: Dict[str, FrozenSet[Any]],
) -> int:
    """
    Integer scaled version of the summary field distance.

    Same logic as before:
    - compare predicates, types, pred_to_dst_types
    - ignore fields empty on both sides
    - average over active fields
    - multiply by 6 so result is integral

    CHANGED: accepts already-built signature sets.
    WHY: avoid repeated JSON -> set conversion inside the O(|V_d|*|V_i|) loop.
    """
    diffs = []

    for key in ("pred_to_dst_types", "predicates", "types"):
        A = sig_i[key]
        B = sig_j[key]

        if not A and not B:
            continue

        diffs.append(len(A ^ B))

    if not diffs:
        return 0

    return (6 * sum(diffs)) // len(diffs)


def earth_mover_distance_signature(
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
) -> float:
    """
    Faster signature-based EMD.

    SAME METRIC IDEA:
    - supply / demand comes from vcp differences
    - transport cost comes from average fieldwise signature difference
    - min cost flow still computes the optimal transport

    CHANGED:
    1) precompute signatures once
    2) aggregate by signature
    3) cancel zero-cost identical-signature transport before flow
    4) cache pairwise signature costs

    WHY:
    The old version built a dense sid-to-sid flow problem and recomputed
    the same signature fields many times. This version shrinks the flow
    graph to unique changed signatures only.
    """
    size_t = _vcp_map(pkg_t)
    size_tp1 = _vcp_map(pkg_tp1)

    raw_nodes_t = pkg_t.get("nodes") or {}
    raw_nodes_tp1 = pkg_tp1.get("nodes") or {}

    # CHANGED: precompute normalized signature sets and immutable keys once
    # WHY: repeated extraction inside the nested loop was costly
    sig_sets_t: Dict[int, Dict[str, FrozenSet[Any]]] = {}
    sig_key_t: Dict[int, Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]]] = {}

    for sid_str, node in raw_nodes_t.items():
        sid = int(sid_str)
        sig = _build_signature_sets(node)
        sig_sets_t[sid] = sig
        sig_key_t[sid] = _build_signature_key(sig)

    sig_sets_tp1: Dict[int, Dict[str, FrozenSet[Any]]] = {}
    sig_key_tp1: Dict[int, Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]]] = {}

    for sid_str, node in raw_nodes_tp1.items():
        sid = int(sid_str)
        sig = _build_signature_sets(node)
        sig_sets_tp1[sid] = sig
        sig_key_tp1[sid] = _build_signature_key(sig)

    # CHANGED: aggregate changed mass by signature key instead of sid
    # WHY: many sids may share the exact same signature, so they are
    # indistinguishable for this transport cost
    supply_by_sig: Dict[
        Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]], int
    ] = {}
    demand_by_sig: Dict[
        Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]], int
    ] = {}

    # Representative signature-sets per key, used for cost computation
    rep_sig_t: Dict[
        Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]],
        Dict[str, FrozenSet[Any]]
    ] = {}
    rep_sig_tp1: Dict[
        Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]],
        Dict[str, FrozenSet[Any]]
    ] = {}

    all_sids = set(size_t.keys()) | set(size_tp1.keys())

    for sid in all_sids:
        s_t = size_t.get(sid, 0)
        s_tp1 = size_tp1.get(sid, 0)
        delta = s_tp1 - s_t

        if delta < 0:
            key_t = sig_key_t.get(sid)
            if key_t is None:
                key_t = _build_signature_key(EMPTY_SIG)
                rep_sig_t[key_t] = EMPTY_SIG
            else:
                rep_sig_t[key_t] = sig_sets_t[sid]

            supply_by_sig[key_t] = supply_by_sig.get(key_t, 0) + (-delta)

        elif delta > 0:
            key_tp1 = sig_key_tp1.get(sid)
            if key_tp1 is None:
                key_tp1 = _build_signature_key(EMPTY_SIG)
                rep_sig_tp1[key_tp1] = EMPTY_SIG
            else:
                rep_sig_tp1[key_tp1] = sig_sets_tp1[sid]

            demand_by_sig[key_tp1] = demand_by_sig.get(key_tp1, 0) + delta

    if not supply_by_sig and not demand_by_sig:
        return 0.0

    # CHANGED: cancel identical signatures first
    # WHY: transport between exactly equal signatures has cost 0, so we
    # should remove that mass before building the expensive flow graph
    common_keys = set(supply_by_sig.keys()) & set(demand_by_sig.keys())
    for key in list(common_keys):
        matched = min(supply_by_sig[key], demand_by_sig[key])

        if matched > 0:
            supply_by_sig[key] -= matched
            demand_by_sig[key] -= matched

            if supply_by_sig[key] == 0:
                del supply_by_sig[key]
            if demand_by_sig[key] == 0:
                del demand_by_sig[key]

    if not supply_by_sig and not demand_by_sig:
        return 0.0

    total_supply = int(sum(supply_by_sig.values()))
    total_demand = int(sum(demand_by_sig.values()))

    F = nx.DiGraph()
    source = "SOURCE"
    target = "TARGET"

    F.add_node(source, demand=-total_supply)
    F.add_node(target, demand=total_demand)

    # Keep same imbalance handling logic, but scaled consistently by 6
    if total_supply < total_demand:
        dummy_supply = "DUMMY_SUPPLY"
        missing = total_demand - total_supply
        F.add_node(dummy_supply, demand=-missing)

        # CHANGED: weight=6 instead of 1
        # WHY: all real transport costs are scaled by 6, so dummy penalty
        # should live on the same scale before dividing final cost by 6
        F.add_edge(dummy_supply, target, weight=6, capacity=missing)

    elif total_demand < total_supply:
        dummy_demand = "DUMMY_DEMAND"
        extra = total_supply - total_demand
        F.add_node(dummy_demand, demand=extra)

        # CHANGED: weight=6 instead of 1
        # WHY: same reason as above
        F.add_edge(source, dummy_demand, weight=6, capacity=extra)

    supply_keys = list(supply_by_sig.keys())
    demand_keys = list(demand_by_sig.keys())

    # source -> supply nodes
    for idx, sig_key in enumerate(supply_keys):
        nd = f"d_{idx}"
        F.add_node(nd)
        F.add_edge(source, nd, weight=0, capacity=int(supply_by_sig[sig_key]))

    # demand nodes -> target
    for idx, sig_key in enumerate(demand_keys):
        ni = f"i_{idx}"
        F.add_node(ni)
        F.add_edge(ni, target, weight=0, capacity=int(demand_by_sig[sig_key]))

    # CHANGED: much smaller bipartite graph, now unique-signature to unique-signature
    # WHY: this is the main runtime win
    cap = max(1, min(total_supply, total_demand))

    # CHANGED: pairwise cost cache
    # WHY: compute each unique signature-pair distance only once
    cost_cache: Dict[
        Tuple[
            Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]],
            Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]],
        ],
        int
    ] = {}

    for i, skey in enumerate(supply_keys):
        nd = f"d_{i}"
        sig_i = rep_sig_t.get(skey, EMPTY_SIG)

        for j, dkey in enumerate(demand_keys):
            ni = f"i_{j}"

            pair = (skey, dkey)
            if pair not in cost_cache:
                sig_j = rep_sig_tp1.get(dkey, EMPTY_SIG)
                cost_cache[pair] = _summary_field_distance_scaled_cached(sig_i, sig_j)

            F.add_edge(nd, ni, weight=cost_cache[pair], capacity=cap)

    flow_cost, _ = nx.network_simplex(F)

    # undo x6 scaling
    return float(flow_cost) / 6.0

    # 6 Sid-degree EMD
def sid_overlap_stats(
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
) -> Dict[str, int]:
        
    """
    Report exact-SID overlap statistics between two summaries.

    Returns:
        - sids_t: number of SIDs in S_t
        - sids_tp1: number of SIDs in S_{t+1}
        - shared_sids: number of SIDs appearing in both snapshots
        - unchanged_payload: payload mass preserved exactly by shared SIDs
        - residual_supply: payload mass left on the t side after exact SID cancellation
        - residual_demand: payload mass left on the t+1 side after exact SID cancellation
    """

    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)

    shared = set(size_t.keys()) & set(size_tp1.keys())

    unchanged_payload = 0
    for sid in shared:
        unchanged_payload += min(size_t[sid], size_tp1[sid])

    total_t = sum(size_t.values())
    total_tp1 = sum(size_tp1.values())

    residual_supply = total_t - unchanged_payload
    residual_demand = total_tp1 - unchanged_payload

    return {
        "sids_t": len(size_t),
        "sids_tp1": len(size_tp1),
        "shared_sids": len(shared),
        "unchanged_payload": int(unchanged_payload),
        "residual_supply": int(residual_supply),
        "residual_demand": int(residual_demand),
    }

def earth_mover_distance_sid_then_degree(
    pkg_t: Dict[str, Any],
    pkg_tp1: Dict[str, Any],
    *,
    deg_t: Optional[Dict[int, float]] = None,
    deg_tp1: Optional[Dict[int, float]] = None,
) -> float:
    """
    Hybrid EMD:
    1) First, match payload with identical SIDs exactly at zero cost.
    2) Then, transport the remaining unmatched payload between EQCs using
       degree-based cost.

    WHY THIS:
    Considers weighted degree-based structural edits, to preserve structural strength
    stable cost, meaning fallback costs. 
    This preserves exact SID overlap whenever possible, and only uses
    degree-based fallback transport for the residual unmatched mass.
    """
    size_t = _eqc_size_map(pkg_t)
    size_tp1 = _eqc_size_map(pkg_tp1)

    if deg_t is None:
        deg_t = _degree_map_weighted_from_pkg(pkg_t, mode="log_weighted_out")
    if deg_tp1 is None:
        deg_tp1 = _degree_map_weighted_from_pkg(pkg_tp1, mode="log_weighted_out")

    # CHANGED: cancel exact shared SID payload first
    # WHY CHANGED: identical SIDs should preserve common mass at zero cost
    residual_t: Dict[int, int] = {}
    residual_tp1: Dict[int, int] = {}

    all_sids = set(size_t.keys()) | set(size_tp1.keys())
    for sid in all_sids:
        a = size_t.get(sid, 0)
        b = size_tp1.get(sid, 0)
        common = min(a, b)

        rt = a - common
        rp = b - common

        if rt > 0:
            residual_t[sid] = rt
        if rp > 0:
            residual_tp1[sid] = rp

    V_d: Dict[int, int] = {}
    V_i: Dict[int, int] = {}

    all_residual_sids = set(residual_t.keys()) | set(residual_tp1.keys())
    for sid in all_residual_sids:
        a = residual_t.get(sid, 0)
        b = residual_tp1.get(sid, 0)
        delta = b - a

        if delta > 0:
            V_i[sid] = delta
        elif delta < 0:
            V_d[sid] = -delta

    if not V_d and not V_i:
        return 0.0

    supply_by_degree: Dict[float, int] = defaultdict(int)
    demand_by_degree: Dict[float, int] = defaultdict(int)

    for sid, mass in V_d.items():
        supply_by_degree[float(deg_t.get(sid, 0.0))] += mass

    for sid, mass in V_i.items():
        demand_by_degree[float(deg_tp1.get(sid, 0.0))] += mass

    return _emd_degree_1d_exact(supply_by_degree, demand_by_degree)

def _emd_min_cost_flow_degree_nodes(
    V_d: Dict[int, int],
    V_i: Dict[int, int],
    deg_t: Dict[int, float],
    deg_tp1: Dict[int, float],
) -> float:
    """
    Node-level min-cost flow over residual EQCs.

    Cost:
        0 if same SID
        abs(deg_t[i] - deg_tp1[j]) + 1 otherwise

    NOTE:
    After exact SID overlap cancellation, same-SID transport is usually no longer
    needed, but we keep the rule for completeness.
    """
    F = nx.DiGraph()

    total_supply = int(sum(V_d.values()))
    total_demand = int(sum(V_i.values()))

    source = "SOURCE"
    target = "TARGET"

    F.add_node(source, demand=-total_supply)
    F.add_node(target, demand=total_demand)

    if total_supply < total_demand:
        dummy_supply = "DUMMY_SUPPLY"
        missing = total_demand - total_supply
        F.add_node(dummy_supply, demand=-missing)
        F.add_edge(dummy_supply, target, weight=1000, capacity=missing)

    elif total_demand < total_supply:
        dummy_demand = "DUMMY_DEMAND"
        extra = total_supply - total_demand
        F.add_node(dummy_demand, demand=extra)
        F.add_edge(source, dummy_demand, weight=1000, capacity=extra)

    # source -> supply nodes
    for sid, mass in V_d.items():
        nd = f"d_{sid}"
        F.add_node(nd)
        F.add_edge(source, nd, weight=0, capacity=int(mass))

    # demand nodes -> target
    for sid, mass in V_i.items():
        ni = f"i_{sid}"
        F.add_node(ni)
        F.add_edge(ni, target, weight=0, capacity=int(mass))

    cap = max(1, min(total_supply, total_demand)) if total_supply and total_demand else 1

    for sid_i in V_d.keys():
        nd = f"d_{sid_i}"
        di = float(deg_t.get(sid_i, 0.0))

        for sid_j in V_i.keys():
            ni = f"i_{sid_j}"
            dj = float(deg_tp1.get(sid_j, 0.0))

            if sid_i == sid_j:
                cost = 0.0
            else:
                cost = abs(di - dj) + 1.0

            # integer weights for network_simplex
            weight = int(round(cost * 1000))
            F.add_edge(nd, ni, weight=weight, capacity=cap)

    flow_cost, _ = nx.network_simplex(F)
    return float(flow_cost) / 1000.0