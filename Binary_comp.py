from __future__ import annotations

import math
from typing import Dict, Any, Set, Tuple, List

from Unary_comp import (
    _entity_ids,          # NEW: reuse your V^E(S) helper
    edge_set,             # NEW: reuse structural edge extraction
    eqc_size_list,        # NEW: reuse EQC sizes
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
    """
    Turn sid -> size into a probability distribution P(eq).

    P(eq) = |eq| / sum_{eq'} |eq'|   over V^E(S).
    """
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
