import os, json
from typing import Dict, Any, Union, Type
import networkx as nx

#change this
#  project dependencies ===
from Collector import SchemaCollector
from AttCollection import SEAttributeCollection
from ClassCollection import SEClassCollection
from CompAttClassCollection import SEComplexAttributeClassCollection
from AttClassCollection import SEAttributeClassCollection
from Summarizer import group_by_signature, build_summary_graph

def _sorted_list(x) -> list:
    # Convert any iterable (e.g., set) into a sorted list for deterministic JSON.
    return sorted(list(x))

def _canon_pred2dst(pred2dst: Dict[str, set]) -> Dict[str, list]:
    # {predicate -> set(dst_types)} into
    # {predicate -> sorted list(dst_types)}, with predicates sorted too.
    return {p: _sorted_list(v) for p, v in sorted(pred2dst.items())}

# Registry of strategy names -> classes
_STRATEGY_REGISTRY: Dict[str, Type] = {
    "attribute": SEAttributeCollection,
    "class": SEClassCollection,
    "attributeclass": SEAttributeClassCollection,
    "complexattributeclass": SEComplexAttributeClassCollection,  # formerly "attributeclass"
}

# save summary for a specific graph for a specific strategy
# path + metadata
# save a canonical JSON package (schema + payload + edges + membership)
def save_summary_for_strategy(
    G: nx.MultiDiGraph,
    strategy: Union[str, Type, object] = "attributeclass",
    out_path: str = "summaries/summary.json",
    *,
    meta: Dict[str, Any] = None,
    include_membership: bool = False,
    only_id: bool = False,               # <<< existing: lightweight export
    include_edges: bool = True,          # ADDED: toggle for saving edges
) -> Dict[str, Any]:

    # 1) Get strategy instance
    if isinstance(strategy, str):
        key = strategy.strip().lower()
        if key not in _STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy}', choose one of {list(_STRATEGY_REGISTRY)}")
        strat = _STRATEGY_REGISTRY[key]()
        strategy_name = key
    elif isinstance(strategy, type):
        strat = strategy()
        strategy_name = strategy.__name__
    else:
        strat = strategy
        strategy_name = type(strat).__name__

    # 2) Collect per-vertex summaries with this strategy
    collector = SchemaCollector(strat)
    per_v_summary, per_v_id = collector.collect(G)

    # 3) Group nodes (supernodes)
    groups = group_by_signature(per_v_id)

    # 4) make into JSON-serializable structure
    nodes: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}  # may remain empty if include_edges=False

    if not only_id:
        # Full (default) export: schema + payload
        for sid, members in groups.items():
            types, preds, dsp = set(), set(), set()
            pred2dst: Dict[str, set] = {}
            for n in members:
                vs = per_v_summary.get(n)
                if not vs:
                    continue
                types |= set(vs.types)
                preds |= set(vs.neighbors.keys())
                dsp |= set(vs.payload)
                for p, dsts in vs.neighbor_dst_types.items():
                    pred2dst.setdefault(p, set()).update(dsts)
            nodes[str(sid)] = {
                "sid": int(sid),
                "vcp": len(members),                           # payload: vertex count
                "dsp": _sorted_list(dsp),                      # payload: data sources
                "types": _sorted_list(types),                  # schema: rdf:types
                "predicates": _sorted_list(preds),             # schema: property set
                "pred_to_dst_types": _canon_pred2dst(pred2dst) # schema: predicate→dst types
            }
    else:
        # Lightweight export: only IDs + counts
        for sid, members in groups.items():
            nodes[str(sid)] = {
                "sid": int(sid),
                "vcp": len(members),
            }

    # 5) Build and include edges only if requested  ---------------------  #
    if include_edges:  # ADDED: guard the whole edge computation
        super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)  # MOVED under guard
        for su, dsts in super_counts.items():
            for sv, total in dsts.items():
                if not only_id:
                    pred_counts = {
                        p: int(c) for p, c in sorted(
                            super_pred.get(su, {}).get(sv, {}).items()
                        )
                    }
                    edges[f"{su}->{sv}"] = {
                        "src": int(su),
                        "dst": int(sv),
                        "count": int(total),        # payload: rolled edge count
                        "pred_counts": pred_counts  # payload: histogram per predicate
                    }
                else:
                    edges[f"{su}->{sv}"] = {
                        "src": int(su),
                        "dst": int(sv),
                        "count": int(total),
                    }
    # ------------------------------------------------------------------- #

    pkg_meta = {
        "strategy": strategy_name,
        "only_id": bool(only_id),
        "include_edges": bool(include_edges),   # ADDED: record edge choice
        **(meta or {})
    }
    pkg = {
        "meta": pkg_meta,
        "nodes": dict(sorted(nodes.items())),
        # CHANGED: include 'edges' only if we actually built them (omit when empty)
        **({"edges": dict(sorted(edges.items()))} if edges else {}), 
    }

    if include_membership:
        membership = {}
        for node, sid in per_v_id.items():
            label = G.nodes[node].get("label", node) if node in G.nodes else str(node)
            membership[str(label)] = int(sid)
        pkg["membership"] = dict(sorted(membership.items()))

    # 6) Save JSON
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[save] wrote summary ({strategy_name}, only_id={only_id}, include_edges={include_edges}) → {out_path}")  # CHANGED: log flag
    return pkg

def load_summary_package(path: str) -> Dict[str, Any]:
    # Load a previously saved summary JSON package back into a dict
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)