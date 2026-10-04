import os, json, gzip
from typing import Dict, Any, Union, Type, Optional
import networkx as nx

#  project dependencies ===
from Collector import SchemaCollector
from AttCollection import SEAttributeCollection
from ClassCollection import SEClassCollection
from CompAttClassCollection import SEComplexAttributeClassCollection
from AttClassCollection import SEAttributeClassCollection
from Summarizer import group_by_signature, build_summary_graph

# Separator for the (P, O) keys of the mass payload. The same character
# get_id() uses, so it cannot occur inside an IRI or a type-set serialisation.
SEP = "\u241E"


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
    "complexattributeclass": SEComplexAttributeClassCollection,
}


def build_package(
    G: nx.MultiDiGraph,
    per_v_summary: Dict[Any, Any],
    per_v_id: Dict[Any, int],
    *,
    strategy_name: str,
    meta: Dict[str, Any] = None,
    include_membership: bool = False,
    only_id: bool = False,
    include_edges: bool = True,
    include_pred_counts: bool = True,
) -> Dict[str, Any]:
    # Build the canonical package from summaries that have ALREADY been
    # collected. Split out of save_summary_for_strategy so that run_summary.py
    # can export without running the collector a second time.
    #
    # The mass payload (mpp), the bins and the edge multiplicities are taken
    # from members[0], the same representative the representative error of
    # Section 4.5 is measured against, and are emitted only when the vertices
    # actually carry mass. A deterministic run over an unweighted graph
    # therefore produces exactly the package this file produced before.
    groups = group_by_signature(per_v_id)

    nodes: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}

    if not only_id:
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

            entry = {
                "sid": int(sid),
                "vcp": len(members),                            # payload: vertex count
                "dsp": _sorted_list(dsp),                       # payload: data sources
                "types": _sorted_list(types),                   # schema: rdf:types
                "predicates": _sorted_list(preds),              # schema: property set
                "pred_to_dst_types": _canon_pred2dst(pred2dst), # schema: predicate→dst types
            }

            rep = per_v_summary.get(members[0])
            if rep is not None and rep.mass:
                entry["rep"] = str(G.nodes[members[0]].get("label", members[0]))
                entry["mpp"] = {f"{p}{SEP}{o}": v for (p, o), v in rep.mass.items()}
                entry["emp"] = dict(rep.neighbors)              # payload: edge multiplicity
                if rep.bins is not None:
                    entry["bins"] = {f"{p}{SEP}{o}": v for (p, o), v in rep.bins.items()}

            nodes[str(sid)] = entry
    else:
        for sid, members in groups.items():
            nodes[str(sid)] = {"sid": int(sid), "vcp": len(members)}

    if include_edges:
        super_counts, super_pred = build_summary_graph(G, groups, by_predicate=True)
        for su, dsts in super_counts.items():
            for sv, total in dsts.items():
                entry = {"src": int(su), "dst": int(sv), "count": int(total)}
                if not only_id and include_pred_counts:
                    entry["pred_counts"] = {
                        p: int(c) for p, c in sorted(
                            super_pred.get(su, {}).get(sv, {}).items()
                        )
                    }
                edges[f"{su}->{sv}"] = entry
    pkg_meta = {
        "strategy": strategy_name,
        "only_id": bool(only_id),
        "include_edges": bool(include_edges),
        **(meta or {})
    }
    pkg = {
        "meta": pkg_meta,
        "nodes": dict(sorted(nodes.items())),
        **({"edges": dict(sorted(edges.items()))} if edges else {}),
    }

    if include_membership:
        membership = {}
        for node, sid in per_v_id.items():
            label = G.nodes[node].get("label", node) if node in G.nodes else str(node)
            membership[str(label)] = int(sid)
        pkg["membership"] = dict(sorted(membership.items()))

    return pkg


def write_package(pkg: Dict[str, Any], out_path: str) -> None:
    # Write a package to disk, gzipping it when out_path ends in .gz.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False, indent=2, sort_keys=True)


def save_summary_for_strategy(
    G: nx.MultiDiGraph,
    strategy: Union[str, Type, object] = "attributeclass",
    out_path: str = "summaries/summary.json",
    *,
    meta: Dict[str, Any] = None,
    include_membership: bool = False,
    only_id: bool = False,
    include_edges: bool = True,
    binning: Optional[object] = None,   # ADDED: None reproduces the old behaviour
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
    per_v_summary, per_v_id = collector.collect(G, binning=binning)

    # 3) Build and write
    pkg = build_package(
        G, per_v_summary, per_v_id,
        strategy_name=strategy_name,
        meta=meta,
        include_membership=include_membership,
        only_id=only_id,
        include_edges=include_edges,
    )
    write_package(pkg, out_path)

    print(f"[save] wrote summary ({strategy_name}, only_id={only_id}, "
          f"include_edges={include_edges}, binning={'none' if binning is None else getattr(binning, 'family', '?')}) → {out_path}")
    return pkg


def load_summary_package(path: str) -> Dict[str, Any]:
    # Load a previously saved summary JSON package back into a dict.
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)