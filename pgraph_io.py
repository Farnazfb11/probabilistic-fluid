#!/usr/bin/env python3
"""
Pgraph_io.py
===========

Rebuild the weighted graph from the files written by ``edge_weights.py``,
without re-parsing the source dataset.

    from graph_io import load_weighted_graph
    G = load_weighted_graph(
        "out/dyldo/graph_weighted.wnt",
        "out/dyldo/vertex_types.tsv",
    )

Preprocessing is performed once per dataset: parsing, deduplication, term
normalisation and weight assignment all happen in ``edge_weights.py`` and are
independent of the binning under study.  This module reads the result back, so
that a run over several binnings and several summary models pays that cost
once rather than once per configuration.

Files read
----------
graph_weighted.wnt
    One line per triple, ``<subject> <predicate> [probability] <object>``.
    Type triples do not appear: they carry no probability mass.

vertex_types.tsv
    One line per typed vertex, tab separated: the vertex label followed by its
    type IRIs.  Needed because the equivalence relations OC_type of Table 1
    read types, which the weighted-triple file does not carry.

What is reconstructed
---------------------
Nodes carry ``label`` and, where known, ``types``.  Edges carry ``predicate``,
``prob`` and ``multiplicity``.  Vertices with no outgoing triple are given
``self_loop_prob = 1.0``, matching the case-based row-stochastic condition of
Definition 1; the self-loop is not added as an edge, since it is not a triple
of the data graph.

The named graph of the source quad is not reconstructed.  It is not part of a
triple and plays no role in the equivalence; ``edge_weights.save_weighted_edges_tsv``
retains it if it is needed for payload.

Only dependency: networkx.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import networkx as nx


def load_types(path: str) -> dict:
    """Read vertex_types.tsv into {vertex label: set of type IRIs}."""
    types: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            types[parts[0]] = set(parts[1:])
    return types


def load_weighted_graph(
    wnt_path: str,
    types_path: Optional[str] = None,
    progress_every: int = 0,
) -> nx.MultiDiGraph:
    """Rebuild the weighted MultiDiGraph from the preprocessed files.

    types_path may be omitted for the attribute collection, which does not
    read types; every other model needs it.

    progress_every prints a line every N triples when non-zero, which is
    worth setting on the larger datasets.
    """
    G = nx.MultiDiGraph()
    node_types = load_types(types_path) if types_path else {}

    def ensure(label: str):
        if label not in G:
            G.add_node(label, label=label)
            t = node_types.get(label)
            if t:
                G.nodes[label]["types"] = set(t)
        return label

    n = 0
    with open(wnt_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            s, p, w, o = parts[0], parts[1], parts[2], parts[3]
            prob = float(w.strip("[]"))
            ensure(s)
            ensure(o)
            G.add_edge(s, o, predicate=p, prob=prob)
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[graph_io] {n} triples read")

    # Types may be recorded for vertices that appear in no triple at all.
    for label, t in node_types.items():
        if label not in G:
            G.add_node(label, label=label, types=set(t))
        elif "types" not in G.nodes[label]:
            G.nodes[label]["types"] = set(t)

    _restore_multiplicity(G)
    _restore_self_loops(G)
    print(f"[graph_io] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(node_types)} typed vertices")
    return G


def _restore_multiplicity(G: nx.MultiDiGraph) -> None:
    """Recompute the per-(predicate, target) multiplicity used by the payload.

    Not stored in the weighted-triple file, since it is derivable: it is the
    number of parallel edges sharing a predicate and a target, which arises
    where several literal values were collapsed onto one vertex.
    """
    for u in G.nodes():
        mult = Counter(
            (d.get("predicate"), v) for _, v, d in G.out_edges(u, data=True)
        )
        for _, v, k, d in G.out_edges(u, keys=True, data=True):
            G[u][v][k]["multiplicity"] = mult[(d.get("predicate"), v)]


def _restore_self_loops(G: nx.MultiDiGraph) -> int:
    """Mark dead ends with the assumed self-loop of Definition 1.

    A vertex with no outgoing triple carries its whole mass on a self-loop, so
    that the row-stochastic condition holds. The loop is not a triple of the
    data graph: it has no predicate, is not added as an edge, and does not
    enter the outgoing mass compared by the equivalence.
    """
    n = 0
    for u in G.nodes():
        if G.out_degree(u) == 0:
            G.nodes[u]["self_loop_prob"] = 1.0
            n += 1
    return n


def check_row_stochastic(G: nx.MultiDiGraph, tol: float = 1e-9) -> tuple:
    """Verify the loaded graph still satisfies Definition 1.

    Cheap, and worth running after a load: a truncated or partly written
    weighted-triple file shows up here rather than as a puzzling partition.
    """
    n_weighted = n_dead = 0
    max_err = 0.0
    for u in G.nodes():
        out = list(G.out_edges(u, data=True))
        if not out:
            n_dead += 1
            total = G.nodes[u].get("self_loop_prob", 0.0)
        else:
            n_weighted += 1
            total = sum(d.get("prob", 0.0) for _, _, d in out)
        max_err = max(max_err, abs(total - 1.0))
    if max_err > tol:
        raise AssertionError(f"rows not stochastic (max err {max_err:.2e})")
    return n_weighted, n_dead, max_err


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wnt", help="graph_weighted.wnt")
    ap.add_argument("--types", default=None, help="vertex_types.tsv")
    ap.add_argument("--progress", type=int, default=0)
    args = ap.parse_args()

    G = load_weighted_graph(args.wnt, args.types, progress_every=args.progress)
    w, d, err = check_row_stochastic(G)
    print(f"[ok] {w} weighted subjects, {d} dead ends, max row error {err:.2e}")