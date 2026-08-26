#!/usr/bin/env python3
"""
pagerank_weights.py
===================

Add random-surfer (PageRank) probability weights to the edges of the
``nx.MultiDiGraph`` produced by ``RDFGraphBuilder``, and (optionally) build the
row-stochastic Markov transition matrix.

Designed to sit downstream of your existing pipeline:

    RDFParser(path).parse()  ->  RDFGraphBuilder().build(...)  ->  G (MultiDiGraph)
    add_pagerank_weights(G)                      # <-- this module, mutates G in place
    P, nodes = to_transition_matrix(G)           # optional Markov matrix

Two roles of the "random surfer":
  1. PAGERANK SCORES  --  classic damped surfer (alpha=0.85) with uniform
     teleport; dangling nodes redistribute uniformly.  Gives PR(v) per vertex.
  2. EDGE WEIGHTS     --  for a node u WITH out-edges, no teleport is mixed in:
         prob(u -> v) = PR(v) / sum_{w in out(u)} PR(w)
     over the DISTINCT out-neighbours of u (mass-based, EQR_p flavour).
     Dangling nodes (no out-edges) have nothing to weight; they only get a
     random/uniform row when you build the full transition matrix.

Edge attributes written onto G (per edge key of a MultiDiGraph):
  prob          transition probability to neighbour v.  Same value on every
                parallel u->v edge (the *mass* toward v).  Over distinct
                neighbours of u this sums to 1.
  prob_edge     prob / multiplicity(u,v).  Per-edge share; over ALL out-edges
                of u (parallel included) this sums to 1.
  multiplicity  number of parallel u->v edges (the PAY-style multiplicity;
                lets you keep the mass-vs-multiset distinction downstream).

Graph specifics handled:
  * rdf:type triples are not edges in your builder, so PageRank runs over the
    object-property graph only -- correct.
  * The collapsed "__LITERAL__" node (include_literals=False) is a dangling
    sink; by default it participates as a normal dangling node.  Use
    drop_literals=True to exclude literal-typed nodes from the walk entirely.
  * Reverse edges (attribute _reversed=True) are weighted by default; pass
    skip_reverse=True to ignore them.
  * Works with either numeric (hashed) or string node ids.

Only dependencies: networkx, numpy, scipy.
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


LITERAL_TYPE = "Literal"   # matches the type your builder puts on literal nodes


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _is_literal_node(G: nx.MultiDiGraph, n) -> bool:
    """True iff node n was tagged as a literal by the builder."""
    return LITERAL_TYPE in G.nodes[n].get("types", ())


def _walkable_targets(
    G: nx.MultiDiGraph, u, skip_reverse: bool, drop_literals: bool
) -> Dict[object, List]:
    """For node u, return {v: [edge_keys]} of the out-edges the surfer may use."""
    targets: Dict[object, List] = {}
    for v in G.successors(u):
        if drop_literals and _is_literal_node(G, v):
            continue
        for k, data in G[u][v].items():
            if skip_reverse and data.get("_reversed"):
                continue
            targets.setdefault(v, []).append(k)
    # drop any target that ended up with no usable keys
    return {v: ks for v, ks in targets.items() if ks}


def _collapsed_digraph(
    G: nx.MultiDiGraph, skip_reverse: bool, drop_literals: bool
) -> nx.DiGraph:
    """Collapse the MultiDiGraph to a simple DiGraph with a 'count' attribute.

    Used only to feed nx.pagerank (parallel edges become one edge carrying the
    parallel-edge count).  Node set is preserved so every vertex gets a score.
    """
    H = nx.DiGraph()
    keep_nodes = [
        n for n in G.nodes()
        if not (drop_literals and _is_literal_node(G, n))
    ]
    H.add_nodes_from(keep_nodes)
    for u in keep_nodes:
        for v, keys in _walkable_targets(G, u, skip_reverse, drop_literals).items():
            H.add_edge(u, v, count=len(keys))
    return H


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #

def compute_pagerank(
    G: nx.MultiDiGraph,
    damping: float = 0.85,
    mode: str = "pagerank",
    skip_reverse: bool = False,
    drop_literals: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Dict[object, float]:
    """Return {node: pagerank}.  'count' mode weights the surfer by parallel-edge
    multiplicity; other modes treat each distinct neighbour equally."""
    H = _collapsed_digraph(G, skip_reverse, drop_literals)
    if H.number_of_nodes() == 0:
        return {}
    weight = "count" if mode == "count" else None
    return nx.pagerank(H, alpha=damping, weight=weight,
                       max_iter=max_iter, tol=tol)


# --------------------------------------------------------------------------- #
# main entry point: annotate the graph
# --------------------------------------------------------------------------- #

def add_pagerank_weights(
    G: nx.MultiDiGraph,
    damping: float = 0.85,
    mode: str = "pagerank",
    skip_reverse: bool = False,
    drop_literals: bool = False,
    pr: Optional[Dict[object, float]] = None,
) -> Dict[object, float]:
    """Mutate G in place, writing prob / prob_edge / multiplicity onto its edges.

    mode:
      'pagerank' (default) prob(u->v) = PR(v) / sum_{w in out(u)} PR(w)
      'uniform'            prob(u->v) = 1 / |out(u)|            (structural surfer)
      'count'              prob(u->v) = mult(u,v) / sum mult    (multiset surfer)

    Returns the PageRank dict (also useful for to_transition_matrix).
    """
    if pr is None:
        pr = compute_pagerank(G, damping, mode, skip_reverse, drop_literals)

    for u in G.nodes():
        targets = _walkable_targets(G, u, skip_reverse, drop_literals)
        if not targets:
            continue  # dangling: nothing to weight here

        if mode == "pagerank":
            denom = sum(pr.get(w, 0.0) for w in targets)
            if denom > 0:
                probs = {v: pr.get(v, 0.0) / denom for v in targets}
            else:  # all neighbours had zero mass -> fall back to uniform
                probs = {v: 1.0 / len(targets) for v in targets}
        elif mode == "uniform":
            probs = {v: 1.0 / len(targets) for v in targets}
        elif mode == "count":
            total = sum(len(ks) for ks in targets.values())
            probs = {v: len(ks) / total for v, ks in targets.items()}
        else:
            raise ValueError(f"unknown mode {mode!r}")

        for v, keys in targets.items():
            mult = len(keys)
            p = probs[v]
            for k in keys:
                G[u][v][k]["prob"] = p
                G[u][v][k]["prob_edge"] = p / mult
                G[u][v][k]["multiplicity"] = mult

    return pr


# --------------------------------------------------------------------------- #
# optional: full Markov transition matrix (dangling nodes handled here)
# --------------------------------------------------------------------------- #

def to_transition_matrix(
    G: nx.MultiDiGraph,
    pr: Optional[Dict[object, float]] = None,
    mode: str = "pagerank",
    damping: float = 0.85,
    dangling: str = "uniform",
    skip_reverse: bool = False,
    drop_literals: bool = False,
    seed: int = 0,
) -> Tuple[sparse.csr_matrix, List]:
    """Build a row-stochastic Markov matrix P and the node ordering.

    Connected rows come from prob(u->v); a node with no out-edges gets a
    'uniform' (1/n) or 'random' row so every row sums to 1.

    Returns (P as scipy CSR, list of node ids in matrix-row order).
    """
    # make sure weights exist
    already = any("prob" in d for _, _, d in G.edges(data=True))
    if not already:
        pr = add_pagerank_weights(G, damping, mode, skip_reverse, drop_literals, pr)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = sparse.lil_matrix((n, n), dtype=np.float64)

    for u, v, data in G.edges(data=True):
        p = data.get("prob")
        if p is None:
            continue
        P[idx[u], idx[v]] = p  # same on parallel edges; idempotent

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    dangling_rows = np.where(row_sums == 0)[0]
    if len(dangling_rows):
        rng = np.random.default_rng(seed)
        if dangling == "uniform":
            for i in dangling_rows:
                P[i, :] = 1.0 / n
        elif dangling == "random":
            for i in dangling_rows:
                r = rng.random(n)
                P[i, :] = r / r.sum()
        else:
            raise ValueError(f"unknown dangling mode {dangling!r}")

    P = P.tocsr()
    err = float(np.abs(np.asarray(P.sum(axis=1)).ravel() - 1.0).max()) if n else 0.0
    assert err < 1e-9, f"rows not stochastic (max err {err:.2e})"
    return P, nodes


# --------------------------------------------------------------------------- #
# persistence helpers
# --------------------------------------------------------------------------- #

def save_weighted_edges_tsv(G: nx.MultiDiGraph, path: str) -> None:
    """Dump the weighted edges to TSV (uses node 'label' if present)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("src\tdst\tpredicate\tprob\tprob_edge\tmultiplicity\n")
        for u, v, data in G.edges(data=True):
            if "prob" not in data:
                continue
            su = G.nodes[u].get("label", u)
            sv = G.nodes[v].get("label", v)
            pred = data.get("predicate", "")
            fh.write(f"{su}\t{sv}\t{pred}\t{data['prob']:.10g}\t"
                     f"{data['prob_edge']:.10g}\t{data['multiplicity']}\n")


def save_graph_pickle(G: nx.MultiDiGraph, path: str) -> None:
    """nx.write_gpickle was removed in networkx 3.x, so use plain pickle."""
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------- #
# end-to-end wiring  (adjust the two imports to your module filenames)
# --------------------------------------------------------------------------- #

def build_and_weight(
    file_path: str,
    include_literals: bool = False,
    add_reverse_edges: bool = False,
    use_numeric_ids: bool = True,
    **weight_kwargs,
) -> Tuple[nx.MultiDiGraph, Dict[object, float]]:
    """Parse -> build -> weight, returning (G, pagerank).

    NOTE: change these two imports to match your filenames.
    """
    from RDFparser import RDFParser          # <-- your quad parser module
    from RDFGraphBuilder import RDFGraphBuilder  # <-- your builder module

    parser = RDFParser(file_path)
    builder = RDFGraphBuilder(
        include_literals=include_literals,
        use_numeric_ids=use_numeric_ids,
        add_reverse_edges=add_reverse_edges,
    )
    G = builder.build(parser.parse())
    pr = add_pagerank_weights(G, **weight_kwargs)
    return G, pr


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="path to .nq.gz / .nt.gz")
    ap.add_argument("--include-literals", action="store_true")
    ap.add_argument("--add-reverse-edges", action="store_true")
    ap.add_argument("--mode", choices=["pagerank", "uniform", "count"],
                    default="pagerank")
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--skip-reverse", action="store_true")
    ap.add_argument("--drop-literals", action="store_true")
    ap.add_argument("--dangling", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--matrix", action="store_true",
                    help="also build + save the Markov matrix")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    G, pr = build_and_weight(
        args.file,
        include_literals=args.include_literals,
        add_reverse_edges=args.add_reverse_edges,
        mode=args.mode, damping=args.damping,
        skip_reverse=args.skip_reverse, drop_literals=args.drop_literals,
    )
    print(f"[info] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    ep = os.path.join(args.outdir, "edges_weighted.tsv")
    save_weighted_edges_tsv(G, ep)
    print(f"[ok] wrote {ep}")

    if args.matrix:
        P, nodes = to_transition_matrix(
            G, pr=pr, mode=args.mode, dangling=args.dangling,
            skip_reverse=args.skip_reverse, drop_literals=args.drop_literals,
        )
        mp = os.path.join(args.outdir, "transition_matrix_pagerank.npz")
        sparse.save_npz(mp, P)
        with open(os.path.join(args.outdir, "node_order.tsv"), "w",
                  encoding="utf-8") as fh:
            fh.write("row\tnode_id\tlabel\n")
            for i, nd in enumerate(nodes):
                fh.write(f"{i}\t{nd}\t{G.nodes[nd].get('label', '')}\n")
        print(f"[ok] wrote {mp} ({P.shape[0]}x{P.shape[1]}, row-stochastic)")