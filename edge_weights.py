#!/usr/bin/env python3
"""
edge_weights.py
===============

Assign transition probabilities to the edges of the ``nx.MultiDiGraph``
produced by ``RDFGraphBuilder``, and optionally build the sparse transition
matrix.

    RDFParser(path).parse() -> RDFGraphBuilder().build(...) -> G
    add_edge_weights(G)                    # mutates G in place
    save_weighted_triples(G, "graph.wnt")  # <s> <p> [prob] <o>
    P, nodes = to_transition_matrix(G)     # optional, substochastic

Model
-----
By Definition 1 of the thesis, an uncertain semantic graph is a pair
G_p = (G, pi) where G is a *set of triples* and pi assigns a weight to each
triple, row-stochastic per subject:

    sum_{(s,p,o) in G} pi(s,p,o) = 1   for every s with at least one triple.

The default weighting is uniform over triples:

    pi(s,p,o) = 1 / d_out(s),  d_out(s) = number of triples with subject s.

Note that the denominator counts TRIPLES, not distinct neighbours.  A subject
reaching the same object through two different predicates has two triples and
therefore d_out = 2.

Dead ends
---------
Vertices with no outgoing triple get no weights and a zero row in the matrix,
so the matrix is SUBSTOCHASTIC.  This is intended: pi constrains only vertices
with at least one outgoing triple.  No random-surfer / dangling correction is
applied, because partition refinement needs no power iteration and
redistributing mass would give dead ends behaviour absent from the data.
Pass dangling="uniform" only if some downstream computation genuinely needs a
stochastic matrix -- and build that matrix separately, never feeding it back
into summarisation.

Deduplication
-------------
Duplicate triples cannot occur in a data graph, since G is a set.  For quad
sources (DyLDO, BTC150M) the same triple may appear under several named
graphs; that deduplication MUST happen upstream, on the original (s, p, o)
and BEFORE literals are collapsed to __LITERAL__.  It cannot be done here:
after collapsing, (s, foaf:name, "Alice") and (s, foaf:name, "Bob") are
indistinguishable parallel edges and merging them would lose a real statement.
Use ``report_parallel_edges`` to check what the builder handed over.

Only dependencies: networkx, numpy, scipy.
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


LITERAL_TYPE = "Literal"   # matches the type the builder puts on literal nodes


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _is_literal_node(G: nx.MultiDiGraph, n) -> bool:
    """True iff node n was tagged as a literal by the builder."""
    return LITERAL_TYPE in G.nodes[n].get("types", ())


def _out_triples(
    G: nx.MultiDiGraph, u, skip_reverse: bool = False, drop_literals: bool = False
) -> List[Tuple[object, object, str]]:
    """Out-edges of u as a list of (target, key, predicate), one per triple.

    Unlike a per-neighbour grouping, this keeps (u, p, v) and (u, q, v) as two
    separate entries, which is what row-stochasticity over triples requires.
    """
    out: List[Tuple[object, object, str]] = []
    for _, v, k, data in G.out_edges(u, keys=True, data=True):
        if drop_literals and _is_literal_node(G, v):
            continue
        if skip_reverse and data.get("_reversed"):
            continue
        out.append((v, k, data.get("predicate", "")))
    return out


# --------------------------------------------------------------------------- #
# main entry point: annotate the graph
# --------------------------------------------------------------------------- #

def add_edge_weights(
    G: nx.MultiDiGraph,
    mode: str = "uniform",
    skip_reverse: bool = False,
    drop_literals: bool = False,
) -> None:
    """Mutate G in place, writing ``prob`` and ``multiplicity`` onto its edges.

    mode:
      'uniform' (default)  pi(s,p,o) = 1 / d_out(s), uniform over triples.
      'count'              parallel edges sharing (predicate, target) are
                           weighted by their multiplicity.  Identical to
                           'uniform' unless the builder produced parallel
                           edges (e.g. multiple literal values collapsed onto
                           __LITERAL__ under the same predicate).

    Edge attributes written:
      prob          the triple's weight pi(s,p,o).  Over all out-edges of s
                    this sums to 1 (or to 0 for a dead end).
      multiplicity  number of parallel edges sharing (predicate, target).
                    Recorded for the payload; not used by the equivalence.
    """
    if mode not in ("uniform", "count"):
        raise ValueError(f"unknown mode {mode!r}; use 'uniform' or 'count'")

    for u in G.nodes():
        triples = _out_triples(G, u, skip_reverse, drop_literals)
        if not triples:
            continue  # dead end: no outgoing mass, left unweighted

        mult = Counter((pred, v) for v, _, pred in triples)
        d_out = len(triples)

        for v, k, pred in triples:
            m = mult[(pred, v)]
            if mode == "uniform":
                p = 1.0 / d_out
            else:  # 'count': share the (pred, v) mass across its parallel edges
                p = (m / d_out) / m
            G[u][v][k]["prob"] = p
            G[u][v][k]["multiplicity"] = m


def check_row_stochastic(
    G: nx.MultiDiGraph, tol: float = 1e-9
) -> Tuple[int, int, float]:
    """Verify the model's row-stochastic condition.

    Returns (n_weighted, n_dead_ends, max_error) where max_error is the largest
    deviation from 1.0 among subjects that have at least one outgoing triple.
    Dead ends are expected to sum to 0 and are excluded from the error.
    """
    n_weighted = n_dead = 0
    max_err = 0.0
    for u in G.nodes():
        total = sum(
            d.get("prob", 0.0) for _, _, d in G.out_edges(u, data=True)
        )
        if G.out_degree(u) == 0:
            n_dead += 1
            continue
        n_weighted += 1
        max_err = max(max_err, abs(total - 1.0))
    if max_err > tol:
        raise AssertionError(f"rows not stochastic (max err {max_err:.2e})")
    return n_weighted, n_dead, max_err


def report_parallel_edges(G: nx.MultiDiGraph, limit: int = 10) -> None:
    """Print subjects with parallel (predicate, target) edges.

    These are either legitimate (several literal values collapsed onto
    __LITERAL__ under one predicate) or a sign that quad deduplication did not
    happen upstream.  Inspect a few before trusting the weights.
    """
    shown = 0
    total = 0
    for u in G.nodes():
        mult = Counter(
            (d.get("predicate", ""), v) for _, v, d in G.out_edges(u, data=True)
        )
        dups = {key: c for key, c in mult.items() if c > 1}
        if not dups:
            continue
        total += 1
        if shown < limit:
            label = G.nodes[u].get("label", u)
            print(f"[dup] {label}")
            for (pred, v), c in list(dups.items())[:5]:
                vlabel = G.nodes[v].get("label", v)
                print(f"        {c}x  {pred}  ->  {vlabel}")
            shown += 1
    print(f"[info] {total} subjects have parallel (predicate, target) edges")


# --------------------------------------------------------------------------- #
# optional: sparse transition matrix (substochastic by default)
# --------------------------------------------------------------------------- #

def to_transition_matrix(
    G: nx.MultiDiGraph,
    dangling: Optional[str] = None,
    skip_reverse: bool = False,
    drop_literals: bool = False,
) -> Tuple[sparse.csr_matrix, List]:
    """Build the transition matrix P and the node ordering.

    Parallel edges are summed, so P[u, v] is the total mass from u to v across
    all predicates -- the matrix is a projection of G_p, not a substitute.

    dangling:
      None (default)  dead-end rows stay zero; P is substochastic.  This is
                      what the model specifies.
      'uniform'       dead-end rows become 1/n.  Only for a computation that
                      genuinely needs a stochastic matrix (e.g. a power
                      iteration).  Never feed the result back into
                      summarisation: it invents edges that are not in the data
                      and breaks the B_0 recovery result.
    """
    if not any("prob" in d for _, _, d in G.edges(data=True)):
        add_edge_weights(G, skip_reverse=skip_reverse, drop_literals=drop_literals)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = sparse.lil_matrix((n, n), dtype=np.float64)

    for u, v, data in G.edges(data=True):
        p = data.get("prob")
        if p is None:
            continue
        P[idx[u], idx[v]] += p  # sum parallel edges and distinct predicates

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    n_dead = int((row_sums == 0).sum())

    if dangling == "uniform":
        for i in np.where(row_sums == 0)[0]:
            P[i, :] = 1.0 / n
        print(f"[warn] applied uniform dangling correction to {n_dead} rows; "
              f"do not use this matrix for summarisation")
    elif dangling is not None:
        raise ValueError(f"unknown dangling mode {dangling!r}")
    else:
        print(f"[info] {n_dead} dead-end rows left at zero (substochastic)")

    P = P.tocsr()

    # non-zero rows must still sum to one
    sums = np.asarray(P.sum(axis=1)).ravel()
    nz = sums > 0
    if nz.any():
        err = float(np.abs(sums[nz] - 1.0).max())
        assert err < 1e-9, f"non-zero rows not stochastic (max err {err:.2e})"

    return P, nodes


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def save_weighted_triples(G: nx.MultiDiGraph, path: str) -> int:
    """Write the weighted graph as one line per triple:

        <subject> <predicate> [probability] <object>

    This is the preprocessed representation summarisation reads, so that
    weighting is done once per dataset and is independent of the binning.
    Returns the number of lines written.
    """
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for u, v, data in G.edges(data=True):
            if "prob" not in data:
                continue
            s = G.nodes[u].get("label", u)
            o = G.nodes[v].get("label", v)
            pred = data.get("predicate", "")
            fh.write(f"{s} {pred} [{data['prob']:.10g}] {o}\n")
            written += 1
    return written


def save_weighted_edges_tsv(G: nx.MultiDiGraph, path: str) -> None:
    """Dump the weighted edges to TSV, including multiplicity for the payload."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("src\tpredicate\tprob\tdst\tmultiplicity\n")
        for u, v, data in G.edges(data=True):
            if "prob" not in data:
                continue
            su = G.nodes[u].get("label", u)
            sv = G.nodes[v].get("label", v)
            pred = data.get("predicate", "")
            fh.write(f"{su}\t{pred}\t{data['prob']:.10g}\t{sv}\t"
                     f"{data['multiplicity']}\n")


def save_graph_pickle(G: nx.MultiDiGraph, path: str) -> None:
    """nx.write_gpickle was removed in networkx 3.x, so use plain pickle."""
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------- #
# end-to-end wiring
# --------------------------------------------------------------------------- #

def build_and_weight(
    file_path: str,
    include_literals: bool = False,
    add_reverse_edges: bool = False,
    use_numeric_ids: bool = True,
    **weight_kwargs,
) -> nx.MultiDiGraph:
    """Parse -> build -> weight, returning the annotated graph."""
    from RDFparser import RDFParser
    from RDFGraphBuilder import RDFGraphBuilder

    parser = RDFParser(file_path)
    builder = RDFGraphBuilder(
        include_literals=include_literals,
        use_numeric_ids=use_numeric_ids,
        add_reverse_edges=add_reverse_edges,
    )
    G = builder.build(parser.parse())
    add_edge_weights(G, **weight_kwargs)
    return G


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", help="path to .nq.gz / .nt.gz")
    ap.add_argument("--include-literals", action="store_true")
    ap.add_argument("--add-reverse-edges", action="store_true")
    ap.add_argument("--mode", choices=["uniform", "count"], default="uniform")
    ap.add_argument("--skip-reverse", action="store_true")
    ap.add_argument("--drop-literals", action="store_true")
    ap.add_argument("--matrix", action="store_true",
                    help="also build and save the transition matrix")
    ap.add_argument("--dangling", choices=["uniform"], default=None,
                    help="apply a dangling correction (not for summarisation)")
    ap.add_argument("--check-duplicates", action="store_true",
                    help="report subjects with parallel (predicate, target) edges")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    G = build_and_weight(
        args.file,
        include_literals=args.include_literals,
        add_reverse_edges=args.add_reverse_edges,
        mode=args.mode,
        skip_reverse=args.skip_reverse,
        drop_literals=args.drop_literals,
    )
    print(f"[info] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.check_duplicates:
        report_parallel_edges(G)

    n_weighted, n_dead, err = check_row_stochastic(G)
    share = n_dead / max(1, G.number_of_nodes())
    print(f"[ok] {n_weighted} weighted subjects, {n_dead} dead ends "
          f"({share:.1%}), max row error {err:.2e}")

    wp = os.path.join(args.outdir, "graph_weighted.wnt")
    n = save_weighted_triples(G, wp)
    print(f"[ok] wrote {wp} ({n} triples)")

    tp = os.path.join(args.outdir, "edges_weighted.tsv")
    save_weighted_edges_tsv(G, tp)
    print(f"[ok] wrote {tp}")

    if args.matrix:
        P, nodes = to_transition_matrix(
            G, dangling=args.dangling,
            skip_reverse=args.skip_reverse, drop_literals=args.drop_literals,
        )
        mp = os.path.join(args.outdir, "transition_matrix.npz")
        sparse.save_npz(mp, P)
        with open(os.path.join(args.outdir, "node_order.tsv"), "w",
                  encoding="utf-8") as fh:
            fh.write("row\tnode_id\tlabel\n")
            for i, nd in enumerate(nodes):
                fh.write(f"{i}\t{nd}\t{G.nodes[nd].get('label', '')}\n")
        print(f"[ok] wrote {mp} ({P.shape[0]}x{P.shape[1]})")