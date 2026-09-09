#!/usr/bin/env python3
"""
edge_weights.py
===============

Assign transition probabilities to the edges of the ``nx.MultiDiGraph``
produced by ``RDFGraphBuilder``, and optionally build the transition matrix.

    RDFParser(path).parse() -> RDFGraphBuilder().build(...) -> G
    add_edge_weights(G)                    # mutates G in place
    save_weighted_triples(G, "graph.wnt")  # <s> <p> [prob] <o>
    P, nodes = to_transition_matrix(G)     # row-stochastic

Model
-----
By Definition 1 of the thesis, a probabilistic semantic graph is a pair
G_p = (G, pi) where G is a *set of triples* and pi assigns a weight to each
triple.  The row-stochastic condition is case-based:

    if s has at least one outgoing triple:
        sum_{(s,p,o) in G} pi(s,p,o) = 1
    if s has no outgoing triple (a dead end / sink vertex):
        pi(s, -, s) = 1        an assumed self-loop

The default weighting is uniform over triples:

    pi(s,p,o) = 1 / d_out(s),  d_out(s) = number of triples with subject s.

The denominator counts TRIPLES, not distinct neighbours.  A subject reaching
the same object through two different predicates has two triples and therefore
d_out = 2.

Dead ends
---------
A vertex with no outgoing triple is assumed to carry a self-loop of weight 1,
so that every row of the transition matrix sums to one.  The self-loop is a
device for satisfying the row-stochastic condition, not a statement about the
data: it has no predicate and by default is not materialised as an edge in G.
It is recorded as the node attribute ``self_loop_prob`` and written into the
diagonal of the transition matrix.

Because a non-existing edge has weight zero and every existing edge a strictly
positive weight, a dead end can never be confused with a vertex that sends an
arbitrarily small amount of mass somewhere.  Preserving that distinction is
why binning functions must be zero-separating.

No random-surfer or dangling correction is applied.  Redistributing mass from
a dead end to every vertex would invent edges absent from the data; the
self-loop keeps the mass where it is.

rdf:type
--------
Type triples carry NO probability mass and do not enter d_out(s).  A weight of
0.2 on "jack is a Person" has no sensible reading, and distributing mass over
type edges would make a vertex's probabilities depend on how richly it happens
to be typed.  ``RDFGraphBuilder`` records types as a node attribute rather than
an edge, so type triples never reach this module; ``check_no_type_edges``
verifies that.

Types are still used in the summarisation phase, as vertex labels.  That is a
separate step and does not affect the weights computed here.

Deduplication
-------------
An edge is a set member: being told about a statement twice does not make it
two statements.  Duplicate quads -- identical (s, p, o) under different named
graphs, or byte-identical lines -- must therefore be removed.  On the DyLDO
2015-10-11 snapshot this matters a great deal: 48,832,721 quads reduce to
31,111,684 distinct lines.

This must happen UPSTREAM, on the original (s, p, o) and BEFORE literals are
collapsed to __LITERAL__.  It cannot be done here: after collapsing,
(s, foaf:name, "Alice") and (s, foaf:name, "Bob") are indistinguishable
parallel edges, and merging them would discard a real statement.  Use
``report_parallel_edges`` to see what the builder handed over.

Only dependencies: networkx, numpy, scipy.
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


LITERAL_TYPE = "Literal"   # matches the type the builder puts on literal nodes
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"

# Predicate used only when self-loops are materialised explicitly.  It is not
# an RDF predicate and must not collide with one in the data.
SELF_LOOP_PREDICATE = "__SELF__"


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
    Materialised self-loops are excluded so that re-weighting is idempotent.
    """
    out: List[Tuple[object, object, str]] = []
    for _, v, k, data in G.out_edges(u, keys=True, data=True):
        pred = data.get("predicate", "")
        if pred == SELF_LOOP_PREDICATE:
            continue
        if drop_literals and _is_literal_node(G, v):
            continue
        if skip_reverse and data.get("_reversed"):
            continue
        out.append((v, k, pred))
    return out


def check_no_type_edges(G: nx.MultiDiGraph) -> int:
    """Verify that rdf:type triples were not turned into edges.

    Types must be node attributes, not weighted edges.  Returns the number of
    offending edges and prints a warning if any are found.
    """
    n = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("predicate", "") == RDF_TYPE
    )
    if n:
        print(f"[warn] {n} rdf:type edges found in the graph. Type triples "
              f"must not carry probability mass; check that "
              f"RDFGraphBuilder.add_statement still records types as node "
              f"attributes and returns before adding an edge.")
    return n


# --------------------------------------------------------------------------- #
# main entry point: annotate the graph
# --------------------------------------------------------------------------- #

def add_edge_weights(
    G: nx.MultiDiGraph,
    mode: str = "uniform",
    self_loops: str = "implicit",
    skip_reverse: bool = False,
    drop_literals: bool = False,
) -> None:
    """Mutate G in place, writing ``prob`` and ``multiplicity`` onto its edges.

    mode:
      'uniform' (default)  pi(s,p,o) = 1 / d_out(s), uniform over triples.
      'count'              parallel edges sharing (predicate, target) split
                           that pair's mass between them.  Under set semantics
                           parallel edges should not survive preprocessing, so
                           this differs from 'uniform' only where literal
                           values were collapsed onto __LITERAL__.

    self_loops:
      'implicit' (default) dead ends get the node attribute
                           ``self_loop_prob = 1.0``; no edge is added to G.
      'explicit'           a self-loop edge with predicate SELF_LOOP_PREDICATE
                           and prob 1.0 is added, for downstream tools that
                           need every row present as an edge.

    Edge attributes written:
      prob          the triple's weight pi(s,p,o).  Over the out-edges of s
                    this sums to 1.
      multiplicity  number of parallel edges sharing (predicate, target).
                    Recorded for the payload; not used by the equivalence.
    """
    if mode not in ("uniform", "count"):
        raise ValueError(f"unknown mode {mode!r}; use 'uniform' or 'count'")
    if self_loops not in ("implicit", "explicit"):
        raise ValueError(
            f"unknown self_loops {self_loops!r}; use 'implicit' or 'explicit'"
        )

    dead_ends = []

    for u in list(G.nodes()):
        triples = _out_triples(G, u, skip_reverse, drop_literals)

        if not triples:
            # Dead end: assume a self-loop of weight 1 so the row sums to one.
            G.nodes[u]["self_loop_prob"] = 1.0
            dead_ends.append(u)
            continue

        G.nodes[u].pop("self_loop_prob", None)
        mult = Counter((pred, v) for v, _, pred in triples)
        d_out = len(triples)

        for v, k, pred in triples:
            m = mult[(pred, v)]
            p = 1.0 / d_out if mode == "uniform" else (m / d_out) / m
            G[u][v][k]["prob"] = p
            G[u][v][k]["multiplicity"] = m

    if self_loops == "explicit":
        for u in dead_ends:
            existing = [
                k for _, v, k, d in G.out_edges(u, keys=True, data=True)
                if v == u and d.get("predicate", "") == SELF_LOOP_PREDICATE
            ]
            if not existing:
                G.add_edge(u, u, predicate=SELF_LOOP_PREDICATE,
                           prob=1.0, multiplicity=1)
            # the loop is now an edge; drop the attribute so it is not
            # counted twice
            G.nodes[u].pop("self_loop_prob", None)


def check_row_stochastic(
    G: nx.MultiDiGraph, tol: float = 1e-9
) -> Tuple[int, int, float]:
    """Verify the case-based row-stochastic condition of Definition 1.

    A vertex with outgoing triples must have weights summing to 1; a dead end
    must carry a self-loop of weight 1, whether implicit or explicit.

    Returns (n_weighted, n_dead_ends, max_error).
    """
    n_weighted = n_dead = 0
    max_err = 0.0

    for u in G.nodes():
        real = [
            d for _, _, d in G.out_edges(u, data=True)
            if d.get("predicate", "") != SELF_LOOP_PREDICATE
        ]
        if not real:
            n_dead += 1
            # A materialised self-loop takes precedence over the attribute, so
            # that a graph carrying both is not counted twice.
            loop = sum(
                d.get("prob", 0.0) for _, _, d in G.out_edges(u, data=True)
                if d.get("predicate", "") == SELF_LOOP_PREDICATE
            )
            total = loop if loop > 0.0 else G.nodes[u].get("self_loop_prob", 0.0)
        else:
            n_weighted += 1
            total = sum(d.get("prob", 0.0) for d in real)
        max_err = max(max_err, abs(total - 1.0))

    if max_err > tol:
        raise AssertionError(f"rows not stochastic (max err {max_err:.2e})")
    return n_weighted, n_dead, max_err


def report_parallel_edges(G: nx.MultiDiGraph, limit: int = 10) -> int:
    """Print subjects with parallel (predicate, target) edges.

    Under set semantics these should not survive preprocessing, except where
    several literal values were collapsed onto __LITERAL__ under one
    predicate.  Anything else indicates that quad deduplication did not happen
    upstream.  Returns the number of affected subjects.
    """
    shown = total = 0
    for u in G.nodes():
        mult = Counter(
            (d.get("predicate", ""), v) for _, v, d in G.out_edges(u, data=True)
            if d.get("predicate", "") != SELF_LOOP_PREDICATE
        )
        dups = {key: c for key, c in mult.items() if c > 1}
        if not dups:
            continue
        total += 1
        if shown < limit:
            print(f"[dup] {G.nodes[u].get('label', u)}")
            for (pred, v), c in list(dups.items())[:5]:
                print(f"        {c}x  {pred}  ->  {G.nodes[v].get('label', v)}")
            shown += 1
    print(f"[info] {total} subjects have parallel (predicate, target) edges")
    return total


# --------------------------------------------------------------------------- #
# transition matrix
# --------------------------------------------------------------------------- #

def to_transition_matrix(
    G: nx.MultiDiGraph,
    skip_reverse: bool = False,
    drop_literals: bool = False,
) -> Tuple[sparse.csr_matrix, List]:
    """Build the row-stochastic transition matrix P and the node ordering.

    Parallel edges and distinct predicates between the same pair are summed,
    so P[u, v] is the total mass from u to v -- a projection of G_p, not a
    substitute for it.  Dead ends contribute P[u, u] = 1 from their assumed
    self-loop, so every row sums to one without any dangling correction.
    """
    if not any("prob" in d for _, _, d in G.edges(data=True)):
        add_edge_weights(G, skip_reverse=skip_reverse, drop_literals=drop_literals)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = sparse.lil_matrix((n, n), dtype=np.float64)

    for u, v, data in G.edges(data=True):
        p = data.get("prob")
        if p is not None:
            P[idx[u], idx[v]] += p

    # implicit self-loops on dead ends
    n_self = 0
    for u in nodes:
        slp = G.nodes[u].get("self_loop_prob")
        if slp and P[idx[u], idx[u]] == 0.0:
            P[idx[u], idx[u]] = slp
            n_self += 1
    print(f"[info] {n_self} dead ends given a self-loop of weight 1")

    P = P.tocsr()
    sums = np.asarray(P.sum(axis=1)).ravel()
    err = float(np.abs(sums - 1.0).max()) if n else 0.0
    assert err < 1e-9, f"rows not stochastic (max err {err:.2e})"
    return P, nodes


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def save_weighted_triples(
    G: nx.MultiDiGraph, path: str, emit_self_loops: bool = False
) -> int:
    """Write the weighted graph as one line per triple:

        <subject> <predicate> [probability] <object>

    This is the preprocessed representation summarisation reads, so that
    weighting is done once per dataset and is independent of the binning.

    Self-loops on dead ends are assumed rather than stated, so by default they
    are not written; a dead end appears in the file only as the object of some
    other triple.  Set emit_self_loops=True to write them explicitly with the
    predicate SELF_LOOP_PREDICATE.
    """
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for u, v, data in G.edges(data=True):
            if "prob" not in data:
                continue
            pred = data.get("predicate", "")
            if pred == SELF_LOOP_PREDICATE and not emit_self_loops:
                continue
            s = G.nodes[u].get("label", u)
            o = G.nodes[v].get("label", v)
            fh.write(f"{s} {pred} [{data['prob']:.10g}] {o}\n")
            written += 1

        if emit_self_loops:
            for u in G.nodes():
                slp = G.nodes[u].get("self_loop_prob")
                if not slp:
                    continue
                s = G.nodes[u].get("label", u)
                fh.write(f"{s} {SELF_LOOP_PREDICATE} [{slp:.10g}] {s}\n")
                written += 1
    return written


def save_weighted_edges_tsv(G: nx.MultiDiGraph, path: str) -> None:
    """Dump the weighted edges to TSV, including multiplicity for the payload."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("src\tpredicate\tprob\tdst\tmultiplicity\n")
        for u, v, data in G.edges(data=True):
            if "prob" not in data:
                continue
            fh.write(
                f"{G.nodes[u].get('label', u)}\t{data.get('predicate', '')}\t"
                f"{data['prob']:.10g}\t{G.nodes[v].get('label', v)}\t"
                f"{data['multiplicity']}\n"
            )


def save_graph_pickle(G: nx.MultiDiGraph, path: str) -> None:
    """nx.write_gpickle was removed in networkx 3.x, so use plain pickle."""
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------- #
# end-to-end wiring
# --------------------------------------------------------------------------- #
def save_types(G: nx.MultiDiGraph, path: str) -> int:
    """Write the type sets recorded by the builder, one line per typed vertex.

    Type triples carry no probability mass and so do not appear in the
    weighted-triple file, but the equivalence relations OC_type of Table 1
    need them. Writing them here means summarisation can read the preprocessed
    graph without re-parsing the source.
    """
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for n, d in G.nodes(data=True):
            types = d.get("types")
            if not types:
                continue
            label = d.get("label", n)
            fh.write(label + "\t" + "\t".join(sorted(types)) + "\n")
            written += 1
    return written

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
    ap.add_argument("file", help="path to .nq.gz / .nt.gz / .nt / .nq")
    ap.add_argument("--include-literals", action="store_true")
    ap.add_argument("--add-reverse-edges", action="store_true")
    ap.add_argument("--mode", choices=["uniform", "count"], default="uniform")
    ap.add_argument("--self-loops", choices=["implicit", "explicit"],
                    default="implicit",
                    help="how dead-end self-loops are represented in G")
    ap.add_argument("--emit-self-loops", action="store_true",
                    help="write self-loops into the .wnt file as well")
    ap.add_argument("--skip-reverse", action="store_true")
    ap.add_argument("--drop-literals", action="store_true")
    ap.add_argument("--matrix", action="store_true",
                    help="also build and save the transition matrix")
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
        self_loops=args.self_loops,
        skip_reverse=args.skip_reverse,
        drop_literals=args.drop_literals,
    )
    print(f"[info] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    check_no_type_edges(G)
    if args.check_duplicates:
        report_parallel_edges(G)

    n_weighted, n_dead, err = check_row_stochastic(G)
    share = n_dead / max(1, G.number_of_nodes())
    print(f"[ok] {n_weighted} weighted subjects, {n_dead} dead ends "
          f"({share:.1%}) with self-loops, max row error {err:.2e}")

    wp = os.path.join(args.outdir, "graph_weighted.wnt")
    n = save_weighted_triples(G, wp, emit_self_loops=args.emit_self_loops)
    print(f"[ok] wrote {wp} ({n} triples)")
    
    yp = os.path.join(args.outdir, "vertex_types.tsv")
    ny = save_types(G, yp)
    print(f"[ok] wrote {yp} ({ny} typed vertices)")

    tp = os.path.join(args.outdir, "edges_weighted.tsv")
    save_weighted_edges_tsv(G, tp)
    print(f"[ok] wrote {tp}")

    if args.matrix:
        P, nodes = to_transition_matrix(
            G, skip_reverse=args.skip_reverse, drop_literals=args.drop_literals,
        )
        mp = os.path.join(args.outdir, "transition_matrix.npz")
        sparse.save_npz(mp, P)
        with open(os.path.join(args.outdir, "node_order.tsv"), "w",
                  encoding="utf-8") as fh:
            fh.write("row\tnode_id\tlabel\n")
            for i, nd in enumerate(nodes):
                fh.write(f"{i}\t{nd}\t{G.nodes[nd].get('label', '')}\n")
        print(f"[ok] wrote {mp} ({P.shape[0]}x{P.shape[1]}, row-stochastic)")