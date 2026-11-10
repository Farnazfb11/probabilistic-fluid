# collector.py
from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
from VertexSummary import VertexSummary  # class with merge(), seal() and get_id()
from SchemaExtraction import SchemaExtraction  # Protocol or base class


class SchemaCollector:
    # GraphX simulator for python:
    # walks all edges (u -> v) in G
    # uses `strategy.send_message(...)` to build a per-edge VertexSummary
    # merges messages per SOURCE vertex using `strategy.merge_messages(...)`
    #
    # The work is split in two, so that the bin boundaries can be fitted to
    # the collected masses before any vertex is binned:
    #   collect_masses(G)          node -> merged VertexSummary, exact masses
    #   signatures(summaries, b)   node -> signature id under binning b
    # collect(G, binning) does both in one call, as before, for callers that
    # already have a binning.
    def __init__(self, strategy: SchemaExtraction):
        self.strategy = strategy

    def collect_masses(self, G: nx.MultiDiGraph) -> Dict[object, VertexSummary]:
        mailboxes: Dict[object, list[VertexSummary]] = {}

        # 1) send (GraphX sendMessage)
        for u, v, edata in G.edges(data=True):
            msg = self.strategy.send_message(G, u, v, edata)
            mailboxes.setdefault(u, []).append(msg)

        # 2) merge (GraphX mergeMessage)
        per_vertex_summary: Dict[object, VertexSummary] = {}

        for node in G.nodes:
            msgs = mailboxes.get(node, [])
            if hasattr(self.strategy, "seed_vertex"):
                merged = self.strategy.seed_vertex(G, node)  # CHANGED: new hook
            else:
                merged = VertexSummary(types=set(G.nodes[node].get("types", [])))  # original default
                merged.add_instance(G.nodes[node].get("label"))                    # original default

            for m in msgs:
                merged = self.strategy.merge_messages(merged, m)

            per_vertex_summary[node] = merged

        return per_vertex_summary

    @staticmethod
    def signatures(per_vertex_summary: Dict[object, VertexSummary],
                   binning=None) -> Dict[object, int]:
        # 3) bin the exact masses (or clear the bins, for binning=None) and
        #    hash every vertex's signature
        per_vertex_id: Dict[object, int] = {}
        for node, vs in per_vertex_summary.items():
            vs.seal(binning)
            per_vertex_id[node] = vs.get_id()
        return per_vertex_id

    def collect(self, G: nx.MultiDiGraph, binning=None) -> Tuple[Dict[object, VertexSummary], Dict[object, int]]:
        per_vertex_summary = self.collect_masses(G)
        per_vertex_id = self.signatures(per_vertex_summary, binning)
        return per_vertex_summary, per_vertex_id


def mass_values(per_vertex_summary: Dict[object, VertexSummary]) -> np.ndarray:
    # The multiset X the data-dependent binnings are fitted to: every non-zero
    # cell mass mu(s, P, O) of every vertex, one value per cell. Dead-ends
    # contribute nothing, as zero is the zero bin by construction.
    vals = np.fromiter(
        (v for vs in per_vertex_summary.values() for v in vs.mass.values()),
        dtype=float,
    )
    return vals[vals > 0.0]
