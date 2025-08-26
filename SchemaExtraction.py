from __future__ import annotations
import networkx as nx
from typing import Protocol #define flexible interfaces
from VertexSummary import VertexSummary

class SchemaExtraction(Protocol):
    # Scala trait analogue:
    #   - send_message: build a per-edge summary (for the source vertex).
    #   - merge_messages: combine two summaries into one (default = union).

    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        #VertexSummary from ONE edge (src->edata()'predicate')-> dst)
        # Parameters
        # G: NetworkX MultiDiGraph(built RDF graph)
        # src: source node id (hash int or string URI)
        # dst: destination node id
        # edata: edge attribute dict-has:
        #  'predicate': str (IRI in <>)
        #  'graph'    : str (named graph IRI in <>)

        # Returns
        # VertexSummary describing the source vertex contribution of the specific edge.
        raise NotImplementedError

    def merge_messages(self, a: VertexSummary, b: VertexSummary) -> VertexSummary:
        # 2 summaries with the same vertex -> new merged summary and can be strategized
        # merge two partial summaries for the same vertex into one.
        # default behavior is set-union + count-sum, so delegate to VertexSummary.merge.
        return a.merge(b)
