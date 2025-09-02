# collector.py
from typing import Dict, Tuple
import networkx as nx
from VertexSummary import VertexSummary  # class with merge() and get_id()
from SchemaExtraction import SchemaExtraction  # Protocol or base class


class SchemaCollector:
    # GraphX simulator for python:
    # walks all edges (u -> v) in G
    # uses `strategy.send_message(...)` to build a per-edge VertexSummary
    # merges messages per SOURCE vertex using `strategy.merge_messages(...)`
    # returns:
    # per_vertex_summary: node -> merged VertexSummary
    # per_vertex_id:      node -> signature id (for FLUID grouping)
    def __init__(self, strategy: SchemaExtraction):
        self.strategy = strategy

    def collect(self, G: nx.MultiDiGraph) -> Tuple[Dict[object, VertexSummary], Dict[object, int]]:
        mailboxes: Dict[object, list[VertexSummary]] = {}

        # 1) send (GraphX sendMessage)
        for u, v, edata in G.edges(data=True):
            msg = self.strategy.send_message(G, u, v, edata)
            mailboxes.setdefault(u, []).append(msg)

        # 2) merge (GraphX mergeMessage)
        per_vertex_summary: Dict[object, VertexSummary] = {}
        per_vertex_id: Dict[object, int] = {}

        for node in G.nodes:
            msgs = mailboxes.get(node, [])
            merged = VertexSummary(types=set(G.nodes[node].get("types", [])))
            merged.add_instance(G.nodes[node].get("label"))
            for m in msgs:
                merged = self.strategy.merge_messages(merged, m)
            per_vertex_summary[node] = merged
            per_vertex_id[node] = merged.get_id()

        return per_vertex_summary, per_vertex_id
