import networkx as nx
from VertexSummary import VertexSummary
from SchemaExtraction import SchemaExtraction

class SEAttributeCollection(SchemaExtraction):
    # Attribute-only strategy
    # take src types (from node attrs)
    # record the predicate used on this edge
    # record the graph
    # record the src instance (its label/IRI)
    # """
    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        src_types = set(G.nodes[src].get("types", []))
        vs = VertexSummary(types=src_types)
        vs.add_instance(G.nodes[src].get("label"))
        vs.add_neighbor(edata.get("predicate"))
        vs.add_payload(edata.get("graph"))
        return vs
