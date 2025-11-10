# SEAttributeClassCollection.py
import networkx as nx
from VertexSummary import VertexSummary
from SchemaExtraction import SchemaExtraction

class SEAttributeClassCollection(SchemaExtraction):
    # Attribute + Class Collection (ACC):
    #   include source rdf:types
    #   include outgoing predicate set
    #   DO NOT include per-predicate destination type sets (that’s SchemEX)

    def seed_vertex(self, G: nx.MultiDiGraph, node) -> VertexSummary:
        # Seed with the node's types so even vertices with no outgoing edges
        # still form a class-based signature
        vs = VertexSummary()
        vs.add_types(set(G.nodes[node].get("types", [])))
        vs.add_instance(G.nodes[node].get("label"))
        return vs

    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        vs_src = VertexSummary()

        # Record source (src) types
        src_types = set(G.nodes[src].get("types", []))
        vs_src.add_types(src_types)

        # Record the outgoing predicate (but NOT dst types)
        predicate = edata.get("predicate")
        if predicate is not None:
            vs_src.add_neighbor(predicate)

        # Optional payload: graph IRI
        g = edata.get("graph")
        if g is not None:
            vs_src.add_payload(g)

        # Record instance label for payload convenience
        vs_src.add_instance(G.nodes[src].get("label"))

        return vs_src
    # merge_messages uses VertexSummary.merge() by default (set unions + count sums)
