# SEComplexAttributeClassCollection.py
import networkx as nx
from VertexSummary import VertexSummary
from SchemaExtraction import SchemaExtraction

class SEComplexAttributeClassCollection(SchemaExtraction):
    # Complex Attribute+Class strategy (Scala SE_ComplexAttributeClassCollection analogue)

    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        vs_src = VertexSummary()

        # Get origin (src) types
        src_types = set(G.nodes[src].get("types", []))
        vs_src.add_types(src_types)

        # Get destination (dst) types
        dst_types = set(G.nodes[dst].get("types", []))

        # Add neighbor predicate
        predicate = edata.get("predicate")
        vs_src.add_neighbor(predicate)

        # Attach destination's types to this predicate
        vs_src.add_dst_types(predicate, dst_types)

        # Add payload (graph IRI)
        vs_src.add_payload(edata.get("graph"))

        # Add instance (the actual subject URI)
        vs_src.add_instance(G.nodes[src].get("label"))

        vs_src.add_mass(predicate, frozenset(dst_types), edata.get("prob"))

        return vs_src
    # merge_messages just defaults to VertexSummary.merge via SchemaExtraction