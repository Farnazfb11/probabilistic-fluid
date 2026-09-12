# SEClassCollection.py
import networkx as nx
from VertexSummary import VertexSummary
from SchemaExtraction import SchemaExtraction

class SEClassCollection(SchemaExtraction):
    # Class-only collection strategy (Scala SE_ClassCollection analogue)

    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        # Start a new summary element
        vs = VertexSummary()

        # Collect origin (src) types
        src_types = set(G.nodes[src].get("types", []))
        vs.add_types(src_types)

        # Add payload: which named graph (dataset source)
        vs.add_payload(edata.get("graph"))

        # Add instance: the actual subject URI
        vs.add_instance(G.nodes[src].get("label"))

        vs.add_mass("*", "*", edata.get("prob"))

        return vs

    # merge_messages just defaults to VertexSummary.merge via SchemaExtraction
