import networkx as nx
from VertexSummary import VertexSummary
from SchemaExtraction import SchemaExtraction

class SEAttributeCollection(SchemaExtraction):
    # Attribute-only strategy
    # take src types (from node attrs)
    # record the predicate used on this edge
    # record the graph
    # record the src instance (its label/IRI)
    # CHANGED (new): give the strategy control over per-vertex seeding.
    # We intentionally DO NOT seed types here, so vertices start with empty type sets.
    def seed_vertex(self, G: nx.MultiDiGraph, node) -> VertexSummary:
        vs = VertexSummary()
        # Instances don’t affect grouping, but are useful payload.
        vs.add_instance(G.nodes[node].get("label"))
        return vs

    def send_message(self, G: nx.MultiDiGraph, src, dst, edata: dict) -> VertexSummary:
        # Predicate-only: we only record the outgoing predicate on this edge (and payload graph).
        vs = VertexSummary()

        pred = edata.get("predicate")
        if pred is not None:
            vs.add_neighbor(pred)

        g = edata.get("graph")
        if g is not None:
            vs.add_payload(g)
        vs.add_mass(pred, "*", edata.get("prob"))
        # IMPORTANT: Do NOT add types (src or dst). Do NOT add dst types per predicate.
        return vs

    # merge_messages uses default (set/union + count/sum) → OK for predicate-only.