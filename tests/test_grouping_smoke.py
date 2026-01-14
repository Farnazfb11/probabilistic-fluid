import pytest

from Collector import SchemaCollector
from AttCollection import SEAttributeCollection
from Summarizer import group_by_signature

def test_group_by_signature_smoke(build_minigraph):
    G = build_minigraph
    per_v, per_id = SchemaCollector(SEAttributeCollection()).collect(G)
    groups = group_by_signature(per_id)

    # Every node must appear exactly once
    all_members = set().union(*groups.values())
    assert all_members == set(G.nodes)

    # a and b same group for attribute-only
    sid_a = per_id["a"]
    sid_b = per_id["b"]
    assert sid_a == sid_b
    assert "a" in groups[sid_a] and "b" in groups[sid_a]