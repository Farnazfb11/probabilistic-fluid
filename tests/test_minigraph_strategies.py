import pytest
from Collector import SchemaCollector
from AttCollection import SEAttributeCollection
from ClassCollection import SEClassCollection
from AttClassCollection import SEAttributeClassCollection  # NEW
from CompAttClassCollection import SEComplexAttributeClassCollection

# IRI constants used by the fixture
KNOWS   = "http://ex/knows"
WORKSAT = "http://ex/worksAt"
NAME    = "http://ex/name"
T_PERSON = "http://ex/Person"
T_ORG    = "http://ex/Org"

SUBJECTS = ("a", "b", "e", "g", "d")  # we assert presence explicitly (fail fast)

def _subj_sid_map(per_id):
    # Explicitly require all subjects to exist in per_id; raises KeyError if missing
    return {n: per_id[n] for n in SUBJECTS}

#attribute collection (predicate-only))
def test_attribute_predicate_only(build_minigraph):
    G = build_minigraph
    strat = SEAttributeCollection()
    per_v, per_id = SchemaCollector(strat).collect(G)

    subj = _subj_sid_map(per_id)

    # --- SID relations (groups) among subjects ---
    # a and b share {knows, worksAt}
    assert subj["a"] == subj["b"]
    # g has only {knows}
    assert subj["g"] != subj["a"]
    # e has only {name}
    assert subj["e"] not in {subj["a"], subj["g"]}
    # d has empty predicate set
    assert subj["d"] not in {subj["a"], subj["e"], subj["g"]}

    # Among subjects: exactly 4 distinct groups -> {a,b}, {e}, {g}, {d}
    assert len(set(subj.values())) == 4

    # --- Predicate recognition/classification (per_v checks) ---
    # Attribute-only: predicates recorded, types and dst-types NOT recorded
    # a: {knows, worksAt}
    assert set(per_v["a"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert per_v["a"].types == set()  # strategy seeds no types
    assert per_v["a"].neighbor_dst_types == {}  # no dst-type attachment

    # b: {knows, worksAt}
    assert set(per_v["b"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert per_v["b"].types == set()
    assert per_v["b"].neighbor_dst_types == {}

    # g: {knows}
    assert set(per_v["g"].neighbors.keys()) == {KNOWS}
    assert per_v["g"].types == set()
    assert per_v["g"].neighbor_dst_types == {}

    # e: {name}
    assert set(per_v["e"].neighbors.keys()) == {NAME}
    assert per_v["e"].types == set()
    assert per_v["e"].neighbor_dst_types == {}

    # d: ∅ predicates
    assert per_v["d"].neighbors == {}
    assert per_v["d"].types == set()
    assert per_v["d"].neighbor_dst_types == {}

#class collection (type-only)
def test_class_only(build_minigraph):
    G = build_minigraph
    strat = SEClassCollection()
    per_v, per_id = SchemaCollector(strat).collect(G)

    subj = _subj_sid_map(per_id)

    # --- SID relations ---
    person_sid = subj["a"]
    for p in ("b", "e", "d"):
        assert subj[p] == person_sid
    assert subj["g"] != person_sid  # g is Org

    # Among subjects: exactly 2 groups -> Person vs Org
    assert len(set(subj.values())) == 2

    # --- Classification correctness (per_v checks) ---
    # Class-only: source types recorded; predicates ignored
    for n in ("a", "b", "e", "d"):
        assert set(per_v[n].types) == {T_PERSON}
        assert per_v[n].neighbors == {}  # no predicates
        assert per_v[n].neighbor_dst_types == {}

    assert set(per_v["g"].types) == {T_ORG}
    assert per_v["g"].neighbors == {}
    assert per_v["g"].neighbor_dst_types == {}

def test_attribute_class_non_complex(build_minigraph):
    """
    SEAttributeClassCollection (ACC):
    signature = (src types) + (outgoing predicate set)
    NOT dst types.
    """
    G = build_minigraph
    strat = SEAttributeClassCollection()
    per_v, per_id = SchemaCollector(strat).collect(G)

    subj = _subj_sid_map(per_id)

    # --- Expected grouping logic ---
    # Person with {knows, worksAt}: a and b
    assert subj["a"] == subj["b"]

    # Org with {knows}: g must differ from Person-{knows,worksAt}
    assert subj["g"] != subj["a"]

    # Person with {name}: e must differ from (a,b) and from g
    assert subj["e"] not in {subj["a"], subj["g"]}

    # Person with empty predicate set: d must be its own group
    assert subj["d"] not in {subj["a"], subj["e"], subj["g"]}

    # Among subjects: exactly 4 groups -> {a,b}, {g}, {e}, {d}
    assert len(set(subj.values())) == 4

    # --- Per-vertex summary checks ---
    # ACC records source types and predicate set, but NOT dst types
    assert set(per_v["a"].types) == {T_PERSON}
    assert set(per_v["a"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert per_v["a"].neighbor_dst_types == {}

    assert set(per_v["b"].types) == {T_PERSON}
    assert set(per_v["b"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert per_v["b"].neighbor_dst_types == {}

    assert set(per_v["g"].types) == {T_ORG}
    assert set(per_v["g"].neighbors.keys()) == {KNOWS}
    assert per_v["g"].neighbor_dst_types == {}

    assert set(per_v["e"].types) == {T_PERSON}
    assert set(per_v["e"].neighbors.keys()) == {NAME}
    assert per_v["e"].neighbor_dst_types == {}

    assert set(per_v["d"].types) == {T_PERSON}
    assert per_v["d"].neighbors == {}
    assert per_v["d"].neighbor_dst_types == {}

# complex attribute+class collection (predicate + type + dst-type)
def test_attribute_class_complex(build_minigraph):
    G = build_minigraph
    strat = SEComplexAttributeClassCollection()
    per_v, per_id = SchemaCollector(strat).collect(G)

    subj = _subj_sid_map(per_id)

    # --- SID relations ---
    # a != b because their knows→dst_type differs (Person vs Org)
    assert subj["a"] != subj["b"]
    # g (Org) != a (Person)
    assert subj["g"] != subj["a"]

    # Among subjects: all five distinct (a, b, e, g, d)
    assert len(set(subj.values())) == 5

    # --- Predicate recognition & dst-type classification (per_v checks) ---
    # a:
    assert set(per_v["a"].types) == {T_PERSON}
    assert set(per_v["a"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert set(per_v["a"].neighbor_dst_types[KNOWS]) == {T_PERSON}
    assert set(per_v["a"].neighbor_dst_types[WORKSAT]) == {T_ORG}

    # b:
    assert set(per_v["b"].types) == {T_PERSON}
    assert set(per_v["b"].neighbors.keys()) == {KNOWS, WORKSAT}
    assert set(per_v["b"].neighbor_dst_types[KNOWS]) == {T_ORG}
    assert set(per_v["b"].neighbor_dst_types[WORKSAT]) == {T_ORG}

    # g:
    assert set(per_v["g"].types) == {T_ORG}
    assert set(per_v["g"].neighbors.keys()) == {KNOWS}
    assert set(per_v["g"].neighbor_dst_types[KNOWS]) == {T_PERSON}

    # e:
    assert set(per_v["e"].types) == {T_PERSON}
    assert set(per_v["e"].neighbors.keys()) == {NAME}
    # name points to untyped (empty set)
    assert set(per_v["e"].neighbor_dst_types.get(NAME, set())) == set()

    # d:
    assert set(per_v["d"].types) == {T_PERSON}
    assert per_v["d"].neighbors == {}
    assert per_v["d"].neighbor_dst_types == {}

