import json
from pathlib import Path
import pytest
import networkx as nx

from save import save_summary_for_strategy, load_summary_package
from AttCollection import SEAttributeCollection
from ClassCollection import SEClassCollection
from AttClassCollection import SEAttributeClassCollection  # NEW
from CompAttClassCollection import SEComplexAttributeClassCollection

T_PERSON = "http://ex/Person"
T_ORG    = "http://ex/Org"
KNOWS    = "http://ex/knows"
WORKSAT  = "http://ex/worksAt"
NAME     = "http://ex/name"

SUBJECTS = ("a","b","e","g","d")

def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def test_attribute_onlyid_and_full(tmp_path, build_minigraph):
    G = build_minigraph

    # --- only_id=True (lightweight) ---
    out1 = tmp_path / "summaries" / "mini_attribute_onlyid.json"
    pkg1 = save_summary_for_strategy(
        G, strategy="attribute", out_path=str(out1), only_id=True, include_membership=False,
        meta={"datasets": ["unit"]}
    )
    assert out1.exists()

    # meta flags
    assert pkg1["meta"]["strategy"] == "attribute"
    assert pkg1["meta"]["only_id"] is True

    # nodes contain only sid + vcp (no schema fields)
    assert pkg1["nodes"]
    any_node = next(iter(pkg1["nodes"].values()))
    assert set(any_node.keys()) == {"sid", "vcp"}

    # --- only_id=False (full) ---
    out2 = tmp_path / "summaries" / "mini_attribute_full.json"
    pkg2 = save_summary_for_strategy(
        G, strategy=SEAttributeCollection, out_path=str(out2), only_id=False, include_membership=True,
        meta={"datasets": ["unit"]}
    )
    assert out2.exists()

    # meta flags
    assert pkg2["meta"]["strategy"] in ("SEAttributeCollection", "attribute")  # name may differ for class input
    assert pkg2["meta"]["only_id"] is False

    # full nodes: sid,vcp,dsp,types,predicates,pred_to_dst_types
    node_val = next(iter(pkg2["nodes"].values()))
    assert {"sid","vcp","dsp","types","predicates","pred_to_dst_types"} <= set(node_val.keys())

    # attribute strategy stores predicates only (types & pred_to_dst_types empty at per-vertex level)
    # but here nodes are supernodes; still types/pred_to_dst_types should be [] / {}
    for nd in pkg2["nodes"].values():
        assert isinstance(nd["predicates"], list)
        assert isinstance(nd["types"], list) and nd["types"] in ([],)  # union stays empty
        assert isinstance(nd["pred_to_dst_types"], dict) and nd["pred_to_dst_types"] in ({},)

    # membership exists and maps raw node labels to SID
    assert "membership" in pkg2 and pkg2["membership"]
    for s in SUBJECTS:
        assert s in pkg2["membership"]


def test_class_full_export(tmp_path, build_minigraph):
    G = build_minigraph

    out = tmp_path / "out" / "deep" / "mini_class.json"
    pkg = save_summary_for_strategy(
        G, strategy="class", out_path=str(out), only_id=False, include_membership=True,
        meta={"include_literals": False}
    )
    assert out.exists()

    # full nodes present with type info; no predicates for class-only
    for nd in pkg["nodes"].values():
        assert isinstance(nd["types"], list)
        # class-only ignores predicates
        assert nd["predicates"] == []
        assert nd["pred_to_dst_types"] == {}

    # check Person/Org presence indirectly via membership SIDs
    memb = pkg["membership"]
    sid_person = memb["a"]
    sid_org    = memb["g"]
    assert sid_person != sid_org
    for p in ("b","e","d","x"):
        assert memb[p] == sid_person
    assert memb["y"] == sid_org
    assert memb["n1"] not in (sid_person, sid_org)  # untyped separate


def test_attributeclass_full_export_and_dst_types(tmp_path, build_minigraph):
    G = build_minigraph
    out = tmp_path / "s" / "mini_attrclass.json"
    pkg = save_summary_for_strategy(
        G, strategy=SEComplexAttributeClassCollection(), out_path=str(out),
        only_id=False, include_membership=True
    )
    assert out.exists()

    # dst-type classification must distinguish a vs b on KNOWS
    memb = pkg["membership"]
    sid_a = memb["a"]
    sid_b = memb["b"]
    assert sid_a != sid_b

    # Find the two supernodes and inspect schema fields
    nd_a = pkg["nodes"][str(sid_a)]
    nd_b = pkg["nodes"][str(sid_b)]

    assert T_PERSON in nd_a["types"]
    assert set(nd_a["predicates"]) == {KNOWS, WORKSAT}
    assert nd_a["pred_to_dst_types"][KNOWS] == [T_PERSON]   # a.knows → Person
    assert nd_a["pred_to_dst_types"][WORKSAT] == [T_ORG]    # a.worksAt → Org

    assert T_PERSON in nd_b["types"]
    assert set(nd_b["predicates"]) == {KNOWS, WORKSAT}
    assert nd_b["pred_to_dst_types"][KNOWS] == [T_ORG]      # b.knows → Org
    assert nd_b["pred_to_dst_types"][WORKSAT] == [T_ORG]


def test_unknown_strategy_raises(tmp_path, build_minigraph):
    G = build_minigraph
    with pytest.raises(ValueError):
        save_summary_for_strategy(G, strategy="nope", out_path=str(tmp_path / "x.json"))


def test_roundtrip_and_determinism(tmp_path, build_minigraph):
    G = build_minigraph
    out = tmp_path / "summaries" / "mini_attrclass_round.json"

    # Write twice; contents should be byte-identical (sorted keys, deterministic)
    pkg1 = save_summary_for_strategy(G, strategy="attributeclass", out_path=str(out), only_id=False)
    bytes1 = _read_bytes(out)
    pkg2 = save_summary_for_strategy(G, strategy="attributeclass", out_path=str(out), only_id=False)
    bytes2 = _read_bytes(out)
    assert bytes1 == bytes2

    # Round-trip load equals what we just saved
    loaded = load_summary_package(str(out))
    assert loaded == pkg2

def test_attributeclass_non_complex_full_export(tmp_path, build_minigraph):
    """
    Export should include:
    - types
    - predicates
    - pred_to_dst_types should be empty {}
    """
    G = build_minigraph
    out = tmp_path / "s" / "mini_acc.json"

    pkg = save_summary_for_strategy(
        G,
        strategy=SEAttributeClassCollection(),
        out_path=str(out),
        only_id=False,
        include_membership=True,
    )
    assert out.exists()

    # meta sanity
    assert pkg["meta"]["only_id"] is False

    # ACC should store types + predicates, but NOT dst-types
    for nd in pkg["nodes"].values():
        assert isinstance(nd["types"], list)
        assert isinstance(nd["predicates"], list)
        assert isinstance(nd["pred_to_dst_types"], dict)
        assert nd["pred_to_dst_types"] == {}  # key ACC property

    # Check key expected groups exist indirectly via membership
    memb = pkg["membership"]
    sid_a = memb["a"]
    sid_b = memb["b"]
    sid_e = memb["e"]
    sid_g = memb["g"]
    sid_d = memb["d"]

    # a and b same (Person + {knows,worksAt})
    assert sid_a == sid_b

    # e different (Person + {name})
    assert sid_e != sid_a

    # d different (Person + empty set)
    assert sid_d != sid_a
    assert sid_d != sid_e

    # g different (Org + {knows})
    assert sid_g != sid_a
    assert sid_g != sid_e
    assert sid_g != sid_d

    # Also verify the schema fields of the (a,b) supernode look right
    nd_ab = pkg["nodes"][str(sid_a)]
    assert T_PERSON in nd_ab["types"]
    assert set(nd_ab["predicates"]) == {KNOWS, WORKSAT}
    assert nd_ab["pred_to_dst_types"] == {}

def test_class_full_export_via_object(tmp_path, build_minigraph):
    """
    Same as test_class_full_export, but passes the strategy as a class/object
    instead of the string "class".
    This catches bugs in save_summary_for_strategy's strategy dispatch.
    """
    G = build_minigraph
    out = tmp_path / "out" / "deep" / "mini_class_obj.json"

    pkg = save_summary_for_strategy(
        G,
        strategy=SEClassCollection(),   # <- object form
        out_path=str(out),
        only_id=False,
        include_membership=True,
        meta={"include_literals": False},
    )
    assert out.exists()

    # meta sanity
    assert pkg["meta"]["only_id"] is False

    # full nodes present with type info; no predicates for class-only
    for nd in pkg["nodes"].values():
        assert isinstance(nd["types"], list)
        assert nd["predicates"] == []
        assert nd["pred_to_dst_types"] == {}

    # check Person/Org presence indirectly via membership SIDs
    memb = pkg["membership"]
    sid_person = memb["a"]
    sid_org    = memb["g"]
    assert sid_person != sid_org
    for p in ("b", "e", "d", "x"):
        assert memb[p] == sid_person
    assert memb["y"] == sid_org
    assert memb["n1"] not in (sid_person, sid_org)