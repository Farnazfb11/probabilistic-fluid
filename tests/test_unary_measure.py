# tests/test_unary_comp.py  (drop-in replacement)
import math
import pytest

from Unary_measure import (
    count_equivalence_classes,
    eqc_size_list,
    average_eqc_size,
    std_eqc_size,
    edge_set,
    average_out_degree,
    average_in_degree,
    average_degree,
    tmh_tendency_to_make_hubs,
    connected_components_count,
    _build_digraph,
)


def _toy_pkg_all_features():
    return {
        "nodes": {
            "1": {"sid": 1, "vcp": 2},
            "2": {"sid": 2, "vcp": 3},
            "3": {"sid": 3, "vcp": 1},
            "4": {"sid": 4, "vcp": 4},
        },
        "edges": {
            "e1": {"src": 1, "dst": 2, "count": 2},
            "e2": {"src": 2, "dst": 3, "count": 3},
            "e3": {"src": 3, "dst": 1, "count": 4},
            "e4": {"src": 1, "dst": 2, "count": 1},   # parallel edge
            "e5": {"src": 2, "dst": 99, "count": 10}, # literal sink (dst not in nodes)
        },
    }


def _toy_pkg_membership_fallback():
    # vcp missing -> must fall back to membership counts
    return {
        "nodes": {
            "10": {"sid": 10},
            "20": {"sid": 20},
            "30": {"sid": 30},
        },
        "membership": {
            "a": 10,
            "b": 10,
            "c": 20,
            "d": 30,
            "e": 30,
            "f": 30,
        },
        "edges": {},
    }


def test_eqc_count_and_sizes():
    pkg = _toy_pkg_all_features()
    assert count_equivalence_classes(pkg) == 4
    assert sorted(eqc_size_list(pkg)) == [1, 2, 3, 4]
    assert average_eqc_size(pkg) == pytest.approx((2 + 3 + 1 + 4) / 4)


def test_membership_fallback_sizes_and_avg():
    pkg = _toy_pkg_membership_fallback()
    # sizes are derived from membership: sid10=2, sid20=1, sid30=3
    assert sorted(eqc_size_list(pkg)) == [1, 2, 3]
    assert average_eqc_size(pkg) == pytest.approx((2 + 1 + 3) / 3)


def test_std_eqc_size_sample_and_population():
    pkg = _toy_pkg_all_features()

    # sizes: [2,3,1,4], mean=2.5
    # sample variance = (0.25+0.25+2.25+2.25)/3 = 5/3
    assert std_eqc_size(pkg, population=False) == pytest.approx(math.sqrt(5.0 / 3.0))

    # population variance = (0.25+0.25+2.25+2.25)/4 = 5/4
    assert std_eqc_size(pkg, population=True) == pytest.approx(math.sqrt(5.0 / 4.0))


def test_edge_set_unweighted_and_weighted_are_merged_option_b():
    pkg = _toy_pkg_all_features()

    # Unweighted collapses parallel edges by (src,dst)
    assert edge_set(pkg, include_weights=False) == {(1, 2), (2, 3), (3, 1), (2, 99)}

    # Weighted (Option B) merges parallel edges too: 1->2 has 2+1 = 3
    assert edge_set(pkg, include_weights=True) == {(1, 2, 3), (2, 3, 3), (3, 1, 4), (2, 99, 10)}


def test_parallel_edges_are_merged_in_build_digraph():
    pkg = _toy_pkg_all_features()
    G = _build_digraph(pkg)
    assert G[1][2]["weight"] == 3


def test_average_out_in_total_degree_weighted():
    pkg = _toy_pkg_all_features()

    # Weighted out over entities {1,2,3,4}
    # out(1)=3, out(2)=3+10=13, out(3)=4, out(4)=0
    assert average_out_degree(pkg, weighted=True) == pytest.approx((3 + 13 + 4 + 0) / 4)

    # Weighted in over entities
    # in(1)=4, in(2)=3, in(3)=3, in(4)=0
    assert average_in_degree(pkg, weighted=True) == pytest.approx((4 + 3 + 3 + 0) / 4)

    # Total degrees (in+out): [7,16,7,0]
    assert average_degree(pkg, weighted=True) == pytest.approx((7 + 16 + 7 + 0) / 4)


def test_average_out_in_total_degree_unweighted():
    pkg = _toy_pkg_all_features()

    # Unweighted out-degree in merged DiGraph:
    # out(1)=1 (to {2}), out(2)=2 (to {3,99}), out(3)=1 (to {1}), out(4)=0
    assert average_out_degree(pkg, weighted=False) == pytest.approx((1 + 2 + 1 + 0) / 4)

    # Unweighted in-degree:
    # in(1)=1 (from {3}), in(2)=1 (from {1}), in(3)=1 (from {2}), in(4)=0
    assert average_in_degree(pkg, weighted=False) == pytest.approx((1 + 1 + 1 + 0) / 4)

    # Total unweighted degrees: node1=2, node2=3, node3=2, node4=0
    assert average_degree(pkg, weighted=False) == pytest.approx((2 + 3 + 2 + 0) / 4)


def test_tmh_matches_paper_degree_in_plus_out_weighted():
    pkg = _toy_pkg_all_features()

    # totals per entity (weighted): [7,16,7,0]
    # TMH = (49 + 256 + 49 + 0) / (7 + 16 + 7 + 0) = 354/30
    assert tmh_tendency_to_make_hubs(pkg, weighted=True) == pytest.approx(354 / 30)


def test_tmh_matches_paper_degree_in_plus_out_unweighted():
    pkg = _toy_pkg_all_features()

    # totals per entity (unweighted): [2,3,2,0]
    # TMH = (4 + 9 + 4 + 0) / (2 + 3 + 2 + 0) = 17/7
    assert tmh_tendency_to_make_hubs(pkg, weighted=False) == pytest.approx(17 / 7)


def test_components_cycle_and_isolated():
    pkg = _toy_pkg_all_features()

    # Weak components (undirected):
    # {1,2,3,99} and {4} -> both contain an entity EQC -> 2
    assert connected_components_count(pkg, weak=True) == 2

    # Strong components:
    # SCC1: {1,2,3} (cycle), SCC2: {4}, SCC3: {99}
    # Count only SCCs containing an entity EQC -> {1,2,3} and {4} => 2
    assert connected_components_count(pkg, weak=False) == 2
