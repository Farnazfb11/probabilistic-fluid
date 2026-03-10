# test_binary_metrics.py
import pytest
import math
from Binary_measure import (
    jaccard_vertices,
    jaccard_edges,
    graph_edit_distance_structural,
    symmetric_kl_divergence,
    earth_mover_distance_degree,
)


# ============================================================================
# FIXTURES - Test Data Packages
# ============================================================================

@pytest.fixture
def pkg_empty():
    """Empty package - no nodes or edges."""
    return {
        "nodes": {},
        "edges": {},
        "membership": {}
    }


@pytest.fixture
def pkg_single_class():
    """Single equivalence class with 100 entities."""
    return {
        "nodes": {
            "1": {"vcp": 100, "dsp": 0}
        },
        "edges": {},
        "membership": {f"e{i}": 1 for i in range(100)}
    }


@pytest.fixture
def pkg_two_classes_no_edges():
    """Two classes, no edges between them."""
    return {
        "nodes": {
            "1": {"vcp": 50, "dsp": 0},
            "2": {"vcp": 50, "dsp": 0}
        },
        "edges": {},
        "membership": {
            **{f"e{i}": 1 for i in range(50)},
            **{f"f{i}": 2 for i in range(50)}
        }
    }


@pytest.fixture
def pkg_two_classes_with_edge():
    """Two classes connected by one edge."""
    return {
        "nodes": {
            "1": {"vcp": 60, "dsp": 1},
            "2": {"vcp": 40, "dsp": 1}
        },
        "edges": {
            "e1": {"src": 1, "dst": 2, "predicate": "http://ex/knows", "weight": 10}
        }
    }


@pytest.fixture
def pkg_three_classes_complex():
    """Three classes with multiple edges forming a triangle."""
    return {
        "nodes": {
            "1": {"vcp": 30, "dsp": 2},
            "2": {"vcp": 40, "dsp": 2},
            "3": {"vcp": 30, "dsp": 2}
        },
        "edges": {
            "e1": {"src": 1, "dst": 2, "predicate": "http://ex/knows", "weight": 5},
            "e2": {"src": 2, "dst": 3, "predicate": "http://ex/worksAt", "weight": 8},
            "e3": {"src": 3, "dst": 1, "predicate": "http://ex/knows", "weight": 3}
        }
    }


# ============================================================================
# JACCARD VERTICES TESTS
# ============================================================================

class TestJaccardVertices:
    
    def test_identical_packages(self, pkg_two_classes_no_edges):
        """Identical packages should have distance 0."""
        result = jaccard_vertices(pkg_two_classes_no_edges, pkg_two_classes_no_edges)
        assert result == 0.0
    
    def test_empty_packages(self, pkg_empty):
        """Two empty packages should have distance 0."""
        result = jaccard_vertices(pkg_empty, pkg_empty)
        assert result == 0.0
    
    def test_completely_different_classes(self):
        """No overlap in equivalence classes → distance 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"3": {"vcp": 30}, "4": {"vcp": 70}},
            "edges": []
        }
        result = jaccard_vertices(pkg_t, pkg_tp1)
        assert result == 1.0
    
    def test_partial_overlap(self):
        """Partial overlap: {1,2} vs {2,3} → distance = 1 - 1/3."""
        pkg_t = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"2": {"vcp": 60}, "3": {"vcp": 40}},
            "edges": []
        }
        result = jaccard_vertices(pkg_t, pkg_tp1)
        # Intersection: {2}, Union: {1,2,3}
        # Jaccard = 1 - 1/3 = 2/3
        assert abs(result - 2/3) < 1e-9
    
    def test_one_class_added(self):
        """Adding one class: {1} vs {1,2} → distance = 1 - 1/2."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        result = jaccard_vertices(pkg_t, pkg_tp1)
        assert result == 0.5


# ============================================================================
# JACCARD EDGES TESTS
# ============================================================================

class TestJaccardEdges:
    
    def test_identical_edges(self, pkg_two_classes_with_edge):
        """Identical edge sets → distance 0."""
        result = jaccard_edges(pkg_two_classes_with_edge, pkg_two_classes_with_edge)
        assert result == 0.0
    
    def test_no_edges_both(self, pkg_two_classes_no_edges):
        """No edges in either package → distance 0."""
        result = jaccard_edges(pkg_two_classes_no_edges, pkg_two_classes_no_edges)
        assert result == 0.0
    
    def test_completely_different_edges(self):
        """No edge overlap → distance 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {"e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1}}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {"e1": {"src": 2, "dst": 1, "predicate": "p", "weight": 1}}
        }
        result = jaccard_edges(pkg_t, pkg_tp1)
        # Different edges: (1,2) vs (2,1)
        assert result == 1.0
    
    def test_partial_edge_overlap(self):
        """Partial edge overlap: 2 common, 1 different each."""
        pkg_t = {
            "nodes": {"1": {"vcp": 30}, "2": {"vcp": 30}, "3": {"vcp": 40}},
            "edges": {
                "e1": {"src": 1, "dst": 2, "predicate": "p1", "weight": 1},
                "e2": {"src": 2, "dst": 3, "predicate": "p2", "weight": 1},
                "e3": {"src": 1, "dst": 3, "predicate": "p3", "weight": 1}
            }
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 30}, "2": {"vcp": 30}, "3": {"vcp": 40}},
            "edges": {
                "e1": {"src": 1, "dst": 2, "predicate": "p1", "weight": 1},
                "e2": {"src": 2, "dst": 3, "predicate": "p2", "weight": 1},
                "e4": {"src": 3, "dst": 1, "predicate": "p4", "weight": 1}
            }
        }
        result = jaccard_edges(pkg_t, pkg_tp1)
        # Intersection: {(1,2), (2,3)} = 2
        # Union: {(1,2), (2,3), (1,3), (3,1)} = 4
        # Jaccard distance = 1 - 2/4 = 0.5
        assert result == 0.5


# ============================================================================
# GRAPH EDIT DISTANCE TESTS
# ============================================================================

class TestGraphEditDistance:
    
    def test_identical_graphs(self, pkg_two_classes_with_edge):
        """Identical graphs → GED = 0."""
        result = graph_edit_distance_structural(
            pkg_two_classes_with_edge, 
            pkg_two_classes_with_edge
        )
        assert result == 0
    
    def test_one_edge_added(self):
        """Adding one edge → GED = 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {"e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1}}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        assert result == 1
    
    def test_one_edge_removed(self):
        """Removing one edge → GED = 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {"e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1}}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 50}},
            "edges": {}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        assert result == 1
    
    def test_edge_moved(self):
        """Moving edge from (1,2) to (1,3) → GED = 2 (1 delete + 1 add)."""
        pkg_t = {
            "nodes": {"1": {"vcp": 30}, "2": {"vcp": 30}, "3": {"vcp": 40}},
            "edges": {"e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1}}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 30}, "2": {"vcp": 30}, "3": {"vcp": 40}},
            "edges": {"e2": {"src": 1, "dst": 3, "predicate": "p", "weight": 1}}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        assert result == 2
    
    def test_isolated_class_created(self):
        """Creating isolated class (no edges) → GED = 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": {}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 100}, "2": {"vcp": 50}},
            "edges": {}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        assert result == 1
    
    def test_isolated_class_deleted(self):
        """Deleting isolated class → GED = 1."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}, "2": {"vcp": 50}},
            "edges": {}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 100}},
            "edges": {}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        assert result == 1
    
    def test_class_created_with_edges(self):
        """Creating class with edges → edges counted, not class creation."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": {}
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": {"e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1}}
        }
        result = graph_edit_distance_structural(pkg_t, pkg_tp1)
        # Only the edge addition counts (class has edges, so not isolated)
        assert result == 1


# ============================================================================
# SYMMETRIC KL DIVERGENCE TESTS
# ============================================================================

class TestSymmetricKLDivergence:
    
    def test_identical_distributions(self, pkg_two_classes_no_edges):
        """Identical distributions → KLD = 0."""
        result = symmetric_kl_divergence(pkg_two_classes_no_edges, pkg_two_classes_no_edges)
        assert result == 0.0
    
    def test_size_change_same_classes(self):
        """Same classes, different sizes → non-zero KLD."""
        pkg_t = {
            "nodes": {"1": {"vcp": 60}, "2": {"vcp": 40}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 40}, "2": {"vcp": 60}},
            "edges": []
        }
        result = symmetric_kl_divergence(pkg_t, pkg_tp1)
        # Should be positive and symmetric
        assert result > 0
        # Verify symmetry
        reverse = symmetric_kl_divergence(pkg_tp1, pkg_t)
        assert abs(result - reverse) < 1e-9
    
    def test_class_appears(self):
        """New class appears → KLD > 0."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        result = symmetric_kl_divergence(pkg_t, pkg_tp1)
        assert result > 0
    
    def test_class_disappears(self):
        """Class disappears → KLD > 0."""
        pkg_t = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        result = symmetric_kl_divergence(pkg_t, pkg_tp1)
        assert result > 0
    
    def test_symmetry_property(self):
        """KLD should be symmetric."""
        pkg_t = {
            "nodes": {"1": {"vcp": 30}, "2": {"vcp": 70}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 60}, "3": {"vcp": 40}},
            "edges": []
        }
        kld_forward = symmetric_kl_divergence(pkg_t, pkg_tp1)
        kld_backward = symmetric_kl_divergence(pkg_tp1, pkg_t)
        assert abs(kld_forward - kld_backward) < 1e-9
    
    def test_using_membership_fallback(self):
        """Should work when vcp not available, using membership."""
        pkg_t = {
            "nodes": {"1": {}, "2": {}},  # No vcp
            "membership": {
                **{f"e{i}": 1 for i in range(60)},
                **{f"f{i}": 2 for i in range(40)}
            }
        }
        pkg_tp1 = {
            "nodes": {"1": {}, "2": {}},
            "membership": {
                **{f"e{i}": 1 for i in range(40)},
                **{f"f{i}": 2 for i in range(60)}
            }
        }
        result = symmetric_kl_divergence(pkg_t, pkg_tp1)
        assert result > 0


# ============================================================================
# EARTH MOVER'S DISTANCE TESTS
# ============================================================================

class TestEarthMoverDistance:
    
    def test_identical_distributions(self, pkg_two_classes_no_edges):
        """Identical distributions → EMD = 0."""
        result = earth_mover_distance_degree(pkg_two_classes_no_edges, pkg_two_classes_no_edges)
        assert result == 0.0
    
    def test_no_change(self, pkg_single_class):
        """No change in sizes → EMD = 0."""
        result = earth_mover_distance_degree(pkg_single_class, pkg_single_class)
        assert result == 0.0
    
    def test_payload_shift_same_class(self):
        """Payload changes within same class → EMD = 0 (distance to self = 0)."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 100}},  # Same size
            "edges": []
        }
        result = earth_mover_distance_degree(pkg_t, pkg_tp1)
        assert result == 0.0
    
    def test_payload_moved_between_classes(self):
        """Payload moves between different classes → EMD > 0."""
        pkg_t = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 60}, "2": {"vcp": 40}},
            "edges": []
        }
        result = earth_mover_distance_degree(pkg_t, pkg_tp1)
        # 20 entities moved from class 1 to class 2
        # With simplified uniform distance: should be 20
        assert result == 20.0
    
    def test_new_class_appears(self):
        """New class appears → cost of creating entities."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        result = earth_mover_distance_degree(pkg_t, pkg_tp1)
        # 20 entities moved from class 1 to new class 2
        assert result == 20.0
    
    def test_class_disappears(self):
        """Class disappears → cost of removing entities."""
        pkg_t = {
            "nodes": {"1": {"vcp": 80}, "2": {"vcp": 20}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 100}},
            "edges": []
        }
        result = earth_mover_distance_degree(pkg_t, pkg_tp1)
        # 20 entities moved from class 2 to class 1
        assert result == 20.0
    
    def test_complex_redistribution(self):
        """Complex redistribution across multiple classes."""
        pkg_t = {
            "nodes": {
                "1": {"vcp": 50},
                "2": {"vcp": 30},
                "3": {"vcp": 20}
            },
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {
                "1": {"vcp": 30},  # -20
                "2": {"vcp": 40},  # +10
                "3": {"vcp": 30}   # +10
            },
            "edges": []
        }
        result = earth_mover_distance_degree(pkg_t, pkg_tp1)
        # Total movement: 20 entities redistributed
        assert result == 20.0
    
    def test_symmetry_property(self):
        """EMD should be symmetric."""
        pkg_t = {
            "nodes": {"1": {"vcp": 70}, "2": {"vcp": 30}},
            "edges": []
        }
        pkg_tp1 = {
            "nodes": {"1": {"vcp": 40}, "2": {"vcp": 60}},
            "edges": []
        }
        emd_forward = earth_mover_distance_degree(pkg_t, pkg_tp1)
        emd_backward = earth_mover_distance_degree(pkg_tp1, pkg_t)
        assert abs(emd_forward - emd_backward) < 1e-9


# ============================================================================
# INTEGRATION TESTS - Multiple Metrics Together
# ============================================================================

class TestIntegration:
    
    def test_all_metrics_on_identical_packages(self, pkg_three_classes_complex):
        """All metrics should return 0 for identical packages."""
        pkg = pkg_three_classes_complex
        
        assert jaccard_vertices(pkg, pkg) == 0.0
        assert jaccard_edges(pkg, pkg) == 0.0
        assert graph_edit_distance_structural(pkg, pkg) == 0
        assert symmetric_kl_divergence(pkg, pkg) == 0.0
        assert earth_mover_distance_degree(pkg, pkg) == 0.0
    
    def test_metrics_increase_with_change(self):
        """As changes increase, most metrics should increase."""
        pkg_t = {
            "nodes": {"1": {"vcp": 100}},
            "edges": {}
        }
        
        # Small change
        pkg_small = {
            "nodes": {"1": {"vcp": 90}, "2": {"vcp": 10}},
            "edges": {}
        }
        
        # Large change
        pkg_large = {
            "nodes": {"1": {"vcp": 50}, "2": {"vcp": 30}, "3": {"vcp": 20}},
            "edges": {
                "e1": {"src": 1, "dst": 2, "predicate": "p", "weight": 1},
                "e2": {"src": 2, "dst": 3, "predicate": "p", "weight": 1}
            }
        }
        
        # Jaccard vertices
        jac_small = jaccard_vertices(pkg_t, pkg_small)
        jac_large = jaccard_vertices(pkg_t, pkg_large)
        assert jac_large > jac_small
        
        # EMD
        emd_small = earth_mover_distance_degree(pkg_t, pkg_small)
        emd_large = earth_mover_distance_degree(pkg_t, pkg_large)
        assert emd_large > emd_small


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    
    def test_empty_vs_nonempty(self, pkg_empty, pkg_single_class):
        """Comparing empty to non-empty packages."""
        jac_v = jaccard_vertices(pkg_empty, pkg_single_class)
        assert jac_v == 1.0  # Completely different
        
        jac_e = jaccard_edges(pkg_empty, pkg_single_class)
        assert jac_e == 0.0  # Both have no edges
    
    def test_missing_vcp_and_membership(self):
        """Should raise error when both vcp and membership missing."""
        pkg_invalid = {
            "nodes": {"1": {}},  # No vcp
            "edges": {}
            # No membership either
        }
        
        with pytest.raises(ValueError, match="Cannot derive EQC size map"):
            symmetric_kl_divergence(pkg_invalid, pkg_invalid)
    
    def test_single_node_graphs(self):
        """Single node graphs should work correctly."""
        pkg = {
            "nodes": {"1": {"vcp": 100}},
            "edges": {}
        }
        
        assert jaccard_vertices(pkg, pkg) == 0.0
        assert graph_edit_distance_structural(pkg, pkg) == 0
        assert symmetric_kl_divergence(pkg, pkg) == 0.0