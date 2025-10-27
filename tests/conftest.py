# tests/conftest.py
# --- Make project imports work when running `python -m pytest` from repo root ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Shared fixtures and your minigraph ---
import pytest
import networkx as nx
import gzip

from RDFGraphBuilder import RDF_TYPE
from RDFGraphBuilder import RDFGraphBuilder  # (not used below but handy to have)

@pytest.fixture
def build_minigraph():
    """
    Nodes:
      a(Person):  knows -> x(Person), worksAt -> y(Org)
      b(Person):  knows -> y(Org),    worksAt -> y(Org)
      e(Person):  name  -> n1(untyped)
      g(Org):     knows -> x(Person)
      d(Person):  (no outgoing)
    Objects / targets:
      x(Person), y(Org), n1(untyped)
    """
    G = nx.MultiDiGraph()

    # Subjects
    G.add_node("a", label="a", types=["http://ex/Person"])
    G.add_node("b", label="b", types=["http://ex/Person"])
    G.add_node("e", label="e", types=["http://ex/Person"])
    G.add_node("g", label="g", types=["http://ex/Org"])
    G.add_node("d", label="d", types=["http://ex/Person"])  # no outgoing

    # Objects / targets
    G.add_node("x", label="x", types=["http://ex/Person"])
    G.add_node("y", label="y", types=["http://ex/Org"])
    G.add_node("n1", label="n1", types=[])

    # Edges with datasets (graphs)
    G.add_edge("a", "x", predicate="http://ex/knows",    graph="G1")
    G.add_edge("a", "y", predicate="http://ex/worksAt",  graph="G1")

    G.add_edge("b", "y", predicate="http://ex/knows",    graph="G2")
    G.add_edge("b", "y", predicate="http://ex/worksAt",  graph="G2")

    G.add_edge("e", "n1", predicate="http://ex/name",    graph="G3")

    G.add_edge("g", "x", predicate="http://ex/knows",    graph="G4")

    return G
