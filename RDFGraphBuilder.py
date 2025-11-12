# The module
import networkx as nx
from hashing import hash_id

# Constants for RDF predicates for class signal
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"

#constructor with 3 switches 
class RDFGraphBuilder:
    def __init__(self, include_literals=False, use_numeric_ids=True, add_reverse_edges=False):
        #include_literals: bool = False = skip edges to literal objs
        #TD : literal should be collapced : as one big vertex
        #use_numeric_ids: bool = True = use numeric IDs for nodes saves memory
        #add_reverse_edges: bool = True = add reverse copy of each none-literal edges can add memory overhead
        #put the default on false
        self.include_literals = include_literals
        self.use_numeric_ids = use_numeric_ids
        self.add_reverse_edges = add_reverse_edges
        self.G = nx.MultiDiGraph()

        # NEW: canonical id (token) used when include_literals == False
        #      All literal objects will point to this single node.
        self._CANON_LIT = "__LITERAL__"  # human-readable label
        self._CANON_LIT_ID = self._node_id(self._CANON_LIT)  # respects numeric-id setting

    def _node_id(self, term: str):
        #if numeric id is true hashed int otherwise return the term as is
        return hash_id(term) if self.use_numeric_ids else term

    def _is_literal(self, term: str) -> bool:
        # Check if the term is a literal by looking for quotes
        return term.startswith('"')
    
    # def _is_blank_node(self, term: str) -> bool:
    #     #check if a node is a blank node
    #     return term.startswith("_:")

    def add_statement(self, s: str, p: str, o: str, g: str):
        #types to vertices not edges
        if p == RDF_TYPE and not self._is_literal(o):
            sid = self._node_id(s)
            self.G.add_node(sid, label=s)
            tset = self.G.nodes[sid].setdefault("types", set())
            tset.add(o)  # store the class URI (with <>)
            #print(f"Added type {o} to node {s}")
            return

        # CHANGED: do NOT drop literal objects when include_literals == False.
        #          Instead, collapse ALL literals into a single canonical node.
        is_lit_obj = self._is_literal(o)
        if is_lit_obj:
            if self.include_literals:
                # keep the concrete literal as its own node
                oid = self._node_id(o)
                oid_label = o
                oid_types = {"Literal"}  # NEW: give literal nodes a 'Literal' type so dstTypes can see it
            else:
                # collapse all literals to a single shared node
                oid = self._CANON_LIT_ID
                oid_label = self._CANON_LIT
                oid_types = {"Literal"}
        else:
            oid = self._node_id(o)
            oid_label = o
            oid_types = None  # non-literal: types handled via rdf:type triples

        #adds the predicate and context on each edge
        sid = self._node_id(s)
        self.G.add_node(sid, label=s)
        self.G.add_node(oid, label=oid_label)
        if oid_types is not None:
            # NEW: ensure the literal node (concrete or canonical) has a type so SchemEX dstTypes sees 'Literal'
            tset = self.G.nodes[oid].setdefault("types", set())
            tset.update(oid_types)

        self.G.add_edge(sid, oid, predicate=p, graph=g)

        #if reverse edges are enabled, add a reverse edge for non-literal objects
        #graph in both directions
        #but can increase memory usage significantly
        #only adds if the object is not a literal
        #and if add_reverse_edges is true
        # CHANGED: use 'is_lit_obj' to avoid adding reverse edges to the canonical literal node.
        if self.add_reverse_edges and not is_lit_obj:
            self.G.add_edge(oid, sid, predicate=p, graph=g, _reversed=True)

    def build(self, triple_iterable):
        for s, p, o, g in triple_iterable:
            self.add_statement(s, p, o, g)
        return self.G
