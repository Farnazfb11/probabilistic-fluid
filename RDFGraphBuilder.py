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
        #if we decide include_literals is false this skips literals obj
        if self._is_literal(o) and not self.include_literals:
            return
        #adds the predicate and context on each edge
        sid = self._node_id(s)
        oid = self._node_id(o)
        self.G.add_node(sid, label=s)
        self.G.add_node(oid, label=o)
        self.G.add_edge(sid, oid, predicate=p, graph=g)
        #if reverse edges are enabled, add a reverse edge for non-literal objects
        #graph in both directions
        #but can increase memory usage significantly
        #only adds if the object is not a literal
        #and if add_reverse_edges is true
        if self.add_reverse_edges and not self._is_literal(o):
            self.G.add_edge(oid, sid, predicate=p, graph=g, _reversed=True)

    def build(self, triple_iterable):
        for s, p, o, g in triple_iterable:
            self.add_statement(s, p, o, g)
        return self.G
