from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Iterable, Optional
from hashing import hash_id

# Destination types in the object slot. "union" keeps the union of the types of
# all successors reached by a predicate, which is the behaviour of FluidSpark's
# schema elements up to merge order; "sets" keeps one type set per successor,
# the literal reading of Definition 4. See Appendix~\ref{app:dsttypes}.
DST_TYPE_MODE = "union"


class VertexSummary:
    #FLUID-style summarization
    #types: rdf:type class IRIs of the source vertex
    #neighbors: predicate--count of outgoing uses
    #neighbor_dst_types: predicate -- set of destination class IRIs (dst rdf:types)
    #payload:set of named graph IRIs(sources)
    #instances:set of subject IRIs this summary represents
    #labels:for debug can hold human labels or class IRIs

    # Typ signature (get_id):
    #   (types, set(predicates), per-predicate dst type sets)

    # constructor
    def __init__(self, types: Optional[Iterable[str]] = None):
        self.types: Set[str] = set(types or [])
        self.neighbors: Dict[str, int] = defaultdict(int)
        # predicate -> set of destination CLASSES, each a frozenset of types.
        # A set of type sets, not a flat union: two objects typed {Person} and
        # {Employee} are two classes under ~_o, and must not collapse into one
        # object typed {Person, Employee}.
        self.neighbor_dst_types: Dict[str, Set[frozenset]] = defaultdict(set)
        self.payload: Set[str] = set()
        self.instances: Set[str] = set()
        self.labels: Set[str] = set()  # optional; you may ignore if unused
        # mu(s, P, O): mass sent toward object class O via predicate class P.
        # Summed here, binned once in seal(), never per edge.
        self.mass: Dict[tuple, float] = defaultdict(float)
        self.bins: Optional[Dict[tuple, int]] = None

    #adds rdf:type class to the summary
    def add_type(self, t: Optional[str]) -> None:
        if t:
            self.types.add(t)

    def add_types(self, ts: Iterable[str]) -> None:
        for t in ts:
            if t:
                self.types.add(t)
    #neighbors increase the count and update the set
    def add_neighbor(self, predicate_iri: Optional[str]) -> None:
        if predicate_iri:
            self.neighbors[predicate_iri] += 1
    #for the given predicate (union in dst type)
    def add_dst_types(self, predicate_iri: Optional[str], dst_types: Iterable[str]) -> None:
        if not predicate_iri:
            return
        bucket = self.neighbor_dst_types[predicate_iri]
        if DST_TYPE_MODE == "sets":
            bucket.add(frozenset(t for t in dst_types if t))
        else:
            for t in dst_types:
                if t:
                    bucket.add(t)
    def add_mass(self, pred_key, dst_key, prob: Optional[float]) -> None:
        # Accumulate the probability mass of one triple into its (P, O) cell.
        if pred_key is None or not prob:
            return
        self.mass[(pred_key, dst_key)] += prob
    #record source
    def add_payload(self, graph_iri: Optional[str]) -> None:
        if graph_iri:
            self.payload.add(graph_iri)
    #subject IRI that this summary blongs to
    def add_instance(self, subject_iri: Optional[str]) -> None:
        if subject_iri:
            self.instances.add(subject_iri)
    #opt human readable names
    def add_label(self, label: Optional[str]) -> None:
        if label:
            self.labels.add(label)

    #merge -> Scala-mergeMessage
    #merges 2 summarias and updates on vertex -> combines sets inside one summary
    def merge(self, other: "VertexSummary") -> "VertexSummary":
        #Union of sets and sum neighbor predicate counts 
        #union "per predicat" of neighbor_dst_types
        if other is self:
            return self
        self.types |= other.types
        self.payload |= other.payload
        self.instances |= other.instances
        self.labels |= other.labels
        for p, c in other.neighbors.items():
            self.neighbors[p] += c
        for p, dsts in other.neighbor_dst_types.items():
            self.neighbor_dst_types[p].update(dsts)
        for k, v in other.mass.items():
            self.mass[k] += v
        return self

    def seal(self, b) -> "VertexSummary":
        # Apply the binning once all messages for this vertex are merged.
        # Must not run per edge: b is not additive, so b(0.3) and b(0.3) say
        # nothing about b(0.6). Not calling seal leaves the summary
        # deterministic.
        self.bins = {k: b(v) for k, v in self.mass.items()}
        return self
    def _dst_bucket_parts(self, bucket) -> list[str]:
        # deterministic serialisation of one predicate's destination-type bucket
        if DST_TYPE_MODE == "sets":
            return sorted("\u241E".join(sorted(ts)) for ts in bucket)
        return sorted(bucket)
    #signature / ID
    def get_id(self) -> int:
        # build signature string and hash it
        # signature components:
        # 1-sorted(types)
        # 2-sorted(outgoing predicate keys)
        # 3-for each predicate (sorted by name), the sorted dst type set
        parts: list[str] = []

        # 1 types
        if self.types:
            parts.extend(sorted(self.types))
        parts.append("|#|")

        # 2 outgoing predicates (set)
        if self.neighbors:
            parts.extend(sorted(self.neighbors.keys()))
        parts.append("|@|")

        # 3 predicate -> dst type sets (schema-aware behavior)
        if self.neighbor_dst_types:
            for p in sorted(self.neighbor_dst_types.keys()):
                parts.append(p)
                parts.extend(self._dst_bucket_parts(self.neighbor_dst_types[p]))
        if self.bins:
            parts.append("|b|")
            for key in sorted(self.bins, key=lambda k: (str(k[0]), str(k[1]))):
                p, o = key
                parts.append(f"{p}\u241E{o}\u241E{self.bins[key]}")
        signature = "\u241F".join(parts)  # reduce collisions
        return hash_id(signature)

    # export
    def as_dict(self) -> Dict[str, object]:
        return {
            "types": sorted(self.types),
            "predicates": sorted(self.neighbors.keys()),
            "predicate_counts": dict(self.neighbors),
            "predicate_to_dst_types": {
                p: self._dst_bucket_parts(bucket)
                for p, bucket in self.neighbor_dst_types.items()
            },
            "payload_graphs": sorted(self.payload),
            "instances": sorted(self.instances),
            "labels": sorted(self.labels),
            "mass": {f"{p}|{o}": v for (p, o), v in self.mass.items()},
            "bins": None if self.bins is None
                    else {f"{p}|{o}": v for (p, o), v in self.bins.items()},
            "id": self.get_id(),
        }

    def __repr__(self) -> str:
        return f"VertexSummary(id={self.get_id()}, types={len(self.types)}, preds={len(self.neighbors)})"
