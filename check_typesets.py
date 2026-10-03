#!/usr/bin/env python3
"""
check_typesets.py
=================

Reconcile the class collection's class count with the distinct_type_sets
figure reported by dataset_stats.py.

    python check_typesets.py out/watdiv10m/graph_weighted.wnt \
                             out/watdiv10m/vertex_types.tsv

Why they can disagree
---------------------
CC signs a vertex by its type set alone, so its class count is the number of
distinct type sets over the vertices the summariser sees, plus one for the
vertices carrying no type at all. The summariser sees every vertex of the
weighted-triple file *and* every vertex named in vertex_types.tsv, since
pgraph_io adds typed vertices that appear in no triple.

A statistic computed over a narrower population -- only subjects, say, or only
vertices that occur in the triple file -- will report fewer distinct type sets
than CC produces classes. This prints the count under each population so the
difference can be attributed rather than guessed at.
"""

import sys
from collections import Counter

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

wnt_path, types_path = sys.argv[1], sys.argv[2]

# ---- the types file --------------------------------------------------- #
type_sets = {}
with open(types_path, encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        type_sets[parts[0]] = frozenset(p for p in parts[1:] if p)

# ---- the weighted-triple file ----------------------------------------- #
subjects, objects = set(), set()
with open(wnt_path, encoding="utf-8") as fh:
    for line in fh:
        parts = line.split()
        if len(parts) < 4:
            continue
        subjects.add(parts[0])
        objects.add(parts[3])

graph_vertices = subjects | objects
all_vertices = graph_vertices | set(type_sets)        # what pgraph_io builds


def distinct_sets(population) -> int:
    return len({type_sets[v] for v in population if v in type_sets})


def typed_count(population) -> int:
    return sum(1 for v in population if v in type_sets)


print(f"{'entries in vertex_types.tsv':<42}{len(type_sets):>12,}")
print(f"{'distinct type sets in that file':<42}{len(set(type_sets.values())):>12,}")
print()
print(f"{'vertices in the .wnt (subjects + objects)':<42}{len(graph_vertices):>12,}")
print(f"{'vertices the summariser sees':<42}{len(all_vertices):>12,}")
print(f"{'typed vertices in no triple':<42}"
      f"{len(set(type_sets) - graph_vertices):>12,}")
print()
print("distinct type sets, by population")
for name, pop in (("subjects only", subjects),
                  ("objects only", objects),
                  ("vertices in the .wnt", graph_vertices),
                  ("all vertices the summariser sees", all_vertices)):
    print(f"  {name:<40}{distinct_sets(pop):>12,}   "
          f"({typed_count(pop):,} typed)")

untyped = len(all_vertices) - typed_count(all_vertices)
expected_cc = distinct_sets(all_vertices) + (1 if untyped else 0)
print()
print(f"{'vertices with no type at all':<42}{untyped:>12,}")
print(f"{'=> CC should produce':<42}{expected_cc:>12,}   classes")
print()

# A type set occurring on exactly one vertex is worth seeing: if the gap is
# made of singletons, it is likely a handful of unusual vertices rather than a
# systematic difference in how the two scripts select vertices.
counts = Counter(type_sets[v] for v in all_vertices if v in type_sets)
singletons = [ts for ts, c in counts.items() if c == 1]
print(f"type sets carried by exactly one vertex: {len(singletons)}")
for ts in singletons[:10]:
    print("   " + ", ".join(sorted(ts)))