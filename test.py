#!/usr/bin/env python3
"""Show the members and signature of a few equivalence classes."""
import sys
from collections import defaultdict
from binning import make_binning
from pgraph_io import load_weighted_graph
from Collector import SchemaCollector
from CompAttClassCollection import SEComplexAttributeClassCollection

def short(x):
    if isinstance(x, frozenset):
        return "{" + ", ".join(sorted(short(t) for t in x)) + "}"
    s = str(x)
    return s.rstrip(">").rsplit("/", 1)[-1].rsplit("#", 1)[-1] or s

wnt, types = sys.argv[1], sys.argv[2]
G = load_weighted_graph(wnt, types)
b = make_binning("equal_width", 5)

summaries, ids = SchemaCollector(SEComplexAttributeClassCollection()).collect(G, binning=b)

groups = defaultdict(list)
for node, sid in ids.items():
    groups[sid].append(node)

for sid, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]:
    print(f"\n=== class {sid} : {len(members)} members ===")
    print("  sample:", members[:3])
    vs = summaries[members[0]]
    print("  types:", sorted(short(t) for t in vs.types))
    print("  predicates:", sorted(short(p) for p in vs.neighbors))
    print("  mass:", {f"{short(p)}|{short(o)}": round(v, 4) for (p, o), v in vs.mass.items()})
    print("  bins:", {f"{short(p)}|{short(o)}": v for (p, o), v in vs.bins.items()})
    # every member must agree on the binned profile
    print("  all members agree on bins:",
          all(summaries[m].bins == vs.bins for m in members))