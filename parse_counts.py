#!/usr/bin/env python3
"""parse_counts.py -- report RDFParser's deduplication counters for one file."""
import re, sys
from RDFparser import RDFParser

path = sys.argv[1]
p = RDFParser(path)
if "--keep-lang" in sys.argv:
    p._strip_lang_tag = re.compile(r"(?!x)(x)(x)")     # never matches
for _ in p.parse():
    pass
print(f"{path}")
print(f"  lines        {p.counter_lines:,}")
print(f"  distinct     {p.counter_yielded:,}")
print(f"  duplicates   {p.counter_duplicates:,}")
print(f"  non-matching {p.counter_NM:,}")