import re, gzip
from typing import Iterator, Tuple, Optional
class RDFParser: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.counter_NM = 0
        # CHANGED: counter for how many non-matching lines we actually log
        # WHY: avoid massive logs by only printing a small sample.
        self.counter_NM_logged = 0

        # Strip any language tag after a literal, keep datatype if present
        # "foo"@en          -> "foo"
        # "foo"@en^^<dt>    -> "foo"^^<dt>
        self._strip_lang_tag = re.compile(r'("(?:(?:[^"\\]|\\.)*)")\s*@[^\s^]+')
        # CHANGED: generic pattern to strip ANY language tag after a literal
        # Example:
        #   "text"@en         -> "text"
        #   "23/swe"@*        -> "23/swe"
        #   "foo"@en-US       -> "foo"
        #   "foo"@en^^<...>   -> "foo"^^<...>   (datatype preserved)
        #
        # It matches: " ... " @ <any run of non-space, non-^ chars>
        # and keeps only the " ... " part.

        # Permissive N-Quads/N-Triples pattern
        self.pattern = re.compile(r"""
            ^\s*
            (?P<s>(<[^>]*>|_:[^\s]+))
            \s+
            (?P<p><[^>]*>)
            \s+
            (?P<o>
                "(?:[^"\\]|\\.)*"
                (?:\s*@[A-Za-z0-9_][A-Za-z0-9_-]*(?:@[A-Za-z0-9_][A-Za-z0-9_-]*)*)?
                (?:\^\^<[^>]*>)?
              |
                <[^>]*>
              |
                _:[^\s]+
            )
            (?:\s+(?P<g>(<[^>]*>|_:[^\s]+)))?
            \s*\.\s*$
        """, re.UNICODE | re.VERBOSE)

        # For fast checks
        self._iri_trailing_cleanup = re.compile(r'([/#])>$')  # "<.../>" or "<...#>" before ">"
        self._is_iri = re.compile(r'^<[^>]*>$')
        self._is_bnode = re.compile(r'^_:[^\s]+$')
        self._is_literal = re.compile(r'^"')

    def _canonicalize_iri(self, term: str) -> str:
        """
        Justin-like behavior:
        - treat <foo>, <foo/>, <foo#> as equal
        - ignore IRI casing (lowercase)
        """
        if not self._is_iri.match(term):
            return term
        t = term.strip()

        # remove trailing / or # right before >
        t = self._iri_trailing_cleanup.sub('>', t)

        # lowercase the IRI content
        inner = t[1:-1].lower()
        return f"<{inner}>"
    
    def _canonicalize_bnode(self, term: str) -> str:
        """
        Optional: if you want case-insensitive blank node ids too.
        Usually _:-ids are case-sensitive, but Justin’s “uppercase whole line
        stays equal” test suggests he might be normalizing these as well.
        """
        if not self._is_bnode.match(term):
            return term
        return "_:" + term[2:].lower()

    def _canonicalize_term(self, term: str) -> str:
        """
        Canonicalize IRIs and blank nodes.
        Keep literals as-is (except language tag stripping done earlier).
        """
        if not term:
            return term
        if term.startswith('"'):
            return term
        if term.startswith("<"):
            return self._canonicalize_iri(term)
        if term.startswith("_:"):
            return self._canonicalize_bnode(term)
        return term

    def parse(self) -> Iterator[Tuple[str, str, str, Optional[str]]]:
        with gzip.open(self.file_path, 'rt', encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # 1) Strip language tags on literals
                line = self._strip_lang_tag.sub(r"\1", line)

                m = self.pattern.match(line)
                if not m:
                    self.counter_NM += 1
                    if self.counter_NM_logged < 200:
                        print("No match found", line)
                        self.counter_NM_logged += 1
                    continue

                # 2) Canonicalize parsed terms
                s = self._canonicalize_term(m.group("s"))
                p = self._canonicalize_term(m.group("p"))
                o = self._canonicalize_term(m.group("o"))
                g_raw = m.group("g")
                g = self._canonicalize_term(g_raw) if g_raw else None

                yield (s, p, o, g)

        if self.counter_NM > 0:
            print(f"[RDFParser] Total non-matching lines: {self.counter_NM}")