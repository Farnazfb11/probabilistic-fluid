import re, gzip

class RDFParser: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.counter_NM = 0

        # CHANGED: counter for how many non-matching lines we actually log
        # WHY: avoid massive logs by only printing a small sample.
        self.counter_NM_logged = 0  # CHANGED

        # CHANGED: generic pattern to strip ANY language tag after a literal
        # Example:
        #   "text"@en         -> "text"
        #   "23/swe"@*        -> "23/swe"
        #   "foo"@en-US       -> "foo"
        #   "foo"@en^^<...>   -> "foo"^^<...>   (datatype preserved)
        #
        # It matches: " ... " @ <any run of non-space, non-^ chars>
        # and keeps only the " ... " part.
        self._strip_lang_tag = re.compile(
            r'("(?:(?:[^"\\]|\\.)*)")@[^\s^]+'   # CHANGED
        )

        # Permissive N-Quads/N-Triples pattern
        self.pattern = re.compile(r"""
            ^\s*
            (?P<s>(<[^>]*>|_:[^\s]+))            # subject: IRI or blank node
            \s+
            (?P<p><[^>]*>)                       # predicate: IRI
            \s+
            (?P<o>
                "(?:[^"\\]|\\.)*"               # literal (handles escapes)
                (?:                              # optional language tag(s)
                    @[A-Za-z0-9_][A-Za-z0-9_-]* # allow underscore + hyphen
                    (?:@[A-Za-z0-9_][A-Za-z0-9_-]*)*
                )?
                (?:\^\^<[^>]*>)?                # optional datatype
              |
                <[^>]*>                         # OR IRI object
              |
                _:[^\s]+                        # OR blank node object
            )
            (?:\s+(?P<g>(<[^>]*>|_:[^\s]+)))?   # optional graph term
            \s*\.\s*$
        """, re.UNICODE | re.VERBOSE)

    def parse(self):
        with gzip.open(self.file_path, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # CHANGED: strip ANY language tag after a literal
                # so "literal"@something  -> "literal"
                # This is exactly the "ignore @..." behaviour you asked for.
                line = self._strip_lang_tag.sub(r'\1', line)  # CHANGED

                m = self.pattern.match(line)
                if m:
                    s = m.group('s')
                    p = m.group('p')
                    o = m.group('o')
                    g = m.group('g') if m.group('g') else None
                    yield (s, p, o, g)
                else:
                    self.counter_NM += 1
                    # CHANGED: only log a few examples, not every bad line.
                    if self.counter_NM_logged < 200:          # CHANGED
                        print("No match found", line)        # CHANGED
                        self.counter_NM_logged += 1          # CHANGED

        if self.counter_NM > 0:
            print(f"[RDFParser] Total non-matching lines: {self.counter_NM}")
