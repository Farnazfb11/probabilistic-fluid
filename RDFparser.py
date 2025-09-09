import re, gzip

class RDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
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
                m = self.pattern.match(line)
                if m:
                    s = m.group('s')
                    p = m.group('p')
                    o = m.group('o')
                    g = m.group('g') if m.group('g') else None
                    yield (s, p, o, g)
                else:
                    print("No match found", line)
