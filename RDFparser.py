import re, gzip

class RDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        # More permissive N-Quads/N-Triples pattern
        self.pattern = re.compile(r"""
            ^\s*
            (?P<s>(<[^>]*>|_:[^\s]+))            # subject: IRI or blank node
            \s+
            (?P<p><[^>]*>)                       # predicate: IRI
            \s+
            (?P<o>                               
                "(?:[^"\\]|\\.)*"               # literal string (handles escapes)
                (?:@[A-Za-z0-9][A-Za-z0-9-]*    # 1st language tag segment
                   (?:@[A-Za-z0-9][A-Za-z0-9-]*)* # allow extra @segments (nonstandard)
                )?
                (?:\^\^<[^>]*>)?                # optional datatype
              |
                <[^>]*>                         # OR IRI object
              |
                _:[^\s]+                        # OR blank node object
            )
            (?:\s+(?P<g>(<[^>]*>|_:[^\s]+)))?   # optional graph term (IRI or blank)
            \s*\.\s*$
        """, re.UNICODE | re.VERBOSE)

    def parse(self):
        # tolerant decoding; replace undecodable bytes instead of failing
        with gzip.open(self.file_path, 'rt', encoding='utf-8', errors='replace') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = self.pattern.match(line)
                if m:
                    s = m.group('s')
                    p = m.group('p')
                    o = m.group('o')
                    g = m.group('g') if m.group('g') else None
                    # Return a 4-tuple; use None when graph is absent
                    yield (s, p, o, g)
                else:
                    # Keep the log, but it won't crash your pipeline
                    print("No match found", line)
