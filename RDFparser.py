import re, gzip

class RDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pattern = re.compile(
            r'^(<[^>]+>|_:[^\s]+)\s+'              # G1 subject-> can also be a blank 
            r'(<[^>]+>)\s+'                        # Group 2: Predicate
            r'('
                r'"[^"\\]*(?:\\.[^"\\]*)*"'       # Literal string 
                r'(?:@[a-zA-Z\-]+)?'               # could language tag
                r'(?:\^\^<[^>]+>)?'                # could datatype 
            r'|<[^>]+>'                           # OR URI object
            r'|_:[^\s]+'                         # OR blank node object
            r')\s+'
            r'(<[^>]+>)\s*\.$'                   # Group 4: Graph
        )
#(nU)prefix based on the context(source) and ID use as the IRI
    def parse(self):
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as file:
            for line in file:
                match = self.pattern.match(line.strip())
                if match:
                    yield match.groups()
                else:
                    print("No match found", line)
                    break