import re, gzip, io
from typing import Iterator, Tuple, Optional, TextIO


class RDFParser:
    """
    Lightweight RDF N-Quads / N-Triples parser for large .gz files.

    What it does
    1) Reads `.nq.gz` or `.nt.gz` as a stream (no full file in memory)
    2) Handles a tricky real world case where a `.gz` contains another `.gz`
       (nested gzip), which otherwise produces garbage characters and regex misses
    3) Normalizes some syntax so you get fewer “different looking but same” nodes:
       - Strips language tags from literals but keeps datatype if present
         Example:  "foo"@en^^<dt>  ->  "foo"^^<dt>
       - Canonicalizes IRIs:
         - makes IRIs case-insensitive by lowercasing the inner part
         - treats trailing `/` or `#` as equivalent:
           <http://x/y>, <http://x/y/>, <http://x/y#> become the same token

    Output format
    Yields 4-tuples: (s, p, o, g)
    - s: subject term as string ("<...>" or "_:...")
    - p: predicate IRI as string "<...>"
    - o: object term as string (literal, IRI, or blank node)
    - g: optional graph name (IRI or blank node), or None if absent

    Notes
    - This is a permissive regex-based parser, not a full RDF spec implementation.
    - It is designed for DYLD/DyLDO style N-Quads where most lines are simple.
    """

    def __init__(self, file_path: str):
        """
        Parameters
        file_path
            Path to a gzip-compressed N-Quads/N-Triples file.
            Works with normal gzip and nested gzip files.
        """
        self.file_path = file_path

        # Count of lines that did not match the N-Quads/N-Triples regex.
        self.counter_NM = 0

        # We only print a small sample of bad lines to avoid huge logs.
        self.counter_NM_logged = 0

        # Strip language tag but keep optional datatype
        # "foo"@en          -> "foo"
        # "foo"@en^^<dt>    -> "foo"^^<dt>
        self._strip_lang_tag = re.compile(
            r'("(?:(?:[^"\\]|\\.)*)")\s*@[^\s^]+(\^\^<[^>]*>)?'
        )

        # Permissive N-Quads / N-Triples pattern.
        # Supports:
        # - subject: IRI or blank node
        # - predicate: IRI
        # - object: literal (with optional lang + optional datatype), IRI, or blank node
        # - optional graph term: IRI or blank node
        # - ends with "."
        self.pattern = re.compile(
            r"""
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
            """,
            re.UNICODE | re.VERBOSE,
        )

        # Helpers for IRI canonicalization
        self._iri_trailing_cleanup = re.compile(r"([/#])>$")  # "<.../>" or "<...#>" before ">"
        self._is_iri = re.compile(r"^<[^>]*>$")

    def _canonicalize_iri(self, term: str) -> str:
        """
        Canonicalize an IRI token.

        Rules
        - Only applies to tokens that look like "<...>"
        - Removes a trailing "/" or "#" right before the closing ">"
          <http://a/b/>  -> <http://a/b>
          <http://a/b#>  -> <http://a/b>
        - Lowercases the inner IRI string so casing does not create new nodes
          <HTTP://EXAMPLE.ORG/X> -> <http://example.org/x>

        Returns
        Canonicalized IRI token as "<...>" or the original term if it is not an IRI.
        """
        if not self._is_iri.match(term):
            return term

        t = term.strip()

        # Remove trailing / or # immediately before ">"
        t = self._iri_trailing_cleanup.sub(">", t)

        # Lowercase the IRI content (between < and >)
        inner = t[1:-1].lower()
        return f"<{inner}>"

    def _canonicalize_term(self, term: str) -> str:
        """
        Canonicalize a parsed RDF term.

        - Literals are left unchanged (except lang tags are stripped earlier)
        - IRIs are canonicalized with `_canonicalize_iri`
        - Blank nodes are left unchanged (case-sensitive)

        Returns
        Canonicalized term string.
        """
        if not term or term.startswith('"'):
            return term
        if term.startswith("<"):
            return self._canonicalize_iri(term)
        return term

    def _open_text_stream(self) -> TextIO:
        """
        Open the file as a text stream, handling nested gzip automatically.

        Why this exists
        Some files look like `.gz`, pass `gzip -t`, but when you decompress them
        once, the output starts with gzip magic bytes again (`1f 8b`).
        That means: gzip inside gzip.

        Logic
        - Open outer gzip in binary mode
        - Peek at the next bytes of the decompressed stream
        - If they start with gzip magic, wrap another gzip decompressor

        Returns
        A TextIO stream yielding decoded UTF-8 lines (invalid bytes replaced).
        """
        f1 = gzip.open(self.file_path, "rb")

        # `peek` returns bytes that would be read next from the decompressed stream.
        head = f1.peek(2)[:2]

        # If after one gzip layer we still see gzip magic, it is nested.
        if head == b"\x1f\x8b":
            f2 = gzip.GzipFile(fileobj=f1, mode="rb")
            return io.TextIOWrapper(f2, encoding="utf-8", errors="replace")

        return io.TextIOWrapper(f1, encoding="utf-8", errors="replace")

    def parse(self) -> Iterator[Tuple[str, str, str, Optional[str]]]:
        """
        Stream-parse the file and yield RDF statements.

        Yields
        (s, p, o, g)
            s, p, o are strings; g is string or None.

        Behavior on bad lines
        - Increments `counter_NM`
        - Prints up to 200 examples: "No match found <line>"
        - Continues parsing the rest of the file
        """
        with self._open_text_stream() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # Strip language tags but keep datatype if present
                line = self._strip_lang_tag.sub(r"\1\2", line)

                m = self.pattern.match(line)
                if not m:
                    self.counter_NM += 1
                    if self.counter_NM_logged < 200:
                        print("No match found", line)
                        self.counter_NM_logged += 1
                    continue

                s = self._canonicalize_term(m.group("s"))
                p = self._canonicalize_term(m.group("p"))
                o = self._canonicalize_term(m.group("o"))
                g_raw = m.group("g")
                g = self._canonicalize_term(g_raw) if g_raw else None

                yield (s, p, o, g)

        if self.counter_NM > 0:
            print(f"[RDFParser] Total non-matching lines: {self.counter_NM}")
