import gzip
import hashlib
import io
import re
from typing import Iterator, Optional, TextIO, Tuple


class RDFParser:
    """
    Lightweight RDF N-Quads / N-Triples parser for large files.

    What it does
    1) Reads plain, gzip, or nested-gzip `.nq` / `.nt` files as a stream
       (no full file in memory)
    2) Normalizes syntax so you get fewer "different looking but same" nodes:
       - Strips language tags from literals but keeps datatype if present
         Example:  "foo"@en^^<dt>  ->  "foo"^^<dt>
       - Canonicalizes IRIs (see `_canonicalize_iri`)
    3) Deduplicates triples, because a data graph is a *set* of triples

    Output format
    Yields 4-tuples: (s, p, o, g)
    - s: subject term as string ("<...>" or "_:...")
    - p: predicate IRI as string "<...>"
    - o: object term as string (literal, IRI, or blank node)
    - g: optional graph name (IRI or blank node), or None if absent

    Deduplication
    -------------
    By Definition 1 of the thesis a data graph is a finite SET of triples, so
    being told about a statement twice does not make it two statements. Two
    kinds of repetition occur in crawled data:

      - byte-identical lines, the same statement from the same document
      - the same (s, p, o) under different named graphs, i.e. the same
        statement found in several documents

    Both are dropped. On the DyLDO 2015-10-11 snapshot this is a large effect:
    48,832,721 quads contain only 31,111,684 distinct lines, and a further
    7,771,321 triples repeat across named graphs. Since pi(s,p,o) = 1/d_out(s)
    divides by the number of outgoing triples, leaving duplicates in place
    would inflate the outdegree of every affected subject and depress every
    probability on it.

    Deduplication happens HERE, on the original (s, p, o), and therefore
    before the builder collapses literal objects onto `__LITERAL__`. That
    order matters: after collapsing, (s, foaf:name, "Alice") and
    (s, foaf:name, "Bob") are indistinguishable, and merging them would
    discard a real statement.

    The first named graph in which a triple is seen is the one reported; the
    graph component is not part of the deduplication key.

    Notes
    - This is a permissive regex-based parser, not a full RDF spec
      implementation. It is designed for DyLDO-style N-Quads where most lines
      are simple.
    """

    def __init__(
        self,
        file_path: str,
        dedupe: bool = True,
        lowercase_full_iri: bool = False,
    ):
        """
        Parameters
        file_path
            Path to an N-Quads/N-Triples file, plain or gzip-compressed.
        dedupe
            Drop repeated triples (default). Keeps a set of 128-bit digests,
            roughly 2 GB for 30 million distinct triples. Set False to stream
            without deduplication if memory is tight, but note that the
            resulting probabilities will not match Definition 1.
        lowercase_full_iri
            Reproduce the earlier normalisation, which lowercased the whole
            IRI. Off by default; see `_canonicalize_iri` for why.
        """
        self.file_path = file_path
        self.dedupe = dedupe
        self.lowercase_full_iri = lowercase_full_iri

        # Count of lines that did not match the N-Quads/N-Triples regex.
        self.counter_NM = 0

        # We only print a small sample of bad lines to avoid huge logs.
        self.counter_NM_logged = 0

        # Deduplication statistics, filled in during parse().
        self.counter_lines = 0
        self.counter_yielded = 0
        self.counter_duplicates = 0

        # Strip language tag but keep optional datatype
        # "foo"@en          -> "foo"
        # "foo"@en^^<dt>    -> "foo"^^<dt>
        self._strip_lang_tag = re.compile(
            r'("(?:(?:[^"\\]|\\.)*)")\s*@[^\s^]+(\^\^<[^>]*>)?'
        )

        # Permissive N-Quads / N-Triples pattern.
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
        self._iri_trailing_cleanup = re.compile(r"[/#]+>$")
        self._is_iri = re.compile(r"^<[^>]*>$")
        # scheme "://" authority, then the rest
        self._iri_authority = re.compile(
            r"^([A-Za-z][A-Za-z0-9+.\-]*)://([^/?#]*)(.*)$"
        )
        # scheme ":" rest, for schemes without an authority such as acct: or urn:
        self._iri_scheme_only = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):(.*)$")

    # ----------------------------------------------------------------- #
    # normalisation
    # ----------------------------------------------------------------- #

    def _canonicalize_iri(self, term: str) -> str:
        """
        Canonicalize an IRI token.

        Rules
        - Only applies to tokens that look like "<...>"
        - Removes trailing "/" or "#" characters before the closing ">"
          <http://a/b/>  -> <http://a/b>
          <http://a/b#>  -> <http://a/b>
        - Lowercases the scheme and the authority (host), which RFC 3986
          §6.2.2.1 declares case-insensitive, and leaves the path unchanged
          <HTTP://EXAMPLE.ORG/Country> -> <http://example.org/Country>

        Why the path keeps its case
        The earlier implementation, inherited from the FluidSpark parser,
        lowercased the whole IRI. That merges resources the path distinguishes.
        DBpedia separates classes from properties by capitalisation of the
        local name, so dbo:Country (a class) and dbo:country (a property)
        collapsed into one token: 69 such pairs on DBpedia 3.9, and 6,967
        path-case collisions on the DyLDO 2015-10-11 snapshot, among them
        <...atom-owl/2004-06-22/ns#Content> with <...ns#content>. Resolving
        the two DBpedia IRIs in a browser returns two different pages, so
        these are genuinely distinct resources.

        Set lowercase_full_iri=True to reproduce the earlier behaviour.
        """
        if not self._is_iri.match(term):
            return term

        t = self._iri_trailing_cleanup.sub(">", term.strip())
        inner = t[1:-1]

        if self.lowercase_full_iri:
            return f"<{inner.lower()}>"

        m = self._iri_authority.match(inner)
        if m:
            scheme, authority, rest = m.groups()
            return f"<{scheme.lower()}://{authority.lower()}{rest}>"

        m = self._iri_scheme_only.match(inner)
        if m:
            scheme, rest = m.groups()
            return f"<{scheme.lower()}:{rest}>"

        return f"<{inner}>"

    def _canonicalize_term(self, term: str) -> str:
        """
        Canonicalize a parsed RDF term.

        - Literals are left unchanged (except lang tags are stripped earlier)
        - IRIs are canonicalized with `_canonicalize_iri`
        - Blank nodes are left unchanged. DyLDO encodes the source document
          into each blank node label, so labels are already scoped per
          document and no renaming apart is required.
        """
        if not term or term.startswith('"'):
            return term
        if term.startswith("<"):
            return self._canonicalize_iri(term)
        return term

    # ----------------------------------------------------------------- #
    # input
    # ----------------------------------------------------------------- #

    def _open_text_stream(self) -> TextIO:
        """
        Open the file as a text stream, handling plain text, gzip, and nested
        gzip automatically.

        Why this exists
        The input may be uncompressed (`.nt`, `.nq`) or gzip-compressed. Some
        files look like `.gz`, pass `gzip -t`, but when you decompress them
        once, the output starts with gzip magic bytes again (`1f 8b`).
        That means: gzip inside gzip.

        Logic
        - Sniff the first two bytes rather than trusting the file extension
        - If they are not gzip magic, open as plain text
        - Otherwise open the outer gzip and peek at the decompressed stream
        - If that also starts with gzip magic, wrap another decompressor
        """
        with open(self.file_path, "rb") as probe:
            magic = probe.read(2)

        if magic != b"\x1f\x8b":
            return io.open(self.file_path, "rt", encoding="utf-8",
                           errors="replace")

        f1 = gzip.open(self.file_path, "rb")

        # `peek` returns bytes that would be read next from the decompressed
        # stream.
        head = f1.peek(2)[:2]

        # If after one gzip layer we still see gzip magic, it is nested.
        if head == b"\x1f\x8b":
            f2 = gzip.GzipFile(fileobj=f1, mode="rb")
            return io.TextIOWrapper(f2, encoding="utf-8", errors="replace")

        return io.TextIOWrapper(f1, encoding="utf-8", errors="replace")

    # ----------------------------------------------------------------- #
    # deduplication
    # ----------------------------------------------------------------- #

    @staticmethod
    def _triple_digest(s: str, p: str, o: str) -> int:
        """128-bit digest of a canonicalized triple.

        Storing digests rather than the strings keeps the seen-set to roughly
        2 GB at 30 million triples instead of about 5 GB. At 128 bits the
        chance of a collision over 10^8 triples is on the order of 10^-23.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
        h.update(o.encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    # ----------------------------------------------------------------- #
    # main loop
    # ----------------------------------------------------------------- #

    def parse(self) -> Iterator[Tuple[str, str, str, Optional[str]]]:
        """
        Stream-parse the file and yield distinct RDF statements.

        Yields
        (s, p, o, g)
            s, p, o are strings; g is string or None. When a triple occurs
            under several named graphs, the first one seen is reported.

        Behaviour on bad lines
        - Increments `counter_NM`
        - Prints up to 200 examples: "No match found <line>"
        - Continues parsing the rest of the file
        """
        seen = set() if self.dedupe else None
        self.counter_lines = 0
        self.counter_yielded = 0
        self.counter_duplicates = 0

        with self._open_text_stream() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                self.counter_lines += 1

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

                if seen is not None:
                    digest = self._triple_digest(s, p, o)
                    if digest in seen:
                        self.counter_duplicates += 1
                        continue
                    seen.add(digest)

                self.counter_yielded += 1
                yield (s, p, o, g)

        if self.counter_NM > 0:
            print(f"[RDFParser] Total non-matching lines: {self.counter_NM}")
        if self.dedupe:
            share = self.counter_duplicates / max(1, self.counter_lines)
            print(f"[RDFParser] {self.counter_lines} lines, "
                  f"{self.counter_yielded} distinct triples, "
                  f"{self.counter_duplicates} duplicates dropped ({share:.1%})")