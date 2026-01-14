import gzip
from pathlib import Path

import pytest

from RDFparser import RDFParser


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(text.encode("utf-8"))


def _write_nested_gz(path: Path, text: str) -> None:
    """
    Create a gzip file whose decompressed content is itself gzip bytes.
    This triggers your nested gzip logic.
    """
    inner = gzip.compress(text.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(inner)


def test_parse_ntriples_basic(tmp_path: Path):
    fp = tmp_path / "mini.nt.gz"
    _write_gz(
        fp,
        """
        <HTTP://EXAMPLE.ORG/A> <http://EXAMPLE.ORG/P> <http://example.org/B> .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    assert rows == [
        ("<http://example.org/a>", "<http://example.org/p>", "<http://example.org/b>", None)
    ]
    assert p.counter_NM == 0


def test_parse_nquads_with_graph(tmp_path: Path):
    fp = tmp_path / "mini.nq.gz"
    _write_gz(
        fp,
        """
        <http://example.org/s> <http://example.org/p> <http://example.org/o> <HTTP://EXAMPLE.ORG/G> .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    assert rows == [
        ("<http://example.org/s>", "<http://example.org/p>", "<http://example.org/o>", "<http://example.org/g>")
    ]


def test_blank_nodes_in_s_o_g(tmp_path: Path):
    fp = tmp_path / "bnode.nq.gz"
    _write_gz(
        fp,
        """
        _:s1 <http://ex/p> _:o1 _:g1 .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    assert rows == [("_:s1", "<http://ex/p>", "_:o1", "_:g1")]


def test_literal_lang_tag_is_stripped(tmp_path: Path):
    fp = tmp_path / "lang.nq.gz"
    _write_gz(
        fp,
        """
        <http://ex/s> <http://ex/p> "Hello"@en <http://ex/g> .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    # language tag removed, graph canonicalized
    assert rows == [("<http://ex/s>", "<http://ex/p>", '"Hello"', "<http://ex/g>")]


def test_literal_lang_tag_with_datatype_keeps_datatype(tmp_path: Path):
    fp = tmp_path / "lang_dt.nq.gz"
    _write_gz(
        fp,
        """
        <http://ex/s> <http://ex/p> "Hello"@en^^<http://ex/dt> <http://ex/g> .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    # "Hello"@en^^<dt> -> "Hello"^^<dt>
    assert rows == [
        ("<http://ex/s>", "<http://ex/p>", '"Hello"^^<http://ex/dt>', "<http://ex/g>")
    ]


def test_literal_escapes_are_kept(tmp_path: Path):
    fp = tmp_path / "esc.nq.gz"
    _write_gz(
        fp,
        '<http://ex/s> <http://ex/p> "He said \\"hi\\"." <http://ex/g> .\n'
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    assert rows == [
        ("<http://ex/s>", "<http://ex/p>", '"He said \\"hi\\"."', "<http://ex/g>")
    ]

def test_iri_trailing_slash_and_hash_are_removed(tmp_path: Path):
    fp = tmp_path / "trail.nq.gz"
    _write_gz(
        fp,
        """
        <http://ex/A/> <http://ex/p#> <http://ex/B#> <http://ex/G/> .
        """.strip() + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    # trailing "/" or "#" before ">" removed, and lowercased
    assert rows == [
        ("<http://ex/a>", "<http://ex/p>", "<http://ex/b>", "<http://ex/g>")
    ]


def test_nonmatching_lines_are_counted_and_skipped(tmp_path: Path):
    fp = tmp_path / "badmix.nq.gz"
    _write_gz(
        fp,
        "\n".join([
            "<http://ex/s> <http://ex/p> <http://ex/o> .",
            "THIS IS NOT RDF",
            "<http://ex/s2> <http://ex/p2> \"ok\"@en .",
            "also bad",
        ]) + "\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    # Two good statements should parse
    assert rows == [
        ("<http://ex/s>", "<http://ex/p>", "<http://ex/o>", None),
        ("<http://ex/s2>", "<http://ex/p2>", '"ok"', None),
    ]

    # Two bad lines
    assert p.counter_NM == 2


def test_nested_gzip_is_supported(tmp_path: Path):
    fp = tmp_path / "nested.nq.gz"
    _write_nested_gz(
        fp,
        "<http://ex/s> <http://ex/p> <http://ex/o> <http://ex/g> .\n"
    )

    p = RDFParser(str(fp))
    rows = list(p.parse())

    assert rows == [
        ("<http://ex/s>", "<http://ex/p>", "<http://ex/o>", "<http://ex/g>")
    ]
    assert p.counter_NM == 0
