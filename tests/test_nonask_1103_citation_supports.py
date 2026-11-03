"""NONASK-1103 WP6 (issue g): EH added `supports` (the answer sentence a citation backs) to the internal citation
records built by vera/fulldoc_answer.build_citations. The evidence trace reads it; the public /ask `citations` field
(main.EvidenceCitation) keeps its documented key set and does not expose it (docs/api.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTR = {"creator": "A. Author", "title": "T", "source": "https://doi.org/10.1/x", "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "none"}
INTERNAL = {"claim_id": 1, "n": 1, "span_id": "s1", "document_id": "d1", "canonical_source_id": "doi:10.1/x",
            "licence": "CC BY 4.0", "span": "chars=0-10", "verdict": "supported", "quote": "q", "quote_span": "chars=0-1",
            "attribution": ATTR, "supports": "Beta doubles throughput [1]."}


def test_internal_citation_records_carry_supports():
    src = (ROOT / "vera" / "fulldoc_answer.py").read_text(encoding="utf-8")
    assert '"supports": r["sentence"]' in src


def test_public_citation_model_does_not_expose_supports():
    import main
    out = main.EvidenceCitation(**INTERNAL).model_dump()
    assert "supports" not in out
    assert set(out) == set(INTERNAL) - {"supports"}


def test_docs_describe_supports_as_internal():
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    assert "`supports`" in api and "does not include it" in api
