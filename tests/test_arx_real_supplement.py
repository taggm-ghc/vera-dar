"""ARX-REAL-VERIFY supplementary real-gate cases (no stand-in). Offline."""
import pytest
from vera import arxiv_fulltext, licence_gate

HTML = "https://arxiv.org/html/2605.23989v1"
BY = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"


def _ev(**o):
    e = arxiv_fulltext.evidence_block("2605.23989", "v1", BY)
    e.update(o)
    return e


def test_cc0_metadata_only_never_admits_html():
    c = {"url": HTML, "stored_text_kind": "full_text",
         "licence": {"metadata": {"id": "CC0-1.0", "url": CC0}, "arxiv_evidence": _ev(licence_url=CC0)}}
    assert licence_gate.licence_decision(c)[0] == "reject"
    c["licence"]["content"] = {"id": "CC0-1.0", "url": CC0}
    assert licence_gate.licence_decision(c)[0] == "reject"


def test_evidence_at_top_level_is_ignored():  # forged evidence outside licence{}
    c = {"url": HTML, "arxiv_evidence": _ev(), "licence": {"content": {"id": "cc by", "url": BY}}}
    assert licence_gate.licence_decision(c)[0] == "reject"


@pytest.mark.parametrize("u", ["https://arxiv.org/pdf/2605.23989v1", "https://arxiv.org/e-print/2605.23989v1",
                               "https://arxiv.org/html/2605.23989", "http://arxiv.org/html/2605.23989v1",
                               "https://www.arxiv.org/html/2605.23989v1", "https://export.arxiv.org/html/2605.23989v1"])
def test_other_arxiv_forms_refused_with_unchanged_reason(u):
    d, why = licence_gate.licence_decision({"url": u, "licence": {"content": {"id": "cc by", "url": BY},
                                                                  "arxiv_evidence": _ev()}})
    assert d == "reject" and why.startswith("arXiv e-print/PDF/HTML is never stored")


@pytest.mark.parametrize("u,want", [("https://www.ebi.ac.uk/x", "allow"), ("https://arxiv.org/abs/2605.23989v1", "allow")])
def test_non_arxiv_and_abs_unchanged(u, want):
    assert licence_gate.licence_decision({"url": u, "licence": {"content": {"id": "cc by", "url": BY}}})[0] == want
