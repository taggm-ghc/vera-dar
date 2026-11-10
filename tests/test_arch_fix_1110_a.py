"""ARCH-FIX-1110 A: one membership label per reading-list row (R-ACQ-KNOWN and the stored set agree); arXiv URL == DOI."""
import pytest  # noqa: F401

from tests.test_discovery_panel import render, texts
from vera import grounded_ask as ga
from vera.dar_corpus import doc_key_set

ARX = "10.48550/arxiv.2601.13671"


def test_arxiv_url_and_doi_are_one_document():
    want = {"doi:" + ARX}
    for url in ("https://arxiv.org/abs/2601.13671", "http://arxiv.org/abs/2601.13671v3", "https://arxiv.org/pdf/2601.13671v2",
                "https://arxiv.org/pdf/2601.13671.pdf", "https://www.arxiv.org/html/2601.13671v1/", "https://doi.org/10.48550/arXiv.2601.13671"):
        assert doc_key_set(None, url) == want, url
    for doi in ("10.48550/arXiv.2601.13671", "doi:10.48550/ARXIV.2601.13671v2"):
        assert doc_key_set(doi, None) == want, doi
    assert doc_key_set(None, "https://arxiv.org/abs/2601.13672") != want


def test_stored_acquired_doc_absent_from_snapshot_is_in_corpus():
    class Base:
        def doc_keys(self):
            return frozenset({"doi:10.1/other"})
    res = [{"url": "https://arxiv.org/abs/2601.13671", "title": "t"}, {"url": "https://x.org/n"}]
    out = ga._mark_in_corpus(res, Base(), extra_keys=doc_key_set(None, "https://doi.org/10.48550/arXiv.2601.13671"))
    assert out[0].get("in_corpus") is True and not out[1].get("in_corpus")
    assert ga._mark_in_corpus(res, Base(), extra_keys=frozenset()) is res


def test_known_row_never_prints_both_membership_labels():
    url = "https://arxiv.org/abs/2601.13671"
    data = {"result_kind": "reading_list", "answer": "x",
            "candidates": [{"title": "T", "url": url, "acquisition": "not_acquired"}],
            "contract": {"inline_acquisition": {"status": "no_evidence", "discovered": [
                {"url": url, "service": "s", "record_id": "1", "outcome": "already_stored", "reason": "R-ACQ-KNOWN"}]}}}
    t = " | ".join(texts(render(data)))
    assert "Already in the collection" not in t or "not in VERA-DAR's collection" not in t
    assert "not in VERA-DAR's collection" not in t
