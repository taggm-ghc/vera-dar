from vera import ui_evidence as u


import pytest


@pytest.fixture(autouse=True)
def _legacy_flag_off(monkeypatch):
    """NOANSWER-RC-1107: these pin the legacy scope wording, which now ships only with count_matching_documents off."""
    cfg = dict(u._cfg(), count_matching_documents=False)
    monkeypatch.setattr(u, "_cfg", lambda: cfg)


def _d(kind="bundle", n=9, below=1):
    return {"evidence_status": "insufficient", "examined_full_documents": [{"document_id": i} for i in range(n)],
            "admitted_evidence": [], "corpus_source": {"kind": kind},
            "contract": {"inline_acquisition": {"documents": [{"persisted": False, "reason": "below_persist_min"}] * below}}}


def test_split_counts():
    assert u.evidence_scope_line(_d()) == ("Evidence scope: 9 documents from VERA-DAR's collection and 1 fetched for this question "
                                           "inspected; no passages selected as possible evidence.")


def test_no_inline_unchanged():
    assert u.evidence_scope_line(_d(below=0)).startswith("Evidence scope: 9 documents inspected;")


def test_no_double_count_when_not_base():
    assert u.evidence_scope_line(_d(kind="db")).startswith("Evidence scope: 9 documents inspected;")


def test_no_backend_names():
    assert "pinned" not in u.evidence_scope_line(_d()).lower()


def test_headline_split_and_unchanged():
    from types import SimpleNamespace as N

    from vera import ask_service as a
    ev = N(corpus_source="bundle", examined_full_documents=tuple(range(9)))
    ia = {"documents": [{"persisted": False, "reason": "below_persist_min"}]}
    t = "VERA-DAR searched 9 full documents but found no passage."
    assert a._with_fetched(t, ev, ia) == ("VERA-DAR searched 9 full documents from VERA-DAR's collection and 1 fetched "
                                          "for this question but found no passage.")
    assert a._with_fetched(t, ev, {"documents": []}) == t
    assert a._with_fetched(t, N(corpus_source="db", examined_full_documents=tuple(range(9))), ia) == t
    assert a._plural_docs(1) == "1 full document"
