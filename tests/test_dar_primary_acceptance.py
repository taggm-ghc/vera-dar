"""AE-1020/WP-C: PRIMARY functional acceptance for the DAR /ask path (P1, P1b, P2, P3 of FDC-1020 acceptance-plan).

What "primary success" means here: an evidence-grounded answer produced from inspected FULL documents, with no fallback.
These tests drive the real main:app pipeline with the L3 seam UNPATCHED (answer_contract.examined_full_documents /
result_kind_for are the original functions; a guard fails otherwise). Offline: synthetic bundle, stub LLM and checker, no
network, no DB.

CAVEAT: tests.fulldoc_bundle_synth.synth_cfg() force-enables fulldoc.enabled, so P1 proves the MECHANISM works when the
flag is on, not the shipped configuration. The real-bundle (DAR-9) run is owner-run.
`outcome` (C-2 enum answer / insufficient_evidence / corpus_unusable / reading_list / no_sources / decline) comes from
WP-B; assertions on it fail until WP-B lands (deliberately not xfail/skip).
Contract-rendering tests elsewhere that open the seam are not functional acceptance; this file is.
"""
import pytest

from tests import fulldoc_bundle_synth as synth
from tests.test_fulldoc_e2e import (ABS, GROUNDING, Q_EPI, Q_NONE, SPAN, WEB, WEB_URL, Harness, _ids,  # noqa: F401
                                    _texts)
from vera import answer_contract as ac

ORIG_EFD, ORIG_RKF = ac.examined_full_documents, ac.result_kind_for  # captured at import, before any monkeypatch
NON_DOC_KINDS = ("abstract_metadata", "abstract", "snippet", "metadata", "index_record")


@pytest.fixture
def h(monkeypatch, tmp_path):
    from vera import fulldoc_corpus
    fulldoc_corpus.reset_for_tests()
    monkeypatch.delenv("VERA_FULLDOC_BUNDLE_DIR", raising=False)
    yield Harness(monkeypatch, tmp_path)
    fulldoc_corpus.reset_for_tests()


def _seam_unpatched():
    assert ac.examined_full_documents is ORIG_EFD and ac.result_kind_for is ORIG_RKF, "L3 seam is patched"


def is_primary_success(body, texts=None) -> bool:
    """The P1 predicate: a real answer, no fallback, inspected full documents, every citation bound to an admitted span
    with a verbatim quote at the span's offsets (checked against `texts` when given), claim check ran."""
    try:
        if body.get("outcome") != "answer" or body.get("result_kind") != "answer":
            return False
        if (body.get("contract") or {}).get("fallback") is not None:
            return False
        if not (body.get("evidence_status") and body.get("examined_full_documents")):
            return False
        adm = {a["span_id"]: a for a in body.get("admitted_evidence") or []}
        cits = body.get("citations") or []
        if not adm or not cits:
            return False
        for c in cits:
            if c.get("span_id") not in adm or not c.get("quote"):
                return False
            if texts is not None:
                m = SPAN.match(c.get("span") or "")
                doc = texts.get(adm[c["span_id"]]["document_id"])
                if not m or doc is None or c["quote"] not in doc[int(m.group(1)):int(m.group(2))]:
                    return False
        cc = body.get("claim_check") or {}
        return cc.get("outcome") == "checked" and cc.get("removed", 0) < cc.get("checked", 0)
    except (KeyError, TypeError, AttributeError):
        return False


def _invariant(body):
    """P1b: answer implies evidence present and every citation bound to an admitted span."""
    if body.get("result_kind") != "answer":
        return
    adm = {a["span_id"] for a in body.get("admitted_evidence") or []}
    assert body.get("evidence_status") and body.get("examined_full_documents"), "answer without inspected full documents"
    assert all(c["span_id"] in adm for c in body.get("citations") or []), "citation not bound to an admitted span"


# ---------------------------------------------------------------- P1
def test_p1_primary_answer_is_full_document_grounded(h):
    root = h.bundle()
    texts = _texts(root)
    _seam_unpatched()
    raw, b = h.ask(Q_EPI)
    _seam_unpatched()
    assert b["result_kind"] == "answer" and b["contract"]["fallback"] is None
    assert b.get("outcome") == "answer", "WP-B: outcome field"
    assert b["evidence_status"] and b["examined_full_documents"]
    for d in b["examined_full_documents"]:
        assert d["document_id"] in texts and d["text_sha256"] and d["chars"] > 0
    adm = {a["span_id"]: a for a in b["admitted_evidence"]}
    assert adm and b["citations"]
    for c in b["citations"]:
        assert c["span_id"] in adm and c["quote"]
        m = SPAN.match(c["span"])
        assert m, c["span"]
        s, e = int(m.group(1)), int(m.group(2))
        assert c["quote"] in texts[adm[c["span_id"]]["document_id"]][s:e], "quote not verbatim at the span offsets"
    assert b["sources"] and all(s["span_id"] in adm for s in b["sources"])
    assert b["claim_check"]["outcome"] == "checked" and b["claim_check"]["removed"] < b["claim_check"]["checked"]
    assert ABS not in raw and WEB_URL not in raw
    assert is_primary_success(b, texts)


def test_p1_fails_when_evidence_check_is_opened(h, monkeypatch):
    """Meta: with the evidence check forced open and NO bundle, the pipeline may say 'answer', but the predicate must
    still reject it (no admitted evidence, no citations)."""
    monkeypatch.setattr(ac, "examined_full_documents", lambda c: [{"document_id": "fake"}])
    monkeypatch.setattr(ac, "result_kind_for", lambda c: "answer")
    try:
        _, b = h.ask(Q_EPI)
    except Exception:  # noqa: BLE001  (a crash is also not primary success)
        return
    assert is_primary_success(b) is False


# ---------------------------------------------------------------- P1b
def _stored(kind, admitted):
    return {"stored_text_kind": kind, "admission_passed": admitted, "fetch_status": "ok", "acquisition": ac.ACQ_ACQUIRED}


@pytest.mark.parametrize("variant", ["web_only", "abstract_only", "stored_full_text_dict", "stored_abstract_dict"])
def test_p1b_answer_implies_evidence_without_bundle(h, variant, monkeypatch):
    _seam_unpatched()
    entry = dict(WEB[0])
    if variant == "abstract_only":
        entry.update(stored_text_kind="abstract_metadata", admission_passed=True)
    elif variant == "stored_full_text_dict":
        entry.update(_stored("full_text", True))
    elif variant == "stored_abstract_dict":
        entry.update(_stored("abstract", True))
    monkeypatch.setattr("tests.test_fulldoc_e2e.WEB", [entry])
    h.install = _wrap_install(h, entry)
    raw, b = h.ask(Q_EPI)
    assert b["result_kind"] != "answer", "an answer without a full-document EvidenceBundle (defect for WP-B)"
    _invariant(b)
    assert is_primary_success(b) is False


def _wrap_install(h, entry):
    orig = h.install

    def install(*a, **k):
        main = orig(*a, **k)
        h.mp.setattr(main, "retrieve_all", lambda q, cfg: ([dict(entry)], []))
        return main
    return install


def test_p1b_invariant_holds_on_grounded_answer(h):
    h.bundle()
    _, b = h.ask(Q_EPI)
    assert b["result_kind"] == "answer"
    _invariant(b)


# ---------------------------------------------------------------- P2
def _s_reading_list(h):
    return h.ask(Q_EPI)[1], "reading_list"


def _s_insufficient(h):
    h.bundle()
    return h.ask(Q_NONE)[1], "insufficient_evidence"


def _s_unusable(h):
    h.bundle(tamper={d: "\nINJECTED" for d in ("syn_epi", "syn_clim", "syn_repro")})
    return h.ask(Q_EPI)[1], "corpus_unusable"


def _s_checker_unavailable(h):
    h.bundle()
    h.check = lambda t, s, c: (None, {"checked": 0, "outcome": "checker_unavailable"})
    return h.ask(Q_EPI)[1], "insufficient_evidence"


def _s_checker_disabled(h):
    h.bundle()
    return h.ask(Q_EPI, grounding={**GROUNDING, "claim_check": {"enabled": False}})[1], "insufficient_evidence"


FALLBACK_SCENARIOS = {"reading_list": _s_reading_list, "insufficient": _s_insufficient, "unusable": _s_unusable,
                      "checker_unavailable": _s_checker_unavailable, "checker_disabled": _s_checker_disabled}
EXPECT_TOKEN = {"reading_list": None, "insufficient": "insufficient_full_document_evidence",
                "unusable": "full_document_corpus_unusable", "checker_unavailable": "checker_unavailable",
                "checker_disabled": "claim_check_disabled"}


@pytest.mark.parametrize("name", sorted(FALLBACK_SCENARIOS))
def test_p2_fallback_is_not_primary_success(h, name):
    _seam_unpatched()
    b, _ = FALLBACK_SCENARIOS[name](h)
    assert b["result_kind"] == "reading_list" and b["contract"]["fallback"] == EXPECT_TOKEN[name]
    assert not b.get("citations") and b["sources"] == [] and b["claim_check"] is None
    assert is_primary_success(b) is False
    if name == "insufficient":
        assert h.answer_calls == []


@pytest.mark.parametrize("name", sorted(FALLBACK_SCENARIOS))
def test_p2_fallback_outcome_enum(h, name):
    b, want = FALLBACK_SCENARIOS[name](h)
    assert b.get("outcome") == want, "WP-B: outcome field / C-2 mapping"


# ---------------------------------------------------------------- P3
@pytest.mark.parametrize("kind", NON_DOC_KINDS)
def test_p3a_abstract_only_bundle_is_refused(h, kind):
    """Abstract-only content, labelled with a non-document text_kind: nothing examined, never an answer."""
    kw = {d: {"with_body": False} for d in ("syn_epi", "syn_clim", "syn_repro")}
    h.bundle(jats_over=kw, text_kind=kind)
    raw, b = h.ask(Q_EPI)
    assert b["result_kind"] != "answer" and not b.get("examined_full_documents")
    assert b["candidates"] and all("full document not acquired" in str(c) or c.get("acquisition") != ac.ACQ_ACQUIRED
                                   for c in b["candidates"])
    assert is_primary_success(b) is False


@pytest.mark.parametrize("kind", NON_DOC_KINDS)
def test_p3a_label_alone_must_refuse_full_body(h, kind):
    """A manifest/meta text_kind in the non-document kinds on a document WITH a body: the loader is expected to refuse
    it. If it is acquired, that is a defect (route to WP-A / fulltext_loader), not a test to relax."""
    h.bundle(text_kind=kind)
    _, b = h.ask(Q_EPI)
    assert not b.get("examined_full_documents"), f"text_kind={kind!r} document was treated as a full document"
    assert b["result_kind"] != "answer"


@pytest.mark.parametrize("kind", NON_DOC_KINDS)
def test_p3b_stored_non_document_kind_not_acquired(kind):
    assert ac.acquisition_state({"stored_text_kind": kind, "admission_passed": True}) == ac.ACQ_NOT_ACQUIRED
    assert ac.representation(ac.ACQ_NOT_ACQUIRED) != ac.representation(ac.ACQ_ACQUIRED)
    assert ac.examined_full_documents([{"stored_text_kind": kind, "admission_passed": True}]) == []
    assert ac.result_kind_for([{"stored_text_kind": kind, "admission_passed": True}]) == "reading_list"


def test_p3b_not_admitted_or_missing_kind_not_acquired():
    assert ac.acquisition_state({"stored_text_kind": "full_text", "admission_passed": False}) == ac.ACQ_NOT_ACQUIRED
    assert ac.acquisition_state({"stored_text_kind": "full_text"}) == ac.ACQ_NOT_ACQUIRED
    assert ac.acquisition_state({"admission_passed": True}) == ac.ACQ_NOT_ACQUIRED
    assert ac.acquisition_state({}) == ac.ACQ_NOT_ACQUIRED


def test_p3c_mixed_bundle_abstract_doc_never_examined_or_cited(h):
    root = h.bundle(jats_over={"syn_clim": {"with_body": False}}, text_kind={"syn_clim": "abstract"})
    texts = _texts(root)
    raw, b = h.ask(Q_EPI)
    assert b["result_kind"] == "answer"
    assert "syn_clim" not in _ids(b["examined_full_documents"]) and "syn_clim" not in _ids(b["admitted_evidence"])
    assert all("syn_clim" not in (c.get("span_id", "") + c.get("canonical_source_id", "")) for c in b["citations"])
    assert "syn_epi" in _ids(b["examined_full_documents"])
    assert is_primary_success(b, texts)


def test_p3d_admit_exception_fails_closed(h, monkeypatch):
    h.bundle()
    from vera import fulldoc_admission

    def boom(*a, **k):
        raise RuntimeError("admit failed")

    monkeypatch.setattr(fulldoc_admission, "admit", boom)
    raw, b = h.ask(Q_EPI)
    assert b["result_kind"] != "answer" and b["contract"]["fallback"]
    assert h.answer_calls == [] and not b.get("citations")
    assert is_primary_success(b) is False


def test_p3d_missing_answer_config_fails_closed(h, monkeypatch):
    h.bundle()
    from vera import fulldoc_corpus
    cfg = {k: v for k, v in synth.synth_cfg().items() if k != "answer"}
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    fulldoc_corpus.reset_for_tests()
    try:
        raw, b = h.ask(Q_EPI)
    except Exception:  # noqa: BLE001  (an exception is not an answer either)
        return
    assert b["result_kind"] != "answer" and h.answer_calls == []
    assert is_primary_success(b) is False


def test_p3e_fetch_status_alone_never_acquired(h):
    for c in ({"fetch_status": "ok"}, {"fetch_status": "ok", "stored_text_kind": None},
              {"fetch_status": "ok", "admission_passed": True}, {"fetch_status": "ok", "stored_text_kind": "abstract"}):
        assert ac.acquisition_state(c) == ac.ACQ_NOT_ACQUIRED
    assert ac.result_kind_for([{"fetch_status": "ok", "acquisition": ac.ACQ_ACQUIRED}]) == "reading_list"
