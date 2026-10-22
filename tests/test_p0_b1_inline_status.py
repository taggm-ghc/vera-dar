"""P0-B1 (demo-1022): an in-scope question the stored corpus cannot answer must carry the inline acquisition outcome in
its contract, so the Report states whether a new search was made (off / none / ...) instead of showing an ordinary
abstention that hides a gated or failed acquisition. Offline: synthetic fixtures only (no network, DB or LLM)."""
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from tests.test_dar_inline_acquire import rig  # noqa: F401  (fixture)
from vera import ask_service
from vera.m7 import live_report

IN = dict(accept=True, layer="layer2_llm", reason="llm_in:x", category="in_scope")


def _report_status(r):
    data = {"result_kind": r.result_kind, "contract": r.contract, "candidates": list(r.candidates),
            "tokens_used": r.tokens_used, "cost_usd": r.cost_usd}
    return ((live_report.build_live_report(data, "q", 1.0).get("search") or {}).get("inline") or {}).get("status")


def test_acquisition_off_is_reported_not_silent(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: None)  # flag, kill switch or DB env: effectively off
    r = rig.run(**IN)
    assert r.result_kind != "answer" and rig.search_calls == []
    assert r.contract["inline_acquisition"] == {"status": "off"}
    assert _report_status(r) == live_report.INLINE_STATUS["off"]


def test_inline_search_with_nothing_admitted_is_reported_as_none(rig):
    rig.setup(topic="clim")  # licensed document on an unrelated topic: the unchanged relevance gate rejects it
    r = rig.run(**IN)
    assert r.result_kind != "answer" and rig.search_calls
    assert r.contract["inline_acquisition"]["status"] == "none"
    assert rig.cfg["scope_expansion"]["inline_notice_none"] in r.answer
    assert _report_status(r) == live_report.INLINE_STATUS["none"]


def test_layer1_dirty_question_gets_no_inline_block(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: False)
    r = rig.run(**IN)
    assert "inline_acquisition" not in (r.contract or {}) and rig.search_calls == []
