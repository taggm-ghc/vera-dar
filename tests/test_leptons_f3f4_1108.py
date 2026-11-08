"""LEPTONS-F3F4-1108: refusal reason reaches the per-passage label; truthful pacing and not-reached wording."""
import json

from vera import evidence_trace as et
from vera import research_outcome as ro
from vera import search_providers as sp
from vera import rate_limit, discovery_panel
from vera.fulldoc_evidence import EvidenceBundle

SPAN = "Leptons are elementary particles."


def _ev():
    adm = ({"span_id": "S1", "document_id": "D1", "text": SPAN, "start": 0, "end": len(SPAN), "relevance": 0.9},)
    return EvidenceBundle(question="q", examined_full_documents=({"document_id": "D1"},), admitted=adm, rejected=(),
                          unusable=(), status="admitted")


def _rows(refusal):
    t = et.project(_ev(), None, None, None, None, True, sources=[{"n": 1, "snippet": SPAN}], refusal=refusal)
    return t["passages"][0]


def test_refusal_reason_label_from_config():
    lab = et.load_labels()
    for code, text in lab["by_refusal"].items():
        assert _rows(code)["reason"] == text
    assert _rows("model_insufficient")["reason"] != lab["not_cited"]["unused"]
    assert "no reason recorded" not in _rows("model_insufficient")["reason"]


def test_unknown_refusal_neutral_no_leak():
    r = _rows("internal_weird_code_x")["reason"]
    assert r == et.load_labels()["refusal_other"] and "internal" not in r and "_" not in r


def test_no_refusal_keeps_default():
    assert _rows(None)["reason"] == et.load_labels()["not_cited"]["unused"]


def test_fulldoc_fall_passes_token(monkeypatch):
    from vera import fulldoc_answer as fa
    seen = {}
    monkeypatch.setattr(et, "project", lambda *a, **k: seen.update(k) or {"ok": 1})
    assert fa._trace(_ev(), None, None, None, None, refusal="model_insufficient") == {"ok": 1}
    assert seen["refusal"] == "model_insufficient"


def test_own_pacing_vs_provider_limit():
    assert sp._own_pacing(rate_limit.RateLimited("europepmc", "pacing", 5))
    assert not sp._own_pacing(rate_limit.RateLimited("europepmc", "cooldown", 5))
    assert not sp._own_pacing(RuntimeError("HTTP 429 too many requests"))


def test_paced_cause_text_distinct():
    att = [ro.PortalAttempt("europepmc", "paced", 0)]
    assert ro._causes(att, False) == ["paced"]
    t = ro._CAUSE_TEXT["paced"]
    assert "paced" in t and "limit was reached" not in t
    assert ro._CAUSE_TEXT["quota"] != t


def test_not_reached_label_not_too_many():
    assert "too many" not in discovery_panel.REASON_TEXT["R-ACQ-STOPPED"]
    import inspect
    from vera import inline_acquire
    src = inspect.getsource(inline_acquire._evaluate_stream)
    assert '"doc_budget" if len(got) >= budget else "stopped"' in src
