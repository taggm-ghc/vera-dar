"""PLAN-FDA INT: the fulldoc.enabled flag (F13), the shipped config state, and the no-write guarantee.
In-process service calls with stubs only: no TestClient, no app, no DB, no network."""
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from tests import fulldoc_bundle_synth as synth
from vera import ask_service, claim_check, fulldoc_corpus, scope_router
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

CONFIG = Path(fulldoc_corpus.CONFIG_PATH)
V0 = CONFIG.with_name("fulldoc_answer.v0.json")
P0_SHA = "669526db208c4bd5f92cd335832d381f6cb857b7e3c46c7325bb95ff29b5cf37"  # git blob fda-p0:config/fulldoc_answer.json
PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
WEB = [{"n": 1, "title": "Web Paper One", "url": "https://example.org/web/one", "published": "2025-01-01",
        "snippet": "web abstract", "abstract": "web abstract", "identifier": "10.1234/web", "provider": "arxiv",
        "licence_decision": "allow", "retrieved_at": "2026-10-01"}]
GROUNDING = {"insufficient_text": "INSUFFICIENT", "removed_citation_note": "NOTE", "no_sources_text": "NO SOURCES",
             "claim_check": {"enabled": True, "unavailable_text": "UNAVAILABLE"}}
Q = "How was the transmission of the pathogen related to incidence during an outbreak?"


def test_shipped_config_state():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["fulldoc"]["enabled"] is False  # pending acceptance; INT-final sets true
    adm = cfg["admission"]
    assert adm["excluded_section_headings"] == ["How to cite", "Code availability", "Supporting information"]
    assert adm["excluded_paragraph_labels"] == ["How to cite this article:", "How to cite:"]
    assert adm.get("signal", "legacy") == "legacy" and adm["pointer_filter"]["enabled"] is False
    assert "PENDING CAL" in cfg["_calibration_note"] and cfg["_calibration_note"].startswith("admission thresholds: probed")
    assert cfg["answer"]["per_part"]["enabled"] is True and cfg["fallback_tokens"]["model_insufficient"] == "model_insufficient"
    assert cfg["retrieval"]["candidate_k"] == 12 and adm["max_spans_per_doc"] == 2  # priors stay inactive


def test_v0_is_the_p0_config_byte_copy():
    assert hashlib.sha256(V0.read_bytes()).hexdigest() == P0_SHA
    assert "fulldoc" not in json.loads(V0.read_text(encoding="utf-8"))


def _no_store(mp):
    from vera import corpus_admission, dar_store

    def boom(*a, **k):
        raise AssertionError("store/writer touched (DB write)")

    mp.setattr(corpus_admission.WoCorpusStore, "__init__", boom)
    mp.setattr(dar_store.DarWriter, "__init__", boom)
    mp.setattr(corpus_admission, "store_from_env", boom)


def _ask(mp, calls):
    mp.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    mp.setattr(ask_service, "complete", lambda *a, **k: calls.append("llm") or AskResult("x [1]", 1, 0.0))
    mp.setattr(claim_check, "check_answer", lambda *a, **k: calls.append("check") or (None, {"outcome": "x"}))
    return ask_service.answer_in_scope(Q, "m", PRICING, grounding=(lambda: [dict(w) for w in WEB], GROUNDING))


@pytest.fixture(autouse=True)
def _reset():
    fulldoc_corpus.reset_for_tests()
    yield
    fulldoc_corpus.reset_for_tests()


def _fields(r):
    return {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}


def test_disabled_equals_pre_fda_path_and_makes_no_writes(monkeypatch, tmp_path):
    _no_store(monkeypatch)
    root = synth.write_bundle(tmp_path / "b")
    # pre-FDA: the bundle variable is unset
    monkeypatch.delenv("VERA_FULLDOC_BUNDLE_DIR", raising=False)
    base_calls: list = []
    base = _ask(monkeypatch, base_calls)
    # flag off, bundle variable SET: must be identical to the pre-FDA result
    cfg = synth.synth_cfg()
    cfg["fulldoc"]["enabled"] = False  # explicit: synth_cfg() enables the flag on its own copy
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["fulldoc"]["enabled"] is False  # the SHIPPED file is false
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("VERA_FULLDOC_BUNDLE_DIR", str(root))
    off_calls: list = []
    off = _ask(monkeypatch, off_calls)
    assert _fields(off) == _fields(base) and off_calls == base_calls == []
    assert off.evidence is None and off.result_kind == base.result_kind


def test_enabled_makes_no_writes_and_flag_is_what_switches_it(monkeypatch, tmp_path):
    _no_store(monkeypatch)
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    cfg["fulldoc"]["enabled"] = True
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("VERA_FULLDOC_BUNDLE_DIR", str(root))
    calls: list = []
    on = _ask(monkeypatch, calls)  # the checker stub returns None => a fallback, but the path ran
    assert on.evidence is not None and on.result_kind in ("answer", "reading_list")
    assert fulldoc_corpus.get_corpus(cfg) is not None
    cfg["fulldoc"]["enabled"] = False
    fulldoc_corpus.reset_for_tests()
    assert fulldoc_corpus.get_corpus(cfg) is None
