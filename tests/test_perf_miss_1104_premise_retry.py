"""PERF-MISS-1104-FIX (OD-1104-TRACE) regression: the two live USA 2026 midterm questions (build 53ff077) that were
refused with the premise_unverified text and ran no live search.

Replay seams (offline: no DB, no network, no LLM): the shipped config/fulldoc_answer.json (flag forced on), a weak
admitted bundle (off-subject 2022 passages that only share words with the question), a stubbed answer writer and
claim checker, a stubbed inline_acquire.acquire (counts acquisition attempts; finds nothing) and a stubbed reading
list. Real: ask_service.answer_in_scope -> _fulldoc -> fulldoc_answer.answer (premise wrapper, per-part handler,
claim check wiring) and the inline-retry gate.
"""
import json
from types import SimpleNamespace

import pytest

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests.test_fulldoc_answer import ATTR  # noqa: F401
from tests.test_fulldoc_answer import span
from vera import ask_service, claim_check, fulldoc_answer as fa, fulldoc_corpus, inline_acquire, scope_router
from vera.ask_service import AskResult
from vera.fulldoc_evidence import EvidenceBundle

QA = "What is newsworthy about rethe USA 2026 midterm elections?"
QB = "What make the news about the USA 2026 midterm elections?"
WEAK = (span(1, "Rumours about the 2022 midterm elections spread quickly on social media platforms in several states."),
        span(2, "An ethics index ranks election administration across states using survey data from 2021."))
GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 10,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}
REPLIES = {  # what the writer might say on these weak passages; none may become a premise refusal
    "unsupported_cited": "PREMISE: unsupported - The passages do not support anything about the 2026 midterms [1].",
    "unsupported_uncited": "PREMISE: unsupported - nothing about the 2026 midterms is in the passages.",
    "false_uncited": "PREMISE: false - The 2026 midterm elections have not taken place yet.",
    "false_cited_fails_check": "PREMISE: false - The midterm elections discussed took place in 2022, not 2026 [1].",
}


def _weak_bundle(question):
    docs = tuple({"document_id": s["document_id"], "canonical_source_id": s["canonical_source_id"], "title": s["title"],
                  "url": s["url"], "doi": None, "published": "2022", "licence": "CC-BY-4.0", "licence_decision": "allow",
                  "text_sha256": "cd" * 32, "chars": 999, "chunks_inspected": 5, "candidate_spans": 3,
                  "admitted_spans": 1} for s in WEAK)
    return EvidenceBundle(question=question, examined_full_documents=docs, admitted=WEAK, status="admitted")


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")
    cfg = fulldoc_corpus.load_config()
    cfg.setdefault("fulldoc", {})["enabled"] = True
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(fulldoc_corpus, "get_corpus", lambda *a, **k: object())
    monkeypatch.setattr(fulldoc_corpus, "get_acquired_corpus", lambda *a, **k: None)
    monkeypatch.setattr(ask_service, "_base_warming", lambda: False)
    from vera import fulldoc_admission
    monkeypatch.setattr(fulldoc_admission, "admit", lambda q, corpus, c: _weak_bundle(q))
    acq = A.cfg()
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: acq)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "inline_need_s", lambda *a, **k: 1.0)
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ask_service, "_reading_list", lambda *a, **k: AskResult(
        "Reading list.", 0, 0.0, result_kind="reading_list", contract={"layers": []}))
    st = SimpleNamespace(acquire=[], writer=[], checker=[], reply=None, verdict="unsupported")

    def acquire(q, a, **kw):
        st.acquire.append(q)
        return SimpleNamespace(corpus=None, status="none", candidates=None, public=lambda: {"status": "none"})

    monkeypatch.setattr(inline_acquire, "acquire", acquire)
    monkeypatch.setattr(inline_acquire, "lane_pass", lambda *a, **k: None)
    monkeypatch.setattr(inline_acquire, "background_note", lambda *a, **k: "")

    def complete(messages, model, pricing, **kw):
        st.writer.append(messages)
        return AskResult(st.reply, 10, 0.0)

    def check_call(chk, msgs):
        payload = json.loads(msgs[1]["content"])
        st.checker.append((msgs, payload))
        return json.dumps({"claims": [{"id": c["id"], "verdict": st.verdict, "reason": "r"} for c in payload["claims"]]})

    monkeypatch.setattr(ask_service, "complete", complete)
    real_check = claim_check.check_answer
    monkeypatch.setattr(claim_check, "check_answer", lambda *a, **k: real_check(*a, **{**k, "call": check_call}))
    monkeypatch.setattr(claim_check, "utc_now", lambda: __import__("datetime").datetime(
        2026, 11, 4, 12, 0, tzinfo=__import__("datetime").timezone.utc))
    return st


@pytest.mark.parametrize("question", [QA, QB])
@pytest.mark.parametrize("kind", sorted(REPLIES))
def test_midterm_premise_flag_on_weak_passages_starts_one_inline_search_and_no_premise_refusal(replay, question, kind):
    replay.reply = REPLIES[kind]
    r = ask_service.answer_in_scope(question, "m", e2e.PRICING, grounding=(None, GCFG))
    assert len(replay.acquire) == 1, f"{kind}: exactly one inline acquisition attempt"
    assert replay.writer, "the writer ran on the weak stored passages"
    assert fa.PREMISE_UNVERIFIED_TEXT not in (r.answer or "")
    assert "premise" not in (r.answer or "").lower()
    assert not (r.answer or "").startswith(fa.PREMISE_PREFIX)
    assert r.result_kind != "answer"
    assert (r.contract or {}).get("fallback") in ("premise_unverified", "model_insufficient", "no_valid_citations")
    assert (r.contract or {}).get("inline_acquisition", {}).get("status") == "none"
    assert ((r.contract or {}).get("no_answer_cause") or {}).get("code") != "premise_unverified"


def test_writer_and_checker_get_the_injected_utc_date_and_own_knowledge_rule(replay):
    replay.reply = REPLIES["false_cited_fails_check"]
    ask_service.answer_in_scope(QB, "m", e2e.PRICING, grounding=(None, GCFG))
    sys_w = replay.writer[0][0]["content"]
    assert "Today's date (UTC) is 2026-11-04." in sys_w and "from your own knowledge" in sys_w
    msgs, payload = replay.checker[0]
    assert payload["current_date_utc"] == "2026-11-04"
    assert "never judge whether an event" in msgs[0]["content"] and "own knowledge" in msgs[0]["content"]


def test_cited_verified_false_premise_correction_is_still_shown(replay):
    replay.reply = "PREMISE: false - The passage says the rumours concerned the 2022 midterm elections [1]."
    replay.verdict = "supported"
    r = ask_service.answer_in_scope(QB, "m", e2e.PRICING, grounding=(None, GCFG))
    assert r.result_kind == "answer" and r.answer.startswith(fa.PREMISE_PREFIX) and "[1]" in r.answer
    assert r.claim_check["premise_status"] == "false" and r.claim_check["premise_verified"] is True
    # OD-1109-SIX: a verified answer still runs the ONE research attempt for new documents (never the retry loop)
    assert len(replay.acquire) == 1
