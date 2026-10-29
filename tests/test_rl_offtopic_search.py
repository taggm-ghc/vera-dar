"""RL-REL-1021, OFFLINE: the off-topic reading list ANDs distinctive terms (OpenAlex), in-scope searches unchanged.
Synthetic fixtures only; classifier and transport are stubs. No network, no model."""
import json
from pathlib import Path

import pytest

from vera import ask_service, scope_router as sr, search_providers as sp
from vera.pricing.config import PricingRecord

ROOT = Path(__file__).resolve().parent.parent
RAW = json.loads((ROOT / "config" / "scope_router.json").read_text())
QT = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["query_terms"]
OS = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]["openalex"]
PRICING = PricingRecord(provider="p", model="m", input=0.1, output=0.4, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
Q = "What are key differences between boxing and mma?"
BODY = json.dumps({"results": [{"id": "https://openalex.org/W1", "display_name": "Synthetic Title",
                                "doi": "https://doi.org/10.9999/x", "publication_year": 2024}]})


class _HTTP:
    status_code, text, headers = 200, BODY, {"Content-Type": "application/json"}


def _grounding(calls):
    def tx(url, params, headers, timeout):
        calls.append(dict(params))
        return _HTTP()
    prov = sp.OpenAlexProvider(OS, transport=tx, query_terms=QT)

    def sources_fn(restricted=False):
        prov.call(Q, 10, 5)
        return []
    return sources_fn, {"no_sources_text": "NO"}


def _route(monkeypatch, cfg_raw, reply):
    cfg = sr.parse_config(cfg_raw)
    monkeypatch.setattr(sr, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(sr, "default_chat", lambda m, mo, l2: (reply, 10, 1))


OUT = '{"in_scope": false, "reason": "sports"}'


def test_shipped_profile_present_without_pmc_filter():
    p = sr.offtopic_openalex_profile(sr.parse_config(RAW))
    assert p["and_extra_max"] == 4 and "key" in p["generic_terms"] and "filter" not in p


@pytest.mark.parametrize("bad", [None, {}, {"and_extra_max": 0}, {"and_extra_max": "x"}, "s"])
def test_missing_or_malformed_profile_is_none(bad):
    raw = {k: v for k, v in RAW.items() if k != "offtopic_openalex_profile"}
    if bad is not None:
        raw["offtopic_openalex_profile"] = bad
    assert sr.offtopic_openalex_profile(sr.parse_config(raw)) is None


def test_offtopic_reading_list_search_is_term_and(monkeypatch):
    _route(monkeypatch, RAW, OUT)
    calls = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_grounding(calls))
    assert [c["search"] for c in calls][0] == "boxing AND mma"  # empty fixture -> B fallback adds the OR call (below)
    assert [c["search"] for c in calls][1:] == ["(key OR differences OR boxing OR mma)"]
    assert "S2764455111" not in calls[0]["filter"]  # no PMC-only filter on the list


def test_offtopic_profile_removed_restores_or_query(monkeypatch):
    _route(monkeypatch, {k: v for k, v in RAW.items() if k != "offtopic_openalex_profile"}, OUT)
    calls = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_grounding(calls))
    assert calls[0]["search"] == "(key OR differences OR boxing OR mma)"


def _route_no_catstep(monkeypatch):
    # NEXTREL-1028: these tests exercise the D-P10b off-topic list; CATSTEP-1028 (step on) routes a missing category elsewhere.
    raw = json.loads(json.dumps(RAW))
    raw["unknown_category_step"]["enabled"] = False
    _route(monkeypatch, raw, OUT)


def test_profile_is_reset_after_the_offtopic_list_even_on_error(monkeypatch):
    _route_no_catstep(monkeypatch)
    seen = []

    def boom(restricted=False):
        seen.append(sp.OPENALEX_PROFILE.get())
        raise RuntimeError("x")
    res = ask_service.answer_in_scope(Q, "m", PRICING, grounding=(boom, {}))
    assert seen and seen[0] is not None and seen[0]["and_extra_max"] == 4
    assert res.result_kind == "decline" and sp.OPENALEX_PROFILE.get() is None


def test_in_scope_search_is_byte_identical(monkeypatch):
    _route(monkeypatch, RAW, '{"in_scope": true, "reason": "ok"}')
    seen = []
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: None)

    def sources_fn(restricted=False):
        seen.append(sp.OPENALEX_PROFILE.get())
        return []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=(sources_fn, {"no_sources_text": "NO"}))
    assert seen == [None]
    assert sp.build_openalex_search(Q, QT) == ("(key OR differences OR boxing OR mma)", "extra_only")


# ---- RL-REL-1021-B: one fallback call (old OR query) when the AND-profiled off-topic search is empty ----
def _seq(replies, seen):
    def sources_fn(restricted=False):
        seen.append((restricted, sp.OPENALEX_PROFILE.get()))
        return replies[min(len(seen), len(replies)) - 1]
    return sources_fn, {"no_sources_text": "NO"}


SRC = [{"title": "Synthetic Title", "doi": "10.9999/x", "year": 2024}]


def test_fallback_fires_on_empty_and_result_with_profile_cleared(monkeypatch):
    _route(monkeypatch, RAW, OUT)
    seen = []
    res = ask_service.answer_in_scope(Q, "m", PRICING, grounding=_seq([[], SRC], seen))
    assert len(seen) == 2 and seen[0][1] is not None and seen[1] == (True, None)
    assert res.contract.get("offtopic") is True and res.result_kind != "decline"
    assert sp.OPENALEX_PROFILE.get() is None


def test_fallback_does_not_fire_on_nonempty_result(monkeypatch):
    _route(monkeypatch, RAW, OUT)
    seen = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_seq([SRC, SRC], seen))
    assert len(seen) == 1 and seen[0][1] is not None


def test_fallback_is_at_most_one_extra_call_and_resets(monkeypatch):
    _route(monkeypatch, RAW, OUT)
    seen = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_seq([[], [], []], seen))
    assert len(seen) == 2 and sp.OPENALEX_PROFILE.get() is None


def test_no_fallback_when_profile_absent(monkeypatch):
    _route(monkeypatch, {k: v for k, v in RAW.items() if k != "offtopic_openalex_profile"}, OUT)
    seen = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_seq([[], SRC], seen))
    assert len(seen) == 1


def test_fallback_error_resets_profile(monkeypatch):
    _route_no_catstep(monkeypatch)
    n = []

    def fn(restricted=False):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("x")
        return []
    res = ask_service.answer_in_scope(Q, "m", PRICING, grounding=(fn, {}))
    assert res.result_kind == "decline" and len(n) == 2 and sp.OPENALEX_PROFILE.get() is None


def test_in_scope_never_falls_back(monkeypatch):
    _route(monkeypatch, RAW, '{"in_scope": true, "reason": "ok"}')
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: None)
    seen = []
    ask_service.answer_in_scope(Q, "m", PRICING, grounding=_seq([[], SRC], seen))
    assert len(seen) == 1 and seen[0] == (False, None)


# ---- RL-REL-1021-C: main.py's real sources_fn keys its restricted cache by the active OpenAlex profile ----
MSRC = [{"n": 1, "title": "Synthetic Title", "url": "https://arxiv.org/abs/2401.00001", "published": "2025-01-01",
         "snippet": "S", "identifier": "arXiv:2401.00001", "provider": "arxiv", "retrieved_at": "2026-10-01"}]


def _main_ask(monkeypatch, chain, by_profile):
    """POST /ask through main.ask with retrieve_all stubbed (synthetic rows keyed on the active profile)."""
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from fastapi.testclient import TestClient
    from vera import public_mode as pm
    pm.reset_limiter(None)
    calls = []

    def fake_retrieve_all(question, cfg, **kw):
        prof = sp.OPENALEX_PROFILE.get()
        calls.append(prof)
        return by_profile(prof), []
    monkeypatch.setattr(main, "GROUNDING", {"no_sources_text": "NO", "providers": []})
    monkeypatch.setattr(main, "retrieve_all", fake_retrieve_all)
    monkeypatch.setattr(main, "answer_via_chain", chain)
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    r = TestClient(main.app).post("/ask", json={"question": Q}, headers={"X-API-Key": "k" * 20})
    return r, calls


def _decline():
    return ask_service.AskResult(answer="D", tokens_used=0, cost_usd=0.0, result_kind="decline")


def test_main_fallback_gets_fresh_or_results(monkeypatch):
    _route(monkeypatch, RAW, OUT)
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: None)
    chain = lambda q, e, g: (ask_service.answer_in_scope(q, "m", PRICING, grounding=g), "x")  # noqa: E731
    r, calls = _main_ask(monkeypatch, chain, lambda prof: [] if prof is not None else MSRC)
    assert r.status_code == 200, r.text
    assert len(calls) == 2 and calls[0] is not None and calls[1] is None
    assert r.json()["result_kind"] == "reading_list" and len(r.json()["candidates"]) == 1


def test_main_profiled_call_after_unprofiled_is_not_cached(monkeypatch):
    seen = []

    def chain(q, e, g):
        fn = g[0]
        seen.append(fn(restricted=True))
        tok = sp.OPENALEX_PROFILE.set({"and_extra_max": 4, "generic_terms": []})
        try:
            seen.append(fn(restricted=True))
        finally:
            sp.OPENALEX_PROFILE.reset(tok)
        return _decline(), "x"
    r, calls = _main_ask(monkeypatch, chain, lambda prof: [] if prof is None else MSRC)
    assert r.status_code == 200, r.text
    assert len(calls) == 2 and calls[0] is None and calls[1] is not None and seen == [[], MSRC]


def test_main_repeated_identical_call_is_cached(monkeypatch):
    seen = []

    def chain(q, e, g):
        fn = g[0]
        tok = sp.OPENALEX_PROFILE.set({"and_extra_max": 4, "generic_terms": ["key"]})
        try:
            seen.extend([fn(restricted=True), fn(restricted=True)])
        finally:
            sp.OPENALEX_PROFILE.reset(tok)
        seen.extend([fn(restricted=True), fn(restricted=True)])
        return _decline(), "x"
    r, calls = _main_ask(monkeypatch, chain, lambda prof: MSRC if prof else [])
    assert r.status_code == 200, r.text
    assert len(calls) == 2 and seen == [MSRC, MSRC, [], []]
