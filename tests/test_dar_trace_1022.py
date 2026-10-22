"""DAR-TRACE-1022: inline answer-path drop points (offline, synthetic fixtures only; no network, DB or model call)."""
import json

from tests import acq_support as A
from tests.test_dar_inline_acquire import OFF_Q, direct, rig  # noqa: F401  (fixtures)
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from tests.test_dar_inline_r2 import _first_run_plan, _new_process
from vera import ask_service, auto_acquire as aa, inline_acquire
from vera import search_providers as sp

OTHER_Q = "What are the different types of horses?"


# ---- 1. a stored document is reused only when it passes THIS question's relevance gate -----------------------------
def test_stored_document_irrelevant_to_the_new_question_is_not_reused(rig):
    plan = _first_run_plan(rig)
    _store, calls = _new_process(rig, plan)
    o = inline_acquire.acquire(OTHER_Q, rig.cfg, env=A.ENV_ON, deps_factory=lambda a, e: rig.deps)
    assert calls, "stored rows were looked up"
    assert o.reused == 0 and o.codes.get(inline_acquire.REUSE_IRRELEVANT) == 1
    assert o.corpus is None  # nothing relevant: no ephemeral corpus of an off-question stored document


def test_stored_document_relevant_to_the_question_is_still_reused(rig):
    plan = _first_run_plan(rig)
    _new_process(rig, plan)
    o = inline_acquire.acquire(OFF_Q, rig.cfg, env=A.ENV_ON, deps_factory=lambda a, e: rig.deps)
    assert o.reused == 1 and o.corpus is not None and inline_acquire.REUSE_IRRELEVANT not in o.codes


def test_irrelevant_stored_documents_do_not_use_up_the_fresh_budget(rig):
    """The 65af0ac defect: unchecked reuse filled max_docs_per_job, so fresh_budget was 0 and nothing fresh ran."""
    plan = _first_run_plan(rig)
    _new_process(rig, plan)
    rig.cfg["bounds"]["max_docs_per_job"] = 1  # fix2 verifier must-fix 3: one irrelevant stored doc WOULD fill the cap
    seen = []
    orig = inline_acquire._evaluate_parallel

    def spy(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, budget, count, ms):
        seen.append(budget)
        return orig(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, budget, count, ms)
    import pytest
    mp = pytest.MonkeyPatch()
    mp.setattr(inline_acquire, "_evaluate_parallel", spy)
    # a second, fresh (not stored) candidate so the fresh stage has something to evaluate
    fresh = dict(rig.deps.search(OTHER_Q, 10, 5)[0]) if rig.deps.search(OTHER_Q, 10, 5) else None
    try:
        if fresh is not None:
            fresh.update(doi="10.9999/acq.fresh1", url="https://doi.org/10.9999/acq.fresh1")
            base = rig.deps.search
            rig.deps.search = lambda q, n, t: list(base(q, n, t)) + [fresh]
        inline_acquire.acquire(OTHER_Q, rig.cfg, env=A.ENV_ON, deps_factory=lambda a, e: rig.deps)
    finally:
        mp.undo()
    assert seen and seen[0] >= 1  # review M2: the fresh stage ran with a budget an irrelevant stored doc did not eat


# ---- 2. failed outcomes are replayed for at most failure_ttl_s --------------------------------------------------------
def test_failed_outcome_cache_expires_after_failure_ttl(monkeypatch):
    inline_acquire.reset_for_tests()
    t = [1000.0]
    cfg = A.cfg()
    cfg["inline"] = {**cfg.get("inline", {}), "enabled": True, "result_ttl_s": 600, "failure_ttl_s": 60}
    runs = []

    def fake_run(question, acq, env, clock, deps_factory, served=None):
        runs.append(1)
        return inline_acquire.Outcome("search_failed")
    monkeypatch.setattr(inline_acquire, "_run", fake_run)
    clock = lambda: t[0]  # noqa: E731
    inline_acquire.acquire("q one", cfg, env=A.ENV_ON, clock=clock)
    t[0] += 30
    inline_acquire.acquire("q one", cfg, env=A.ENV_ON, clock=clock)
    assert len(runs) == 1  # replayed inside 60 s
    t[0] += 31
    inline_acquire.acquire("q one", cfg, env=A.ENV_ON, clock=clock)
    assert len(runs) == 2  # a failure is not replayed for the full 600 s


def test_deterministic_none_keeps_the_full_result_ttl(monkeypatch):
    """Review S1: only transient failures (search_failed, deadline) get failure_ttl_s; "none" is deterministic."""
    inline_acquire.reset_for_tests()
    t = [1000.0]
    cfg = A.cfg()
    cfg["inline"] = {**cfg.get("inline", {}), "enabled": True, "result_ttl_s": 600, "failure_ttl_s": 60}
    runs = []

    def fake_run(question, acq, env, clock, deps_factory, served=None):
        runs.append(1)
        return inline_acquire.Outcome("none")
    monkeypatch.setattr(inline_acquire, "_run", fake_run)
    clock = lambda: t[0]  # noqa: E731
    inline_acquire.acquire("q none", cfg, env=A.ENV_ON, clock=clock)
    t[0] += 300
    inline_acquire.acquire("q none", cfg, env=A.ENV_ON, clock=clock)
    assert len(runs) == 1


def test_successful_outcome_still_cached_for_result_ttl(monkeypatch):
    inline_acquire.reset_for_tests()
    t = [1000.0]
    cfg = A.cfg()
    cfg["inline"] = {**cfg.get("inline", {}), "enabled": True, "result_ttl_s": 600, "failure_ttl_s": 60}
    runs = []

    def fake_run(question, acq, env, clock, deps_factory, served=None):
        runs.append(1)
        return inline_acquire.Outcome("ok", corpus=object(), admitted=1)
    monkeypatch.setattr(inline_acquire, "_run", fake_run)
    clock = lambda: t[0]  # noqa: E731
    inline_acquire.acquire("q two", cfg, env=A.ENV_ON, clock=clock)
    t[0] += 300
    inline_acquire.acquire("q two", cfg, env=A.ENV_ON, clock=clock)
    assert len(runs) == 1


# ---- 3. capped notice --------------------------------------------------------------------------------------------------
def test_rate_and_daily_cap_get_their_own_notices_not_the_background_search_one():
    from vera import acq_config
    cfg = acq_config.load_config()
    se = cfg["scope_expansion"]
    assert "no background search was started" in se["inline_notice_capped"]
    assert "no background search was started" in se["inline_notice_doc_cap"]
    assert inline_acquire.notice(cfg, "rate") == se["inline_notice_capped"]  # job caps: no search was made
    assert inline_acquire.notice(cfg, "daily_cap") == se["inline_notice_doc_cap"]  # review S2: a search DID run
    assert inline_acquire.notice(cfg, "busy") is None and inline_acquire.notice(cfg, "off") is None


# ---- 4. pre-fetch ranking ---------------------------------------------------------------------------------------------
def test_rank_question_drops_generic_terms_only():
    cfg = {"search": {"openalex_profile": {"generic_terms": ["different", "types", "key", "differences"]}}}
    assert aa.rank_question("What are the different types of horses?", cfg) == "What are the of horses?"
    assert aa.rank_question("types", cfg) == "types"  # nothing distinctive left: unchanged
    assert aa.rank_question("horse breeds", {}) == "horse breeds"


def test_ties_keep_provider_order_not_doi_order():
    cands = [{"doi": f"10.9999/z.{9 - i}", "url": f"https://doi.org/10.9999/z.{9 - i}", "title": "Unrelated title",
              "provider": "europepmc", "oa": {"epmc_ft": True}} for i in range(5)]
    out = aa.rank_candidates("widgets", cands, A.cfg(), set(), aa._fd_cfg(), lambda c: None)
    assert [d for _s, d, _c, _i in out] == [c["doi"] for c in cands]  # 9, 8, 7, ... (provider order), not sorted


# ---- 5. Europe PMC OR retry -----------------------------------------------------------------------------------------
class _Resp:
    status_code = 200

    def __init__(self, rows):
        self.text = json.dumps({"resultList": {"result": rows}})


def _epmc_cfg():
    cfg = A.cfg()
    s = cfg.setdefault("search", {})
    s["openalex_profile"] = {"and_extra_max": 4, "generic_terms": ["types"]}
    s["ddg_first"] = {"enabled": True}
    s["europepmc"] = {"enabled": True, "search_url": "https://example.invalid/epmc", "query_suffix": "OPEN_ACCESS:y",
                      "page_size": 5, "refuse_sources": ["PPR"]}
    cfg.setdefault("bounds", {})["max_search_calls_per_job"] = 5
    return cfg


def _terms():
    return sp._default_terms_and_profile(A.cfg())[0]


PROFILE = {"and_extra_max": 4, "generic_terms": ["types"]}
Q_NO_CORE = "horse breeds classification"


def _run_epmc(first, ddg_results, question=Q_NO_CORE, terms=None):
    """first: a list of rows (HTTP 200) or an exception to raise on the FIRST Europe PMC call."""
    queries = []

    def transport(url, params, headers, timeout):
        queries.append(params["query"])
        if len(queries) == 1 and isinstance(first, Exception):
            raise first
        return _Resp(first if len(queries) == 1 else [
            {"pmcid": "PMC123", "isOpenAccess": "Y", "source": "MED", "doi": "10.9999/or.1", "title": "Or hit"}])
    t = _terms() if terms is None else (None if terms == "legacy" else terms)
    search = sp.acquisition_search(lambda q, n, tm: [], _epmc_cfg(), transport=transport,
                                   terms_fn=lambda: (t, PROFILE), ddg_search=lambda q, n, tm: ddg_results)
    return search(question, 5, 5), queries


def test_or_query_differs_for_a_question_without_a_core_phrase():
    assert sp._or_differs(Q_NO_CORE, _terms(), PROFILE, "S")
    assert not sp._or_differs(Q_NO_CORE, _terms(), None, "S")


def test_epmc_and_empty_retries_once_with_or_when_ddg_gave_nothing():
    sp._QUOTA.reset()
    out, queries = _run_epmc([], [])
    assert len(queries) == 2 and queries[0] != queries[1] and [c["doi"] for c in out] == ["10.9999/or.1"]


def test_epmc_no_retry_when_first_query_has_hits():
    sp._QUOTA.reset()
    rows = [{"pmcid": "PMC9", "isOpenAccess": "Y", "source": "MED", "doi": "10.9999/and.1", "title": "And hit"}]
    out, queries = _run_epmc(rows, [])
    assert len(queries) == 1 and [c["doi"] for c in out] == ["10.9999/and.1"]


def test_epmc_no_retry_after_a_failed_first_call():
    """Review M3(a): a 429 / transport error / timeout on the first call is never retried at once."""
    sp._QUOTA.reset()
    _out, queries = _run_epmc(sp.ProviderResponseError("europepmc search HTTP 429"), [])
    assert len(queries) == 1


def test_epmc_no_retry_when_the_or_query_is_identical():
    """Review M3(b): no terms config -> legacy words, identical with or without the profile -> no wasted call."""
    sp._QUOTA.reset()
    _out, queries = _run_epmc([], [], terms="legacy")
    assert len(queries) == 1


# ---- 6. tokens_used keeps the writer/check spend after an inline fallback -----------------------------------------
def test_inline_note_carries_spend():
    n = ask_service._Note("notice", 1998, 0.0012)
    assert str(n) == "notice" and ask_service._spent(n) == {"tokens_used": 1998, "cost_usd": 0.0012}
    assert ask_service._spent("plain") == {} and ask_service._spent(None) == {}


def test_in_scope_tokens_used_adds_the_inline_spend_to_the_base_attempt(monkeypatch):
    """Review M1: in-scope, the base writer spend (tuple) PLUS the inline spend, the classifier counted once."""
    from vera import acq_config, fulldoc_corpus, scope_router
    ft = fulldoc_corpus.load_config()["fallback_tokens"]
    dec = scope_router.ScopeDecision(True, "layer2_llm", "llm_in:x", 400, 0.004, 1, "in_scope")
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: dec)
    cfg = acq_config.load_config()
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: cfg)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: (None, ft["insufficient"], 1500, 0.015, "insufficient"))
    monkeypatch.setattr(ask_service, "_inline", lambda *a, **k: ask_service._Note("note", 900, 0.009))
    r = ask_service.answer_in_scope("q", "m", None)
    assert r.tokens_used == 1500 + 900 - 400 and abs(r.cost_usd - (0.015 + 0.009 - 0.004)) < 1e-9


def test_off_topic_tokens_used_is_the_inline_total(monkeypatch):
    from vera import acq_config, scope_router
    dec = scope_router.ScopeDecision(False, "layer2_llm", "llm_out:x", 400, 0.004, 1, "off_topic")
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: dec)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: acq_config.load_config())
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: None)
    monkeypatch.setattr(ask_service, "_inline", lambda *a, **k: ask_service._Note("note", 1998, 0.0))
    monkeypatch.setattr(ask_service, "_offtopic_reading_list", lambda d, g: ask_service.AskResult(
        answer="list", tokens_used=400, cost_usd=0.004, result_kind="reading_list"))
    r = ask_service.answer_in_scope("q", "m", None)
    assert r.tokens_used == 1998 and r.answer.endswith("note")


# ---- 7. reuse: reader refusals are not relevance misses; the reuse check honours the deadline -------------------------
def test_reuse_reader_refusal_is_counted_unusable_not_irrelevant(rig):
    plan = _first_run_plan(rig)
    _store, _calls = _new_process(rig, plan)
    bad = rig.deps.stored_rows

    def tampered(cus):
        srcs, spans = bad(cus)
        for r in srcs:
            r["content"] = r["content"] + " tampered"  # hash no longer matches: the reader refuses the row
        return srcs, spans
    rig.deps.stored_rows = tampered
    o = inline_acquire.acquire(OFF_Q, rig.cfg, env=A.ENV_ON, deps_factory=lambda a, e: rig.deps)
    assert o.codes.get(inline_acquire.REUSE_UNUSABLE) == 1 and inline_acquire.REUSE_IRRELEVANT not in o.codes


def test_reuse_check_stops_at_the_deadline(rig):
    plan = _first_run_plan(rig)
    _new_process(rig, plan)
    ranked = [(1.0, "10.9999/acq.epi4", {"doi": "10.9999/acq.epi4", "url": "https://doi.org/10.9999/acq.epi4"}, None)]
    codes = {}
    reused, rest = inline_acquire._stored_for(ranked, rig.cfg, rig.deps, lambda c: codes.__setitem__(c, codes.get(c, 0) + 1),
                                              question=OFF_Q, fdcfg=aa._fd_cfg(), icfg={}, clock=lambda: 10.0,
                                              deadline=5.0)
    assert reused[0] == [] and codes.get("R-ACQ-DEADLINE") == 1
