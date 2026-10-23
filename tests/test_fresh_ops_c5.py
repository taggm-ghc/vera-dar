"""FRESH-C5 (part of FRESH-1023): pre-deploy OPERATIONAL tests of the freshness / insufficiency triggers on POST /ask.

Risk: nearly every question with stored evidence now triggers external discovery + acquisition. OFFLINE (same harness as
tests/test_fresh_acceptance_c4.py, itself built on tests/test_darb_http_e2e.py stubs + DB/socket guards). Config values come from
config/auto_acquire.json (read, never edited): bounds.jobs_per_hour 6, jobs_per_day 30, max_docs_per_job 2, docs_per_day null,
inline.deadline_s 45, job_deadline_s 120; public_mode.ASK_TIMEOUT_S 180 (the UI's client window). Kill switch VERA_AUTOACQ_DISABLED
(acq_config.killed); write switch VERA_DB_URL_WO (acq_config.db_configured: WO and RO env both needed).
Inline jobs count against the SAME caps (auto_acquire.reserve_job, clock injectable); the fake clock is passed as inline_acquire
acquire(clock=...), the freshness clock is the single hook vera.freshness.utcnow.
Fail-on-base contract: every test asserts a FEATURE field (triggers / research_outcome) before or with its operational claim.
"""
import json
import time

import pytest

from tests import accept_predicate as ap
from tests import acq_support as A
from tests import fulltext_synth as fs
from tests import test_fulldoc_e2e as e2e
from tests.test_fresh_acceptance_c4 import (HOURS, NEW_DOI, NOW, YEARS_AGO, World, _doc_ids, _grounded, _need,  # noqa: F401
                                            clock, no_db_no_socket, world)
from tests._pipeline_support import h, rig  # noqa: F401  (fixtures)
from vera import acq_config, auto_acquire, fulldoc_corpus, fulldoc_db, inline_acquire, public_mode

Q_REPL = "Why do many scientific studies fail to replicate?"
REPL_WORDS = ["replicate", "scientific", "studies", "fail", "findings", "reproducible", "research", "bias", "power",
              "published", "false", "results", "computational", "rules", "significance"]
OLD_DOI = "10.9999/old.epi1"
EXTRA = ["10.9999/acq.epi5", "10.9999/acq.epi6"]
HONEST = ("incomplete_search", "qualified_older_only")


_REAL_DB_CONFIGURED = acq_config.db_configured  # captured at import, before any harness stub replaces it


class FakeClock:
    """The inline-acquisition cap clock (time.monotonic in production): auto_acquire.reserve_job prunes 3600 s / 86400 s windows."""

    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _install_cap_clock(world, fc):
    prev = inline_acquire.acquire  # the rig's spy (records outcomes) around the real acquire

    def acquire(q, acq, **kw):
        kw["clock"] = fc
        return prev(q, acq, **kw)
    world.mp.setattr(inline_acquire, "acquire", acquire)


def _three_candidates(world):
    for i, d in enumerate(EXTRA):
        w2, _ = A.epi_setup(doi=d, pmcid=f"PMC91000{i}", jats_kw={"seed": 200 + i})
        world.web.records.update(w2.records)
        world.web.jats.update(w2.jats)
    world.cands = [A.cand(NEW_DOI)] + [A.cand(d) for d in EXTRA]


@pytest.fixture(autouse=True)
def _reset_state():
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _stale_world(world, fc):
    # The HTTP layer has its OWN public cap (public_mode.Caps: 6 req/min, 40 req/day per client): lift it for the storm so the test
    # exercises the acquisition caps, not the request limiter (reported separately as a second guard).
    public_mode.reset_limiter(public_mode.CapLimiter(public_mode.Caps(req_per_min_per_client=10**6, req_per_day_per_client=10**6,
                                                                       req_per_day_global=10**6)))
    world.mp.setattr(public_mode, "reset_limiter", public_mode.reset_limiter)
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30)))  # added 30 h ago -> 'stale' (corpus_added_at) on every ask
    world.rig.cfg["inline"]["result_ttl_s"] = 0  # no outcome reuse: every ask is a real acquisition attempt
    world.rig.cfg["inline"]["failure_ttl_s"] = 0
    _three_candidates(world)
    _install_cap_clock(world, fc)


def _ask_n(world, fc, n, step_s):
    bodies = []
    for i in range(n):
        fc.t += step_s
        bodies.append(world.ask())  # same question: result_ttl_s 0 defeats outcome reuse
    return bodies


def _jobs_run(world):
    return [o for o in world.rig.outcomes if o.status != "rate"]


# ---------------------------------------------------------------------------------------------------- shipped bounds pin
def test_shipped_bounds_and_budget_values():
    c = acq_config.load_config()
    b = lambda k: int(acq_config.bound(c, k))  # noqa: E731
    assert (b("jobs_per_hour"), b("jobs_per_day"), b("max_docs_per_job")) == (6, 30, 2)
    assert acq_config.docs_per_day_cap(c) is None
    assert float(c["inline"]["deadline_s"]) == 45 and b("job_deadline_s") == 120
    assert public_mode.ASK_TIMEOUT_S == 180
    from vera import ask_service
    need = ask_service.inline_need_s(c, object())
    assert need is not None and need <= public_mode.ASK_TIMEOUT_S, need  # worst inline retry fits the UI's 180 s window


# ---------------------------------------------------------------------------------------------------- (1) cap storm
def test_cap_storm_ten_stale_asks_in_one_hour(world):
    fc = FakeClock()
    _stale_world(world, fc)
    bodies = _ask_n(world, fc, 10, 300.0)  # 10 asks over 45 min
    assert all(_need(b, "triggers") == ["stale"] for b in bodies)  # the stale trigger fires every time ...
    assert len(_jobs_run(world)) <= 6, [o.status for o in world.rig.outcomes]  # ... but at most jobs_per_hour jobs run
    assert len(_jobs_run(world)) == 6
    plans_after_6 = len(world.rig.store.plans)
    assert plans_after_6 <= 6 * 2  # <= max_docs_per_job per job
    for b in bodies[6:]:
        ro = _need(b, "research_outcome")
        assert ro["kind"] == "incomplete_search" and ro["complete"] is False and ro["user_text"], ro
        assert not any(c["terminal_reason"] == "admitted" for c in ro.get("candidates") or [])
        if b["result_kind"] == "answer":  # older evidence may still answer, but only a grounded one: never fabricated
            _grounded(b)
            assert _doc_ids(b) <= {a["document_id"] for a in b["admitted_evidence"]}
            assert all(d.startswith("acq-") for d in _doc_ids(b))
    assert len(world.rig.store.plans) == plans_after_6  # no inserts after the cap is spent
    per_job = [o.stored for o in _jobs_run(world)]
    assert all(s <= 2 for s in per_job), per_job


def test_cap_storm_across_a_simulated_day(world):
    fc = FakeClock()
    _stale_world(world, fc)
    bodies = _ask_n(world, fc, 40, 1500.0)  # 40 asks spaced 25 min: <= 3 per hour, 16.7 h in total (inside one 24 h window)
    assert all(_need(b, "triggers") == ["stale"] for b in bodies)
    run = _jobs_run(world)
    assert len(run) <= 30 and len(run) == 30, [o.status for o in world.rig.outcomes]  # jobs_per_day
    assert len(world.rig.store.plans) <= 30 * 2
    assert [o.status for o in world.rig.outcomes[30:]] == ["rate"] * 10
    for b in bodies[30:]:
        ro = _need(b, "research_outcome")
        assert ro["kind"] == "incomplete_search" and ro["complete"] is False
    fc.t += 24 * 3600.0  # the window rolls: acquisition is possible again (caps are limits, not a lock-out)
    n_before = len(world.rig.outcomes)
    world.ask()
    assert world.rig.outcomes[n_before].status != "rate"


# ---------------------------------------------------------------------------------------------------- (2) latency bound
def test_slow_portals_honour_inline_deadline_and_return_honest_outcome(world):
    class Slow(A.Web):
        def __call__(self, *a, **k):
            time.sleep(0.4)
            return super().__call__(*a, **k)
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30)))
    world.rig.cfg["inline"]["deadline_s"] = 0.05  # the pipeline's own deadline mechanism (shipped value: 45 s)
    t0 = time.monotonic()
    body = world.ask(web=Slow(world.web.records, world.web.jats))
    took = time.monotonic() - t0
    assert _need(body, "triggers") == ["stale"]
    ro = _need(body, "research_outcome")
    assert ro["kind"] in HONEST and ro["user_text"], ro  # never 'hang', never 'no_valuable_source' for a search that ran out of time
    if ro["kind"] == "incomplete_search":
        assert ro["complete"] is False
    assert took < 15.0, took  # the slow fakes (0.4 s each) must not be waited out one by one
    assert world.rig.store.plans == []  # nothing admitted after the deadline
    assert world.rig.outcomes and world.rig.outcomes[0].status == "deadline"
    if body["result_kind"] == "answer":
        _grounded(body)  # qualified answer from the older stored passages only


def test_freshness_trigger_does_not_remove_the_retry_budget_guard():
    """The in-scope retry still needs deadline_s + 3 x chain timeout + 2 x grounding deadline <= the 180 s window."""
    from vera import ask_service
    c = acq_config.load_config()
    need = ask_service.inline_need_s(c, object())
    assert need == pytest.approx(45 + 3 * ask_service._chain_timeout_s() + 2 * 10)
    assert public_mode.ASK_TIMEOUT_S - need >= 40, (need, public_mode.ASK_TIMEOUT_S)


# ---------------------------------------------------------------------------------------------------- (3) kill switch
def test_kill_switch_stale_trigger_but_no_external_work(world, monkeypatch):
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30)))
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", "1")
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    assert _need(body, "freshness")["any_stale_or_unknown"] is True
    e = world.external
    assert e["acq_searches"] == 0 and e["http_urls"] == 0 and e["plans_written"] == 0, e
    assert e["discovery_searches"] == 0, e  # the disabled switch also stops the reading-list discovery the trigger would run
    ro = _need(body, "research_outcome")
    assert ro["kind"] in HONEST, ro  # honest: the search did not run -> not 'no_valuable_source', not 'answered_with_new'
    assert ro["kind"] != "no_valuable_source" and ro["user_text"]
    if ro["kind"] == "incomplete_search":
        assert ro["complete"] is False
    assert not any(c.get("terminal_reason") == "admitted" for c in ro.get("candidates") or [])
    _grounded(body)
    assert _doc_ids(body) and all(d.startswith("acq-") for d in _doc_ids(body))


# ---------------------------------------------------------------------------------------------------- (4) write switch absent
def test_no_wo_url_nothing_persisted_and_citations_are_admitted_only(world, monkeypatch):
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30)))
    monkeypatch.setattr(acq_config, "db_configured", _REAL_DB_CONFIGURED)  # honour the env, not the harness stub
    monkeypatch.delenv("VERA_DB_URL_WO", raising=False)
    monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://ro.invalid/x")  # harmless placeholder; the guards forbid any connection
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    assert world.rig.store.plans == []  # acquisition may run (answer-only) but NOTHING is written
    assert body["contract"].get("inline_acquisition", {}).get("persisted") in (None, False)
    ro = _need(body, "research_outcome")
    assert ro["kind"] in HONEST + ("answered_with_new",), ro
    adm = {a["document_id"] for a in body.get("admitted_evidence") or []}
    assert _doc_ids(body) <= adm  # cites only admitted passages
    _grounded(body)


# ---------------------------------------------------------------------------------------------------- (5) replication question
class ReplWorld(World):
    """Two stored documents on the replication topic, published 2005 and 2016 (year-only / full date)."""

    def store_repl(self, *specs):
        self.mp.setitem(fs.TOPICS, "repl", REPL_WORDS)
        plans = []
        for i, s in enumerate(specs, start=1):
            web, _ = A.epi_setup(doi=s["doi"], pmcid=f"PMC80000{i}", topic="repl", jats_kw={"seed": 300 + i})
            st = A.Store()
            from vera import auto_acquire as aa
            aa.run_job(A.job(Q_REPL), A.cfg(), A.deps([A.cand(s["doi"], title=s["title"], snippet="studies replicate findings")],
                                                      web, store=st), clock=A.clock_at(), env=A.ENV_ON)
            assert st.plans, f"fixture: {s['doi']} not admitted"
            p = st.plans[0]
            prov = p["source"]["provenance"]
            prov["published"] = prov["item"]["published_year"] = s["published"]
            t = HOURS(30).isoformat(timespec="seconds")  # FRESH-AMEND-1023: corpus-added >24 h -> stale trigger
            prov["item"]["fetched_at"] = prov["fetched_at"] = prov["retrieved_at"] = prov["acquisition"]["acquired_at"] = t
            plans.append(p)
        src, spans = A.rows_from_plans(plans)
        corpus = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())
        assert len(corpus.acquired) == len(plans), corpus.unusable
        self.mp.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg=None, env=None, now=None: corpus)
        return self


@pytest.mark.parametrize("pub", [("2005", "2016"), ("2005-03-01", "2016-06-15")])
def test_replication_question_stale_unknown_qualified_older_only(rig, monkeypatch, tmp_path, clock, no_db_no_socket, pub):
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    w = ReplWorld(rig, monkeypatch, tmp_path)
    w.store_repl(dict(doi="10.9999/rep.tsr", title="Ten Simple Rules for Reproducible Computational Research", published=pub[1]),
                 dict(doi="10.9999/rep.wmpf", title="Why Most Published Research Findings Are False", published=pub[0]))
    # no newer source: the only candidate is an off-topic outbreak paper, which the relevance gate rejects
    body = w.ask(q=Q_REPL)
    tr = _need(body, "triggers")
    assert set(tr) & {"stale", "freshness_unknown"}, tr
    assert _need(body, "freshness")["any_stale_or_unknown"] is True
    assert w.triggered(), w.external  # external verification was attempted
    ro = _need(body, "research_outcome")
    assert ro["kind"] == "qualified_older_only" and ro["complete"] is True and ro["user_text"], ro
    assert w.rig.store.plans == []  # nothing newer admitted
    _grounded(body)
    ids = _doc_ids(body)
    assert ids and all(i.startswith("acq-") for i in ids) and ids <= {a["document_id"] for a in body["admitted_evidence"]}
    assert ro["user_text"] in body["answer"] or ro["user_text"] == body.get("research_notice"), "freshness limitation not disclosed"
    assert not (body.get("contract") or {}).get("fallback")
