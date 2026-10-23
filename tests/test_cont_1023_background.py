"""CONT-1023: returned search results never end the acquisition effort.

Candidates a request found but could not finish (inline deadline, per-request candidate cut, transient failure) are
handed to the EXISTING background queue (vera.auto_acquire.enqueue_candidates -> run_job), de-duplicated by identity;
terminal rejections are not. Offline: search, web, lookup and store are synthetic fakes (tests/acq_support.py); the
worker thread is replaced by a no-op and queued jobs are run with auto_acquire.drain_for_tests."""
import pytest

from tests import acq_support as A
from tests import fulltext_synth
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from tests.test_dar_inline_acquire import Rig
from vera import auto_acquire, inline_acquire, research_outcome, research_record

Q = e2e.Q_EPI
QUEUED = inline_acquire.QUEUED


@pytest.fixture
def bg(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)  # jobs are drained by the test
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _dois(n, tag="c"):
    return [f"10.9999/acq.{tag}{i}" for i in range(n)]


def _web(dois, bad_licence=(), fail=()):
    recs, jats, over = {}, {}, {}
    for i, d in enumerate(dois):
        pmc = f"PMC{7000000 + i}"
        recs[d] = A.epmc_record(d, pmcid=pmc, licence="cc by-nd" if d in bad_licence else "cc by")
        jats[pmc] = fulltext_synth.jats(f"acq_{i}", "epi", d)
        if d in fail:
            over[f"/{pmc}/fullTextXML"] = A.Resp(status=503)
    return A.Web(recs, jats, over)


class Clock:
    def __init__(self, t=1.0):
        self.t = t

    def __call__(self):
        return self.t


def _queued_cands():
    q = auto_acquire._S.queue
    return [] if q is None else [c for j in list(q.queue) for c in (j.cands or ())]


def _deadline_run(env, clock, dois, extra=()):
    """Search returns the candidates and then the inline deadline has passed: nothing is evaluated in the request."""
    cfg = A.cfg()
    cands = [A.cand(d) for d in dois] + list(extra)
    d = A.deps(cands, _web(dois))

    def search(q, n, t):
        clock.t += 1000.0
        return list(cands)
    d.search = search
    o = inline_acquire.acquire(Q, cfg, env=env, clock=clock, deps_factory=lambda a, e: d)
    return cfg, o


# ------------------------------------------------------------------------------------------------ deadline
def test_deadline_hit_enqueues_every_remaining_eligible_candidate_deduped(bg):
    dois = _dois(4)
    dup = A.cand(dois[0])  # the same DOI twice: de-duplicated, never queued twice
    cfg, o = _deadline_run(dict(A.ENV_ON), Clock(), dois, extra=[dup])
    assert o.status == "deadline"
    queued = [c for c in o.candidates if c["outcome"] == QUEUED]
    assert sorted(c["doi"] for c in queued) == sorted(dois)
    assert all(c["unfinished"] == "not_evaluated" and c["reason"] == "R-ACQ-DEADLINE" for c in queued)
    assert sorted(A_doi for A_doi in (auto_acquire._doi(c) for c in _queued_cands())) == sorted(dois)
    assert not [c for c in o.candidates if c["outcome"] == "not_evaluated"]  # none dropped
    assert o.background == {"status": "ACQ-QUEUED", "queued": 4, "already_queued": 0, "refused": 0}
    assert o.public()["background"]["queued"] == 4


def test_second_identical_request_queues_no_duplicate_jobs(bg):
    dois = _dois(4)
    _deadline_run(dict(A.ENV_ON), Clock(), dois)
    n_jobs = auto_acquire._S.queue.qsize()
    inline_acquire.reset_for_tests()  # bypass the outcome cache: a genuinely new request
    _cfg, o2 = _deadline_run(dict(A.ENV_ON), Clock(2.0), dois)
    assert auto_acquire._S.queue.qsize() == n_jobs and len(_queued_cands()) == 4
    assert [c["background"] for c in o2.candidates] == ["already_queued"] * 4
    assert all(c["outcome"] == QUEUED for c in o2.candidates)
    assert o2.background["queued"] == 0 and o2.background["already_queued"] == 4
    # direct: the same candidates again (also by URL / OpenAlex id only) -> nothing new
    again = auto_acquire.enqueue_candidates(Q, [A.cand(d) for d in dois] + [{"url": f"https://doi.org/{dois[1]}"}],
                                            cfg=A.cfg(), env=A.ENV_ON, clock=Clock(3.0), start_worker=False)
    assert again["queued"] == [] and len(again["already"]) == 5 and auto_acquire._S.queue.qsize() == n_jobs


# ------------------------------------------------------------------------------------------------ candidate cut
def test_per_request_cut_of_six_with_nine_eligible_enqueues_three(bg):
    dois = _dois(9)
    cfg = A.cfg()
    assert int(cfg["bounds"]["max_candidates_resolved"]) == 6
    d = A.deps([A.cand(x) for x in dois], A.Web({}, {}))  # no Europe PMC record: the 6 evaluated end R-ACQ-NOTOA
    o = inline_acquire.acquire(Q, cfg, env=dict(A.ENV_ON), clock=Clock(), deps_factory=lambda a, e: d)
    by = {}
    for c in o.candidates:
        by.setdefault(c["outcome"], []).append(c)
    assert len(by[QUEUED]) == 3 and all(c["reason"] == "R-ACQ-DEFER_CAP" for c in by[QUEUED])
    assert len(by["rejected"]) == 6 and all(c["background"] == "no_route" for c in by["rejected"])
    assert len(_queued_cands()) == 3


# ------------------------------------------------------------------------------------------------ transient vs terminal
def test_transient_5xx_is_enqueued_and_licence_rejected_is_not(bg):
    ok_fail, nd = "10.9999/acq.t5xx", "10.9999/acq.tnd"
    cfg = A.cfg()
    d = A.deps([A.cand(ok_fail), A.cand(nd)], _web([ok_fail, nd], bad_licence={nd}, fail={ok_fail}))
    o = inline_acquire.acquire(Q, cfg, env=dict(A.ENV_ON), clock=Clock(), deps_factory=lambda a, e: d)
    rec = {c["doi"]: c for c in o.candidates}
    assert rec[ok_fail]["outcome"] == QUEUED and rec[ok_fail]["unfinished"] == "rejected"
    assert rec[ok_fail]["reason"] == "R-ACQ-UNREACHABLE"
    assert rec[nd]["outcome"] == "rejected" and rec[nd]["background"] == "terminal"
    assert research_outcome.reason_from_code(rec[nd]["reason"]) == "licence_rejected"
    assert [auto_acquire._doi(c) for c in _queued_cands()] == [ok_fail]


@pytest.mark.parametrize("code,want", [("R-ACQ-UNREACHABLE", True), ("R-ACQ-RATELIMITED", True),
                                       ("R-ACQ-DEADLINE", True), ("R-ACQ-ERR-ReadTimeout", True),
                                       ("R-ACQ-NOLIC", False), ("R-ACQ-INJECT", False), ("R-ACQ-IRRELEVANT", False),
                                       ("R-ACQ-FEWSPANS", False), ("R-ACQ-PREPRINT_NOT_PERMITTED", False),
                                       ("R-ABSTRACT", False), ("R-ACQ-NOTOA", False), ("R-ACQ-KNOWN", False)])
def test_only_reasons_that_can_change_are_eligible(code, want):
    assert inline_acquire.continue_eligible(A.cfg(), {"outcome": "rejected", "reason": code}) is want
    assert inline_acquire.continue_eligible(A.cfg(), {"outcome": "dropped", "reason": code}) is False


def test_url_only_candidate_is_followable_not_terminal():
    c = A.cfg()
    assert inline_acquire.continue_eligible(c, {"outcome": "dropped", "reason": "R-ACQ-NODOI", "url": "https://x.org/a"})
    assert not inline_acquire.continue_eligible(c, {"outcome": "dropped", "reason": "R-ACQ-NODOI", "url": None})


def test_url_only_candidates_are_queued_and_followed_through_the_existing_route(bg):
    pmc_doi = "10.9999/acq.u0"
    web = _web([pmc_doi])  # its Europe PMC record carries PMC7000000
    pmc_url = {"url": "https://europepmc.org/article/PMC/PMC7000000", "title": "SYNTHETIC outbreak transmission",
               "snippet": "transmission incidence outbreak pathogen"}
    bare = {"url": "https://example.org/some/landing-page", "title": "SYNTHETIC outbreak transmission",
            "snippet": "transmission incidence outbreak pathogen"}
    web.records["PMC7000000"] = web.records[pmc_doi]  # the PMCID resolve query matches by PMCID
    cfg = A.cfg()
    d = A.deps([pmc_url, bare], web)
    o = inline_acquire.acquire(Q, cfg, env=dict(A.ENV_ON), clock=Clock(), deps_factory=lambda a, e: d)
    rec = {c["url"]: c for c in o.candidates}
    for u in (pmc_url["url"], bare["url"]):
        assert rec[u]["outcome"] == QUEUED and rec[u]["route"] == "follow_url" and rec[u]["reason"] == "R-ACQ-NODOI"
    store = A.Store()
    jd = A.deps([], web, store=store)
    outs = auto_acquire.drain_for_tests(cfg, jd, clock=Clock(), env=A.ENV_ON)
    assert sum(r["stored"] for r in outs) == 1 and len(store.plans) == 1  # the PMC URL was followed and admitted
    assert sum(r["codes"].get("R-ACQ-NODOI", 0) for r in outs) == 1  # the bare URL awaits a route (no landing fetch)
    assert not any("example.org" in u for u in web.urls)  # never fetched: landing_route stays off


# ------------------------------------------------------------------------------------------------ switches
def test_kill_switch_on_means_nothing_enqueued_and_outcome_says_so(bg):
    env = dict(A.ENV_ON)
    cfg = A.cfg()
    dois = _dois(3, "k")
    cands = [A.cand(x) for x in dois]
    clock = Clock()
    d = A.deps(cands, _web(dois))

    def search(q, n, t):  # the operator pulls the kill switch while the request runs
        env["VERA_AUTOACQ_DISABLED"] = "1"
        clock.t += 1000.0
        return list(cands)
    d.search = search
    o = inline_acquire.acquire(Q, cfg, env=env, clock=clock, deps_factory=lambda a, e: d)
    assert _queued_cands() == [] and o.background["status"] == "R-ACQ-OFF" and o.background["queued"] == 0
    assert all(c["background"] == "R-ACQ-OFF" and c["outcome"] == "not_evaluated" for c in o.candidates)
    assert inline_acquire.background_note(cfg, o) == ""
    r = auto_acquire.enqueue_candidates(Q, cands, cfg=cfg, env=env, start_worker=False)
    assert r["status"] == "R-ACQ-OFF" and r["queued"] == [] and _queued_cands() == []


def test_no_write_account_refuses_the_enqueue_with_a_reason(bg):
    env = {"VERA_DB_URL_RO": A.ENV_ON["VERA_DB_URL_RO"]}
    r = auto_acquire.enqueue_candidates(Q, [A.cand(x) for x in _dois(2, "w")], cfg=A.cfg(), env=env,
                                        start_worker=False)
    assert r["status"] == "R-ACQ-NO_WRITE_DB" and r["queued"] == [] and len(r["refused"]) == 2
    assert _queued_cands() == []


# ------------------------------------------------------------------------------------------------ caps when jobs run
def test_caps_and_gates_apply_when_the_background_jobs_run(bg):
    dois = _dois(4, "r")
    cfg, o = _deadline_run(dict(A.ENV_ON), Clock(), dois)
    assert len(_queued_cands()) == 4
    store = A.Store()
    d = A.deps([], _web(dois, bad_licence={dois[3]}), store=store)

    def no_search(*a):
        raise AssertionError("a continuation job never searches")
    d.search = no_search
    outs = auto_acquire.drain_for_tests(cfg, d, clock=Clock(), env=A.ENV_ON)
    assert len(outs) == 2 and all(r["stored"] <= int(cfg["bounds"]["max_docs_per_job"]) for r in outs)
    assert sorted(p["source"]["provenance"]["doi"] for p in store.plans) == sorted(dois[:3])  # ND never stored
    assert all(r["trigger"] == "continue" for r in outs)


def test_hourly_job_cap_limits_what_is_queued_and_refused_ones_stay_eligible(bg):
    cfg = A.cfg(jobs_per_hour=2)  # the inline run reserves 1: room for ONE continuation job (2 candidates)
    dois = _dois(4, "h")
    cands = [A.cand(x) for x in dois]
    clock = Clock()
    d = A.deps(cands, _web(dois))

    def search(q, n, t):
        clock.t += 1000.0
        return list(cands)
    d.search = search
    o = inline_acquire.acquire(Q, cfg, env=dict(A.ENV_ON), clock=clock, deps_factory=lambda a, e: d)
    assert o.background == {"status": "R-ACQ-RATE", "queued": 2, "already_queued": 0, "refused": 2}
    assert len(_queued_cands()) == 2
    assert sorted(c["background"] for c in o.candidates) == ["R-ACQ-RATE", "R-ACQ-RATE", "queued", "queued"]
    later = auto_acquire.enqueue_candidates(Q, cands, cfg=A.cfg(jobs_per_hour=10), env=A.ENV_ON,
                                            clock=Clock(clock.t + 1), start_worker=False)  # cap raised, same window
    assert len(later["queued"]) == 2 and len(later["already"]) == 2  # refused ones were never marked pending


def test_kill_switch_rechecked_when_the_job_runs(bg):
    dois = _dois(2, "j")
    cfg, _o = _deadline_run(dict(A.ENV_ON), Clock(), dois)
    store = A.Store()
    outs = auto_acquire.drain_for_tests(cfg, A.deps([], _web(dois), store=store), clock=Clock(),
                                        env={**A.ENV_ON, "VERA_AUTOACQ_DISABLED": "1"})
    assert store.plans == [] and outs[0]["codes"] == {"R-ACQ-OFF": 1}


# ------------------------------------------------------------------------------------------------ response
def test_research_outcome_shows_queued_background_and_user_text_discloses_it(bg):
    _cfg, o = _deadline_run(dict(A.ENV_ON), Clock(), _dois(4, "o"))
    rec = research_record.summarize(triggers=["insufficient"], inline=o.public(), used_new=False,
                                    older_available=False, fr_before=None)
    pub = research_record.public_fields(rec)["research_outcome"]
    assert [c["terminal_reason"] for c in pub["candidates"]] == ["queued_background"] * 4
    assert all("pmcid" not in c for c in pub["candidates"])
    assert research_outcome.BACKGROUND_SENTENCE in pub["user_text"]
    assert pub["complete"] is False  # unfinished in the request: never reported as a complete search
    none = research_record.summarize(triggers=["insufficient"], inline={"status": "none", "candidates": [],
                                     "portals": [{"portal": "p", "status": "completed", "results": 0}]},
                                     used_new=False, older_available=False, fr_before=None)
    assert research_outcome.BACKGROUND_SENTENCE not in none["user_text"]


def test_offtopic_notice_discloses_background_acquisition(h, monkeypatch, bg):
    import time

    rig = Rig(h, monkeypatch)

    class Slow(A.Web):
        def __call__(self, *a, **k):
            time.sleep(0.4)
            return super().__call__(*a, **k)
    base = A.epi_setup()[0]
    rig.setup(web=Slow(base.records, base.jats))
    rig.cfg["inline"]["deadline_s"] = 0.05
    r = rig.run()
    assert rig.outcomes[0].status == "deadline" and rig.store.plans == []
    assert rig.cfg["scope_expansion"]["inline_notice_deadline"] in r.answer
    assert inline_acquire.BACKGROUND_NOTE in r.answer
    assert len(_queued_cands()) == 1


def test_url_only_queued_cause_is_awaiting_route_not_budget(bg):
    d = A.deps([{"url": "https://example.org/landing", "title": "SYNTHETIC outbreak transmission",
                 "snippet": "transmission incidence outbreak pathogen"}], A.Web({}, {}))
    o = inline_acquire.acquire(Q, A.cfg(), env=dict(A.ENV_ON), clock=Clock(), deps_factory=lambda a, e: d)
    pub = o.public()
    pub["portals"] = [{"portal": "p", "status": "completed", "results": 1}]
    rec = research_record.summarize(triggers=["insufficient"], inline=pub, used_new=False, older_available=False,
                                    fr_before=None)
    assert rec["complete"] is False and "no usable identifier yet" in rec["user_text"]
    assert "budget ran out" not in rec["user_text"]


def test_pending_candidate_identities_are_hashed(bg):
    auto_acquire.enqueue_candidates(Q, [A.cand("10.9999/acq.hash1")], cfg=A.cfg(), env=A.ENV_ON, start_worker=False)
    keys = list(auto_acquire._S.pending)
    assert keys and not any("10.9999" in k or "doi.org" in k for k in keys)
