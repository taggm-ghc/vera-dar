"""P0DTA-PRIO: candidate prioritisation (D1), search-time reservation (D3) and background quota reserve (D4).

OFFLINE: fake clock, fake portals and fake HTTP (tests/acq_support.py, tests/arxiv_synth.py); no network, DB or model.
The D1/D3/D4 cases are the work/p0dta-1023/d2 reproductions turned round to assert the corrected behaviour."""
import pytest

from tests import acq_support as A
from tests import arxiv_synth as X
from tests import fulltext_synth as fs
from tests.test_acq_discovered import Router, oa_src
from vera import acq_config, arxiv_fulltext, auto_acquire, inline_acquire, search_providers


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)  # never run queued jobs here
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq = A.cfg()
    acq["inline"]["enabled"] = True
    yield acq
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _epmc_hit(doi, pmcid):
    return {"url": f"https://doi.org/{doi}", "title": "SYNTHETIC acquired outbreak study", "snippet": "", "doi": doi,
            "year": 2021, "provider": "europepmc", "pmcid": pmcid,
            "oa": {"is_oa": True, "has_oa_location": True, "licence": None, "pmcid": pmcid, "epmc_ft": True}}


# ---------------------------------------------------------------------------------------------- D1 prioritisation
@pytest.mark.parametrize("kind", ["discovered", "ddg"])
def test_D1_routable_fulltext_candidate_takes_a_slot_before_no_oa_records(env, kind):
    good_doi, good_pmc = "10.9999/d2.good", "PMC7200001"
    web = A.Web({good_doi: A.epmc_record(good_doi, pmcid=good_pmc)}, {good_pmc: fs.jats("d2", "epi", good_doi)})
    snippet = "transmission incidence outbreak pathogen related during"
    low = [{"doi": f"10.9999/d2.paywall{i}", "url": f"https://doi.org/10.9999/d2.paywall{i}",
            "title": "outbreak transmission incidence pathogen", "snippet": snippet} for i in range(6)]
    kw = {}
    if kind == "discovered":
        kw["discovered"] = [dict(r, provider="openalex") for r in low]
        search = [_epmc_hit(good_doi, good_pmc)]
    else:
        search = [dict(r, provider="ddg", acq_source="ddg", ddg_id_kind="doi", snippet="") for r in low]
        search.append(_epmc_hit(good_doi, good_pmc))
    d = A.deps(search, web)
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d, **kw)
    by = {c["doi"]: c for c in out.public()["candidates"]}
    assert by[good_doi]["outcome"] == "admitted" and out.admitted == 1
    assert any(f"/{good_pmc}/fullTextXML" in u for u in web.urls)
    # the no-OA records only fill the 5 slots that remain; the 6th is cut and handed to the background (reserve allows)
    assert sum(by[r["doi"]]["reason"] == "R-ACQ-NOTOA" for r in low) == 5
    cut = [by[r["doi"]] for r in low if by[r["doi"]]["reason"] == "R-ACQ-DEFER_CAP"]
    assert len(cut) == 1 and cut[0]["background"] == "queued"


def test_D1_sort_uses_only_existing_oa_signals(env):
    c = env
    cands = [A.cand("10.9999/p.none0"), A.cand("10.9999/p.none1"),
             A.cand("10.9999/p.isoa", oa={"is_oa": True}),
             A.cand("10.9999/p.pmc", pmcid="PMC7300001"),
             A.cand("10.9999/p.ddg", acq_source="ddg", ddg_id_kind="doi"),
             {"url": X.ABS_URL, "doi": X.DOI, "identifier": f"arXiv:{X.AID}{X.VER}",
              "title": "SYNTHETIC outbreak transmission incidence preprint", "snippet": ""}]  # DDFIX-1023 D6: a real title (a 1-word placeholder is now tier 1)
    ranked = auto_acquire.rank_candidates(A.Q_EPI, cands, c, set(), auto_acquire._fd_cfg(), lambda code: None)
    dois = [r[1] for r in ranked]
    routable = {"10.9999/p.isoa", "10.9999/p.pmc", X.DOI}
    assert set(dois[:3]) == routable and set(dois[3:]) == {"10.9999/p.ddg", "10.9999/p.none0", "10.9999/p.none1"}
    assert dois[3] == "10.9999/p.ddg"  # owner order (DDG first) kept within the no-OA group
    c["arxiv_route"]["enabled"] = False  # the arXiv id counts only when the arXiv route can take it
    assert not auto_acquire.routable_fulltext(cands[-1], c)


# ---------------------------------------------------------------------------------------------- D3 search time
def _openalex_search(clk, calls, spend):
    class P:
        name = "openalex"

    def fake_search_question(question, n, **kw):
        calls.append(kw["timeout_s"])
        clk.t += spend(kw["timeout_s"])
        return []
    return [P()], fake_search_question


@pytest.mark.parametrize("spend,want_calls,want_elapsed", [
    (lambda t: t, [20.0], 20.0),        # AND call ran to its timeout: no OR retry (nothing left)
    (lambda t: 5.0, [20.0, 15.0], 10.0),  # AND call took 5 s: the OR retry gets the 15 s remainder, never a fresh 20
])
def test_D3_openalex_or_retry_gets_only_the_remainder(env, monkeypatch, spend, want_calls, want_elapsed):
    monkeypatch.setattr(search_providers, "_wiki_cfg", lambda: {"enabled": False})
    clk, calls = Clock(), []
    chain, fn = _openalex_search(clk, calls, spend)
    cfg = A.cfg()
    cfg["search"]["ddg_first"]["enabled"] = False
    cfg["search"]["europepmc"]["enabled"] = False
    base = auto_acquire.make_search(cfg, chain_fn=lambda: chain, search_fn=fn, clock=clk)
    search = search_providers.acquisition_search(base, cfg, clock=clk)
    t0 = clk()
    with search_providers.portal_scope():
        search("agentic orchestration archetypes", 10, 20.0)
    assert calls == want_calls and clk() - t0 == want_elapsed


def test_D3_base_search_is_capped_at_the_search_end(env, monkeypatch):
    monkeypatch.setattr(search_providers, "_wiki_cfg", lambda: {"enabled": False})
    clk, got = Clock(), []
    cfg = A.cfg()
    cfg["search"]["ddg_first"]["enabled"] = False

    def terms():  # the main thread spends 3 s before OpenAlex starts
        clk.t += 3.0
        return None, None

    def boom(*a, **k):
        raise RuntimeError("SYNTHETIC europepmc failure")
    search = search_providers.acquisition_search(lambda q, n, t: got.append(t) or [], cfg, clock=clk,
                                                 terms_fn=terms, transport=boom)
    with search_providers.portal_scope():
        search("q", 10, 20.0)
    assert got == [17.0]


@pytest.mark.parametrize("deadline_s,want", [(45, 20.0), (30, 5.0)])
def test_D3_inline_search_leaves_a_full_fetch_reserve(env, deadline_s, want):
    env["inline"]["deadline_s"] = deadline_s
    seen = []
    d = A.deps([], A.Web({}, {}))
    d.search = lambda q, n, t: seen.append(t) or []
    inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d)
    assert seen == [want]


def test_D3_no_evaluation_starts_without_the_fetch_reserve(env):
    clk = Clock()
    doi = "10.9999/d3.late"
    web = A.Web({doi: A.epmc_record(doi, pmcid="PMC7300009")}, {"PMC7300009": fs.jats("d3", "epi", doi)})
    d = A.deps([], web)

    def slow(q, n, t):  # the search overruns its 20 s cap: 24 s of the 45 s deadline left < per_doc_timeout_s 25
        clk.t += 21.0
        return [A.cand(doi, oa={"is_oa": True})]
    d.search = slow
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    (c,) = out.public()["candidates"]
    assert out.status == "deadline" and web.urls == []  # nothing resolved or fetched
    assert c["reason"] == "R-ACQ-DEADLINE" and (c["outcome"] == "not_evaluated" or c.get("unfinished") == "not_evaluated")


# ---------------------------------------------------------------------------------------------- D4 quota spend
def test_D4_url_only_handoff_stops_at_the_reserve_so_the_next_request_still_searches(env):
    urls = [{"url": f"https://example.org/post{i}", "title": "t", "snippet": "", "doi": None, "provider": "ddg",
             "acq_source": "ddg", "ddg_id_kind": "url", "discovered_url": f"https://example.org/post{i}"}
            for i in range(8)]
    d = A.deps(urls, A.Web({}, {}))
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d)
    # URL-only candidates are still queued for follow_url (CONT-1023 DF-3), but only while 2 hourly jobs are free:
    # inline 1 + 4 jobs (queue_max 4) = 5 of 6, leaving 1 job for the next request's search
    assert out.status == "none" and len(auto_acquire._S.jobs_hour) == 5
    recs = out.public()["candidates"]
    assert all(c["outcome"] == "queued_background" and c["route"] == "follow_url" for c in recs)
    inline_acquire.reset_for_tests()
    d2 = A.deps([dict(u, url=u["url"] + "x", discovered_url=u["url"] + "x") for u in urls], A.Web({}, {}))
    o2 = inline_acquire.acquire(A.Q_EPI.replace("outbreak", "epidemic"), env, env=A.ENV_ON, clock=Clock(1001.0),
                                deps_factory=lambda a, e: d2)
    assert o2.status != "rate" and len(auto_acquire._S.jobs_hour) == 6  # it searched; its hand-off is blocked
    assert {c["background"] for c in o2.public()["candidates"]} == {"skipped_quota"}


def test_D4_background_handoff_keeps_one_hourly_job_free(env):
    now = 1000.0
    for _ in range(3):  # 3 of jobs_per_hour 6 already spent this hour
        auto_acquire._S.jobs_hour.append(now)
        auto_acquire._S.jobs_day.append(now)
    dois = [f"10.9999/d4.r{i}" for i in range(10)]
    # 4 routable + 6 with no OA evidence; Europe PMC has no record for any of them
    cands = [A.cand(x, oa={"is_oa": True}) if i >= 6 else A.cand(x) for i, x in enumerate(dois)]
    clk = Clock(now)
    d = A.deps(cands, A.Web({}, {}))

    def slow(q, n, t):  # the fetch reserve is gone: every candidate is unfinished (eligible for the background)
        clk.t += 21.0
        return list(cands)
    d.search = slow
    out = inline_acquire.acquire(A.Q_EPI, env, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    # inline 1 (4 used, 2 free) -> ONE job of 2 candidates (5 used, 1 free) -> the next job is refused for quota
    assert out.background["queued"] == 2 and out.background["status"] == acq_config.reason(env, "rate")
    assert len(auto_acquire._S.jobs_hour) == 5
    by = {c["doi"]: c for c in out.public()["candidates"]}
    assert {x for x in dois if by[x]["background"] == "queued"} <= set(dois[6:])  # routable full text queued first
    assert any(c["background"] == "skipped_quota" for c in by.values())  # only what the reserve blocks
    inline_acquire.reset_for_tests()
    d.search = lambda q, n, t: list(cands)
    o2 = inline_acquire.acquire(A.Q_EPI + " again", env, env=A.ENV_ON, clock=Clock(clk.t + 1),
                                deps_factory=lambda a, e: d)
    assert o2.status != "rate"  # the reserved job lets the next request search
    assert int(acq_config.bound(env, "jobs_per_hour")) == 6 and int(acq_config.bound(env, "max_docs_per_job")) == 2


# ---------------------------------------------------------------------------------------------- live scenario
LIVE_Q = "Which are the prevailing proven-useful archetypes for agentic AI orchestrated agents?"


def test_live_agentic_archetypes_question_selects_fetches_and_admits_the_routable_arxiv_candidate(env, monkeypatch):
    """The live case: 9 corpus documents, 0 admitted, freshness + insufficiency triggers (prefer_recent, the served
    keys of the 9 documents), 2 discovered candidates (one arXiv:2605.23989v1), one portal timing out. The routable
    arXiv candidate is evaluated first, fetched inside the inline deadline and admitted by the REAL gates."""
    monkeypatch.setitem(fs.TOPICS, "agent", ["agentic", "orchestrated", "agents", "archetypes", "prevailing",
                                             "proven", "useful", "orchestrator", "planner", "worker", "pattern",
                                             "delegation", "agentic", "archetypes", "agents"])
    clk = Clock()
    t0 = clk()
    other_doi = "10.9999/live.paywalled"
    at: dict = {}

    class Net(Router):  # every request costs 2 s of the fake clock
        def __call__(self, url, **kw):
            clk.t += 2.0
            at.setdefault(url, clk.t)
            return super().__call__(url, **kw)
    web = Net({}, {}, X.routes(html=X.html_resp(X.latexml_html(topic="agent"))))
    d = A.deps([], web)

    def search(q, n, t):  # Europe PMC times out (consuming the capped search time); OpenAlex finds nothing new
        clk.t += t
        search_providers.record_portal("europepmc", "timeout", 0)
        search_providers.record_portal("openalex", "completed", 0)
        return []
    d.search = search
    cfg = env
    cfg["inline"]["use_discovered"] = True
    cfg["inline"]["fetch_workers"] = 1  # serial, so the evaluation order is observable
    served = {f"doi:10.9999/live.corpus{i}" for i in range(9)}
    other = oa_src(doi=other_doi, title="SYNTHETIC agentic agents orchestration overview",
                   snippet="agentic orchestrated agents archetypes")
    arx = dict(X.discovery_src(), snippet="agentic orchestrated agents archetypes")
    out = inline_acquire.acquire(LIVE_Q, cfg, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d, served=served,
                                 discovered=[other, arx], prefer_recent=True)
    pub = out.public()
    assert {"portal": "europepmc", "status": "timeout", "results": 0} in pub["portals"]
    assert out.status == "ok" and out.admitted == 1, pub
    assert X.HTML_URL in at and at[X.HTML_URL] - t0 <= cfg["inline"]["deadline_s"]  # fetched within the deadline
    epmc = [u for u in web.urls if "europepmc" in u]
    assert web.urls.index(X.HTML_URL) < (web.urls.index(epmc[0]) if epmc else len(web.urls))  # routable first
    adm = [c for c in pub["candidates"] if c["outcome"] == "admitted"]
    assert len(adm) == 1 and X.AID in str(adm[0]["doi"] or adm[0]["url"])
