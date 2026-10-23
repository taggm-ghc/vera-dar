"""FRESH-1023 C3: portal attempt recorder, per-candidate records, drops, prefer_recent. Offline, synthetic."""
import pytest

from tests import acq_support as A
from vera import acq_config, auto_acquire, inline_acquire, search_providers
from vera.m7 import live_report


def test_record_portal_without_scope_is_a_noop_and_scope_collects():
    search_providers.record_portal("openalex", "completed", 3)  # no scope: no raise
    with search_providers.portal_scope() as lst:
        search_providers.record_portal("ddg", "quota")
        search_providers.record_portal("openalex", "completed", 2)
    assert lst == [{"portal": "ddg", "status": "quota", "results": 0},
                   {"portal": "openalex", "status": "completed", "results": 2}]
    assert search_providers.PORTALS.get() is None


def test_acquisition_search_records_each_portal_once():
    cfg = {"search": {"openalex_profile": False, "europepmc": {"enabled": False}, "ddg_first": {"enabled": False}},
           "bounds": {"max_search_calls_per_job": 1}}
    s = search_providers.acquisition_search(lambda q, n, t: [{"doi": "10.1/x"}], cfg)
    with search_providers.portal_scope() as lst:
        s("q", 5, 5.0)
    assert [p["portal"] for p in lst].count("openalex") == 1
    assert lst[0] == {"portal": "openalex", "status": "completed", "results": 1}

    def boom(q, n, t):
        raise TimeoutError("t")
    s2 = search_providers.acquisition_search(boom, cfg)
    with search_providers.portal_scope() as lst2:
        with pytest.raises(TimeoutError):
            s2("q", 5, 5.0)
    assert lst2[0]["status"] == "timeout"


def test_rank_candidates_drops_record_reason_and_defer_cap():
    cfg = A.cfg()
    cfg["bounds"]["max_candidates_resolved"] = 1
    good = [A.cand("10.9999/a.%d" % i) for i in range(2)]
    nodoi = {"title": "x", "url": "https://example.org/x"}
    drops = {}
    out = auto_acquire.rank_candidates("outbreak transmission incidence", [nodoi] + good, cfg, set(),
                                       auto_acquire._fd_cfg(), lambda c: None, drops=drops)
    assert len(out) == 1
    assert drops[id(nodoi)] == acq_config.reason(cfg, "no_doi")
    assert list(drops.values()).count(acq_config.reason(cfg, "defer_cap")) == 1


def test_inline_outcome_has_independent_candidate_records_and_portals(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    cfg = A.cfg()
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    web = A.epi_setup()[0]
    nodoi = {"title": "no id", "url": "https://example.org/x", "provider": "openalex"}
    good = A.cand("10.9999/acq.epi4", provider="openalex", year=2024)
    d = A.deps([nodoi, good], web)
    inner = d.search

    def search(q, n, t):
        search_providers.record_portal("openalex", "completed", 2)
        return inner(q, n, t)
    d.search = search
    o = inline_acquire.acquire("outbreak transmission incidence question", cfg, env=A.ENV_ON,
                               deps_factory=lambda a, e: d)
    pub = o.public()
    assert pub["portals"] == [{"portal": "openalex", "status": "completed", "results": 2}]
    by = {c["doi"]: c for c in pub["candidates"]}
    assert len(pub["candidates"]) == 2
    assert by[None]["outcome"] == "dropped" and by[None]["reason"] == acq_config.reason(cfg, "no_doi")
    assert by["10.9999/acq.epi4"]["outcome"] in ("admitted", "rejected", "reused")
    assert by["10.9999/acq.epi4"]["published"] == 2024
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def test_live_report_research_projection_is_allowlisted():
    contract = {"research": {"summary": "S" * 700, "triggers": ["stale", "bogus"], "all_portals_completed": False,
                             "portals": [{"portal": "openalex", "status": "quota", "results": 0},
                                         {"portal": "secret_svc", "status": "weird"}],
                             "candidates": [{"doi": "10.1/x", "outcome": "rejected", "reason": "R-NOPE",
                                             "url": "https://leak.example/x"}]}}
    rep = live_report.build_live_report({"contract": contract}, "q", 1.0)["research"]
    assert len(rep["outcome_text"]) <= 600
    assert rep["triggers"] == ["evidence older than 24 hours"]
    assert rep["portals"] == [{"label": "scholarly index", "status": "rate limit reached"},
                              {"label": "other", "status": "other"}]
    assert rep["candidates"][0]["identifier"] == "10.1/x" and "leak" not in str(rep) and "openalex" not in str(rep)
    assert live_report.build_live_report({}, "q", 1.0)["research"] is None
