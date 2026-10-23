"""ROUTE-R3 (P0): DDG-first multi-portal discovery. OFFLINE unit tests: chain order, neutral RRF tie-break, soft
per-portal share, identifier preservation, arXiv rank tier, DDG request cache + quota status, acquisition portal
records in start order with not_attempted."""
import pytest
import requests

import vera.search_and_fetch as sf
import vera.search_providers as sp
from tests.config_support import real_run
from tests.test_item71_ddg import QT, R, link, page
from tests.test_m2_search import FakeProvider, ok
from vera.eval_config import load_eval_config
from vera.search_providers import DuckDuckGoLiteProvider, ProviderUnavailable, build_chain

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
Q = "What does published research show about AI coding assistants and developer productivity?"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    for name in ("arxiv_default_transport", "openalex_default_transport", "ddg_default_transport"):
        monkeypatch.setattr(sp, name, boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(sf, "_CHAIN_CACHE", {})


# ------------------------------------------------------------------------------------------------ 1 chain order
def test_reading_list_chain_is_ddg_first_arxiv_last():
    s = real_run()["search"]
    names = [p.name for p in build_chain(s, repo_root=ROOT)]
    assert names[0] == "duckduckgo_lite" and names[-1] == "arxiv" and "openalex" in names, names
    assert names.index("openalex") < names.index("arxiv")
    # grounding / off-topic lists only FILTER the chain: their call order is the chain order
    sub = [p.name for p in build_chain(s, repo_root=ROOT) if p.name in {"arxiv", "openalex", "duckduckgo_lite"}]
    assert sub == ["duckduckgo_lite", "openalex", "arxiv"]


def test_soft_share_and_memory_cache_are_config_and_marked_uncalibrated():
    s = load_eval_config().search
    assert isinstance(s["soft_max_share"]["_default"], int) and "UNCALIBRATED" in s["soft_max_share_note"]
    assert s["duckduckgo_lite"]["memory_cache_ttl_s"] > 0 and "UNCALIBRATED" in s["duckduckgo_lite"]["memory_cache_note"]
    assert s["duckduckgo_lite"]["daily_query_cap"] == 8  # unchanged


# ------------------------------------------------------------------------------------------------ 2 RRF ties
def _it(u, **k):
    return {"url": u, "title": u, "snippet": "s", **k}


def test_rrf_tie_break_is_independent_of_provider_position():
    per = {"arxiv": [_it("https://arxiv.org/abs/2401.00001"), _it("https://arxiv.org/abs/2401.00002")],
           "openalex": [_it("https://doi.org/10.9/a")], "duckduckgo_lite": [_it("https://news.example.com/x")]}
    rev = dict(reversed(list(per.items())))
    for n in (1, 2, 3, 4):
        a = [x["url"] for x in sf.rrf_merge(per, 60, n)]
        b = [x["url"] for x in sf.rrf_merge(rev, 60, n)]
        assert a == b, (n, a, b)  # chain position never decides a tie


def test_rrf_tie_round_robins_across_portals():
    per = {"arxiv": [_it(f"https://arxiv.org/abs/2401.0000{i}") for i in range(1, 4)],
           "openalex": [_it("https://doi.org/10.9/a"), _it("https://doi.org/10.9/b")]}
    out = sf.rrf_merge(per, 60, 4)
    provs = [x["found_by_providers"][0] for x in out]
    assert sorted(provs[:2]) == ["arxiv", "openalex"] and sorted(provs[2:4]) == ["arxiv", "openalex"], provs


def test_rrf_merge_keeps_every_members_identifiers():
    per = {"duckduckgo_lite": [_it("https://doi.org/10.9/x")],
           "openalex": [_it("https://doi.org/10.9/x", doi="10.9/x", openalex_id="https://openalex.org/W1", pmcid="PMC1")]}
    (rec,) = sf.rrf_merge(per, 60, 5)
    assert rec["doi"] == "10.9/x" and rec["openalex_id"].endswith("W1") and rec["pmcid"] == "PMC1"
    assert set(rec["found_by_providers"]) == {"duckduckgo_lite", "openalex"}


# ------------------------------------------------------------------------------------------------ soft share
def test_soft_share_backfills_and_never_drops():
    items = [{"url": f"https://arxiv.org/abs/{i}", "found_by_providers": ["arxiv"]} for i in range(5)] + \
            [{"url": "https://doi.org/10.9/a", "found_by_providers": ["openalex"]}]
    out = sf.apply_soft_share(items, {"_default": 3})
    assert [x["found_by_providers"][0] for x in out[:4]] == ["arxiv"] * 3 + ["openalex"] and len(out) == 6
    only = items[:5]
    assert sf.apply_soft_share(only, {"_default": 3}) == only  # no other eligible result: nothing withheld


def test_fan_out_diversity_under_cap_and_portal_records_in_call_order():
    ax = FakeProvider("arxiv", [ok(*[f"https://arxiv.org/abs/2401.0000{i}" for i in range(1, 6)])])
    oa = FakeProvider("openalex", [ok("https://doi.org/10.9/a", "https://doi.org/10.9/b")])
    ddg = FakeProvider("duckduckgo_lite", [ProviderUnavailable("duckduckgo_lite daily query cap (quota) of 8 spent")])
    res = sf.search_question("q", 5, providers=[ddg, oa, ax], sleep=lambda s: None, mode="fan_out", max_share={},
                             soft_max_share={"_default": 3})
    provs = [x["provider"] for x in res]
    assert provs.count("openalex") == 2 and provs.count("arxiv") == 3, provs
    assert oa.calls and ax.calls  # DDG-first is not DDG-only
    assert [(p["portal"], p["status"], p["n_results"]) for p in res.meta["portals"]] == [
        ("duckduckgo_lite", "quota", 0), ("openalex", "completed", 2), ("arxiv", "completed", 5)]


def test_fan_out_completed_zero_timeout_and_error_are_distinct():
    a = FakeProvider("openalex", [ok()])
    b = FakeProvider("arxiv", [requests.Timeout("timed out")])
    c = FakeProvider("duckduckgo_lite", [ProviderUnavailable("service down")])
    res = sf.search_question("q", 5, providers=[c, a, b], sleep=lambda s: None, mode="fan_out", max_retries=1)
    assert [(p["portal"], p["status"], p["n_results"]) for p in res.meta["portals"]] == [
        ("duckduckgo_lite", "error", 0), ("openalex", "completed", 0), ("arxiv", "timeout", 0)]


# ------------------------------------------------------------------------------------------------ 3 rank tier
def test_discovered_and_listed_arxiv_rank_behind_routable_candidates():
    disc_ax = {"doi": "10.1145/1234", "url": "https://arxiv.org/abs/2401.00001", "acq_source": "discovered",
               "discovery": {"service": "arxiv", "record_id": "arXiv:2401.00001"}}
    disc_oa = {"doi": "10.9/route", "url": "https://doi.org/10.9/route", "acq_source": "discovered",
               "discovery": {"service": "openalex", "record_id": "https://openalex.org/W1"}}
    listed_ax = {"provider": "arxiv", "doi": "10.48550/arxiv.2401.00002", "url": "https://arxiv.org/abs/2401.00002"}
    oa = {"provider": "openalex", "doi": "10.9/oa", "url": "https://doi.org/10.9/oa"}
    assert sp.acq_rank_tier(disc_ax) == 2 and sp.acq_rank_tier(listed_ax) == 2
    assert sp.acq_rank_tier(disc_oa) == 1 and sp.acq_rank_tier(oa) == 1
    assert sp.acq_rank_tier({**disc_ax, "pmcid": "PMC123"}) == 1  # routable via Europe PMC
    assert sp.acq_rank_tier({"acq_source": "ddg", "ddg_id_kind": "doi", "doi": "10.9/d"}) == 0  # DDG still first
    ranked = sorted([listed_ax, disc_ax, oa, disc_oa], key=sp.acq_rank_tier)
    assert ranked[:2] == [oa, disc_oa]


# ------------------------------------------------------------------------------------------------ ids + portals in merge
def test_merge_sources_keeps_ids_and_portal_provenance():
    ddg = {"url": "https://doi.org/10.9/x", "doi": "10.9/x", "provider": "ddg", "acq_source": "ddg"}
    oa = {"url": "https://doi.org/10.9/x", "doi": "10.9/X", "provider": "openalex", "openalex_id": "https://openalex.org/W9",
          "oa": {"pmcid": "PMC9"}}
    ep = {"url": "https://europepmc.org/x", "doi": "10.9/x", "pmcid": "PMC9", "provider": "europepmc"}
    (m,) = sp.merge_sources([ddg], [oa], [ep])
    assert m["openalex_id"].endswith("W9") and m["pmcid"] == "PMC9" and m["portals"] == ["ddg", "openalex", "epmc"]
    ax = {"provider": "arxiv", "url": "https://arxiv.org/abs/2401.00003", "doi": "10.48550/arxiv.2401.00003"}
    assert sp.acq_portals(ax) == ["arxiv"]


# ------------------------------------------------------------------------------------------------ DDG budget
def test_ddg_same_question_twice_spends_the_daily_cap_once():
    ddg_cfg = load_eval_config().search["duckduckgo_lite"]
    html = page(link("https://doi.org/10.9/abc", "A study"))
    calls = []

    def tx(url, data, headers, timeout):
        calls.append(data["q"])
        return R(text=html)
    p = DuckDuckGoLiteProvider({**ddg_cfg, "daily_query_cap": 2}, transport=tx, query_terms=QT, sleep=lambda s: None)
    first = p.parse(p.call(Q, 10, 5.0), 10)       # acquisition
    assert p.gate_exempt(Q)
    second = p.parse(p.call(Q, 10, 5.0), 10)      # reading list, same request
    assert first == second and len(calls) == len(set(calls)) and p._used == len(calls) <= 2
    off = DuckDuckGoLiteProvider({**ddg_cfg, "daily_query_cap": 2, "memory_cache_ttl_s": 0}, transport=tx,
                                 query_terms=QT, sleep=lambda s: None)
    off.call(Q, 10, 5.0)
    with pytest.raises(ProviderUnavailable, match="quota"):
        off.call(Q, 10, 5.0)  # exhaustion names the quota (recorded as status quota)


# ------------------------------------------------------------------------------------------------ acquisition records
def test_malformed_pmcid_is_omitted_from_public_output():
    from vera import grounded_ask as ga
    good = {"n": 1, "title": "T", "url": "https://example.org/a", "pmcid": "pmc123456", "provider": "openalex"}
    bad = {**good, "pmcid": "not-a-pmcid"}
    assert ga.public_candidates([good])[0]["pmcid"] == "PMC123456" and ga.public_sources([good])[0]["pmcid"] == "PMC123456"
    assert "pmcid" not in ga.public_candidates([bad])[0] and "pmcid" not in ga.public_sources([bad])[0]
    assert ga.provenance({"url": "https://example.org/a", "pmcid": "not-a-pmcid"})["identifier"] == "https://example.org/a"
    assert ga.provenance({"url": "https://example.org/a", "pmcid": "PMC9"})["identifier"] == "pmcid:PMC9"


def _acq_cfg(cap):
    return {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}},
            "bounds": {"max_search_calls_per_job": cap}}


def _run_acq(monkeypatch, cap, ddg):
    monkeypatch.setattr(sp, "_wiki_cfg", lambda: {"enabled": False})
    sp._QUOTA.reset()
    log = []

    def base(q, n, t):
        log.append("openalex")
        return [{"provider": "openalex", "doi": "10.9/o", "url": "https://doi.org/10.9/o"}]

    def ddg_fn(q, n, t):
        log.append("ddg")
        if isinstance(ddg, Exception):
            raise ddg
        return ddg
    search = sp.acquisition_search(base, _acq_cfg(cap), ddg_search=ddg_fn)
    with sp.portal_scope() as recs:
        out = search(Q, 5, 5.0)
    return log, recs, out


def test_acquisition_ddg_starts_first_and_records_follow_call_order(monkeypatch):
    for _ in range(5):
        log, recs, _ = _run_acq(monkeypatch, 6, [])
        assert log == ["ddg", "openalex"], log
        assert [(r["portal"], r["status"]) for r in recs] == [("ddg", "completed"), ("openalex", "completed")]
        assert recs[1]["results"] == 1


def test_acquisition_ddg_failure_continues_to_secondary_portals(monkeypatch):
    log, recs, out = _run_acq(monkeypatch, 6, RuntimeError("ddg down"))
    assert log == ["ddg", "openalex"] and out and out[0]["doi"] == "10.9/o"
    assert [(r["portal"], r["status"]) for r in recs] == [("ddg", "failed"), ("openalex", "completed")]


def test_acquisition_portal_without_call_slot_is_not_attempted(monkeypatch):
    log, recs, _ = _run_acq(monkeypatch, 1, [])
    assert log == ["openalex"]
    assert [(r["portal"], r["status"]) for r in recs] == [("openalex", "completed"), ("ddg", "not_attempted")]
    from vera import research_record as rr
    rec = rr.summarize(triggers=["insufficient"], inline={"status": "ok", "portals": recs, "candidates": []},
                       used_new=False, older_available=False, fr_before=None)
    assert [p["status"] for p in rec["portals"]] == ["completed", "not_attempted"] and rec["complete"] is False
    assert all("n_results" in p for p in rec["portals"])
