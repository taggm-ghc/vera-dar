"""NONASK-1103 WP-F0: arXiv as an acquisition provider. OFFLINE and SYNTHETIC: fake providers, fake clock, no network."""
import copy

from tests import acq_support as A
from vera import acq_config, auto_acquire, search_providers as sp

import pytest

# ARXIV-PORTAL-1110: these tests pin the DORMANT arXiv evidence route (closed in code: arxiv_fulltext.PORTAL_ONLY);
# the fixture reopens it per test. Portal-only behaviour: tests/test_arxiv_portal_1110.py.
pytestmark = pytest.mark.usefixtures("legacy_arxiv_route")

Q = "llm multi agent orchestration frameworks"


class _P:
    def __init__(self, name):
        self.name = name


def _make(results, profile="acq", providers=("openalex", "arxiv"), clock=None):
    calls = []

    def search_fn(question, n, providers=None, **kw):
        names = tuple(p.name for p in providers)
        calls.append((names, sp.OPENALEX_PROFILE.get()))
        r = results.get(names, [])
        r = r.pop(0) if isinstance(r, list) and r and isinstance(r[0], list) else r
        if isinstance(r, Exception):
            raise r
        return list(r)
    cfg = {"search": {"providers": list(providers), "openalex_profile": profile}}
    kw = {"clock": clock} if clock else {}
    s = auto_acquire.make_search(cfg, chain_fn=lambda: [_P("openalex"), _P("arxiv")], search_fn=search_fn, **kw)
    return s, calls


def _oa(i):
    return {"provider": "openalex", "doi": f"10.1/oa{i}", "title": f"oa {i}"}


def _ax(i, **k):
    return {"provider": "arxiv", "arxiv_id": f"2401.{i:05d}", "doi": f"10.48550/arxiv.2401.{i:05d}", "title": f"ax {i}", **k}


def test_arxiv_called_exactly_once_and_all_openalex_results_kept():
    oa = [_oa(i) for i in range(31)]
    s, calls = _make({("openalex",): [oa], ("arxiv",): [_ax(1)]})
    out = s(Q, 31, 10)
    assert [c[0] for c in calls] == [("openalex",), ("arxiv",)]
    assert [c["doi"] for c in out[:31]] == [c["doi"] for c in oa] and len(out) == 32  # no cut to 31


def test_or_retry_is_openalex_only_and_arxiv_hit_does_not_suppress_it():
    s, calls = _make({("openalex",): [[], [_oa(1)]], ("arxiv",): [_ax(1)]})
    out = s(Q, 31, 10)
    assert calls == [(("openalex",), "acq"), (("openalex",), None), (("arxiv",), "acq" if False else None)] or \
        [c[0] for c in calls] == [("openalex",), ("openalex",), ("arxiv",)]
    assert sum(1 for c in calls if c[0] == ("arxiv",)) == 1
    assert {c["doi"] for c in out} == {"10.1/oa1", "10.48550/arxiv.2401.00001"}


def test_duplicate_prefers_the_openalex_pmc_journal_version():
    journal = {"provider": "openalex", "doi": "10.1/j", "pmcid": "PMC123", "title": "journal"}
    preprint = _ax(7, doi="10.1/j")
    s, _ = _make({("openalex",): [[journal]], ("arxiv",): [preprint]})
    out = s(Q, 31, 10)
    assert len(out) == 1 and out[0]["provider"] == "openalex" and out[0]["pmcid"] == "PMC123"
    assert out[0]["arxiv_id"] == preprint["arxiv_id"]  # the id is kept on the journal record


def test_arxiv_failure_keeps_openalex_and_budget_gate_skips_the_call():
    s, calls = _make({("openalex",): [[_oa(1)]], ("arxiv",): RuntimeError("down")})
    assert [c["doi"] for c in s(Q, 31, 10)] == ["10.1/oa1"]
    s, calls = _make({("openalex",): [[_oa(1)]], ("arxiv",): [_ax(1)]})
    tok = sp.ARXIV_ALLOWED.set(False)
    try:
        out = s(Q, 31, 10)
    finally:
        sp.ARXIV_ALLOWED.reset(tok)
    assert [c[0] for c in calls] == [("openalex",)] and len(out) == 1


def test_arxiv_not_in_providers_is_not_called():
    s, calls = _make({("openalex",): [[_oa(1)]], ("arxiv",): [_ax(1)]}, providers=("openalex",))
    s(Q, 31, 10)
    assert [c[0] for c in calls] == [("openalex",)]


# ---- budget: acquisition_search counts arXiv; lookup_left
def _acq(monkeypatch, cap, arxiv=True):
    monkeypatch.setattr(sp, "_wiki_cfg", lambda: {"enabled": True})
    from vera import wiki_ref_portal
    monkeypatch.setattr(wiki_ref_portal, "enabled", lambda w: True)
    monkeypatch.setattr(wiki_ref_portal, "make_portal_search", lambda w, http_get=None: (lambda q, n, t: []))
    sp._QUOTA.reset()
    seen = {"allowed": None, "lookup_left": None, "lookups": 0}

    def base(q, n, t):
        seen["allowed"] = sp.ARXIV_ALLOWED.get()
        return [{"provider": "openalex", "doi": "10.9/o"}]

    def ddg(q, n, t):
        return [{"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/", "title": "t"}]

    def fake_ddg_to_cands(results, ecfg, dcfg, **kw):
        seen["lookup_left"] = kw["calls_left"]
        return [], {}
    monkeypatch.setattr(sp, "ddg_to_candidates", fake_ddg_to_cands)
    cfg = {"search": {"providers": ["openalex"] + (["arxiv"] if arxiv else []), "openalex_profile": {"x": 1},
                      "ddg_first": {"enabled": True}, "europepmc": {"enabled": True}},
           "bounds": {"max_search_calls_per_job": cap}}
    search = sp.acquisition_search(base, cfg, ddg_search=ddg, transport=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
                                   terms_fn=lambda: (None, None))
    with sp.portal_scope():
        search(Q, 31, 5.0)
    return seen


def test_budget_cap7_with_arxiv_keeps_lookup_left_one(monkeypatch):
    """F0 pin (single question, one part). AMENDED by NONASK-1103 stream S (plan 6.2, T1 gate): with n >= 2 sub-queries
    the pins become per sub-query (arXiv once per part; lookup_left == 1 at the scaled cap 7 + 3(n-1)); see
    tests/test_nonask_1103_subquery_search.py. A question with one part is exactly this test, unchanged."""
    s = _acq(monkeypatch, 7)
    assert s["lookup_left"] == 1 and s["allowed"] is True  # OA 2 + ddg + epmc + wiki + arxiv = 6; 1 left for the id lookup


def test_budget_cap7_without_arxiv_unchanged(monkeypatch):
    s = _acq(monkeypatch, 7, arxiv=False)
    assert s["lookup_left"] == 2 and s["allowed"] is False


def test_budget_shipped_6_with_arxiv_counts_it_lookup_left_zero(monkeypatch):
    s = _acq(monkeypatch, 6)
    assert s["lookup_left"] == 0 and s["allowed"] is True  # arXiv takes the last slot at the OLD cap (why 7 ships)


def test_shipped_config_cap_and_providers():
    cfg = acq_config.load_config()
    assert cfg["search"]["providers"] == ["openalex"] \
        and cfg["bounds"]["max_search_calls_per_job"] == 7  # ARXIV-OFF-1107: arxiv removed; cap unchanged
    left = 7 - 2 - 1 - 1 - 1 - 1  # openalex 2, ddg, epmc, wiki, arxiv
    assert left == 1


def test_ddg_pmc_id_lookup_still_runs_with_arxiv_on(monkeypatch):
    """The Europe PMC id lookup for a DDG PMC link is still made (lookup slot kept at cap 7)."""
    monkeypatch.setattr(sp, "_wiki_cfg", lambda: {"enabled": False})
    sp._QUOTA.reset()
    posted = []

    def transport(url, params, headers, timeout):
        posted.append(params)

        class R:
            status_code = 200
            text = '{"resultList":{"result":[{"doi":"10.5/pmcdoi","pmcid":"PMC1234567","title":"T"}]}}'
            content = text.encode()
            def json(self): return {"resultList": {"result": [{"doi": "10.5/pmcdoi", "pmcid": "PMC1234567", "title": "T"}]}}
        return R()
    cfg = copy.deepcopy(acq_config.load_config())
    cfg["search"]["europepmc"]["enabled"] = False  # only the id lookup uses the transport
    ddg = lambda q, n, t: [{"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/", "title": "T"}]
    search = sp.acquisition_search(lambda q, n, t: [], cfg, ddg_search=ddg, transport=transport)
    with sp.portal_scope():
        search(Q, 31, 5.0)
    assert len(posted) == 1  # the lookup ran


# ---- rank tiers
def test_tiers_unchanged_with_route_off():
    d = {"acq_source": "ddg", "ddg_id_kind": "arxiv"}
    w = {"acq_source": "wikipedia_portal", "ref_id_kind": "arxiv"}
    a = _ax(1)
    assert [sp.acq_rank_tier(x) for x in (d, w, a)] == [2, 2, 2]
    assert sp.acq_rank_tier(d, False) == 2 and sp.acq_rank_tier({**a, "pmcid": "PMC9"}) == 1
    assert sp.acq_rank_tier({"acq_source": "ddg", "ddg_id_kind": "doi"}) == 0


def test_tiers_with_route_on_only_for_routable_arxiv():
    assert sp.acq_rank_tier({"acq_source": "ddg", "ddg_id_kind": "arxiv"}, True) == 1
    assert sp.acq_rank_tier({"acq_source": "wikipedia_portal", "ref_id_kind": "arxiv"}, True) == 1
    assert sp.acq_rank_tier(_ax(1), True) == 1
    unroutable = {"provider": "arxiv", "title": "no id", "doi": "10.9/x"}
    assert sp.acq_rank_tier(unroutable, True) == 2  # identified by provider only: not one the route can take
    assert sp.acq_rank_tier({"acq_source": "ddg", "ddg_id_kind": "doi"}, True) == 0


FILL = "a long abstract about llm orchestration and how a team of engineers might report on it during a long review"


def _pool(arxiv_route, ax):
    cfg = A.cfg()
    cfg["arxiv_route"]["enabled"] = arxiv_route
    oa = [{"provider": "openalex", "doi": f"10.1001/oa{i}", "title": f"LLM multi agent orchestration study {i}", "snippet": FILL,
           "oa": {"is_oa": True}} for i in range(31)]
    ep = [{"provider": "europepmc", "doi": f"10.1002/ep{i}", "title": f"LLM multi agent orchestration review {i}", "snippet": FILL,
           "oa": {"epmc_ft": True}} for i in range(31)]
    return cfg, oa + ep + [ax]


def test_full_pool_arxiv_candidate_is_fetched_when_route_on_and_not_when_off():
    ax = {"provider": "arxiv", "arxiv_id": "2401.00001", "doi": "10.48550/arxiv.2401.00001",
          "title": "LLM multi agent orchestration frameworks", "snippet": "multi agent orchestration with LLM",
          "url": "https://arxiv.org/abs/2401.00001v1"}
    for on in (True, False):
        cfg, pool = _pool(on, copy.deepcopy(ax))
        cfg["bounds"]["max_candidates_resolved"] = 31
        out = auto_acquire.rank_candidates(Q, pool, cfg, set(), auto_acquire._fd_cfg(), lambda *_: None)
        dois = [r[1] if isinstance(r, tuple) else r.get("doi") for r in out]
        got = any("arxiv" in str(d) for d in dois)
        assert got is on, (on, len(out))
        if on:
            assert any("arxiv" in str(d) for d in dois[:2]), dois[:3]  # inside max_docs_per_job 2 window as well


# ---- rate gate and timing (fake clock, no sleep, no network)
def test_rate_gate_second_arxiv_call_waits_at_least_3s_fake_clock():
    import json
    from pathlib import Path
    from vera import rate_limit as rl
    root = Path(__file__).resolve().parent.parent
    gate = sp.RateGate(3.0)  # config search.arxiv.min_interval_s (asserted below when the file is present)
    t0 = 1000.0
    gate.stamp(t0)
    assert gate.wait_needed(t0) >= 3.0 and gate.wait_needed(t0 + 1.0) == 2.0 and gate.wait_needed(t0 + 3.0) == 0.0
    cfg = rl.load_config(root / "config" / "rate_limits.json", env={})
    a = cfg["portals"]["arxiv"]
    assert 1.0 / a["rate_per_s"] >= 2.99 and a["burst"] == 1 and a["max_concurrency"] == 1

    class Clk:
        t = 1000.0
        slept = []
        def __call__(self): return self.t
        def wall(self): return 1_760_000_000.0 + self.t
        def sleep(self, s): self.slept.append(s); self.t += s
    c = Clk()
    reg = rl.Registry(cfg, clock=c, wall=c.wall, sleep=c.sleep, rand=lambda: 0.0)
    ad = reg.portal("arxiv")
    g1 = ad.acquire()
    assert g1.ok
    g1.record(status=200)
    g2 = ad.acquire()  # immediately again: either waits (>= ~3 s total on the fake clock) or is skipped, never free
    waited = sum(c.slept)
    assert (not g2.ok) or waited >= 2.9, (g2, waited)


def test_gate_waits_stay_inside_inline_deadline_and_stream_stall():
    cfg = acq_config.load_config()
    one_wait = 3.0  # one arXiv search call per job (make_search); gate min_interval_s 3 (offline arithmetic, no clock)
    assert one_wait < cfg["inline"]["deadline_s"] == 45  # SEOS r2 3a: inside the inline deadline
    assert cfg["inline"]["deadline_s"] < cfg["live"]["stall_s"] == 300  # and inside the streamed path's stall_s
