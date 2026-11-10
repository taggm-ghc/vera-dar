"""SEARCHCAPS-1110 (1) per-service visitor reserve (OD-1110-SEARCHCAPS). Offline: fake transports, fake clocks."""
import json

import pytest
import requests

from vera import rate_limit, request_origin
import vera.search_providers as sp
from vera.eval_config import load_eval_config
from vera.search_providers import DuckDuckGoLiteProvider, OpenAlexProvider, ProviderUnavailable

CFG = load_eval_config().search


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    for name in ("openalex_default_transport", "ddg_default_transport"):
        monkeypatch.setattr(sp, name, boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={})))
    yield
    rate_limit.set_registry_for_tests(None)


class R:
    def __init__(self, status=200, text='{"results": []}'):
        self.status_code, self.text, self.headers, self.url = status, text, {}, ""


def _oa(budget):
    sent = []
    p = OpenAlexProvider({**CFG["openalex"], "daily_search_budget": budget},
                         transport=lambda u, params, h, t: (sent.append(1), R())[1], today=lambda: "2026-11-10")
    return p, sent


def test_shipped_config_policy():
    raw = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))["portals"]
    assert raw["duckduckgo"]["origin"] == {"seed_allowed": False, "visitor_reserve_frac": 1.0}
    assert raw["openalex"]["origin"] == {"seed_allowed": True, "visitor_reserve_frac": 0.5}
    assert request_origin.policy("duckduckgo")["seed_allowed"] is False
    assert request_origin.policy("openalex")["visitor_reserve_frac"] == 0.5


def test_seed_cap_math():
    assert request_origin.seed_cap(90, 0.5) == 45
    assert request_origin.seed_cap(1000, 0.5) == 500
    assert request_origin.seed_cap(8, 1.0) == 0
    assert request_origin.seed_cap(7, 0.5) == 3


def test_openalex_seed_stops_at_share_visitor_still_served():
    p, sent = _oa(4)
    with request_origin.scope("seed"):
        p.call("quarks", 5, 5.0)
        p.call("quarks", 5, 5.0)
        with pytest.raises(ProviderUnavailable) as ei:
            p.call("quarks", 5, 5.0)
    assert "seed share" in str(ei.value) and sp._own_decision(ei.value) == "own_cap"
    assert len(sent) == 2
    p.call("quarks", 5, 5.0)  # the visitor reserve is still there
    p.call("quarks", 5, 5.0)
    assert len(sent) == 4
    with pytest.raises(ProviderUnavailable, match="daily search budget of 4 spent"):
        p.call("quarks", 5, 5.0)


def test_openalex_seed_share_counts_visitor_spend():
    p, sent = _oa(4)
    p.call("quarks", 5, 5.0)
    p.call("quarks", 5, 5.0)
    with request_origin.scope("seed"), pytest.raises(ProviderUnavailable):
        p.call("quarks", 5, 5.0)
    assert len(sent) == 2


def _ddg():
    sent = []
    p = DuckDuckGoLiteProvider(CFG["duckduckgo_lite"], transport=lambda *a: (sent.append(1), R(text=""))[1],
                               today=lambda: "2026-11-10", sleep=lambda s: None, robots_check=lambda u, ua: True)
    return p, sent


def test_ddg_never_used_by_seed():
    p, sent = _ddg()
    with request_origin.scope("seed"), pytest.raises(ProviderUnavailable) as ei:
        p.call("coral reef bleaching", 5, 5.0)
    assert sent == [] and sp._own_decision(ei.value) == "own_cap"
    p.call("coral reef bleaching", 5, 5.0)  # a visitor question still uses DuckDuckGo
    assert sent


def test_ddg_spend_refuses_seed_even_when_called_directly():
    p, _ = _ddg()
    with request_origin.scope("seed"), pytest.raises(ProviderUnavailable):
        p._spend()
    p._spend()


def test_adapter_seed_share_and_exclusion():
    cfg = rate_limit.load_config(env={})
    reg = rate_limit.Registry(cfg, clock=lambda: 0.0, wall=lambda: 1_800_000_000.0, sleep=lambda s: None,
                              rand=lambda: 0.0)
    ddg = reg.portal("duckduckgo")
    with request_origin.scope("seed"):
        sk = ddg.acquire(deadline=None)
    assert not sk.ok and sk.reason == "seed_excluded" and ddg.n["requests"] == 0
    assert sp._own_decision(rate_limit.RateLimited("duckduckgo", sk.reason)) == "seed_excluded"
    pm = ddg.acquire(deadline=None)
    assert pm.ok
    pm.record(status=200)


def test_adapter_seed_share_of_daily_cap():
    raw = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))
    raw["portals"]["openalex"] = {**raw["portals"]["openalex"], "daily_cap": 4, "burst": 10, "rate_per_s": 100.0,
                                  "max_concurrency": 4}
    reg = rate_limit.Registry(rate_limit.parse_config(raw, env={}), clock=lambda: 0.0, wall=lambda: 1_800_000_000.0,
                              sleep=lambda s: None, rand=lambda: 0.0)
    a = reg.portal("openalex")
    with request_origin.scope("seed"):
        for _ in range(2):
            a.acquire(deadline=None).record(status=200)
        sk = a.acquire(deadline=None)
    assert not sk.ok and sk.reason == "daily_cap"
    a.acquire(deadline=None).record(status=200)  # visitor: the reserved half is still there
    a.acquire(deadline=None).record(status=200)
    assert a.acquire(deadline=None).reason == "daily_cap"


def test_bad_origin_config_refused():
    raw = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))
    raw["portals"]["openalex"] = {**raw["portals"]["openalex"], "origin": {"visitor_reserve_frac": 1.5}}
    with pytest.raises(rate_limit.RateLimitConfigError):
        rate_limit.parse_config(raw, env={})


def test_unreadable_policy_fails_closed_for_seeds_only(monkeypatch):
    monkeypatch.setattr(request_origin, "_portal_cfg", lambda name: None)
    p, sent = _oa(4)
    with request_origin.scope("seed"), pytest.raises(ProviderUnavailable):
        p.call("quarks", 5, 5.0)
    p.call("quarks", 5, 5.0)
    assert len(sent) == 1


def test_scope_rejects_unknown_origin():
    with pytest.raises(ValueError):
        with request_origin.scope("admin"):
            pass
    assert request_origin.current() == "visitor"


def test_acquisition_search_budget_drops_ddg_for_seed():
    calls = []

    def ddg_search(q, n, t):
        calls.append("ddg")
        return []
    cfg = {"search": {"ddg_first": {"enabled": True}, "europepmc": {"enabled": False}, "providers": []},
           "bounds": {"max_search_calls_per_job": 7}}
    s = sp.acquisition_search(lambda q, n, t: [], cfg, ddg_search=ddg_search)
    with request_origin.scope("seed"):
        s("coral reef bleaching", 5, 5.0)
    assert calls == []
    s("coral reef bleaching", 5, 5.0)
    assert calls == ["ddg"]
