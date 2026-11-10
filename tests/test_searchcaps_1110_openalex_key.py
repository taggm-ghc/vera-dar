"""SEARCHCAPS-1110 (4): OpenAlex API key -> cap 1000/day (half reserved for visitors), pacing <= 5 req/s; no key ->
unchanged 90/day (OD-1110-SEARCHCAPS). Offline: fake transport; the key is a test sentinel, never a real value."""
import pytest
import requests

from vera import rate_limit, request_origin
import vera.search_providers as sp
from vera.eval_config import load_eval_config
from vera.search_providers import OpenAlexProvider, ProviderUnavailable

CFG = load_eval_config().search
SENT = "testkeySENTINEL0123"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(sp, "openalex_default_transport", boom)
    monkeypatch.setattr(requests, "get", boom)
    yield
    rate_limit.set_registry_for_tests(None)


class R:
    def __init__(self):
        self.status_code, self.text, self.headers, self.url = 200, '{"results": []}', {}, ""


def _registry(env):
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env=env)))


def _provider(sent):
    return OpenAlexProvider(CFG["openalex"], transport=lambda u, params, h, t: (sent.append(params), R())[1],
                            today=lambda: "2026-11-10")


def test_without_key_cap_and_rate_unchanged(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    _registry({})
    c = rate_limit.get_registry().portal("openalex").c
    assert c["search_daily_cap"] is None and c["rate_per_s"] == 1.0
    sent = []
    p = _provider(sent)
    assert p._budget == CFG["openalex"]["daily_search_budget"] == 90
    p.call("quarks", 5, 5.0)
    assert "api_key" not in sent[0]


def test_with_key_cap_1000_and_pacing_at_most_5_per_second(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", SENT)
    _registry({"OPENALEX_API_KEY": SENT})
    c = rate_limit.get_registry().portal("openalex").c
    assert c["search_daily_cap"] == 1000
    assert c["rate_per_s"] * 1.0 + c["burst"] <= 5.0  # token bucket: at most burst + rate x 1 s in any 1 s window
    sent = []
    p = _provider(sent)
    assert p._budget == 1000
    p.call("quarks", 5, 5.0)
    assert sent[0]["api_key"] == SENT


def test_with_key_seed_share_is_half(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", SENT)
    _registry({"OPENALEX_API_KEY": SENT})
    p = _provider([])
    p._day, p._used = "2026-11-10", 499
    with request_origin.scope("seed"):
        p.call("quarks", 5, 5.0)
        with pytest.raises(ProviderUnavailable) as ei:
            p.call("quarks", 5, 5.0)
    assert SENT not in str(ei.value) and "seed share of 1000" in str(ei.value)
    p.call("quarks", 5, 5.0)  # visitor


def test_token_bucket_paces_to_five_per_second(monkeypatch):
    t = {"now": 0.0}
    cfg = rate_limit.load_config(env={"OPENALEX_API_KEY": SENT})
    reg = rate_limit.Registry(cfg, clock=lambda: t["now"], wall=lambda: 1_800_000_000.0,
                              sleep=lambda s: t.__setitem__("now", t["now"] + s), rand=lambda: 0.0)
    a = reg.portal("openalex")
    stamps = []
    for _ in range(12):
        pm = a.acquire(deadline=None)
        assert pm.ok
        stamps.append(t["now"])
        pm.record(status=200)
    for i in range(len(stamps)):
        assert sum(1 for x in stamps if stamps[i] <= x < stamps[i] + 1.0) <= 5
