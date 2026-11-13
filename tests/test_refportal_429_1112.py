"""REFPORTAL-429-1112: a server throttle (429, or 503 with Retry-After, the portal's throttle_statuses) on the
tdmrep.json read is a rate limit, not a verdict about the file: never cached as "unreadable".

Plan: work/REFPORTAL-CLOCK-1112/PLAN.md (429 amendment). Mirrors acq_discovered._passthrough. Retry-After reaches the
limiter's cooldown through the existing rate_limit.limited wrapper (no new mechanism). Offline, fake clock, SYNTHETIC
values only (World harness of test_refportal_clock_1112)."""
from __future__ import annotations

import pytest

from vera import rate_limit
from vera import reference_portal as rp
from tests.test_refportal_clock_1112 import (  # noqa: F401
    HOST, TDM_URL, PAGE_URL, World, Resp, rcfg, _reset, TDM_RULES, PAGE)


def _throttling(wd, status, retry_after=None, first_only=True):
    """The tdmrep/page server answers `status` (with Retry-After) once, then normally."""
    real = wd.server
    state = {"n": 0}

    def server(url, **kw):
        wd.sent.append((wd.clk.t, url))
        state["n"] += 1
        if state["n"] == 1 or not first_only:
            r = Resp(status, b"")
            if retry_after is not None:
                r.headers["Retry-After"] = str(retry_after)
            return r
        wd.sent.pop()
        return real(url, **kw)
    wd.server = server


@pytest.mark.parametrize("status,ra", [(429, None), (429, 20), (503, 20)])
def test_throttled_tdmrep_raises_and_is_not_cached(status, ra):
    wd = World()
    _throttling(wd, status, ra)
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    assert HOST not in rp._TDM
    assert len(wd.sent) == 1  # the request was sent once; the throttle is recorded, not retried


@pytest.mark.parametrize("status", [429, 503])
def test_next_call_after_cooldown_rereads_tdmrep(status):
    wd = World()
    _throttling(wd, status, 20)
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    wd.clk.t += 120.0  # Retry-After 20 s (+ jitter) honoured by the limiter's cooldown, then passed
    assert wd.tdm(rcfg()) is True
    assert [u for _t, u in wd.sent] == [TDM_URL, TDM_URL]
    assert rp._TDM[HOST][1] != "unreadable"


def test_retry_after_feeds_the_limiter_cooldown_and_blocks_an_early_retry():
    wd = World()
    _throttling(wd, 429, 60)
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    wd.clk.t += 5.0  # inside the 60 s Retry-After: the limiter skips, nothing more is sent, still nothing cached
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    assert len(wd.sent) == 1 and HOST not in rp._TDM


def test_fetch_page_reports_rate_limited_for_a_throttled_tdmrep():
    wd = World()
    _throttling(wd, 429, 20)
    with pytest.raises(rp.Skip) as e:
        wd.page(rcfg())
    assert e.value.code == "rate_limited"
    assert HOST not in rp._TDM


@pytest.mark.parametrize("status", [429, 503])
def test_fetch_page_page_get_throttle_is_rate_limited(status):
    wd = World()
    real = wd.server

    def server(url, **kw):
        if url == PAGE_URL:
            wd.sent.append((wd.clk.t, url))
            return Resp(status, b"")
        return real(url, **kw)
    wd.server = server
    with rate_limit.stream_wait_scope():
        with pytest.raises(rp.Skip) as e:
            wd.page(rcfg())
    assert e.value.code == "rate_limited"


def test_page_get_500_is_still_unreachable():
    wd = World()
    real = wd.server

    def server(url, **kw):
        return Resp(500, b"") if url == PAGE_URL else real(url, **kw)
    wd.server = server
    with rate_limit.stream_wait_scope():
        with pytest.raises(rp.Skip) as e:
            wd.page(rcfg())
    assert e.value.code == "unreachable"


# ---- unchanged behaviour (current behaviour, kept) ----

@pytest.mark.parametrize("status", [500, 502, 504])
def test_genuine_5xx_without_throttle_semantics_is_still_cached_unreadable(status):
    wd = World()
    wd.server = lambda u, **k: Resp(status, b"")
    assert wd.tdm(rcfg()) is False
    assert rp._TDM[HOST][1] == "unreadable"


def test_slow_tdmrep_host_is_still_cached_unreadable():
    wd = World(slow={"tdmrep": 5.5})
    assert wd.tdm(rcfg()) is False
    assert rp._TDM[HOST][1] == "unreadable"


@pytest.mark.parametrize("st", [404, 410])
def test_404_410_still_mean_no_reservation(st):
    wd = World()
    wd.server = lambda u, **k: Resp(st, b"")
    assert wd.tdm(rcfg()) is True
    assert rp._TDM[HOST][1] == []
