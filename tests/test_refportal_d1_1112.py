"""REFPORTAL-D1-1112: a limiter skip (rate_limit.RateLimited) of the tdmrep.json read is not a verdict about the file.

Plan: work/REFPORTAL-CLOCK-1112/PLAN.md (D1 amendment). Defect: reference_portal._tdmrep_ok caught RateLimited in its
bare except, cached the host as "unreadable" for _TDM_TTL_S with nothing sent, and fetch_page reported tdm_reservation.
Offline, fake clock, SYNTHETIC values only (reuses the World harness of test_refportal_clock_1112)."""
from __future__ import annotations

import pytest

from vera import rate_limit
from vera import reference_portal as rp
from tests.test_refportal_clock_1112 import (  # noqa: F401
    HOST, TDM_URL, PAGE_URL, World, Resp, rcfg, _reset)


def _skipped_world():
    """Portal rate 0.1/s with its token spent: outside a stream scope the 10 s wait exceeds max_wait_s, so the
    limiter skips the request (RateLimited) before anything is sent."""
    wd = World(rate=0.1)
    wd.use_token()
    return wd


def test_ratelimited_tdmrep_raises_sends_nothing_and_is_not_cached():
    wd = _skipped_world()
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    assert wd.sent == []
    assert HOST not in rp._TDM


def test_next_call_after_the_limiter_frees_sends_the_tdmrep_request():
    wd = _skipped_world()
    with pytest.raises(rate_limit.RateLimited):
        wd.tdm(rcfg())
    wd.clk.t += 30.0  # the token has refilled
    assert wd.tdm(rcfg()) is True
    assert [u for _t, u in wd.sent] == [TDM_URL]
    assert rp._TDM[HOST][1] != "unreadable"


def test_fetch_page_reports_the_rate_limited_code_not_tdm_reservation():
    wd = _skipped_world()
    with pytest.raises(rp.Skip) as e:
        wd.page(rcfg())
    assert e.value.code == "rate_limited" and str(e.value) == "rate_limited"
    assert wd.sent == [] and HOST not in rp._TDM


def test_fetch_page_works_once_the_limiter_frees():
    wd = _skipped_world()
    with pytest.raises(rp.Skip):
        wd.page(rcfg())
    wd.clk.t += 30.0
    with rate_limit.stream_wait_scope():  # the page slot follows the tdmrep slot: 10 s apart at 0.1/s
        assert "synthetic zorblat page" in wd.page(rcfg())
    assert [u for _t, u in wd.sent] == [TDM_URL, PAGE_URL]


def test_slow_tdmrep_host_is_still_cached_unreadable():
    wd = World(slow={"tdmrep": 5.5})
    assert wd.tdm(rcfg()) is False
    assert rp._TDM[HOST][1] == "unreadable"


def test_other_exception_in_tdmrep_is_still_cached_unreadable():
    wd = World()

    def boom(u, **k):
        raise OSError("synthetic connection failure")
    wd.server = boom
    assert wd.tdm(rcfg()) is False
    assert rp._TDM[HOST][1] == "unreadable"


@pytest.mark.parametrize("st", [404, 410])
def test_404_410_still_mean_no_reservation(st):
    wd = World()
    wd.server = lambda u, **k: Resp(st, b"")
    assert wd.tdm(rcfg()) is True
    assert rp._TDM[HOST][1] == []
