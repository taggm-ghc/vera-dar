"""LANDING-CLOCK-1112: our own pacing (host-slot wait, limiter wait) is not hop time in landing_doi's robots check.

Plan: work/REFPORTAL-CLOCK-1112/PLAN.md ("landing amendment"). Sibling of tests/test_epmc_clock_1112.py.
Offline: one fake clock; the real landing_doi, the real acq_discovered robots checker and _b_get; a fake transport that
advances the fake clock for "limiter wait" (credited through rate_limit.WAIT_TALLY exactly as rate_limit.limited does)
and for server time. SYNTHETIC host names only.
Defect: _landing_robots fixed its deadline (landing start + timeout_s) before the host-slot wait and the limiter wait,
so a paced robots.txt read ended as "deadline" (not cached by _passthrough) and the landing DOI was skipped."""
from __future__ import annotations

import contextlib

import pytest

from vera import acq_discovered as ad
from vera import rate_limit
from vera import search_providers as sp

HOST = "journal.example.org"
PAGE = f"https://{HOST}/article/1"
ROBOTS = f"https://{HOST}/robots.txt"
HTML = b'<html><head><meta name="citation_doi" content="10.5555/synthetic.1"></head><body>x</body></html>'
ROBOTS_OK = b"User-agent: *\nAllow: /\n"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Resp:
    def __init__(self, status=200, body=b"", ctype="text/html"):
        self.status_code, self._b, self.headers = status, body, {"Content-Type": ctype}

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


class World:
    def __init__(self, limiter_wait=0.0, robots_server=0.0, page_server=0.0, slot_wait=0.0):
        self.clock = Clock()
        self.limiter_wait, self.robots_server, self.page_server, self.slot_wait = \
            limiter_wait, robots_server, page_server, slot_wait
        self.log = []

    def get(self, url, **kw):
        self.log.append(url)
        box = rate_limit.WAIT_TALLY.get()
        if box is not None:  # what rate_limit.limited does: wait for the slot, tally it, then call the server
            self.clock.t += self.limiter_wait
            box[0] += self.limiter_wait
        if url == ROBOTS:
            self.clock.t += self.robots_server
            return Resp(200, ROBOTS_OK, "text/plain")
        self.clock.t += self.page_server
        return Resp(200, HTML)

    def slot(self, host, timeout):
        @contextlib.contextmanager
        def cm():
            self.clock.t += self.slot_wait  # our own per-host pacing: the slot was busy this long
            yield True
        return cm()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("_ROBOTS", "_TDMREP"):
        d = getattr(ad, name, None)
        if isinstance(d, dict):
            d.clear()
    monkeypatch.setattr(ad, "crawl_wait", lambda *a, **k: 0.0)
    monkeypatch.setattr(sp, "_crawl_gate", lambda *a, **k: True)
    monkeypatch.setattr(ad, "is_denied", lambda **k: False)
    yield


def run(w, deadline_after=None, timeout_s=3.0):
    monkey = sp._host_slot
    sp._host_slot = w.slot
    try:
        dl = None if deadline_after is None else w.clock() + deadline_after
        return sp.landing_doi(PAGE, timeout_s, 65536, get=w.get, url_check=lambda u: (True, ""),
                              deadline=dl, clock=w.clock)
    finally:
        sp._host_slot = monkey


def test_limiter_wait_on_robots_does_not_skip_landing():
    w = World(limiter_wait=2.5, robots_server=1.0)
    assert run(w, deadline_after=10) == "10.5555/synthetic.1"
    assert w.log[0] == ROBOTS and w.log[1] == PAGE  # robots before page


def test_host_slot_wait_does_not_eat_robots_budget():
    w = World(slot_wait=2.8, robots_server=0.5)
    assert run(w, deadline_after=10) == "10.5555/synthetic.1"
    assert w.log[0] == ROBOTS


def test_no_caller_deadline_limiter_wait_credited():
    w = World(limiter_wait=2.5, robots_server=1.0)
    assert run(w, deadline_after=None) == "10.5555/synthetic.1"


def test_caller_deadline_still_caps_the_credit():
    w = World(limiter_wait=2.5, robots_server=2.0)  # 4.5 s elapsed > the caller's 4 s
    assert run(w, deadline_after=4) is None
    assert PAGE not in w.log  # fail closed: no page request


def test_genuinely_slow_robots_host_still_times_out():
    w = World(robots_server=6.0)  # server time beyond the per-fetch timeout is never credited
    assert run(w, deadline_after=30) is None
    assert PAGE not in w.log


def test_no_pacing_normal_path_unchanged():
    w = World()
    assert run(w, deadline_after=10) == "10.5555/synthetic.1"
    assert w.log == [ROBOTS, PAGE]
