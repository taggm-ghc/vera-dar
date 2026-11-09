"""RETRIEVAL-1109 (NOANSWER-ROOT-1109 plan A2, A3; owner H-T, H-4): streamed-path limiter waits instead of skipping,
per-request limiter accounting, the keep-searching deadline extension, and variant-round reporting-lane passes.
Offline: fake clocks and sleeps, stub dependencies; no network, no database, no env files."""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
from types import SimpleNamespace

import pytest

from vera import acq_discovered, ask_service, inline_acquire as ia, lane_evidence, rate_limit, reporting_lane, stream_ctx

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = json.loads((ROOT / "config/rate_limits.json").read_text())
EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{n}/fullTextXML"


class FakeTime:
    def __init__(self, t=1000.0):
        self.t = float(t)
        self.sleeps: list = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(float(s))
        self.t += float(s)


def registry(ft: FakeTime, raw=None) -> rate_limit.Registry:
    return rate_limit.Registry(rate_limit.parse_config(copy.deepcopy(raw or RAW), env={}), clock=ft.clock,
                               wall=lambda: 1.7e9, sleep=ft.sleep, rand=lambda: 0.0)


# ------------------------------------------------------------------ A2 / H-T: config and the switch
def test_shipped_stream_cap_is_30_and_non_stream_cap_unchanged():
    cfg = rate_limit.parse_config(copy.deepcopy(RAW), env={})
    assert RAW["defaults"]["max_wait_stream_s"] == 30.0
    assert cfg["portals"]["europepmc"]["max_wait_stream_s"] == 30.0
    assert cfg["portals"]["europepmc"]["max_wait_s"] == 2.0  # non-streamed path unchanged


def test_absent_key_means_old_behaviour():
    raw = copy.deepcopy(RAW)
    del raw["defaults"]["max_wait_stream_s"]
    cfg = rate_limit.parse_config(raw, env={})
    assert all(c["max_wait_stream_s"] == c["max_wait_s"] for c in cfg["portals"].values())


@pytest.mark.parametrize("stream,expect", [(False, "pacing"), (True, "permit")])
def test_second_epmc_call_skips_off_stream_and_waits_on_stream(stream, expect):
    ft = FakeTime()
    ad = registry(ft).portal("europepmc")  # 0.1/s, burst 1
    with rate_limit.stream_wait_scope(stream):
        first = ad.acquire()
        first.record(status=200)
        second = ad.acquire()
    assert isinstance(first, rate_limit.Permit)
    if expect == "pacing":
        assert isinstance(second, rate_limit.Skip) and second.reason == "pacing"
        assert ft.sleeps == []
    else:
        assert isinstance(second, rate_limit.Permit)
        assert ft.sleeps == [pytest.approx(10.0)]  # exactly the documented spacing, never less


def test_stream_cap_switch_off_restores_skip():
    raw = copy.deepcopy(RAW)
    raw["defaults"]["max_wait_stream_s"] = raw["defaults"]["max_wait_s"]  # the one-line switch
    ft = FakeTime()
    ad = registry(ft, raw).portal("europepmc")
    with rate_limit.stream_wait_scope(True):
        ad.acquire().record(status=200)
        s = ad.acquire()
    assert isinstance(s, rate_limit.Skip) and s.reason == "pacing"


def test_stream_wait_never_beyond_cap_or_deadline():
    ft = FakeTime()
    ad = registry(ft).portal("openaire")  # 1 per 60 s: longer than the 30 s cap
    with rate_limit.stream_wait_scope(True):
        ad.acquire().record(status=200)
        s = ad.acquire()
    assert isinstance(s, rate_limit.Skip) and s.reason == "pacing" and ft.sleeps == []
    ft2 = FakeTime()
    ad2 = registry(ft2).portal("europepmc")
    with rate_limit.stream_wait_scope(True), rate_limit.deadline_scope(ft2.t + 5.0):
        ad2.acquire(deadline=ft2.t + 5.0).record(status=200)
        s2 = ad2.acquire(deadline=ft2.t + 5.0)
    assert isinstance(s2, rate_limit.Skip) and s2.reason == "deadline" and ft2.sleeps == []


@pytest.mark.parametrize("state", ["cooldown", "daily_cap"])
def test_stream_scope_never_waits_out_cooldown_or_daily_cap(state):
    ft = FakeTime()
    ad = registry(ft).portal("duckduckgo")
    if state == "cooldown":
        ad.acquire().record(status=429)
    else:
        ad.day_used = ad.c["daily_cap"]
        ad._roll_day = lambda: None
    with rate_limit.stream_wait_scope(True):
        s = ad.acquire()
    assert isinstance(s, rate_limit.Skip) and s.reason == state and ft.sleeps == []


# ------------------------------------------------------------------ hadrons-shaped: Europe PMC 0.1/s, Crawl-delay 10
def _read_texts(monkeypatch, stream: bool, n_texts=3, cap=None):
    """Serially read n_texts Europe PMC full texts, 3 host requests each (robots, tdmrep, XML), each through the H-5
    Crawl-delay slot (acq_discovered.crawl_wait) and the europepmc limiter, with an acquisition SEARCH call (limiter
    only) between texts, as on the inline path. Returns (texts read, send times, skips, elapsed)."""
    ft = FakeTime()
    raw = copy.deepcopy(RAW)
    if cap is not None:
        raw["defaults"]["max_wait_stream_s"] = cap
    reg = registry(ft, raw)
    rate_limit.set_registry_for_tests(reg)
    acq_discovered.reset_for_tests()
    monkeypatch.setattr(acq_discovered, "_sleep", ft.sleep)
    monkeypatch.setattr(acq_discovered, "crawl_interval",  # robots Crawl-delay 10 (stricter of it and 1/rate)
                        lambda url, rp=None: max(10.0, acq_discovered._documented_spacing(url)))
    acq = {"crawl_delay": {"enforce_on_acquisition": True}}
    ad = reg.portal("europepmc")
    sends, skips, read = [], [], 0
    t0 = ft.t
    try:
        with rate_limit.stream_wait_scope(stream), rate_limit.request_scope() as led:
            for i in range(n_texts):
                s = ad.acquire()  # the acquisition search for the next candidate (no crawl slot of its own)
                if isinstance(s, rate_limit.Permit):
                    sends.append(ft.t)
                    s.record(status=200)
                ok = True
                for _hop in ("robots", "tdmrep", "xml"):
                    acq_discovered.crawl_wait(EPMC_URL.format(n=i), acq, ft.clock, None)
                    p = ad.acquire()
                    if not isinstance(p, rate_limit.Permit):
                        skips.append(p.reason)
                        ok = False
                        break
                    sends.append(ft.t)
                    p.record(status=200)
                read += 1 if ok else 0
        return read, sends, skips, ft.t - t0, led.summary()
    finally:
        rate_limit.set_registry_for_tests(None)
        acq_discovered.reset_for_tests()


def test_hadrons_shape_stream_reads_three_texts_at_cap_30_without_violation(monkeypatch):
    read, sends, skips, elapsed, summ = _read_texts(monkeypatch, stream=True)
    assert read >= 3 and skips == []
    gaps = [b - a for a, b in zip(sends, sends[1:])]
    assert all(g >= 10.0 - 1e-9 for g in gaps), gaps  # never faster than 0.1/s nor Crawl-delay 10
    assert elapsed <= 300.0  # well inside the streamed stall rule (stream_ctx stall_s 300)
    assert summ["skips"] == {} and summ["permits"]["europepmc"] == len(sends)
    assert summ["waited_s"] > 0  # the limiter waits are accounted to THIS request


def test_hadrons_shape_old_cap_skips(monkeypatch):
    read, _sends, skips, _e, summ = _read_texts(monkeypatch, stream=True, cap=2.0)
    assert read < 3 and "pacing" in skips
    assert summ["skips"]["europepmc"]["pacing"] >= 1


def test_hadrons_shape_non_stream_unchanged(monkeypatch):
    read, _s, skips, _e, _summ = _read_texts(monkeypatch, stream=False)
    assert read < 3 and skips and set(skips) == {"pacing"}


# ------------------------------------------------------------------ per-request accounting, no leak across requests
def test_two_sequential_requests_keep_global_politeness_but_own_causes():
    ft = FakeTime()
    reg = registry(ft)
    ddg, epmc = reg.portal("duckduckgo"), reg.portal("europepmc")
    with rate_limit.request_scope() as r1:  # request 1 (non-streamed): DDG throttled, then Europe PMC paced
        ddg.acquire().record(status=429)
        assert ddg.acquire().reason == "cooldown"
        epmc.acquire().record(status=200)
        assert epmc.acquire().reason == "pacing"
    with rate_limit.request_scope() as r2, rate_limit.stream_wait_scope(True):  # request 2 (streamed)
        p = epmc.acquire()
        p.record(status=200)
    s1, s2 = r1.summary(), r2.summary()
    assert s1["skips"] == {"duckduckgo": {"cooldown": 1}, "europepmc": {"pacing": 1}}
    assert s2["skips"] == {}  # nothing of request 1 is reported by request 2
    assert isinstance(p, rate_limit.Permit) and s2["waited_s"] == pytest.approx(10.0)  # global token state kept
    assert ddg.cooling_down()  # global politeness kept: the cooldown still applies process-wide
    assert rate_limit.REQUEST_LEDGER.get() is None


def test_inline_run_sets_stream_wait_and_reports_limiter_per_run(monkeypatch):
    seen = []

    def body(*a, **k):
        seen.append((rate_limit.STREAM_WAIT.get(), rate_limit.REQUEST_LEDGER.get()))
        rate_limit.REQUEST_LEDGER.get().skip("duckduckgo", "daily_cap")
        return ia.Outcome("none")

    monkeypatch.setattr(ia, "_run_body", body)
    o1 = ia._run("q", {"inline": {}}, {}, lambda: 0.0, None)
    ctx = stream_ctx.StreamCtx(300.0)
    tok = stream_ctx.CTX.set(ctx)
    try:
        o2 = ia._run("q", {"inline": {}}, {}, lambda: 0.0, None)
    finally:
        stream_ctx.CTX.reset(tok)
    assert [s[0] for s in seen] == [False, True]
    assert seen[0][1] is not seen[1][1]
    assert o1.limiter["skips"] == {"duckduckgo": {"daily_cap": 1}} == o2.limiter["skips"]  # one each, not two
    assert "limiter" not in o1.public()


# ------------------------------------------------------------------ H-4: keep searching extends the inline deadline
def _acq(**inl):
    a = json.loads((ROOT / "config/auto_acquire.json").read_text())
    a["inline"].update(inl)
    return a


def test_h4_shipped_on_and_bounded():
    a = _acq()
    assert a["inline"]["persist_extends_deadline"] is True and a["inline"]["persist_deadline_max_s"] == 100
    ext = ask_service.persist_acq(a, True)
    assert ext["inline"]["deadline_s"] == 100 and a["inline"]["deadline_s"] == 45  # copy, original untouched
    # the request budget still bounds the round: _round_acq caps at the remaining time
    assert ask_service._round_acq(ext, 60.0)["inline"]["deadline_s"] == 60.0
    assert ask_service._round_acq(ext, 500.0)["inline"]["deadline_s"] == 100


@pytest.mark.parametrize("persist,inl", [(False, {}), (True, {"persist_extends_deadline": False}),
                                         (True, {"persist_deadline_max_s": "x"}), (True, {"persist_deadline_max_s": 0}),
                                         (True, {"persist_extends_deadline": 1})])
def test_h4_off_or_unreadable_is_unchanged(persist, inl):
    a = _acq(**inl)
    assert ask_service.persist_acq(a, persist) is a


def test_h4_never_shortens():
    a = _acq(deadline_s=120)
    assert ask_service.persist_acq(a, True)["inline"]["deadline_s"] == 120


# ------------------------------------------------------------------ A3: variant rounds query the reporting lane
_spec = importlib.util.spec_from_file_location("_acq6_rounds_harness", ROOT / "tests/test_acq6_rounds_1103.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)
NEWS_Q = "What are implications of the US 2026 Midterm elections?"


class LaneSpy:
    def __init__(self, page=True):
        self.calls, self.page = [], page

    def __call__(self, question, plan=None, **k):
        self.calls.append((question, tuple(p.text for p in plan.parts)))
        i = len(self.calls)
        pages = [SimpleNamespace(url=f"https://www.voanews.com/a/x/{i}.html", text="T")] if self.page else []
        return lane_evidence.LaneResult(links=[{"lane": "reporting", "url": f"https://www.voanews.com/a/x/{i}.html"},
                                               {"lane": "reporting", "url": "https://apnews.com/same"}],
                                        pages=pages, attributions=[{"name": "GDELT"}])


def _run_news(monkeypatch, spy, **inline):
    h = _h.H(monkeypatch, [[_h.cand("s0", usable=False)]], endless=True)
    monkeypatch.setattr(_h, "Q", NEWS_Q)
    monkeypatch.setattr(lane_evidence, "collect", spy)
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ia, "_web_read", lambda *a, **k: None)
    a = _h.cfg(**inline)
    a["bounds"]["acq_round_target_new_docs"] = 1000
    o = h.run(a, stream=True)
    return h, o


def test_a3_news_variant_rounds_each_run_one_lane_pass(monkeypatch):
    spy = LaneSpy()
    h, o = _run_news(monkeypatch, spy)
    rounds = len(h.searches)
    assert rounds >= 2 and len(spy.calls) == rounds - 1  # one lane pass per variant round, none for round 0
    assert all(q == NEWS_Q for q, _ in spy.calls)  # news intent decided on the question
    assert [parts[0] for _, parts in spy.calls] == [s["q"] for s in h.searches[1:]]  # the variant text is the query
    assert all(s["scholarly_only"] is False for s in h.searches[1:])  # news rounds keep web routes (F5)
    assert o.no_cache is True and len(o.lanes_extra) == rounds - 1
    merged = ia.with_web(o, None, None)
    assert len(merged.pages) == rounds - 1
    assert sum(1 for lk in merged.links if lk["url"] == "https://apnews.com/same") == 1  # deduplicated
    assert merged.attributions == [{"name": "GDELT"}]


def test_a3_lane_pages_count_toward_target(monkeypatch):
    spy = LaneSpy()
    h = _h.H(monkeypatch, [[_h.cand("s0", usable=False)]], endless=True)
    monkeypatch.setattr(_h, "Q", NEWS_Q)
    monkeypatch.setattr(lane_evidence, "collect", spy)
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ia, "_web_read", lambda *a, **k: None)
    a = _h.cfg()
    a["bounds"]["acq_round_target_new_docs"] = 2
    o = h.run(a, stream=True)
    assert len(spy.calls) == 2 and o.acquisition["stop"] == "met"


@pytest.mark.parametrize("inline,lane_on", [({"variant_lane_enabled": False}, True), ({}, False)])
def test_a3_switch_off_or_lane_off_no_extra_pass(monkeypatch, inline, lane_on):
    spy = LaneSpy()
    h = _h.H(monkeypatch, [[_h.cand("s0", usable=False)]], endless=True)
    monkeypatch.setattr(_h, "Q", NEWS_Q)
    monkeypatch.setattr(lane_evidence, "collect", spy)
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: lane_on)
    monkeypatch.setattr(ia, "_web_read", lambda *a, **k: None)
    o = h.run(_h.cfg(**inline), stream=True)
    assert spy.calls == [] and not o.lanes_extra


def test_a3_non_news_question_no_lane_pass(monkeypatch):
    spy = LaneSpy()
    h = _h.H(monkeypatch, [[_h.cand("s0", usable=False)]], endless=True)
    monkeypatch.setattr(lane_evidence, "collect", spy)
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: True)
    h.run(_h.cfg(), stream=True)
    assert len(h.searches) >= 2 and spy.calls == []


def test_a3_shipped_switch_on():
    assert _acq()["inline"]["variant_lane_enabled"] is True


def test_a3_non_streamed_path_no_variant_lane(monkeypatch):
    spy = LaneSpy()
    h = _h.H(monkeypatch, [[_h.cand("s0", usable=False)]], endless=True, search_cost=1.0)
    monkeypatch.setattr(_h, "Q", NEWS_Q)
    monkeypatch.setattr(lane_evidence, "collect", spy)
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ia, "_web_read", lambda *a, **k: None)
    h.run(_h.cfg(), stream=False)
    assert len(h.searches) >= 2 and spy.calls == []
