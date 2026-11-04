"""STREAM-1031 core (OD-1031-NODEADLINE, OD-1031-STREAM-ON, OD-1031-CEIL31, HITL4 item 4): inline acquisition on the streamed
path has no 45 s cut, stops on stall / disconnect, is bounded by counts, and emits counts-only events; the legacy path is
unchanged. T1, T2, T3, T4, T5, T10a, T10c, T10d, T12, TB1. Offline: fake clocks, stubbed evaluate/search/store."""
from __future__ import annotations

import json

import pytest

from tests import acq_support as A
from vera import acq_config, ask_service, auto_acquire, inline_acquire, progress, rate_limit, stream_ctx


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    monkeypatch.setattr(inline_acquire, "ephemeral_corpus", lambda *a, **k: object())
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    yield
    stream_ctx.CTX.set(None)


@pytest.fixture
def events(monkeypatch):
    ev = []
    monkeypatch.setattr(progress, "emit", lambda stage, state, **c: ev.append((stage, state, dict(c))))
    return ev


def plan_for(doi, spans=3):
    return {"admitted_spans": spans, "canonical_url": f"https://doi.org/{doi}", "spans": [],
            "source": {"content": "x", "content_hash": "h", "provenance": {}}}


def stub_eval(monkeypatch, calls, clock=None, step=0.0, reject=()):
    def ev(job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clk, deadline, dur, min_spans=None):
        calls.append(doi)
        if clock is not None:
            clock.t += step
        if doi in reject:
            raise auto_acquire._Stop("R-ACQ-IRRELEVANT")
        return plan_for(doi), None
    monkeypatch.setattr(auto_acquire, "evaluate_candidate", ev)


def run(n=3, clk=None, search=None, cfg=None, store=None, stream=True, stall_s=300, workers=1, deps_mut=None):
    clk = clk or Clock()
    if cfg is None:
        cfg = A.cfg()
        cfg["inline"]["acq6_rounds_enabled"] = False  # A6 kill switch off: the pre-A6 single-search contract (rounds: test_acq6_rounds_1103)
    cfg["inline"]["fetch_workers"] = workers
    cands = [A.cand(f"10.9999/sc.{i}") for i in range(n)]
    d = A.deps(cands, A.Web({}, {}), store=store or A.Store())
    if search is not None:
        d.search = lambda q, k, tm: search(cands)
    if deps_mut:
        deps_mut(d)
    ctx = stream_ctx.StreamCtx(stall_s, clock=clk) if stream else None
    stream_ctx.CTX.set(ctx)
    o = inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    return o, ctx, d


def test_T1_thirty_minutes_on_the_stream_path_has_no_deadline(monkeypatch, events):
    clk, calls, seen = Clock(), [], {}
    stub_eval(monkeypatch, calls, clk, step=60.0)  # 31 candidates x 60 s = 31 min of fake time

    def search(c):
        seen["scope"] = rate_limit.DEADLINE.get()
        clk.t += 60
        return c
    o, _ctx, _d = run(n=40, clk=clk, search=search)
    assert seen["scope"] is None  # D2: no deadline scope on the stream path
    assert clk.t - 1000.0 >= 1800 and acq_config.reason(A.cfg(), "deadline") not in o.codes
    assert o.status == "ok" and o.stored == 31 and o.acquisition["status"] == "met"
    assert "deadline" not in json.dumps(o.public()["acquisition"]) and o.public()["acquisition"] == {
        "status": "met", "stored": 31, "target": 6}


def test_T2_count_stops_31_stored_and_short_supply_is_resource_limited(monkeypatch, events):
    calls = []
    stub_eval(monkeypatch, calls)
    o, _c, _d = run(n=45)
    assert o.stored == 31 and len(calls) == 31 and o.acquisition["status"] == "met"
    assert events[-1] == ("acquire", "done", {"stored": 31, "target": 6})
    events.clear()
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    o, _c, _d = run(n=3)
    assert o.stored == 3 and o.acquisition == {"status": "resource_limited", "stored": 3, "target": 6}
    assert events[-1] == ("acquire", "limited", {"stored": 3, "target": 6, "why": 1})


def test_T3_rate_limit_wait_is_bounded_by_max_wait_with_no_deadline_scope():
    with rate_limit.deadline_scope(None):
        reg = rate_limit.get_registry()
        a = reg.portal("openalex")
        assert a._budget(1000.0, rate_limit.DEADLINE.get()) == float(a.c["max_wait_s"]) <= 2.0


def test_T4_events_are_counts_only_and_name_nothing(monkeypatch, events):
    stub_eval(monkeypatch, [], reject=("10.9999/sc.2",))
    o, _c, _d = run(n=40)
    blob = json.dumps(events)
    for bad in ("10.9999", "doi.org", "http", "openalex", "europepmc", "arxiv", "duckduckgo", A.Q_EPI[:20], "SYNTHETIC"):
        assert bad.lower() not in blob.lower()
    assert all(type(v) is int for _s, _t, c in events for v in c.values())
    assert events[0] == ("search", "start", {"step": 1})
    assert events[1] == ("search", "done", {"found": 40, "shortlisted": 31})
    ups = [c for s, t, c in events if (s, t) == ("acquire", "update")]
    assert len(ups) == 31 and [u["checked"] for u in ups] == list(range(1, 32))
    assert all(set(u) == {"checked", "usable", "held", "stored", "target", "shortlisted"} for u in ups)
    assert ups[-1]["usable"] == 30 == ups[-1]["stored"] and ups[-1]["shortlisted"] == 31 and ups[-1]["target"] == 6
    assert events[-1][:2] == ("acquire", "done")


def test_T5_stream_budget_has_no_deadline_and_one_inline_per_request(monkeypatch):
    assert ask_service.request_budget().deadline != float("inf")
    stream_ctx.CTX.set(stream_ctx.StreamCtx(300))
    b = ask_service.request_budget()
    assert b.deadline == float("inf") and b.inline_used is False
    from vera import fulldoc_corpus
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(ask_service, "inline_need_s", lambda acq, g: 125.0)
    monkeypatch.setattr(inline_acquire, "acquire", lambda *a, **k: inline_acquire.Outcome("none"))
    cfg = A.cfg()
    st, _o = ask_service._research_attempt("what is the pathogen transmission", None, cfg, {}, {}, b)
    assert st["status"] != "deadline" and b.inline_used is True
    st2, o2 = ask_service._research_attempt("what is the pathogen transmission", None, cfg, {}, {}, b)
    assert st2 == {"status": "busy"} and o2 is None  # the one-inline-per-request guard stays
    stream_ctx.CTX.set(None)
    st3, _ = ask_service._research_attempt("what is the pathogen transmission", None, cfg, {}, {},
                                           ask_service.RequestBudget(__import__("time").monotonic() + 1))
    assert st3 == {"status": "deadline"}  # legacy: the 125 s guard is unchanged


def test_T5b_cancelled_client_stops_before_writer_and_check():
    ctx = stream_ctx.StreamCtx(300)
    stream_ctx.CTX.set(ctx)
    ask_service._pre_llm()  # not cancelled: no-op
    ctx.cancel()
    with pytest.raises(stream_ctx.StreamCancelled):
        ask_service._pre_llm()
    with pytest.raises(stream_ctx.StreamCancelled):
        ask_service._pre_llm(writer=True)


def test_T10a_two_busy_slots_third_gets_the_busy_event(events):
    inline_acquire._I.active = 2
    o, _c, _d = run(n=3)
    assert o.status == "busy" and events == [("acquire", "busy", {})]


def test_T10a_waiter_on_the_stream_path_ends_with_the_holder(monkeypatch, events):
    import threading
    stub_eval(monkeypatch, [])
    holder_ctx = stream_ctx.StreamCtx(300)
    key = inline_acquire._cache_key(A.Q_EPI, None)
    slot = [threading.Event(), None, 0, holder_ctx]
    inline_acquire._I.inflight[key] = slot
    inline_acquire._I.active = 1
    stream_ctx.CTX.set(stream_ctx.StreamCtx(300))
    holder_ctx.stop(stream_ctx.STALLED)  # the holder stalled: the waiter is released within a tick, not 45 s
    import time
    t = time.monotonic()
    o = inline_acquire.acquire(A.Q_EPI, A.cfg(), env=A.ENV_ON, deps_factory=lambda a, e: None)
    assert time.monotonic() - t < 5 and o.status == "busy"


def test_T10c_legacy_path_still_cuts_at_45s_even_with_stream_config_on(monkeypatch, events):
    monkeypatch.setattr(progress, "stream_enabled", lambda: True, raising=False)
    clk, calls, seen = Clock(), [], {}
    stub_eval(monkeypatch, calls, clk)

    def search(c):
        seen["scope"] = rate_limit.DEADLINE.get()
        clk.t += 46.0  # the search alone outlasts inline.deadline_s 45
        return c
    o, ctx, _d = run(n=3, clk=clk, search=search, stream=False)
    assert ctx is None and seen["scope"] is not None  # D2 deadline scope present
    assert o.acquisition["status"] == "incomplete_deadline" and calls == [] and events == []  # legacy emits nothing here
    assert ask_service.request_budget().deadline != float("inf")


def test_T10d_disconnect_during_search_starts_no_fetch(monkeypatch, events):
    calls, holder = [], {}
    stub_eval(monkeypatch, calls)

    def search(c):
        holder["ctx"].cancel()
        return c
    clk = Clock()
    holder["ctx"] = ctx = stream_ctx.StreamCtx(300, clock=clk)
    stream_ctx.CTX.set(ctx)
    cfg = A.cfg()
    d = A.deps([A.cand(f"10.9999/sc.{i}") for i in range(5)], A.Web({}, {}))
    d.search = lambda q, k, t: search([A.cand(f"10.9999/sc.{i}") for i in range(5)])
    o = inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    assert calls == [] and d.store.plans == []
    assert o.acquisition["status"] == "incomplete_disconnected"
    assert not [e for e in events if e[:2] == ("acquire", "update")]


def test_T12_stall_stops_new_candidates_and_persists_admitted(monkeypatch, events):
    clk, calls = Clock(), []
    stub_eval(monkeypatch, calls)
    n_up = {"n": 0}

    def emit(stage, state, **c):
        events.append((stage, state, dict(c)))
        if (stage, state) == ("acquire", "update"):
            n_up["n"] += 1
            if n_up["n"] == 2:
                clk.t += 301.0  # no progress for longer than stall_s
    monkeypatch.setattr(progress, "emit", emit)
    o, ctx, d = run(n=10, clk=clk, workers=1)
    assert len(calls) == 2 and len(d.store.plans) == 2 and o.stored == 2  # admitted documents persisted
    assert o.acquisition == {"status": "resource_limited", "stored": 2, "target": 6, "stop": "stalled"}
    assert events[-1] == ("acquire", "limited", {"stored": 2, "target": 6, "why": 4})


def test_providers_exhausted_stop_is_resource_limited_with_stop(monkeypatch, events):
    calls, holder = [], {}
    stub_eval(monkeypatch, calls)

    def ev(job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clk, deadline, dur, min_spans=None):
        calls.append(doi)
        holder["ctx"].stop(stream_ctx.PROVIDERS_EXHAUSTED)
        return plan_for(doi), None
    monkeypatch.setattr(auto_acquire, "evaluate_candidate", ev)
    clk = Clock()
    holder["ctx"] = ctx = stream_ctx.StreamCtx(300, clock=clk)
    stream_ctx.CTX.set(ctx)
    d = A.deps([A.cand(f"10.9999/sc.{i}") for i in range(6)], A.Web({}, {}))
    cfg = A.cfg()
    cfg["inline"]["fetch_workers"] = 1
    o = inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=clk, deps_factory=lambda a, e: d)
    assert len(calls) == 1 and o.acquisition["stop"] == "providers_exhausted" and o.acquisition["status"] == "resource_limited"
    assert events[-1][2]["why"] == 5


def test_TB1_single_pass_evaluations_and_reuse_checks_are_bounded(monkeypatch, events):
    from vera.corpus_admission import canonical_url
    calls, checks = [], []
    stub_eval(monkeypatch, calls)
    monkeypatch.setattr(inline_acquire, "_reuse_relevant", lambda *a, **k: checks.append(1) or inline_acquire.REUSE_IRRELEVANT)

    def rows(d):
        d.stored_rows = lambda cus: ([{"source_id": i + 1, "canonical_url": u, "content": "x"} for i, u in enumerate(cus)], [])
    cfg = A.cfg()
    cfg["live"] = {**acq_config.live(cfg), "max_reuse_checks_per_prompt": 5}
    o, _c, _d = run(n=60, cfg=cfg, deps_mut=rows)
    assert len(checks) == 5  # reuse checks bounded by live.max_reuse_checks_per_prompt on the stream path
    assert len(calls) == 0 and canonical_url({"doi": "10.9999/sc.0", "url": "https://doi.org/10.9999/sc.0"})
    # default bound: never more than 31 checks even with 60 stored candidates, and evaluations never exceed the shortlist
    auto_acquire.reset_for_tests(); inline_acquire.reset_for_tests(); checks.clear(); calls.clear()
    o, _c, _d = run(n=60, cfg=A.cfg(), deps_mut=rows)
    assert len(checks) <= 31 == acq_config.live(A.cfg())["max_reuse_checks_per_prompt"]
    auto_acquire.reset_for_tests(); inline_acquire.reset_for_tests(); calls.clear()
    o, _c, _d = run(n=60)
    assert len(calls) <= 31 and o.stored <= 31
    assert len(calls) + 31 + 5 <= 129  # steps <= R + E + reuse for the single-pass shape (R=1 here)
