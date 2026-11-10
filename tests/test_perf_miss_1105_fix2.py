"""PERF-MISS-1105-FIX2: offline replay of the 2026-11-04 live miss ("What are the implications of the US 2026 Midterm
elections?" -> "Saved 0 of 6: sources are busy right now", then a silent page) with stubbed portals:
single-connection arXiv / DOI-landing hosts, a slow GDELT, and scholarly candidates that fail relevance.

F1 no self-inflicted R-ACQ-RATELIMITED (per-portal slot queueing; per-reason codes; only real cooldown/breaker/daily
   cap is "busy"); F2 the lane pass runs concurrently with acquisition and is traced; F4 progress at least every
   activity tick (5 s shipped) in the lane and reading-list phases, and a final event on every stream path;
F5 a news-intent variant round is not scholarly-only and its web results go through the gated web route.
Owner decision 2026-11-05 (news sources): link-only outlets are listed, never fetched; allow outlets are read;
excluded outlets are absent; the GDELT credit is present.
No network, no DB, no env files."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from types import SimpleNamespace

import pytest

import main
from vera import acq_config, acq_rounds, auto_acquire as aa, inline_acquire as ia, lane_evidence, official_lane
from vera import progress, public_fetch, query_variants6, rate_limit, reporting_lane, request_trace, scope_router
from vera import stream_ctx, web_route

Q = "What are the implications of the US 2026 Midterm elections?"
ENV = {"VERA_DB_URL_RO": "dummy-not-a-url", "VERA_DB_URL_WO": "dummy-not-a-url"}
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
RL_DEFAULTS = {"increase_frac": 0.1, "decrease_factor": 0.5, "min_rate_factor": 0.1, "breaker_failures": 3,
               "breaker_open_s": 10.0, "breaker_open_max_s": 40.0, "half_open_max": 1, "jitter_frac": 0.0,
               "max_wait_s": 0.1, "min_call_s": 0.0, "retry_after_max_s": 600.0, "base_cooldown_s": 30.0,
               "max_cooldown_s": 120.0, "poll_s": 0.01}
SRC = [{"url": "https://example.invalid/docs", "read": "2026-11-05"}]
FETCH_S = 0.25  # each stubbed full-text fetch holds its single portal slot this long (> max_wait_s 0.1)


@pytest.fixture(autouse=True)
def _iso():
    tok = progress.SINK.set(None)
    yield
    progress.SINK.reset(tok)
    rate_limit.set_registry_for_tests(None)


def _registry():
    """doi.org and arxiv.org: max_concurrency 1 (as config/rate_limits.json doi_landing / arxiv_web), short max_wait_s
    so the pre-fix limiter would skip the 2nd..4th parallel worker; real clock (threads)."""
    p = {"rate_per_s": 1000.0, "burst": 100, "max_concurrency": 1, "sources": SRC}
    cfg = rate_limit.parse_config({"defaults": RL_DEFAULTS, "portals": {
        "doi_landing": dict(p, hosts=["doi.org"]), "arxiv_web": dict(p, hosts=["arxiv.org"])}}, env={})
    return rate_limit.Registry(cfg)


def cfg(**inline):
    a = json.loads((ROOT / "config/auto_acquire.json").read_text())
    a["enabled"] = True
    a["inline"].update({"enabled": True, "acq6_rounds_enabled": True, **inline})
    a["bounds"]["docs_per_day"] = None
    return a


def scholarly(i):
    host = "doi.org" if i % 2 else "arxiv.org"
    return {"doi": f"10.99999/mid.{i}", "title": f"Elections paper {i}", "url": f"https://{host}/10.99999/mid.{i}"}


class Sink(main._StreamSink):
    """The real stream sink, plus a monotonic timestamp per appended event."""
    def __init__(self):
        super().__init__(10_000)
        self.ts: list = []

    def append(self, stage, state, counts):
        self.ts.append((time.monotonic(), stage, state, dict(counts)))
        return super().append(stage, state, counts)


class Harness:
    """inline_acquire.acquire on the streamed path; evaluate_candidate fetches through the shared limiter (the
    live _guarded_get mapping: RateLimited -> _Stop(rate_limited_code)), then fails relevance (scholarly papers on
    past elections do not pass a 2026-midterm relevance gate)."""

    def __init__(self, mp, pools, reg):
        aa.reset_for_tests()
        ia.reset_for_tests()
        acq_rounds.reset_for_tests()
        self.pools, self.searches, self.fetches, self.evaluated = list(pools), [], [], []
        self.lock = threading.Lock()
        get = rate_limit.limited_by_url(self._get, registry=reg)
        self.deps = SimpleNamespace(search=self.search, corpus_keys=lambda: set(), stored_rows=lambda cus: ([], []),
                                    store=SimpleNamespace(write=lambda p: {"status": "stored"}), close=None,
                                    refresh=None, url_check=None, http_get=get,
                                    lookup=SimpleNamespace(known=lambda *a, **k: False, docs_last_24h=lambda: 0))
        mp.setattr(scope_router, "intent_clean", lambda q, *a, **k: True)
        mp.setattr(aa, "_worker_loop", lambda *a, **k: None)
        mp.setattr(aa, "_topic_fit_active", lambda *a, **k: None)
        mp.setattr(aa, "rank_candidates", lambda q, cs, *a, **k: [(1.0, c.get("doi"), c, None) for c in cs if c.get("doi")])
        mp.setattr(aa, "reserve_reads", lambda ranked, n, *a, **k: list(ranked)[:max(0, int(n))])
        mp.setattr(aa, "evaluate_candidate", self.evaluate)
        mp.setattr(ia, "_stored_for", lambda ranked, *a, **k: (([], []), list(ranked)))
        mp.setattr(ia, "ephemeral_corpus", lambda plans, *a, **k: SimpleNamespace(plans=list(plans)) if plans else None)
        mp.setattr(ia, "_continue", lambda *a, **k: None)

    def _get(self, url, **kw):
        time.sleep(FETCH_S)
        with self.lock:
            self.fetches.append(url)
        return SimpleNamespace(status_code=200, headers={})

    def search(self, query, *a, **k):
        from vera import search_providers as sp
        i = len(self.searches)
        assert i < 50
        self.searches.append({"q": query, "scholarly_only": sp.SCHOLARLY_ONLY.get()})
        return [dict(c) for c in self.pools[i]] if i < len(self.pools) else []

    def evaluate(self, job, score, doi, c, isbn, acq, deps, icfg, fdcfg, clock, deadline, d, min_spans=1, **k):
        with self.lock:
            self.evaluated.append(doi)
        try:
            deps.http_get(c["url"], headers={}, timeout=5, allow_redirects=False, stream=True)
        except rate_limit.RateLimited as e:  # the live mapping (auto_acquire._guarded_get)
            raise aa._Stop(acq_config.rate_limited_code(acq, e.reason)) from None
        raise aa._Stop(acq_config.reason(acq, "irrelevant"))

    def run(self, acq):
        ctx = stream_ctx.StreamCtx(300.0)
        tok = stream_ctx.CTX.set(ctx)
        try:
            return ia.acquire(Q, acq, env=ENV, deps_factory=lambda a, e: self.deps)
        finally:
            stream_ctx.CTX.reset(tok)


def _limited_events(sink):
    return [c for _t, st, sa, c in sink.ts if st == "acquire" and sa == "limited"]


# ------------------------------------------------------------------------------------------------ F1
def test_f1_replay_no_self_inflicted_ratelimit_and_not_busy(monkeypatch):
    reg = _registry()
    h = Harness(monkeypatch, [[scholarly(i) for i in range(5)]], reg)
    monkeypatch.setattr(query_variants6, "variants", lambda q: [])  # round 0 only for this check
    sink = Sink()
    progress.SINK.set(sink)
    o = h.run(cfg())
    assert sorted(h.evaluated) == sorted(scholarly(i)["doi"] for i in range(5))
    assert len(h.fetches) == 5  # every candidate was fetched: none was skipped by our own limiter
    codes = dict(o.codes or {})
    assert not [c for c in codes if acq_config.is_rate_limited(cfg(), c)], codes
    assert codes.get("R-ACQ-IRRELEVANT") == 5
    assert reg.portal("doi_landing").skips == {} and reg.portal("arxiv_web").skips == {}
    lim = _limited_events(sink)
    assert lim and all(e.get("why") != ia.WHY_BUSY for e in lim), lim


def test_f1_without_queueing_the_limiter_self_skips_and_the_code_is_not_busy(monkeypatch):
    """Control: the pre-fix behaviour (no queue scope) reproduces the self-skip; the per-reason code it now gets
    (R-ACQ-RATELIMITED-CONCURRENCY) is NOT 'sources are busy'."""
    reg = _registry()
    ad = reg.portal("doi_landing")
    out = []

    def worker():
        try:
            rate_limit.call_with_limit(ad, lambda: time.sleep(FETCH_S))
            out.append("ok")
        except rate_limit.RateLimited as e:
            out.append(acq_config.rate_limited_code(cfg(), e.reason))
    ts = [threading.Thread(target=worker) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out.count("ok") == 1 and out.count("R-ACQ-RATELIMITED-CONCURRENCY") == 2
    assert not ia._provider_busy(cfg(), {"R-ACQ-RATELIMITED-CONCURRENCY": 2, "R-ACQ-RATELIMITED-PACING": 1})
    assert not ia._provider_busy(cfg(), {"R-ACQ-RATELIMITED": 1})  # the legacy plain code: unknown reason, not busy


def test_f1_queue_scope_admits_all_within_budget_and_skips_at_deadline():
    reg = _registry()
    ad = reg.portal("arxiv_web")
    out = []

    def worker(budget):
        with rate_limit.queue_scope(time.monotonic() + budget):
            try:
                rate_limit.call_with_limit(ad, lambda: time.sleep(FETCH_S))
                out.append("ok")
            except rate_limit.RateLimited as e:
                out.append(e.reason)
    ts = [threading.Thread(target=worker, args=(5.0,)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out == ["ok"] * 4
    out.clear()
    ts = [threading.Thread(target=worker, args=(5.0,))] + [threading.Thread(target=worker, args=(0.05,))]
    ts[0].start()
    time.sleep(0.02)
    ts[1].start()
    [t.join() for t in ts]
    assert "ok" in out and len(out) == 2
    assert set(out) - {"ok"} <= {"concurrency", "wait_too_long", "deadline_passed"}  # bounded: never waits past its own budget; never "busy"
    assert not (set(out) & rate_limit.PROVIDER_BUSY_REASONS)


@pytest.mark.parametrize("why,busy", [("cooldown", True), ("breaker_open", True), ("half_open_busy", True),
                                      ("daily_cap", True), ("concurrency", False), ("pacing", False),
                                      ("deadline", False)])
def test_f1_per_reason_codes_and_busy_classification(why, busy):
    a = cfg()
    code = acq_config.rate_limited_code(a, why)
    assert code == "R-ACQ-RATELIMITED-" + why.upper()
    assert acq_config.is_rate_limited(a, code) and ia._transient(a, code)  # still background-eligible
    assert ia._provider_busy(a, {code: 1}) is busy
    o = ia.Outcome("none", acquisition={"status": "resource_limited", "stored": 0, "target": 6})
    sink = Sink()
    progress.SINK.set(sink)
    stream_tok = stream_ctx.CTX.set(stream_ctx.StreamCtx(300.0))
    try:
        ia._emit_final(o, a, {code: 1}, None)
    finally:
        stream_ctx.CTX.reset(stream_tok)
    assert _limited_events(sink)[0]["why"] == (ia.WHY_BUSY if busy else ia.WHY_NO_MORE)


def test_f1_guarded_get_carries_the_skip_reason():
    def get(*a, **k):
        raise rate_limit.RateLimited("doi_landing", "concurrency", 0.01)
    deps = SimpleNamespace(http_get=get, url_check=lambda u: (True, ""))
    with pytest.raises(aa._Stop) as e:
        aa._guarded_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?q=x", cfg(), deps, 1000, 5.0, True)
    assert e.value.code == "R-ACQ-RATELIMITED-CONCURRENCY"


# ------------------------------------------------------------------------------------------------ F5
def test_f5_news_round_is_not_scholarly_only_and_web_results_use_the_gated_route(monkeypatch):
    # RETRIEVAL-1109 A3: this test is about the scholarly/web routing; the variant-round reporting-lane pass is off here
    # (it would call GDELT on the streamed path; covered by its own tests).
    monkeypatch.setattr(ia, "variant_lane_on", lambda acq: False)
    reg = _registry()
    web = {"title": "Midterm coverage", "url": "https://news.example.org/midterms-2026"}
    h = Harness(monkeypatch, [[scholarly(0)], [scholarly(1), dict(web)], []], reg)
    monkeypatch.setattr(query_variants6, "variants", lambda q: ["2026 midterm elections implications", "midterms 2026"])
    seen = []

    def read_web(question, cands, should_stop=None, max_fetches=None):
        seen.append(([c.get("url") for c in cands], max_fetches))
        return web_route.WebResult()
    monkeypatch.setattr(web_route, "read_web", read_web)
    h.run(cfg())
    assert len(h.searches) >= 2
    assert h.searches[0]["scholarly_only"] is False and h.searches[1]["scholarly_only"] is False
    assert [web["url"]] in [u for u, _m in seen]  # the variant round's web result went through web_route.read_web
    assert all(m is None or m >= 0 for _u, m in seen)
    assert scholarly(1)["doi"] in h.evaluated and None not in h.evaluated  # the web item was not evaluated as a paper


def test_f5_non_news_rounds_stay_scholarly_only(monkeypatch):
    reg = _registry()
    h = Harness(monkeypatch, [[scholarly(0)], [scholarly(1)]], reg)
    monkeypatch.setattr(query_variants6, "variants", lambda q: ["variant one"])
    ctx = stream_ctx.StreamCtx(300.0)
    tok = stream_ctx.CTX.set(ctx)
    try:
        ia.acquire("What are the de facto best practices for agent archetypes in orchestration?", cfg(), env=ENV,
                   deps_factory=lambda a, e: h.deps)
    finally:
        stream_ctx.CTX.reset(tok)
    assert h.searches[0]["scholarly_only"] is False
    assert all(s["scholarly_only"] is True for s in h.searches[1:]) and len(h.searches) >= 2


# ------------------------------------------------------------------------------------------------ F2 / F4 lane
GDELT_ARTS = {"articles": [
    {"url": "https://globalvoices.org/2026/11/03/us-midterms/", "title": "GV midterms", "seendate": "20261103T100000Z"},
    {"url": "https://www.nytimes.com/2026/11/03/us/politics/midterms.html", "title": "NYT midterms",
     "seendate": "20261103T110000Z"},
    {"url": "https://unknown-outlet.example/midterms", "title": "Unlisted midterms", "seendate": "20261103T120000Z"},
    {"url": "https://excluded-outlet.example/midterms", "title": "Excluded", "seendate": "20261103T130000Z"},
]}
GDELT_S = 0.8  # slow GDELT


@pytest.fixture
def lane_env(monkeypatch):
    """Shipped reporting-lane config (emit_link_only true, outlet_query all since 2026-11-05) plus one test outlet
    marked exclude; official lane off; a slow GDELT stub; a recording page fetch. Nothing leaves the process."""
    monkeypatch.setattr(official_lane, "lane_enabled", lambda cfg=None: False)
    monkeypatch.setattr(scope_router, "intent_clean", lambda q, *a, **k: True)
    calls, fetched = [], []

    def gdelt(url, headers, timeout):
        calls.append((time.monotonic(), url))
        time.sleep(GDELT_S)
        return 200, {}, json.dumps(GDELT_ARTS).encode()

    def fetch(url, **kw):
        fetched.append(url)
        return {"text": "Midterm elections 2026 implications reported. " * 40, "url": url, "content_type": "text/html"}
    monkeypatch.setattr(reporting_lane, "_default_http_get", gdelt)
    monkeypatch.setattr(reporting_lane, "_SHARED", reporting_lane._new_state())
    monkeypatch.setattr(public_fetch, "fetch", fetch)
    c = reporting_lane.load_config()
    c["outlets"] = list(c["outlets"]) + [{"host": "excluded-outlet.example", "terms_verdict": "exclude",
                                          "terms_evidence": "test: terms forbid linking"}]
    monkeypatch.setattr(reporting_lane, "load_config", lambda path=reporting_lane.CONFIG_PATH: dict(c))
    return SimpleNamespace(calls=calls, fetched=fetched)


def test_owner_news_decision_shipped_config():
    c = reporting_lane.load_config()
    assert c["emit_link_only"] is True and c["outlet_query"] == "all" and c["unlisted_verdict"] == "links_only"
    assert set(reporting_lane.allow_hosts(c)) == {"globalvoices.org", "en.wikinews.org", "voanews.com"}  # H5


def test_owner_news_decision_link_only_never_fetched_allow_read_excluded_absent_credit(lane_env):
    res = lane_evidence.collect(Q)
    q = lane_env.calls[0][1]
    assert "domain%3A" not in q  # outlet_query all: every outlet is searched
    by = {lk["url"]: lk for lk in res.links}
    assert by["https://www.nytimes.com/2026/11/03/us/politics/midterms.html"]["read"] is False  # links_only outlet
    assert by["https://unknown-outlet.example/midterms"]["read"] is False  # unlisted = deny-all: link only
    assert not any("excluded-outlet" in u for u in by)  # exclude outlet: absent
    assert not any("nytimes" in u or "unknown-outlet" in u or "excluded-outlet" in u for u in lane_env.fetched)
    gv = "https://globalvoices.org/2026/11/03/us-midterms/"
    assert by[gv]["read"] is True and [p.url for p in res.pages] == [gv]  # allow outlet: read in full
    assert all(u.startswith("https://globalvoices.org/") for u in lane_env.fetched)  # its page and robots only
    assert res.attributions and res.attributions[0]["url"] == "https://www.gdeltproject.org/"
    pub = res.public()
    assert pub["credits"] and pub["n_read"] == 1
    assert {d["reason"] for d in res.diag} == {"ok"} and "diag" not in pub


def test_f2_lane_runs_concurrently_with_acquisition_and_is_traced(lane_env, monkeypatch):
    events = []
    monkeypatch.setattr(request_trace, "event", lambda stage, code, counts=None, detail=None:
                        events.append((stage, code, dict(counts or {}), dict(detail or {}))))
    t0 = time.monotonic()
    task = ia.start_lane_pass(Q, {})
    time.sleep(GDELT_S)  # the acquisition rounds, in the caller's thread
    res = task.result()
    took = time.monotonic() - t0
    assert lane_env.calls and lane_env.calls[0][0] - t0 < 0.3  # GDELT called at once, not after acquisition
    assert took < 2 * GDELT_S  # overlapped, not GDELT_S + GDELT_S
    assert res is not None and res.pages
    lane = [e for e in events if e[0] == "lane"]
    assert len(lane) == 1
    _s, code, counts, detail = lane[0]
    assert code == "done" and detail == {"portal": "gdelt", "status": "called", "reason": "ok", "news": True}
    assert counts["pages"] == 1 and counts["links"] == 3 and counts["items"] == 3 and counts["calls"] == 1


def test_f2_lane_skip_reason_is_traced(lane_env, monkeypatch):
    events = []
    monkeypatch.setattr(request_trace, "event", lambda stage, code, counts=None, detail=None:
                        events.append((stage, code, dict(detail or {}))))
    ia.start_lane_pass(Q, {}).result()
    ia.start_lane_pass(Q, {}).result()  # within min_interval_s 6: the portal is not called
    assert [e[2]["reason"] for e in events if e[0] == "lane"] == ["ok", "spacing"]
    assert [e[1] for e in events if e[0] == "lane"] == ["done", "skipped"]
    assert len(lane_env.calls) == 1


def _max_gap(sink, t_from, t_to):
    ts = [t_from] + [t for t, *_ in sink.ts if t_from <= t <= t_to] + [t_to]
    return max(b - a for a, b in zip(ts, ts[1:]))


def test_f4_lane_wait_shows_news_activity_at_every_tick(lane_env, monkeypatch):
    tick = 0.15
    monkeypatch.setattr(progress, "CFG", progress.Config(**{**progress.CFG.__dict__, "stream": progress.StreamCfg(
        **{**progress.CFG.stream.__dict__, "activity_tick_s": tick})}))
    sink = Sink()
    progress.SINK.set(sink)
    task = ia.start_lane_pass(Q, {})
    t0 = time.monotonic()
    task.result()  # the caller had nothing else to do: it waits on the lane the whole GDELT call
    t1 = time.monotonic()
    acts = [c.get("act") for _t, _s, _st, c in sink.ts]
    assert progress.ACT_NEWS in acts
    assert _max_gap(sink, t0, t1) <= tick + 0.2


def test_f4_reading_list_phase_is_never_silent_and_stream_always_ends(monkeypatch):
    """A slow reading-list fallback (the grounding search after 'Saved 0 of 6') inside the real stream worker: the
    reading-list activity is re-set every tick, and the worker always ends with a done event plus a result."""
    from vera import ask_service
    tick = 0.15
    monkeypatch.setattr(progress, "CFG", progress.Config(**{**progress.CFG.__dict__, "stream": progress.StreamCfg(
        **{**progress.CFG.stream.__dict__, "activity_tick_s": tick})}))
    marks = {}

    def body(*a, **k):
        marks["t0"] = time.monotonic()
        time.sleep(0.9)
        marks["t1"] = time.monotonic()
        return ask_service.AskResult(answer="No answer yet.", tokens_used=0, cost_usd=0.0, result_kind="reading_list")
    monkeypatch.setattr(ask_service, "_reading_list_body", body)
    monkeypatch.setattr(main, "_ask_impl", lambda req, request, bg: ask_service._reading_list(
        Q, "", None, SimpleNamespace(tokens_used=0, cost_usd=0.0), {}, None, None))
    monkeypatch.setattr(main, "_schedule_live_stats", lambda *a, **k: None)
    monkeypatch.setattr(main, "_trace_finish", lambda resp: None)
    monkeypatch.setattr(main, "jsonable_encoder", lambda r: {"answer": getattr(r, "answer", "")})
    for raise_it in (False, True):
        sink = Sink()
        if raise_it:
            monkeypatch.setattr(ask_service, "_reading_list_body", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        # as in /ask/stream: the worker runs in a copied context (it sets the stream ctx and progress sink)
        contextvars.copy_context().run(main._stream_worker, sink, stream_ctx.StreamCtx(300.0), SimpleNamespace(),
                                       None, None)
        items = []
        while not sink.items.empty():
            items.append(sink.items.get_nowait())
        dones = [i for i in items if i[0] == "event" and i[1] == "done"]
        results = [i for i in items if i[0] == "result"]
        assert len(dones) == 1 and len(results) == 1, items  # a final event and exactly one result, every path
        if not raise_it:
            assert results[0][1] == 200
            assert progress.ACT_READING_LIST in [c.get("act") for _t, _s, _st, c in sink.ts]
            assert _max_gap(sink, marks["t0"], marks["t1"]) <= tick + 0.2
        else:
            assert results[0][1] == 500 and dones[0][2] == "error"


def test_f4_shipped_tick_is_5s_and_labels_exist():
    assert progress.CFG is None or progress.CFG.stream.activity_tick_s == 5
    raw = json.loads((ROOT / "config/progress.json").read_text())
    assert progress.validate_config(raw).stream.activity_tick_s == 5
    from vera.progress_ui import Labels
    lab = Labels.load()
    assert "news" in lab.activity_label({"act": progress.ACT_NEWS, "elapsed": 7})
    assert "reading list" in lab.activity_label({"act": progress.ACT_READING_LIST, "elapsed": 7})
    for kind in ("orphaned", "cancelled"):
        t = lab.terminal_text(kind)
        assert t and t != lab.terminal_text("error")


def test_f4_orphaned_stream_gets_its_own_terminal():
    from vera import progress_ui
    h = progress_ui.StreamHolder(t0=0.0, last_read=0.0)
    clock = [0.0]

    def stream_fn(*a, should_stop=None):
        clock[0] = 500.0  # the page stopped reading (no fragment tick for > orphan_s)
        yield {"type": "hb", "seq": 1}
    progress_ui.stream_worker(h, stream_fn, (), lambda: clock[0], 120.0)
    assert h.terminal == "orphaned"
    assert "stopped receiving updates" in progress_ui.Labels.load().terminal_text(h.terminal)
