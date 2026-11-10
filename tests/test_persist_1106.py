"""MIDTERMS-WP8-1106 (owner OD-1106-MIDTERMS-D1D4, D1/D2): a keep-searching request (persist) overrides only the
provisional no-progress heuristic; every other bound still stops the rounds; a heuristic stop is disclosed; the
background note never claims more than the real best-effort hand-off count. Offline: stubs only, no network, no DB."""
from __future__ import annotations

from types import SimpleNamespace

from tests.test_acq6_rounds_1103 import ENV, H, Q, cand, cfg
from vera import ask_service, auto_acquire as aa, inline_acquire as ia, query_variants6, stream_ctx, ui_evidence

VARIANTS = [f"variant query {i}" for i in range(5)]


def _run(h, acq, persist=False, stream=True):
    h.ctx, tok = None, None
    if stream:
        h.ctx = stream_ctx.StreamCtx(300.0, clock=h.clock)
        tok = stream_ctx.CTX.set(h.ctx)
    try:
        return ia.acquire(Q, acq, env=ENV, clock=h.clock, deps_factory=lambda a, e: h.deps,
                          **({"persist": True} if persist else {}))
    finally:
        if tok is not None:
            stream_ctx.CTX.reset(tok)


def _h(monkeypatch, **k):
    monkeypatch.setattr(query_variants6, "variants", lambda q: list(VARIANTS))
    return H(monkeypatch, [[cand("z0", usable=False)]], **k)  # every variant round finds nothing


def test_1_plain_request_stops_on_the_heuristic_and_discloses_it(monkeypatch):
    h = _h(monkeypatch)
    o = _run(h, cfg())
    assert len(h.searches) == 1 + 2
    assert o.acquisition["stop"] == "no_progress" and o.acquisition["heuristic"] is True
    assert "persist" not in o.acquisition


def test_2_persist_searches_every_variant(monkeypatch):
    h = _h(monkeypatch)
    o = _run(h, cfg(), persist=True)
    assert len(h.searches) == 1 + len(VARIANTS)
    assert o.acquisition["stop"] == "variants_exhausted" and o.acquisition["persist"] is True
    assert "heuristic" not in o.acquisition


def test_2b_heuristic_override_recorded_in_r6(monkeypatch):
    seen = {}
    real = ia._extra_rounds

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(a[8]["acq6"])  # st
        return out
    monkeypatch.setattr(ia, "_extra_rounds", spy)
    _run(_h(monkeypatch), cfg(), persist=True)
    assert seen["heuristic_overridden"] == 1 and seen["persist"] is True


def test_3_persist_still_stops_on_rate(monkeypatch):
    h = _h(monkeypatch)
    real = aa.reserve_job
    n = {"i": 0}

    def reserve(acq, q, clock=None, floor=0):
        n["i"] += 1
        return real(acq, q, clock=clock, floor=floor) if n["i"] <= 2 else "R-ACQ-RATE"
    monkeypatch.setattr(aa, "reserve_job", reserve)
    a = cfg()
    a["bounds"]["acq_rounds_one_job"] = False  # the per-round job rule, under which a round can hit the rate cap
    o = _run(h, a, persist=True)
    assert o.acquisition["stop"] == "rate" and len(h.searches) == 2


def test_3_persist_still_stops_on_disconnect(monkeypatch):
    h = _h(monkeypatch)
    h_search = h.search

    def search(query, *a, **k):
        out = h_search(query, *a, **k)
        if len(h.searches) == 3:
            h.ctx.cancel()
        return out
    h.deps.search = search
    o = _run(h, cfg(), persist=True)
    assert len(h.searches) == 3 and o.acquisition["status"] == "incomplete_disconnected"


def test_3_persist_still_stops_on_met_and_ceiling(monkeypatch):
    monkeypatch.setattr(query_variants6, "variants", lambda q: list(VARIANTS))
    h = H(monkeypatch, [[cand("m0"), cand("m1")], [cand("m2"), cand("m3")], [cand("m4"), cand("m5")]])
    o = _run(h, cfg(), persist=True)
    assert o.acquisition["status"] == "met" and len(h.searches) == 3
    h2 = H(monkeypatch, [[cand(f"c{i}")] for i in range(10)])
    a = cfg()
    a["bounds"]["acq_round_target_new_docs"] = 1000
    monkeypatch.setattr(ia.acq_config, "inline_fresh_docs", lambda acq: 3)
    o2 = _run(h2, a, persist=True)
    assert o2.acquisition["stop"] == "ceiling31"


def test_3_persist_still_stops_on_the_nonstream_deadline(monkeypatch):
    h = _h(monkeypatch, search_cost=12.0)
    o = _run(h, cfg(deadline_s=45, per_doc_timeout_s=5), persist=True, stream=False)
    assert o.acquisition["stop"] == "deadline" and len(h.searches) < 1 + len(VARIANTS)


def test_5_cache_isolation_plain_then_persist(monkeypatch):
    h = _h(monkeypatch)
    calls = []
    real = ia._run
    monkeypatch.setattr(ia, "_run", lambda *a, **k: calls.append(k.get("persist")) or real(*a, **k))
    a = cfg()
    _run(h, a, stream=False)
    _run(h, a, stream=False)  # a plain repeat replays the cache
    _run(h, a, persist=True, stream=False)
    assert calls == [None, True]
    assert ia._cache_key(Q, None) + ("persist",) in {k for k in ia._I.results}


def test_6_ask_service_inline_passes_persist_only_when_set(monkeypatch):
    seen = []

    def fake_acquire(question, acq, **kw):
        seen.append(dict(kw))
        return ia.Outcome("off")
    monkeypatch.setattr(ia, "acquire", fake_acquire)
    monkeypatch.setattr(ia, "start_lane_pass", lambda q, a: SimpleNamespace(result=lambda: None))
    monkeypatch.setattr(ask_service, "_rt_event", lambda *a, **k: None)
    from vera import fulldoc_corpus
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda c: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: {})
    dec = SimpleNamespace(topic_key="")
    tok = ask_service._REQUEST_PERSIST.set(True)
    try:
        ask_service._inline(Q, "m", None, dec, {}, None, {}, info={})
    finally:
        ask_service._REQUEST_PERSIST.reset(tok)
    ask_service._inline(Q, "m", None, dec, {}, None, {}, info={})
    assert seen[0].get("persist") is True and "persist" not in seen[1]


def _line(aq, ledger=None):
    ia_ = {"acquisition": aq, **({"ledger": ledger} if ledger else {})}
    return ui_evidence.acquisition_notice_line({"contract": {"inline_acquisition": ia_}})


def test_7_notice_stop_clauses_and_legacy_goldens():
    base = {"status": "resource_limited", "stored": 1, "target": 6}
    assert _line(base) == "VERA-DAR obtained 1 new full documents for this question (aim: at least 6); supply was limited."
    assert _line({**base, "stop": "no_progress"}).endswith(
        'stopped after two further searches found nothing new; add "keep searching" to your question to try every '
        "remaining search.")
    assert _line({**base, "stop": "no_progress", "persist": True}) == _line(base)
    assert _line({**base, "stop": "variants_exhausted", "persist": True}).endswith(
        " Every search option available for this question was tried.")
    assert _line({**base, "stop": "variants_exhausted"}) == _line(base)
    assert _line({**base, "stop": "ceiling31"}).endswith(" The per-question reading limit was reached.")
    assert _line({**base, "stop": "rate"}).endswith(" The current search limit was reached; try again later.")
    for x in ("gdelt", "openalex", "http", "R-ACQ"):
        assert x not in _line({**base, "stop": "no_progress"})


def test_8_background_note_never_claims_more_than_the_hand_off():
    a = {}
    persist = {"status": "resource_limited", "persist": True}
    o0 = ia.Outcome("none", acquisition=persist, background={"status": "ACQ-QUEUED", "queued": 0, "already_queued": 0})
    assert ia.background_note(a, o0) == ""
    o3 = ia.Outcome("none", acquisition=persist, background={"status": "ACQ-QUEUED", "queued": 2, "already_queued": 1})
    t = ia.background_note(a, o3)
    assert " 3 source(s) " in t and "will not update" in t
    assert "in progress" not in t and "searching continues" not in t.lower()
    off = ia.Outcome("none", acquisition=persist, background={"status": "R-ACQ-DISABLED", "queued": 0})
    assert ia.background_note(a, off) == ""
    plain = ia.Outcome("none", acquisition={"status": "resource_limited"},
                       background={"status": "ACQ-QUEUED", "queued": 2, "already_queued": 0})
    assert ia.background_note(a, plain) == ia.BACKGROUND_NOTE


def test_9_final_event_why_heuristic(monkeypatch):
    seen = []
    monkeypatch.setattr(ia, "_emit", lambda stage, state, **c: seen.append((stage, state, c)))
    o = ia.Outcome("none", acquisition={"status": "resource_limited", "stored": 0, "target": 6, "stop": "no_progress"})
    ia._emit_final(o, {}, {}, None)
    assert seen[-1][2]["why"] == ia.WHY_HEURISTIC == 6
    ia._emit_final(o, {}, {}, stream_ctx.STALLED)
    assert seen[-1][2]["why"] == ia.WHY_STALLED  # never over a stall


def test_f1_persist_note_claims_exhaustion_only_on_variants_exhausted():
    def o(stop):
        return ia.Outcome("none", acquisition={"status": "resource_limited", "persist": True, "stop": stop},
                          background={"status": "ACQ-QUEUED", "queued": 2, "already_queued": 0})
    assert ia.background_note({}, o("variants_exhausted")).startswith("VERA-DAR has tried every search available")
    for stop, why in (("deadline", "time limit"), ("ceiling31", "reading limit"), ("rate", "search limit"),
                      ("disconnect", "connection ended"), ("search_failed", "could not be completed"),
                      (None, "stopping point")):
        t = ia.background_note({}, o(stop))
        assert "tried every search" not in t and why in t and " 2 source(s) " in t, (stop, t)


def test_f1_search_failure_never_leaves_the_variants_exhausted_default(monkeypatch):
    h = _h(monkeypatch)
    real = h.search

    def search(query, *a, **k):
        if len(h.searches) >= 2:
            h.searches.append({"q": query})
            raise TimeoutError("search timed out")
        return real(query, *a, **k)
    h.deps.search = search
    o = _run(h, cfg(), persist=True)
    assert o.acquisition["stop"] == "search_failed"


def test_f2_plain_heuristic_stop_is_disclosed_on_the_non_answer_path(monkeypatch):
    acq = cfg()
    acq["scope_expansion"]["inline_notice_none"] = "VERA-DAR searched and found nothing it could use."
    o = ia.Outcome("none", candidates=None,
                   acquisition={"status": "weird_status", "stored": 0, "target": 6, "stop": "no_progress",
                                "heuristic": True})
    t = ask_service._inline_notice(acq, o)
    assert t.startswith("VERA-DAR searched and found nothing it could use.")
    assert t.endswith('stopped after two further searches found nothing new; add "keep searching" to your question '
                      "to try every remaining search.")
    # the shortfall line discloses it even with a status that has no status clause
    assert "keep searching" in _line({"status": "weird_status", "stored": 0, "target": 6, "stop": "no_progress"})
    assert _line({"status": "weird_status", "stored": 0, "target": 6}) is None
    o2 = ia.Outcome("none", acquisition={"status": "resource_limited", "stop": "no_progress", "persist": True})
    assert "keep searching" not in (ask_service._inline_notice(acq, o2) or "")
