"""MIDTERMS-WP10-1106 (owner OD-1106-MIDTERMS-D1D4, D3): the examined-source ledger. Only qualifying full- or
partial-text sources actually examined count toward the six-source target; duplicates, metadata-only and merely
discovered results never count. Offline: pure data, stubs only."""
from __future__ import annotations

import json
from types import SimpleNamespace

from vera import ask_service, evidence_ledger as el, inline_acquire as ia, request_trace, ui_evidence

DOI = "10.1234/abc.5"


def _add(led, ids, kind="scholarly", depth="full_text", q=True, **k):
    return led.add(ids=ids, kind=kind, depth=depth, qualified=q, origin=k.pop("origin", "inline"), **k)


def test_1_dedup_doi_url_fragment_tracking_and_upgrade():
    led = el.Ledger()
    assert _add(led, {"doi": DOI})
    assert not _add(led, {"url": f"https://doi.org/{DOI}"})
    _add(led, {"url": "http://News.Example.org/a/b/?utm_source=x&id=1#top"}, kind="news")
    assert not _add(led, {"url": "https://news.example.org/a/b?id=1&fbclid=z"}, kind="news")
    assert led.counted() == 2 and led.duplicates == 2
    led2 = el.Ledger()
    _add(led2, {"url": "https://x.example/p"}, kind="web", q=False)
    assert led2.counted() == 0
    _add(led2, {"url": "https://x.example/p/"}, kind="web", q=True)  # a later qualified duplicate upgrades
    assert led2.counted() == 1 and led2.summary()["duplicates"] == 1


def test_2_depth_rules():
    led = el.Ledger()
    _add(led, {"doi": "10.1/a"})
    _add(led, {"doi": "10.1/b"}, depth="partial_text")
    _add(led, {"doi": "10.1/c"}, depth="metadata_only", q=True)
    _add(led, {"doi": "10.1/d"}, q=False)
    s = led.summary()
    assert s["counted"] == 2 and s["full_text"] == 1 and s["partial_text"] == 1
    assert s["metadata_only"] == 1 and s["examined_unqualified"] == 1


def test_3_kind_of_and_count_kinds():
    kw = {"outlet_hosts": {"news.example"}, "official_hosts": {"gov.example"}}
    assert el.kind_of("https://a.example/x", "reporting", **kw) == "news"
    assert el.kind_of("https://a.example/x", "official", **kw) == "official"
    assert el.kind_of("https://news.example/x", "web", **kw) == "news"
    assert el.kind_of("https://gov.example/x", "web", **kw) == "official"
    assert el.kind_of("https://other.example/x", "web", **kw) == "web"
    assert el.kind_of("https://other.example/x", None, **kw) == "scholarly"
    acq = {"bounds": {"acq_round_count_kinds": ["scholarly", "news", "official"]}}
    led = el.Ledger(el.count_kinds(acq))
    _add(led, {"url": "https://w.example/1"}, kind="web")
    _add(led, {"url": "https://n.example/1"}, kind="news")
    assert led.counted() == 1 and el.count_kinds({}) == el.KINDS


def _page(url, lane="reporting"):
    return SimpleNamespace(url=url, lane=lane, title="t", text="x", published=None)


def _news_case():
    exam = tuple({"document_id": f"lane-reporting-{i}", "url": f"https://n{i}.example/a", "doi": None,
                  "canonical_source_id": f"https://n{i}.example/a", "admitted_spans": 2, "lane": "reporting"}
                 for i in range(2))
    ev = SimpleNamespace(examined_full_documents=exam)
    pages = [_page("https://n0.example/a"), _page("https://n1.example/a"), _page("https://n2.example/a")]
    links = [{"lane": "reporting", "url": f"https://l{i}.example/x", "read": False} for i in range(3)]
    lanes = SimpleNamespace(pages=pages, links=links)
    o = ia.Outcome("none", candidates=[], acquisition={"status": "resource_limited", "stored": 0, "target": 6})
    return o, ev, lanes


def test_4_news_question_counts_examined_reporting_pages_not_zero():
    o, ev, lanes = _news_case()
    led = el.from_inline(o, ev=ev, lanes=lanes, acq={}, host_lists=((), ()))
    s = led.summary()
    assert s["counted"] == 2 and s["examined_unqualified"] == 1 and s["metadata_only"] == 3 and s["news"] == 2
    pub = ask_service._with_ledger(o.public(), led, {})
    line = ui_evidence.acquisition_notice_line({"contract": {"inline_acquisition": pub}})
    assert line.startswith("VERA-DAR obtained 2 documents for this question (aim: at least 6);") and "0 of 6" not in line


def test_5_candidates_and_discovered_only_count_zero():
    recs = [{"doi": "10.1/r", "outcome": "rejected", "reason": "R-ACQ-IRRELEVANT"},
            {"doi": "10.1/n", "outcome": "rejected", "reason": "R-ACQ-NOLIC"},
            {"doi": "10.1/d", "outcome": "dropped", "reason": "R-ACQ-DEFER_CAP"}]
    o = ia.Outcome("none", candidates=recs, discovered=[{"url": "https://d.example/1"}, {"url": None}])
    s = el.from_inline(o, acq={}).summary()
    assert s["counted"] == 0 and s["examined_unqualified"] == 1 and s["metadata_only"] == 4


def test_5b_admitted_candidates_count_only_without_answer_time_evidence():
    recs = [{"doi": "10.1/a", "outcome": "admitted", "persisted": True}]
    o = ia.Outcome("ok", candidates=recs)
    assert el.from_inline(o, acq={}).summary()["stored"] == 1
    ev = SimpleNamespace(examined_full_documents=({"doi": "10.1/a", "url": None, "admitted_spans": 0},))
    s = el.from_inline(o, ev=ev, acq={}).summary()
    assert s["counted"] == 0 and s["examined_unqualified"] == 1  # answer-time admission decides, no upgrade


def test_6_acquire_rounds_dedups_across_rounds_and_targets_on_counted(monkeypatch):
    calls = {"n": 0}

    def fake_inline(question, model, pricing, decision, chained, grounding, acq, tried=None, in_scope=False, info=None):
        calls["n"] += 1
        led = el.Ledger()
        _add(led, {"doi": DOI})
        info["_ledger"] = led
        assert calls["n"] < 10, "unbounded loop"
        info["tried_keys"] = {f"k{calls['n']}"} if calls["n"] < 3 else set()  # round 3 adds nothing new: stop
        info["outcome"] = {"status": "none", "acquisition": {"status": "incomplete_deadline", "stored": 0, "target": 6}}
        return "notice"
    monkeypatch.setattr(ask_service, "_inline", fake_inline)
    monkeypatch.setattr(ask_service, "_retry_tokens", lambda *a: True)
    monkeypatch.setattr(ask_service, "_rt_event", lambda *a, **k: None)
    monkeypatch.setattr(ask_service, "inline_need_s", lambda acq, g: 0.0)
    monkeypatch.setattr(ia, "icfg_of", lambda acq: {"in_scope": True})
    budget = SimpleNamespace(inline_used=False, deadline=float("inf"))
    _note, out = ask_service._acquire_rounds("q", "m", None, SimpleNamespace(tokens_used=0, cost_usd=0.0), {}, None,
                                             {"inline": {"deadline_s": 45}}, budget, {}, None, False)
    assert calls["n"] >= 2 and out["ledger"]["counted"] == 1 and out["acquisition"]["counted"] == 1


def test_7_notice_ledger_goldens_and_legacy():
    def line(led, aq=None):
        aq = aq or {"status": "resource_limited", "stored": 1, "target": 6}
        return ui_evidence.acquisition_notice_line({"contract": {"inline_acquisition": {"acquisition": aq,
                                                                                       **({"ledger": led} if led else {})}}})
    z = {k: 0 for k in el.SUMMARY_KEYS}
    assert line({**z, "counted": 3, "full_text": 3, "scholarly": 3}) == "VERA-DAR obtained 3 documents for this question (aim: at least 6); supply was limited."
    assert line({**z, "counted": 3, "full_text": 2, "partial_text": 1, "scholarly": 3, "stored": 1}) == (
        "VERA-DAR obtained 3 documents for this question (aim: at least 6) (2 in full, 1 in part), 1 saved to VERA-DAR's collection; supply was limited.")
    assert line({**z, "counted": 3, "full_text": 3, "scholarly": 1, "news": 2}) == (
        "VERA-DAR obtained 3 documents for this question (aim: at least 6) (scholarly 1, news 2); supply was limited.")
    assert line({**z, "counted": 6, "full_text": 6, "news": 6}) is None
    assert line(None) == "VERA-DAR obtained 1 new full document for this question (aim: at least 6); supply was limited."
    assert line({"counted": "x"}) == "VERA-DAR obtained 1 new full document for this question (aim: at least 6); supply was limited."


def test_8_need_to_know_public_summary_and_trace():
    o, ev, lanes = _news_case()
    led = el.from_inline(o, ev=ev, lanes=lanes, acq={}, host_lists=((), ()))
    blob = json.dumps(ask_service._with_ledger(o.public(), led, {})["ledger"])
    for bad in ("http", "10.", "example", "lane-"):
        assert bad not in blob
    assert set(request_trace.clean_counts(led.summary())) == set(el.SUMMARY_KEYS)


def test_9_status_met_when_counted_reaches_target_without_mutating_outcome():
    led = el.Ledger()
    for i in range(6):
        _add(led, {"url": f"https://n{i}.example/a"}, kind="news")
    o = ia.Outcome("none", acquisition={"status": "resource_limited", "stored": 0, "target": 6})
    pub = ask_service._with_ledger(o.public(), led, {})
    assert pub["acquisition"]["status"] == "met" and o.acquisition["status"] == "resource_limited"
    assert ui_evidence.acquisition_notice_line({"contract": {"inline_acquisition": pub}}) is None


def test_10_progress_bar_read_replaces_saved_once_counted_arrives():
    from vera import progress, progress_ui
    assert "counted" in progress.COUNT_KEYS
    labels = progress_ui.Labels.load() if hasattr(progress_ui.Labels, "load") else None
    holder = SimpleNamespace(events=[{"stage": "acquire", "state": "update", "counts": {"stored": 1, "target": 6}}],
                             last_line_at=None, t0=0.0, activity=None)
    lab = labels or SimpleNamespace(stage_label=lambda *a: None, activity_label=lambda a: None)
    assert progress_ui.stream_view(holder, lab, 1.0)["bars"][0][0] == "Saved 1 of 6"
    holder.events.append({"stage": "acquire", "state": "update", "counts": {"counted": 2, "target": 6}})
    assert [b[0] for b in progress_ui.stream_view(holder, lab, 1.0)["bars"]] == ["Read 2 of 6"]
    assert ia.acq_config is not None and el.count_kinds(json.load(open(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "config/auto_acquire.json"))) == el.KINDS
    cfg = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[1] / "config/auto_acquire.json"))
    assert cfg["scope_expansion"]["inline_notice_background_persist"] == ia.BACKGROUND_NOTE_PERSIST


def test_f3_keyless_rows_never_count_and_m2_linked_keys():
    led = el.Ledger()
    _add(led, {})
    _add(led, {"title": "no identifier"})
    assert led.counted() == 0 and len(led.entries()) == 2
    o = ia.Outcome("ok", candidates=[{"title": "x", "outcome": "admitted"}])
    assert el.from_inline(o, acq={}).counted() == 0
    r1, r2 = el.Ledger(), el.Ledger()
    _add(r1, {"doi": DOI, "url": "https://pub.example/a"})
    _add(r2, {"url": "https://pub.example/a/"})  # a later URL-only sighting of a record that carried both
    assert r1.merge(r2).counted() == 1
