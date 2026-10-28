"""ROUNDS-1028: the 6-document acquisition round tracker wired into the inline and background acquisition paths
(observational; in-process, DV-ACQ-FLOOR6-1). OFFLINE: fakes only, no network, no DB, no model."""
import json
import logging

import pytest

from tests import acq_support as A
from tests import fulltext_synth
from vera import acq_config, acq_rounds, auto_acquire, inline_acquire

DOIS = [f"10.9999/acq.rounds{i}" for i in range(8)]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq_rounds.reset_for_tests()
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq_rounds.reset_for_tests()


def _web(dois):
    recs, jats = {}, {}
    for i, d in enumerate(dois):
        pm = f"PMC77{i:05d}"
        recs[d] = A.epmc_record(d, pmcid=pm)
        jats[pm] = fulltext_synth.jats(f"acq_rounds{i}", "epi", d)
    return A.Web(recs, jats)


def _bg(dois, cfg=None, question=A.Q_EPI):
    d = A.deps([A.cand(x) for x in dois], _web(dois))
    return auto_acquire.run_job(A.job(question), cfg or A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)


def _key(question=A.Q_EPI):
    return acq_rounds.topic_key(question, A.cfg())


# ------------------------------------------------------------------------------------------------ lifecycle
def test_three_background_jobs_of_two_documents_meet_the_round():
    outs = [_bg(DOIS[2 * i:2 * i + 2]) for i in range(3)]
    assert [o["stored"] for o in outs] == [2, 2, 2]
    closed = acq_rounds.tracker().closed()
    assert len(closed) == 1
    r = closed[0]
    assert r["outcome"] == "met" and r["new_docs"] == 6 and r["jobs"] == 3 and r["target"] == 6
    assert r["topic_key"] == _key() and r["paths"] == {"background": 3}
    assert acq_rounds.tracker().open_round(_key()) is None


def test_round_stays_open_while_jobs_fill_their_document_budget():
    _bg(DOIS[0:2])
    r = acq_rounds.tracker().open_round(_key())
    assert r["outcome"] == "open" and r["new_docs"] == 2 and r["jobs"] == 1  # budget filled: not "routes exhausted"


def test_fewer_documents_with_routes_exhausted_is_resource_limited():
    _bg(DOIS[0:2])
    out = _bg(DOIS[2:3])  # one candidate only: every route used, 1 stored
    assert out["stored"] == 1
    r = acq_rounds.tracker().closed()[0]
    assert r["outcome"] == "resource_limited" and r["new_docs"] == 3 and r["jobs"] == 2


def test_empty_search_is_resource_limited_zero_not_an_error():
    out = _bg([])
    assert out["stored"] == 0
    assert acq_rounds.tracker().closed()[0]["outcome"] == "resource_limited"


def test_inline_and_background_jobs_share_one_round_per_topic():
    class Rig:
        pass
    d = A.deps([A.cand(DOIS[0]), A.cand(DOIS[1])], _web(DOIS[0:2]))
    o = inline_acquire.acquire(A.Q_EPI, A.cfg(), env=A.ENV_ON, deps_factory=lambda a, e: d)
    assert o.status == "ok" and o.stored == 2
    _bg(DOIS[2:4])
    _bg(DOIS[4:6])
    r = acq_rounds.tracker().closed()[0]
    assert r["outcome"] == "met" and r["new_docs"] == 6 and r["paths"] == {"inline": 1, "background": 2}


def test_inline_few_documents_exhausted_is_resource_limited():
    d = A.deps([A.cand(DOIS[0])], _web(DOIS[0:1]))
    o = inline_acquire.acquire(A.Q_EPI, A.cfg(), env=A.ENV_ON, deps_factory=lambda a, e: d)
    assert o.stored == 1
    assert acq_rounds.tracker().closed()[0]["outcome"] == "resource_limited"


def test_topic_key_registry_anchor_catstep_hint_and_unassigned():
    cfg = A.cfg()
    anchors = json.load(open("config/micro_topic_anchors.json", encoding="utf-8"))["entries"]
    q = anchors[0]["anchor_text"]
    k = acq_rounds.topic_key(q, cfg)
    assert k != acq_rounds.UNASSIGNED and ":" in k
    assert acq_rounds.topic_key("zzzz qqqq xxxx", cfg) == "unassigned"
    assert acq_rounds.topic_key("zzzz", cfg, hint="field:economics") == "field:economics"  # catstep key wins
    assert acq_rounds.topic_key("zzzz", cfg, hint="bad key with spaces!") == "unassigned"


# ------------------------------------------------------------------------------------------------ bound and eviction
def test_open_rounds_are_bounded_and_oldest_is_evicted():
    t = acq_rounds.RoundTracker(max_open=3)
    for k in ("a", "b", "c"):
        t.record_job(k, 1, False)
    t.record_job("a", 1, False)  # a is now the most recently updated; b the oldest
    t.record_job("d", 1, False)
    assert t.open_count() == 3 and t.evicted == 1
    assert t.open_round("b") is None and t.open_round("a")["new_docs"] == 2
    for i in range(50):
        t.record_job(f"k{i}", 1, False)
    assert t.open_count() == 3 and t.evicted == 1 + 50


def test_max_open_comes_from_config_and_defaults():
    assert acq_config.load_config()["bounds"]["acq_round_max_open"] == 200
    assert acq_rounds.max_open_of(A.cfg(acq_round_max_open=5)) == 5
    assert acq_rounds.max_open_of({}) == acq_rounds.DEFAULT_MAX_OPEN
    assert acq_rounds.max_open_of({"bounds": {"acq_round_max_open": True}}) == acq_rounds.DEFAULT_MAX_OPEN
    acq_rounds.reset_for_tests()
    assert acq_rounds.tracker(A.cfg(acq_round_max_open=7))._max_open == 7


# ------------------------------------------------------------------------------------------------ no behaviour change
def test_caps_and_results_unchanged_by_tracking(monkeypatch):
    b = acq_config.load_config()["bounds"]
    assert (b["jobs_per_hour"], b["jobs_per_day"], b["max_docs_per_job"], b["queue_max"]) == (6, 30, 2, 4)
    with_t = [_bg(DOIS[2 * i:2 * i + 2]) for i in range(3)]
    state_with = (len(auto_acquire._S.docs_day), len(auto_acquire._S.jobs_hour), auto_acquire._S.docs_reserved)
    auto_acquire.reset_for_tests()
    monkeypatch.setattr(acq_rounds, "tracker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    without = [_bg(DOIS[2 * i:2 * i + 2]) for i in range(3)]  # a raising observer cannot break a job
    state_without = (len(auto_acquire._S.docs_day), len(auto_acquire._S.jobs_hour), auto_acquire._S.docs_reserved)
    strip = lambda o: {k: v for k, v in o.items() if k != "durations_ms"}  # noqa: E731
    assert [strip(o) for o in with_t] == [strip(o) for o in without]
    assert state_with == state_without == (6, 0, 0)


def test_the_two_documents_per_job_cap_still_binds_with_many_candidates():
    out = _bg(DOIS[0:5])
    assert out["stored"] == 2  # max_docs_per_job
    assert acq_rounds.tracker().open_round(_key())["new_docs"] == 2


def test_tracker_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(acq_rounds, "tracker", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    assert acq_rounds.observe("q", A.cfg(), 1, False, "inline") is None


# ------------------------------------------------------------------------------------------------ logging
def test_round_log_line_has_class_tokens_and_no_raw_text(caplog):
    caplog.set_level(logging.INFO)
    _bg(DOIS[0:2])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("autoacq round ")]
    assert len(lines) == 1
    ln = lines[0]
    assert f"topic={_key()}" in ln and "new_docs=2" in ln and "jobs=1" in ln and "target=6" in ln
    assert "outcome=open" in ln and "path=background" in ln and "exhausted=false" in ln
    assert A.SENTINEL not in ln and "How was" not in ln and "transmission" not in ln
    assert A.SENTINEL not in caplog.text
