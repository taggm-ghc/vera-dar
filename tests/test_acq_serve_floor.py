"""FLOOR6-1028: owner 2026-10-28, 6 docs is a floor, not a ceiling. Serving: no per-topic ceiling, topic-balanced
order, only the global totals bind. Acquisition: interim per-topic rounds targeting >= 6 new docs. Offline only."""
import threading

import pytest

from tests.test_acq_serve_per_topic import MT, _Conn, _row, _rows, _rows_full
from vera import acq_config, acq_rounds, fulldoc_db


def _topic(g):
    return "A" if g <= 20 else "B" if g <= 40 else "C" if g <= 58 else f"u{g}"


def _counts(got):
    out = {}
    for g in got:
        out[_topic(g)] = out.get(_topic(g), 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------ serving
def test_shipped_config_has_no_ceiling_balanced_order_totals_unchanged():
    b = acq_config.load_config()["bounds"]
    assert b["served_per_topic_max"] is None and b["served_topic_balanced"] is True
    assert "served_per_topic_min" not in b  # the floor is an acquisition target, not a serving rule
    assert b["served_acquired_max"] == 30 and b["served_acquired_max_bytes"] == 50000000
    assert "FLOOR6-1028" in b["_served_topic_balanced_note"] and "OD-1027(4)" in b["_served_topic_balanced_note"]
    assert fulldoc_db._per_topic_mode({"bounds": b}) == (True, None)


def test_no_ceiling_when_budget_remains():
    got = fulldoc_db.select_served(_rows(), None, 30, MT)
    c = _counts(got)
    assert len(got) == 30 and len(set(got)) == 30
    assert c["A"] + c["B"] + c["C"] == 28 and max(c["A"], c["B"], c["C"]) > 6  # beyond 6: no ceiling
    assert sorted(fulldoc_db.select_served(_rows(), None, 1000, MT)) == list(range(1, 61))  # all served
    assert len(fulldoc_db.select_served(_rows(), 6, 1000, MT)) == 20  # contrast: the A4 ceiling stops at 3x6+2


def test_balanced_order_newest_first_within_topic():
    got = fulldoc_db.select_served(_rows(), None, 30, MT)
    assert got[:5] == [60, 59, 58, 40, 20]  # round 1: every topic once, newest-member topic first
    for ids in (range(1, 21), range(21, 41), range(41, 59)):
        mine = [g for g in got if g in ids]
        assert mine == sorted(ids, reverse=True)[: len(mine)]
    assert max(_counts(got)[t] for t in "ABC") - min(_counts(got)[t] for t in "ABC") <= 1  # depth-balanced


def test_budget_exhaustion_is_deterministic_newest_topics_win():
    rows = [_row(t * 10 + i, f"T{t}") for t in range(1, 6) for i in range(1, 7)]
    got = fulldoc_db.select_served(rows, None, 10, MT)
    assert got == [56, 46, 36, 26, 16, 55, 45, 35, 25, 15]
    assert got == fulldoc_db.select_served(list(reversed(rows)), None, 10, MT)  # input order does not matter
    assert fulldoc_db.select_served(rows, None, 3, MT) == [56, 46, 36]
    assert fulldoc_db.select_served(rows, None, 0, MT) == [] and fulldoc_db.select_served([], None, 5, MT) == []


def test_unkeyed_rows_are_each_their_own_topic():
    rows = [_row(i) for i in range(1, 11)] + [_row(i, "K") for i in range(11, 21)]
    got = fulldoc_db.select_served(rows, None, 16, MT)
    assert set(range(1, 11)) <= set(got) and sum(1 for g in got if g > 10) == 6


def test_legacy_behaviour_unchanged():
    assert fulldoc_db.select_served(_rows(), 3, 4, MT) == [60, 59, 58, 40]  # A4 ceiling when a number is set
    assert fulldoc_db._per_topic_mode({"bounds": {"served_per_topic_max": None}}) == (False, None)  # newest-N
    assert fulldoc_db._per_topic_mode({"bounds": {}}) == (False, None) and fulldoc_db._per_topic_mode({}) == (False, None)
    assert fulldoc_db._per_topic_mode({"bounds": {"served_topic_balanced": "yes"}}) == (False, None)  # strict bool
    assert fulldoc_db._per_topic_mode({"bounds": {"served_per_topic_max": 6, "served_topic_balanced": True}}) == (True, 6)


def test_load_acquired_db_switches(monkeypatch):
    seen = []
    monkeypatch.setattr(fulldoc_db, "fetch_acquired_rows", lambda *a, **k: seen.append(k) or ([], []))
    monkeypatch.setattr(fulldoc_db, "load_acquired_rows", lambda *a, **k: "ok")
    base = {"marker": "m", "micro_topic": MT}
    fulldoc_db.load_acquired_db(None, 1, {}, {**base, "bounds": {"served_acquired_max": 5, "served_per_topic_max": None}})
    fulldoc_db.load_acquired_db(None, 1, {}, {**base, "bounds": {"served_acquired_max": 5, "served_per_topic_max": None,
                                                                "served_topic_balanced": True}})
    assert seen[0] == {}
    assert seen[1]["per_topic_max"] is None and seen[1]["topic_balanced"] is True


def test_fetch_path_serves_balanced_without_ceiling():
    log = []
    conn = _Conn(_rows_full(), log)
    s, _ = fulldoc_db.fetch_acquired_rows(lambda: conn, 999, "m", 30, per_topic_max=None, topic_balanced=True,
                                          scan_max=2000, micro_cfg=MT)
    assert fulldoc_db.SQL_ACQ_INDEX in log and fulldoc_db.SQL_ACQ_SOURCES not in log
    assert sorted(x["source_id"] for x in s) == sorted(fulldoc_db.select_served(_rows(), None, 30, MT))
    log2 = []
    s2, _ = fulldoc_db.fetch_acquired_rows(lambda: _Conn(_rows_full(), log2), 999, "m", 5)  # legacy newest-N
    assert log2[0] == fulldoc_db.SQL_ACQ_SOURCES and [x["source_id"] for x in s2] == [60, 59, 58, 57, 56]


# ------------------------------------------------------------------------------------- acquisition rounds (interim)
def test_shipped_round_target_is_six_and_job_caps_unchanged():
    b = acq_config.load_config()["bounds"]
    assert b["acq_round_target_new_docs"] == 6 and acq_rounds.target_of({"bounds": b}) == 6
    assert (b["jobs_per_hour"], b["jobs_per_day"], b["max_docs_per_job"]) == (6, 30, 2)
    assert "FLOOR6-1028" in b["_acq_round_target_note"] and "in-process" in b["_acq_round_target_note"]
    assert acq_rounds.target_of({}) == 6 and acq_rounds.target_of({"bounds": {"acq_round_target_new_docs": True}}) == 6


def test_round_spans_jobs_and_is_met_at_six():
    t = acq_rounds.RoundTracker()
    assert t.record_job("k", 2, False)["outcome"] == "open"
    assert t.record_job("k", 2, False, path="inline")["outcome"] == "open"
    r = t.record_job("k", 2, False)
    assert r["outcome"] == "met" and r["new_docs"] == 6 and r["jobs"] == 3 and r["paths"] == {"background": 2, "inline": 1}
    r2 = t.record_job("k", 1, False)  # a later round of the same topic opens fresh, again targeting 6
    assert r2["round_id"] != r["round_id"] and r2["new_docs"] == 1 and r2["outcome"] == "open"


def test_resource_limited_is_not_failure_and_closes_round():
    t = acq_rounds.RoundTracker()
    t.record_job("k", 2, False)
    r = t.record_job("k", 1, True)
    assert r["outcome"] == "resource_limited" and r["new_docs"] == 3
    assert t.open_round("k") is None and [c["outcome"] for c in t.closed()] == ["resource_limited"]
    assert t.record_job("k", 0, True)["outcome"] == "resource_limited"  # zero found: still a resource limit
    assert t.record_job("j", 6, True)["outcome"] == "met"  # target reached wins over exhaustion


def test_rounds_are_per_topic_and_thread_safe():
    t = acq_rounds.RoundTracker()
    th = [threading.Thread(target=lambda: t.record_job("a", 1, False)) for _ in range(5)]
    [x.start() for x in th]
    [x.join() for x in th]
    assert t.open_round("a")["new_docs"] == 5 and t.open_round("b") is None
    with pytest.raises(ValueError):
        t.record_job("", 1, False)


def test_job_exhausted_reads_candidate_outcomes():
    assert acq_rounds.job_exhausted([{"outcome": "admitted"}, {"outcome": "rejected"}])
    assert acq_rounds.job_exhausted([]) and acq_rounds.job_exhausted(None)
    assert not acq_rounds.job_exhausted([{"outcome": "not_evaluated"}])
    assert not acq_rounds.job_exhausted([{"outcome": "admitted"}, {"outcome": "not_used"}])
