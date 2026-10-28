"""ACQ S4-0 L9: per-micro-topic serving (offline, fake rows and fake DB-API connection)."""
import copy
import json
import time

from tests import acq_support as A
from vera import acq_config, fulldoc_db

MT = copy.deepcopy(acq_config.load_config()["micro_topic"])


def _row(sid, topic=None):
    p = {"acquisition": {"marker": "m"}}
    if topic:
        p["acquisition"]["micro_topic"] = [{"scheme": "openalex_topic", "id": topic, "score": 1.0}]
    return {"source_id": sid, "run_id": sid, "provenance": json.dumps(p)}


def _rows():
    # topics: A = ids 1..20 (oldest), B = 21..40, C = 41..58, unkeyed 59, 60 (newest)
    return ([_row(i, "A") for i in range(1, 21)] + [_row(i, "B") for i in range(21, 41)]
            + [_row(i, "C") for i in range(41, 59)] + [_row(59), _row(60)])


def test_older_topics_served_and_per_topic_cap():
    got = fulldoc_db.select_served(_rows(), 3, 30, MT)
    assert len(got) == 3 * 3 + 2 and len(set(got)) == len(got)
    assert sum(1 for g in got if g <= 20) == 3 and not {1, 2, 3} & set(got)  # topic A: only its newest three
    for lo, hi in ((1, 20), (21, 40), (41, 58)):
        assert sum(1 for g in got if lo <= g <= hi) == 3
    assert {59, 60} <= set(got)  # unkeyed rows are their own topics
    assert sorted((g for g in got if g <= 20), reverse=True) == [20, 19, 18]  # newest first within a topic
    assert got[0] == 60  # topics ordered by newest member first (round 1)


def test_total_cap_and_round_robin():
    got = fulldoc_db.select_served(_rows(), 3, 4, MT)
    assert got == [60, 59, 58, 40]
    assert fulldoc_db.select_served(_rows(), 3, 0, MT) == [] and fulldoc_db.select_served([], 3, 5, MT) == []


def test_legacy_newest_n_unchanged_when_null():
    plans = None
    log = []
    conn = _Conn(_rows_full(), log)
    s, _ = fulldoc_db.fetch_acquired_rows(lambda: conn, 999, "m", 5)
    assert log[0] == fulldoc_db.SQL_ACQ_SOURCES and [x["source_id"] for x in s] == [60, 59, 58, 57, 56]
    s2, _ = fulldoc_db.fetch_acquired_rows(lambda: conn, 999, "m", 5, per_topic_max=3, scan_max=2000, micro_cfg=MT)
    assert fulldoc_db.SQL_ACQ_INDEX in log and [x["source_id"] for x in s2] == [60, 59, 58, 40, 20]
    assert len(s2) <= 5  # rebuild-size bound: never more than the total cap


def test_load_acquired_db_uses_config_switch(monkeypatch):
    seen = []
    monkeypatch.setattr(fulldoc_db, "fetch_acquired_rows", lambda *a, **k: seen.append(k) or ([], []))
    monkeypatch.setattr(fulldoc_db, "load_acquired_rows", lambda *a, **k: "ok")
    base = {"marker": "m", "micro_topic": MT}
    fulldoc_db.load_acquired_db(None, 1, {}, {**base, "bounds": {"served_acquired_max": 5}})
    fulldoc_db.load_acquired_db(None, 1, {}, {**base, "bounds": {"served_acquired_max": 5, "served_per_topic_max": None}})
    fulldoc_db.load_acquired_db(None, 1, {}, {**base, "bounds": {"served_acquired_max": 5, "served_per_topic_max": 3,
                                                                "served_scan_max": 77}})
    assert seen[0] == {} and seen[1] == {}
    assert seen[2]["per_topic_max"] == 3 and seen[2]["scan_max"] == 77


def test_selection_over_2000_rows_is_fast():
    idx = [_row(i, f"T{i % 40}") for i in range(1, 2001)]
    t = time.perf_counter()
    got = fulldoc_db.select_served(idx, 3, 30, MT)
    assert time.perf_counter() - t < 0.5
    assert len(got) == 30


def test_load_acquired_rows_orders_by_topic_when_set():
    src = [{"source_id": i, "run_id": i, "provenance": _row(i, "A" if i < 10 else "B")["provenance"]} for i in range(1, 20)]
    picked = fulldoc_db.select_served(src, 2, 3, MT)
    assert picked == [19, 9, 18]


def _rows_full():
    out = []
    for r in _rows():
        out.append({**r, "canonical_url": f"u{r['source_id']}", "content": "c", "content_hash": "h"})
    return out


class _Conn:
    def __init__(self, rows, log):
        self.rows, self.log, self._r = rows, log, []

    def cursor(self):
        return self

    def close(self):
        pass

    def fetchall(self):
        return list(self._r)

    def execute(self, sql, params):
        self.log.append(sql)
        rs = sorted(self.rows, key=lambda r: -r["source_id"])
        full = lambda r: (r["run_id"], r["source_id"], r["canonical_url"], r["content"], r["content_hash"], r["provenance"])
        if sql == fulldoc_db.SQL_ACQ_SOURCES:
            self._r = [full(r) for r in rs][: params["lim"]]
        elif sql == fulldoc_db.SQL_ACQ_INDEX:
            self._r = [(r["source_id"], r["run_id"], r["provenance"]) for r in rs][: params["scan"]]
        elif sql == fulldoc_db.SQL_ACQ_SOURCES_BY_ID:
            self._r = [full(r) for r in rs if r["source_id"] in set(params["sids"])]
        else:
            self._r = []


# D1: production-shaped rows: content "" and the text only in evidence_spans; keys must come from title + first span.
EPI = ("epidemiology infectious disease outbreak transmission incidence prevalence surveillance cohort pathogen "
       "infection epidemic vaccination mortality")


def _prod_prov(title):
    return json.dumps({"title": title, "acquisition": {"marker": "m"}})


def test_select_served_keys_from_first_span_when_content_empty():
    rows = [{"source_id": i, "provenance": _prod_prov("Qzxv"), "content": ""} for i in range(1, 7)]
    with_text = [{**r, "content": EPI} for r in rows[:3]]
    keyed = fulldoc_db.select_served(with_text + rows[3:], 1, 10, MT)
    assert len(keyed) == 4 and 3 in keyed and 1 not in keyed and 2 not in keyed  # 1 keyed topic + 3 unkeyed rows
    assert len(fulldoc_db.select_served(rows, 1, 10, MT)) == 6  # no text: every row its own topic
    assert len(fulldoc_db.select_served(with_text, 1, 10, MT)) == 1  # same text => same micro-topic


def test_load_acquired_rows_keys_from_spans_when_content_empty():
    srcs = [{"source_id": i, "run_id": i, "provenance": _prod_prov("Qzxv"), "content": ""} for i in range(1, 7)]
    spans = ([{"source_id": i, "start_index": 10, "end_index": 20, "text": "zzz"} for i in range(1, 7)]
             + [{"source_id": i, "start_index": 0, "end_index": 10, "text": EPI} for i in range(1, 4)])
    first = fulldoc_db._first_span_texts(spans)
    assert first[1] == EPI and first[5] == "zzz"
    pick = fulldoc_db.select_served([{**s, "content": s["content"] or first.get(s["source_id"])} for s in srcs], 1, 10, MT)
    assert 3 in pick and 2 not in pick and 1 not in pick and len(pick) == 4


def test_per_topic_index_fetches_first_span_text_with_guarded_select():
    log = []

    class C(_Conn):
        def execute(self, sql, params):
            if sql == fulldoc_db.SQL_ACQ_FIRST_SPANS:
                self.log.append(sql)
                assert "%(ids)s" in sql and "%(win)s" in sql and sql.lstrip().upper().startswith("SELECT")
                self._r = [(i, EPI) for i in params["ids"] if i <= 3]
                return
            super().execute(sql, params)
    rows = [{"source_id": i, "run_id": i, "provenance": _prod_prov("Qzxv"), "canonical_url": f"u{i}", "content": "",
             "content_hash": "h"} for i in range(1, 7)]
    conn = C(rows, log)
    ids = fulldoc_db.acquired_ids_per_topic(conn, conn, 999, "m", 1, 2000, 10, MT)
    assert fulldoc_db.SQL_ACQ_FIRST_SPANS in log
    assert 3 in ids and 1 not in ids and 2 not in ids and len(ids) == 4
    assert fulldoc_db.SQL_ACQ_FIRST_SPANS.count("%(") == 2 and "{" not in fulldoc_db.SQL_ACQ_FIRST_SPANS


def test_first_span_failure_rolls_back_so_later_sql_still_works():
    state = {"aborted": False, "rollbacks": 0}

    class A(_Conn):
        def execute(self, sql, params):
            if sql == fulldoc_db.SQL_ACQ_FIRST_SPANS:
                state["aborted"] = True
                raise RuntimeError("boom")
            if state["aborted"]:
                raise RuntimeError("current transaction is aborted")
            super().execute(sql, params)

        def rollback(self):
            state["rollbacks"] += 1
            state["aborted"] = False
    rows = [{"source_id": i, "run_id": i, "provenance": _prod_prov("Qzxv"), "canonical_url": f"u{i}", "content": "",
             "content_hash": "h"} for i in range(1, 4)]
    conn = A(rows, [])
    ids = fulldoc_db.acquired_ids_per_topic(conn, conn, 999, "m", 1, 2000, 10, MT)
    assert state["rollbacks"] == 1 and len(ids) == 3
    conn.execute(fulldoc_db.SQL_ACQ_SOURCES_BY_ID, {"sids": [1]})  # post-fallback SQL accepted


def test_first_span_failure_with_failing_rollback_is_still_soft():
    class B(_Conn):
        def execute(self, sql, params):
            if sql == fulldoc_db.SQL_ACQ_FIRST_SPANS:
                raise RuntimeError("boom")
            super().execute(sql, params)

        def rollback(self):
            raise RuntimeError("rollback failed")
    rows = [{"source_id": i, "run_id": i, "provenance": _prod_prov("Qzxv"), "canonical_url": f"u{i}", "content": "",
             "content_hash": "h"} for i in range(1, 4)]
    conn = B(rows, [])
    assert len(fulldoc_db.acquired_ids_per_topic(conn, conn, 999, "m", 1, 2000, 10, MT)) == 3


# CAP6-1028: OD-1027(4) "at least 6 docs per micro-topic" -> shipped served_per_topic_max is 6 (added tests; none edited)
def test_shipped_per_topic_cap_is_six_and_totals_unchanged():
    b = acq_config.load_config()["bounds"]
    assert b["served_per_topic_max"] == 6
    assert b["served_acquired_max"] == 30 and b["served_acquired_max_bytes"] == 50000000
    assert "OD-1027(4)" in b["_served_per_topic_note"]


def test_six_per_topic_serves_six_and_respects_total_cap():
    got = fulldoc_db.select_served(_rows(), 6, 30, MT)
    for lo, hi in ((1, 20), (21, 40), (41, 58)):
        assert sum(1 for g in got if lo <= g <= hi) == 6
    assert {59, 60} <= set(got) and len(got) == 3 * 6 + 2 and len(set(got)) == len(got)
    assert len(fulldoc_db.select_served(_rows(), 6, 10, MT)) == 10  # total cap still binds
