"""FRESH-AMEND-1023: corpus_added_at backfill (owner refinement 2026-10-23). Fake clock, fake store; no DB, no network."""
import threading
from datetime import datetime, timedelta, timezone

from vera import corpus_stamp
from vera import research_record as rr

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
CFG = {"enabled": True, "freshness_max_age_hours": 24, "freshness_future_skew_seconds": 300}


class FakeStore:
    """Models SQL_STAMP: a conditional single-row UPDATE (only NULL / future created_at), atomic per row."""

    def __init__(self, rows):
        self.rows, self.calls, self.lock = dict(rows), [], threading.Lock()

    def stamp(self, span_id, now, skew_s):
        with self.lock:
            self.calls.append(span_id)
            cur = self.rows.get(span_id, "absent")
            if cur == "absent":
                return 0
            if cur is None or cur > now + timedelta(seconds=skew_s):
                self.rows[span_id] = now
                return 1
            return 0


def _adm(stamps):
    return [{"document_id": "d1", "published": "2019-05-01", "corpus_chunks": stamps}]


def test_missing_stamp_triggers_and_is_set_to_now():
    st = {"span_id": 7, "corpus_added_at": None, "basis": "chunk"}
    fr = rr.assess(_adm([st]), CFG, NOW)
    assert rr.triggers(True, fr) == ["freshness_unknown"]
    store = FakeStore({7: None})
    bf = rr.apply_backfill(fr, CFG, NOW, {}, stamper=store)
    assert st["corpus_added_at"] == NOW.isoformat() and bf["count"] == 1 and bf["chunk_ids"] == [7]
    assert bf["persist"] == {"status": "done", "updated": 1, "not_persistable": 0, "reason": ""}
    assert store.rows[7] == NOW and store.calls == [7]  # exactly that chunk, one row
    assert rr.triggers(True, rr.assess(_adm([st]), CFG, NOW + timedelta(hours=23))) == []  # next request within 24 h


def test_invalid_future_and_partial_stamps_are_revised():
    for v in ("garbage", (NOW + timedelta(days=3)).isoformat(), "2026", "2026-09", "2026-10-23"):
        st = {"span_id": 8, "corpus_added_at": v, "basis": "chunk"}
        fr = rr.assess(_adm([st]), CFG, NOW)
        assert rr.triggers(True, fr) == ["freshness_unknown"], v
        future = NOW + timedelta(days=3)
        store = FakeStore({8: future if "T" in v else None})
        bf = rr.apply_backfill(fr, CFG, NOW, {}, stamper=store)
        assert st["corpus_added_at"] == NOW.isoformat() and bf["count"] == 1, v
        assert store.rows[8] == NOW, v


def test_valid_value_never_overwritten():
    t = (NOW - timedelta(hours=30)).isoformat()
    good = {"span_id": 9, "corpus_added_at": t, "basis": "chunk"}
    bad = {"span_id": 10, "corpus_added_at": None, "basis": "chunk"}
    fr = rr.assess(_adm([good, bad]), CFG, NOW)
    store = FakeStore({9: NOW - timedelta(hours=30), 10: None})
    bf = rr.apply_backfill(fr, CFG, NOW, {}, stamper=store)
    assert good["corpus_added_at"] == t and store.rows[9] == NOW - timedelta(hours=30)
    assert bf["chunk_ids"] == [10] and store.calls == [10]
    # a stamp that became valid after assessment (another request) is not overwritten either
    late = {"span_id": 11, "corpus_added_at": None}
    fr = rr.assess(_adm([late]), CFG, NOW)
    late["corpus_added_at"] = (NOW - timedelta(hours=1)).isoformat()
    assert rr.apply_backfill(fr, CFG, NOW, {}, stamper=FakeStore({}))["count"] == 0
    assert late["corpus_added_at"] == (NOW - timedelta(hours=1)).isoformat()


def test_no_wo_url_in_memory_only_persist_skipped_with_reason(monkeypatch):
    monkeypatch.delenv("VERA_DB_URL_WO", raising=False)
    st = {"span_id": 12, "corpus_added_at": None}
    fr = rr.assess(_adm([st]), CFG, NOW)
    bf = rr.apply_backfill(fr, CFG, NOW, {"db_url_env_wo": "VERA_DB_URL_WO"})
    assert st["corpus_added_at"] == NOW.isoformat()
    assert bf["persist"]["status"] == "skipped" and bf["persist"]["reason"] == "VERA_DB_URL_WO not set"


def test_in_memory_only_chunks_are_not_persisted():
    st = {"span_id": None, "document_id": "d1", "corpus_added_at": None}
    store = FakeStore({})
    bf = rr.apply_backfill(rr.assess(_adm([st]), CFG, NOW), CFG, NOW, {}, stamper=store)
    assert bf["count"] == 1 and bf["persist"]["status"] == "not_needed" and bf["persist"]["not_persistable"] == 1
    assert store.calls == []


def test_idempotent_under_concurrency_conditional_update():
    store = FakeStore({13: None})
    results = []
    times = [NOW, NOW + timedelta(seconds=5)]

    def run(t):  # two server processes, each with its own in-memory copy of the stamp
        st = {"span_id": 13, "corpus_added_at": None}
        results.append(corpus_stamp.backfill([st], t, CFG, store))
    th = [threading.Thread(target=run, args=(t,)) for t in times]
    for x in th:
        x.start()
    for x in th:
        x.join()
    assert sum(r["persist"]["updated"] for r in results) == 1  # only the first conditional UPDATE matched
    assert store.rows[13] in times


def test_stamp_sql_is_single_row_and_conditional():
    sql = corpus_stamp.SQL_STAMP
    assert sql.startswith("UPDATE vera_vjay.evidence_spans SET created_at")
    assert "WHERE span_id = :span_id AND (created_at IS NULL OR created_at > " in sql


def test_backfill_record_is_public_without_internal_refs():
    st = {"span_id": None, "corpus_added_at": None}
    fr = rr.assess(_adm([st]), CFG, NOW)
    bf = rr.apply_backfill(fr, CFG, NOW, {})
    rec = rr.summarize(triggers=["freshness_unknown"], inline={"status": "off"}, used_new=False, older_available=True,
                       fr_before=fr, backfill=bf)
    pf = rr.public_fields(rec)
    assert pf["freshness"]["corpus_added_backfill"]["count"] == 1
    assert "_backfill" not in rec["freshness"]["before"]


def test_kill_switch_keeps_memory_stamp_but_skips_persist(monkeypatch):
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", "1")
    st = {"span_id": 14, "corpus_added_at": None}
    store = FakeStore({14: None})
    bf = rr.apply_backfill(rr.assess(_adm([st]), CFG, NOW), CFG, NOW, {}, stamper=store)
    assert st["corpus_added_at"] == NOW.isoformat() and bf["count"] == 1
    assert bf["persist"]["status"] == "skipped" and bf["persist"]["reason"] == "kill switch"
    assert store.calls == [] and store.rows[14] is None
    assert corpus_stamp.default_stamper({}, {"VERA_AUTOACQ_DISABLED": "1", "VERA_DB_URL_WO": "postgresql://wo.invalid/x"}) \
        == (None, "kill switch")


def test_missing_privilege_is_recorded_harmlessly():
    class Denied(Exception):
        pgcode = "42501"

    class NoPriv:
        def stamp(self, *a):
            raise Denied("permission denied for table evidence_spans")
    st = {"span_id": 15, "corpus_added_at": "2026"}
    bf = rr.apply_backfill(rr.assess(_adm([st]), CFG, NOW), CFG, NOW, {}, stamper=NoPriv())
    assert st["corpus_added_at"] == NOW.isoformat()  # in-memory stamp kept
    assert bf["persist"]["status"] == "failed" and bf["persist"]["reason"] == "no privilege"


def test_partial_stamps_are_set_to_now():
    for v in ("2026", "2026-09"):
        st = {"span_id": None, "corpus_added_at": v}
        fr = rr.assess(_adm([st]), CFG, NOW)
        assert rr.triggers(True, fr) == ["freshness_unknown"], v
        rr.apply_backfill(fr, CFG, NOW, {})
        assert st["corpus_added_at"] == NOW.isoformat(), v


def test_default_stamper_reuses_one_engine(monkeypatch):
    import sqlalchemy
    made = []
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url, **kw: made.append(url) or object())
    corpus_stamp.reset_for_tests()
    env = {"VERA_DB_URL_WO": "postgresql://wo.invalid/x"}
    a, _ = corpus_stamp.default_stamper({}, env)
    b, _ = corpus_stamp.default_stamper({}, env)
    assert a.engine is b.engine and made == ["postgresql://wo.invalid/x"]
    corpus_stamp._ENGINES.clear()
