"""WP-L live aggregate statistics: offline tests with fakes; no database, no network."""
import json
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from vera import live_stats as ls

ROOT = Path(__file__).resolve().parent.parent
FWD = ROOT / "db/migrations/pending/013-DRAFT-vera-live-ask-stats.sql.draft"
RBK = ROOT / "db/migrations/rollback/013-DRAFT-vera-live-ask-stats-rollback.sql.draft"
ON = {"enabled": True, "min_count": 5, "cache_ttl_s": 60, "window_days_max": 30, "hard_timeout_s": 1}
ENV = {"VERA_DB_URL_WO": "sqlite://", "VERA_DB_URL_RO": "sqlite://"}
OWNER = "o" * 40
API = "a" * 40


class FakeConn:
    def __init__(self, eng): self.eng = eng
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None):
        self.eng.calls.append((str(sql), params))
        if self.eng.sleep: time.sleep(self.eng.sleep)
        if self.eng.boom: raise RuntimeError("postgresql://secret-host/db")
        if self.eng.rows is not None and "SELECT" in str(sql) and "SET LOCAL" not in str(sql):
            return [type("R", (), {"_mapping": r})() for r in self.eng.rows]


class FakeEngine:
    def __init__(self, rows=None, sleep=0, boom=False): self.calls, self.rows, self.sleep, self.boom = [], rows, sleep, boom
    def begin(self): return FakeConn(self)
    def connect(self): return FakeConn(self)


@pytest.fixture(autouse=True)
def _reset():
    ls.reset_for_tests()
    yield
    ls.reset_for_tests()


def factory(eng): return lambda url, **kw: eng


BODY = {"result_kind": "answer", "research_outcome": {"kind": "abstain", "user_text": "free text"}, "tokens_used": 12,
        "cost_usd": 0.01, "claim_check": {"supported": 2, "partial": 1}, "citations": [1, 2], "admitted_evidence": [1],
        "answer": "QUESTION-TEXT-SECRET", "question": "QUESTION-TEXT-SECRET"}


def test_config_ships_off_with_enabled_key():
    cfg = json.loads((ROOT / "config/live_stats.json").read_text())
    assert cfg["enabled"] is False and cfg["min_count"] >= 1
    assert ls.enabled(ls.load_cfg()) is False


def test_disabled_means_no_write():
    eng = FakeEngine()
    assert ls.record(BODY, 5, cfg={"enabled": False}, env=ENV, engine_factory=factory(eng)) is False
    assert eng.calls == []


def test_row_is_enums_and_ints_only():
    row = ls.project(dict(BODY, result_kind="weird"), 123)
    assert row["result_kind"] == "other" and row["research_outcome"] == "abstain"
    assert set(row) == {"result_kind", "research_outcome", "claims", "citations", "evidence_passages", "latency_ms",
                        "tokens", "cost_usd"}
    assert row["claims"] == 3 and row["citations"] == 2 and row["evidence_passages"] == 1
    assert "QUESTION-TEXT-SECRET" not in json.dumps(row)
    eng = FakeEngine()
    assert ls.record(BODY, 7, cfg=ON, env=ENV, engine_factory=factory(eng)) is True
    assert "QUESTION-TEXT-SECRET" not in repr(eng.calls)


def test_writer_is_fail_open_and_logs_no_body(caplog):
    eng = FakeEngine(boom=True)
    assert ls.record(BODY, 7, cfg=ON, env=ENV, engine_factory=factory(eng)) is False
    assert "secret-host" not in caplog.text and "QUESTION-TEXT" not in caplog.text and "LS_WRITE_ERROR" in caplog.text


def test_record_stops_at_its_timeout(caplog):
    eng = FakeEngine(sleep=3)
    t0 = time.monotonic()
    assert ls.record(BODY, 7, cfg=dict(ON, hard_timeout_s=0.2), env=ENV, engine_factory=factory(eng)) is False
    assert time.monotonic() - t0 < 1.0 and "LS_TIMEOUT" in caplog.text


# ---- hooks in main.py
def _main(monkeypatch, enabled=True):
    import main
    monkeypatch.setattr(ls, "enabled", lambda cfg=None: enabled)
    return main


def _resp():
    import main
    return main.AskResponse(answer="a", model="m", tokens_used=1, cost_usd=0.0)


def test_ask_schedules_the_write_as_a_background_task_not_inline(monkeypatch):
    main = _main(monkeypatch)
    calls = []
    monkeypatch.setattr(ls, "record", lambda body, ms: calls.append(ms))
    monkeypatch.setattr(main, "_ask_impl", lambda req, request, bg: _resp())
    bg = BackgroundTasks()
    out = main.ask(MagicMock(), MagicMock(), bg)
    assert out.answer == "a" and calls == []          # nothing ran inline
    assert len(bg.tasks) == 1 and bg.tasks[0].func is ls.record
    assert bg.tasks[0].args[0]["result_kind"] == out.result_kind


def test_ask_schedules_nothing_when_flag_off_or_on_error(monkeypatch):
    main = _main(monkeypatch, enabled=False)
    monkeypatch.setattr(main, "_ask_impl", lambda req, request, bg: _resp())
    bg = BackgroundTasks()
    main.ask(MagicMock(), MagicMock(), bg)
    assert bg.tasks == []
    main = _main(monkeypatch, enabled=True)
    def boom(req, request, bg): raise RuntimeError("x")
    monkeypatch.setattr(main, "_ask_impl", boom)
    with pytest.raises(RuntimeError):
        main.ask(MagicMock(), MagicMock(), bg)
    assert bg.tasks == []


class Sink:
    def __init__(self, bg): self.bg, self.results, self.seen = bg, [], []
    def put_result(self, status, body=None, kind=None):
        self.seen.append(len(self.bg.tasks)); self.results.append(status)
    def close(self): pass
    def append(self, *a, **k): pass
    def emit(self, *a, **k): pass


def _worker(main, monkeypatch, impl, cancelled=False):
    monkeypatch.setattr(main, "_ask_impl", impl)
    ctx = MagicMock(); ctx.cancelled = cancelled
    bg = BackgroundTasks(); sink = Sink(bg)
    monkeypatch.setattr(main.progress, "emit", lambda *a, **k: None)
    monkeypatch.setattr(main.progress, "terminal_counts", lambda *a, **k: {})
    import contextvars
    contextvars.copy_context().run(main._stream_worker, sink, ctx, MagicMock(), MagicMock(), bg)  # worker sets ContextVars
    return bg, sink


def test_stream_records_once_on_success_before_the_terminal_put(monkeypatch):
    main = _main(monkeypatch)
    bg, sink = _worker(main, monkeypatch, lambda *a: _resp())
    assert sink.results == [200] and len(bg.tasks) == 1 and sink.seen == [1]   # queued BEFORE put_result(200)


def test_stream_records_nothing_on_error_or_abort(monkeypatch):
    main = _main(monkeypatch)
    def boom(*a): raise RuntimeError("x")
    bg, sink = _worker(main, monkeypatch, boom)
    assert sink.results == [500] and bg.tasks == []
    from vera import stream_ctx
    def cancel(*a): raise stream_ctx.StreamCancelled()
    bg, sink = _worker(main, monkeypatch, cancel)
    assert sink.results == [0] and bg.tasks == []
    bg, sink = _worker(main, monkeypatch, lambda *a: _resp(), cancelled=True)   # client gone before the terminal event
    assert bg.tasks == []


# ---- reader
def rows(*cells): return [{"day": d, "result_kind": k, "n": n, "median_latency_ms": m, "tokens": 10, "cost_usd": 0.5}
                          for d, k, n, m in cells]


def test_suppression_cells_complementary_and_median():
    out = ls.suppress_map({"answer": 40, "decline": 2, "reading_list": 9}, 5)
    assert out == {"answer": 40, "decline": "<5", "reading_list": "<5"}      # small cell + complementary smallest
    assert ls.suppress_map({"answer": 40, "decline": 9}, 5) == {"answer": 40, "decline": 9}
    eng = FakeEngine(rows=rows(("d1", "answer", 40, 9), ("d1", "decline", 2, 7), ("d1", "reading_list", 9, 8)))
    o = ls.read_aggregates(7, cfg=ON, env=ENV, engine_factory=factory(eng))
    assert o["by_result_kind"]["decline"] == "<5" and o["typical_latency_ms"]["decline"] == "<5"   # median suppressed
    assert o["typical_latency_ms"]["answer"] == 9 and o["total"] == 51 and o["enabled"] is True


def test_public_shape_contract_enabled_total_and_label_maps():
    eng = FakeEngine(rows=rows(("d1", "answer", 50, 9), ("d1", "decline", 1, 3)))
    o = ls.read_aggregates(7, cfg=ON, env=ENV, engine_factory=factory(eng))
    assert set(o) == {"enabled", "total", "by_result_kind", "by_day", "typical_latency_ms"}
    assert all(isinstance(v, (int, str)) for sec in ("by_result_kind", "by_day", "typical_latency_ms") for v in o[sec].values())
    ls.reset_for_tests()
    tiny = ls.read_aggregates(30, cfg=ON, env=ENV, engine_factory=factory(FakeEngine(rows=rows(("d1", "answer", 2, 9)))))
    assert tiny["total"] == "<5"


def test_ttl_cache_runs_one_query_for_two_calls():
    eng = FakeEngine(rows=rows(("d1", "answer", 50, 9)))
    kw = dict(cfg=ON, env=ENV, engine_factory=factory(eng))
    a = ls.read_aggregates(7, **kw); b = ls.read_aggregates(7, **kw)
    assert a is b and sum(1 for s, _ in eng.calls if "FROM" in s) == 1


def test_window_is_capped():
    eng = FakeEngine(rows=rows(("d1", "answer", 50, 9)))
    out = ls.read_aggregates(9999, cfg=ON, env=ENV, engine_factory=factory(eng))
    assert [p for q, p in eng.calls if "FROM" in q][0] == {"d": 30}


# ---- routes
@pytest.fixture
def client(monkeypatch):
    import main
    monkeypatch.setenv("VERA_API_KEY", API)
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    eng = FakeEngine(rows=rows(("d1", "answer", 50, 9), ("d1", "decline", 40, 3)))
    monkeypatch.setattr(ls, "load_cfg", lambda path=None: dict(ON))
    monkeypatch.setattr(ls.os, "environ", ENV, raising=False) if False else None
    orig = ls.read_aggregates
    monkeypatch.setattr(main.live_stats, "read_aggregates",
                        lambda days, owner=False: orig(days, owner=owner, cfg=dict(ON), env=ENV, engine_factory=factory(eng)))
    return TestClient(main.app)


def test_public_route_needs_no_key_and_has_no_tokens_or_cost(client):
    r = client.get("/trace/live")
    assert r.status_code == 200 and r.json()["enabled"] is True and r.json()["total"] == 90
    assert "tokens" not in r.text and "cost" not in r.text


def test_owner_route_auth_and_totals(client, monkeypatch):
    assert client.get("/trace/live/owner").status_code == 401
    assert client.get("/trace/live/owner", headers={"X-Owner-Key": "w" * 40}).status_code == 401
    ok = client.get("/trace/live/owner", headers={"X-Owner-Key": OWNER})
    assert ok.status_code == 200 and ok.json()["tokens_total"] == 20 and ok.json()["cost_usd_total"] == 1.0
    monkeypatch.delenv("VERA_OWNER_KEY")
    assert client.get("/trace/live/owner", headers={"X-Owner-Key": OWNER}).status_code == 503


def test_owner_route_declares_owner_dependency_only():
    import main
    route = next(r for r in main.app.routes if getattr(r, "path", "") == "/trace/live/owner")
    deps = [d.call for d in route.dependant.dependencies]
    assert main.verify_owner_key in deps and main.verify_api_key not in deps
    pub = next(r for r in main.app.routes if getattr(r, "path", "") == "/trace/live")
    assert pub.dependant.dependencies == []


# ---- static migration checks
def _sql(p): return "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("--"))


def test_migration_pair_names_header_and_preflights():
    assert FWD.parent.name == "pending" and RBK.parent.name == "rollback"
    assert FWD.name.endswith(".sql.draft") and RBK.name.endswith(".sql.draft")
    assert "NOT EXECUTED" in "\n".join(FWD.read_text().splitlines()[:3]) and "NOT EXECUTED" in RBK.read_text().splitlines()[0]
    s = _sql(FWD)
    assert "already applied or name collision" in s and "pg_has_role" in s and "RAISE EXCEPTION" in s
    assert ":'app_wo_login'" in s and ":'app_ro_login'" in s
    assert "vera_vjay.live_ask_stats" in s and "CREATE TABLE live_ask_stats" not in s


def test_grants_follow_012_pattern_and_revoke_precedes_them():
    s = _sql(FWD)
    rev, g_use = s.index("REVOKE ALL ON vera_vjay.live_ask_stats FROM PUBLIC"), s.index("GRANT USAGE ON SCHEMA vera_vjay TO vera_vjay_wo, vera_vjay_ro")
    ins = s.index("GRANT INSERT ON vera_vjay.live_ask_stats TO vera_vjay_wo")
    sel = s.index("GRANT SELECT ON vera_vjay.live_ask_stats TO vera_vjay_ro")
    assert rev < ins and rev < sel and g_use < ins
    assert len(re.findall(r"^GRANT ", s, re.M)) == 3     # nothing else is granted
    assert not re.search(r"(TO|FROM) vera_pipeline_wo", s)   # logins are used only via psql variables


def test_append_only_trigger_and_enum_only_columns():
    s = _sql(FWD)
    assert "BEFORE UPDATE OR DELETE ON vera_vjay.live_ask_stats" in s and "BEFORE TRUNCATE" in s
    cols = s.split("CREATE TABLE vera_vjay.live_ask_stats (")[1].split(");")[0].upper()
    assert "TEXT NOT NULL CONSTRAINT live_ask_stats_result_kind_chk".upper() in cols
    for forbidden in ("QUESTION", "CLIENT", " IP ", "TEXT[]", "JSON"):
        assert forbidden not in cols.replace("RESEARCH_OUTCOME TEXT", "")


def test_rollback_reverses_every_created_object():
    f, r = _sql(FWD), _sql(RBK)
    for name in re.findall(r"CREATE (?:UNIQUE )?(?:TABLE|INDEX|FUNCTION|TRIGGER) (?:vera_vjay\.)?(\w+)", f):
        assert name in r, name
    assert "CASCADE" not in r


def test_secret_and_host_scan():
    pat = re.compile(r"postgres(ql)?://|onrender\.com|password\s*=|\bsk-[A-Za-z0-9]{10}|://\d+\.\d+\.\d+\.\d+")
    for p in (ROOT / "config/live_stats.json", FWD, RBK, ROOT / "vera/live_stats.py", Path(__file__)):
        text = p.read_text()
        if p == Path(__file__):
            text = text.replace("postgresql://secret-host/db", "")
        assert not pat.search(text), p


# ---- WP6b security hardening
def test_migration_truncates_ts_to_the_hour_and_rollback_drops_it():
    f, r = _sql(FWD), _sql(RBK)
    assert "date_trunc('hour'" in f and "live_ask_stats_hour_trg BEFORE INSERT" in f and "COMMENT ON COLUMN" in f
    assert "live_ask_stats_hour_trg" in r and "live_ask_stats_hour_trgfn" in r
    assert "NOT EXECUTED" in FWD.read_text().splitlines()[0] and "NOT EXECUTED" in RBK.read_text().splitlines()[0]


def test_writer_never_sends_a_timestamp():
    eng = FakeEngine()
    assert ls.record(BODY, 5, cfg=ON, env=ENV, engine_factory=factory(eng)) is True
    ins = [(s, p) for s, p in eng.calls if "INSERT" in s][0]
    assert "ts" not in ins[1] and " ts" not in ins[0].split("VALUES")[0]


def test_wording_is_per_answer_record_rounded_to_hour():
    from vera import help_pages
    from vera.trace_eval import ui
    for s in (help_pages.LIVE_STATS_LINE, ui.LIVE_INTRO_TEXT):
        assert "one anonymous record per answer" in s.lower() and "rounded to the hour" in s
        assert "no question text or identifiers" in s and "aggregate usage counts" not in s


def test_failing_reader_is_bounded_and_negative_cached():
    seen = {}

    def fac(url, **kw):
        seen.update(kw)
        return FakeEngine(boom=True)
    t0 = time.monotonic()
    a = ls.read_aggregates(7, cfg=ON, env=ENV, engine_factory=fac)
    assert time.monotonic() - t0 < 2.5 and a["total"] == 0
    assert seen["pool_timeout"] <= 2.5 and seen["connect_args"]["connect_timeout"] <= 3
    calls = []
    ls.reset_for_tests()
    eng = FakeEngine(boom=True)
    kw = dict(cfg=ON, env=ENV, engine_factory=factory(eng))
    x = ls.read_aggregates(7, **kw); y = ls.read_aggregates(7, **kw)
    assert x is y and sum(1 for s, _ in eng.calls if "SET LOCAL" in s) == 1


def test_trace_live_fallback_reports_the_real_flag(monkeypatch):
    import main
    monkeypatch.setattr(main.live_stats, "read_aggregates", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(main.live_stats, "enabled", lambda cfg=None: False)
    c = TestClient(main.app)
    assert c.get("/trace/live").json()["enabled"] is False
    monkeypatch.setattr(main.live_stats, "enabled", lambda cfg=None: True)
    assert c.get("/trace/live").json()["enabled"] is True
