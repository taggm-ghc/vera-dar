"""NONASK-1103 WP6 (issue b): /trace/live producer (main.py + vera/live_stats.py, WP-L) versus the Trace Eval Live tab
consumer (vera/trace_eval/ui.py, WP2). Offline: the producer reads a fake engine; the consumer gets the route's real JSON."""
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from vera import live_stats as ls

ON = {"enabled": True, "min_count": 5, "cache_ttl_s": 60, "window_days_max": 30, "hard_timeout_s": 1}
ENV = {"VERA_DB_URL_WO": "sqlite://", "VERA_DB_URL_RO": "sqlite://"}


class _Conn:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, sql, params=None):
        if "SELECT" in str(sql) and "SET LOCAL" not in str(sql):
            return [type("R", (), {"_mapping": r})() for r in self.rows]


class _Eng:
    def __init__(self, rows): self.rows = rows
    def connect(self): return _Conn(self.rows)
    def begin(self): return _Conn(self.rows)


def _rows():
    return [{"day": "2026-11-01", "result_kind": "answer", "n": 50, "median_latency_ms": 9100, "tokens": 777111,
             "cost_usd": 42.4242},
            {"day": "2026-11-02", "result_kind": "reading_list", "n": 30, "median_latency_ms": 4300, "tokens": 777111,
             "cost_usd": 42.4242},
            {"day": "2026-11-02", "result_kind": "decline", "n": 2, "median_latency_ms": 1200, "tokens": 1, "cost_usd": 0}]


@pytest.fixture
def live_body(monkeypatch):
    """The JSON body GET /trace/live really returns for the fake rows."""
    import main
    ls.reset_for_tests()
    orig = ls.read_aggregates
    eng = _Eng(_rows())
    monkeypatch.setattr(main.live_stats, "read_aggregates",
                        lambda days, owner=False: orig(days, owner=owner, cfg=dict(ON), env=ENV,
                                                       engine_factory=lambda url, **kw: eng))
    r = TestClient(main.app).get("/trace/live")
    ls.reset_for_tests()
    assert r.status_code == 200
    return r.json()


def _live_app(base_url):
    from vera.trace_eval.ui import _render_live
    _render_live(base_url)


def _render(monkeypatch, body) -> str:
    from vera.trace_eval import ui as _ui
    monkeypatch.setattr(_ui, "fetch_live", lambda *a, **k: (200, body))
    at = AppTest.from_function(_live_app, args=("http://x.invalid:9",), default_timeout=30).run()
    assert not at.exception
    texts = [str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.info)]
    return "\n".join(texts + [t.value.to_string() for t in at.table])


def test_producer_shape_is_what_the_consumer_reads(live_body):
    assert set(live_body) == {"enabled", "total", "by_result_kind", "by_day", "typical_latency_ms"}
    assert isinstance(live_body["typical_latency_ms"], dict)
    assert all(isinstance(v, (int, str)) for k in ("by_result_kind", "by_day", "typical_latency_ms")
               for v in live_body[k].values())


def test_consumer_renders_every_producer_cell_and_no_tokens_or_cost(monkeypatch, live_body):
    t = _render(monkeypatch, live_body)
    assert "not yet collecting" not in t and str(live_body["total"]) in t
    for section, label in (("by_result_kind", "Questions by result"), ("by_day", "Questions by day (UTC)"),
                           ("typical_latency_ms", "Typical response time (ms)")):
        for k, v in live_body[section].items():
            assert f"{label}: {k}" in t and str(v) in t
    assert "9100" in t and "<5" in t                       # a median and a suppressed cell both reach the page
    assert "777111" not in t and "42.42" not in t           # tokens and cost never reach the producer body or the page


def test_consumer_renders_a_scalar_typical_latency(monkeypatch):
    t = _render(monkeypatch, {"enabled": True, "total": 80, "by_result_kind": {"answer": 80}, "typical_latency_ms": 6400})
    assert "Typical response time (ms)" in t and "6400" in t and "Questions by result: answer" in t


def test_consumer_drops_owner_only_totals_if_ever_present(monkeypatch, live_body):
    t = _render(monkeypatch, dict(live_body, tokens_total=777111, cost_usd_total=42.4242))
    assert "777111" not in t and "42.42" not in t
