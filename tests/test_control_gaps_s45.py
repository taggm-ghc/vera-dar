"""CONTROL-GAPS-1107 S45: (a) ND lane pages carry licence_class/canonical_text so nd_guard (fulldoc_answer) applies;
(b) memory recall re-applies the write-time licence control at read. Fakes only: no DB, network or env."""
from datetime import datetime, timezone

from vera import fulldoc_answer as fa, lane_evidence as le, memory_store as ms, nd_guard, reporting_lane as rl
from vera.official_lane import LanePage

CFG = {"admission": {"min_relevance": 0.0, "min_matched_stems": 1, "max_spans_per_doc": 2, "max_spans_total": 6}}


def _page(text="Kubernetes autoscaling adjusts replica counts based on observed metrics and load."):
    return LanePage(lane="reporting", entity="", title="T", url="https://news.example/a", outlet="news.example", text=text)


def _spans(monkeypatch, row):
    monkeypatch.setattr(rl, "outlet_row", row)
    from vera import fulldoc_admission as fa_
    monkeypatch.setattr(fa_, "_routing", lambda cfg: {"identification": {}, "corpus": {"relevance_stem_len": 5}})
    monkeypatch.setattr(fa_, "_relevance", lambda *a: 1.0)
    monkeypatch.setattr(fa_, "_matched", lambda *a: (5, 5))
    monkeypatch.setattr(le, "_chunk_cfg", lambda: (400, 10))
    adm, _ = le.admit_pages("q", [_page()], CFG)
    return adm


def test_nd_outlet_page_is_marked_for_nd_guard(monkeypatch):
    adm = _spans(monkeypatch, lambda url, cfg=None: {"access": {"x": 1}})
    assert adm and adm[0]["licence_class"] == "nd" and adm[0]["canonical_text"].startswith("Kubernetes")
    srcs = fa.sources_from(type("E", (), {"admitted": adm})())
    assert nd_guard.nd_sources(srcs) and srcs[0]["canonical_text"]


def test_outlet_lookup_error_fails_closed_as_nd(monkeypatch):
    def boom(url, cfg=None): raise RuntimeError("x")
    assert _spans(monkeypatch, boom)[0]["licence_class"] == "nd"


def test_non_nd_outlet_unchanged(monkeypatch):
    adm = _spans(monkeypatch, lambda url, cfg=None: {"terms_verdict": "allow"})
    assert "licence_class" not in adm[0]
    assert "licence_class" not in fa.sources_from(type("E", (), {"admitted": adm})())[0]


CFGM = {"query_stopwords": [], "min_query_word_len": 3, "max_query_terms": 8, "question_text": "qt", "status_live": "live",
        "ttl_days": 30, "min_matched_terms": 1, "min_rank": 0.0, "half_life_days": 30, "recall_k": 3}


def _src_read(sql):
    """ARXIV-DEFENSE-1113: answer the portal-check source read with a non-portal doi.org source (real column order)."""
    return [(1, {"document_id": "acq-1"}, "https://doi.org/10.1000/example.1")] if "c.claim_id = ANY(:ids)" in sql else None


def test_recall_filters_on_licence_decision_at_read():
    r = ms.RoMemoryReader(object(), CFGM)
    seen = []
    r._rows = lambda sql, p: seen.append((sql, p)) or _src_read(sql) or [("c", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "u", 1)]
    assert r.recall("quantized inference latency")
    sql, p = seen[0]
    assert "provenance ->> 'licence_decision' = ANY" in sql and p["okdec"] == ["allow"]


def test_recall_fails_closed_when_gate_config_unavailable(monkeypatch):
    from vera import memory_gate
    def boom(*a, **k): raise memory_gate.MemoryGateConfigError("x")
    monkeypatch.setattr(memory_gate, "load_gate_config", boom)
    r = ms.RoMemoryReader(object(), CFGM)
    r._rows = lambda sql, p: [("c", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "u", 1)]
    assert r.recall("quantized inference latency") == []


def test_recall_logs_missing_licence_count_numbers_only(caplog):
    import logging
    r = ms.RoMemoryReader(object(), CFGM)
    row = ("SECRET claim", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "https://secret.example/u", 1)
    r._rows = lambda sql, p: [(4,)] if "count(DISTINCT" in sql else (_src_read(sql) or [row])
    with caplog.at_level(logging.INFO, logger="vera"):
        out = r.recall("quantized inference latency secretterm")
    assert out and out[0]["claim_text"] == "SECRET claim"  # recall unchanged
    msgs = [m.getMessage() for m in caplog.records if "missing" in m.getMessage()]
    assert msgs == ["memory recall: 4 live claim(s) not recalled, source licence_decision missing"]
    assert not any(w in caplog.text for w in ("SECRET", "secret.example", "secretterm", "quantized"))


def test_count_failure_never_breaks_recall(caplog):
    import logging
    r = ms.RoMemoryReader(object(), CFGM)
    row = ("c", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "u", 1)

    def rows(sql, p):
        if "count(DISTINCT" in sql:
            raise RuntimeError("db detail SECRET")
        return _src_read(sql) or [row]
    r._rows = rows
    with caplog.at_level(logging.INFO, logger="vera"):
        assert len(r.recall("quantized inference latency")) == 1
    assert "missing_licence_count_failed" in caplog.text and "SECRET" not in caplog.text
