"""Item #79 phase C: corpus admission (Gate A + licence gate, write-only store). Offline only."""
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from vera import corpus_admission as ca, source_resolver as sr
from vera.ask_service import AskResult
from vera.search_providers import OPENALEX_METADATA_LICENCE, licence_record

ROOT = Path(__file__).resolve().parent.parent
CFG = {"enabled": True, "db_url_env": "VERA_DB_URL_WO", "question_text": "FIXED CORPUS QUESTION",
       "gate_a": {"fetch_threshold": 0.6, "reject_threshold": 0.25, "max_fetch": 8}}
CC0 = licence_record(metadata=OPENALEX_METADATA_LICENCE)


def _src(n, url, licence=CC0, snippet="an abstract"):
    return {"n": n, "title": f"T{n}", "url": url, "snippet": snippet, "licence": licence,
            "stored_text_kind": "abstract_metadata", "doi": None}


def _llm(scores):
    def call(messages):
        rows = [{"id": i, "query_fit": s, "evidentiary_value": s, "cost_risk": s, "uncertainty": "low",
                 "rationale": {"query_fit": "r", "evidentiary_value": "r", "cost_risk": "r"}} for i, s in enumerate(scores)]
        return json.dumps({"scores": rows}), 10, 10
    return call


class FakeStore:
    def __init__(self):
        self.rows = []

    def admit(self, c):
        self.rows.append(c)
        return "admitted" if c["gate_a_decision"] == "fetch" else f"recorded_{c['gate_a_decision']}"


def test_gauntlet_admits_only_relevant_and_licensed():
    srcs = [_src(1, "https://arxiv.org/abs/1"), _src(2, "https://doi.org/10.1/b", licence=None),
            _src(3, "https://doi.org/10.1/c")]
    store = FakeStore()
    counts = ca.admit("visitor question", srcs, _llm([0.9, 0.9, 0.1]), CFG, store)
    by_url = {r["url"]: r for r in store.rows}
    assert by_url["https://arxiv.org/abs/1"]["gate_a_decision"] == "fetch"
    assert by_url["https://doi.org/10.1/b"]["gate_a_decision"] == "defer"  # no declared licence: held for review
    assert by_url["https://doi.org/10.1/c"]["gate_a_decision"] == "reject"  # low score, low uncertainty
    assert counts == {"admitted": 1, "recorded_defer": 1, "recorded_reject": 1}


def test_visitor_question_is_never_passed_to_storage():
    store = FakeStore()
    ca.admit("SECRET VISITOR TEXT", [_src(1, "https://arxiv.org/abs/1")], _llm([0.9]), CFG, store)
    assert "SECRET VISITOR TEXT" not in json.dumps(store.rows, default=str)


def test_admission_never_raises():
    def boom(messages):
        raise RuntimeError("provider down")

    class BadStore:
        def admit(self, c):
            raise RuntimeError("db down")

    assert ca.admit("q", [_src(1, "https://arxiv.org/abs/1")], boom, CFG, BadStore())["error"] == "RuntimeError"


def test_store_disabled_without_db_url():
    assert ca.store_from_env(CFG, env={}) is None
    assert ca.store_from_env(None, env={"VERA_DB_URL_WO": "x"}) is None


def test_store_sql_is_write_only_compatible():
    """vera_pipeline_wo may INSERT/UPDATE and read back ids only: no SELECT statements and no WHERE on content."""
    src = (ROOT / "vera" / "corpus_admission.py").read_text()
    sql = " ".join(re.findall(r'"((?:INSERT|UPDATE|SELECT|ON CONFLICT|VALUES|RETURNING)[^"]*)"', src))
    assert "SELECT" not in sql.upper().replace("RETURNING", "")
    assert "DELETE" not in src.upper() and "ON CONFLICT" not in sql.upper()  # ON CONFLICT needs SELECT on its columns
    assert re.findall(r"WHERE (\w+)", src) == ["candidate_id"]


def test_canonical_url_prefers_doi():
    assert ca.canonical_url({"doi": "10.1/X", "url": "https://x/"}) == "https://doi.org/10.1/x"
    assert ca.canonical_url({"url": "https://arxiv.org/abs/1/"}) == "https://arxiv.org/abs/1"


def test_resolver_licence_follows_abstract_origin():
    oa = sr._record("10.1/a", "T", None, "openalex abstract")
    assert oa["licence"] == CC0 and oa["stored_text_kind"] == "abstract_metadata"

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"abstract": "s2 abstract"}
    s2 = sr.with_abstract(sr._record("10.1/b", "T", None, ""), 1, http_get=lambda *a, **k: R)
    assert s2["snippet"] == "s2 abstract" and s2["licence"] is None


def test_ask_schedules_admission_only_with_a_store(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    src = [_src(1, "https://arxiv.org/abs/1")]
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (AskResult("A [1]", 5, 0.0, sources=tuple(src)), "groq:m"))
    calls = []
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a: calls.append(a))
    c, h = TestClient(main.app), {"X-API-Key": "k" * 20}
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    assert c.post("/ask", json={"question": "q"}, headers=h).status_code == 200 and calls == []
    monkeypatch.setattr(main, "store_from_env", lambda cfg: FakeStore())
    assert c.post("/ask", json={"question": "q"}, headers=h).status_code == 200 and len(calls) == 1
