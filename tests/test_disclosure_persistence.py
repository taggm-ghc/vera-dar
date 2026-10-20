"""R-028 (Demo Day Data Retention and Disclosure, Option C): with the SHIPPED config, a public POST /ask writes nothing.

Everything is mocked: no DB, no network, no provider call. `sqlalchemy.create_engine` is replaced by a recorder, so any
engine VERA-DAR tried to create (and any SQL it tried to run) is captured; filesystem writes are recorded by wrapping the
write primitives. Labels: every claim here is "verified by test (mocked)"; the deployed environment is not exercised.
"""
import builtins
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_item71_doaj import PAGE, Q as DDG_Q
from vera import ask_service
from vera import answer_contract as ac
from vera import corpus_admission, memory_store
from vera import search_providers as sp
from vera.ask_service import AskResult
from vera.eval_config import load_eval_config
from vera.grounded_ask import public_candidates

ROOT = Path(ask_service.__file__).resolve().parent.parent
SENT = "postgresql://sentinel_user:sentinel_pw@db-sentinel.invalid:5432/sentinel_db"
QUESTION = "What does published research show about SENTINELQ transformer quantization latency?"
SRC = [{"n": 1, "title": "Paper One", "url": "https://arxiv.org/abs/2401.00001", "published": "2025-01-01",
        "snippet": "an abstract", "abstract": "an abstract", "identifier": "2401.00001", "provider": "arxiv",
        "licence_decision": "allow", "licence": "cc0", "stored_text_kind": "abstract_metadata", "doi": None,
        "retrieved_at": "2026-10-01"}]
WRITE_TABLES = ("questions", "runs", "candidates", "canonical_sources", "sources", "licence_observations", "claims",
                "answers", "evidence_spans")


class RecConn:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, stmt, params=None, *a, **k):
        self.log.append(str(stmt))
        if not str(stmt).lstrip().lower().startswith(("select", "with")):
            raise AssertionError("a write statement was attempted: " + str(stmt)[:80])  # reads get an empty result
        return type("Empty", (), {"fetchall": lambda self: [], "fetchone": lambda self: None, "first": lambda self: None,
                                  "scalar": lambda self: None, "__iter__": lambda self: iter(())})()
    def begin(self): return self
    def begin_nested(self): return self


class RecEngine:
    def __init__(self, log): self.log = log
    def begin(self): return RecConn(self.log)
    def connect(self): return RecConn(self.log)
    def dispose(self): pass


@pytest.fixture
def recorder(monkeypatch):
    """Records engines created, SQL attempted and filesystem writes under the repo."""
    rec = {"engines": [], "sql": [], "files": []}

    def fake_create_engine(url, *a, **k):
        rec["engines"].append("engine")
        return RecEngine(rec["sql"])
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)
    monkeypatch.setattr(memory_store, "_SERVICE", {})
    from vera import dar_runtime  # the DAR runtime caches a corpus built from (fake) reads: keep it out of other tests
    monkeypatch.setattr(dar_runtime._DEFAULT, "_state", {})
    monkeypatch.setattr(corpus_admission, "_STORE", {})

    def under_root(p):
        try:
            return ROOT in Path(os.path.abspath(str(p))).parents
        except (TypeError, ValueError):
            return False
    real_open = builtins.open

    def spy_open(f, mode="r", *a, **k):
        if isinstance(f, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+") and under_root(f):
            rec["files"].append(("open", str(f)))
        return real_open(f, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", spy_open)
    for name in ("write_text", "write_bytes", "mkdir", "touch"):
        real = getattr(Path, name)

        def make(real=real, name=name):
            def spy(self, *a, **k):
                if under_root(self):
                    rec["files"].append((name, str(self)))
                return real(self, *a, **k)
            return spy
        monkeypatch.setattr(Path, name, make())
    for name in ("replace", "rename", "makedirs"):
        real = getattr(os, name)

        def make2(real=real, name=name):
            def spy(p, *a, **k):
                if under_root(p):
                    rec["files"].append((name, str(p)))
                return real(p, *a, **k)
            return spy
        monkeypatch.setattr(os, name, make2())
    return rec


def _post(monkeypatch, result, spies, call_real_admit=True):
    import main
    from vera import public_mode as pm
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)  # FAKE, well-formed
    monkeypatch.setenv("VERA_DB_URL_WO", SENT)  # sentinels: the recorder, not a database, receives any attempt
    monkeypatch.setenv("VERA_DB_URL_RO", SENT)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "GROUNDING", {"no_sources_text": "x", "insufficient_text": "x", "removed_citation_note": "x"})
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg, **k: ([dict(s) for s in SRC], []))
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (g[0]() and result, "groq:m"))
    def _no_llm(messages):
        raise AssertionError("Gate A model call")
    monkeypatch.setattr(main, "llm_call_for", lambda e: _no_llm)
    real_admit = main.admit_to_corpus
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a, **k: spies.append("admit") or (real_admit(*a, **k) if call_real_admit else None))
    return TestClient(main.app).post("/ask", json={"question": QUESTION}, headers={"X-API-Key": "k" * 20})


def _results():
    cands = public_candidates([dict(s) for s in SRC])
    reading = AskResult("list", 1, 0.0, sources=tuple(dict(s) for s in SRC), accepted=True, result_kind="reading_list",
                        candidates=cands, contract={"layers": ["L0"], "fallback": None})
    # worst case for the write paths: an accepted result carrying sources, claims and candidates
    answer = AskResult("A [1].", 1, 0.0, sources=tuple(dict(s) for s in SRC), accepted=True, result_kind="answer",
                       claims=({"id": 0, "sentence": "Quantization lowers latency in transformer models [1].",
                                "citations": [1], "verdict": "supported"},), candidates=cands)
    discovery = AskResult("list", 1, 0.0, sources=tuple(dict(s) for s in SRC), accepted=True, result_kind="discovery",
                          candidates=cands, contract={"layers": ["L0"], "fallback": None})
    return {"reading_list": reading, "answer_with_claims": answer, "discovery": discovery}


@pytest.mark.parametrize("kind", ["reading_list", "answer_with_claims", "discovery"])
def test_shipped_config_public_ask_makes_zero_writes(monkeypatch, recorder, kind):
    """Verified by test (mocked): no engine is even created, so no write to questions, runs, candidates,
    canonical_sources, sources, licence_observations, claims, answers (or evidence_spans); no DAR hook; no file write."""
    spies = []
    r = _post(monkeypatch, _results()[kind], spies)
    assert r.status_code == 200, r.text
    # the only engine allowed is the read-only DAR registry/health probe (SELECT); no statement may be a write
    writes = [s for s in recorder["sql"] if not s.lstrip().lower().startswith(("select", "with"))]
    assert writes == [], writes
    assert not [s for s in recorder["sql"] if any(t in s.lower() for t in ("insert", "update", "delete"))]
    assert spies == [], "corpus admission (and with it Gate A, licence_observations and the DAR hook) ran"
    assert recorder["files"] == [], recorder["files"]
    assert r.json()["memory_written"] == 0 and r.json()["memory_submitted"] == 0
    assert r.json()["discovery_note_submitted"] == 0 and r.json()["recalled_discovery"] == []


def test_control_recorder_detects_writes_when_corpus_is_switched_on(monkeypatch, recorder):
    """Control (mocked): flipping ONLY corpus.enabled makes the same request call corpus admission,
    so the zero-write result above comes from the shipped switch, not from a blind recorder."""
    import main
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["corpus"]
    monkeypatch.setattr(main, "CORPUS", dict(cfg, enabled=True))
    spies = []
    r = _post(monkeypatch, _results()["reading_list"], spies, call_real_admit=False)
    assert r.status_code == 200
    assert spies == ["admit"]


def test_control_recorder_sees_inserts_from_the_memory_store():
    log = []
    store = memory_store.WoMemoryStore(RecEngine(log), {"question_text": "q", "gate_version": "g1"})
    with pytest.raises(AssertionError):
        store.answer_id()
    assert log and "questions" in log[0].lower()  # the recorder captures an INSERT into questions


def test_shipped_switches_for_every_db_write_path():
    chain = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert chain["corpus"]["enabled"] is False          # corpus admission, licence_observations, Gate A, DAR admission hook
    assert chain["memory"]["store"]["enabled"] is False  # memory claim writes and discovery notes (flag, not just unreachable)
    assert ac.layers() == ["L0"]                         # L2 discovery notes and recall
    from vera.grounded_ask import load_corpus_cfg
    assert load_corpus_cfg() is None
    assert memory_store.memory_from_env(memory_store.load_memory_block(),
                                        env={"VERA_DB_URL_RO": SENT, "VERA_DB_URL_WO": SENT}) is None


def test_ddg_search_cache_is_off_in_the_shipped_search_config(monkeypatch, recorder):
    """Verified by test (mocked): the provider built from the config the public /ask uses has no cache dir, so a search
    writes no file even when it succeeds; the request words are not persisted."""
    settings = load_eval_config().search
    assert settings["duckduckgo_lite"]["cache_dir"] is None
    p = sp.PROVIDER_FACTORIES["duckduckgo_lite"](settings, transport=lambda *a: type("R", (), {"status_code": 200, "text": PAGE, "headers": {}})(),
                                                  sleep=lambda s: None)
    assert p.cache_dir is None
    p.parse(p.call(DDG_Q, 10, 5.0), 10)
    assert recorder["files"] == []
    assert not (ROOT / "tmp" / "search_cache").exists() or not list((ROOT / "tmp" / "search_cache").rglob("*.json"))


def test_openai_error_log_has_type_and_status_only(monkeypatch, caplog):
    import main
    from openai import OpenAIError

    class Boom(OpenAIError):
        status_code = 503

    def chain(q, e, g):
        raise Boom("SENTINEL-PROVIDER-ERROR-BODY echoing the question")
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(main, "answer_via_chain", chain)
    with caplog.at_level("INFO"):
        r = TestClient(main.app).post("/ask", json={"question": QUESTION}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 502
    assert "SENTINEL-PROVIDER-ERROR-BODY" not in caplog.text and "SENTINELQ" not in caplog.text
    assert "Boom" in caplog.text and "503" in caplog.text


def test_dependent_flags_still_true_but_inert_under_demo_config():
    """Verified by test (mocked, with the zero-write requests above): corpus.licence_observations_enabled and
    discovery_memory.write_on_l0 stay true in the shipped config, yet neither can cause a write, because each is read
    only behind corpus admission (corpus.enabled=false) and behind L2 (layers=["L0"]) respectively."""
    chain = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert chain["corpus"].get("licence_observations_enabled") is True and ac.write_on_l0() is True
    assert corpus_admission.store_from_env(None, env={"VERA_DB_URL_WO": SENT}) is None
    assert corpus_admission.store_from_env(dict(chain["corpus"]), env={"VERA_DB_URL_WO": SENT}) is None  # enabled=false
    import main
    memory = type("M", (), {"reader": object(), "store": object()})()
    res = _results()["discovery"]
    assert main._discovery_note_wanted(memory, res) is False  # write_on_l0 true, but L2 absent


def test_unhandled_exception_log_has_type_only(monkeypatch, caplog):
    """Verified by test (mocked): an unexpected exception whose message quotes the question is logged by type only."""
    import main
    from vera import public_mode as pm

    def chain(q, e, g):
        raise RuntimeError("GET https://example.invalid/search?q=SENTINELQ-IN-EXC-MESSAGE failed")
    pm.reset_limiter(None)
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(main, "answer_via_chain", chain)
    with caplog.at_level("INFO"):
        r = TestClient(main.app, raise_server_exceptions=False).post("/ask", json={"question": QUESTION},
                                                                     headers={"X-API-Key": "k" * 20})
    assert r.status_code == 500
    assert "SENTINELQ" not in caplog.text and "Traceback" not in caplog.text and "RuntimeError" in caplog.text


def test_access_log_disabled_and_uvicorn_error_path_sanitized(monkeypatch):
    """Verified by test (mocked, logger level): an access-log record is not emitted, and a uvicorn.error record that carries an
    exception whose message holds the canary question leaves the canary in no emitted record or handler output."""
    import io
    import logging
    import main
    main._harden_server_logging()
    canary = "SENTINELQ-CANARY-IN-EXC"
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    emitted = []

    class Cap(logging.Handler):
        def emit(self, record):
            emitted.append(record)
    cap = Cap()
    names = ("uvicorn.access", "uvicorn.error", "uvicorn")
    saved = {n: (logging.getLogger(n).handlers[:], logging.getLogger(n).propagate) for n in names}
    try:
        for n in names:
            lg = logging.getLogger(n)
            lg.handlers = [h, cap]  # as uvicorn's own handlers would be
            lg.setLevel(logging.INFO)
        # uvicorn.access (the real call shape in uvicorn's h11/httptools impl)
        logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "203.0.113.9:5555", "POST", "/ask", "1.1", 200)
        # uvicorn.error: h11_impl logs the unhandled ASGI exception with exc_info
        try:
            raise RuntimeError(f"GET https://example.invalid/search?q={canary} failed")
        except RuntimeError:
            logging.getLogger("uvicorn.error").error("Exception in ASGI application\n", exc_info=True)
            logging.getLogger("uvicorn").exception("again %s", "x")
    finally:
        for n, (hs, pr) in saved.items():
            logging.getLogger(n).handlers, logging.getLogger(n).propagate = hs, pr
    out = stream.getvalue() + "\n".join(r.getMessage() + str(r.exc_info) + str(r.exc_text) for r in emitted)
    assert canary not in out and "Traceback" not in out and "203.0.113.9" not in out and "/ask" not in out
    assert "RuntimeError" in out and all(r.exc_info is None for r in emitted)
    assert not [r for r in emitted if r.name == "uvicorn.access"]


def test_start_sh_unchanged_pinned_hash():
    import hashlib
    from tests.vera_test_boundary import START_SH_SHA
    assert hashlib.sha256((ROOT / "start.sh").read_bytes()).hexdigest() == START_SH_SHA
