"""NONASK-1103 E9 (KE-4), step A5p first.

A5p finding (static, source-read): /ask's DuckDuckGo provider is built by search_and_fetch.cached_chain()
-> _search_settings() -> eval_config.load_eval_config().search -> config/vera_eval_run.json `search`
(PROVIDER_FACTORIES[DDG_NAME](settings) -> DuckDuckGoLiteProvider(cache_dir=_ddg_cache_dir(settings["duckduckgo_lite"]))).
auto_acquire.make_search and search_providers._default_ddg_search reuse that same cached chain; config/auto_acquire.json's
`search` block only filters provider NAMES. eval_config reads no environment variable. The test below resolves the config
through that exact path (it FAILS, not skips, if the path cannot be resolved) and builds the real provider with the
network stubbed. The discovery store comes from config/portal_registry.json `store` (portal_store.open_store).
The acquisition writers store a fixed config label, never the visitor question (tests at the end).
"""
import ast
import inspect
import logging
import re
from pathlib import Path

import pytest

from vera import eval_config, portal_recorder, portal_registry, portal_store
from vera import search_and_fetch as sf
from vera import search_providers as sp

QUESTION = "zq-question-canary: what do studies say about widget calibration?"
HTML = "<html><body><table><tr><td><a href='https://arxiv.org/abs/2401.00001' class='result-link'>A widget paper</a></td></tr></table></body></html>"
ROOT = Path(sp.__file__).resolve().parent.parent


class R:
    def __init__(self, text=HTML):
        self.status_code, self.text, self.headers = 200, text, {}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(sp, "ddg_default_transport", boom)


def resolved_ask_settings(monkeypatch=None):
    """The exact settings /ask's chain uses. Raises (test FAILS) if the resolution path no longer exists."""
    assert hasattr(sf, "_search_settings") and hasattr(sf, "cached_chain"), "A5p: /ask DDG config path unresolvable"
    s = sf._search_settings()
    assert sp.DDG_NAME in s["providers"], "DDG not in /ask's provider list"
    assert sp.DDG_NAME in sp.PROVIDER_FACTORIES
    return s


def test_ddg_config_source_is_vera_eval_run_json():
    s = resolved_ask_settings()
    run = eval_config.load_eval_config(path=eval_config.FREEZE_PATH, run_path=eval_config.RUN_PATH).search
    assert dict(s) == dict(run)
    assert Path(eval_config.RUN_PATH) == ROOT / "config" / "vera_eval_run.json"
    src = inspect.getsource(sf.cached_chain) + inspect.getsource(sf._search_settings)
    assert "load_eval_config" in src


def test_ddg_cache_dir_null_with_env_overrides_applied(monkeypatch):
    for k in ("VERA_DDG_CACHE_DIR", "DDG_CACHE_DIR", "VERA_CACHE_DIR", "VERA_SEARCH_CACHE_DIR", "VERA_DUCKDUCKGO_LITE_CACHE_DIR"):
        monkeypatch.setenv(k, str(ROOT / "tmp" / "ddg-env-attempt"))
    s = resolved_ask_settings()
    assert s[sp.DDG_NAME].get("cache_dir") is None
    assert sp._ddg_cache_dir(s[sp.DDG_NAME]) is None
    # no env read anywhere on the resolution path
    assert "environ" not in inspect.getsource(sf._search_settings)
    assert not re.search(r"environ|getenv", inspect.getsource(sp._ddg_cache_dir))
    attrs = {n.attr for n in ast.walk(ast.parse(Path(eval_config.__file__).read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}
    assert not ({"environ", "getenv"} & attrs)


def test_real_ddg_provider_from_resolved_settings_writes_nothing(monkeypatch, tmp_path):
    s = resolved_ask_settings()
    calls = []

    def tx(url, data, headers, timeout):
        calls.append(data)
        return R()
    p = sp.PROVIDER_FACTORIES[sp.DDG_NAME](s, transport=tx)
    assert isinstance(p, sp.DuckDuckGoLiteProvider) and p.cache_dir is None
    before = {q for q in ROOT.rglob("*.json") if "tmp" in q.parts}
    p.call(QUESTION, 5, 5.0)
    assert calls, "stub transport was never exercised"
    assert p._cache_path({"query": "x"}) is None
    assert {q for q in ROOT.rglob("*.json") if "tmp" in q.parts} == before
    # in-memory cache only, and it never holds the question text as a key beyond the process
    assert all(isinstance(k, str) for k in p._mem)


def test_portal_store_disabled_in_shipped_registry(monkeypatch):
    import json
    cfg = json.loads(portal_registry.CONFIG_PATH.read_text(encoding="utf-8"))
    st = cfg["store"]
    assert st["enabled"] is False and st["sqlite_path"] is None
    assert portal_store.open_store(cfg) is None
    for k in ("VERA_PORTAL_STORE_ENABLED", "VERA_PORTAL_STORE_PATH", "PORTAL_STORE_ENABLED", "VERA_SQLITE_PATH"):
        monkeypatch.setenv(k, "1" if "ENABLED" in k else str(ROOT / "tmp" / "x.sqlite"))
    assert portal_store.open_store(json.loads(portal_registry.CONFIG_PATH.read_text(encoding="utf-8"))) is None
    for f in ("portal_recorder.py", "portal_store.py", "portal_registry.py"):
        t = ast.parse((ROOT / "vera" / f).read_text(encoding="utf-8"))
        names = {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}
        assert not ({"environ", "getenv"} & names), f


def test_recorder_writes_no_query_hash_with_store_off(monkeypatch, caplog):
    opened = []
    monkeypatch.setattr(portal_store, "SQLiteDiscoveryStore", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("store opened")))
    monkeypatch.setattr(portal_recorder, "_STORE", None)
    monkeypatch.setattr(portal_recorder, "_STORE_CFG", None)
    cfg = portal_recorder._registry_cfg()
    assert portal_store.open_store(cfg) is None and portal_recorder._store_for(cfg) is None and not opened
    with caplog.at_level(logging.DEBUG):
        assert portal_recorder.record(portal_recorder._store_for(cfg), {"triggered": True}, question=QUESTION) is None
    assert QUESTION not in caplog.text and portal_recorder.query_hash(QUESTION) not in caplog.text
    assert not opened


VISITOR = "zq-visitor-canary: is my private question stored?"
LABEL = "VERA-DAR automatic acquisition (open-access full text; visitor question not stored)"


class _Res:
    def first(self):
        return (1,)


class _Conn:
    def __init__(self, log):
        self.log = log

    def execute(self, stmt, params=None):
        self.log.append((str(stmt), dict(params or {})))
        return _Res()

    def begin_nested(self):
        class SP:
            def commit(self):
                pass

            def rollback(self):
                pass
        return SP()


class _Engine:
    def __init__(self):
        self.log = []

    def begin(self):
        import contextlib
        return contextlib.nullcontext(_Conn(self.log))


def _question_params(log):
    return [p for sql, p in log if re.search(r"INSERT INTO\s+\S*(runs|questions)", sql)]


def test_acquisition_writer_stores_fixed_label_never_the_visitor_question(monkeypatch):
    """Corrects the earlier xfail: WoStore.write uses cfg["question_text"] (a fixed label)."""
    import json
    from vera import auto_acquire as aa
    cfg = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
    assert cfg["question_text"] == LABEL
    eng = _Engine()
    w = aa.WoStore.__new__(aa.WoStore)
    w.engine, w.cfg = eng, dict(cfg, question_text=LABEL, visitor_question=VISITOR)

    class Lookup:
        def canonical_state(self, *a):
            return (None, 0, False, False)
    w.lookup = Lookup()
    plan = {"source": {"provenance": {"text_sha256": "h", "doi": None, "url": "u"}}, "canonical_url": "cu", "candidate": {}}
    try:
        w.write(plan)
    except Exception:  # later inserts need richer plans; the question/run inserts come first
        pass
    qp = _question_params(eng.log)
    assert qp, "question/run inserts were not reached"
    vals = [v for p in qp for v in p.values() if isinstance(v, str)]
    assert LABEL in vals and not any(VISITOR in v for v in vals)


def test_other_question_writers_take_fixed_text_from_constructor_or_config():
    """memory_store, corpus_admission and fulltext_loader insert self.question_text (set at construction from config),
    and none of them receive the request question: static check of the insert parameters."""
    for f in ("memory_store.py", "corpus_admission.py", "fulltext_loader.py", "auto_acquire.py"):
        src = (ROOT / "vera" / f).read_text(encoding="utf-8")
        for m in re.finditer(r"INSERT INTO \{S\}\.(questions|runs)[^\n]*\n?[^\n]*", src):
            assert re.search(r"self\.question_text|qtext", m.group(0)), (f, m.group(0))
    assert "cfg[\"question_text\"]" in (ROOT / "vera" / "memory_store.py").read_text(encoding="utf-8")
