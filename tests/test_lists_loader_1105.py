"""LTD-LOADER-1105: list_tables loader landed with read_from_db OFF. No DB; fake engines only."""
import json
import logging
import threading
from pathlib import Path

import pytest

from vera import list_tables as lt
from vera import source_permissions as sp

ROOT = Path(__file__).resolve().parents[1]
ENV = {"VERA_DB_URL_RO": "postgresql://x/y"}


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class Eng:
    def __init__(self, lists, items, boom=None):
        self.lists, self.items, self.boom, self.connects = lists, items, boom, 0

    def connect(self):
        self.connects += 1
        if self.boom:
            raise self.boom
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        self.rows = self.lists if "lists_v" in sql else self.items
        return self

    def fetchall(self):
        return self.rows


def _spec(shape, **kw):
    d = {"file": "f.json", "path": ["a"], "shape": shape, "state_field": None, "key_field": None,
         "key_lower": False, "combine": "replace"}
    d.update(kw)
    return d


def _cfg(**lists):
    return {"read_from_db": True, "db_url_env": "VERA_DB_URL_RO", "cache_ttl_s": 300, "failure_retry_s": 60, "lists": lists}


@pytest.fixture(autouse=True)
def _reset():
    lt.reset_cache()
    yield
    lt.reset_cache()


def _get(name, c, eng, seed, clock=None):
    return lt.get(name, seed, cfg=c, env=ENV, engine_factory=lambda *a, **k: eng, clock=clock or Clock())


# ---- flag off: byte-identical -------------------------------------------------------------------------------------

def test_shipped_flag_is_off_and_registry_complete():
    c = json.loads((ROOT / "config" / "list_tables.json").read_text())
    # repository.pdf_hosts added by H20 D-CFG (2026-11-05)
    assert c["read_from_db"] is False and len(c["lists"]) == 11
    assert lt._CFG["read_from_db"] is False


def test_overlay_flag_off_equals_input_for_every_real_config_file():
    c = json.loads((ROOT / "config" / "list_tables.json").read_text())
    for fname in sorted({s["file"] for s in c["lists"].values()}):
        p = ROOT / "config" / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        before = json.dumps(data, sort_keys=True)
        out = lt.overlay(fname, data)
        assert json.dumps(out, sort_keys=True) == before, fname
        assert json.dumps(data, sort_keys=True) == before


def test_get_flag_off_returns_seed_object_without_touching_db():
    seed = ["x"]
    assert lt.get("news.fallback_order", seed, env={}) is seed
    assert lt.get("unknown", seed) is seed


def test_only_literal_true_enables():
    eng = Eng([("l", "strings", "replace", 1)], [("l", "db", 1, 1, {})])
    for flag in ("true", 1, "yes", None, False):
        lt.reset_cache()
        c = dict(_cfg(l=_spec("strings")), read_from_db=flag)
        assert _get("l", c, eng, ["seed"]) == ["seed"]
    assert eng.connects == 0


def test_patched_readers_flag_off_equal_file_values():
    assert sp.load_config() == json.loads((ROOT / "config" / "source_permissions.json").read_text())
    reg = json.loads((ROOT / "config" / "entity_registry.json").read_text())
    assert sp.load_registry() == sp._rows(reg)
    assert sp._rows(None) == sp._rows(reg)


# ---- flag on, fake engine -----------------------------------------------------------------------------------------

def test_flag_on_db_rows_used_and_union_keeps_seed():
    eng = Eng([("s", "strings", "replace", 1), ("d", "strings", "union", 1)],
              [("s", "b", 2, 1, {}), ("s", "a", 1, 1, {}), ("d", "evil.com", 1, 1, {})])
    c = _cfg(s=_spec("strings"), d=_spec("strings", combine="union"))
    assert _get("s", c, eng, ["seed"]) == ["a", "b"]
    assert _get("d", c, eng, ["bad.org", "evil.com"]) == ["evil.com", "bad.org"]


def test_flag_on_via_patched_reader(monkeypatch):
    orig = lt._fetch
    eng = Eng([("permissions.deny_hosts", "strings", "union", 1)], [("permissions.deny_hosts", "dbhost.example", 1, 1, {})])
    monkeypatch.setattr(lt, "_CFG", dict(json.loads((ROOT / "config" / "list_tables.json").read_text()), read_from_db=True))
    monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://x/y")
    monkeypatch.setattr(lt, "_fetch", lambda cfg, env, f: orig(cfg, env, lambda *a, **k: eng))
    file_hosts = json.loads((ROOT / "config" / "source_permissions.json").read_text()).get("deny_hosts", [])
    got = sp.load_config()["deny_hosts"]
    assert "dbhost.example" in got and all(h in got for h in file_hosts)


STAGES = [("t", "s", 1, 1, {"stage": "s", "reason": "r"}), ("t", "s:k", 2, 1, {"stage": "s", "key": "k"})]
SEED_NESTED = {"s": {"keys": ["k"], "reason": "r", "nested": {"p": ["n1"], "q": ["m1"]}}}


def test_f3_missing_nested_parent_gives_empty_list():
    rows = STAGES + [("t", "s:p.n1", 3, 1, {"stage": "s", "parent": "p", "key": "n1"})]
    out = _get("t", _cfg(t=_spec("stage_keys")), Eng([("t", "stage_keys", "replace", 1)], rows), SEED_NESTED)
    assert out == {"s": {"keys": ["k"], "reason": "r", "nested": {"p": ["n1"], "q": []}}}


def test_f3_all_nested_retired_gives_empty_parents_not_missing():
    out = _get("t", _cfg(t=_spec("stage_keys")), Eng([("t", "stage_keys", "replace", 1)], STAGES), SEED_NESTED)
    assert out["s"]["nested"] == {"p": [], "q": []}


def test_db_error_falls_back_to_config_with_log_line(caplog):
    eng = Eng([], [], boom=RuntimeError("secret-url-should-not-appear"))
    with caplog.at_level(logging.WARNING, logger="vera.list_tables"):
        assert _get("l", _cfg(l=_spec("strings")), eng, ["seed"]) == ["seed"]
    assert "LT_DB_ERROR (RuntimeError)" in caplog.text and "secret-url" not in caplog.text


def test_f7_malformed_log_rate_limited(caplog):
    eng = Eng([("a", "records", "replace", 1)], [("a", "k", 1, 1, {"id": "other"})])
    c = _cfg(a=_spec("records", key_field="id"))
    clk = Clock()
    with caplog.at_level(logging.WARNING, logger="vera.list_tables"):
        for _ in range(5):
            assert _get("a", c, eng, "SEED", clk) == "SEED"
        assert caplog.text.count("LT_LIST_ERROR") == 1
        clk.t += 61
        _get("a", c, eng, "SEED", clk)
        assert caplog.text.count("LT_LIST_ERROR") == 2


def test_f7_single_flight_other_threads_do_not_block_on_refresh():
    started, release = threading.Event(), threading.Event()

    class Slow(Eng):
        def connect(self):
            started.set()
            release.wait(5)
            return super().connect()

    eng = Slow([("l", "strings", "replace", 1)], [("l", "db", 1, 1, {})])
    c = _cfg(l=_spec("strings"))
    res = {}
    t = threading.Thread(target=lambda: res.update(first=_get("l", c, eng, ["seed"])))
    t.start()
    assert started.wait(5)
    assert _get("l", c, eng, ["seed"]) == ["seed"]  # returns immediately with the seed while the refresh is in flight
    release.set()
    t.join(5)
    assert res["first"] == ["db"] and eng.connects == 1


# ---- LTD-TRACE-HUNK-1105: request_trace key_allowlist via the loader ----------------------------------------------

TSTAGES = [("trace.key_allowlist",) + r[1:] for r in STAGES]


def _rt_on(monkeypatch, eng):
    from vera import request_trace as rt
    orig = lt._fetch
    monkeypatch.setattr(lt, "_CFG", dict(json.loads((ROOT / "config" / "list_tables.json").read_text()), read_from_db=True))
    monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://x/y")
    monkeypatch.setattr(lt, "_fetch", lambda cfg, env, f: orig(cfg, env, lambda *a, **k: eng))
    return rt


def test_trace_flag_off_gives_config_allowlist():
    from vera import request_trace as rt
    assert rt._key_allowlist() is rt.CFG.get("key_allowlist")
    for st, ent in rt.CFG["key_allowlist"].items():
        if not isinstance(ent, dict):
            continue
        assert rt._stage_keys(st) == frozenset(k for k in ent["keys"] if isinstance(k, str))


def test_trace_flag_on_gives_db_rows(monkeypatch):
    eng = Eng([("trace.key_allowlist", "stage_keys", "replace", 1)],
              TSTAGES + [("trace.key_allowlist", "s:p.n1", 3, 1, {"stage": "s", "parent": "p", "key": "n1"})])
    rt = _rt_on(monkeypatch, eng)
    assert rt._stage_keys("s") == frozenset({"k"})
    assert rt._nested_keys("s", "p") == ["n1"] or set(rt._nested_keys("s", "p")) == {"n1"}


def test_trace_missing_parent_gives_empty(monkeypatch):
    seed = {"s": {"keys": ["k"], "reason": "r", "nested": {"p": ["n1"], "q": ["m1"]}}}
    eng = Eng([("trace.key_allowlist", "stage_keys", "replace", 1)],
              TSTAGES + [("trace.key_allowlist", "s:p.n1", 3, 1, {"stage": "s", "parent": "p", "key": "n1"})])
    rt = _rt_on(monkeypatch, eng)
    monkeypatch.setitem(rt.CFG, "key_allowlist", seed)
    assert list(rt._nested_keys("s", "q") or []) == []
    assert lt.get("trace.key_allowlist", seed)["s"]["nested"]["q"] == []


def test_trace_db_error_falls_back_to_seed(monkeypatch):
    eng = Eng([], [], boom=RuntimeError("down"))
    rt = _rt_on(monkeypatch, eng)
    assert rt._key_allowlist() == rt.CFG.get("key_allowlist")
    assert rt._stage_keys("nonexistent") == frozenset()
