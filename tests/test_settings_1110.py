"""SETTINGS-W3-1110: loader tests with a fake DB (injected fetch/clock/spawn); no network, no real DB."""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vera import settings as st

ROOT = Path(__file__).resolve().parent.parent
NOW = 1_800_000_000.0
DT = datetime.fromtimestamp(NOW, timezone.utc)

REG = {
    "m.cap": {"module": "m", "path": "cap", "type": "int", "min": 1, "max": 100, "clamp": "lower_only", "validator": True},
    "m.floor": {"module": "m", "path": "floor", "type": "float", "min": 0.0, "max": 1.0, "clamp": "raise_only"},
    "m.flag": {"module": "m", "path": "sub.flag", "type": "bool", "allowed": [True, False], "clamp": None},
    "m.open": {"module": "m", "path": "open", "type": "int_or_null", "min": 1, "max": 50, "clamp": None},
}
FILE = {"cap": 10, "floor": 0.5, "sub": {"flag": False}, "open": None, "hosts": ["a.org", "b.org", "c.org"]}


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    clk = Clock()
    box = {"s": [], "d": [], "err": None, "calls": 0}

    def fetch():
        box["calls"] += 1
        if box["err"]:
            raise box["err"]
        return list(box["s"]), list(box["d"])

    monkeypatch.setattr(st, "_FETCH", fetch)
    monkeypatch.setattr(st, "_SPAWN", lambda fn: fn())
    monkeypatch.setattr(st, "_CLOCK", clk)
    monkeypatch.setattr(st, "_WALL", lambda: NOW)
    st.reset(cfg={"read_from_db": {"m": True}}, registry=REG)
    box["clk"] = clk
    yield box
    st.reset(cfg={"read_from_db": {}}, registry={})


def row(key, val, ver=1):
    return (key, val, ver)


def deny(lst, pat, days=10, wild=False, created=None):
    c = created or DT
    return (lst, pat, c, c + timedelta(days=days), wild)


def test_allowlist_enforcement_non_registry_key_ignored(env, caplog):
    env["s"] = [row("m.secret", 1), row("licence_rules.x", 1)]
    assert st.get("m", "secret", 5) == 5
    assert st.get("licence_rules", "x", 5) == 5
    assert st.overlay("m", dict(FILE)) == FILE


def test_db_value_used_when_valid(env):
    env["s"] = [row("m.cap", 7), row("m.floor", 0.6), row("m.flag", True), row("m.open", 20)]
    out = st.overlay("m", FILE)
    assert (out["cap"], out["floor"], out["sub"]["flag"], out["open"]) == (7, 0.6, True, 20)
    assert FILE["cap"] == 10 and FILE["sub"]["flag"] is False  # input never mutated


def test_one_way_caps(env):
    env["s"] = [row("m.cap", 50), row("m.floor", 0.2)]  # cap raise and floor lower are both refused
    assert st.get("m", "cap", 10) == 10
    assert st.get("m", "floor", 0.5) == 0.5
    env["clk"].t = 1000
    env["s"] = [row("m.cap", 3), row("m.floor", 0.9)]
    assert st.get("m", "cap", 10) == 3 and st.get("m", "floor", 0.5) == 0.9


def test_invalid_rows_ignored(env):
    for bad in ("7", True, 7.5, 0, 101, None, [1]):
        st.reset(cfg={"read_from_db": {"m": True}}, registry=REG)
        env["s"] = [row("m.cap", bad)]
        assert st.get("m", "cap", 10) == 10, bad
    st.reset(cfg={"read_from_db": {"m": True}}, registry=REG)
    env["s"] = [row("m.flag", "yes"), row("m.floor", float("nan"))]
    assert st.get("m", "sub.flag", False) is False and st.get("m", "floor", 0.5) == 0.5


def test_module_validator_refusal_falls_back_to_file(env):
    env["s"] = [row("m.cap", 7)]

    def v(cfg):
        raise ValueError("no")
    assert st.overlay("m", FILE, validate=v) is FILE
    assert st.overlay("m", FILE, validate=lambda c: True)["cap"] == 7


def test_deny_only_lists(env):
    env["d"] = [deny("m.hosts", "B.org"), deny("m.hosts", "evil.com")]
    assert st.filter_allow("m.hosts", FILE["hosts"], "m") == ["a.org", "c.org"]
    # a deny entry for a host not in the allow-list adds nothing: no allow path exists
    assert "evil.com" not in st.filter_allow("m.hosts", FILE["hosts"], "m")
    assert st.extend_deny("m.hosts", ["x.org"], "m") == ["x.org", "b.org", "evil.com"]


def test_deny_rows_expired_overlong_or_unconfirmed_wildcard_ignored(env):
    old = DT - timedelta(days=40)
    env["d"] = [deny("m.hosts", "a.org", created=old, days=5),      # expired
                deny("m.hosts", "b.org", days=100),                 # TTL over the 90-day hard maximum
                deny("m.hosts", "*.org", wild=False),               # wildcard without confirmation
                ("m.hosts", "c.org", DT, DT + timedelta(days=1)),   # malformed / allow-shaped (no wildcard field)
                deny("m.hosts", "", days=1)]
    assert st.filter_allow("m.hosts", FILE["hosts"], "m") is FILE["hosts"]
    env["clk"].t = 1000
    env["d"] = [deny("m.hosts", "*.org", wild=True)]
    assert st.filter_allow("m.hosts", FILE["hosts"], "m") == []
    assert st.extend_deny("m.hosts", ["x"], "m") == ["x"]  # wildcards never extend a deny list


def test_db_down_keeps_last_good_snapshot_and_denies(env, caplog):
    env["s"] = [row("m.cap", 4)]
    env["d"] = [deny("m.hosts", "a.org")]
    assert st.get("m", "cap", 10) == 4
    env["err"] = RuntimeError("down")
    env["clk"].t = 100
    with caplog.at_level(logging.WARNING):
        assert st.get("m", "cap", 10) == 4
        assert st.filter_allow("m.hosts", FILE["hosts"], "m") == ["b.org", "c.org"]
        env["clk"].t = 120
        st.get("m", "cap", 10)
    assert len([r for r in caplog.records if "SET_DB_ERROR" in r.message]) == 1  # warn once


def test_db_down_no_snapshot_uses_file(env):
    env["err"] = TimeoutError()
    assert st.get("m", "cap", 10) == 10
    assert st.filter_allow("m.hosts", FILE["hosts"], "m") is FILE["hosts"]
    assert st.overlay("m", FILE) is FILE


def test_flags_off_byte_identical_and_no_db_touch(env):
    env["s"] = [row("m.cap", 3)]
    env["d"] = [deny("m.hosts", "a.org")]
    st.reset(cfg={"read_from_db": {"m": False}}, registry=REG)
    assert st.get("m", "cap", 10) == 10
    assert st.overlay("m", FILE) is FILE
    assert st.filter_allow("m.hosts", FILE["hosts"], "m") is FILE["hosts"]
    assert st.version() is None and env["calls"] == 0


def test_shipped_flags_all_off():
    assert json.loads((ROOT / "config" / "settings.json").read_text())["read_from_db"] == {}
    assert st._CFG["read_from_db"] == {} and not st.any_enabled()


def test_single_flight_and_ttl(env, monkeypatch):
    pending = []
    monkeypatch.setattr(st, "_SPAWN", pending.append)
    env["s"] = [row("m.cap", 4)]
    assert st.get("m", "cap", 10) == 10          # first call never blocks: file value, refresh queued
    st.get("m", "cap", 10); st.get("m", "cap", 10)
    assert len(pending) == 1                       # single flight
    pending.pop()()
    assert st.get("m", "cap", 10) == 4 and env["calls"] == 1
    env["clk"].t = 59
    st.get("m", "cap", 10)
    assert not pending
    env["clk"].t = 61
    st.get("m", "cap", 10)
    assert len(pending) == 1                       # TTL elapsed -> one refresh


def test_change_logged_and_version(env, caplog):
    with caplog.at_level(logging.INFO):
        env["s"] = [row("m.cap", 4, 1)]
        st.get("m", "cap", 10)
        v1 = st.version()
        env["clk"].t = 100
        env["s"] = [row("m.cap", 5, 2)]
        st.get("m", "cap", 10)
    msgs = [r.getMessage() for r in caplog.records if "settings.changed" in r.getMessage()]
    assert any("key=m.cap file->1" in m for m in msgs) and any("key=m.cap 1->2" in m for m in msgs)
    assert v1 and st.version() != v1


def test_trace_start_records_settings_version(env, monkeypatch):
    from vera import request_trace as rt
    lines = []
    monkeypatch.setattr(rt, "_log_line", lines.append)
    monkeypatch.setitem(rt.CFG, "enabled", True)
    env["s"] = [row("m.cap", 4, 1)]
    st.get("m", "cap", 10)
    if rt.enabled():
        rt.start("q")
        assert lines and lines[0].get("settings_version") == st.version()
        rt.end_context()


def test_registry_pin_and_static_checks(tmp_path):
    reg = json.loads((ROOT / "config" / "settings_registry.json").read_text())
    assert st.registry_digest(reg["keys"]) == reg["sha256"] == st.REGISTRY_SHA256
    assert st.load_registry() == reg["keys"]
    tampered = dict(reg["keys"]); tampered["x.y"] = {"module": "x", "path": "y", "type": "bool", "allowed": [True]}
    p = tmp_path / "tampered_registry.json"
    p.write_text(json.dumps({"keys": tampered}))
    assert st.load_registry(p) == {}
    for k, e in reg["keys"].items():
        assert e["module"] not in st.REFUSED_STEMS and not k.startswith(st.REFUSED_PREFIXES), k
        cfg = json.loads((ROOT / e["file"]).read_text())
        for part in e["path"].split("."):
            cfg = cfg[part]  # key exists in the shipped file
        if e["type"] != "bool":
            assert isinstance(e["min"], (int, float)) and isinstance(e["max"], (int, float)), k
            if cfg is not None:
                assert e["min"] <= cfg <= e["max"], k


def test_chunk_taxo_upper_bounds_and_nullable_bounds():
    keys = json.loads((ROOT / "config" / "settings_registry.json").read_text())["keys"]
    for k in ("k_min", "k_max", "patience", "min_support", "term_count", "kmeans_max_iter"):
        assert keys["chunk_taxo.params." + k]["max"] is not None
    nul = [k for k, e in keys.items() if e["type"].endswith("_or_null")]
    assert nul and all(keys[k]["min"] is not None and keys[k]["max"] is not None for k in nul)


def test_chunk_taxo_overlay_stays_valid_with_real_validator(env):
    from vera import chunk_taxo
    reg = st.load_registry()
    st.reset(cfg={"read_from_db": {"chunk_taxo": True}}, registry=reg)
    env["s"] = [row("chunk_taxo.params.patience", 3), row("chunk_taxo.params.k_max", 99999)]
    raw = json.loads((ROOT / "config" / "chunk_taxo.json").read_text())
    out = st.overlay("chunk_taxo", raw)
    assert out["params"]["patience"] == 3 and out["params"]["k_max"] == raw["params"]["k_max"]
    assert chunk_taxo.load_cfg is not None


def test_sql_matches_migration_026_shapes():
    assert "set_settings" in st.SQL_SETTINGS and "set_keys" in st.SQL_SETTINGS and "'active'" in st.SQL_SETTINGS
    assert "set_deny_active" in st.SQL_DENY and "wildcard_confirmed" in st.SQL_DENY
    assert st._CFG is not None and json.loads((ROOT / "config" / "settings.json").read_text())["deny_max_ttl_days"] == 90
