"""TOPIC-ENQ-1105: the ingest and /ask enqueue hooks are inert when the flag is off and non-blocking when on."""
import json
import pathlib
import time

import pytest

from vera import topic_tagging as tt

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _uninstall():
    tt.install(None)
    yield
    tt.install(None)


def _boom(*a, **k):
    raise AssertionError("engine/DB must not be touched")


def test_flag_off_no_enqueue_no_db():
    cfg = json.loads((REPO / "config" / "topic_tagging.json").read_text())
    assert cfg["enabled"] is False
    assert tt.build_worker(cfg, env={}, engine_factory=_boom) is None
    tt.install(None)
    assert tt.notify_ingested(5) is False
    assert tt.schedule_refresh(["a" * 64]) == 0


def test_hooks_wired_in_sources():
    assert "topic_tagging.notify_ingested(sid)" in (REPO / "vera" / "auto_acquire.py").read_text()
    m = (REPO / "main.py").read_text()
    assert "topic_tagging.schedule_refresh(" in m and "_toptag.build_worker(" in m


def test_flag_on_fake_engine_enqueue_does_not_block():
    cfg = {**tt.load_cfg(), "enabled": True, "limits_confirmed": True, "tokens_per_minute": 1000}
    env = {cfg["db_url_env_ro"]: "x://ro", cfg["db_url_env_wo"]: "x://wo"}
    w = tt.build_worker(cfg, env=env, engine_factory=lambda *a, **k: object(),
                        micro_cfg={"registry_table": {"read_from_db": True}})
    assert w is not None
    tt.install(w)  # not started: nothing consumes the inbox, so a blocking hand-off would hang
    t0 = time.monotonic()
    assert tt.notify_ingested(7) is True
    assert tt.schedule_refresh(["b" * 64, "c" * 64, "bad"]) == 2
    assert time.monotonic() - t0 < 0.5
    assert w.inbox.qsize() == 3


# ---- TT-R6-FIX-1105 (convergence round 6) ------------------------------------------------------------------------------------
def test_r6_a1_ask_hook_sends_full_document_hashes():
    """A1: the /ask hook sends examined_full_documents[*].text_sha256 (sources.provenance hash), never passage hashes."""
    import types
    import main as app_main

    calls = []
    orig = tt.schedule_refresh
    tt.schedule_refresh = lambda hashes: calls.append(list(hashes)) or 0
    try:
        ev = types.SimpleNamespace(admitted=({"text_sha256": "p" * 64},),
                                   examined_full_documents=({"document_id": 1, "text_sha256": "d" * 64}, "junk"))
        res = types.SimpleNamespace(evidence=ev, claims=(), result_kind="answer", sources=(), candidates=())
        app_main._after_response(None, None, res, "q")
    finally:
        tt.schedule_refresh = orig
    assert calls == [["d" * 64]]


def test_r6_c4_enabled_must_be_literal_true():
    cfg = {**tt.load_cfg(), "enabled": "yes", "limits_confirmed": True, "tokens_per_minute": 1000}
    env = {cfg["db_url_env_ro"]: "x://ro", cfg["db_url_env_wo"]: "x://wo"}
    assert tt.build_worker(cfg, env=env, engine_factory=_boom, micro_cfg={"registry_table": {"read_from_db": True}}) is None


def test_r6_a2_worker_requires_db_registry_source(caplog):
    """A2: no tagging worker unless micro_topic.registry_table.read_from_db is literally true (tagger and digest same source)."""
    cfg = {**tt.load_cfg(), "enabled": True, "limits_confirmed": True, "tokens_per_minute": 1000}
    env = {cfg["db_url_env_ro"]: "x://ro", cfg["db_url_env_wo"]: "x://wo"}
    for mc in ({}, {"registry_table": {"read_from_db": False}}, {"registry_table": {"read_from_db": "true"}}):
        with caplog.at_level("WARNING", logger="vera"):
            assert tt.build_worker(cfg, env=env, engine_factory=_boom, micro_cfg=mc) is None
    assert "read_from_db" in caplog.text
    # the shipped auto_acquire.json (read_from_db false) blocks the worker too
    assert tt.build_worker(cfg, env=env, engine_factory=_boom) is None
    # read-only paths (--status, --retag-inactive) are not gated: they use the DB active keys, not the seed
    assert tt.build_worker(cfg, env=env, engine_factory=lambda *a, **k: object(), require_limits=False, micro_cfg={}) is not None
