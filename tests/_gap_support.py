"""Helpers for test_dar_pipeline_audit_gaps.py (DAR-PIPE-1022-GAPTESTS). Offline: no socket, DB, env-file read or app server.
Only the scope classifier, the HTTP/DNS boundary, the DB store, the answer model and the claim checker are faked."""
from __future__ import annotations

import json
import re
from pathlib import Path

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests._pipeline_support import Q_EPI, _decide
from vera import acq_config, grounded_ask

REAL_LOAD_CONFIG = acq_config.load_config  # captured at import (before any fixture patches it)
SHIPPED_TEXT = (Path(acq_config.__file__).resolve().parent.parent / "config" / "auto_acquire.json").read_text(encoding="utf-8")


def enabled_config_text(*, override_layer2: bool | None = None) -> str:
    """The shipped config FILE TEXT with only inline.use_discovered flipped false -> true (an owner-style edit of the
    file, not an in-memory dict). Optionally also scope_expansion.override_layer2_declines."""
    # DAR-B-1023: the shipped file now has use_discovered true (ENAB-1023); force true either way, exactly one key.
    t, n = re.subn(r'("use_discovered"\s*:\s*)(?:false|true)', r"\1true", SHIPPED_TEXT, count=1)
    assert n == 1, "shipped config no longer has a single use_discovered key"
    if override_layer2 is not None:
        t, n = re.subn(r'("override_layer2_declines"\s*:\s*)(true|false)', r"\1" + ("true" if override_layer2 else "false"), t, count=1)
        assert n == 1
    return t


def disabled_config_text() -> str:
    """The shipped config FILE TEXT with only inline.use_discovered set false (the rollback-lever state)."""
    t, n = re.subn(r'("use_discovered"\s*:\s*)(?:false|true)', r"\1false", SHIPPED_TEXT, count=1)
    assert n == 1
    return t


def use_config_file(rig, monkeypatch, tmp_path, *, override_layer2: bool = True) -> Path:
    """Write the edited config to a file and make the code under test read it through the REAL acq_config.load_config(path)."""
    p = tmp_path / "auto_acquire.json"
    p.write_text(enabled_config_text(override_layer2=override_layer2), encoding="utf-8")
    loaded = REAL_LOAD_CONFIG(p)
    assert loaded and loaded["inline"]["use_discovered"] is True  # parsed from the file, true boolean
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: REAL_LOAD_CONFIG(p))
    rig.cfg = loaded
    return p


def http_ask(rig, monkeypatch, raws, *, q=Q_EPI, offtopic=True, key="k" * 20):
    """ONE POST /ask through main.app (TestClient). Discovery runs the REAL grounded_ask.retrieve_all; only its provider
    search (`search=` seam) returns `raws` (OpenAlex-shaped provider records). Returns (response, searches)."""
    import main
    from fastapi.testclient import TestClient
    real, searches = grounded_ask.retrieve_all, []
    shipped = grounded_ask.load_grounding() or {}

    def wrapped(question, cfg, **k):
        searches.append({"providers": list(cfg.get("providers") or []), "resolve": dict(cfg.get("resolve") or {})})
        k.pop("search", None)
        return real(question, cfg, search=lambda qq: [dict(r) for r in raws], **k)
    monkeypatch.setattr(main, "retrieve_all", wrapped)
    monkeypatch.setattr(main, "GROUNDING", {**shipped, **e2e.GROUNDING})
    if offtopic:
        _decide(rig.mp, False, "layer2_llm", "llm_out:x", "off_topic")
    else:
        _decide(rig.mp, True, "layer1", "ok", "")
    resp = TestClient(main.app).post("/ask", json={"question": q}, headers={"X-API-Key": key})
    return resp, searches


def stored_text(rig) -> str:
    return "".join(s["text"] for p in rig.store.plans for s in p["spans"])


def injected_jats(raw: bytes, payload: str) -> bytes:
    """Put an injection payload into the JATS BODY (first body paragraph)."""
    head, sep, body = raw.partition(b"<body>")
    assert sep, "fixture JATS has no body"
    return head + sep + body.replace(b"<p>", b"<p>" + payload.encode() + b" ", 1)
