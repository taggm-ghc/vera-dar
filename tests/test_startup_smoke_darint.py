"""DARINT startup smoke (PRB-DARINT-002 wiring): API /health (in-process), import-time env audit, UI render, UI-1 warning.
Runs only inside scripts/vera_test_runner.py. No DB keys are ever set; no value is printed.
D-DARINT-043 (owner, OD-DARINT-SMOKE-TEST option A): this file launches NO subprocess. The INT4 boundary refuses loopback
TCP and child processes in tests, so the real-server (uvicorn) and real-Streamlit startup checks are skipped and
explicitly UNVERIFIED here; the API is checked in-process through its ASGI app instead. INT4 protections are unchanged."""
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
_DB_URL_RE = re.compile(r"postgres(?:ql)?(?:\+\w+)?://", re.I)
_UNVERIFIED_REAL = ("UNVERIFIED (INT4 boundary): a real server needs a child process and loopback TCP, which the test "
                    "boundary refuses by design; not bypassed (D-DARINT-043). The in-process checks in this file stand in.")


def _drop_db_keys(monkeypatch):
    for k in list(os.environ):
        if fnmatch.fnmatch(k, "*DB_URL*") or k == "DATABASE_URL" or k.startswith("PG"):
            monkeypatch.delenv(k)


def test_a_api_health_ok_in_process(monkeypatch):
    """In-process replacement for the real /health server check (owner D-DARINT-043)."""
    _drop_db_keys(monkeypatch)
    from fastapi.testclient import TestClient
    import main
    r = TestClient(main.app).get("/health")
    assert r.status_code == 200, r.text[:300]
    assert r.json().get("status") == "ok"
    assert not _DB_URL_RE.search(r.text), "a DB URL value appeared in the /health response"


@pytest.mark.skip(reason=_UNVERIFIED_REAL)
def test_a2_real_uvicorn_health():
    """Real `uvicorn main:app` on 127.0.0.1 + /health poll: UNVERIFIED inside the runner (see module docstring)."""


def test_b_import_main_no_db_keys(monkeypatch):
    _drop_db_keys(monkeypatch)
    import main  # noqa: F401
    from vera import env_precedence as ep
    audit = ep.load_env()                              # real environ: returns the first (import-time) audit
    d = audit.as_dict()
    assert d["context"] and d["selected_by"]
    restricted = [n for n in list(d["sources"]) + d["applied"]
                  if fnmatch.fnmatch(n, "*DB_URL*") or n == "DATABASE_URL" or n.startswith("PG")]
    assert restricted == [], f"restricted names loaded: {restricted}"


def _apptest(monkeypatch, public, api_down=False):
    from streamlit.testing.v1 import AppTest
    import requests
    if public:
        monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
        monkeypatch.setenv("VERA_API_KEY", "k")
    else:
        monkeypatch.setenv("VERA_PUBLIC_MODE", "0")
        monkeypatch.delenv("VERA_API_KEY", raising=False)
    monkeypatch.delenv("VERA_UI_AUTOSTART_API", raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    if api_down:                                     # nothing listening: health probe sees ConnectionError
        def _down(*a, **k):
            raise requests.ConnectionError("down")
        monkeypatch.setattr(requests, "get", _down)
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()


def test_c_ui_dev_renders(monkeypatch):
    at = _apptest(monkeypatch, public=False)
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert at.title


@pytest.mark.skip(reason=_UNVERIFIED_REAL)
def test_c2_real_streamlit_health():
    """Real `streamlit run` + /_stcore/health poll: UNVERIFIED inside the runner (see module docstring)."""


UI1 = "api_kickstart" in (ROOT / "streamlit_app.py").read_text()


@pytest.mark.skipif(not UI1, reason="UI-1 diff not applied in this variant")
def test_d_ui1_autostart_off_warns_and_spawns_nothing(monkeypatch):
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(RuntimeError("spawn")))
    monkeypatch.setattr(os, "posix_spawn", lambda *a, **k: spawned.append(a), raising=False)
    at = _apptest(monkeypatch, public=False, api_down=True)
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    warns = [str(w.value) for w in at.sidebar.warning]
    assert any("auto-start is off" in w for w in warns), warns
    assert spawned == []
