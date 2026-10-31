"""RENDER-XFF-1031: /diag/header-shape is owner-key protected, config-disabled by default, shapes only."""
import json
from pathlib import Path

import main

ROOT = Path(main.__file__).resolve().parent


def test_ships_disabled():
    assert json.loads((ROOT / "config" / "header_echo.json").read_text())["enabled"] is False


def test_route_needs_owner_key_and_returns_shapes_only(monkeypatch):
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    assert c.get("/diag/header-shape").status_code in (401, 403, 404, 503)
    monkeypatch.setattr(main, "_header_echo_enabled", lambda: True)
    app = main.app
    app.dependency_overrides[main.verify_owner_key] = lambda: None
    try:
        r = c.get("/diag/header-shape", headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1", "X-Echo-Probe": "9.9.9.9", "Via": "x"})
    finally:
        app.dependency_overrides.pop(main.verify_owner_key, None)
    assert r.status_code == 200
    b = r.json()
    assert b["xff_entries"] == 2 and b["leftmost_is_probe"] is True
    assert b["present"]["Via"] is True and b["present"]["X-Real-IP"] is False
    assert "9.9.9.9" not in r.text and "10.0.0.1" not in r.text
