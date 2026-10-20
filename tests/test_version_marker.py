"""Deployed-commit marker: /version, helper, and UI footer wiring."""
from pathlib import Path

from fastapi.testclient import TestClient

from vera.version_info import UNKNOWN, deployed_commit, short_commit

SHA = "d46c726b9da53e2e8d86d574b72e1394ad1c32b1"


def test_deployed_commit_reads_render_env():
    assert deployed_commit({"RENDER_GIT_COMMIT": SHA}) == SHA
    assert short_commit(SHA) == "d46c726"


def test_unset_or_malformed_is_unknown():
    for env in ({}, {"RENDER_GIT_COMMIT": ""}, {"RENDER_GIT_COMMIT": "https://x.onrender.com"}, {"RENDER_GIT_COMMIT": "zz"}):
        assert deployed_commit(env) == UNKNOWN
    assert short_commit(UNKNOWN) == UNKNOWN


def test_version_endpoint(monkeypatch):
    import main
    c = TestClient(main.app)
    monkeypatch.setenv("RENDER_GIT_COMMIT", SHA)
    r = c.get("/version")
    assert r.status_code == 200 and r.json() == {"commit": SHA, "short": "d46c726"}
    monkeypatch.delenv("RENDER_GIT_COMMIT")
    assert c.get("/version").json() == {"commit": "unknown", "short": "unknown"}


def test_version_exposes_nothing_else(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", SHA)
    monkeypatch.setenv("VERA_API_KEY", "secret-sentinel")
    import main
    body = TestClient(main.app).get("/version").text
    assert "secret-sentinel" not in body and "onrender" not in body


def test_ui_footer_shows_short_commit():
    src = (Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text()
    assert "VERA-DAR build:" in src and "short_commit" in src
