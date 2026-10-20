"""scripts/check_config.py (PLAN-ENV v2.1 §4, §5) on a synthetic layout under tmp_path only. Never touches the real
R/config/env. Zero socket calls, no connect call, no git child (--no-git), values never printed."""
import builtins
import importlib.util
import io
import json
import pathlib
import secrets
import socket
from pathlib import Path

import pytest

from safety import endpoint_class as ec

ROOT = Path(__file__).resolve().parents[1]
GLOBS = '(".env", ".env.*", "*.env", ".env-*", "*.env-*")'


def _load():
    spec = importlib.util.spec_from_file_location("_t_check_config", ROOT / "scripts" / "check_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_layout(tmp_path, *, vera_text=None, intern_text=None):
    h = secrets.token_hex(4)
    secrets_ = {"pw": f"cnry-{secrets.token_hex(8)}", "host": f"canary-prod-{h}.invalid", "db": f"canaryprod_{h}",
                "role": f"canary_rw_{h}"}
    r = tmp_path / "Rfake"
    repo = r / "vera"
    for d in (repo / "scripts", repo / "tests", repo / "db" / "accounts", repo / "vera", repo / ".git",
              r / "config" / "env", r / "work"):
        d.mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "vera_test_runner.py").write_text(f"CRED_MASK_GLOBS = {GLOBS}\n")
    (repo / "tests" / "vera_test_boundary.py").write_text(f"CRED_FILE_GLOBS = {GLOBS}\n")
    (repo / "db" / "accounts" / "create_vera_accounts.sql").write_text("-- roles only\n")
    (repo / "vera" / "x.py").write_text("X = 1\n")
    url = f"postgresql://{secrets_['role']}:{secrets_['pw']}@{secrets_['host']}:5432/{secrets_['db']}"
    env = r / "config" / "env"
    (env / ".env-vera").write_text(vera_text if vera_text is not None else
                                   f"VERA_DB_URL_RO={url}\nVERA_DB_URL_WO={url}\nOPENAI_API_KEY=sk-{secrets_['pw']}\n")
    (env / ".env-intern").write_text(intern_text if intern_text is not None else
                                     f"DB_HOST={secrets_['host']}\nDB_NAME={secrets_['db']}\nDB_ACCOUNT={secrets_['role']}\n")
    for f in (".env-vera", ".env-intern"):
        (env / f).chmod(0o600)
    return r, repo, secrets_


def _run(mod, r, repo, *extra, environ=None, capsys=None):
    out = io.StringIO()
    rc = mod.main(["--root", str(r), "--repo", str(repo), "--no-git", *extra], environ=environ or {}, out=out)
    return rc, out.getvalue()


@pytest.fixture
def no_net(monkeypatch):
    calls = []

    def deny(*a, **k):
        calls.append("net")
        raise OSError("no network in this test")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    import psycopg2
    monkeypatch.setattr(psycopg2, "_connect", lambda *a, **k: calls.append("db"))
    return calls


def test_clean_layout_exit_0_names_only(tmp_path, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path)
    rc, text = _run(mod, r, repo)
    assert rc == 0, text
    assert "names: OPENAI_API_KEY VERA_DB_URL_RO VERA_DB_URL_WO" in text
    assert "class VERA_DB_URL_RO .env-vera PRODUCTION" in text and "DISPOSABLE_VERIFIED" not in text
    assert "parse VERA_DB_URL_RO .env-vera ok" in text
    for v in s.values():
        assert v not in text
    assert no_net == []


def test_json_mode_and_db_side_unverified(tmp_path, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path)
    rc, text = _run(mod, r, repo, "--json", environ={"VERA_DB_URL_RO": "postgresql://u:p@other.test/db",
                                                     "PGHOST": "x", "HTTPS_PROXY": "y"})
    rep = json.loads(text)
    assert rc in (0, 1) and rep["findings"] == [f for f in rep["findings"] if "VERA_DB_URL_RO (process)" not in f]
    assert {d["status"] for d in rep["db_side"]} == {"UNVERIFIED"}
    assert all(c["cls"] == ec.PRODUCTION for c in rep["classification"])
    assert set(rep["c3_preview"]["would_strip"]) == {"VERA_DB_URL_RO", "PGHOST", "HTTPS_PROXY"}
    assert rep["c3_preview"]["would_override"] == ["VERA_DB_URL_RO"]
    assert any(c["source"] == "process" and "pg_env" in c["reasons"] for c in rep["classification"])
    for v in s.values():
        assert v not in text
    assert no_net == []


def test_findings_exit_1(tmp_path, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path, vera_text="A=1\nA=2\nDB_ADMIN_PASSWORD=x\nPGHOST=y\nVERA_ENV_CONTEXT=z\n"
                                                 "VERA_DB_URL_RO=postgresql://u:p@a.test,b.test/db\n")
    (r / "config" / "env" / ".env-vera").chmod(0o644)
    (repo / "vera" / ".env").write_text("X=1\n")
    (repo / "vera" / "legacy.py").write_text("P = 'AI-Internship/week-1v2'\n")
    (repo / "tests" / "vera_test_boundary.py").write_text("CRED_FILE_GLOBS = ('.env',)\n")
    rc, text = _run(mod, r, repo)
    assert rc == 1
    for frag in ("mode is not 0600", "duplicate names", "admin-floor names", "control names", "C3-denied",
                 "R-G1", "legacy path literals", "C10", "VERA_DB_URL_RO (.env-vera): parse multihost"):
        assert frag in text, frag
    assert no_net == []


def test_missing_env_vera_is_a_finding(tmp_path, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path)
    (r / "config" / "env" / ".env-vera").unlink()
    rc, text = _run(mod, r, repo)
    assert rc == 1 and ".env-vera: missing" in text


def test_marker_file_is_never_read(tmp_path, monkeypatch, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path)
    marker = r / "config" / "env" / ".test-db-marker"
    marker.write_text("NONCE-SENTINEL")
    marker.chmod(0o600)
    opened = []
    real_open, real_io_open = builtins.open, io.open

    def spy(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_io(f, *a, **k):
        opened.append(str(f))
        return real_io_open(f, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy_io)
    real_read_text = pathlib.Path.read_text

    def spy_read_text(self, *a, **k):
        opened.append(str(self))
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", spy_read_text)
    rc, text = _run(mod, r, repo)
    assert not any(o.endswith(".test-db-marker") for o in opened)
    assert "NONCE-SENTINEL" not in text and "contents=not read" in text


def test_verify_disposable_refuses_exit_3(tmp_path, no_net):
    mod = _load()
    r, repo, s = make_layout(tmp_path)
    rc, text = _run(mod, r, repo, "--verify-disposable",
                    environ={"VERA_TEST_DB_URL": "postgresql://t:x@localhost:5433/vera_test_x"})
    assert rc == 3 and ec.VERIFY_DISABLED in text
    assert no_net == []


def test_usage_error_exit_2(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit) as ei:
        mod.main(["--bogus"], environ={}, out=io.StringIO())
    assert ei.value.code == 2
