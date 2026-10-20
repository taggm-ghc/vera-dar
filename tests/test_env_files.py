"""PLAN-ENV s5: vera/env_files.py. Offline; every layout is a fake tree in tmp_path with FAKE values. The real
R/config/env is never opened (the module constants are monkeypatched to the fake tree)."""
from __future__ import annotations

import builtins
import importlib.util
import logging
import os
import stat
import subprocess
from pathlib import Path

import pytest

from vera import env_files as ef
from vera import env_precedence as ep

VERA_DIR = Path(__file__).resolve().parent.parent
FAKE_URL = "postgresql://fakeuser:fakepw-ZZ9@db.invalid/fakedb"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A fake R/config/env with a 0700 dir; returns a writer for .env-vera."""
    old = os.umask(0o077)
    env_dir = tmp_path / "R" / "config" / "env"
    env_dir.mkdir(parents=True)
    os.chmod(env_dir, 0o700)
    monkeypatch.setattr(ef, "ENV_DIR", env_dir)
    monkeypatch.setattr(ef, "VERA_ENV", env_dir / ".env-vera")
    monkeypatch.setattr(ef, "INTERN_ENV", env_dir / ".env-intern")

    def write(text, mode=0o600):
        p = env_dir / ".env-vera"
        p.write_text(text, encoding="utf-8")
        os.chmod(p, mode)
        return p

    write.dir = env_dir
    yield write
    os.umask(old)


def _load(environ, **kw):
    return ef.load_vera_env(environ, log=kw.pop("log", False), modules={}, **kw)


# --- paths come from __file__ only -------------------------------------------------------------------------------
def test_paths_derive_from_file_only(tmp_path, monkeypatch):
    here = Path(ef.__file__).resolve()
    assert ef.REPO == here.parents[1] and ef.R_ROOT == here.parents[2]
    assert ef.ENV_DIR == ef.R_ROOT / "config" / "env"
    assert ef.VERA_ENV == ef.ENV_DIR / ".env-vera" and ef.INTERN_ENV == ef.ENV_DIR / ".env-intern"
    # A fresh copy of the module imported with another cwd and a hostile HOME / env override yields the same paths.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VERA_ENV_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("vera_env_files_copy", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.REPO, mod.R_ROOT, mod.ENV_DIR, mod.VERA_ENV, mod.INTERN_ENV) == (
        here.parents[1], here.parents[2], here.parents[2] / "config" / "env",
        here.parents[2] / "config" / "env" / ".env-vera", here.parents[2] / "config" / "env" / ".env-intern")


# --- refusal codes E1-E6 ------------------------------------------------------------------------------------------
def _code(fn, *a, **k):
    with pytest.raises(ef.EnvFilesError) as ei:
        fn(*a, **k)
    assert ei.value.__suppress_context__ is True or ei.value.__cause__ is None
    return ei.value.code


def test_e1_dir_missing(layout, monkeypatch, tmp_path):
    monkeypatch.setattr(ef, "ENV_DIR", tmp_path / "nowhere")
    assert _code(ef.preflight_dir) == "E1"


def test_e1_dir_symlink(layout, monkeypatch, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(layout.dir)
    monkeypatch.setattr(ef, "ENV_DIR", link)
    assert _code(ef.preflight_dir) == "E1"


def test_e1_dir_group_or_other_writable(layout):
    os.chmod(layout.dir, 0o770)
    assert _code(ef.preflight_dir) == "E1"
    os.chmod(layout.dir, 0o702)
    assert _code(ef.preflight_dir) == "E1"


def test_e1_dir_wrong_owner(layout, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(layout.dir).st_uid + 1)
    assert _code(ef.preflight_dir) == "E1"


def test_e2_not_a_vera_checkout(layout, monkeypatch, tmp_path):
    monkeypatch.setattr(ef, "REPO", tmp_path / "not-vera")
    assert _code(ef.preflight_dir) == "E2"


def test_e3_file_missing(layout):
    assert _code(ef.preflight_file, ef.VERA_ENV) == "E3"
    assert _code(_load, {"VERA_ENV_CONTEXT": "local_dev"}) == "E3"


def test_e4_symlink_or_not_regular(layout):
    real = layout.dir / "real"
    real.write_text("A=1\n")
    os.chmod(real, 0o600)
    (layout.dir / ".env-vera").symlink_to(real)
    assert _code(ef.preflight_file, ef.VERA_ENV) == "E4"
    (layout.dir / ".env-vera").unlink()
    (layout.dir / ".env-vera").mkdir()
    assert _code(ef.preflight_file, ef.VERA_ENV) == "E4"


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644, 0o660, 0o700, 0o602])
def test_e5_mode_must_be_exactly_0600(layout, mode):
    layout("OTHER_PLAIN=x\n", mode=mode)
    assert _code(ef.preflight_file, ef.VERA_ENV) == "E5"


@pytest.mark.parametrize("name", ["DB_ADMIN_URL", "APP_DBADMIN_URL", "VERA_DB_URL_ADMIN", "VERA_MIGRATE_PRODUCTION"])
def test_e6_admin_floor_names(layout, name):
    layout(f"{name}={FAKE_URL}\nOTHER_PLAIN=x\n")
    with pytest.raises(ef.EnvFilesError) as ei:
        _load({"VERA_ENV_CONTEXT": "local_dev"})
    assert ei.value.code == "E6" and name in str(ei.value)
    assert "fakepw" not in str(ei.value) and "fakeuser" not in str(ei.value)


def test_messages_carry_no_values(layout):
    layout(f"VERA_DB_URL_ADMIN={FAKE_URL}\n", mode=0o644)
    with pytest.raises(ef.EnvFilesError) as ei:
        _load({})
    assert "fakepw" not in str(ei.value) and "db.invalid" not in str(ei.value)


# --- contexts that load no file never touch ENV_DIR ---------------------------------------------------------------
@pytest.mark.parametrize("environ", [
    {"RENDER": "true"},
    {"VERA_TEST_RUNNER": "1"},
    {"CLAUDECODE": "1"},
    {"VERA_ENV_CONTEXT": "production"},
])
def test_no_file_contexts_never_stat_env_dir(layout, monkeypatch, tmp_path, environ):
    gone = tmp_path / "container" / "config" / "env"       # container layout: no R/config at all
    monkeypatch.setattr(ef, "ENV_DIR", gone)
    monkeypatch.setattr(ef, "VERA_ENV", gone / ".env-vera")
    touched = []
    real_lstat, real_stat, real_open = os.lstat, os.stat, os.open

    def spy(real):
        def inner(path, *a, **k):
            if "config" in str(path) and "env" in str(path) and str(tmp_path) in str(path):
                touched.append(str(path))
            return real(path, *a, **k)
        return inner

    monkeypatch.setattr(os, "lstat", spy(real_lstat))
    monkeypatch.setattr(os, "stat", spy(real_stat))
    monkeypatch.setattr(os, "open", spy(real_open))
    audit = _load(dict(environ))
    assert audit.files == [] and not touched


def test_render_container_layout_no_config_dir_boots(tmp_path, monkeypatch):
    """C11: a container tree with no R/config and RENDER=true loads without a stat of the config dir."""
    root = tmp_path / "app"
    (root / "vera").mkdir(parents=True)
    monkeypatch.setattr(ef, "ENV_DIR", root.parent / "config" / "env")
    seen = []
    real = os.lstat
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: (seen.append(str(p)), real(p, *a, **k))[1])
    env = {"RENDER": "true", "INTERNAL_DB_URL": FAKE_URL}
    audit = _load(env)
    assert audit.context == "production" and env["INTERNAL_DB_URL"] == FAKE_URL
    assert not [s for s in seen if str(root.parent / "config") in s]


# --- planted legacy files are never opened -----------------------------------------------------------------------
def test_legacy_files_never_opened(layout, tmp_path, monkeypatch):
    planted = [tmp_path / "R" / "vera" / ".env", tmp_path / "AI-Internship" / "week-1v2" / ".env",
               tmp_path / "AI-Internship" / "week-1v2" / ".env.db-accounts", tmp_path / ".env"]
    for p in planted:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("PLANTED_ONLY=1\n")
        os.chmod(p, 0o600)
    layout("OTHER_PLAIN=from-file\n")
    opened = []
    real_os_open, real_open = os.open, builtins.open
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (opened.append(str(p)), real_os_open(p, *a, **k))[1])
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1])
    monkeypatch.chdir(tmp_path / "R" / "vera")
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {"VERA_ENV_CONTEXT": "local_dev"}
    _load(env)
    assert env["OTHER_PLAIN"] == "from-file" and "PLANTED_ONLY" not in env
    assert not [o for o in opened if any(str(p) in o for p in planted)]
    assert any(o.endswith(".env-vera") for o in opened)


# --- C3 precedence-change table, row by row ----------------------------------------------------------------------
FILE = f"OTHER_PLAIN=from-file\nVERA_DB_URL_RW={FAKE_URL}\n"


def test_row_production_unchanged(layout):
    env = {"RENDER": "true", "VERA_DB_URL_RW": "inh", "PGHOST": "h", "OTHER_PLAIN": "inh"}
    before = dict(env)
    audit = _load(env)
    assert env == before and audit.context == "production" and audit.stripped == []


def test_row_agent_unchanged_d036(layout):
    env = {"CLAUDECODE": "1", "VERA_OWNER_KEY": "k", "EXTERNAL_DB_URL": "e", "VERA_DB_URL_RO": "ro",
           "VERA_DB_URL_WO": "wo", "OTHER_PLAIN": "inh"}
    audit = _load(env)
    assert audit.context == "agent" and audit.files == []
    assert sorted(audit.stripped) == ["EXTERNAL_DB_URL", "VERA_OWNER_KEY"]
    assert env["VERA_DB_URL_RO"] == "ro" and env["OTHER_PLAIN"] == "inh" and "VERA_OWNER_KEY" not in env


def test_row_local_dev_explicit_file_wins_and_warns_by_name(layout, caplog):
    layout(FILE)
    env = {"VERA_ENV_CONTEXT": "local_dev", "OTHER_PLAIN": "inherited-VAL", "VERA_DB_URL_RW": "inh-url",
           "PGHOST": "inh-host", "KRB5CCNAME": "k", "HTTPS_PROXY": "p", "SSL_CERT_FILE": "c"}
    with caplog.at_level(logging.INFO, logger="vera.env_precedence"):
        audit = _load(env, log=True)
    assert audit.selected_by == "explicit"
    assert env["OTHER_PLAIN"] == "from-file" and env["VERA_DB_URL_RW"] == FAKE_URL
    for gone in ("PGHOST", "KRB5CCNAME", "HTTPS_PROXY", "SSL_CERT_FILE"):
        assert gone not in env
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "env: file .env-vera overrides inherited OTHER_PLAIN" in warns
    text = " ".join(r.getMessage() for r in caplog.records)
    for secret in ("inherited-VAL", "from-file", "fakepw", "inh-url", "inh-host"):
        assert secret not in text


def test_row_local_dev_default_no_db_url_and_strips(layout):
    layout(FILE)
    env = {"OTHER_PLAIN": "inh", "VERA_DB_URL_RW": "inh-url", "DATABASE_URL": "d", "PGPASSWORD": "p",
           "DB_ADMIN_PW": "a", "VERA_MIGRATE_PRODUCTION": "1"}
    audit = _load(env)
    assert audit.selected_by == "default" and audit.context == "local_dev"
    assert env["OTHER_PLAIN"] == "from-file"
    for gone in ("VERA_DB_URL_RW", "DATABASE_URL", "PGPASSWORD", "DB_ADMIN_PW", "VERA_MIGRATE_PRODUCTION"):
        assert gone not in env
    assert "VERA_DB_URL_RW" in audit.skipped


def test_row_agent_explicit_file_wins_owner_and_db_urls_skipped(layout):
    layout("OTHER_PLAIN=from-file\nVERA_DB_URL_RW=" + FAKE_URL + "\n")
    env = {"CLAUDECODE": "1", "VERA_ENV_CONTEXT": "local_dev", "OTHER_PLAIN": "inh", "VERA_OWNER_KEY": "k",
           "EXTERNAL_DB_URL": "e"}
    audit = _load(env)
    assert audit.selected_by == "agent_explicit"
    assert env["OTHER_PLAIN"] == "from-file"
    assert "VERA_DB_URL_RW" not in env and "VERA_OWNER_KEY" not in env and "EXTERNAL_DB_URL" not in env


def test_row_test_context_loads_no_file(layout):
    env = {"VERA_TEST_RUNNER": "1", "OTHER_PLAIN": "inh"}
    audit = _load(env)
    assert audit.context == "test" and audit.files == [] and env["OTHER_PLAIN"] == "inh"


@pytest.mark.parametrize("line", ["SSL_CERT_FILE=/x", "REQUESTS_CA_BUNDLE=/x", "CURL_CA_BUNDLE=/x", "HTTP_PROXY=h",
                                  "https_proxy=h", "ALL_PROXY=h", "NO_PROXY=h", "PGHOST=h", "KRB5_CONFIG=k",
                                  "GSS_NAME=1"])
def test_tls_proxy_libpq_names_refused_in_file(layout, line):
    layout(line + "\nOTHER_PLAIN=x\n")
    with pytest.raises(ep.EnvPrecedenceError):
        _load({"VERA_ENV_CONTEXT": "local_dev"})


# --- start.sh (C11) -----------------------------------------------------------------------------------------------
# The test boundary refuses any child process other than the pinned interpreter, so start.sh is checked statically here
# (its refusal and allow paths were exercised by hand outside the runner; see the owner's private implementation notes, not published).
def test_start_sh_gate_precedes_every_launch_and_is_names_only():
    lines = (VERA_DIR / "start.sh").read_text(encoding="utf-8").splitlines()
    code = [(i, l) for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]
    gate = next(i for i, l in code if "RENDER" in l and "VERA_ENV_CONTEXT" in l)
    launches = [i for i, l in code if "exec " in l]
    assert launches and gate < min(launches)
    block = "\n".join(lines[gate:gate + 4])
    assert '"${RENDER:-}" != "true"' in block and '"${VERA_ENV_CONTEXT:-}" != "production"' in block
    assert "refusing" in block and "exit 3" in block
    echo = next(l for l in lines[gate:gate + 4] if "echo" in l)
    assert "$" not in echo          # the message interpolates no variable, so it cannot print a value
