"""PLAN-ENV ENV-C: db/apply_migrations.py env wiring and disabled paths (offline; fakes only, no DB, no real env file).

Owner decisions 2026-10-19: D-ENV-5 = DEFER (production path disabled, code D-ENV-5-pending); disposable verification
refuses unconditionally (DISPOSABLE-VERIFY-DISABLED-pending-isolation-review). Every fixture file lives under tmp_path.
"""
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_apply_migrations_env_under_test", ROOT / "db" / "apply_migrations.py")
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)
g = am.g
VD = "DISPOSABLE-VERIFY-DISABLED-pending-isolation-review"


class Boom(AssertionError):
    pass


def make_ctx(tmp_path, *, with_sources=True, environ=None, local=True):
    mig = tmp_path / "migrations"
    mig.mkdir(exist_ok=True)
    (mig / "005_vera_x.sql").write_text("SELECT 1;\n")
    prod = tmp_path / "env-vera"
    prod.write_text("DB_HOST=db.example.invalid\nDB_NAME=vera_prod\nDB_ADMIN_ROLE=vera_admin_x\n" if with_sources else "")
    loc = tmp_path / "env-local"
    loc.write_text("LOCAL_PG_HOST=127.0.0.1\nLOCAL_PG_DB=vera_local_t\nLOCAL_PG_ROLE=vera_local_r\n" if local else "")
    calls = {"connect": 0, "guard": []}

    def connect(*a, **k):
        calls["connect"] += 1
        raise Boom("connect must never be called")

    out, err = io.StringIO(), io.StringIO()
    ctx = am.Ctx(migrations_dir=mig, prod_env_files=[prod], local_env_file=loc, marker_file=tmp_path / "no-marker",
                 environ={} if environ is None else environ, connect=connect, isatty=lambda: False,
                 prompt=lambda t: Boom, getpass=lambda t: "", out=out, err=err,
                 install_guard=lambda c: calls["guard"].append(c))
    return ctx, calls, out, err


def test_week1_is_gone_and_defaults_are_under_config_env():
    assert not hasattr(am, "WEEK1")
    ef = am._ef
    assert am.default_prod_env_files() == [ef.INTERN_ENV, ef.VERA_ENV]
    for p in am.default_prod_env_files() + [am.LOCAL_ENV_FILE, am.MARKER_FILE]:
        assert p.parent == ef.ENV_DIR
    assert am.LOCAL_ENV_FILE.name == ".env-local-pg" and am.MARKER_FILE.name == ".test-db-marker"
    src = (ROOT / "db" / "apply_migrations.py").read_text()
    assert "WEEK1" not in src and "AI-Internship" not in src and ".env.db-accounts" not in src


def test_ctx_defaults_use_the_new_paths():
    c = am.Ctx(environ={})
    assert c.local_env_file == am.LOCAL_ENV_FILE and c.marker_file == am.MARKER_FILE


def test_plan_stays_offline_and_works(tmp_path):
    ctx, calls, out, _ = make_ctx(tmp_path, with_sources=False)
    assert am.main(["plan", "005"], ctx) == 0
    assert "connection made" in out.getvalue()
    assert calls["connect"] == 0 and calls["guard"] == []


def test_rollback_inspect_stays_offline(tmp_path):
    ctx, calls, _, err = make_ctx(tmp_path)
    am.main(["rollback", "005"], ctx)   # no rollback file: a refusal, never a connect
    assert calls["connect"] == 0 and calls["guard"] == []


@pytest.mark.parametrize("argv", [["inspect", "--target", "local"], ["rehearse", "--target", "local", "005"]])
def test_local_loopback_target_is_refused_as_not_verifiable(tmp_path, argv):
    ctx, calls, _, err = make_ctx(tmp_path)
    rc = am.main(argv, ctx)
    assert rc == g.EXIT_IDENTITY
    assert VD in err.getvalue()
    assert calls["connect"] == 0
    assert calls["guard"] == ["migration"]   # the migration connect guard is installed before the refusal


def test_local_marker_file_is_never_read(tmp_path, monkeypatch):
    ctx, calls, _, err = make_ctx(tmp_path)
    ctx.marker_file.write_text("planted-token")
    real = Path.read_text
    seen = []

    def spy(self, *a, **k):
        seen.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    am.main(["inspect", "--target", "local"], ctx)
    assert ctx.marker_file.name not in seen
    assert VD in err.getvalue()


@pytest.mark.parametrize("argv", [["inspect", "--target", "local"], ["rehearse", "--target", "local", "005"]])
def test_no_sources_refuses_every_connecting_op(tmp_path, argv):
    ctx, calls, _, err = make_ctx(tmp_path, with_sources=False)
    rc = am.main(argv, ctx)
    assert rc == g.EXIT_REFUSED
    assert am.NO_SOURCES in err.getvalue()
    assert calls["connect"] == 0 and calls["guard"] == []


def test_guard_unavailable_still_refuses_with_the_same_code(tmp_path):
    ctx, calls, _, err = make_ctx(tmp_path)

    def bad(_c):
        raise RuntimeError("secret-looking-detail")

    ctx.install_guard = bad
    assert am.main(["inspect", "--target", "local"], ctx) == g.EXIT_IDENTITY
    assert VD in err.getvalue() and "secret-looking-detail" not in err.getvalue()
    assert calls["connect"] == 0


@pytest.mark.parametrize("argv", [
    ["apply", "--target", "production", "--owner-approval", "OD-1", "--approve-sha", "005=" + "0" * 64, "005"],
    ["inspect", "--target", "production"],
])
def test_production_path_ships_disabled(tmp_path, argv):
    assert am.PRODUCTION_PATH_ENABLED is False
    ctx, calls, _, err = make_ctx(tmp_path, environ={"VERA_MIGRATE_PRODUCTION": "I_ACCEPT_OD-1"})
    rc = am.main(argv, ctx)
    assert rc == g.EXIT_REFUSED and "D-ENV-5-pending" in err.getvalue()
    assert calls["connect"] == 0


def test_no_environment_or_argument_enables_the_production_path(tmp_path):
    env = {"PRODUCTION_PATH_ENABLED": "1", "VERA_MIGRATE_PRODUCTION": "I_ACCEPT_OD-1", "D_ENV_5": "1"}
    ctx, calls, _, err = make_ctx(tmp_path, environ=env)
    assert am.main(["inspect", "--target", "production"], ctx) == g.EXIT_REFUSED
    assert "D-ENV-5-pending" in err.getvalue()
    with pytest.raises(g.Refusal):
        am.parse(["apply", "--target", "production", "--enable-production", "005"])


@pytest.mark.parametrize("marker", ["CLAUDECODE", "AI_AGENT"])
def test_enabled_production_path_still_refuses_under_an_agent(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(am, "PRODUCTION_PATH_ENABLED", True)   # the reviewed one-line change, simulated
    ctx, calls, _, err = make_ctx(tmp_path, environ={marker: "1"})
    rc = am.main(["apply", "--target", "production", "--owner-approval", "OD-1", "--approve-sha", "005=" + "0" * 64,
                  "005"], ctx)
    assert rc == g.EXIT_REFUSED and marker in err.getvalue() and "agent" in err.getvalue()
    assert calls["connect"] == 0


def test_enabled_production_path_refuses_ambient_pg_names(tmp_path, monkeypatch):
    monkeypatch.setattr(am, "PRODUCTION_PATH_ENABLED", True)
    ctx, calls, _, err = make_ctx(tmp_path, environ={"PGHOST": "x"})
    rc = am.main(["inspect", "--target", "production"], ctx)
    assert rc == g.EXIT_REFUSED and "PGHOST" in err.getvalue()
    assert calls["connect"] == 0


def test_strip_inherited_removes_names_only_and_reports_names():
    env = {"EXTERNAL_DB_URL": "u", "PGHOST": "h", "KRB5CCNAME": "k", "HOME": "/h", "VERA_DB_URL_RW": "v"}
    names = g.strip_inherited(env, am.STRIP_PATTERNS)
    assert names == ["EXTERNAL_DB_URL", "KRB5CCNAME", "PGHOST", "VERA_DB_URL_RW"] and env == {"HOME": "/h"}


def test_strip_runs_after_the_credential_refusals(tmp_path):
    # a DB URL carrying a password is still REFUSED by name (checked before the strip), not silently stripped.
    ctx, calls, _, err = make_ctx(tmp_path, environ={"DATABASE_URL": "postgresql://u:pw@h/d"})
    rc = am.main(["inspect", "--target", "local"], ctx)
    assert rc == g.EXIT_REFUSED and "DATABASE_URL" in err.getvalue() and "pw" not in err.getvalue()
    assert calls["guard"] == []


def test_classify_local_is_a_routing_hint_only():
    ep = g.endpoint("127.0.0.1", "5432", "vera_local_t", "r")
    assert g.classify(ep, set()) == "local"
    assert "ROUTING HINT" in (g.classify.__doc__ or "")
