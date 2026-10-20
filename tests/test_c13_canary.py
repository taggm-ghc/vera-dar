"""C13 / OR-4 negative canary (PLAN-ENV v2.1 §5.1), with the binding owner decision of 2026-10-19 applied: disposable
verification refuses unconditionally, so case (a) is "refuses, driver never called" and the HMAC helper is exercised
directly as inert code.

Everything is fake: random values, .invalid/.test hosts, a fake root under tmp_path. No DB, no network, no real env
file. The leak scanner is itself self-tested against a deliberately leaky sentinel.
"""
import importlib
import importlib.util
import io
import logging
import os
import secrets
import shutil
import socket
import sys
import traceback
from pathlib import Path
from urllib.parse import quote

import pytest

from safety import connect_guard as cg
from safety import endpoint_class as ec
from safety import sql_guard as sg

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------- leak scanner
def _forms(s: str) -> set:
    return {s, quote(s, safe=""), quote(s), s.upper(), s.lower()}


def scan_leaks(secret_values, channels: dict) -> list:
    """Names of channels that contain any secret (raw, URL-encoded, or case-folded)."""
    bad = []
    forms = set()
    for s in secret_values:
        if s:
            forms |= _forms(s)
    for name, text in channels.items():
        t = text if isinstance(text, str) else repr(text)
        if any(f in t for f in forms):
            bad.append(name)
    return bad


def _exc_channels(prefix: str, excs: list) -> dict:
    out = {}
    for i, e in enumerate(excs):
        out[f"{prefix}{i}.str"] = str(e)
        out[f"{prefix}{i}.repr"] = repr(e)
        out[f"{prefix}{i}.tb"] = "".join(traceback.format_exception(type(e), e, e.__traceback__, chain=True))
        ctx = e.__context__ or e.__cause__
        while ctx is not None:
            out[f"{prefix}{i}.ctx"] = out.get(f"{prefix}{i}.ctx", "") + repr(ctx) + str(ctx)
            ctx = ctx.__context__ or ctx.__cause__
    return out


def test_leak_scanner_self_test(caplog):
    pw = "cnry-" + secrets.token_hex(8)

    def leaky_sentinel(url):
        logging.getLogger("leaky").warning("connecting to %s", url)
        raise RuntimeError(f"failed for {url}")

    with caplog.at_level(logging.DEBUG):
        try:
            leaky_sentinel(f"postgresql://u:{quote(pw, safe='')}@h.invalid/db")
        except RuntimeError as e:
            exc = e
    hits = scan_leaks([pw], {"caplog": caplog.text, **_exc_channels("e", [exc])})
    assert "caplog" in hits and "e0.str" in hits, "the scanner must catch a deliberate leak"
    assert scan_leaks([pw], {"clean": "connect refused: bad_param"}) == []


# ---------------------------------------------------------------------------------------------- fixture
@pytest.fixture
def canary(tmp_path, monkeypatch):
    h = secrets.token_hex(4)
    v = {
        "pw": "cnry-" + secrets.token_hex(8),
        "host": f"canary-prod-{h}.invalid",
        "host2": f"canary-db-{h}.test",
        "db": f"canaryprod_{h}",
        "role": f"canary_rw_{h}",
        "admin": f"canary_admin_{h}",
        "api": "sk-cnry" + secrets.token_hex(12),
    }
    r = tmp_path / "Rfake"
    stub = r / "vera-stub"
    env = r / "config" / "env"
    for d in (stub / "vera", stub / "scripts", stub / "tests", stub / "db" / "accounts", stub / ".git", env):
        d.mkdir(parents=True, exist_ok=True)
    env.chmod(0o700)
    globs = '(".env", ".env.*")'
    (stub / "scripts" / "vera_test_runner.py").write_text(f"CRED_MASK_GLOBS = {globs}\n")
    (stub / "tests" / "vera_test_boundary.py").write_text(f"CRED_FILE_GLOBS = {globs}\n")
    (stub / "main.py").write_text("")
    (stub / "vera" / "env_precedence.py").write_text("")
    url = f"postgresql://{v['role']}:{quote(v['pw'], safe='')}@{v['host']}:5432/{v['db']}"
    url2 = f"postgresql://{v['role']}:{quote(v['pw'], safe='')}@{v['host2']}:5432/{v['db']}"
    lines = [f"{n}={url if i % 2 else url2}" for i, n in enumerate(
        ("EXTERNAL_DB_URL", "INTERNAL_DB_URL", "DATABASE_URL", "VERA_DB_URL_RO", "VERA_DB_URL_RW", "VERA_DB_URL_WO"))]
    (env / ".env-vera").write_text("\n".join(lines + [f"OPENAI_API_KEY={v['api']}"]) + "\n")
    (env / ".env-intern").write_text("\n".join(lines + [
        f"VERA_DB_URL_ADMIN=postgresql://{v['admin']}:{v['pw']}@{v['host']}/{v['db']}",
        f"DB_HOST={v['host']}", "DB_PORT=5432", f"DB_NAME={v['db']}", f"DB_ACCOUNT={v['role']}",
        f"DB_ADMIN_ROLE={v['admin']}"]) + "\n")
    for f in (".env-vera", ".env-intern"):
        (env / f).chmod(0o600)
    # the agent shell: conflicting inherited URLs
    monkeypatch.setenv("VERA_DB_URL_RO", f"postgresql://{v['role']}:{v['pw']}x@{v['host2']}/{v['db']}")
    monkeypatch.setenv("VERA_DB_URL_WO", f"postgresql://{v['role']}:{v['pw']}y@{v['host']}/{v['db']}")
    for n in list(os.environ):
        if n.upper().startswith(ec.AMBIENT_PREFIXES):
            monkeypatch.delenv(n)
    net = []

    def no_net(*a, **k):
        net.append("net")
        raise OSError("canary: no network")

    monkeypatch.setattr(socket, "getaddrinfo", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    import psycopg2
    raw_calls = []
    monkeypatch.setattr(psycopg2, "_connect", lambda *a, **k: raw_calls.append(1))
    sources = ec.Sources.from_files(env / ".env-vera", env / ".env-intern", dict(os.environ))
    secret_list = [v["pw"], v["host"], v["host2"], v["db"], v["role"], v["admin"], v["api"]]
    secret_list += [ec._h(x)[:8] for x in (v["host"], v["host2"], v["db"], v["role"])]
    return dict(v=v, r=r, stub=stub, env=env, url=url, url2=url2, sources=sources, net=net, raw=raw_calls,
                secrets=secret_list, tmp=tmp_path)


def _assert_absent(c, channels: dict):
    hits = scan_leaks(c["secrets"], channels)
    assert hits == [], f"canary leaked into channels: {hits}"


def _tmp_files_channel(c) -> dict:
    planted = {c["env"] / ".env-vera", c["env"] / ".env-intern"}
    out = {}
    for p in c["tmp"].rglob("*"):
        if p.is_file() and p not in planted:
            out[f"file:{p.relative_to(c['tmp'])}"] = p.read_text(errors="replace")
    return out


# ---------------------------------------------------------------------------------------------- paths
def test_canary_classify_every_url_is_production(canary, caplog, capsys):
    c, v = canary, canary["v"]
    urls = [c["url"], c["url2"],
            f"postgresql://{v['role']}:{v['pw']}@localhost:5433/{v['db']}",
            f"postgresql://x:y@{v['host2']}/{v['db']}?hostaddr=10.1.2.3",
            f"postgresql://x:y@{v['host2']}/postgres?dbname={v['db']}",
            f"postgresql://x:y@{v['host']}./{v['db']}",
            f"postgresql://x:y@{quote(v['host'], safe='')}%2E/{v['db']}",
            f"postgresql://x:y@h.test/{v['db']}%5F",
            f"postgresql://{v['role']}%40:y@h.test/db",
            f"postgresql://x:y@/{v['db']}?host=/var/run/postgresql",
            f"postgresql://x:y@/{v['db']}"]
    verdicts = []
    with caplog.at_level(logging.DEBUG):
        for u in urls:
            vd = ec.classify(u, sources=c["sources"], environ={"PGHOST": "/tmp"} if u.endswith(f"@/{v['db']}") else {})
            assert vd.cls == ec.PRODUCTION and vd.cls != ec.DISPOSABLE_VERIFIED
            assert vd.reasons and set(vd.reasons) <= ec.REASONS, vd.reasons
            verdicts.append(vd)
        for vd in verdicts[:2]:   # the planted fixture URLs must hit real indicators, not just 'inconclusive'
            assert set(vd.reasons) & set(ec.INDICATOR_REASONS), vd.reasons
        tunnel = ec.classify(urls[2], sources=c["sources"], environ={})
        assert {"dbname", "role"} <= set(tunnel.reasons)
    out = capsys.readouterr()
    ch = {"caplog": caplog.text, "out": out.out, "err": out.err, "sources": repr(c["sources"])}
    for i, vd in enumerate(verdicts):
        ch[f"v{i}"] = repr(vd) + str(vd.as_dict())
    _assert_absent(c, ch)


def test_canary_connect_guard_refuses_before_driver(canary, caplog, capsys):
    c = canary
    import types
    calls = []
    mod = types.SimpleNamespace(connect=lambda *a, **k: calls.append(k))
    excs = []
    with caplog.at_level(logging.DEBUG):
        for ctx in ("test", "check", "calibration", "migration"):
            g = cg.Guard()
            g.install(ctx, sqlalchemy=False, drivers=[(mod, "connect")])
            for u in (c["url"], c["url2"], c["url"] + "?hostaddr=10.0.0.1"):
                with pytest.raises(cg.ConnectRefused) as ei:
                    mod.connect(u)
                excs.append(ei.value)
            g.uninstall()
        g = cg.Guard()
        g.install("production", sqlalchemy=False, drivers=[(mod, "connect")])
        for u, kw in ((c["url"] + "?options=x", {}), (c["url"], {"host": c["v"]["host2"]}),
                      ("postgresql://u:p@a.test,b.test/db", {})):
            with pytest.raises(cg.ConnectRefused) as ei:
                mod.connect(u, **kw)
            excs.append(ei.value)
        # SQLAlchemy hook path with an engine recorded by the guard's create_engine wrapper
        g2 = cg.Guard()
        g2.install("test", sqlalchemy=True, drivers=[])
        try:
            import sqlalchemy
            eng = sqlalchemy.create_engine(c["url"].replace("postgresql://", "postgresql+psycopg2://"))
            with pytest.raises(cg.ConnectRefused) as ei:
                g2.on_do_connect(eng.dialect, type("R", (), {"info": {}})(), (), {})
            excs.append(ei.value)
        finally:
            g2.uninstall()
    assert calls == [] and c["raw"] == [] and c["net"] == []
    assert {e.code for e in excs} >= {ec.VERIFY_DISABLED, "bad_param", "multihost"}
    out = capsys.readouterr()
    _assert_absent(c, {"caplog": caplog.text, "out": out.out, "err": out.err, **_exc_channels("e", excs),
                       **_tmp_files_channel(c)})


@pytest.mark.boundary_probe
def test_canary_real_sqlalchemy_path(canary, caplog):
    """The process-global path in the test process (boundary outermost, guard installed in 'test'): both refuse;
    psycopg2's C entry point is never called and no exception text carries a canary value. The raw psycopg2 path is
    exercised with a fake driver module (tests/test_env_guard.py bans literal connector calls in tests)."""
    c = canary
    from sqlalchemy import create_engine
    excs = []
    with caplog.at_level(logging.DEBUG):
        eng = create_engine(c["url"].replace("postgresql://", "postgresql+psycopg2://"))
        with pytest.raises(Exception) as ei:
            eng.connect()
        excs.append(ei.value)
    assert all(type(e).__name__ in ("ConnectRefused", "DBBoundaryError") for e in excs)
    assert c["raw"] == []
    _assert_absent(c, {"caplog": caplog.text, **_exc_channels("e", excs)})


class _Cur:
    def __init__(self, behaviour, c, tag, log):
        self.b, self.c, self.tag, self.log, self._n = behaviour, c, tag, log, None

    def execute(self, q, p=None):
        self.log.append(q)
        v = self.c["v"]
        if self.b == "timeout" and "disposable_marker" in q:
            raise TimeoutError("statement timeout")
        if self.b == "exception":
            raise RuntimeError(f"could not connect to {v['host']} as {v['role']} password {v['pw']}")
        if q.startswith("SHOW"):
            self._n = ("off" if self.b == "read_only_off" else "on",)
        elif "current_database(), current_user, " in q:
            db = v["db"] if self.b == "pd" else "vera_test_x"
            oid = 999 if self.b == "dns" else 41
            self._n = (db, "tester", oid, 5433)
        elif "disposable_marker" in q:
            if self.b == "missing":
                raise RuntimeError("relation does not exist")
            self._n = ("00" * 32,) if self.b == "wrong" else (self.tag,)

    def fetchone(self):
        return self._n


class _Conn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur


@pytest.mark.parametrize("case", ["a_correct", "wrong", "missing", "timeout", "exception", "read_only_off", "pd",
                                  "dns"])
def test_canary_verify_disposable_cases(canary, case, caplog, capsys):
    c = canary
    nonce = secrets.token_bytes(32)
    tag = ec._inert_marker_tag(nonce, "vera_test_x", 41)
    marker = {"nonce": nonce.hex(), "dbname": "vera_test_x", "oid": 41, "port": 5433}
    c["secrets"] += [nonce.hex(), tag]
    log = []
    conn = _Conn(_Cur(case, c, tag, log))
    with caplog.at_level(logging.DEBUG):
        vd = ec.verify_disposable(conn, "postgresql://t:x@localhost:5433/vera_test_x", marker_file=marker,
                                  sources=c["sources"], environ={})
        assert vd.cls == ec.PRODUCTION and ec.VERIFY_DISABLED in vd.reasons
        assert log == []                       # verification never touches the connection (owner decision)
        inert = ec._inert_marker_check(conn, marker, sources=c["sources"])
    expected = {"a_correct": "marker_ok", "wrong": "tag_mismatch", "missing": "probe_error", "timeout": "probe_error",
                "exception": "probe_error", "read_only_off": "not_read_only", "pd": "negative_check",
                "dns": "tag_mismatch"}[case]
    assert inert == expected
    # even the inert helper's success can never reach classification: the verdict above stays PRODUCTION
    out = capsys.readouterr()
    _assert_absent(c, {"caplog": caplog.text, "out": out.out, "err": out.err, "verdict": repr(vd) + str(vd.as_dict()),
                       "inert": inert})
    assert c["raw"] == [] and c["net"] == []


def test_canary_sql_guard_on_production_tagged_connection(canary):
    c = canary
    log = []

    class Raw:
        def cursor(self, *a, **k):
            return type("C", (), {"execute": lambda s, q, p=None: log.append(q),
                                  "executemany": lambda s, q, p: log.append(q)})()

    conn = sg.wrap_connection(Raw(), ec.Verdict(ec.PRODUCTION, ("dbname",)))
    cur = conn.cursor()
    excs = []
    for q in ("DELETE FROM vera_vjay.runs", "/*x*/DELETE FROM t", "WITH d AS (DELETE FROM t RETURNING *) SELECT 1",
              f"SELECT 1; DROP TABLE {c['v']['db']}", f"UPDATE t SET a='{c['v']['pw']}' -- WHERE x",
              "UPDATE t SET a=1 WHERE true", "TRUNCATE t", "DO $$ BEGIN END $$", "COPY t TO STDOUT", "GRANT ALL ON t TO x"):
        with pytest.raises(sg.DestructiveRefused) as ei:
            cur.execute(q)
        excs.append(ei.value)
    assert log == []
    _assert_absent(c, _exc_channels("e", excs))


def test_canary_check_config_json_and_text(canary, capsys):
    """DEVIATION (ENV-B): plan §5.1 runs check_config as a subprocess. tests/test_env_guard.py pins every test spawn
    site, and adding one is ENV-D's edit, so this runs main() in-process on the fake root until ENV-D pins
    'test_c13_canary.py': {'subprocess.run'}; the subprocess variant is then a direct swap."""
    c = canary
    spec = importlib.util.spec_from_file_location("_canary_check_config", ROOT / "scripts" / "check_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    outs = {}
    for extra in (["--json"], []):
        buf = io.StringIO()
        rc = mod.main(["--root", str(c["r"]), "--repo", str(c["stub"]), "--no-git", *extra], environ=dict(os.environ),
                      out=buf)
        assert rc in (0, 1)
        outs[f"out{extra}"] = buf.getvalue()
    cap = capsys.readouterr()
    outs["stdout"], outs["stderr"] = cap.out, cap.err
    assert "PRODUCTION" in outs["out[]"] and "DISPOSABLE_VERIFIED" not in outs["out[]"]
    assert c["net"] == [] and c["raw"] == []
    _assert_absent(c, {**outs, **_tmp_files_channel(c)})


def _load_env_files_copy(c):
    src = ROOT / "vera" / "env_files.py"
    if not src.is_file():
        pytest.skip("ENV-A vera/env_files.py not present")
    dst = c["stub"] / "vera" / "env_files.py"
    shutil.copyfile(src, dst)
    spec = importlib.util.spec_from_file_location("_canary_env_files", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ENV_DIR == c["env"]
    return mod


def _allowed_only_env_text(c) -> str:
    """A fake .env-vera with ONLY names the local_dev context may supply (fake values, .invalid/.test hosts)."""
    v = c["v"]
    url = f"postgresql://{v['role']}:{quote(v['pw'], safe='')}@{v['host']}:5432/{v['db']}"
    names = ("EXTERNAL_DB_URL", "VERA_DB_URL_RO", "VERA_DB_URL_WO", "VERA_DB_URL_RW")
    return "\n".join([f"{n}={url}" for n in names] + [f"OPENAI_API_KEY={v['api']}"]) + "\n"


def test_canary_load_vera_env_success_case(canary, caplog, capsys):
    """(i) SUCCESS: only allowed names. The load succeeds, the success audit and the 'file overrides inherited'
    warning are emitted (by name only), and every channel is leak-scanned."""
    c = canary
    ef = _load_env_files_copy(c)
    vf = c["env"] / ".env-vera"
    vf.write_text(_allowed_only_env_text(c))
    vf.chmod(0o600)
    fake = {"VERA_ENV_CONTEXT": "local_dev", "VERA_DB_URL_RO": os.environ["VERA_DB_URL_RO"],
            "VERA_DB_URL_WO": os.environ["VERA_DB_URL_WO"]}
    with caplog.at_level(logging.DEBUG):
        audit = ef.load_vera_env(fake, log=True, modules={})     # must NOT raise
    assert audit.context == "local_dev" and audit.selected_by == "explicit"
    assert [f["status"] for f in audit.files] == ["loaded"]
    assert {"EXTERNAL_DB_URL", "VERA_DB_URL_RO", "VERA_DB_URL_WO", "VERA_DB_URL_RW", "OPENAI_API_KEY"} <= set(audit.applied)
    assert fake["OPENAI_API_KEY"] == c["v"]["api"] and "INTERNAL_DB_URL" not in fake and "DATABASE_URL" not in fake
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("env precedence: context=local_dev") for m in msgs), "success audit line not emitted"
    over = [m for m in msgs if "overrides inherited" in m]
    assert any(m.endswith("VERA_DB_URL_RO") for m in over) and any(m.endswith("VERA_DB_URL_WO") for m in over)
    out = capsys.readouterr()
    ch = {"caplog": caplog.text, "out": out.out, "err": out.err, "audit": repr(audit) + str(audit.as_dict())}
    _assert_absent(c, ch)


def test_canary_load_vera_env_failure_paths(canary, caplog, capsys):
    """(ii) REFUSAL: INTERNAL_DB_URL / DATABASE_URL (F4) and an admin-floor name (E6) are refused; plus file-mode,
    symlink and missing-file failures. Refusal texts are leak-scanned."""
    c = canary
    ef = _load_env_files_copy(c)
    excs = []
    with caplog.at_level(logging.DEBUG):
        vf = c["env"] / ".env-vera"
        # F4: the default fixture file holds INTERNAL_DB_URL and DATABASE_URL, which local_dev may not supply
        vf.chmod(0o600)
        with pytest.raises(Exception) as ei:
            ef.load_vera_env({"VERA_ENV_CONTEXT": "local_dev"}, log=True, modules={})
        assert getattr(ei.value, "code", None) == "F4"
        assert "INTERNAL_DB_URL" in str(ei.value) and "DATABASE_URL" in str(ei.value)
        excs.append(ei.value)
        # E6: an admin-floor name is refused by the preflight, before anything is applied
        vf.write_text(_allowed_only_env_text(c) + f"DB_ADMIN_PASSWORD={c['v']['pw']}\n")
        vf.chmod(0o600)
        applied = {"VERA_ENV_CONTEXT": "local_dev"}
        with pytest.raises(Exception) as ei:
            ef.load_vera_env(applied, log=True, modules={})
        assert getattr(ei.value, "code", None) == "E6" and "DB_ADMIN_PASSWORD" in str(ei.value)
        assert applied == {"VERA_ENV_CONTEXT": "local_dev"}, "nothing may be applied on an E6 refusal"
        excs.append(ei.value)
        # combined: INTERNAL_DB_URL + DATABASE_URL + admin-floor together: the E6 floor check runs first
        vf.write_text(vf.read_text() + f"INTERNAL_DB_URL={c['url']}\nDATABASE_URL={c['url2']}\n")
        vf.chmod(0o600)
        with pytest.raises(Exception) as ei:
            ef.load_vera_env({"VERA_ENV_CONTEXT": "local_dev"}, log=True, modules={})
        assert getattr(ei.value, "code", None) == "E6"
        excs.append(ei.value)
        vf.write_text(_allowed_only_env_text(c))
        # mode 0644, symlink, missing file
        vf.chmod(0o644)
        with pytest.raises(Exception) as ei:
            ef.preflight_file(vf)
        excs.append(ei.value)
        vf.chmod(0o600)
        link = c["env"] / ".env-link"
        link.symlink_to(vf)
        with pytest.raises(Exception) as ei:
            ef.preflight_file(link)
        excs.append(ei.value)
        link.unlink()
        vf.unlink()
        with pytest.raises(Exception) as ei:
            ef.preflight_file(vf)
        excs.append(ei.value)
    codes = {getattr(e, "code", None) for e in excs}
    assert codes >= {"F4", "E6", "E3", "E4", "E5"}, codes
    out = capsys.readouterr()
    _assert_absent(c, {"caplog": caplog.text, "out": out.out, "err": out.err, **_exc_channels("e", excs)})


def test_canary_apply_migrations_plan_via_ctx(canary):
    c = canary
    try:
        am = importlib.import_module("db.apply_migrations")
    except Exception:
        pytest.skip("ENV-C db.apply_migrations not importable as a module")
    if not hasattr(am, "Ctx") or not hasattr(am, "main"):
        pytest.skip("ENV-C Ctx seam not present")
    out, err = io.StringIO(), io.StringIO()
    calls = []
    ctx = am.Ctx(prod_env_files=[c["env"] / ".env-intern", c["env"] / ".env-vera"],
                 local_env_file=c["env"] / ".env-local-pg", marker_file=c["env"] / ".test-db-marker",
                 environ={"VERA_DB_URL_RO": os.environ["VERA_DB_URL_RO"]}, connect=lambda *a, **k: calls.append(1),
                 isatty=lambda: False, out=out, err=err, install_guard=lambda *a, **k: None)
    excs = []
    try:
        am.main(["plan"], ctx)
    except BaseException as e:      # SystemExit or a refusal: both acceptable, text still scanned
        excs.append(e)
    assert calls == []
    _assert_absent(c, {"out": out.getvalue(), "err": err.getvalue(), **_exc_channels("e", excs)})
