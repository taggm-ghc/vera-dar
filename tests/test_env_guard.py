"""K2.S-a: the test DB boundary (tests/vera_test_boundary.py) and its install ordering.

Every value here is a FAKE sentinel: hosts under .invalid, a Render-pattern host that is only ever compared (the
boundary refuses before any resolution), and made-up credentials. Nothing here connects anywhere; the tests prove that
refusals happen before any socket opens.
"""
import ast
import importlib.util
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

bt = sys.modules["tests.vera_test_boundary"]  # the single instance loaded by the conftests

REPO = Path(__file__).resolve().parent.parent
SENT = "postgresql://sentinel_user:sentinel_pw@db-sentinel.invalid:5432/sentinel_db"
SENT_RENDER = "postgresql://sentinel_user:sentinel_pw@dpg-sentinel0000-a.oregon-postgres.render.com:5432/sentinel_db"
SENT_ADMIN = "postgresql://sentinel_dbadmin:sentinel_pw@db-sentinel-test.invalid:5432/sentinel_test_db"
SENT_TEST = "postgresql://sentinel_tester:sentinel_pw@db-sentinel-test.invalid:5432/sentinel_test_db"
SECRET_BITS = ("sentinel_pw", "db-sentinel", "sentinel_user", "sentinel_db", "GUARD-SENTINEL")


@pytest.fixture
def no_sockets(monkeypatch):
    """Any socket use fails the test: proves a refusal happened before a connection attempt."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError("a socket was used")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket.socket, "connect_ex", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    return calls


SHIM = '''import pathlib
_real = pathlib.Path(r"%s")
_g = {"__file__": str(_real), "__name__": "real_vera_conftest"}
exec(compile(_real.read_text(), str(_real), "exec"), _g)
globals().update({k: v for k, v in _g.items() if k.startswith(("pytest_", "_vera_"))})
''' % (REPO / "tests" / "conftest.py")


def run_child(tmp_path, files: dict, args=(), env=None, shim=True, cwd=None, pre=""):
    """Run a child pytest in tmp_path with an explicit minimal environment (no outer secrets). `pre` is Python source
    run in the child's conftest before the real one (used to force the isolation status)."""
    if shim:
        (tmp_path / "conftest.py").write_text(pre + SHIM)
    for name, body in files.items():
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    e = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
         "HOME": str(tmp_path)}
    e.update(env or {})
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
                          cwd=cwd or tmp_path, env=e, capture_output=True, text=True, timeout=180)


def no_secrets(text: str):
    for bit in SECRET_BITS:
        assert bit not in text, bit


# ------------------------------------------------------------------------------------------------ G1 ordering
def test_install_precedes_app_imports():
    assert bt.STATE["installed"] is True and bt.STATE["app_modules_at_install"] == []
    import dotenv
    import dotenv.main
    assert dotenv.load_dotenv is bt._noop_load_dotenv and dotenv.main.load_dotenv is bt._noop_load_dotenv
    assert dotenv.dotenv_values is bt._noop_dotenv_values and dotenv.main.dotenv_values is bt._noop_dotenv_values
    assert bt.install() is bt.STATE  # idempotent


def test_install_refuses_when_app_module_already_imported(monkeypatch):
    monkeypatch.setitem(bt.STATE, "installed", False)
    monkeypatch.setitem(sys.modules, "vera.fake_probe_module", type(sys)("vera.fake_probe_module"))
    with pytest.raises(RuntimeError, match="after application import"):
        bt.install()


def test_polluted_outer_env_is_invisible_to_tests():
    assert [n for n in os.environ if bt.guarded(n)] == []


# ------------------------------------------------------------------------------------------------ G2 ordering (child)
PROBE_ENV = ('VERA_DB_URL_WO=%s\nGROQ_API_KEY=GUARD-SENTINEL-KEY\nDB_ADMIN_PASSWORD=GUARD-SENTINEL\n' % SENT)
PROBE_OK = '''import os
from dotenv import load_dotenv
load_dotenv()                      # module import time, before any fixture: must be a no-op
import vera.memory_store           # an application import after install
def test_clean():
    assert "VERA_DB_URL_WO" not in os.environ and "GROQ_API_KEY" not in os.environ
    assert "DB_ADMIN_PASSWORD" not in os.environ
    assert not any("guard-sentinel" in v.lower() or "db-sentinel" in v.lower() for v in os.environ.values())
'''
PROBE_SWALLOW = '''from sqlalchemy import create_engine
def test_swallowed_refusal():
    try:
        create_engine("%s").connect()
    except Exception:
        pass
''' % SENT


def test_ordering_with_sentinel_dotenv_subprocess(tmp_path):
    r = run_child(tmp_path, {".env": PROBE_ENV, "test_probe.py": PROBE_OK})
    assert r.returncode == 0, r.stdout + r.stderr
    no_secrets(r.stdout + r.stderr)


def test_swallowed_refusal_still_fails_the_test(tmp_path):
    r = run_child(tmp_path, {"test_swallow.py": PROBE_SWALLOW})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "test boundary violation" in out and "1 DB" in out, out
    no_secrets(out)


# ------------------------------------------------------------------------------------------------ G3 engine paths
@pytest.mark.boundary_probe
def test_db_boundary_covers_every_engine_path(no_sockets):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    import psycopg2

    engines = {
        "connect": lambda: create_engine(SENT).connect(),
        "begin": lambda: create_engine(SENT).begin().__enter__(),
        "session": lambda: Session(create_engine(SENT)).execute(__import__("sqlalchemy").text("select 1")),
        "pre_ping": lambda: create_engine(SENT, pool_pre_ping=True).connect(),
        "render_host": lambda: create_engine(SENT_RENDER).connect(),
        "creator": lambda: create_engine(
            "postgresql+psycopg2://", creator=lambda: psycopg2.connect(host="db-sentinel.invalid", user="sentinel_user",
                                                                         dbname="sentinel_db")).connect(),
        "direct_driver": lambda: psycopg2.connect(host="db-sentinel.invalid", user="sentinel_user", dbname="sentinel_db"),
        "direct_dsn": lambda: psycopg2.connect(SENT),
    }
    for name, fn in engines.items():
        with pytest.raises(bt.DBBoundaryError):
            fn()
    assert len(bt.STATE["violations"]) >= len(engines)
    assert no_sockets == []


def test_no_async_or_other_drivers_installed():
    for mod in ("asyncpg", "psycopg", "pg8000"):
        assert importlib.util.find_spec(mod) is None, f"{mod} is importable: extend the boundary (B2) and G3 first"


# ------------------------------------------------------------------------------------------------ G6 production URLs
def _decide(target, test_url, **over):
    env = {"VERA_TEST_DB_DISPOSABLE": "1", "VERA_TEST_DB_URL": test_url}
    kw = dict(live_item=True, live_enabled=True, env=env, deny_fp=frozenset(), deny_user_fp=frozenset())
    kw.update(over)
    return bt.decide(target, **kw)


def _pre(target, test_url, **over):
    env = {"VERA_TEST_DB_DISPOSABLE": "1", "VERA_TEST_DB_URL": test_url}
    kw = dict(live_item=True, live_enabled=True, env=env, deny_fp=frozenset(), deny_user_fp=frozenset())
    kw.update(over)
    return bt.live_preconditions(target, **kw)


def test_decide_never_allows_the_live_db_path_owner_decision():
    """Owner decision 2026-10-19: an unverified DB is PRODUCTION and disposable verification is disabled, so decide()
    is False even when every static precondition holds (VERA_TEST_DB_DISPOSABLE=1 can never allow)."""
    t = ("db-sentinel-test.invalid", 5432, "sentinel_test_db", "sentinel_tester")
    assert _decide(t, SENT_TEST) is False
    assert _pre(t, SENT_TEST) == bt.VERIFY_DISABLED == "DISPOSABLE-VERIFY-DISABLED-pending-isolation-review"
    assert _pre(("other.invalid", 5432, "sentinel_test_db", "sentinel_tester"), SENT_TEST) == "not_test_url"
    assert _pre(t, SENT_TEST, live_item=False) == "not_live"
    assert _pre(t, SENT_TEST, live_enabled=False) == "not_live"
    assert _pre(t, "") == "no_test_url"
    assert _pre(t, SENT_TEST, deny_all=True) == "deny_all"
    assert _pre(t, SENT_TEST, env={"VERA_TEST_DB_URL": SENT_TEST}) == "no_disposable_flag"
    for over in ({}, {"live_item": False}, {"deny_all": True}):
        assert _decide(t, SENT_TEST, **over) is False


def test_decide_has_no_allow_path_in_source():
    import inspect
    src = inspect.getsource(bt.decide)
    assert "return True" not in src and src.rstrip().endswith("return False")


@pytest.mark.parametrize("label,test_url,user", [
    ("render_host", SENT_RENDER, "sentinel_user"),
    ("admin_user", SENT_ADMIN, "sentinel_dbadmin"),
    ("production_fingerprint", SENT, "sentinel_user"),
])
def test_test_db_url_pointing_at_production_rejected(label, test_url, user):
    parsed = bt._from_url(test_url)
    deny_fp = frozenset({bt.fingerprint(*parsed[:3])}) if label == "production_fingerprint" else frozenset()
    target = (parsed[0], parsed[1], parsed[2], user)
    assert _decide(target, test_url, deny_fp=deny_fp) is False


def test_denied_user_fingerprint_rejected():
    t = ("db-sentinel-test.invalid", 5432, "sentinel_test_db", "app_account_sentinel")
    url = "postgresql://app_account_sentinel:x@db-sentinel-test.invalid:5432/sentinel_test_db"
    assert _pre(t, url) == bt.VERIFY_DISABLED and _decide(t, url) is False
    assert _pre(t, url, deny_user_fp=frozenset({bt._h("app_account_sentinel")})) == "role"
    assert _decide(t, url, deny_user_fp=frozenset({bt._h("app_account_sentinel")})) is False


def test_dbname_in_pd_suffix_and_ip_rejected():
    t = ("db-sentinel-test.invalid", 5432, "sentinel_test_db", "sentinel_tester")
    assert _pre(t, SENT_TEST, deny_db_fp=frozenset({bt._h("sentinel_test_db")})) == "dbname"
    assert _pre(t, SENT_TEST, deny_suffix=frozenset({(len(".invalid"), bt._h(".invalid"))})) == "suffix"
    ip = "postgresql://sentinel_tester:x@10.9.8.7:5432/sentinel_test_db"
    assert _pre(("10.9.8.7", 5432, "sentinel_test_db", "sentinel_tester"), ip,
                deny_ip_fp=frozenset({bt._h("10.9.8.7")})) == "ip"
    trailing = ("dpg-sentinel0000-a.oregon-postgres.render.com.", 5432, "sentinel_db", "sentinel_user")
    assert _pre(trailing, SENT_RENDER.replace("render.com:", "render.com.:")) == "host_pattern"


@pytest.mark.parametrize("url", [SENT_RENDER, SENT_ADMIN])
def test_production_test_url_fails_loudly_in_a_live_session(tmp_path, url):
    child = ('import os, pytest\n@pytest.mark.live\ndef test_c():\n'
             '    from sqlalchemy import create_engine\n    create_engine(os.environ["VERA_TEST_DB_URL"]).connect()\n')
    r = run_child(tmp_path, {"test_c.py": child}, args=("--run-live-stages",),
                  env={"VERA_TEST_DB_URL": url, "VERA_TEST_DB_DISPOSABLE": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "test boundary violation" in out, out
    no_secrets(out)


# ------------------------------------------------------------------------------------------------ G7 missing config
def test_m2_style_skip_reason_names_the_missing_variable():
    spec = importlib.util.spec_from_file_location("m2_probe", REPO / "tests" / "test_m2_integration.py")
    src = (REPO / "tests" / "test_m2_integration.py").read_text()
    assert "VERA_TEST_DB_URL" in src and "VERA_TEST_DB_DISPOSABLE" in src and spec is not None


@pytest.mark.boundary_probe
def test_missing_config_never_connects(no_sockets, monkeypatch):
    from sqlalchemy import create_engine
    monkeypatch.delenv("VERA_TEST_DB_URL", raising=False)
    monkeypatch.delenv("VERA_TEST_DB_DISPOSABLE", raising=False)
    with pytest.raises(bt.DBBoundaryError):
        create_engine(SENT).connect()
    assert no_sockets == [] and bt.STATE["violations"][-1]["kind"] == "db"


# ------------------------------------------------------------------------------------------------ G8 skip is not a pass
def test_violation_fails_even_if_test_skips(tmp_path):
    child = ('import pytest\nfrom sqlalchemy import create_engine\ndef test_s():\n    try:\n'
             '        create_engine("%s").connect()\n    except Exception:\n        pass\n    pytest.skip("not a pass")\n' % SENT)
    r = run_child(tmp_path, {"test_s.py": child})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "failed" in out and "skipped" not in out, out


def test_probe_marker_suppresses_the_failure(tmp_path):
    child = ('import pytest\nfrom sqlalchemy import create_engine\n@pytest.mark.boundary_probe\ndef test_p():\n'
             '    with pytest.raises(Exception):\n        create_engine("%s").connect()\n' % SENT)
    r = run_child(tmp_path, {"test_p.py": child})
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------------------------------------ G10 /ask, G15 memory
def _ask_app(monkeypatch, result):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)  # FAKE, well-formed
    monkeypatch.setenv("VERA_DB_URL_WO", SENT)
    monkeypatch.setenv("VERA_DB_URL_RO", SENT)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from fastapi.testclient import TestClient
    from tests.test_item79_corpus_admission import _llm
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (result, "groq:m"))
    monkeypatch.setattr(main, "llm_call_for", lambda entry: _llm([0.9]))  # no provider SDK call
    return TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})


@pytest.mark.boundary_probe
def test_unstubbed_ask_cannot_reach_db_or_provider(monkeypatch, no_sockets):
    from tests.test_item79_corpus_admission import _src
    from vera.ask_service import AskResult
    src = _src(1, "https://arxiv.org/abs/1")
    src["provider"] = "arxiv"
    res = AskResult("A", 1, 0.0, sources=(src,), accepted=True, result_kind="reading_list")
    import main
    # R-028: the shipped Demo Day config never reaches the DB; to keep this boundary probe non-vacuous the corpus switch
    # (only) is turned on here. The DB refusal and no-socket assertions below are unchanged.
    import json
    cfg = json.loads((Path(main.__file__).parent / "config" / "ask-provider-chain.json").read_text())["corpus"]
    monkeypatch.setattr(main, "CORPUS", dict(cfg, enabled=True))
    r = _ask_app(monkeypatch, res)
    assert r.status_code == 200, r.text
    assert any(v["kind"] == "db" for v in bt.STATE["violations"]), "admission/memory never tried the DB: test is vacuous"
    assert no_sockets == []


@pytest.mark.boundary_probe
def test_unstubbed_memory_from_env_cannot_connect(monkeypatch, no_sockets):
    from vera import memory_store as ms
    monkeypatch.setattr(ms, "_SERVICE", {})
    monkeypatch.setenv("VERA_DB_URL_RO", SENT)
    monkeypatch.setenv("VERA_DB_URL_WO", SENT)
    from vera import answer_contract as ac
    blk = ms.load_memory_block()
    # R-028: the shipped config has memory.store off and layers L0; this probe turns both on (only) so the DB refusals
    # it asserts are still exercised. Refusal and no-socket assertions are unchanged.
    blk = dict(blk, store=dict(blk["store"], enabled=True))
    monkeypatch.setattr(ac, "layers", lambda *a, **k: ["L0", "L2"])
    svc = ms.memory_from_env(blk)
    assert svc is not None
    assert svc.recall("quantized inference latency") == []          # V3: evidence recall off, no SQL
    assert svc.reader.recall_discovery.__name__ == "recall_discovery"
    assert svc.recall_discovery("quantized inference latency transformer") == []   # swallowed refusal
    cand = {"title": "Efficient Int8 Weight Compression for Language Models", "url": "https://arxiv.org/abs/2401.00001",
            "provider": "arxiv", "licence_decision": "allow", "identifier": "arXiv:2401.00001"}
    n, skipped = svc.write_discovery_note([cand], "unrelated words entirely different")
    assert n == 0
    assert sum(1 for v in bt.STATE["violations"] if v["kind"] == "db") >= 2
    assert no_sockets == []


# ------------------------------------------------------------------------------------------------ G11 messages
@pytest.mark.boundary_probe
def test_boundary_error_texts_have_no_secrets():
    from sqlalchemy import create_engine
    with pytest.raises(bt.DBBoundaryError) as ei:
        create_engine(SENT_RENDER).connect()
    for text in (str(ei.value), repr(ei.value), bt.REFUSAL_TEXT):
        no_secrets(text)
        assert "render.com" not in text and "dpg-" not in text
    bt.STATE["violations"].append({"kind": "db", "phase": "call"})
    no_secrets(bt.violations_message("call"))
    assert bt.violations_message("call", probe=True) is None


# ------------------------------------------------------------------------------------------------ G12 m6 writer
def test_m6_store_results_cannot_write_without_url():
    from vera.m6.m6_runner import store_results
    with pytest.raises(RuntimeError, match="is not set"):
        store_results("r1", {"response_text": "x"}, {})


@pytest.mark.boundary_probe
def test_m6_store_results_refused_with_sentinel_url(monkeypatch, no_sockets):
    from vera.m6.m6_runner import store_results
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("EXTERNAL_DB_URL", SENT)
    with pytest.raises(bt.DBBoundaryError):
        store_results("r1", {"response_text": "x"}, {})
    assert no_sockets == [] and bt.STATE["violations"]


# ------------------------------------------------------------------------------------------------ G13 collection
def test_root_conftest_ignores_db_and_scripts_dirs():
    tree = ast.parse((REPO / "conftest.py").read_text())
    vals = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "collect_ignore" for t in n.targets)]
    assert vals and {"db", "scripts", "vera/m6/test_m6_schema_sync.py"} <= set(vals[0])


def test_db_and_scripts_dirs_not_collected_by_bare_pytest(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "vera_test_boundary.py").write_text((REPO / "tests" / "vera_test_boundary.py").read_text())
    (tmp_path / "conftest.py").write_text((REPO / "conftest.py").read_text())
    boom = "raise RuntimeError('collected an operator dir')\ndef test_x():\n    pass\n"
    files = {"db/test_x.py": boom, "scripts/test_x.py": boom, "tests/test_ok.py": "def test_ok():\n    pass\n"}
    r = run_child(tmp_path, files, args=("--collect-only", "tests"), shim=False)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "test_ok" in out and "db/test_x" not in out and "scripts/test_x" not in out, out
    # M3-A': a bare run (args = the invocation dir) and explicit operator paths are refused before any import
    for args in (("--collect-only",), ("--collect-only", "db/test_x.py"), ("--collect-only", "scripts/test_x.py::test_x"),
                 ("--collect-only", "tests/../db/test_x.py")):
        r = run_child(tmp_path, {}, args=args, shim=False)
        out = r.stdout + r.stderr
        assert r.returncode == 4 and "only `tests` or paths under tests/" in out, (args, out)
        assert "collected an operator dir" not in out, (args, out)


# ------------------------------------------------------------------------------------------------ G14 pass-through
@pytest.mark.boundary_probe
def test_sqlite_allowed(no_sockets):
    from sqlalchemy import create_engine, text
    with create_engine("sqlite://").connect() as c:
        assert c.execute(text("select 1")).scalar() == 1
    assert bt.STATE["violations"] == []


def test_explicit_disposable_path_is_refused_owner_decision(monkeypatch):
    """Formerly the one allowed configuration. Owner decision 2026-10-19: still refused (VERIFY_DISABLED), recorded,
    and nothing is pinned or resolved."""
    resolved = []
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: resolved.append(host) or [(2, 1, 6, "", ("127.0.0.1", 0))])
    monkeypatch.setenv("VERA_TEST_DB_URL", SENT_TEST)
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    monkeypatch.setitem(bt.STATE, "deny_all", False)
    monkeypatch.setitem(bt._SEALED, "deny_all", False)
    bt.begin_item(live_item=True)
    try:
        cp = {"host": "db-sentinel-test.invalid", "port": 5432, "dbname": "sentinel_test_db", "user": "sentinel_tester"}
        with pytest.raises(bt.DBBoundaryError):
            bt.check((), cp)
        assert "hostaddr" not in cp and resolved == [] and bt.STATE["violations"][-1]["kind"] == "db"
    finally:
        bt.begin_item(live_item=False)                    # leave the recorder clean for the next test


def test_pattern_catches_future_names():
    for n in ("VERA_DB_URL_ADMIN", "VERA_DB_PASSWORD_vera_eval_ro", "DB_ADMIN_ROLE", "DB_ACCOUNT", "FOO_API_KEY",
              "NEW_DB_URL_RO", "PGPASSWORD", "X_TOKEN", "OPENROUTER_API_KEY", "VERA_UI_URL", "DATABASE_URL"):
        assert bt.guarded(n), n
    for n in ("PATH", "HOME", "VERA_TEST_DB_URL", "VERA_TEST_DB_DISPOSABLE", "VERA_PUBLIC_MODE", "PYTHONDONTWRITEBYTECODE"):
        assert not bt.guarded(n), n


# ------------------------------------------------------------------------------------------------ G16 static
def _code_token_hits(path: Path) -> list:
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.Name, ast.Attribute)):
            ident = node.id if isinstance(node, ast.Name) else node.attr
            if ident in ("get_admin_engine", "ro_connection"):
                hits.append((node.lineno, ident))
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in ("get_admin_engine", "ro_connection"):
            hits.append((getattr(node, "lineno", 0), node.name))
        elif isinstance(node, ast.Call):
            f = ast.unparse(node.func)
            if f == "psycopg2.connect":
                hits.append((node.lineno, f))
            # INT2 F-1: raw driver entry points that skip psycopg2.connect (and so the boundary's check())
            if f.endswith("._connect") or f == "_connect" or f.endswith("extensions.connection") \
                    or f.endswith("_ext.connection"):
                hits.append((node.lineno, f))
            if f in ("os.getenv", "os.environ.get", "getenv", "env.get") and node.args \
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value in ("EXTERNAL_DB_URL", "INTERNAL_DB_URL"):
                hits.append((node.lineno, node.args[0].value))
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ" \
                and isinstance(node.slice, ast.Constant) and node.slice.value in ("EXTERNAL_DB_URL", "INTERNAL_DB_URL"):
            hits.append((node.lineno, node.slice.value))
    return hits


def test_no_production_connector_in_tests():
    bad = {}
    for p in sorted((REPO / "tests").glob("*.py")):
        if p.name in ("vera_test_boundary.py", "test_env_guard.py"):
            continue
        h = _code_token_hits(p)
        if h:
            bad[p.name] = h
    assert bad == {}, bad


def test_no_creator_engines_in_repo_code():
    """B3: no create_engine(..., creator=...) outside tests that go through the covered psycopg2.connect wrapper."""
    offenders = []
    files = [*(REPO / "vera").rglob("*.py"), *(REPO / "scripts").glob("*.py"), REPO / "main.py", REPO / "streamlit_app.py"]
    for p in files:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("create_engine") \
                    and any(k.arg == "creator" for k in node.keywords):
                offenders.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert offenders == []


# ------------------------------------------------------------------------------------------------ G17 m4/m5
def test_m4_m5_targets_disposable_db_only():
    src = (REPO / "tests" / "test_m4_m5_integration.py").read_text()
    assert 'os.environ["VERA_TEST_DB_URL"]' in src and "VERA_TEST_DB_DISPOSABLE" in src
    assert "pytest.mark.live" in src and "pytest.mark.skipif" in src
    assert "get_admin_engine" not in src and "week-1v2" not in src and "ai-engineering-bootcamp" not in src


def test_m4_m5_admin_target_refused_in_a_live_session(tmp_path):
    body = (REPO / "tests" / "test_m4_m5_integration.py").read_text()
    r = run_child(tmp_path, {"test_m4_m5_integration.py": body}, args=("--run-live-stages",),
                  env={"VERA_TEST_DB_URL": SENT_ADMIN, "VERA_TEST_DB_DISPOSABLE": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "test boundary violation" in out and "1 DB" in out, out
    no_secrets(out)


# ------------------------------------------------------------------------------------------------ G18 operator script
def _script():
    spec = importlib.util.spec_from_file_location("live_catalog_check_probe", REPO / "scripts" / "live_catalog_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_catalog_script_static_safety(monkeypatch, no_sockets):
    mod = _script()
    reads = []
    monkeypatch.setattr(os, "getenv", lambda *a, **k: reads.append(a) or None)
    monkeypatch.setattr(os.environ, "get", lambda *a, **k: reads.append(a) or None, raising=False)
    import builtins
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: reads.append(a) or real_open(*a, **k))
    assert mod.main([]) == 2 and reads == []              # no flag: exit 2 before reading any setting
    monkeypatch.undo()
    assert no_sockets == []

    src = (REPO / "scripts" / "live_catalog_check.py").read_text()
    assert mod.ROLE == "vera_eval_ro"
    body = src.split("def _run(")[1]
    assert body.index("privilege_ok(conn)") < body.index("live_catalog(conn)")


def test_live_catalog_script_connect_is_read_only_first_and_uses_ro_url_only():
    src = (REPO / "scripts" / "live_catalog_check.py").read_text()
    assert "VERA_DB_URL_RO" in src and "u.username" in src
    for gone in ("EXTERNAL_DB_URL", "VERA_DB_PASSWORD_vera_eval_ro", "SET SESSION CHARACTERISTICS"):
        assert gone not in src.split('"""', 2)[2], gone
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "connect")
    calls = sorted((n.lineno, ast.unparse(n.func)) for n in ast.walk(fn) if isinstance(n, ast.Call))
    names = [c[1] for c in calls]
    assert "conn.set_session" in names and not any(n.endswith(".execute") or n.endswith(".cursor") for n in names)
    assert "default_transaction_read_only=on" in src and "statement_timeout=" in src
    for col in ("transaction_read_only", "has_table_privilege"):
        assert col in _script().GUARD_QUERY


class _FakeCursor:
    def __init__(self, conn): self.conn = conn
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None):
        if self.conn.raise_on == "guard" and "current_user" in sql:
            raise RuntimeError("secret message must not print")
        if self.conn.raise_on == "catalog" and "information_schema" in sql:
            raise RuntimeError("secret message must not print")
        self.sql = sql
    def fetchone(self): return self.conn.guard_row
    def fetchall(self): return self.conn.catalog_rows


class _FakeConn:
    def __init__(self, guard_row, catalog_rows, raise_on=None):
        self.guard_row, self.catalog_rows, self.raise_on, self.closed = guard_row, catalog_rows, raise_on, False
    def cursor(self): return _FakeCursor(self)
    def close(self): self.closed = True


GOOD_GUARD = ("vera_eval_ro", False, False, False, False, "on", False)


def _catalog_rows(drop=None):
    from tests import sql_contract_support as sc
    rows = [(t, c) for (t, c) in sorted(sc.collect()) if (t, c) != drop]
    return rows


def _drive(monkeypatch, capsys, conn=None, connect_raises=False, collect_raises=False):
    mod = _script()
    if connect_raises:
        monkeypatch.setattr(mod, "connect", lambda: (_ for _ in ()).throw(RuntimeError("secret message must not print")))
    else:
        monkeypatch.setattr(mod, "connect", lambda: conn)
    if collect_raises:
        from tests import sql_contract_support as sc
        monkeypatch.setattr(sc, "collect", lambda: (_ for _ in ()).throw(RuntimeError("secret message must not print")))
    code = mod.main(["--i-confirm-owner-run"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("live_catalog_check script_sha256=")
    assert sum(1 for ln in lines if ln.startswith("RESULT:")) == 1 and lines[-1].startswith("RESULT: ")
    assert "Traceback" not in out and "secret message" not in out
    return code, lines[-1], out


def test_result_pass_when_catalog_satisfies_contract(monkeypatch, capsys):
    conn = _FakeConn(GOOD_GUARD, _catalog_rows())
    code, last, _ = _drive(monkeypatch, capsys, conn)
    assert (code, last) == (0, "RESULT: PASS") and conn.closed


def test_result_fail_when_a_column_is_missing(monkeypatch, capsys):
    from tests import sql_contract_support as sc
    drop = sorted(sc.collect())[0]
    code, last, out = _drive(monkeypatch, capsys, _FakeConn(GOOD_GUARD, _catalog_rows(drop)))
    assert (code, last) == (1, "RESULT: FAIL") and f"missing: {drop[0]}.{drop[1]}" in out


@pytest.mark.parametrize("label,row", [
    ("wrong_user", ("vera_rw",) + GOOD_GUARD[1:]),
    ("superuser", ("vera_eval_ro", True) + GOOD_GUARD[2:]),
    ("createrole", ("vera_eval_ro", False, True) + GOOD_GUARD[3:]),
    ("createdb", ("vera_eval_ro", False, False, True) + GOOD_GUARD[4:]),
    ("schema_create", GOOD_GUARD[:4] + (True,) + GOOD_GUARD[5:]),
    ("session_not_read_only", GOOD_GUARD[:5] + ("off", False)),
    ("table_write_privilege", GOOD_GUARD[:6] + (True,)),
    ("no_row", None),
])
def test_result_cannot_verify_when_guard_fails(monkeypatch, capsys, label, row):
    conn = _FakeConn(row, _catalog_rows())
    code, last, out = _drive(monkeypatch, capsys, conn)
    assert (code, last) == (3, "RESULT: CANNOT_VERIFY") and conn.closed
    assert "checked " not in out  # the catalog query never ran


def test_result_cannot_verify_without_credentials_or_on_connect_error(monkeypatch, capsys):
    assert _drive(monkeypatch, capsys, None)[:2] == (4, "RESULT: CANNOT_VERIFY")
    assert _drive(monkeypatch, capsys, connect_raises=True)[:2] == (4, "RESULT: CANNOT_VERIFY")


@pytest.mark.parametrize("raise_on", ["guard", "catalog"])
def test_result_cannot_verify_when_a_query_raises(monkeypatch, capsys, raise_on):
    conn = _FakeConn(GOOD_GUARD, _catalog_rows(), raise_on=raise_on)
    code, last, out = _drive(monkeypatch, capsys, conn)
    assert (code, last) == (4, "RESULT: CANNOT_VERIFY") and "RuntimeError" in out and conn.closed


def test_result_cannot_verify_on_empty_catalog_or_unexpected_error(monkeypatch, capsys):
    assert _drive(monkeypatch, capsys, _FakeConn(GOOD_GUARD, []))[:2] == (4, "RESULT: CANNOT_VERIFY")
    assert _drive(monkeypatch, capsys, _FakeConn(GOOD_GUARD, _catalog_rows()), collect_raises=True)[:2] == \
        (4, "RESULT: CANNOT_VERIFY")


def test_connect_with_fake_driver_sets_read_only_before_any_statement(monkeypatch):
    mod = _script()
    order = []

    class FakeConn:
        def set_session(self, **kw): order.append(("set_session", kw))
        def cursor(self): order.append(("cursor",))

    fake = type(sys)("psycopg2")
    fake.connect = lambda dsn, **kw: order.append(("connect", {**kw, "dsn": dsn})) or FakeConn()
    monkeypatch.setitem(sys.modules, "psycopg2", fake)
    url = "postgresql://vera_eval_ro:pw-sentinel@db-sentinel.invalid:5432/sentinel_db"
    monkeypatch.setenv("VERA_DB_URL_RO", url)
    assert mod.connect() is not None
    assert [o[0] for o in order] == ["connect", "set_session"]
    kw = order[0][1]
    assert kw["dsn"].startswith("postgresql://vera_eval_ro:pw-sentinel@db-sentinel.invalid:5432/sentinel_db")
    assert kw["sslmode"] == "require" and "statement_timeout" in kw["options"]
    assert "default_transaction_read_only=on" in kw["options"] and order[1][1] == {"readonly": True, "autocommit": True}
    order.clear()
    for bad in ("postgresql://vera_rw:pw@db-sentinel.invalid/sentinel_db", "postgresql://vera_eval_ro@db-sentinel.invalid/x", ""):
        monkeypatch.setenv("VERA_DB_URL_RO", bad)
        assert mod.connect() is None
    assert order == []                      # never connected for a wrong user or a missing password


def test_live_catalog_script_prints_and_sql_are_safe():
    src = (REPO / "scripts" / "live_catalog_check.py").read_text()
    tree = ast.parse(src)
    allowed = {"CONFIRM_FLAG", "type(exc).__name__", "len(found)", "len(problems)", "p", "digest", "result"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "print":
            for sub in ast.walk(node):
                if isinstance(sub, ast.FormattedValue):
                    assert ast.unparse(sub.value) in allowed, ast.unparse(sub.value)
            for a in node.args:
                assert not isinstance(a, (ast.Name, ast.Attribute)) or ast.unparse(a) in allowed
    mod = _script()
    sql_like = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.lstrip().upper().startswith(("SELECT", "SET ", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE"))]
    assert sorted(sql_like) == sorted([mod.GUARD_QUERY, mod.CATALOG_QUERY])
    assert all(q.lstrip().lower().startswith("select") for q in sql_like)
    assert mod.CATALOG_QUERY.startswith("SELECT table_name, column_name FROM information_schema.columns")


def test_live_catalog_script_refused_path_evidence_lines(capsys):
    import hashlib
    mod = _script()
    assert mod.main([]) == 2                      # refused: no database, no setting read
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "live_catalog_check script_sha256=" + hashlib.sha256(
        (REPO / "scripts" / "live_catalog_check.py").read_bytes()).hexdigest()
    assert out[-1] == "RESULT: CANNOT_VERIFY"
    for line in out:
        low = line.lower()
        assert "@" not in line and "://" not in line and "host" not in low and "password" not in low, line


# ================================================================================================ K2.S-b: network
@pytest.fixture
def loopback_server():
    srv = socket.create_server(("127.0.0.1", 0))
    yield srv
    srv.close()


@pytest.mark.boundary_probe
def test_network_boundary_blocks_requests_httpx_and_sdk():
    import asyncio

    import httpx
    import openai
    import requests

    attempts = {
        "requests": lambda: requests.get("https://example.org", timeout=2),
        "httpx": lambda: httpx.get("https://example.org", timeout=2),
        "httpx_async": lambda: asyncio.run(_httpx_async_get()),
        "openai_sdk": lambda: openai.OpenAI(api_key="GUARD-SENTINEL", max_retries=0, timeout=2).models.list(),
        "create_connection": lambda: socket.create_connection(("93.184.216.34", 80), timeout=1),
    }
    for name, fn in attempts.items():
        before = len([v for v in bt.STATE["violations"] if v["kind"] == "net"])
        with pytest.raises(Exception):
            fn()
        after = len([v for v in bt.STATE["violations"] if v["kind"] == "net"])
        assert after > before, f"{name}: no network violation recorded"


async def _httpx_async_get():
    import httpx
    async with httpx.AsyncClient(timeout=2) as c:
        return await c.get("https://example.org")


@pytest.mark.boundary_probe
def test_reserved_tld_names_fail_like_the_real_resolver_without_a_violation():
    for host in ("db-sentinel.invalid", "x.test", "x.example", "x.localhost"):
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo(host, 80)
    assert bt.STATE["violations"] == []


@pytest.mark.boundary_probe
def test_non_loopback_resolution_and_connect_are_refused_and_recorded():
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.org", 443)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bt.NetworkBoundaryError):
            s.connect(("93.184.216.34", 80))
        with pytest.raises(bt.NetworkBoundaryError):
            s.connect_ex(("10.255.255.1", 80))
    finally:
        s.close()
    assert [v["kind"] for v in bt.STATE["violations"]] == ["net", "net", "net"]


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_loopback_and_unix_sockets_still_work(loopback_server, tmp_path, monkeypatch):
    host, port = loopback_server.getsockname()
    c = socket.create_connection((host, port), timeout=2)
    c.close()
    assert socket.getaddrinfo("localhost", port)
    a, b = socket.socketpair()
    a.close(); b.close()
    # The runner's temp dir path can exceed the ~107-byte AF_UNIX limit; chdir into it and use a relative
    # name (same technique as _unix_server below) so the socket file still lives inside the allowed temp area.
    monkeypatch.chdir(tmp_path)
    sp = "u.sock"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sp); srv.listen(1)
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cli.connect(sp)
    cli.close(); srv.close()
    assert bt.STATE["violations"] == []


def test_network_error_text_has_no_secrets():
    for text in (bt.NET_REFUSAL_TEXT, str(bt.NetworkBoundaryError(bt.NET_REFUSAL_TEXT))):
        no_secrets(text)
        assert "://" not in text and "@" not in text


# ------------------------------------------------------------------------------------------------ G5 proxy scrub
PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")


@pytest.mark.parametrize("name", PROXY_NAMES)
def test_proxy_names_are_guarded(name):
    assert bt.guarded(name)


def test_proxy_env_scrubbed_in_a_child_run(tmp_path):
    child = ('import os\ndef test_p():\n    assert not [n for n in os.environ if n.lower().endswith("proxy")]\n')
    r = run_child(tmp_path, {"test_p.py": child},
                  env={"HTTPS_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "x"})
    assert r.returncode == 0, r.stdout + r.stderr


def test_future_proxy_and_provider_names_are_caught_and_plain_names_are_not():
    for n in ("FTP_PROXY", "socks_proxy", "NEW_PROVIDER_API_KEY"):
        assert bt.guarded(n), n
    for n in ("PATH", "HOME", "PYTHONDONTWRITEBYTECODE", "VERA_PUBLIC_MODE"):
        assert not bt.guarded(n), n


# ------------------------------------------------------------------------------------------------ G9 live items
LIVE_CHILD = (
    'import os, pytest, socket\nfrom sqlalchemy import create_engine\nbt = __import__("sys").modules["tests.vera_test_boundary"]\n'
    '@pytest.mark.live\n@pytest.mark.boundary_probe\n'
    'def test_live_item():\n'
    '    assert os.environ.get("GROQ_API_KEY") == "GUARD-SENTINEL-KEY"          # provider key restored\n'
    '    for n in ("VERA_DB_URL_WO", "DB_ADMIN_PASSWORD", "HTTPS_PROXY", "EXTERNAL_DB_URL"):\n'
    '        assert n not in os.environ, n                                         # DB, admin and proxy names never\n'
    '    assert bt.STATE["net_lifted"] is True\n'
    '    with pytest.raises(bt.DBBoundaryError):                                  # the DB boundary is never lifted\n'
    '        create_engine(os.environ["VERA_TEST_DB_URL"]).connect()\n'
    '    with pytest.raises(socket.gaierror):                                     # reserved names still fail, no real DNS\n'
    '        socket.getaddrinfo("x.invalid", 80)\n'
)
NONLIVE_CHILD = (
    'import os, pytest, socket\nbt = __import__("sys").modules["tests.vera_test_boundary"]\n'
    '@pytest.mark.boundary_probe\n'
    'def test_non_live_item():\n'
    '    assert "GROQ_API_KEY" not in os.environ and bt.STATE["net_lifted"] is False\n'
    '    with pytest.raises(socket.gaierror):\n        socket.getaddrinfo("example.org", 443)\n'
    '    assert [v["kind"] for v in bt.STATE["violations"]] == ["net"]\n'
)


def test_live_item_gets_provider_keys_not_db_names(tmp_path):
    env = {"GROQ_API_KEY": "GUARD-SENTINEL-KEY", "VERA_DB_URL_WO": SENT, "EXTERNAL_DB_URL": SENT,
           "DB_ADMIN_PASSWORD": "GUARD-SENTINEL", "HTTPS_PROXY": "http://127.0.0.1:9",
           "VERA_TEST_DB_URL": SENT_ADMIN, "VERA_TEST_DB_DISPOSABLE": "1"}  # admin user: the DB boundary must still refuse it
    r = run_child(tmp_path, {"test_live.py": LIVE_CHILD, "test_nonlive.py": NONLIVE_CHILD}, args=("--run-live-stages",), env=env)
    assert r.returncode == 0 and "2 passed" in r.stdout, r.stdout + r.stderr
    r2 = run_child(tmp_path, {}, env=env)  # without the flag the live item is skipped and nothing is restored
    assert r2.returncode == 0 and "1 passed" in r2.stdout and "1 skipped" in r2.stdout, r2.stdout + r2.stderr


# ------------------------------------------------------------------------------------------------ S-a acceptance notes
def test_setup_phase_refusal_is_attributed_to_the_item(tmp_path):
    child = ('import pytest\nfrom sqlalchemy import create_engine\n@pytest.fixture\ndef f():\n    try:\n'
             '        create_engine("%s").connect()\n    except Exception:\n        pass\n'
             'def test_x(f):\n    pass\n' % SENT)
    r = run_child(tmp_path, {"test_x.py": child})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "test boundary violation" in out and "1 error" in out or "1 failed" in out, out
    no_secrets(out)


def test_violation_outside_any_item_fails_the_session(tmp_path):
    child = ('import socket\ntry:\n    socket.getaddrinfo("example.org", 443)\nexcept OSError:\n    pass\n'
             'def test_ok():\n    pass\n')
    r = run_child(tmp_path, {"test_o.py": child})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "outside any test" in out, out




# ------------------------------------------------------------------------------------------------ deny file: fail closed
















# ------------------------------------------------------------------------------------------------ violation tracking
@pytest.mark.boundary_probe
def test_reset_clears_the_record_but_detection_is_preserved():
    bt.record("db")
    bt.begin_item(live_item=False)                    # what the tryfirst setup hook does before every test
    assert bt.STATE["violations"] == []
    with pytest.raises(bt.DBBoundaryError):
        bt.check((), {"host": "db-sentinel.invalid", "port": 5432, "dbname": "x", "user": "u"})
    assert bt.violations_message("setup") is not None and bt.violations_message("setup", probe=True) is None


def test_second_item_starts_clean_and_first_item_still_fails(tmp_path):
    child = ('from sqlalchemy import create_engine\ndef test_a_violates():\n    try:\n'
             '        create_engine("%s").connect()\n    except Exception:\n        pass\n'
             'def test_b_is_clean():\n    assert True\n' % SENT)
    r = run_child(tmp_path, {"test_two.py": child}, args=("-rA",))
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "FAILED test_two.py::test_a_violates" in out and "PASSED test_two.py::test_b_is_clean" in out, out


def test_violation_before_the_first_item_is_an_orphan_and_fails_the_session(tmp_path):
    child = ('from sqlalchemy import create_engine\ntry:\n    create_engine("%s").connect()\nexcept Exception:\n    pass\n'
             'def test_ok():\n    pass\n' % SENT)
    r = run_child(tmp_path, {"test_orphan.py": child})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "outside any test" in out, out
    no_secrets(out)


# ================================================================================================ E2: R/config/env policy
# PLAN-ENV v2.1 E2 (ENV-D): the deny sources are R/config/env/.env-intern (policy) and .env-vera, located from the
# boundary file's path and read by a password-dropping parser. The legacy AI-Internship policy file and the
# main-checkout .env are gone. Every fixture value is a FAKE sentinel; fixtures are built under tmp_path only.
GOOD_POLICY = ("DB_HOST=db-sentinel.invalid\nDB_NAME=sentinel_db\nDB_ACCOUNT=sentinel_user\n"
               "DB_ADMIN_ROLE=sentinel_dbadmin\nDB_PASSWORD=sentinel_pw\nDB_ADMIN_PASSWORD=sentinel_pw\n")
GOOD_VERA = f"VERA_DB_URL_RO={SENT}\nOPENAI_API_KEY=sk-sentinel_pw\n"


def _env_dir(root: Path, intern=GOOD_POLICY, vera=GOOD_VERA, mode=0o600) -> Path:
    d = root / "config" / "env"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    for name, text in ((".env-intern", intern), (".env-vera", vera)):
        if text is not None:
            (d / name).write_text(text)
            (d / name).chmod(mode)
    return d


def _keep_policy_state(monkeypatch):
    """M2-A (PF-18 M-2): snapshot the session's deny-policy state; monkeypatch restores it at teardown, also when the
    test fails (bt._read_env_files() commits its fixture result to bt.STATE)."""
    monkeypatch.setitem(bt.STATE, "deny_policy", bt.STATE["deny_policy"])
    monkeypatch.setitem(bt.STATE, "deny_all", bt.STATE["deny_all"])


def test_policy_location_is_r_config_env_from_the_file_path_only():
    assert bt.ENV_DIR == Path(bt.__file__).resolve().parent.parent.parent / "config" / "env"
    assert (bt.INTERN_ENV_NAME, bt.VERA_ENV_NAME) == (".env-intern", ".env-vera")
    for gone in ("POLICY_REL", "POLICY_NAME", "policy_path", "resolve_policy", "_validate_policy_text"):
        assert not hasattr(bt, gone), gone
    src = Path(bt.__file__).read_text()
    assert "AI-Internship" not in src.split("# --- E2", 1)[1].split("# --- D1", 1)[0]


def test_policy_ok_from_r_config_env(tmp_path, monkeypatch):
    _keep_policy_state(monkeypatch)
    d = _env_dir(tmp_path)
    status, found = bt.resolve_sources(d)
    assert status == "ok" and len(found) == 2
    bt._read_env_files(d)
    assert bt.STATE["deny_policy"] == "ok" and bt.STATE["deny_all"] is False


def test_policy_parser_drops_passwords_and_keys(tmp_path):
    d = _env_dir(tmp_path)
    _status, found = bt.resolve_sources(d)
    blob = repr(found)
    assert "sentinel_pw" not in blob and "OPENAI_API_KEY" not in blob and "DB_PASSWORD" not in blob
    assert any(m.get("VERA_DB_URL_RO", "").startswith("postgresql://sentinel_user@db-sentinel.invalid") for m in found)
    reduced = bt._reduce({"VERA_DB_URL_RW": SENT, "DB_ADMIN_PASSWORD": "sentinel_pw", "DB_NAME": "sentinel_db"})
    assert "sentinel_pw" not in repr(reduced) and set(reduced) == {"VERA_DB_URL_RW", "DB_NAME"}


def test_policy_missing_fails_closed(monkeypatch, tmp_path):
    _keep_policy_state(monkeypatch)
    assert bt.resolve_sources(tmp_path / "config" / "env")[0] == "missing"            # no directory
    d = _env_dir(tmp_path, intern=None)                                              # .env-vera alone is not policy
    assert bt.resolve_sources(d)[0] == "missing"
    bt._read_env_files(d)
    assert bt.STATE["deny_policy"] == "missing" and bt.STATE["deny_all"] is True


@pytest.mark.parametrize("label,text", [
    ("garbage", "this is not a policy\n??? \n"),
    ("empty", ""),
    ("missing_required_key", "DB_HOST=db-sentinel.invalid\nDB_NAME=sentinel_db\nDB_ACCOUNT=sentinel_user\n"),
    ("empty_value", GOOD_POLICY.replace("DB_NAME=sentinel_db", "DB_NAME=")),
    ("bad_host", GOOD_POLICY.replace("db-sentinel.invalid", "bad host!")),
    ("bad_port_text", GOOD_POLICY + "DB_PORT=abc\n"),
    ("bad_port_range", GOOD_POLICY + "DB_PORT=70000\n"),
])
def test_policy_malformed_fails_closed(tmp_path, label, text, monkeypatch):
    _keep_policy_state(monkeypatch)
    d = _env_dir(tmp_path, intern=text)
    assert bt.resolve_sources(d)[0] == "malformed", label
    bt._read_env_files(d)
    assert bt.STATE["deny_all"] is True


def test_policy_valid_port_and_ip_host_accepted(tmp_path):
    d = _env_dir(tmp_path, intern=GOOD_POLICY.replace("db-sentinel.invalid", "10.1.2.3") + "DB_PORT=5432\n")
    assert bt.resolve_sources(d)[0] == "ok"


def test_policy_unreadable_fails_closed(tmp_path, monkeypatch):
    d = _env_dir(tmp_path)
    f = d / ".env-intern"
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda p, mode, **k: False if str(p).endswith(".env-intern") else real_access(p, mode, **k))
    assert bt.resolve_sources(d)[0] == "unreadable"
    monkeypatch.undo()
    if os.geteuid() != 0:
        f.chmod(0)
        try:
            assert bt.resolve_sources(d)[0] == "unreadable"
        finally:
            f.chmod(0o600)


def test_policy_group_writable_or_foreign_owner_fails_closed(tmp_path, monkeypatch):
    d = _env_dir(tmp_path, mode=0o664)
    assert bt.resolve_sources(d)[0] == "unreadable"
    for name in (".env-intern", ".env-vera"):
        (d / name).chmod(0o602)
    assert bt.resolve_sources(d)[0] == "unreadable"
    for name in (".env-intern", ".env-vera"):
        (d / name).chmod(0o600)
    assert bt.resolve_sources(d)[0] == "ok"
    (d / ".env-vera").chmod(0o620)                                  # an unsafe .env-vera also fails closed
    assert bt.resolve_sources(d)[0] == "unreadable"
    (d / ".env-vera").chmod(0o600)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(d / ".env-intern").st_uid + 1)
    assert bt.resolve_sources(d)[0] == "unreadable"


def test_policy_symlink_file_and_other_name_are_escape(tmp_path):
    d = _env_dir(tmp_path)
    f = d / ".env-intern"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".env-intern").write_text(GOOD_POLICY)
    f.unlink()
    f.symlink_to(outside / ".env-intern")
    assert bt.resolve_sources(d)[0] == "escape"
    f.unlink()
    (d / "real.txt").write_text(GOOD_POLICY)
    f.symlink_to(d / "real.txt")                             # inside the directory but another name
    assert bt.resolve_sources(d)[0] == "escape"


def test_policy_symlinked_env_dir_or_ancestor_is_escape(tmp_path):
    real = _env_dir(tmp_path / "real")
    (tmp_path / "link" / "config").mkdir(parents=True)
    (tmp_path / "link" / "config" / "env").symlink_to(real)
    assert bt.resolve_sources(tmp_path / "link" / "config" / "env")[0] == "escape"
    (tmp_path / "anc").symlink_to(tmp_path / "real")
    assert bt.resolve_sources(tmp_path / "anc" / "config" / "env")[0] == "escape"
    assert bt.resolve_sources(Path("config") / "env")[0] == "escape"          # relative: never cwd-resolved


def test_policy_ignores_cwd_decoy_env_supplied_path_and_legacy_files(tmp_path, monkeypatch, capsys):
    _keep_policy_state(monkeypatch)
    decoy = tmp_path / "cwd_decoy"
    _env_dir(decoy, intern=GOOD_POLICY.replace("db-sentinel", "decoy-sentinel"))
    (decoy / ".env").write_text("DB_HOST=decoy-sentinel.invalid\nDB_NAME=decoy\n")
    legacy = tmp_path / "AI-Internship" / "ai-engineering-bootcamp-v2" / "week-1v2"
    legacy.mkdir(parents=True)
    (legacy / ".env.db-accounts").write_text(GOOD_POLICY.replace("db-sentinel", "decoy-sentinel"))
    monkeypatch.chdir(decoy)
    monkeypatch.setenv("DB_SETTINGS_FILE", str(decoy / "config" / "env" / ".env-intern"))
    monkeypatch.setenv("DOTENV_PATH", str(decoy / ".env"))
    empty = tmp_path / "R" / "config" / "env"
    assert bt.resolve_sources(empty)[0] == "missing"
    found = bt._read_env_files(empty)
    assert found == [] and bt.STATE["deny_all"] is True
    fps, _ = bt._build_deny(found)
    assert bt.fingerprint("decoy-sentinel.invalid", 5432, "decoy") not in fps
    assert capsys.readouterr().out == "" and all("/" not in s and "=" not in s for s in bt.POLICY_STATUSES)


def test_policy_is_read_once(tmp_path, monkeypatch):
    _keep_policy_state(monkeypatch)
    d = _env_dir(tmp_path)
    reads, orig = [], Path.read_text

    def counting(self, *a, **k):
        if self.name in (".env-intern", ".env-vera"):
            reads.append(self.name)
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", counting)
    bt._read_env_files(d)
    assert sorted(reads) == [".env-intern", ".env-vera"]                 # each source read exactly once


def test_deny_sets_gain_suffix_pd_pr_and_pi(tmp_path):
    intern = GOOD_POLICY + "DB_HOST_EXTERNAL_SUFFIX=.ext-sentinel.invalid\n"
    vera = "VERA_DB_URL_WO=postgresql://wo_sentinel:sentinel_pw@10.9.8.7:5432/wo_sentinel_db\n"
    _status, found = bt.resolve_sources(_env_dir(tmp_path, intern=intern, vera=vera))
    s = bt._build_deny_sets(found)
    assert bt.fingerprint("db-sentinel.invalid.ext-sentinel.invalid", 5432, "sentinel_db") in s["fp"]   # host+suffix
    assert (len(".ext-sentinel.invalid"), bt._h(".ext-sentinel.invalid")) in s["suffix"]
    assert {bt._h("sentinel_db"), bt._h("wo_sentinel_db")} <= s["db"]                                    # PD
    assert {bt._h("sentinel_user"), bt._h("sentinel_dbadmin"), bt._h("wo_sentinel")} <= s["user"]        # PR
    assert bt._h("10.9.8.7") in s["ip"]                                                                  # PI literal
    assert bt._build_deny(found) == (s["fp"], s["user"])
    called = []
    bt._build_deny_sets([{"DB_HOST": "h.invalid", "DB_NAME": "x"}])                  # no resolver: no DNS lookup
    s2 = bt._build_deny_sets([{"DB_HOST": "h.invalid", "DB_NAME": "x"}],
                             resolver=lambda h: called.append(h) or ["192.0.2.44"])
    assert called == ["h.invalid"] and bt._h("192.0.2.44") in s2["ip"]


def test_install_builds_deny_sets_without_a_resolver_and_seals_them():
    for k in ("deny_fp", "deny_user_fp", "deny_db_fp", "deny_suffix", "deny_ip_fp"):
        assert isinstance(bt.STATE[k], frozenset), k
    src = Path(bt.__file__).read_text()
    body = src.split("def install() -> dict:", 1)[1].split("\ndef ", 1)[0]
    assert "_build_deny_sets(sources)" in body and "resolver" not in body
    assert body.index("_install_db_listener()") < body.index("install_connect_guard()") < body.index("_install_db_boundary()")


# ---- E2: the connect guard is installed in the "test" context, behind the boundary's own hooks
def test_connect_guard_installed_in_test_context_by_private_path():
    cg = sys.modules[bt._CG_PRIVATE]
    assert cg.installed_context() == "test"
    assert "safety.connect_guard" not in sys.modules or sys.modules["safety.connect_guard"] is not cg
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    assert event.contains(Engine, "do_connect", bt.STATE["db_listener"])
    eng = __import__("sqlalchemy").create_engine(SENT)                      # builds no connection
    order = list(eng.dialect.dispatch.do_connect)
    assert order.index(bt.STATE["db_listener"]) < order.index(cg._GLOBAL.listener)   # boundary listener runs first
    import psycopg2
    assert getattr(psycopg2.connect, "_vera_boundary", False)            # the boundary wrapper is outermost


@pytest.mark.boundary_probe
def test_boundary_refuses_and_records_before_the_connect_guard(no_sockets):
    from sqlalchemy import create_engine
    with pytest.raises(bt.DBBoundaryError):
        create_engine(SENT).connect()
    import psycopg2
    with pytest.raises(bt.DBBoundaryError):
        psycopg2.connect(SENT)
    assert no_sockets == [] and [v["kind"] for v in bt.STATE["violations"][-2:]] == ["db", "db"]


# ---- H6 regression: a loopback tunnel to a production dbname is refused even with every other live precondition
H6_TUNNEL = "postgresql://sentinel_tester:sentinel_pw@127.0.0.1:5433/sentinel_db"


def test_h6_loopback_tunnel_to_production_dbname_is_refused(tmp_path):
    _status, found = bt.resolve_sources(_env_dir(tmp_path, vera=None))
    s = bt._build_deny_sets(found)
    t = ("127.0.0.1", 5433, "sentinel_db", "sentinel_tester")
    kw = dict(live_item=True, live_enabled=True, deny_all=False, deny_fp=s["fp"], deny_user_fp=frozenset(),
              deny_db_fp=s["db"], deny_suffix=s["suffix"], deny_ip_fp=frozenset(),
              env={"VERA_TEST_DB_DISPOSABLE": "1", "VERA_TEST_DB_URL": H6_TUNNEL})
    assert bt.live_preconditions(t, **kw) == "dbname"                 # fp misses (other host/port); PD catches it
    assert bt.decide(t, **kw) is False
    kw["deny_user_fp"] = frozenset({bt._h("sentinel_tester")})
    kw["deny_db_fp"] = frozenset()
    assert bt.live_preconditions(t, **kw) == "role"                   # PR catches the production role too
    assert bt.decide(t, **kw) is False


def test_h6_tunnel_refused_through_check_in_a_live_item(monkeypatch, tmp_path):
    _status, found = bt.resolve_sources(_env_dir(tmp_path, vera=None))
    monkeypatch.setitem(bt.STATE, "deny_db_fp", bt._build_deny_sets(found)["db"])
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    monkeypatch.setitem(bt.STATE, "deny_all", False)
    monkeypatch.setitem(bt._SEALED, "deny_all", False)
    monkeypatch.setenv("VERA_TEST_DB_URL", H6_TUNNEL)
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0))])
    bt.begin_item(live_item=True)
    try:
        cp = {"host": "127.0.0.1", "port": 5433, "dbname": "sentinel_db", "user": "sentinel_tester"}
        with pytest.raises(bt.DBBoundaryError):
            bt.check((), cp)
        assert "hostaddr" not in cp
    finally:
        bt.begin_item(live_item=False)


# ---- C10: the runner's mask globs and the boundary's credential globs are one list
def test_c10_cred_glob_tuples_are_equal_and_cover_env_dash_files():
    spec = importlib.util.spec_from_file_location("_envd_runner_c10", REPO / "scripts" / "vera_test_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.CRED_MASK_GLOBS == bt.CRED_FILE_GLOBS
    assert runner.CRED_MASK_EXCLUDE == bt.CRED_FILE_EXCLUDE
    import fnmatch
    for name in (".env-vera", ".env-intern", "copy.env-vera"):
        assert any(fnmatch.fnmatchcase(name, g) for g in bt.CRED_FILE_GLOBS), name
    assert any(fnmatch.fnmatchcase(".env-vera.example", x) for x in bt.CRED_FILE_EXCLUDE)    # examples stay visible


def test_runner_rg1_refuses_env_files_and_config_env_in_checkout_or_course_service(tmp_path):
    spec = importlib.util.spec_from_file_location("_envd_runner_rg1", REPO / "scripts" / "vera_test_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    r = tmp_path / "R"
    repo = r / "vera"
    (repo / ".git").mkdir(parents=True)
    (repo / ".env.example").write_text("X=\n")
    (repo / ".env-vera.example").write_text("X=\n")
    assert runner.rg1_findings(str(repo)) == []                               # examples are fine
    (repo / ".git" / ".env-vera").write_text("X=\n")                          # root .git is skipped
    assert runner.rg1_findings(str(repo)) == []
    (repo / "sub").mkdir()
    (repo / "sub" / ".env-vera").write_text("X=\n")
    (repo / "config" / "env").mkdir(parents=True)
    (r / "course-service").mkdir()
    (r / "course-service" / ".env").write_text("X=\n")
    got = runner.rg1_findings(str(repo))
    assert len(got) == 3 and all(g.startswith("R-G1: ") for g in got)
    assert any("vera/sub/.env-vera" in g for g in got) and any("config/env" in g for g in got)
    assert any("course-service/.env" in g for g in got)
    assert not any(str(tmp_path) in g for g in got)                            # relative names only
    assert runner.rg1_findings(str(REPO)) == []                                # the real checkout is clean


# ================================================================================================ S-b revision: F2 children
CHILD_PRINT = ('import os, sys\nprint(sorted(n for n in os.environ if n.endswith(("_KEY", "_URL", "_PASSWORD", "PROXY"))))\n'
               'print("tests.vera_test_boundary" in sys.modules)\n')


def _py(code, env=None, cwd=None):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=cwd, timeout=60)


def test_child_inherits_no_guarded_names():
    r = _py(CHILD_PRINT)
    assert r.returncode == 0 and r.stdout.splitlines() == ["[]", "True"], r.stdout + r.stderr


def test_explicit_env_child_still_gets_boundary():
    env = {"PATH": os.environ.get("PATH", ""), "VERA_DB_URL_WO": SENT, "HTTPS_PROXY": "http://127.0.0.1:9", "EXTERNAL_DB_URL": SENT}
    r = _py(CHILD_PRINT, env=env)
    assert r.returncode == 0 and r.stdout.splitlines() == ["[]", "True"], r.stdout + r.stderr


@pytest.mark.boundary_probe
def test_python_child_gets_boundary_db_refused_before_libpq():
    srv = socket.create_server(("127.0.0.1", 0))
    srv.settimeout(0.5)
    port = srv.getsockname()[1]
    code = ('from sqlalchemy import create_engine\nbt = __import__("sys").modules["tests.vera_test_boundary"]\n'
            'try:\n    create_engine("postgresql://u:p@127.0.0.1:%d/d").connect()\nexcept bt.DBBoundaryError:\n    print("refused")\n' % port)
    r = _py(code)
    try:
        with pytest.raises(socket.timeout):
            srv.accept()                                           # nothing ever connected to the listener
    finally:
        srv.close()
    assert r.stdout.strip() == "refused" and r.returncode == 97, r.stdout + r.stderr   # M5: swallowed refusal -> 97


@pytest.mark.boundary_probe
def test_python_child_network_refused():
    code = ('import socket\nbt = __import__("sys").modules["tests.vera_test_boundary"]\ntry:\n'
            '    socket.create_connection(("192.0.2.1", 443), timeout=1)\nexcept bt.NetworkBoundaryError:\n    print("refused")\n')
    r = _py(code)
    assert r.stdout.strip() == "refused" and r.returncode == 97, r.stdout + r.stderr


def test_python_child_dotenv_is_noop(tmp_path):
    (tmp_path / ".env").write_text("GUARD_DOTENV_SENTINEL=1\nVERA_DB_URL_WO=%s\n" % SENT)
    code = 'import os\nfrom dotenv import load_dotenv\nprint(load_dotenv())\nprint("GUARD_DOTENV_SENTINEL" in os.environ, "VERA_DB_URL_WO" in os.environ)\n'
    r = _py(code, cwd=tmp_path)
    assert r.stdout.split() == ["False", "False", "False"], r.stdout + r.stderr


@pytest.mark.boundary_probe
def test_isolated_python_child_with_unpinned_program_refused():
    before = len(bt.STATE["violations"])
    for argv, env in (([sys.executable, "-I", "-c", "pass"], None), ([sys.executable, "-I", "-c", "print(1)"], {"PATH": ""}),
                      ([sys.executable, "-E", "-c", "pass"], None)):
        with pytest.raises(bt.ChildProcessBoundaryError):
            subprocess.run(argv, env=env, capture_output=True)
    assert len(bt.STATE["violations"]) == before + 3 and all(v["kind"] == "child" for v in bt.STATE["violations"][before:])


@pytest.mark.boundary_probe
def test_dash_S_and_combined_flags_refused():
    for flags in (["-S"], ["-IS"], ["-sS"], ["-E"], ["-Ic"]):
        with pytest.raises(bt.ChildProcessBoundaryError):
            subprocess.run([sys.executable, *flags, "pass"], capture_output=True)
    for env in ({"PATH": "/usr/bin", "PYTHONNOUSERSITE": "1"}, {"PATH": "/usr/bin", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}):
        with pytest.raises(bt.ChildProcessBoundaryError):
            subprocess.run([sys.executable, "-c", "pass"], env=env, capture_output=True)
    r = subprocess.run([sys.executable, "-u", "-c", "print('ok')"], capture_output=True, text=True)   # -u is fine
    assert r.stdout.strip() == "ok"


@pytest.mark.boundary_probe
def test_shell_and_non_python_children_refused():
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run(["/bin/sh", "-c", "true"], capture_output=True)
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run("true", shell=True, capture_output=True)
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run(["echo", "x"], capture_output=True)
    with pytest.raises(bt.ChildProcessBoundaryError):
        os.system("true")
    with pytest.raises(bt.ChildProcessBoundaryError):
        os.execv("/bin/true", ["/bin/true"])
    with pytest.raises(bt.ChildProcessBoundaryError):
        os.spawnv(os.P_WAIT, "/bin/true", ["/bin/true"])
    assert len([v for v in bt.STATE["violations"] if v["kind"] == "child"]) == 6


def test_pinned_answer_contract_child_runs_with_stripped_env():
    code = (f"import sys\nsys.path.insert(0, {str(REPO)!r})\nsys.modules['vera.discovery']=None\nsys.modules['vera.discovery_leak_check']=None\n"
            "import vera.answer_contract as a\nassert a.layers()[0]=='L0'\nprint('ok')")
    assert bt._sha(code.replace(repr(str(REPO)), "<ROOT>").encode()) == bt.ANSWER_CONTRACT_CHILD_SHA
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, env={"PATH": ""}, cwd=REPO)
    assert r.stdout.strip() == "ok", r.stderr


@pytest.mark.boundary_probe
def test_pinned_program_needs_the_exact_env_and_text():
    code = (f"import sys\nsys.path.insert(0, {str(REPO)!r})\nsys.modules['vera.discovery']=None\nsys.modules['vera.discovery_leak_check']=None\n"
            "import vera.answer_contract as a\nassert a.layers()[0]=='L0'\nprint('ok')")
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, env={"PATH": "/usr/bin"})
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True)
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run([sys.executable, "-I", "-B", "-c", code + "\nprint('extra')"], capture_output=True, env={"PATH": ""})


def test_allowlisted_start_sh_child_runs():
    r = subprocess.run([str(REPO / "start.sh"), "nope"], capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr


@pytest.mark.boundary_probe
def test_start_sh_hash_change_refused(tmp_path, monkeypatch):
    fake = tmp_path / "repo"
    fake.mkdir()
    (fake / "start.sh").write_text("#!/bin/sh\necho changed\n")
    (fake / "start.sh").chmod(0o755)
    monkeypatch.setattr(bt, "REPO", fake)
    with pytest.raises(bt.ChildProcessBoundaryError, match="allowlisted child changed"):
        subprocess.run([str(fake / "start.sh"), "nope"], capture_output=True)
    assert bt.STATE["violations"][-1]["kind"] == "child"


def test_read_only_git_children_are_allowlisted_and_nothing_else_git():
    assert bt._match_allowed(["git", "rev-parse", "HEAD"], False, {}) == "ok"
    assert bt._match_allowed(["git", "check-ignore", "-q", ".env.db-accounts"], False, {}) == "ok"
    for bad in (["git", "check-ignore", "-q", "../x"], ["git", "check-ignore", "-q", "/etc/x"], ["git", "push"],
                ["git", "status", "--porcelain", "--", "vera", "config", "x"], ["git", "config", "--global", "x", "y"]):
        assert bt._match_allowed(bad, False, {}) is None, bad


def test_exec_and_spawn_entry_points_wrapped():
    import multiprocessing.util as mpu
    assert getattr(subprocess.Popen.__init__, "_vera_boundary", False)
    for name, _a, _e in bt.SPAWN_SITES:
        if hasattr(os, name):
            assert getattr(getattr(os, name), "_vera_boundary", False), name
    assert getattr(mpu.spawnv_passfds, "_vera_boundary", False)


def _mp_probe(conn):
    import sys as _sys
    from sqlalchemy import create_engine
    bt_child = _sys.modules["tests.vera_test_boundary"]
    try:
        create_engine(SENT).connect()
        conn.send("connected")
    except bt_child.DBBoundaryError:
        conn.send("refused")


@pytest.mark.boundary_probe
def test_multiprocessing_spawn_child_has_boundary():
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    a, b = ctx.Pipe()
    p = ctx.Process(target=_mp_probe, args=(b,))
    p.start()
    assert a.poll(60) and a.recv() == "refused"
    p.join(30)


EXPECTED_SPAWN_SITES = {
    "test_item73_deploy.py": {"subprocess.run"}, "test_eval_config.py": {"subprocess.run"},
    "test_answer_contract.py": {"subprocess.run"}, "test_live_optin.py": {"subprocess.run"},
    "test_run_with_env.py": {"subprocess.run"}, "test_item72_scope.py": {"subprocess.run"},
    # K18W review: object constructions inside fakes (D-B5-1 binding tests), not process creation
    "test_release_gate_evidence_binding.py": {"subprocess.CompletedProcess", "subprocess.TimeoutExpired"},
}


def test_every_test_spawn_site_is_covered():
    """A new spawn site in tests/ fails here until it is reviewed against ALLOWED_CHILDREN / the inherit rule."""
    found = {}
    pat = re.compile(r"^(subprocess\.\w+|Popen|os\.system|os\.exec\w*|os\.spawn\w*|os\.posix_spawn\w*|multiprocessing\.\w+)$")
    for f in sorted((REPO / "tests").rglob("*.py")):
        if f.name in ("test_env_guard.py", "vera_test_boundary.py"):
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and pat.match(ast.unparse(node.func)):
                found.setdefault(f.name, set()).add(ast.unparse(node.func))
    assert found == EXPECTED_SPAWN_SITES


# ------------------------------------------------------------------------------------------------ M4 / M5
def test_bootstrap_dir_is_under_gitdir_not_system_tmp():
    d = Path(bt.STATE["bootstrap_dir"])
    gd = bt._gitdir(REPO)
    assert d.is_dir() and bt._under(d, gd / "vera-test-boundary") and (d / "sitecustomize.py").is_file()
    assert oct(d.stat().st_mode & 0o777) == "0o700" and oct((d / "violations.log").stat().st_mode & 0o777) == "0o600"
    tmp = os.path.realpath(tempfile.gettempdir())
    assert not os.path.realpath(d).startswith(tmp + os.sep) or os.path.realpath(d).startswith(os.path.realpath(gd))
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(d) and os.environ[bt.MARKER] == "1"


def test_child_bootstrap_reused_not_recreated():
    base = Path(bt.STATE["bootstrap_dir"]).parent
    before = sorted(p.name for p in base.iterdir())
    r = _py('import os, sys\nbt = sys.modules["tests.vera_test_boundary"]\nprint(bt.STATE["bootstrap_dir"], bt.STATE["is_child"])\n')
    assert r.stdout.split() == [str(bt.STATE["bootstrap_dir"]), "True"], r.stdout + r.stderr
    assert sorted(p.name for p in base.iterdir()) == before


@pytest.mark.boundary_probe
def test_child_swallowed_refusal_exits_97_and_counts_in_parent():
    code = ('from sqlalchemy import create_engine\ntry:\n    create_engine("%s").connect()\nexcept Exception:\n    pass\n' % SENT)
    r = _py(code)
    assert r.returncode == 97 and "sentinel_pw" not in r.stdout + r.stderr
    bt.set_phase("call")
    msg = bt.violations_message("call")
    assert msg is not None and "1 child-process" in msg


def test_child_swallowed_refusal_fails_parent_item(tmp_path):
    child = ('import subprocess, sys\ndef test_runs_a_swallowing_child():\n'
             '    code = "from sqlalchemy import create_engine\\ntry:\\n    create_engine(\\"%s\\").connect()\\nexcept Exception:\\n    pass\\n"\n'
             '    subprocess.run([sys.executable, "-c", code])\n' % SENT)
    r = run_child(tmp_path, {"test_c.py": child})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "test boundary violation" in out and "1 child-process" in out, out
    no_secrets(out)


# ------------------------------------------------------------------------------------------------ M6 / M7 / M8
def _live_state(monkeypatch):
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    monkeypatch.setitem(bt.STATE, "deny_all", False)
    monkeypatch.setitem(bt._SEALED, "deny_all", False)  # K18W M-2: loosening must be explicit on the sealed value too
    bt.begin_item(live_item=True)
    monkeypatch.setenv("VERA_TEST_DB_URL", SENT_TEST)
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")


CP = {"host": "db-sentinel-test.invalid", "port": 5432, "dbname": "sentinel_test_db", "user": "sentinel_tester"}
T_CP = ("db-sentinel-test.invalid", 5432, "sentinel_test_db", "sentinel_tester")


@pytest.mark.parametrize("addr,ok", [("127.0.0.1", True), ("10.1.2.3", True), ("172.16.5.5", True), ("192.168.1.1", True),
                                     ("100.64.0.9", True), ("::1", True), ("fd00::1", True), ("192.0.2.10", False),
                                     ("8.8.8.8", False), ("172.32.0.1", False), ("2001:db8::1", False)])
def test_allowed_target_must_resolve_to_private_or_loopback(monkeypatch, addr, ok):
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", (addr, 0))])
    cp = dict(CP)
    if ok:
        bt.pin_allowed_target(T_CP, cp)
        assert cp["hostaddr"] == addr
    else:
        with pytest.raises(bt.DBBoundaryError):
            bt.pin_allowed_target(T_CP, cp)
        assert "hostaddr" not in cp
    _live_state(monkeypatch)                                 # and check() itself refuses before any pinning
    cp = dict(CP)
    with pytest.raises(bt.DBBoundaryError):
        bt.check((), cp)
    assert "hostaddr" not in cp
    bt.begin_item(live_item=False)


def test_allowed_target_with_one_public_address_or_unresolvable_is_refused(monkeypatch):
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0)), (2, 1, 6, "", ("192.0.2.10", 0))])
    with pytest.raises(bt.DBBoundaryError):
        bt.pin_allowed_target(T_CP, dict(CP))

    def nxdomain(host, port=None):
        raise socket.gaierror(socket.EAI_NONAME, "x")

    monkeypatch.setattr(bt, "_orig_getaddrinfo", nxdomain)
    with pytest.raises(bt.DBBoundaryError):
        bt.pin_allowed_target(T_CP, dict(CP))
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [])
    with pytest.raises(bt.DBBoundaryError):
        bt.pin_allowed_target(T_CP, dict(CP))
    bt.begin_item(live_item=False)                           # pin refusals record "db"; clear the recorder


def test_allowed_target_ignores_pgpass_and_service(monkeypatch):
    for n in ("PGPASSFILE", "PGSERVICEFILE", "PGSYSCONFDIR", "PGSSLROOTCERT", "PGSSLCERT", "PGSSLKEY", "PGOPTIONS"):
        assert bt.guarded(n), n
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0))])
    cp = dict(CP, service="sentinel-svc", passfile="/home/x/.pgpass")
    bt.pin_allowed_target(T_CP, cp)
    assert cp["passfile"] == os.devnull and "service" not in cp and cp["hostaddr"] == "127.0.0.1"


@pytest.mark.boundary_probe
def test_udp_sendto_non_loopback_refused():
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(bt.NetworkBoundaryError):
            u.sendto(b"x", ("192.0.2.1", 9))
        with pytest.raises(bt.NetworkBoundaryError):
            u.sendto(b"x", 0, ("192.0.2.1", 9))
        with pytest.raises(bt.NetworkBoundaryError):
            u.sendmsg([b"x"], [], 0, ("192.0.2.1", 9))
    finally:
        u.close()
    assert [v["kind"] for v in bt.STATE["violations"]] == ["net", "net", "net"]


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_udp_loopback_still_works():
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(2)
    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.sendto(b"ping", srv.getsockname())
    assert srv.recvfrom(10)[0] == b"ping"
    cli.close(); srv.close()
    assert bt.STATE["violations"] == []


# ================================================================================================ 14: OS-level isolation
FORCE_NOT_ISOLATED = ('import sys\n_bt = sys.modules["tests.vera_test_boundary"]\n'
                      '_bt.netns_status = lambda *a, **k: "not_isolated"\n')


def test_session_runs_in_isolated_netns():
    assert bt.netns_status() == "isolated" and bt.STATE["netns"] == "isolated" and bt.STATE["unix_check"] == "clean"


def test_raw_socket_libpq_and_ctypes_cannot_reach_the_network():
    with bt.kernel_layer_probe():   # INT2: proves the KERNEL layer; the audit hook is stepped aside
        import _socket
        import ctypes
        import errno
        import psycopg2
        raw_connect = psycopg2.connect          # E2: boundary wrapper -> connect guard wrapper -> the real driver
        while getattr(raw_connect, "_vera_original", None) is not None:
            raw_connect = raw_connect._vera_original
        s4 = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError) as e1:
                s4.connect(("192.0.2.1", 9))                       # raw base-class socket: no Python wrapper in the way
            assert e1.value.errno == errno.ENETUNREACH
        finally:
            s4.close()
        u = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        try:
            with pytest.raises(OSError) as e2:
                u.sendto(b"x", ("192.0.2.1", 9))
            assert e2.value.errno == errno.ENETUNREACH
        finally:
            u.close()
        s6 = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError) as e3:
                s6.connect(("2001:db8::1", 9))
            assert e3.value.errno in (errno.ENETUNREACH, errno.EADDRNOTAVAIL)
        finally:
            s6.close()
        with pytest.raises(psycopg2.OperationalError):             # libpq, below the Python wrappers
            raw_connect(host="192.0.2.1", port=9, connect_timeout=2, dbname="d", user="u")
        libc = ctypes.CDLL(None, use_errno=True)                    # a ctypes path to libc, TEST-NET only
        fd = libc.socket(2, 1, 0)
        assert fd >= 0
        try:
            sa = (ctypes.c_ubyte * 16)(2, 0, 0, 9, 192, 0, 2, 1)
            assert libc.connect(fd, sa, 16) == -1 and ctypes.get_errno() == errno.ENETUNREACH
        finally:
            libc.close(fd)


NS_DEV = "Inter-|   Receive\n face |bytes\n    lo: 0 0\n"
NS_DEV_HOST = NS_DEV + "  eth0: 1 1\n"
NS_ROUTE = "Iface\tDestination\n"
NS_ROUTE_HOST = NS_ROUTE + "eth0\t00000000\t0102A8C0\n"
NS_R6 = "00000000000000000000000000000001 80 0 0 0 0 0 0 80200001       lo\n"


def _reader(dev=NS_DEV, route=NS_ROUTE, r6=NS_R6):
    files = {"dev": dev, "route": route, "ipv6_route": r6}
    return lambda name: files[name]


def test_netns_status_pure_cases():
    ok = lambda: True       # noqa: E731
    assert bt.netns_status(_reader(), ok) == "isolated"
    assert bt.netns_status(_reader(dev=NS_DEV_HOST), ok) == "not_isolated"
    assert bt.netns_status(_reader(route=NS_ROUTE_HOST), ok) == "not_isolated"
    assert bt.netns_status(_reader(r6=NS_R6 + "fe80 80 0 0 0 0 0 0 80200001       eth0\n"), ok) == "not_isolated"
    assert bt.netns_status(_reader(), lambda: False) == "not_isolated"          # the UDP confirmation failed

    def unreadable(name):
        raise FileNotFoundError(name)

    assert bt.netns_status(unreadable, ok) == "unknown"


def test_reexec_command_shape():
    argv = bt.reexec_argv(["/venv/bin/python", "-m", "pytest", "tests", "-q"], 1001, 1002, "someone")
    files, dirs = bt.escape_lists(1001, "someone")
    assert argv == ["sh", "-c", bt.OUTER, "sh", "-rnm", "sh", "-c", bt.SCRIPT, "sh", "1001", "1002", *files, "--dirs", *dirs,
                    "--cmd", "/venv/bin/python", "-m", "pytest", "tests", "-q"]
    assert "{" not in bt.SCRIPT.replace("fail() {", "").replace("; }", "") and "pytest" not in bt.SCRIPT
    assert all(p not in bt.SCRIPT for p in (*files, *dirs))                      # data is positional, never interpolated


class _FakeCfg:
    class pluginmanager:
        @staticmethod
        def getplugin(name):
            return None


def _fresh_decision(monkeypatch, status, marker=False, live=False):
    monkeypatch.setitem(bt.STATE, "netns_decided", False)
    monkeypatch.setitem(bt.STATE, "netns", "undecided")
    monkeypatch.setitem(bt.STATE, "config", bt.STATE["config"])
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: status)
    env = {"PATH": os.environ.get("PATH", ""), "OUTER_SENTINEL_KEY": "GUARD-SENTINEL", "WSL_INTEROP": "/run/WSL/1_interop",
           bt.RUNNER_MARKER: "1"}                 # INT2: maybe_reexec now requires a sanctioned launch first
    if marker:
        env[bt.REEXEC_MARKER] = "1"
    monkeypatch.setattr(bt, "_ORIG_ENV", env)


class _Stop(Exception):
    pass


def test_reexec_loop_guard_fails_closed(monkeypatch):
    _fresh_decision(monkeypatch, "not_isolated", marker=True)
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: (_ for _ in ()).throw(SystemExit(text)))
    with pytest.raises(SystemExit) as ei:
        bt.maybe_reexec(_FakeCfg, False)
    # INT3 MED-A: a not-isolated session is refused (no re-exec) whatever the markers say
    assert ei.value.code == bt.ISOLATION_REQUIRED_MESSAGE and calls == [] and bt.EXIT_ISOLATION == 78


def test_reexec_uses_original_env_and_unwrapped_exec(monkeypatch):
    """INT3 MED-A: a sanctioned (runner-marked) but NOT isolated session is refused and never execs (it used to
    re-exec into the in-repo chain). The chain's env helper still scrubs guarded and WSL names (library code)."""
    _fresh_decision(monkeypatch, "not_isolated")
    monkeypatch.setitem(bt.STATE, "bootstrap_dir", None)          # never remove the real session's bootstrap
    seen = {}

    def spy(exe, argv, env):
        seen.update(exe=exe, argv=argv, env=env)
        raise _Stop

    monkeypatch.setattr(os, "execve", spy)                         # no _vera_original attribute: used as-is
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: (_ for _ in ()).throw(SystemExit(text)))
    before = len(bt.STATE["violations"])
    with pytest.raises(SystemExit) as ei:
        bt.maybe_reexec(_FakeCfg, False)
    assert ei.value.code == bt.ISOLATION_REQUIRED_MESSAGE and seen == {}
    env = bt.reexec_env(bt._ORIG_ENV)
    # K18W: guarded names (OUTER_SENTINEL_KEY matches the *_KEY pattern) never cross into the namespace either
    expected = {k: v for k, v in bt._ORIG_ENV.items() if not k.startswith("WSL") and k != "OUTER_SENTINEL_KEY"}
    assert env == expected and "PYTHONPATH" not in env
    assert "OUTER_SENTINEL_KEY" in bt._ORIG_ENV and "OUTER_SENTINEL_KEY" not in env
    assert "WSL_INTEROP" in bt._ORIG_ENV and "WSL_INTEROP" not in env            # X1: interop never survives
    assert len(bt.STATE["violations"]) == before                  # the spawn guard was not consulted


def test_live_session_branch_sets_live_state_without_exec(monkeypatch):
    _fresh_decision(monkeypatch, "not_isolated")
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("a live session must not re-exec"))
    bt.maybe_reexec(_FakeCfg, True)
    assert bt.STATE["netns"] == "live-session" and bt.STATE["unix_check"] == "skipped-live"


def _child_python(code, env=None):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)


def test_isolation_unavailable_exit_code_is_78():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "not_isolated"\n'
            'bt._ORIG_ENV["VERA_TEST_NETNS_REEXEC"] = "1"\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, False)\nprint("not reached")\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.ISOLATION_REQUIRED_MESSAGE and "not reached" not in r.stdout


def test_missing_unshare_binary_exits_78():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "not_isolated"\n'
            'bt._ORIG_ENV.pop("VERA_TEST_NETNS_REEXEC", None)\nbt._ORIG_ENV["PATH"] = "/nonexistent"\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.STATE["bootstrap_dir"] = None\nbt.maybe_reexec(C, False)\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.ISOLATION_REQUIRED_MESSAGE       # INT3: refused before any exec


def test_script_from_stdin_cannot_be_reexecd():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "not_isolated"\n'
            'bt._ORIG_ENV.pop("VERA_TEST_NETNS_REEXEC", None)\nsys.orig_argv = ["python", "-"]\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, False)\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.ISOLATION_REQUIRED_MESSAGE       # INT3: refused before any exec


def test_reexec_output_survives_capture(tmp_path):
    pre = FORCE_NOT_ISOLATED + (
        'import os\n_orig_execve = os.execve._vera_original\n'
        'def _fake(exe, argv, env):\n'
        '    _orig_execve(sys.executable, [sys.executable, "-c", "import sys; print(\'X\'); sys.exit(5)"], env)\n'
        'os.execve = _fake\n')
    r = run_child(tmp_path, {"test_a.py": "def test_a():\n    pass\n"}, pre=pre)
    # INT3 MED-A: a not-isolated child session is refused (78, message on stderr) and the exec is never reached
    assert r.returncode == 78 and "X" not in r.stdout.split() and bt.ISOLATION_REQUIRED_MESSAGE in r.stderr, \
        r.stdout + r.stderr


LIVE_FILES = {
    "test_live.py": ('import pytest\n@pytest.mark.live\ndef test_live_one():\n    pass\n'
                     'def test_plain_in_a_live_module():\n    pass\n'),
    "test_plain.py": ('import pathlib\npathlib.Path(__file__).with_name("MODULE_IMPORTED").write_text("x")\n'
                      'def test_plain():\n    pass\n'),
}


def test_live_session_never_imports_non_live_modules(tmp_path):
    r = run_child(tmp_path, dict(LIVE_FILES), args=("--run-live-stages", "-rA"), pre=FORCE_NOT_ISOLATED,
                  env={"GROQ_API_KEY": "GUARD-SENTINEL-KEY"})
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "1 passed" in out and "1 deselected" in out, out
    assert not (tmp_path / "MODULE_IMPORTED").exists()                 # the non-live module was never imported
    assert "live session: 1 non-live items deselected" in out and "test_plain_in_a_live_module" not in out.split("PASSED")[-1]


def test_live_predicate_matches_pytest_option(tmp_path):
    r = run_child(tmp_path, dict(LIVE_FILES), pre=FORCE_NOT_ISOLATED, env={"PYTEST_ADDOPTS": "--run-live-stages"})
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "1 passed" in out and not (tmp_path / "MODULE_IMPORTED").exists(), out


def test_children_inherit_netns():
    r = _py('print(sorted(l.split(":")[0].strip() for l in open("/proc/self/net/dev").read().splitlines()[2:]))')
    assert r.stdout.strip() == "['lo']", r.stdout + r.stderr


def test_uid_mapping_preserves_dac(tmp_path):
    assert os.getuid() != 0
    f = tmp_path / "secret"
    f.write_text("x")
    f.chmod(0)
    try:
        with pytest.raises(PermissionError):
            f.read_text()
    finally:
        f.chmod(0o600)


def test_no_override_variable_exists():
    src = (REPO / "tests" / "vera_test_boundary.py").read_text()
    assert set(re.findall(r"VERA_TEST_NETNS_\w+", src)) == {"VERA_TEST_NETNS_REEXEC"}
    for word in ("SKIP_ISOLATION", "NO_ISOLATION", "ALLOW_NETWORK", "ISOLATION_OFF"):
        assert word not in src


# ------------------------------------------------------------------------------------------------ AF_UNIX allowlist, escape sockets
def _unix_server(path):
    """Bind in the cwd (tests chdir to a short directory) so the sun_path limit of 107 bytes never bites."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(os.path.basename(path))
    srv.listen(1)
    return srv


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_unix_connect_inside_basetemp_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    srv = _unix_server("u.sock")
    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect("u.sock")
        cli.close()
    finally:
        srv.close()
    assert bt.STATE["violations"] == []


@pytest.mark.boundary_probe
@pytest.mark.vera_sockets      # K18W: requested, so the refusal below is the basetemp-roots rule, not the default deny
def test_unix_connect_outside_roots_refused_and_recorded(monkeypatch):
    d = Path(bt.STATE["bootstrap_dir"]).parent / "sentinel-unix-test"
    d.mkdir(exist_ok=True)
    sock = d / "s.sock"
    monkeypatch.chdir(d)
    srv = _unix_server("s.sock")
    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with pytest.raises(bt.NetworkBoundaryError):
            cli.connect("s.sock")
        with pytest.raises(bt.NetworkBoundaryError):
            cli.connect_ex("s.sock")
        cli.close()
    finally:
        srv.close()
        sock.unlink()
        d.rmdir()
    assert [v["kind"] for v in bt.STATE["violations"]] == ["net", "net"]


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_unix_abstract_address_allowed():
    name = b"\0vera-test-abstract-%d" % os.getpid()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(name)
    srv.listen(1)
    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect(name)
        cli.close()
    finally:
        srv.close()
    assert bt.STATE["violations"] == []


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_multiprocessing_tempdir_unix_allowed(monkeypatch):
    import multiprocessing.process as mpp
    d = Path(bt.STATE["bootstrap_dir"]).parent / "sentinel-mp-test"
    d.mkdir(exist_ok=True)
    sock = d / "m.sock"
    monkeypatch.chdir(d)
    srv = _unix_server("m.sock")
    monkeypatch.setitem(mpp.current_process()._config, "tempdir", str(d))
    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect("m.sock")
        cli.close()
    finally:
        srv.close()
        sock.unlink()
        d.rmdir()
    assert bt.STATE["violations"] == []


def test_escape_check_flags_writable_socket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sock = tmp_path / "docker.sock"
    srv = _unix_server("docker.sock")
    try:
        assert bt.escape_capable([str(sock)], []) is True
        (tmp_path / "rundir").mkdir()
        monkeypatch.chdir(tmp_path / "rundir")
        s2 = _unix_server("bus")
        try:
            assert bt.escape_capable([], [str(tmp_path / "rundir")]) is True
        finally:
            s2.close()
    finally:
        srv.close()


def test_escape_check_ignores_unwritable_socket_absent_and_non_socket(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sock = tmp_path / "locked.sock"
    srv = _unix_server("locked.sock")
    try:
        sock.chmod(0o500)
        assert bt.escape_capable([str(sock)], []) is False
    finally:
        srv.close()
    (tmp_path / "plain.sock").write_text("not a socket")
    assert bt.escape_capable([str(tmp_path / "plain.sock"), str(tmp_path / "absent.sock")], [str(tmp_path / "nodir")]) is False


def test_escape_check_fails_closed_with_78_and_no_path_in_message(tmp_path):
    sock = tmp_path / "docker.sock"
    code = ('import socket, sys\nbt = sys.modules["tests.vera_test_boundary"]\n'
            'srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\nsrv.bind("docker.sock")\n'
            'bt.netns_status = lambda *a, **k: "isolated"\nbt.escape_lists = lambda uid, user: ([%r], [])\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, False)\nprint("not reached")\n' % (str(sock),))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert r.returncode == 78 and r.stderr.strip() == bt.ESCAPE_MESSAGE and str(sock) not in r.stderr + r.stdout


def test_reexec_script_masks_known_paths():
    files, dirs = bt.escape_lists(1001, "someone")
    argv = bt.reexec_argv(["python"], 1001, 1002, "someone")
    for entry in (*files, *dirs):
        assert entry in argv and entry not in bt.SCRIPT
    assert {"/var/run/docker.sock", "/run/podman/podman.sock", "/run/user/1001", "/tmp/tmux-1001"} <= set(argv)
    assert "mount --bind /dev/null" in bt.SCRIPT and "tmpfs" in bt.SCRIPT and "|| fail" in bt.SCRIPT


def test_masked_runtime_dir_is_empty_inside_namespace():
    d = Path(f"/run/user/{os.getuid()}")
    assert (not d.exists()) or list(d.iterdir()) == [] or not bt.escape_capable([], [str(d)])


# ------------------------------------------------------------------------------------------------ release gate: isolation evidence
def _gate():
    spec = importlib.util.spec_from_file_location("release_gate_n", REPO / "scripts" / "release_gate.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def _fake_run(returncode, stdout_fn, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.update(kw)
        out = stdout_fn(kw["env"]["VERA_GATE_NONCE"])
        return subprocess.CompletedProcess(cmd, returncode, stdout=out, stderr="")
    return run


@pytest.fixture
def gate_repo(tmp_path):
    """A throwaway root for g.check_tests: it makes <root>/tmp/release-gate, which must never appear in the checkout."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    return tmp_path


def test_gate_strips_pytest_addopts_and_plugins_and_passes_nonce(monkeypatch, gate_repo):
    g = _gate()
    seen = {}
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setenv("PYTEST_PLUGINS", "evil")
    monkeypatch.setattr(g.subprocess, "run", _fake_run(
        0, lambda n: f"1 passed\nVERA-ISOLATION-EVIDENCE netns=isolated nonce={n}\n", seen))
    assert g.check_tests(gate_repo) == []
    env = seen["env"]
    assert "PYTEST_ADDOPTS" not in env and "PYTEST_PLUGINS" not in env and re.fullmatch(r"[0-9a-f]{32}", env["VERA_GATE_NONCE"])
    assert env["PATH"] == "/usr/bin:/bin" and seen["cwd"] == gate_repo                                     # S-2
    monkeypatch.setenv("PYTEST_ADDOPTS", "--noconftest")                       # S-1: refused before anything is spawned
    seen.clear()
    out = g.check_tests(gate_repo)
    assert out and "PYTEST_ADDOPTS" in out[0] and seen == {}


def test_gate_fails_without_isolation_line(monkeypatch, gate_repo):
    g = _gate()
    monkeypatch.setattr(g.subprocess, "run", _fake_run(0, lambda n: "1 passed\n"))
    out = g.check_tests(gate_repo)
    assert out and "did not prove OS-level network isolation" in out[0]


def test_gate_fails_on_wrong_or_duplicated_nonce_line(monkeypatch, gate_repo):
    g = _gate()
    monkeypatch.setattr(g.subprocess, "run", _fake_run(0, lambda n: f"VERA-ISOLATION-EVIDENCE netns=isolated nonce={'0' * 32}\n"))
    assert "did not prove" in g.check_tests(gate_repo)[0]
    monkeypatch.setattr(g.subprocess, "run", _fake_run(
        0, lambda n: (f"VERA-ISOLATION-EVIDENCE netns=isolated nonce={n}\n") * 2))
    assert "did not prove" in g.check_tests(gate_repo)[0]
    monkeypatch.setattr(g.subprocess, "run", _fake_run(0, lambda n: f"VERA-ISOLATION-EVIDENCE netns=live-session nonce={n}\n"))
    assert "did not prove" in g.check_tests(gate_repo)[0]


def test_gate_passes_with_exact_line_and_rc0(monkeypatch, gate_repo):
    g = _gate()
    monkeypatch.setattr(g.subprocess, "run", _fake_run(
        0, lambda n: f"1 passed\n  VERA-ISOLATION-EVIDENCE netns=isolated nonce={n}  \n"))
    assert g.check_tests(gate_repo) == []


def test_gate_reports_exit_78_as_isolation_not_established(monkeypatch, gate_repo):
    g = _gate()
    monkeypatch.setattr(g.subprocess, "run", _fake_run(78, lambda n: ""))
    out = g.check_tests(gate_repo)
    assert out == ["test isolation was not established for this run (exit 78: the runner or the test boundary refused)"]
    assert "unavailable on this host" not in out[0]                     # GATEFIX-1029: no misleading host claim
    def refused(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 64, stdout="", stderr="vera_test_runner: refused: x\x1b[2J\n")
    monkeypatch.setattr(g.subprocess, "run", refused)
    assert g.check_tests(gate_repo) == ["test run refused by the sanctioned runner (exit 64)",
                                        "vera_test_runner: refused: x?[2J"]                 # G-8: printable echo
    monkeypatch.setattr(g.subprocess, "run", _fake_run(1, lambda n: f"1 failed\nVERA-ISOLATION-EVIDENCE netns=isolated nonce={n}\n"))
    assert g.check_tests(gate_repo)[0].startswith("pytest exit 1")


def test_gate_launches_suite_through_sanctioned_runner(monkeypatch, gate_repo):
    """GATEFIX-1029: check_tests runs `<python> -I -B <root>/scripts/vera_test_runner.py -- -q tests --ignore=...`
    (never a bare `python -m pytest`, which the boundary refuses with 78 outside an isolated session), with the
    allowlisted child env, the per-run TMPDIR, cwd=root and the suite timeout. No process is spawned (faked run)."""
    g = _gate()
    calls = []

    def run(cmd, **kw):
        calls.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"VERA-ISOLATION-EVIDENCE netns=isolated nonce="
                                           f"{kw['env']['VERA_GATE_NONCE']}\n", stderr="")
    monkeypatch.setattr(g.subprocess, "run", run)
    assert g.check_tests(gate_repo) == [] and len(calls) == 1
    cmd, kw = calls[0]
    assert cmd == [sys.executable, "-I", "-B", str(gate_repo / "scripts" / "vera_test_runner.py"), "--",
                   "-q", "tests", "--ignore=tests/test_m2_integration.py"]
    assert "-m" not in cmd and "pytest" not in cmd                       # no direct pytest launch
    env = kw["env"]
    assert kw["cwd"] == gate_repo and kw["timeout"] == g.PYTEST_TIMEOUT_S and env["PATH"] == "/usr/bin:/bin"
    assert Path(env["TMPDIR"]).parent == gate_repo / "tmp" / "release-gate" and not Path(env["TMPDIR"]).exists()
    assert re.fullmatch(r"[0-9a-f]{32}", env["VERA_GATE_NONCE"])


def test_gate_suite_args_pass_the_runner_and_the_nonce_is_forwarded():
    """GATEFIX-1029: the gate's suite args are accepted by the runner's own validator (no pinned/refused option), and
    the runner's sandbox env carries the gate child's nonce (and nothing else of it but USER). Pure; nothing spawned."""
    g = _gate()
    spec = importlib.util.spec_from_file_location("vera_test_runner_g", REPO / "scripts" / "vera_test_runner.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    pyargs, bad = r.validate_args(list(g.SUITE_ARGS), str(REPO))
    assert bad == [] and pyargs == ["-q", "tests", "--ignore", "tests/test_m2_integration.py"]   # runner splits =
    nonce = "0123456789abcdef0123456789abcdef"
    child = g.child_env({"HOME": "/h", "USER": "u", "LANG": "C", "SECRET_X": "s"}, nonce, "/gate/tmp")
    assert r.launch_refusals(child, True) == []
    box = r.sandbox_env(child)
    assert box["VERA_GATE_NONCE"] == nonce and box[r.RUNNER_MARKER] == "1" and "SECRET_X" not in box
    assert box["HOME"] == r.SANDBOX_HOME and box["PATH"] == r.FIXED_PATH


def test_a2_unchanged_by_isolation_check():
    import inspect
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(_gate().catalog_evidence_outcome))).body[0]
    fn.body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
    code = ast.unparse(fn)                                    # the docstring is excluded
    assert "subprocess" not in code and "VERA_GATE_NONCE" not in code and "pytest" not in code


def test_evidence_line_written_only_when_isolated_and_nonce_valid(tmp_path):
    nonce = "0123456789abcdef0123456789abcdef"
    files = {"test_a.py": "def test_a():\n    pass\n"}
    ok = run_child(tmp_path / "ok", files, env={"VERA_GATE_NONCE": nonce}) if (tmp_path / "ok").mkdir() is None else None
    assert f"VERA-ISOLATION-EVIDENCE netns=isolated nonce={nonce}" in ok.stdout
    (tmp_path / "bad").mkdir()
    bad = run_child(tmp_path / "bad", files, env={"VERA_GATE_NONCE": "not-a-nonce"})
    assert "VERA-ISOLATION-EVIDENCE" not in bad.stdout and bad.returncode == 0
    (tmp_path / "live").mkdir()
    live = run_child(tmp_path / "live", dict(LIVE_FILES), args=("--run-live-stages",), pre=FORCE_NOT_ISOLATED,
                     env={"VERA_GATE_NONCE": nonce})
    assert "VERA-ISOLATION-EVIDENCE" not in live.stdout and live.returncode == 0, live.stdout + live.stderr


# ------------------------------------------------------------------------------------------------ Render compatibility
def test_application_and_deploy_files_never_touch_the_test_boundary():
    """Namespace creation exists only in test sessions and the release gate; Render builds/starts never reach it."""
    needles = ("vera_test_boundary", "conftest", "unshare", "maybe_reexec", "netns", "from tests", "import tests")
    files = [REPO / "main.py", REPO / "streamlit_app.py", *(REPO / "pages").glob("*.py"), *(REPO / "vera").rglob("*.py")]
    offenders = [f"{f.relative_to(REPO)}:{n}" for f in files for n in needles if n in f.read_text()]
    assert offenders == []
    for name in ("Dockerfile", "start.sh", "run.sh", "run_demo.sh", "render.yaml", "Procfile", "requirements.txt"):
        f = REPO / name
        if f.exists():
            low = f.read_text().lower()
            assert "pytest" not in low and "conftest" not in low and "tests/" not in low and "unshare" not in low, name


def test_importing_main_does_not_decide_isolation_or_install_anything_new():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbefore = (bt.STATE["netns_decided"], bt.STATE["netns"])\n'
            'import main\nprint(before == (bt.STATE["netns_decided"], bt.STATE["netns"]), before)\n')
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO), "VERA_API_KEY": "k" * 20}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("True"), r.stdout + r.stderr


# ================================================================================================ S-b fix round (r4)
WSL_HOST = Path(bt.WSL_INTEROP_BINFMT).exists()


def test_wsl_interop_masked_inside_session():
    """On a WSL host: no interop socket is reachable and WSL_INTEROP is unset. Elsewhere the condition is trivially true."""
    if WSL_HOST:
        d = Path(bt.WSL_INTEROP_DIR)
        assert (not d.exists()) or not [n for n in os.listdir(d)]
        assert "WSL_INTEROP" not in os.environ and not any(k.startswith("WSL") for k in os.environ)
    assert bt.wsl_interop_open() is False


def test_native_windows_exec_blocked():
    with bt.kernel_layer_probe():   # INT2: proves the KERNEL layer; the audit hook is stepped aside
        """ctypes system() bypasses every Python guard; with /run/WSL masked it must not reach Windows (exit 0 only)."""
        import ctypes
        libc = ctypes.CDLL(None)
        cmd = b"/mnt/c/Windows/System32/cmd.exe /c exit 0 >/dev/null 2>&1"
        rc = libc.system(cmd)
        if WSL_HOST:
            assert rc != 0
        assert bt.STATE["violations"] == []


def test_escape_check_fails_when_interop_enabled_and_unmasked(tmp_path):
    binfmt = tmp_path / "WSLInterop"
    binfmt.write_text("enabled\ninterpreter /init\n")
    run_wsl = tmp_path / "WSL"
    run_wsl.mkdir()
    assert bt.wsl_interop_open(str(binfmt), str(run_wsl), environ={}) is False          # enabled but masked/empty
    (run_wsl / "7_interop").write_text("")
    assert bt.wsl_interop_open(str(binfmt), str(run_wsl), environ={}) is True           # unmasked
    assert bt.escape_capable([], [], str(binfmt), str(run_wsl)) is True
    (run_wsl / "7_interop").unlink()
    assert bt.wsl_interop_open(str(binfmt), str(run_wsl), environ={"WSL_INTEROP": "/run/WSL/7_interop"}) is True
    binfmt.write_text("disabled\n")
    assert bt.wsl_interop_open(str(binfmt), str(run_wsl), environ={"WSL_INTEROP": "x"}) is False
    assert bt.wsl_interop_open(str(tmp_path / "absent"), str(run_wsl), environ={"WSL_INTEROP": "x"}) is False


def test_reexec_script_masks_run_wsl():
    files, dirs = bt.escape_lists(1001, "someone")
    argv = bt.reexec_argv(["python"], 1001, 1002, "someone")
    assert "/run/WSL" in dirs and "/run/WSL" in argv and "/dev/vsock" in files and "/run/dbus/system_bus_socket" in files
    assert "unset WSL_INTEROP WSL_DISTRO_NAME WSLENV" in bt.SCRIPT and "/run/WSL" not in bt.SCRIPT


def test_wsl_variables_are_scrubbed_everywhere():
    for n in ("WSL_INTEROP", "WSL_DISTRO_NAME", "WSLENV", "WSL2_GUI_APPS_ENABLED"):
        assert bt.guarded(n), n
    e = bt._inject({"WSL_INTEROP": "x", "WSLENV": "y", "KEEP": "1"})
    assert "WSL_INTEROP" not in e and "WSLENV" not in e and e["KEEP"] == "1"
    assert not any(k.startswith("WSL") for k in bt._git_env())


# ---- OD-S6 (C): vsock
def test_dev_vsock_masked_inside_session():
    assert bt.vsock_reachable() is False


def test_vsock_reachable_pure_cases(tmp_path):
    assert bt.vsock_reachable("/dev/zero") is True            # a character device that is not /dev/null
    assert bt.vsock_reachable("/dev/null") is False           # masked
    assert bt.vsock_reachable(str(tmp_path / "absent")) is False


@pytest.mark.boundary_probe
def test_af_vsock_refused_in_the_python_guard():
    if hasattr(socket, "AF_VSOCK"):
        with pytest.raises(bt.NetworkBoundaryError):
            socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)        # create-only: nothing is ever connected
        assert [v["kind"] for v in bt.STATE["violations"]] == ["net"]
    assert bt._addr_ok(type("S", (), {"family": getattr(socket, "AF_VSOCK", -99)})(), (2, 1234)) is False


def test_unmasked_vsock_fails_closed_with_78():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "isolated"\n'
            'bt.vsock_reachable = lambda *a, **k: True\nbt.escape_lists = lambda uid, user: ([], [])\n'
            'bt.wsl_interop_open = lambda *a, **k: False\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, False)\nprint("not reached")\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.ESCAPE_MESSAGE


# ---- X2: option parsing
ISOLATING = [["-W", "ignore", "-I", "-c", "pass"], ["-Wignore", "-S", "-c", "pass"], ["-X", "dev", "-E", "-c", "pass"],
             ["--check-hash-based-pycs", "always", "-I", "-c", "pass"], ["-X", "dev", "-W", "error", "-IS", "-c", "pass"]]


@pytest.mark.parametrize("flags", ISOLATING)
@pytest.mark.boundary_probe
def test_isolation_flags_after_arg_taking_options_refused(flags):
    before = len(bt.STATE["violations"])
    with pytest.raises(bt.ChildProcessBoundaryError):
        subprocess.run([sys.executable, *flags], capture_output=True)
    assert len(bt.STATE["violations"]) == before + 1 and bt._isolated([sys.executable, *flags], None) is True


@pytest.mark.parametrize("flags", [["-W", "ignore", "-c", "pass"], ["-Wignore", "-c", "pass"], ["-X", "dev", "-u", "-c", "pass"],
                                   ["--check-hash-based-pycs", "always", "-c", "pass"], ["-c", "pass", "-I"], ["-m", "json.tool"]])
def test_benign_option_clusters_are_not_isolated(flags):
    assert bt._isolated([sys.executable, *flags], {"PATH": "/usr/bin", "PYTHONPATH": "x"}) is False


# ---- F4: git children
def test_git_child_env_is_hardened(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/nonexistent")
    monkeypatch.setenv("GIT_SSH_COMMAND", "evil")
    env = bt._spawn_env(["git", "rev-parse", "HEAD"], {"GIT_DIR": "/x", "GIT_ASKPASS": "y", "PATH": "/tmp/shadow"}, via_popen=True)
    assert set(k for k in env if k.startswith("GIT_")) == {"GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT",
                                                           "GIT_OPTIONAL_LOCKS", "GIT_NO_LAZY_FETCH", "GIT_CONFIG_COUNT",
                                                           *(f"GIT_CONFIG_{w}_{i}" for w in ("KEY", "VALUE") for i in range(6))}
    assert env["PATH"] == "/usr/bin:/bin" and env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_OPTIONAL_LOCKS"] == "0" and env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_CONFIG_COUNT"] == "6"
    pairs = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(6)}
    assert pairs["core.fsmonitor"] == "false" and pairs["core.hooksPath"] == "/dev/null" and pairs["protocol.allow"] == "never"
    assert Path(env["HOME"]).is_dir() and bt._under(Path(env["HOME"]), Path(bt.STATE["bootstrap_dir"]))
    assert bt._pinned_git().startswith("/usr/") or bt._pinned_git().startswith("/bin/")
    with pytest.raises(bt.ChildProcessBoundaryError):                      # git is only allowed through Popen
        bt._spawn_env(["git", "rev-parse", "HEAD"], None, via_popen=False)
    bt.STATE["violations"].clear()


def test_git_on_shadowed_path_is_not_used(tmp_path):
    marker = tmp_path / "shadow-ran"
    fake = tmp_path / "git"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\nexit 9\n")
    fake.chmod(0o755)
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                       env={"PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"})
    assert not marker.exists() and r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", r.stdout.strip())


def test_git_fsmonitor_config_is_overridden(tmp_path):
    with bt.kernel_layer_probe():   # INT2: proves the KERNEL layer; the audit hook is stepped aside
        import ctypes
        libc = ctypes.CDLL(None)
        repo, marker, hook = tmp_path / "r", tmp_path / "fsmonitor-ran", tmp_path / "hook.sh"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\nprintf '\\0'\n")
        hook.chmod(0o755)
        setup = (f"git init -q {repo} && cd {repo} && git config core.fsmonitor {hook} && git config user.email a@b.invalid "
                 f"&& git config user.name n && echo x > f && git add f >/dev/null 2>&1")
        assert libc.system(setup.encode()) == 0                       # native system(): only the test's own setup
        marker.unlink(missing_ok=True)                                # the setup itself may have run the hook
        r = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
        assert r.returncode == 0 and not marker.exists()               # the allowlisted child overrides core.fsmonitor
        assert libc.system(f"cd {repo} && git status --porcelain >/dev/null 2>&1".encode()) == 0
        assert marker.exists()                                         # control: without the override the hook does run


# ---- X4: live module selection by AST
def test_live_session_ignores_modules_that_only_mention_mark_live(tmp_path):
    files = {
        "test_mention.py": ('"""mentions pytest.mark.live in prose"""\nimport pathlib\nS = "@pytest.mark.live"\n'
                            'pathlib.Path(__file__).with_name("MENTION_IMPORTED").write_text("x")\ndef test_m():\n    pass\n'),
        "test_marked.py": 'import pytest\n@pytest.mark.live\ndef test_live_one():\n    pass\n',
        "test_pm.py": 'import pytest\npytestmark = pytest.mark.live_api\ndef test_live_two():\n    pass\n',
    }
    r = run_child(tmp_path, files, args=("--run-live-stages",), pre=FORCE_NOT_ISOLATED, env={"API_BASE_URL": "https://x.invalid"})
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 passed" in out and not (tmp_path / "MENTION_IMPORTED").exists(), out


# ---- unshare failure exits 78 (not 1); a failing test run keeps its own exit code
def _shell_status(cmd: str, tmp_path) -> tuple:
    with bt.kernel_layer_probe():   # INT2: proves the KERNEL layer; the audit hook is stepped aside
        import ctypes
        errf = tmp_path / "err.txt"
        status = ctypes.CDLL(None).system(f"{cmd} 2>{errf}".encode())
        return os.waitstatus_to_exitcode(status), errf.read_text() if errf.exists() else ""


def test_failing_unshare_exits_78_with_the_message(tmp_path):
    import shlex
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "unshare").write_text("#!/bin/sh\nexit 1\n")
    (fake / "unshare").chmod(0o755)
    argv = bt.reexec_argv(["true"], os.getuid(), os.getgid(), "")
    rc, err = _shell_status(f"PATH={fake}:/usr/bin:/bin {shlex.join(argv)}", tmp_path)
    assert rc == 78 and err.strip() == bt.NETNS_MESSAGE


def test_inner_exit_code_is_preserved_when_unshare_works(tmp_path):
    import shlex
    argv = bt.reexec_argv(["sh", "-c", "exit 3"], os.getuid(), os.getgid(), "")
    rc, _err = _shell_status(shlex.join(argv), tmp_path)
    assert rc == 3


# ================================================================================================ S-b-F5: seccomp filter
import errno as _errno
import platform as _platform

AF_VSOCK = 40


def test_filter_is_installed_in_this_session():
    assert bt.STATE["seccomp"] == "installed"


def test_native_af_vsock_socket_refused_in_session():
    import _socket
    with pytest.raises(OSError) as e:
        _socket.socket(AF_VSOCK, _socket.SOCK_STREAM)                  # native base class: no Python guard involved
    assert e.value.errno == _errno.EACCES


def test_child_inherits_socket_filter():
    code = ('import _socket, errno\ntry:\n    _socket.socket(40, _socket.SOCK_STREAM)\n    print("created")\n'
            'except OSError as e:\n    print(e.errno == errno.EACCES)\n')
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "True", r.stdout + r.stderr


def test_io_uring_setup_refused():
    with bt.kernel_layer_probe():   # INT2: proves the KERNEL layer; the audit hook is stepped aside
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        params = (ctypes.c_ubyte * 120)()
        nr = bt._ARCHES[_platform.machine()][2]
        assert libc.syscall(nr, 1, params) == -1 and ctypes.get_errno() == _errno.EPERM


@pytest.mark.vera_sockets      # K18W OA-2: local sockets only on explicit request
def test_inet_and_unix_sockets_unaffected_by_filter(tmp_path, monkeypatch):
    import _socket
    for fam in (_socket.AF_INET, _socket.AF_INET6, _socket.AF_UNIX):
        _socket.socket(fam, _socket.SOCK_STREAM).close()
    _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM).close()
    srv = socket.create_server(("127.0.0.1", 0))
    socket.create_connection(srv.getsockname(), timeout=2).close()
    srv.close()
    monkeypatch.chdir(tmp_path)
    u = _unix_server("f.sock")
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect("f.sock")
    c.close()
    u.close()
    a, b = socket.socketpair()
    a.close(); b.close()


def test_filter_install_failure_fails_closed_78():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "isolated"\n'
            'bt.escape_lists = lambda uid, user: ([], [])\nbt.wsl_interop_open = lambda *a, **k: False\n'
            'bt.vsock_reachable = lambda *a, **k: False\nbt.STATE["seccomp"] = "none"\n'
            'def boom():\n    raise OSError(1, "simulated seccomp failure")\nbt.install_socket_filter = boom\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, False)\nprint("not reached")\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.SECCOMP_MESSAGE and "not reached" not in r.stdout


def test_filter_install_failure_in_a_live_session_also_fails_closed_78():
    code = ('import sys\nbt = sys.modules["tests.vera_test_boundary"]\nbt.netns_status = lambda *a, **k: "not_isolated"\n'
            'def boom():\n    raise KeyError("riscv64")\nbt.install_socket_filter = boom\n'
            'class C:\n    class pluginmanager:\n        getplugin = staticmethod(lambda n: None)\n'
            'bt.maybe_reexec(C, True)\n')
    r = _child_python(code)
    assert r.returncode == 78 and r.stderr.strip() == bt.SECCOMP_MESSAGE


def test_filter_installs_on_a_fresh_process_with_tsync_and_no_new_privs():
    code = ('import sys, ctypes\nbt = sys.modules["tests.vera_test_boundary"]\nbt.STATE["seccomp"] = "none"\n'
            'bt.install_socket_filter()\nlibc = ctypes.CDLL(None)\nprint(libc.prctl(39, 0, 0, 0, 0), bt.STATE["seccomp"])\n')
    r = _child_python(code)
    assert r.stdout.split() == ["1", "installed"], r.stdout + r.stderr          # PR_GET_NO_NEW_PRIVS == 1


# ---- architecture and ABI handling, evaluated with a classic-BPF interpreter against the real program text
@pytest.mark.parametrize("machine", sorted(bt._ARCHES))
def test_filter_program_verdicts_per_architecture(machine):
    audit_arch, nr_socket, nr_uring, _seccomp, reject_x32 = bt._ARCHES[machine]
    prog = bt.build_socket_filter(machine)
    run = lambda arch, nr, a0=0: bt.run_socket_filter(prog, arch, nr, a0)          # noqa: E731
    assert run(audit_arch, nr_socket, AF_VSOCK) == bt._ret_errno(_errno.EACCES)
    assert run(audit_arch, nr_socket, (7 << 32) | AF_VSOCK) == bt._ret_errno(_errno.EACCES)   # only the low 32 bits count
    assert run(audit_arch, nr_socket, 2) == bt.SECCOMP_RET_ALLOW                                 # AF_INET
    assert run(audit_arch, nr_socket, 1) == bt.SECCOMP_RET_ALLOW                                 # AF_UNIX
    assert run(audit_arch, nr_uring) == bt._ret_errno(_errno.EPERM)
    assert run(audit_arch, 0) == bt.SECCOMP_RET_ALLOW and run(audit_arch, 1000) == bt.SECCOMP_RET_ALLOW
    for other in (0x40000003, 0xC00000B7 if machine == "x86_64" else 0xC000003E, 0x40000028):   # i386, the other 64-bit arch, arm
        assert run(other, nr_socket, AF_VSOCK) == bt.SECCOMP_RET_KILL_PROCESS
        assert run(other, 1) == bt.SECCOMP_RET_KILL_PROCESS
    if reject_x32:                                                      # x32 ABI: 64-bit arch value, nr carries bit 30
        assert run(audit_arch, bt.X32_SYSCALL_BIT | nr_socket, AF_VSOCK) == bt.SECCOMP_RET_KILL_PROCESS
        assert run(audit_arch, bt.X32_SYSCALL_BIT | 1) == bt.SECCOMP_RET_KILL_PROCESS


def test_unsupported_architecture_is_an_install_error():
    with pytest.raises(KeyError):
        bt.build_socket_filter("riscv64")
