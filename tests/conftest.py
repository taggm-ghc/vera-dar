# K2.S-a: install the test boundary BEFORE anything else can import an application module or call load_dotenv.
import importlib.util
import pathlib

import sys as _sys
_bp = pathlib.Path(__file__).resolve().parent / "vera_test_boundary.py"
if "tests.vera_test_boundary" not in _sys.modules:
    _bs = importlib.util.spec_from_file_location("tests.vera_test_boundary", _bp)
    _bm = importlib.util.module_from_spec(_bs)
    _sys.modules["tests.vera_test_boundary"] = _bm
    _bs.loader.exec_module(_bm)
_boundary = _sys.modules["tests.vera_test_boundary"]
_boundary.install()

# STUB. The M3 instance's original conftest (FakeLLM(**{"m3.x": ...}), domains()) was accidentally
# overwritten by the M4/M5 instance and must be restored by the M3 owner. M4/M5 tests use
# tests/m45_support.py and do not depend on this file.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Owner decision #7: live tests (real LLM/provider calls, deployed API) run only on explicit opt-in:
#   pytest tests --run-live-stages                      (provider-only live tests: marker `live`)
#   API_BASE_URL=... pytest tests --run-live-stages     (tests that call the deployed VERA-DAR API: marker `live_api`)
# `live_api` tests also need API_BASE_URL. No persistent environment switch exists.
import datetime
import os
import re
import warnings

import pytest

# Deprecated alias for the old gate (VERA_INTEGRATION=1); honoured through this date, then ignored.
ALIAS_LAST_DAY = datetime.date(2026, 10, 25)


def pytest_addoption(parser):
    parser.addoption("--run-live-stages", action="store_true", default=False,
                     help="run tests marked 'live'/'live_api' (live_api also needs API_BASE_URL in the environment)")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # 14: before anything else (and before any test module is imported) make the kernel enforce "no network"
    _boundary.maybe_reexec(config, bool(config.getoption("--run-live-stages")) or alias_active(os.environ))
    config.addinivalue_line("markers", "live: makes live provider calls; needs --run-live-stages")
    config.addinivalue_line("markers", "live_api: calls the deployed VERA-DAR API; needs --run-live-stages and API_BASE_URL")
    config.addinivalue_line("markers", "boundary_probe: deliberately triggers a refused DB access; violations are expected")
    config.addinivalue_line("markers", "vera_sockets: this test explicitly requests local sockets (loopback, AF_UNIX under "
                            "the session basetemp); without it every socket connect is refused (owner decision OA-2)")
    _boundary.set_live_enabled(bool(config.getoption("--run-live-stages")) or alias_active(os.environ))


def alias_active(env, today=None) -> bool:
    today = today or datetime.date.today()
    return env.get("VERA_INTEGRATION") == "1" and today <= ALIAS_LAST_DAY


def live_skip_reason(flag: bool, env, needs_api: bool = False) -> str | None:
    """Reason a live test must be skipped, or None. Names the missing piece, never the URL value."""
    missing = []
    if not flag:
        missing.append("--run-live-stages not passed")
    if needs_api and not str(env.get("API_BASE_URL", "")).strip():
        missing.append("API_BASE_URL not set")
    return ("live test skipped: " + "; ".join(missing)) if missing else None


def pytest_collection_modifyitems(config, items):
    flag = config.getoption("--run-live-stages")
    if not flag and alias_active(os.environ):
        warnings.warn("VERA_INTEGRATION=1 is a deprecated alias for --run-live-stages and stops working "
                      "after 2026-10-25; pass --run-live-stages instead.", DeprecationWarning, stacklevel=1)
        flag = True
    for item in items:
        api = "live_api" in item.keywords
        if api or "live" in item.keywords:
            reason = live_skip_reason(flag, os.environ, needs_api=api)
            if reason:
                item.add_marker(pytest.mark.skip(reason=reason))
    if _boundary.STATE["netns"] == "live-session":      # 14 N-4r: a live session runs live-marked items only
        keep = [i for i in items if "live" in i.keywords or "live_api" in i.keywords]
        dropped = [i for i in items if i not in keep]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = keep
        _boundary.STATE["live_deselected"] = len(dropped)
        if not all("live" in i.keywords or "live_api" in i.keywords for i in items):
            raise pytest.UsageError("live session collected a non-live item")


def pytest_ignore_collect(collection_path, config):
    """N-4r: in a live session non-live test modules are never imported (their module-level code would run unisolated)."""
    if _boundary.STATE["netns"] != "live-session":
        return None
    if collection_path.is_file() and collection_path.name.startswith("test_") and collection_path.suffix == ".py":
        return not _module_marks_live(collection_path)
    return None


def _module_marks_live(path) -> bool:
    """X4: True only if the module has a decorator or a `pytestmark` that uses pytest.mark.live / .live_api. Parsed, never
    imported; a string that merely mentions "mark.live" does not count. Unreadable or unparsable: not live."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return False

    def uses_live(node) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr in ("live", "live_api")
                   and isinstance(n.value, ast.Attribute) and n.value.attr == "mark" for n in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(uses_live(d) for d in node.decorator_list):
            return True
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in node.targets) and uses_live(node.value):
            return True
    return False


def pytest_sessionstart(session):
    if _boundary.STATE["netns"] not in ("isolated", "live-session"):
        _boundary._isolation_fail(session.config, _boundary.NETNS_MESSAGE)
    _boundary.seal_session()          # K18W M-2: per-item tamper check compares against this


def pytest_report_header(config):
    return {"isolated": "VERA-DAR test network isolation: isolated (user+net namespace)",
            "live-session": "VERA-DAR test network isolation: live session (live-marked items only)"}.get(
        _boundary.STATE["netns"], "VERA-DAR test network isolation: undecided")


def pytest_terminal_summary(terminalreporter):
    nonce = _boundary._ORIG_ENV.get("VERA_GATE_NONCE", "")
    if _boundary.STATE["tamper_count"]:
        terminalreporter.write_line(f"VERA-DAR isolation evidence withheld: {_boundary.STATE['tamper_count']} test(s) left "
                                    "isolation state changed (see the 'test boundary tamper' failures)")
    elif _boundary.STATE["netns"] == "isolated" and re.fullmatch(r"[0-9a-f]{32}", nonce or ""):
        terminalreporter.write_line(f"VERA-ISOLATION-EVIDENCE netns=isolated nonce={nonce}")
    if _boundary.STATE["live_deselected"]:
        terminalreporter.write_line(f"live session: {_boundary.STATE['live_deselected']} non-live items deselected "
                                    "(they run only in the isolated default session)")


# --- K2.S-a boundary hooks (the logic lives in tests/vera_test_boundary.py) ---------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # S-b: reset the per-item recorder BEFORE any fixture runs, so refusals during fixture setup belong to this item
    _boundary.begin_item(live_item="live" in item.keywords or "live_api" in item.keywords,
                         sockets="vera_sockets" in item.keywords)
    _boundary.set_phase("setup")


def pytest_runtest_logfinish(nodeid, location):
    _boundary.end_item()


def pytest_sessionfinish(session, exitstatus):
    """S-b: a refusal recorded outside any test item (collection, session fixtures) fails the session loudly."""
    msg = _boundary.orphan_message()
    if msg:
        _sys.stderr.write(msg + "\n")
        session.exitstatus = 1
    if _boundary.STATE["tamper_count"]:      # K18W M-2: a persisting isolation change fails the whole session
        session.exitstatus = 1


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    _boundary.set_phase("call")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    _boundary.set_phase("teardown")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    msg = _boundary.violations_message(call.when, probe="boundary_probe" in item.keywords)
    if call.when == "teardown":       # K18W M-2: after fixture teardown (monkeypatch already undone)
        tamper = _boundary.tamper_message()
        if tamper:
            msg = (msg + "\n" + tamper) if msg else tamper
    if msg:  # D2: a refused access fails the phase even if the test skipped or production code swallowed the error
        rep = outcome.get_result()
        rep.outcome = "failed"
        rep.longrepr = msg


_SINGLETONS = (("vera.memory_store", "_SERVICE", "clear"), ("vera.corpus_admission", "_STORE", "clear"),
               ("vera.db", "_engine", None), ("vera.db", "SessionLocal", None), ("db", "_admin_engine", None))


def _reset_singletons(monkeypatch):
    """Only modules already imported; never imports one."""
    for modname, attr, how in _SINGLETONS:
        mod = _sys.modules.get(modname)
        if mod is not None and hasattr(mod, attr):
            if how == "clear":
                getattr(mod, attr).clear()
            else:
                monkeypatch.setattr(mod, attr, None)


@pytest.fixture(autouse=True)
def _vera_boundary_item(request, monkeypatch):
    """A4: re-scrub guarded names that leaked in; clear cached engines/services. A3: a live item in a live session
    gets provider keys back (never DB, admin or proxy names); the per-item reset is in pytest_runtest_setup."""
    for name in list(os.environ):
        if _boundary.guarded(name):
            monkeypatch.delenv(name, raising=False)
    for name, value in _boundary.provider_env_for_live().items():
        monkeypatch.setenv(name, value)
    _reset_singletons(monkeypatch)
    yield
    _reset_singletons(monkeypatch)


@pytest.fixture(autouse=True)
def _reset_dar_runtime_after_test():
    """Teardown only: a cached dar_runtime corpus left by one test must not leak into later tests/selections."""
    yield
    mod = _sys.modules.get("vera.dar_runtime")
    if mod is not None:
        mod.reset_for_tests()
