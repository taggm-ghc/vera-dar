# Root conftest (K2.S-a): install the test boundary first, for bare `pytest` from the repo root too.
import importlib.util, pathlib, sys
_p = pathlib.Path(__file__).resolve().parent / "tests" / "vera_test_boundary.py"
if "tests.vera_test_boundary" not in sys.modules:
    _s = importlib.util.spec_from_file_location("tests.vera_test_boundary", _p)
    _m = importlib.util.module_from_spec(_s); sys.modules["tests.vera_test_boundary"] = _m; _s.loader.exec_module(_m)
sys.modules["tests.vera_test_boundary"].install()
# db/ and scripts/ hold operator scripts that read production settings or run against PRODUCTION; they are never pytest tests.
# vera/m6/test_m6_schema_sync.py is a standalone check script with a module-level sys.exit (pre-existing; it crashes
# bare-pytest collection); it is not a pytest test.
collect_ignore = ["db", "scripts", "vera/m6/test_m6_schema_sync.py"]


# M3-A' (PF-18 M-3): refuse, before collection imports any module, every run whose positional args are not `tests` or
# a path under tests/. An explicit path bypasses collect_ignore (pytest initpaths), and only tests/conftest.py holds the
# netns re-exec, the sessionstart gate and the evidence line. A bare `pytest` (args = the invocation dir) is refused
# too: the supported entry points are the release gate / launcher, whose argv is `tests` (OD-RB4-3).
# S-3: --rootdir / --confcutdir must be the repo root or under tests/, and the inifile must be the tracked root
# pytest.ini (or none). Honest limit: this runs only when THIS conftest loads; a --confcutdir or -c below the repo root
# skips it entirely (route R4c, like --noconftest R4). With the tracked pytest.ini, --rootdir alone no longer moves the
# confcutdir (pytest uses the inifile's directory).
def _vera_resolve(arg, base):
    p = pathlib.Path(str(arg).split("::", 1)[0])
    try:
        return (p if p.is_absolute() else base / p).resolve()
    except (OSError, RuntimeError):
        return None


def _vera_outside_tests(config) -> list:
    root = pathlib.Path(__file__).resolve().parent
    tests_dir = root / "tests"
    base = pathlib.Path(config.invocation_params.dir)
    bad = []
    for arg in config.args:
        r = _vera_resolve(arg, base)
        if r is None or (r != tests_dir and tests_dir not in r.parents):
            bad.append(str(arg))
    for opt in ("rootdir", "confcutdir"):
        v = getattr(config.option, opt, None)
        if v:
            r = _vera_resolve(v, base)
            if r is None or (r != root and r != tests_dir and tests_dir not in r.parents):
                bad.append(f"--{opt}={v}")
    ini = getattr(config, "inipath", None)
    if ini is not None and pathlib.Path(ini).resolve() != root / "pytest.ini":
        bad.append(f"inifile {ini}")
    return bad


import pytest as _pytest  # noqa: E402


@_pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if config.getoption("help", False):
        return
    bad = _vera_outside_tests(config)
    if bad or not config.args:
        raise _pytest.UsageError("VERA-DAR: only `tests` or paths under tests/ may be collected (refused: "
                                 + ", ".join(bad or ["<no args>"]) + "); operator scripts are not pytest tests")



# M3-G (K18W, PF-18 M-3 / O9 redesign at the runner boundary): the root conftest enforces the netns gate itself, so
# dropping tests/conftest.py (for example `-p no:<path>`) no longer removes the only isolation gate. pytest_sessionstart
# runs before collection imports any test module. Policy lives here and in tests/conftest.py, never in
# tests/__init__.py: that package is also imported by authorized operator scripts (scripts/live_catalog_check.py imports
# tests.sql_contract_support), which must stay executable outside the test boundary. --noconftest (R4) and direct
# `python <file>` (R5) bypass every conftest; they are covered only by the runner/sandbox boundary (gate fixed argv,
# env -i, credential masks, netns), not in-repo.
def pytest_sessionstart(session):
    bt = sys.modules["tests.vera_test_boundary"]
    if not bt.STATE["netns_decided"]:     # tests/conftest.py did not run: decide here (same rules, never a live session)
        bt.maybe_reexec(session.config, False)
    if bt.STATE["netns"] not in ("isolated", "live-session"):
        bt._isolation_fail(session.config, bt.NETNS_MESSAGE)
