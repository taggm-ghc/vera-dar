"""K18W INT4 regression tests for the WS-8h findings (each FAILS on INT3, PASSES on INT4).

MED     : the runner accepts an ALLOWLIST of options only. -W/--pythonwarnings (an `action::module.Category` filter
          imports `module`), -o/--override-ini, --config-file, --import-mode, --debug, --trace-config, argparse
          abbreviations and clustered short options are refused, option values are checked, and PYTEST_ADDOPTS /
          PYTEST_PLUGINS in the caller's environment are refused.
LOW-MED : planted_configs(), the runner's credential mask and the boundary's credential scan no longer skip
          node_modules / venv / .venv / __pycache__ directories below the checkout root; only the root .git (and, for
          the planted-config scan, root-level venv/node_modules dirs outside tests/) and the pinned venv are skipped.
Pure functions only: no pytest session is started, no socket of any kind is created, no DB, no network.
Every credential file here is a fake, non-secret literal in tmp_path.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

bt = sys.modules["tests.vera_test_boundary"]
REPO = Path(__file__).resolve().parent.parent
FAKE = "int4-sentinel-not-a-secret\n"


def _runner():
    spec = importlib.util.spec_from_file_location("vera_test_runner_int4", REPO / "scripts" / "vera_test_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_root(tmp_path):
    root = tmp_path / "root"
    (root / "tests" / "real").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "tests" / "real" / "test_a.py").write_text("def test_a():\n    pass\n")
    return root


# ============================================================================== MED: option allowlist
IMPORTING_OR_CONFIG = (
    ["-W", "ignore::tests.evil.Warn", "tests"], ["-Wignore::tests.evil.Warn", "tests"],
    ["-W", "ignore", "tests"], ["--pythonwarnings", "ignore::tests.evil.Warn", "tests"],
    ["--pythonwarnings=ignore::tests.evil.Warn", "tests"], ["-o", "addopts=-p tests.evil", "tests"],
    ["-oaddopts=x", "tests"], ["--override-ini", "pythonpath=tests", "tests"], ["--override-ini=x=y", "tests"],
    ["--config-file", "tests/evil.ini", "tests"], ["--config-file=tests/evil.ini", "tests"],
    ["--import-mode=importlib", "tests"], ["--import-mode", "append", "tests"], ["--debug", "tests"],
    ["--trace-config", "tests"], ["--pyargs", "tests"], ["--override", "x=y", "tests"], ["--pythonw=x", "tests"],
    ["--conf", "tests/evil.ini", "tests"], ["-qp", "tests.evil", "tests"], ["-qW", "ignore", "tests"],
    ["-vpno:x", "tests"], ["--assert=plain", "tests"], ["--plugins", "x", "tests"], ["--runxfail", "tests"],
    ["--setup-plan", "tests"], ["--keep-duplicates", "tests"], ["--capture=no", "tests"], ["-s", "tests"],
    ["--junit-xml=/tmp/x.xml", "tests"], ["--log-file=/tmp/x", "tests"], ["--rootdir=/tmp", "tests"],
    ["-c", "pytest.ini", "tests"], ["-p", "no:cacheprovider", "tests"], ["--", "tests"],
)
BAD_VALUES = (
    ["--tb=evil", "tests"], ["--tb", "-p", "tests"], ["--maxfail=-1", "tests"], ["--maxfail", "x", "tests"],
    ["--durations=x", "tests"], ["-r", "fE;", "tests"], ["-r=fE", "tests"], ["-k", "-p", "tests"],
    ["-m", "--noconftest", "tests"], ["--deselect", "-p", "tests"], ["-k", "", "tests"],
)


def test_int4_runner_refuses_importing_and_config_options():
    R = _runner()
    root = str(REPO)
    for args in IMPORTING_OR_CONFIG + BAD_VALUES:
        _, bad = R.validate_args(args, root)
        assert bad, f"VULNERABLE: option accepted by the runner: {args}"


def test_int4_runner_names_the_warning_filter_refusal():
    R = _runner()
    for args in (["-W", "ignore::tests.evil.Warn", "tests"], ["--pythonwarnings=x", "tests"]):
        _, bad = R.validate_args(args, str(REPO))
        assert any("warning filter" in b for b in bad), bad


def test_int4_runner_refuses_pytest_environment_options():
    R = _runner()
    for env in ({"PYTEST_ADDOPTS": "-p tests.evil"}, {"PYTEST_PLUGINS": "tests.evil"},
                {"PYTEST_ADDOPTS": "", "PATH": "/usr/bin"}):
        assert R.launch_refusals(env, True), f"VULNERABLE: pytest env option accepted: {sorted(env)}"
    assert R.launch_refusals({"PATH": "/usr/bin", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, True) == []   # control


def test_int4_runner_allowlist_keeps_legitimate_options():
    """control: the gate argv and the common options are still accepted unchanged."""
    R = _runner()
    root = str(REPO)
    for args, want in ((["-q", "tests", "--ignore=tests/test_m2_integration.py"], None),
                       (["-q", "-rfE", "-k", "a and not b", "--basetemp", "/tmp/bt", "--tb=short", "--maxfail=3",
                         "--durations", "5", "-mnot slow", "-x", "-vv", "--co", "--strict-markers", "-l",
                         "--deselect", "tests/test_env_guard.py::test_x", "tests/test_env_guard.py::test_x"], None),
                       (["-rA", "-kfoo", "tests"], ["-rA", "-kfoo", "tests"])):
        out, bad = R.validate_args(args, root)
        assert bad == [], (args, bad)
        if want is not None:
            assert out == want


# ============================================================================== LOW-MED: no skipped dirs in tests/
SKIP_NAMES = ("node_modules", "venv", ".venv", "__pycache__")


def test_int4_planted_configs_scans_skip_named_dirs_in_tests(tmp_path):
    R = _runner()
    root = _fake_root(tmp_path)
    assert R.planted_configs(str(root)) == []                                        # control
    planted = []
    for n in SKIP_NAMES:
        (root / "tests" / n).mkdir()
        (root / "tests" / n / "conftest.py").write_text("")
        planted.append(f"tests/{n}/conftest.py")
    (root / "tests" / "venv" / "x.pth").write_text("")
    (root / "tests" / "node_modules" / "sitecustomize.py").write_text("")
    (root / "tests" / "real" / "node_modules").mkdir()
    (root / "tests" / "real" / "node_modules" / "pytest.ini").write_text("[pytest]\naddopts = --noconftest\n")
    planted += ["tests/venv/x.pth", "tests/node_modules/sitecustomize.py", "tests/real/node_modules/pytest.ini"]
    found = R.planted_configs(str(root))
    missing = sorted(set(planted) - set(found))
    assert not missing, f"VULNERABLE: planted config under a skip-named dir in tests/ not flagged: {missing}"


def test_int4_planted_configs_skips_only_the_pinned_venv(tmp_path):
    R = _runner()
    root = _fake_root(tmp_path)
    (root / "tests" / ".venv").mkdir()
    (root / "tests" / ".venv" / "x.pth").write_text("")
    (root / "tests" / "venv").mkdir()
    (root / "tests" / "venv" / "y.pth").write_text("")
    found = R.planted_configs(str(root), venv=str(root / "tests" / ".venv"))
    assert "tests/venv/y.pth" in found and "tests/.venv/x.pth" not in found, found


def _cred_tree(tmp_path):
    root = _fake_root(tmp_path)
    want = []
    for n in SKIP_NAMES:
        (root / "tests" / n).mkdir()
        (root / "tests" / n / "k.pem").write_text(FAKE)
        want.append(root / "tests" / n / "k.pem")
    for n in ("node_modules", "venv"):
        (root / n).mkdir()
        (root / n / "a.key").write_text(FAKE)
        want.append(root / n / "a.key")
    (root / ".venv").mkdir()                                       # the pinned venv: skipped
    (root / ".venv" / "cacert.pem").write_text(FAKE)
    (root / ".git").mkdir()                                        # the root .git: skipped
    (root / ".git" / "x.pem").write_text(FAKE)
    return root, sorted(os.path.realpath(p) for p in want)


def test_int4_runner_credential_mask_covers_skip_named_dirs(tmp_path):
    R = _runner()
    root, want = _cred_tree(tmp_path)
    got = R.credential_files(str(root))
    missing = sorted(set(want) - set(got))
    assert not missing, f"VULNERABLE: credential-type file not masked by the runner: {missing}"
    got = R.credential_files(str(root), venv=str(root / ".venv"))
    assert got == want, got                                         # pinned venv and root .git only are skipped


def test_int4_boundary_credential_scan_covers_skip_named_dirs(tmp_path):
    root, want = _cred_tree(tmp_path)
    got = bt.credential_files_visible(homes=[], repo=str(root))
    missing = sorted(set(want) - set(got))
    assert not missing, f"VULNERABLE: readable credential-type file not seen by the boundary: {missing}"
    got = bt.credential_files_visible(homes=[], repo=str(root), venv=str(root / ".venv"))
    assert got == want, got
