"""DEV-A: the VERA_TEST_RO_BUNDLE selector of scripts/vera_test_runner.py (pure functions; nothing is mounted or run)."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE = str(REPO.parent / "work" / "fd-1020" / "bundle")


def _runner():
    spec = importlib.util.spec_from_file_location("vera_test_runner_ro", REPO / "scripts" / "vera_test_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script(R, extra):
    return R.sandbox_script("/opt/u/repo", "/opt/u/repo/.venv", [], [], 1000, 1000, "/opt/u/repo/.venv/bin/python",
                            ["tests"], R.sandbox_env({}, extra[0] if extra else None), path_exists=lambda p: True,
                            realpath=lambda p: p, ro_extra=extra)


def test_selector_unset_changes_nothing():
    R = _runner()
    assert R.ro_bundle({}, str(REPO)) == (None, [])
    assert "VERA_REAL_DAR9_BUNDLE" not in R.sandbox_env({})
    assert "VERA_REAL_DAR9_BUNDLE" not in R.sandbox_env({"VERA_REAL_DAR9_BUNDLE": "/x"})  # never passed through
    s = _script(R, ())
    assert "fd-1020" not in s and "bundle-recheck" not in s


def test_selector_set_refuses_non_one_and_missing_bundle(tmp_path):
    R = _runner()
    for bad in ("0", "", "/opt/x", "true"):
        path, why = R.ro_bundle({R.RO_BUNDLE_ENV: bad}, str(REPO))
        assert path is None and why
    root = tmp_path / "vera"
    root.mkdir()
    path, why = R.ro_bundle({R.RO_BUNDLE_ENV: "1"}, str(root))  # no R/work/fd-1020/bundle beside it
    assert path is None and why


def test_selector_set_rejects_symlink_and_credential_file(tmp_path):
    R = _runner()
    root = tmp_path / "vera"
    b = tmp_path / "work" / "fd-1020" / "bundle"
    (b / "texts").mkdir(parents=True)
    root.mkdir()
    (b / "texts" / "a.txt").write_text("x")
    root = str(root)
    assert R.ro_bundle({R.RO_BUNDLE_ENV: "1"}, root) == (str(b.resolve()), [])
    (b / "link").symlink_to(b / "texts" / "a.txt")
    path, why = R.ro_bundle({R.RO_BUNDLE_ENV: "1"}, root)
    assert path is None and any("symlink" in w for w in why)
    (b / "link").unlink()
    (b / ".env-x").write_text("k")
    path, why = R.ro_bundle({R.RO_BUNDLE_ENV: "1"}, root)
    assert path is None and any("credential" in w for w in why)


def test_selector_set_binds_read_only_sets_env_and_rechecks():
    R = _runner()
    assert R.sandbox_env({}, BUNDLE)["VERA_REAL_DAR9_BUNDLE"] == BUNDLE
    s = _script(R, (BUNDLE,))
    for needle in (f"--bind {BUNDLE} ", f"--move /mnt/.vera-stage", f"remount,bind,ro {BUNDLE}", "-type l -o -type f -links +1",
                   "bundle-recheck-link", "bundle-recheck-credential", "! -name '*.example'"):
        assert needle in s, needle
    assert s.index(f"remount,bind,ro {BUNDLE}") < s.index("bundle-recheck-link") < s.index("exec /usr/bin/unshare")


def test_runner_tmp_env_and_bind_are_one_run_directory():
    R = _runner()
    run = "/opt/u/work/test-tmp/rab12"
    env = R.sandbox_env({}, None, run)
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == run
    assert R.sandbox_env({})["TMPDIR"] == "/tmp"
    s = R.sandbox_script("/opt/u/repo", "/opt/u/repo/.venv", [], [], 1000, 1000, "/opt/u/repo/.venv/bin/python",
                         ["tests"], env, path_exists=lambda p: True, realpath=lambda p: p, rw_extra=[run])
    assert s.count("--bind /opt/u/work/test-tmp/rab12 ") == 1 and "test-tmp'" not in s
    assert "remount,bind,ro /opt/u/work/test-tmp/rab12" not in s      # the run directory is the one rw extra bind
    assert "tmpfs /tmp " in s or "tmpfs /tmp" in s                       # the /tmp mask is still applied


def test_tmp_path_resolves_under_runner_test_tmp(tmp_path):
    import os
    base = os.path.realpath(REPO.parent / "work" / "test-tmp")
    assert os.path.realpath(tmp_path).startswith(base + os.sep), (tmp_path, base)
    for name in ("TMPDIR", "TEMP", "TMP"):
        assert os.path.realpath(os.environ[name]).startswith(base + os.sep), name
