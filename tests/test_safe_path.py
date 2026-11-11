"""ENV-PATHS-1110: vera.safe_path containment helper and its three call sites. Offline; uses only tmp_path."""
import json
import logging
import os
from pathlib import Path

import pytest

from vera import m2_runner, safe_path as sp, trace_viz
from vera.cost_ledger import CostLedger
from vera.safe_path import PathRefused, safe_path
from vera.trace_eval import ui

SECRET = "SENTINEL-VALUE-9f3a"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(sp, "EXTRA_TEST_ROOTS", [])
    sp.reset_refusal_counts()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "allowed"
    r.mkdir()
    return r


def _refused(value, roots, reason, **kw):
    with pytest.raises(PathRefused) as ei:
        safe_path(value, roots, key="K_TEST", **kw)
    assert ei.value.reason == reason and ei.value.key == "K_TEST"
    return ei.value


def test_inside_root_returns_resolved_path(root):
    f = root / "a.json"
    f.write_text("{}")
    got = safe_path(str(f), [root], key="K_TEST")
    assert got == f.resolve() and isinstance(got, Path)


def test_dotdot_and_absolute_outside_refused(root, tmp_path):
    (tmp_path / "out.json").write_text("x")
    _refused(str(root / ".." / "out.json"), [root], "outside_root")
    _refused(str(tmp_path / "out.json"), [root], "outside_root")


def test_symlink_escape_refused_and_symlinked_allowed_root_ok(root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "s.json").write_text("x")
    (root / "link").symlink_to(out)
    _refused(str(root / "link" / "s.json"), [root], "outside_root")
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    (root / "ok.json").write_text("x")
    assert safe_path(str(alias / "ok.json"), [alias], key="K_TEST") == (root / "ok.json").resolve()


def test_prefix_sibling_is_not_inside(tmp_path):
    a, a2 = tmp_path / "app", tmp_path / "app2"
    a.mkdir()
    a2.mkdir()
    _refused(str(a2 / "f"), [a], "outside_root")


def test_file_uri_forms(root, tmp_path):
    f = root / "a b.json"
    f.write_text("x")
    assert safe_path("file://" + str(root).replace(" ", "%20") + "/a%20b.json", [root], key="K") == f.resolve()
    assert safe_path("file://localhost" + str(f).replace(" ", "%20"), [root], key="K") == f.resolve()
    _refused("file:///etc/passwd", [root], "outside_root")
    _refused("file://evilhost" + str(f), [root], "uri_host")
    _refused("file://" + str(root) + "/%2e%2e/out", [root], "uri_encoded")
    _refused("file://" + str(root) + "/a%2Fb", [root], "uri_encoded")
    _refused("file://" + str(root) + "/a%00b", [root], "uri_encoded")
    _refused("file:///C:/Windows/x", [root], "drive_letter")
    _refused("file://" + str(f) + "?x=1", [root], "uri_extra")
    _refused("file:relative", [root], "uri_form")


def test_non_file_scheme_drive_and_unc_refused(root):
    _refused("http://example.invalid/x", [root], "scheme")
    _refused("ftp://x/y", [root], "scheme")
    _refused("C:\\Users\\x", [root], "drive_letter")
    _refused("file://server/share/x", [root], "uri_host")


@pytest.mark.parametrize("bad,reason", [("", "empty"), ("a\x00b", "nul"), (" /x", "whitespace"), ("/x\n", "whitespace"),
                                        ("\t/x", "whitespace"), ("rel/path", "relative"), (None, "not_str"), (5, "not_str")])
def test_malformed_values(root, bad, reason):
    _refused(bad, [root], reason)


def test_no_roots_and_bad_mode(root):
    _refused("/x", [], "no_roots")
    _refused("/x", None, "no_roots")
    _refused(str(root / "x"), [root], "bad_mode", mode="exec")


def test_write_and_append_modes(root, tmp_path):
    _refused(str(root / "nodir" / "f"), [root], "parent_missing", mode="append")
    (root / "real").write_text("x")
    (root / "ln").symlink_to(root / "real")
    _refused(str(root / "ln"), [root], "symlink_final", mode="append")
    _refused(str(root / "ln"), [root], "symlink_final", mode="write")
    out = tmp_path / "out.txt"
    out.write_text("x")
    (root / "esc").symlink_to(out)
    _refused(str(root / "esc"), [root], "symlink_final", mode="append")
    ok = safe_path(str(root / "new.txt"), [root], key="K", mode="append")
    sp.append_text(ok, "a\n")
    sp.append_text(ok, "b\n")
    assert ok.read_text() == "a\nb\n"
    with pytest.raises(OSError):  # O_NOFOLLOW at the final component even if a link appears after the check
        sp.append_text(root / "ln", "z")


def test_must_exist(root):
    _refused(str(root / "none"), [root], "missing", must_exist=True)


def test_counter_log_and_exception_never_hold_the_value(root, caplog):
    caplog.set_level(logging.DEBUG)
    bad = f"/etc/{SECRET}"
    exc = _refused(bad, [root], "outside_root")
    assert sp.refusal_counts()[("K_TEST", "outside_root")] == 1
    _refused(bad, [root], "outside_root")
    assert sp.refusal_counts()[("K_TEST", "outside_root")] == 2
    texts = [str(exc), repr(exc), caplog.text] + [r.getMessage() for r in caplog.records]
    assert any("safe_path refused key=K_TEST reason=outside_root" in t for t in texts)
    assert all(SECRET not in t and "/etc" not in t for t in texts)


def test_invalid_key_is_not_echoed(root):
    with pytest.raises(PathRefused) as ei:
        safe_path("/x", [root], key=f"bad key {SECRET}")
    assert SECRET not in str(ei.value) and ei.value.key == "invalid-key"


def test_tmp_path_is_refused_without_the_hook_and_accepted_with_it(root, tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text("x")
    _refused(str(f), [root], "outside_root")
    monkeypatch.setattr(sp, "EXTRA_TEST_ROOTS", [tmp_path])
    assert safe_path(str(f), [root], key="K") == f.resolve()


def test_hook_is_not_settable_from_the_environment(root, tmp_path, monkeypatch):
    monkeypatch.setenv("VERA_SAFE_PATH_EXTRA_ROOTS", str(tmp_path))
    monkeypatch.setenv("EXTRA_TEST_ROOTS", str(tmp_path))
    _refused(str(tmp_path / "t.json"), [root], "outside_root")


def test_returned_path_not_the_original_string_is_what_callers_get(root, tmp_path):
    (root / "real.json").write_text("{}")
    (root / "l").symlink_to(root / "real.json")
    got = safe_path(str(root / "sub" / ".." / "l"), [root], key="K") if (root / "sub").exists() else safe_path(str(root / "l"), [root], key="K")
    assert got == (root / "real.json").resolve() and str(got) != str(root / "l")


# ---- call sites -------------------------------------------------------------------------------------------------
def test_trace_viz_env_inside_eval_results_is_used_and_outside_falls_back(monkeypatch, caplog):
    inside = trace_viz.RESULTS_ROOT / "trace_eval_v1.json"
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", str(inside))
    assert trace_viz._env_results_path() == inside.resolve()
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", f"/etc/{SECRET}")
    assert trace_viz._env_results_path() == trace_viz.RESULTS_PATH
    assert sp.refusal_counts()[("VERA_TRACE_EVAL_RESULTS", "outside_root")] == 1
    assert SECRET not in caplog.text and "reason=outside_root" in caplog.text
    monkeypatch.setenv("VERA_TRACE_EVAL_RESULTS", str(trace_viz.ROOT / "config" / "api_kickstart.json"))
    assert trace_viz._env_results_path() == trace_viz.RESULTS_PATH  # the repo root is NOT an allowed root, only eval_results


def test_trace_viz_and_ui_unset_env_defaults_do_not_depend_on_cwd(monkeypatch, tmp_path):
    for n in ("VERA_TRACE_EVAL_RESULTS", ui.RESULTS_ENV, ui.VALIDATION_ENV):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.chdir(tmp_path)  # Render-like probe: cwd is not the repo root
    assert trace_viz._env_results_path() == trace_viz.RESULTS_PATH
    assert ui._resolve_paths(None, None) == (ui.RESULTS_PATH, ui.VALIDATION_PATH)
    assert Path(ui.RESULTS_PATH).is_absolute() and Path(ui.RESULTS_PATH).parent == trace_viz.ROOT / "eval_results"
    assert ui.ROOT == trace_viz.ROOT


def test_ui_env_outside_root_falls_back_and_is_counted(monkeypatch, tmp_path):
    monkeypatch.setenv(ui.RESULTS_ENV, str(tmp_path / "r.json"))
    monkeypatch.setenv(ui.VALIDATION_ENV, str(ui.RESULTS_ROOT / "trace_eval_v1_validation.json"))
    r, v = ui._resolve_paths(None, None)
    assert r == ui.RESULTS_PATH and v == str(ui.RESULTS_ROOT / "trace_eval_v1_validation.json")
    assert sp.refusal_counts() == {(ui.RESULTS_ENV, "outside_root"): 1}


def _ledger():
    return CostLedger()


def test_cost_log_first_write_creates_fixed_dir_and_file(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    monkeypatch.setattr(m2_runner, "COST_LOG_ROOT", logs)
    assert not logs.exists()
    monkeypatch.setenv("VERA_COST_LOG", str(logs / "c.jsonl"))
    m2_runner._append_cost_log(1, 1, "q", _ledger())
    assert logs.is_dir() and json.loads((logs / "c.jsonl").read_text())["run_id"] == 1


def test_cost_log_refused_value_skips_append_and_does_not_crash(monkeypatch, tmp_path, caplog):
    logs = tmp_path / "logs"
    monkeypatch.setattr(m2_runner, "COST_LOG_ROOT", logs)
    elsewhere = tmp_path / f"{SECRET}.jsonl"
    monkeypatch.setenv("VERA_COST_LOG", str(elsewhere))
    caplog.set_level(logging.INFO)
    m2_runner._append_cost_log(1, 1, "q", _ledger())  # must not raise
    assert not elsewhere.exists()
    assert sp.refusal_counts()[("VERA_COST_LOG", "outside_root")] == 1
    assert SECRET not in caplog.text and "m2 cost:" in caplog.text  # the INFO summary line still appears


def test_cost_log_unset_env_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(m2_runner, "COST_LOG_ROOT", tmp_path / "logs")
    monkeypatch.delenv("VERA_COST_LOG", raising=False)
    m2_runner._append_cost_log(1, 1, "q", _ledger())
    assert not (tmp_path / "logs").exists() and sp.refusal_counts() == {}


def test_cost_log_symlinked_final_component_refused(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    target = tmp_path / "victim.txt"
    target.write_text("keep")
    (logs / "c.jsonl").symlink_to(target)
    monkeypatch.setattr(m2_runner, "COST_LOG_ROOT", logs)
    monkeypatch.setenv("VERA_COST_LOG", str(logs / "c.jsonl"))
    m2_runner._append_cost_log(1, 1, "q", _ledger())
    assert target.read_text() == "keep"
    assert sp.refusal_counts()[("VERA_COST_LOG", "symlink_final")] == 1


def test_cost_log_root_is_anchored_on_the_file_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert m2_runner.COST_LOG_ROOT == Path(m2_runner.__file__).resolve().parent.parent / "logs"
    assert os.path.isabs(m2_runner.COST_LOG_ROOT)


# ---- VERA_FULLDOC_BUNDLE_DIR (fulldoc_corpus.get_corpus) -------------------------------------------------------------
def _fd(monkeypatch, tmp_path):
    from vera import fulldoc_corpus as fc
    from tests.fulldoc_bundle_synth import synth_cfg, write_bundle
    cfg = synth_cfg()
    cfg.setdefault("fulldoc", {})["enabled"] = True
    root = tmp_path / "bundleroot"
    root.mkdir()
    monkeypatch.setattr(fc, "BUNDLE_ROOT", root)
    fc.reset_for_tests()
    return fc, cfg, root, write_bundle


def test_bundle_root_parity_with_test_runner():
    import importlib.util
    from vera import fulldoc_corpus as fc
    spec = importlib.util.spec_from_file_location("vtr", Path(fc.__file__).resolve().parent.parent / "scripts" / "vera_test_runner.py")
    vtr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vtr)
    assert tuple(fc.BUNDLE_ROOT_REL) == tuple(vtr.RO_BUNDLE_REL)
    assert fc.BUNDLE_ROOT == fc.ROOT.parent.joinpath(*vtr.RO_BUNDLE_REL)


def test_bundle_inside_root_loads_without_the_hook(monkeypatch, tmp_path):
    fc, cfg, root, write_bundle = _fd(monkeypatch, tmp_path)
    b = write_bundle(root / "b")
    c = fc.get_corpus(cfg, {cfg["bundle_dir_env"]: str(b)})
    assert c is not None and c.status == fc.STATUS_OK and sp.refusal_counts() == {}


def test_bundle_outside_root_is_r_bundle_path_counted_not_cached_value_not_logged(monkeypatch, tmp_path, caplog):
    fc, cfg, root, write_bundle = _fd(monkeypatch, tmp_path)
    out = write_bundle(tmp_path / f"{SECRET}-out")
    caplog.set_level(logging.INFO)
    env = {cfg["bundle_dir_env"]: str(out)}
    c = fc.get_corpus(cfg, env)
    assert c.status == fc.STATUS_UNUSABLE and [u["code"] if isinstance(u, dict) else u.code for u in c.unusable] == ["R-BUNDLE-PATH"]
    assert sp.refusal_counts() == {(cfg["bundle_dir_env"], "outside_root"): 1}
    assert SECRET not in caplog.text and "reason=outside_root" in caplog.text
    assert not fc._CACHE  # nothing cached for a refused value
    fc.get_corpus(cfg, env)
    assert not fc._CACHE and sp.refusal_counts()[(cfg["bundle_dir_env"], "outside_root")] == 2


def test_bundle_symlink_escaping_root_is_refused(monkeypatch, tmp_path):
    fc, cfg, root, write_bundle = _fd(monkeypatch, tmp_path)
    out = write_bundle(tmp_path / "out")
    (root / "lnk").symlink_to(out)
    c = fc.get_corpus(cfg, {cfg["bundle_dir_env"]: str(root / "lnk")})
    assert c.status == fc.STATUS_UNUSABLE and sp.refusal_counts() == {(cfg["bundle_dir_env"], "outside_root"): 1}


def test_bundle_value_with_whitespace_is_refused_not_stripped(monkeypatch, tmp_path):
    fc, cfg, root, write_bundle = _fd(monkeypatch, tmp_path)
    b = write_bundle(root / "b")
    c = fc.get_corpus(cfg, {cfg["bundle_dir_env"]: str(b) + " "})
    assert c.status == fc.STATUS_UNUSABLE and sp.refusal_counts() == {(cfg["bundle_dir_env"], "whitespace"): 1}
    assert fc.get_corpus(cfg, {cfg["bundle_dir_env"]: "   "}) is None  # blank = unset, unchanged


def test_bundle_both_sources_still_gives_r_source_first(monkeypatch, tmp_path):
    fc, cfg, root, write_bundle = _fd(monkeypatch, tmp_path)
    env = {cfg["bundle_dir_env"]: "/etc", fc._db_run_env(cfg): "7"}
    c = fc.get_corpus(cfg, env)
    assert [u["code"] if isinstance(u, dict) else u.code for u in c.unusable] == ["R-SOURCE"] and sp.refusal_counts() == {}


def test_bundle_path_code_has_an_outcome_mapping():
    from vera import research_outcome
    assert research_outcome.__dict__  # module imports
    src = Path(research_outcome.__file__).read_text(encoding="utf-8")
    assert '"R-BUNDLE-PATH": "provenance_rejected"' in src
