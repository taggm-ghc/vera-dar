"""D-B5-1 regression tests: release_gate (a2) binds owner-run catalog evidence to the commit under test.

Synthetic evidence only: every evidence file, sidecar and fake SHA is written under tmp_path. The real repo bytes of
scripts/live_catalog_check.py and scripts/check_catalog_evidence_freshness.py are BYTE-COPIED into tmp_path/scripts
(shutil.copyfile, never imported, never run), so the checker's approved C1 hash and the gate's checker pin hold
(plan r2 RC-1). No DB, no network, no git, no subprocess: git reads are replaced by monkeypatching
release_gate._candidate_state or release_gate.subprocess.run with fakes; the suite runner check_tests is faked.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_gate_b51", ROOT / "scripts" / "release_gate.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

CAND = "c" * 40                                    # fake candidate SHA (gate HEAD)
OTHER = "d" * 40                                   # a different fake SHA
LEGACY_SHA = "9c3dd5652644cf79ab9918ef5b5bcdab3497b1bb"  # recorded fallback id (a public commit id, not evidence)
A2 = "(a2) live catalog evidence (required, owner-run)"


@pytest.fixture(autouse=True)
def _synthetic_environ(monkeypatch):
    """R-5 secret hygiene: every test sees a fully synthetic os.environ, so no real variable name or value can be
    echoed into a log by an assertion message (monkeypatch.setenv in a test then writes into this dict)."""
    monkeypatch.setattr(os, "environ", {"PATH": "/fake/bin", "HOME": "/fake/home"})


def _approved_script_sha() -> str:
    return hashlib.sha256((ROOT / g.CATALOG_SCRIPT).read_bytes()).hexdigest()


def _body(script_sha: str | None = None, count: str = "checked 7 (table, column) pairs; 0 missing") -> str:
    return f"live_catalog_check script_sha256={script_sha or _approved_script_sha()}\n{count}\nRESULT: PASS\n"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    for rel in (g.CATALOG_SCRIPT, g.CATALOG_CHECKER):
        shutil.copyfile(ROOT / rel, r / rel)
    return r


@pytest.fixture
def evdir(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    return d


def _sidecar_fields(ev: Path, sha: str = CAND, **over) -> dict[str, str]:
    f = {
        "candidate_sha": sha, "head_after": sha, "script_sha256": _approved_script_sha(), "evidence_file": ev.name,
        "evidence_sha256": hashlib.sha256(ev.read_bytes()).hexdigest(), "exit_code": "0", "worktree_clean": "1",
        "run_start_utc": "2026-10-17T16:51:13Z", "run_end_utc": "2026-10-17T16:51:14Z",
    }
    f.update(over)
    return f


def _write_sidecar(ev: Path, fields: dict[str, str] | None = None, header: str = "vera-catalog-evidence-sidecar v1",
                   drop: tuple[str, ...] = (), path: Path | None = None, **over) -> Path:
    fields = fields if fields is not None else _sidecar_fields(ev, **over)
    lines = [header] + [f"{k}={v}" for k, v in fields.items() if k not in drop]
    sc = path or ev.with_name(ev.name + ".sidecar")
    sc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sc


def _bound(evdir: Path, sha: str = CAND, body: str | None = None, name: str | None = None, sidecar: bool = True,
           **over) -> Path:
    ev = evdir / (name or f"catalog-evidence-{sha[:12]}.txt")
    ev.write_text(body if body is not None else _body(), encoding="utf-8")
    if sidecar:
        _write_sidecar(ev, sha=sha, **over)
    return ev


def _outcome(ev, repo, cand=CAND):
    return g.catalog_evidence_outcome(ev, repo, cand)


# ------------------------------------------------------------------ pin and happy path

def test_checker_pin_matches_repo_checker():
    assert hashlib.sha256((ROOT / g.CATALOG_CHECKER).read_bytes()).hexdigest() == g.CATALOG_CHECKER_SHA256


def test_bound_evidence_passes_with_prefix_only_note(repo, evdir):
    ev = _bound(evdir)
    o, note = _outcome(ev, repo)
    assert o == "PASSED" and CAND[:12] in note and CAND not in note
    assert g.check_live_catalog_evidence(ev, repo, CAND) == []


# ------------------------------------------------------------------ exact SHA / candidate-to-evidence binding

def test_evidence_for_other_sha_fails(repo, evdir):
    ev = _bound(evdir, sha=OTHER)                 # file + sidecar consistent, but for OTHER
    o, why = _outcome(ev, repo, CAND)
    assert o == "STALE" and "filename SHA" in why


def test_file_named_for_candidate_but_sidecar_for_other_sha_fails(repo, evdir):
    ev = _bound(evdir, sha=CAND, candidate_sha=OTHER)
    o, why = _outcome(ev, repo)
    assert o == "STALE" and "sidecar candidate_sha" in why


@pytest.mark.parametrize("cand", [CAND[:12], CAND[:39], CAND.upper()[:40] + "x", "", "g" * 40])
def test_candidate_must_be_full_40_hex(repo, evdir, cand):
    ev = _bound(evdir)
    o, why = g.catalog_evidence_outcome(ev, repo, cand)
    assert o == "STALE"


def test_candidate_none_fails_closed(repo, evdir):
    ev = _bound(evdir)
    o, why = g.catalog_evidence_outcome(ev, repo, None)
    assert (o, why) == ("STALE", "cannot bind evidence: candidate commit unknown")
    out = g.check_live_catalog_evidence(ev, repo, None, "HEAD moved during the test suite")
    assert out == ["cannot bind evidence: candidate commit unknown (HEAD moved during the test suite)"]


# ------------------------------------------------------------------ legacy evidence and byte-identical content

def test_legacy_name_rejected_for_new_candidate(repo, evdir):
    ev = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    o, why = _outcome(ev, repo, CAND)
    assert o == "STALE" and "legacy" in why and LEGACY_SHA[:12] in why


def test_legacy_name_for_fallback_sha_requires_recorded_bytes(repo, evdir):
    ev = _bound(evdir, name="catalog-evidence.txt", sidecar=False)   # synthetic bytes, not the recorded C4 bytes
    o, why = _outcome(ev, repo, LEGACY_SHA)
    assert o == "STALE" and "legacy evidence sha256" in why


def test_identical_content_copied_under_candidate_name_without_sidecar_fails(repo, evdir):
    legacy = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    copy = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    shutil.copyfile(legacy, copy)
    assert legacy.read_bytes() == copy.read_bytes()
    o, why = _outcome(copy, repo)
    assert o == "STALE" and "sidecar missing" in why


def test_identical_content_with_sidecar_for_other_sha_fails(repo, evdir):
    legacy = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    copy = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    shutil.copyfile(legacy, copy)
    _write_sidecar(copy, sha=OTHER)
    o, why = _outcome(copy, repo)
    assert o == "STALE" and "sidecar candidate_sha" in why


def test_identical_content_with_wrong_sidecar_fields_fails(repo, evdir):
    legacy = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    copy = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    shutil.copyfile(legacy, copy)
    _write_sidecar(copy, evidence_file="catalog-evidence.txt")
    assert _outcome(copy, repo)[0] == "STALE"


def test_residual_F1_identical_content_with_consistent_forged_sidecar_passes(repo, evdir):
    """Characterisation of the ACCEPTED residual (DV-3/F1), not a desired property: fresh evidence for an unchanged
    catalog is byte-identical to old evidence, so content cannot be rejected; a locally written, internally consistent
    but unsigned sidecar binds it. Only signing (OD-5 B) or the slot-7 review catches a deliberate forger."""
    legacy = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    copy = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    shutil.copyfile(legacy, copy)
    _write_sidecar(copy)
    assert _outcome(copy, repo)[0] == "PASSED"


# ------------------------------------------------------------------ evidence content hash and sidecar metadata

def test_evidence_edited_after_sidecar_fails(repo, evdir):
    ev = _bound(evdir)
    ev.write_text(_body(count="checked 8 (table, column) pairs; 0 missing"), encoding="utf-8")
    o, why = _outcome(ev, repo)
    assert o == "STALE" and "evidence_sha256" in why


_FUTURE = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize("over, drop, header", [
    ({"exit_code": "1"}, (), None),
    ({"exit_code": "2"}, (), None),
    ({"exit_code": ""}, (), None),
    ({}, ("exit_code",), None),                                   # missing exit status
    ({"worktree_clean": "0"}, (), None),
    ({"head_after": OTHER}, (), None),
    ({"script_sha256": "0" * 64}, (), None),
    ({"evidence_sha256": "0" * 64}, (), None),
    ({"evidence_file": "catalog-evidence-cccccccccccc.txt.bak"}, (), None),
    ({"run_end_utc": _FUTURE, "run_start_utc": _FUTURE}, (), None),
    ({"run_end_utc": "2026-10-17T16:51:12Z"}, (), None),          # end before start
    ({"run_start_utc": "17 Oct 2026"}, (), None),
    ({}, ("run_start_utc", "run_end_utc"), None),
    ({}, (), "vera-catalog-evidence-sidecar v0"),
    ({}, (), ""),
])
def test_sidecar_metadata_rejections(repo, evdir, over, drop, header):
    ev = _bound(evdir, sidecar=False)
    _write_sidecar(ev, header="vera-catalog-evidence-sidecar v1" if header is None else header, drop=drop, **over)
    o, why = _outcome(ev, repo)
    assert o == "STALE", why


@pytest.mark.parametrize("keep", [0, 1, 3, 6])
def test_truncated_sidecar_fails_closed(repo, evdir, keep):
    ev = _bound(evdir)
    sc = ev.with_name(ev.name + ".sidecar")
    sc.write_text("\n".join(sc.read_text(encoding="utf-8").splitlines()[:keep]) + "\n", encoding="utf-8")
    assert _outcome(ev, repo)[0] == "STALE"


def test_sidecar_byte_truncated_mid_line_fails(repo, evdir):
    ev = _bound(evdir)
    sc = ev.with_name(ev.name + ".sidecar")
    data = sc.read_bytes()
    sc.write_bytes(data[: len(data) - 30])
    assert _outcome(ev, repo)[0] == "STALE"


def test_sidecar_duplicate_key_and_non_utf8_fail(repo, evdir):
    ev = _bound(evdir)
    sc = ev.with_name(ev.name + ".sidecar")
    sc.write_text(sc.read_text(encoding="utf-8") + f"candidate_sha={CAND}\n", encoding="utf-8")
    assert _outcome(ev, repo)[0] == "STALE"
    sc.write_bytes(b"\xff\xfe\x00bad")
    assert _outcome(ev, repo)[0] == "STALE"


@pytest.mark.parametrize("body", [
    "",                                                                          # empty
    "live_catalog_check script_sha256=@\n",                                      # truncated: no count, no RESULT
    "live_catalog_check script_sha256=@\nchecked 7 (table, column) pairs; 0 missing\n",      # no RESULT
    "live_catalog_check script_sha256=@\nRESULT: PASS\n",                        # no count line
    "live_catalog_check script_sha256=@\nchecked 7 (table, column) pairs; 2 missing\nRESULT: PASS\n",
    "live_catalog_check script_sha256=@\nchecked 7 (table, column) pairs; 0 missing\nRESULT: FAIL\n",
    "live_catalog_check script_sha256=@\nchecked 7 (table, column) pairs; 0 missing\nRESULT: PASS\nRESULT: PASS\n",
    "script_sha256=@\nchecked 7 (table, column) pairs; 0 missing\nRESULT: PASS\n",   # bare header form (AT-15)
    "checked 7 (table, column) pairs; 0 missing\nRESULT: PASS\nlive_catalog_check script_sha256=@\n",
])
def test_incomplete_or_failed_inspection_body_never_passes(repo, evdir, body):
    ev = _bound(evdir, body=body.replace("@", _approved_script_sha()))
    o, why = _outcome(ev, repo)
    assert o in ("STALE", "FAILED", "MISSING"), (o, why)


def test_non_utf8_evidence_is_a_clean_stale_not_a_traceback(repo, evdir):
    ev = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    ev.write_bytes(b"\xff\xfe binary")
    assert _outcome(ev, repo) == ("STALE", "evidence is not UTF-8 text")


def test_evidence_path_is_a_directory(repo, evdir):
    assert _outcome(evdir, repo)[0] == "MISSING"


def test_bare_ev_txt_name_now_fails(repo, evdir):
    ev = _bound(evdir, name="ev.txt")
    o, why = _outcome(ev, repo)
    assert o == "STALE" and "filename" in why


# ------------------------------------------------------------------ script hash

def test_repo_script_not_the_approved_c1_hash_fails(repo, evdir):
    (repo / g.CATALOG_SCRIPT).write_text("print('changed')\n")
    changed = hashlib.sha256((repo / g.CATALOG_SCRIPT).read_bytes()).hexdigest()
    ev = _bound(evdir, body=_body(script_sha=changed), script_sha256=changed)  # evidence matches the CURRENT script
    o, why = _outcome(ev, repo)
    assert o == "STALE" and "catalog script hash" in why


def test_evidence_for_other_script_version_fails(repo, evdir):
    ev = _bound(evdir, body=_body(script_sha="0" * 64))
    assert _outcome(ev, repo) == ("STALE", "evidence is for a different version of scripts/live_catalog_check.py")


# ------------------------------------------------------------------ checker integrity, fail closed on errors

def test_checker_missing_fails_closed(repo, evdir):
    ev = _bound(evdir)
    (repo / g.CATALOG_CHECKER).unlink()
    o, why = _outcome(ev, repo)
    assert o == "STALE" and why.startswith("binding checker unavailable")


def test_checker_weakened_is_refused_by_pin(repo, evdir):
    ev = _bound(evdir, sha=OTHER)                                  # would be rejected by the real checker
    (repo / g.CATALOG_CHECKER).write_text(
        "class EvidenceError(Exception):\n    pass\n\ndef check(*a, **k):\n    return 'always'\n")
    o, why = _outcome(ev, repo, CAND)
    assert (o, why) == ("STALE", "binding checker unavailable (ImportError)")


def test_checker_symlink_refused(repo, evdir, tmp_path):
    ev = _bound(evdir)
    real = tmp_path / "elsewhere.py"
    shutil.move(repo / g.CATALOG_CHECKER, real)
    os.symlink(real, repo / g.CATALOG_CHECKER)
    assert _outcome(ev, repo) == ("STALE", "binding checker unavailable (FileNotFoundError)")


def test_checker_syntax_error_with_matching_pin_fails_closed(repo, evdir, monkeypatch):
    ev = _bound(evdir)
    bad = b"def check(:\n"
    (repo / g.CATALOG_CHECKER).write_bytes(bad)
    monkeypatch.setattr(g, "CATALOG_CHECKER_SHA256", hashlib.sha256(bad).hexdigest())
    assert _outcome(ev, repo) == ("STALE", "binding checker unavailable (SyntaxError)")


class _FakeChecker:
    class EvidenceError(Exception):
        pass

    def __init__(self, result):
        self.result = result

    def check(self, *a, **k):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.mark.parametrize("result, want", [
    (ValueError("x /secret/path"), "binding check error (ValueError)"),
    (TypeError("x"), "binding check error (TypeError)"),
    (None, "binding check error (no note)"),
    ("", "binding check error (no note)"),
    (1, "binding check error (no note)"),
])
def test_unexpected_checker_behaviour_fails_closed_without_message(repo, evdir, monkeypatch, result, want):
    ev = _bound(evdir)
    monkeypatch.setattr(g, "_load_checker", lambda root: _FakeChecker(result))
    assert _outcome(ev, repo) == ("STALE", want)


def test_checker_without_api_fails_closed(repo, evdir, monkeypatch):
    ev = _bound(evdir)
    monkeypatch.setattr(g, "_load_checker", lambda root: object())
    assert _outcome(ev, repo) == ("STALE", "binding checker unavailable (AttributeError)")


def test_wrapper_catches_any_outcome_crash(repo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("leaky /path detail")
    monkeypatch.setattr(g, "catalog_evidence_outcome", boom)
    assert g.check_live_catalog_evidence("x", repo, CAND) == ["binding check error (RuntimeError)"]


def test_symlinked_evidence_and_sidecar_refused(repo, evdir, tmp_path):
    ev = _bound(evdir)
    real_ev = tmp_path / "real-ev.txt"
    shutil.move(ev, real_ev)
    os.symlink(real_ev, ev)
    assert _outcome(ev, repo)[0] == "STALE"
    ev.unlink()
    shutil.move(real_ev, ev)
    sc = ev.with_name(ev.name + ".sidecar")
    real_sc = tmp_path / "real.sidecar"
    shutil.move(sc, real_sc)
    os.symlink(real_sc, sc)
    assert _outcome(ev, repo)[0] == "STALE"


# ------------------------------------------------------------------ no subprocess inside (a2) (RC-5)

def test_a2_source_has_no_banned_text():
    fn = ast.parse(textwrap.dedent(inspect.getsource(g.catalog_evidence_outcome))).body[0]
    fn.body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
    code = ast.unparse(fn)
    assert "subprocess" not in code and "VERA_GATE_NONCE" not in code and "pytest" not in code
    for fn2 in (g._load_checker,):
        assert "subprocess" not in inspect.getsource(fn2)


def test_a2_outcomes_unchanged_when_process_spawning_raises(repo, evdir, monkeypatch):
    good = _bound(evdir)
    other = _bound(evdir, sha=OTHER)
    expected = (_outcome(good, repo), _outcome(other, repo))

    def refuse(*a, **k):
        raise AssertionError("process spawn attempted inside (a2)")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    assert (_outcome(good, repo), _outcome(other, repo)) == expected
    assert expected[0][0] == "PASSED" and expected[1][0] == "STALE"


# ------------------------------------------------------------------ MISSING text (AT-22)

def test_missing_text_names_bound_file_and_sidecar():
    for p in (None, "/nonexistent/none.txt"):
        o, why = g.catalog_evidence_outcome(p, ROOT, CAND)
        assert o == "MISSING"
        assert "catalog-evidence-<sha12>.txt" in why and ".sidecar" in why
        assert "--catalog-evidence" in why and "never connects" in why
        assert "pytest" not in why and "subprocess" not in why


# ------------------------------------------------------------------ age limit (DV-6, AT-21)

def test_age_limit_ships_unset_and_boundary_works_when_set(repo, evdir):
    assert g.CATALOG_EVIDENCE_MAX_AGE_HOURS is None
    ev = _bound(evdir)
    checker = g._load_checker(repo)
    end = datetime(2026, 10, 17, 16, 51, 14, tzinfo=timezone.utc)
    limit = 2.0
    ok = checker.check(CAND, ev, script=repo / g.CATALOG_SCRIPT, max_age_hours=limit,
                       now=end + timedelta(hours=limit) - timedelta(seconds=1))
    assert CAND[:12] in ok
    with pytest.raises(checker.EvidenceError):
        checker.check(CAND, ev, script=repo / g.CATALOG_SCRIPT, max_age_hours=limit,
                      now=end + timedelta(hours=limit) + timedelta(seconds=1))


# ------------------------------------------------------------------ candidate state (git reads, faked)

class _CP:
    def __init__(self, rc, out):
        self.returncode, self.stdout, self.stderr = rc, out, ""


def _blob_id(data: bytes, algo=hashlib.sha1) -> str:
    return algo(b"blob %d\0" % len(data) + data).hexdigest()


def _make_tree(root: Path, algo=hashlib.sha1) -> bytes:
    """Synthetic tracked files plus the `git ls-tree -r -z --full-tree` bytes for them (computed here, no git)."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    files = {"a.txt": b"alpha\n", "bin/run.sh": b"#!/bin/sh\necho hi\n"}
    for rel, data in files.items():
        (root / rel).write_bytes(data)
    os.chmod(root / "a.txt", 0o644)
    os.chmod(root / "bin/run.sh", 0o755)
    if not (root / "link").is_symlink():
        os.symlink("a.txt", root / "link")
    rows = [f"100644 blob {_blob_id(files['a.txt'], algo)}\ta.txt",
            f"100755 blob {_blob_id(files['bin/run.sh'], algo)}\tbin/run.sh",
            f"120000 blob {_blob_id(b'a.txt', algo)}\tlink"]
    return ("\0".join(rows) + "\0").encode()


@pytest.fixture
def fixed_git(monkeypatch):
    monkeypatch.setattr(g, "_git_exe", lambda: "/usr/bin/git")


def _fake_git(root, head=CAND, toplevel=None, status="", rc=0, seen=None, ls_files="H a.txt\0", tree=None, tree_rc=0):
    tree_bytes = _make_tree(Path(root)) if tree is None else tree

    def run(cmd, **kw):
        if seen is not None:
            seen.append((cmd, kw))
        if "rev-parse" in cmd:
            return _CP(rc, f"{toplevel if toplevel is not None else root}\n{head}\n")
        if "ls-files" in cmd:
            return _CP(0, ls_files)
        if "ls-tree" in cmd:
            return _CP(tree_rc, tree_bytes)
        return _CP(0, status)
    return run


def test_candidate_state_clean(tmp_path, monkeypatch, fixed_git):
    seen = []
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/elsewhere")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, seen=seen))
    assert g._candidate_state(tmp_path) == (CAND, True)
    allowed = {"GIT_OPTIONAL_LOCKS", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL"}
    for cmd, kw in seen:
        assert isinstance(cmd, list) and cmd[0] == "/usr/bin/git" and kw["shell"] is False
        assert kw["timeout"] == g.GIT_TIMEOUT_S
        assert cmd[1] == "--no-replace-objects" and "core.fsmonitor=false" in cmd
        assert "core.untrackedCache=false" in cmd and str(tmp_path) in cmd
        env = kw["env"]
        assert env["GIT_OPTIONAL_LOCKS"] == "0" and env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert not any(k.startswith("GIT_") and k not in allowed for k in env)
    subcmds = [next(x for x in c[0] if x in ("rev-parse", "ls-files", "ls-tree", "status")) for c in seen]
    assert subcmds == ["rev-parse", "ls-files", "ls-tree", "status"]
    assert seen[1][0][-2:] == ["-v", "-z"] and seen[2][0][-1] == CAND
    assert seen[-1][0][-1] == "--untracked-files=all"


def test_candidate_state_dirty_tree(tmp_path, monkeypatch, fixed_git):
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, status="?? new.py\n"))
    assert g._candidate_state(tmp_path) == (CAND, False)


@pytest.mark.parametrize("ls_files", ["S a.txt\0", "h a.txt\0", "H a.txt\0S bin/run.sh\0", "M a.txt\0", ""])
def test_candidate_state_index_flags_make_tree_unclean(tmp_path, monkeypatch, fixed_git, ls_files):
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, ls_files=ls_files))
    assert g._candidate_state(tmp_path) == (CAND, False)


def test_candidate_state_hidden_edit_detected_without_index(tmp_path, monkeypatch, fixed_git):
    tree = _make_tree(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"alpha EDITED\n")    # status (index-based) still reports nothing
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, tree=tree))
    assert g._candidate_state(tmp_path) == (CAND, False)


def test_candidate_state_ls_tree_failure_unclean(tmp_path, monkeypatch, fixed_git):
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, tree_rc=128))
    assert g._candidate_state(tmp_path) == (CAND, False)


def test_candidate_state_no_git_on_fixed_path(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_git_exe", lambda: None)
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    assert g._candidate_state(tmp_path) == (None, False)
    monkeypatch.setenv("PATH", str(tmp_path))                 # caller PATH is ignored by the real resolver
    assert g.GIT_SEARCH_PATH == "/usr/bin:/bin"


def test_candidate_state_rejects_nested_repo_toplevel(tmp_path, monkeypatch, fixed_git):
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, toplevel=tmp_path.parent))
    assert g._candidate_state(tmp_path) == (None, False)


@pytest.mark.parametrize("head, rc", [("c" * 12, 0), ("HEAD", 0), ("", 0), (CAND, 128)])
def test_candidate_state_bad_rev_parse(tmp_path, monkeypatch, fixed_git, head, rc):
    monkeypatch.setattr(g.subprocess, "run", _fake_git(tmp_path, head=head, rc=rc))
    assert g._candidate_state(tmp_path) == (None, False)


@pytest.mark.parametrize("ls, ok", [
    ("H a\0H b\0", True), ("H a\0", True), ("", False), ("h a\0", False), ("S a\0", False),
    ("H a\0S b\0", False), ("K a\0", False), ("Ha\0", False),
])
def test_index_flags_table(ls, ok):
    assert g._index_flags_ok(ls) is ok


def test_tree_matches_synthetic_tree(tmp_path):
    assert g._tree_matches(tmp_path, _make_tree(tmp_path)) is True
    assert g._tree_matches(tmp_path, _make_tree(tmp_path, algo=hashlib.sha256)) is True   # sha256 repositories


@pytest.mark.parametrize("mutate", ["edit", "delete", "exec_off", "exec_on", "relink", "unlink_to_file"])
def test_tree_matches_detects_disk_changes(tmp_path, mutate):
    tree = _make_tree(tmp_path)
    if mutate == "edit":
        (tmp_path / "a.txt").write_bytes(b"alphA\n")
    elif mutate == "delete":
        (tmp_path / "bin/run.sh").unlink()
    elif mutate == "exec_off":
        os.chmod(tmp_path / "bin/run.sh", 0o644)
    elif mutate == "exec_on":
        os.chmod(tmp_path / "a.txt", 0o755)
    elif mutate == "relink":
        (tmp_path / "link").unlink()
        os.symlink("bin/run.sh", tmp_path / "link")
    else:
        (tmp_path / "link").unlink()
        (tmp_path / "link").write_bytes(b"a.txt")
    assert g._tree_matches(tmp_path, tree) is False


@pytest.mark.parametrize("extra", [
    b"160000 commit " + b"e" * 40 + b"\tsub\0",       # submodule: not verifiable, fail closed
    b"100644 blob " + b"e" * 40 + b"\0",              # no tab / path
    b"040000 tree " + b"e" * 40 + b"\tdir\0",
    b"100644 blob " + b"e" * 39 + b"\tx\0",           # odd oid length
    b"garbage\0",
])
def test_tree_matches_rejects_unverifiable_entries(tmp_path, extra):
    assert g._tree_matches(tmp_path, _make_tree(tmp_path) + extra) is False


def test_tree_matches_empty_listing_fails(tmp_path):
    assert g._tree_matches(tmp_path, b"") is False


@pytest.mark.parametrize("exc", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10), OSError("x")])
def test_candidate_state_git_errors_fail_closed(tmp_path, monkeypatch, fixed_git, exc):
    def run(*a, **k):
        raise exc
    monkeypatch.setattr(g.subprocess, "run", run)
    assert g._candidate_state(tmp_path) == (None, False)


@pytest.mark.parametrize("pre, post, expect, want", [
    ((CAND, True), (CAND, True), None, (CAND, "")),
    ((CAND, True), (CAND, True), CAND.upper(), (CAND, "")),
    ((CAND, True), (OTHER, True), None, (None, "HEAD moved during the test suite")),
    ((CAND, False), (CAND, True), None, (None, "working tree not clean before the test suite")),
    ((CAND, True), (CAND, False), None, (None, "working tree not clean after the test suite")),
    ((None, False), (CAND, True), None, (None, "HEAD of the gate worktree could not be read")),
    ((CAND, True), (None, False), None, (None, "HEAD of the gate worktree could not be read")),
    ((CAND, True), (CAND, True), OTHER, (None, "HEAD is not the --expect-candidate-sha commit")),
    ((CAND, True), (CAND, True), "", (None, "HEAD is not the --expect-candidate-sha commit")),
])
def test_bound_candidate_table(pre, post, expect, want):
    assert g._bound_candidate(pre, post, expect) == want


# ------------------------------------------------------------------ run_offline_checks / main (all other checks faked)

@pytest.fixture
def gate_env(repo, monkeypatch):
    """Point the gate at the tmp repo and fake every check except (a2)."""
    monkeypatch.setattr(g, "ROOT", repo)
    monkeypatch.setattr(g, "check_tests", lambda root: [])
    for name in ("check_frozen", "check_no_env_in_widgets", "check_no_unsafe_html"):
        monkeypatch.setattr(g, name, lambda root: [])

    def no_in_process(root):
        raise AssertionError("candidate-code check ran in the gate process")
    monkeypatch.setattr(g, "check_public_default", no_in_process)
    monkeypatch.setattr(g, "check_live_scope", no_in_process)
    monkeypatch.setattr(g, "_run_child_check", lambda root, name: [])
    monkeypatch.setattr(g, "_isolated", lambda: True)
    states = []
    monkeypatch.setattr(g, "_candidate_state", lambda root: states.pop(0))
    return states


def test_offline_checks_without_tests_make_no_git_call(repo, monkeypatch):
    def boom(root):
        raise AssertionError("git read without the suite")
    monkeypatch.setattr(g, "_candidate_state", boom)
    for name in ("check_frozen", "check_no_env_in_widgets", "check_no_unsafe_html", "check_public_default"):
        monkeypatch.setattr(g, name, lambda root: [])
    res = g.run_offline_checks(repo, run_tests=False)
    assert A2 not in res and all(not v for v in res.values())


def test_main_passes_with_bound_evidence_and_stable_clean_head(gate_env, evdir, capsys):
    ev = _bound(evdir)
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(ev), "--expect-candidate-sha", CAND]) == 0
    out = capsys.readouterr().out
    assert f"PASS  {A2}" in out and "bound: evidence bound to " + CAND[:12] in out and "RESULT: PASS" in out


@pytest.mark.parametrize("states, argv_extra, needle", [
    ([(CAND, True), (OTHER, True)], [], "HEAD moved during the test suite"),
    ([(CAND, True), (CAND, False)], [], "working tree not clean after the test suite"),
    ([(CAND, False), (CAND, True)], [], "working tree not clean before the test suite"),
    ([(None, False), (None, False)], [], "could not be read"),
    ([(CAND, True), (CAND, True)], ["--expect-candidate-sha", OTHER], "--expect-candidate-sha"),
])
def test_main_fails_when_candidate_cannot_be_bound(gate_env, evdir, capsys, states, argv_extra, needle):
    ev = _bound(evdir)
    gate_env[:] = states
    assert g.main(["--catalog-evidence", str(ev)] + argv_extra) == 1
    out = capsys.readouterr().out
    assert f"FAIL  {A2}" in out and needle in out and out.strip().splitlines()[-1].startswith("RESULT: FAIL")


def test_main_fails_on_legacy_evidence_for_candidate(gate_env, evdir, capsys):
    ev = _bound(evdir, name="catalog-evidence.txt", sidecar=False)
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(ev)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL  {A2}" in out and "legacy" in out and "RESULT: FAIL" in out


def test_main_checker_crash_still_prints_result_fail(gate_env, evdir, capsys, monkeypatch):
    ev = _bound(evdir)
    gate_env[:] = [(CAND, True), (CAND, True)]
    monkeypatch.setattr(g, "_load_checker", lambda root: _FakeChecker(ValueError("boom")))
    assert g.main(["--catalog-evidence", str(ev)]) == 1
    out = capsys.readouterr().out
    assert "binding check error (ValueError)" in out and "boom" not in out
    assert out.strip().splitlines()[-1].startswith("RESULT: FAIL")


def test_main_skip_tests_still_fails(gate_env, evdir):
    assert g.main(["--skip-tests", "--catalog-evidence", str(_bound(evdir))]) == 1


# ================================================================== REVISE round 1 (SEC G-1, G-2, G-3, G-8; V Q-1)
import io  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402


# ------------------------------------------------------------------ G-1: no fail-open by exit code

@pytest.mark.parametrize("exc", [SystemExit(0), SystemExit(None), KeyboardInterrupt(), RuntimeError("x")])
def test_main_turns_any_abort_into_result_fail(gate_env, evdir, capsys, monkeypatch, exc):
    gate_env[:] = [(CAND, True), (CAND, True)]

    def boom(root):
        raise exc
    monkeypatch.setattr(g, "check_frozen", boom)
    assert g.main(["--catalog-evidence", str(_bound(evdir))]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == f"RESULT: FAIL (gate aborted: {type(exc).__name__})"


def test_main_usage_error_is_result_fail(capsys):
    assert g.main(["--no-such-flag"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: FAIL (gate aborted: SystemExit)"


def test_main_refuses_non_isolated_interpreter(gate_env, evdir, capsys, monkeypatch):
    monkeypatch.setattr(g, "_isolated", lambda: False)
    monkeypatch.setattr(g, "check_tests", lambda root: (_ for _ in ()).throw(AssertionError("suite ran")))
    assert g.main(["--catalog-evidence", str(_bound(evdir))]) == 1
    out = capsys.readouterr().out
    assert "python -I" in out and out.strip().splitlines()[-1] == "RESULT: FAIL (gate not run isolated)"


def test_isolated_reads_interpreter_flag():
    assert g._isolated() is bool(__import__("sys").flags.isolated)


def test_main_runs_candidate_code_checks_only_in_children(gate_env, evdir, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(g, "_run_child_check", lambda root, name: calls.append(name) or [])
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(_bound(evdir)), "--live"]) == 0
    assert calls == ["c3", "d"]
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: PASS (7 checks, 0 failed)"


def test_main_exact_last_line_without_live(gate_env, evdir, capsys):
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(_bound(evdir))]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: PASS (6 checks, 0 failed)"


def test_child_error_fails_gate(gate_env, evdir, capsys, monkeypatch):
    monkeypatch.setattr(g, "_run_child_check", lambda root, name: ["c3 check did not prove completion (exit 0, 0 result line(s))"])
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(_bound(evdir))]) == 1
    out = capsys.readouterr().out
    assert "FAIL  (c3) public mode default" in out and out.strip().splitlines()[-1].startswith("RESULT: FAIL")


def test_library_run_without_tests_keeps_in_process_c3(repo, monkeypatch):
    monkeypatch.setattr(g, "_run_child_check", lambda *a: (_ for _ in ()).throw(AssertionError("child")))
    for name in ("check_frozen", "check_no_env_in_widgets", "check_no_unsafe_html", "check_public_default"):
        monkeypatch.setattr(g, name, lambda root: [])
    assert all(not v for v in g.run_offline_checks(repo, run_tests=False).values())


def _child_runner(rc, stdout_fn, seen):
    def run(cmd, **kw):
        seen.append((cmd, kw))
        nonce = kw["input"].strip()
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout_fn(nonce), stderr="")
    return run


@pytest.mark.parametrize("rc, out, want", [
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=[]\n", []),
    (3, lambda n: f"layer 1 only ...\nVERA-GATE-CHILD-RESULT nonce={n} errors=[\"bad default\"]\n", ["bad default"]),
    (0, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=[]\n", None),        # exit 0 is never success
    (3, lambda n: "", None),                                                      # os._exit(3) without a line
    (0, lambda n: "", None),                                                      # os._exit(0) / SystemExit(0)
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={'0' * 32} errors=[]\n", None),  # wrong nonce
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=[]\n" * 2, None),     # duplicated line
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=nope\n", "malformed"),
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors={{}}\n", "malformed"),
    (3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=[1]\n", "malformed"),
    (1, lambda n: "RESULT: FAIL (gate aborted: SystemExit)\n", None),
])
def test_run_child_check_requires_nonce_line_and_rc(tmp_path, monkeypatch, rc, out, want):
    seen = []
    monkeypatch.setattr(g.subprocess, "run", _child_runner(rc, out, seen))
    got = g._run_child_check(tmp_path, "c3")
    if want is None:
        assert len(got) == 1 and "did not prove completion" in got[0]
    elif want == "malformed":
        assert got == ["c3 check returned a malformed result"]
    else:
        assert got == want
    cmd, kw = seen[0]
    nonce = kw["input"].strip()
    assert cmd[1] == "-I" and cmd[-2:] == ["--child-check", "c3"] and nonce not in " ".join(cmd)
    assert nonce not in kw["env"].values() and kw["capture_output"] is True and kw["timeout"] == g.CHILD_TIMEOUT_S


@pytest.mark.parametrize("exc, needle", [(subprocess.TimeoutExpired("x", 1), "exceeded"),
                                         (FileNotFoundError("py"), "could not start")])
def test_run_child_check_errors_fail_closed(tmp_path, monkeypatch, exc, needle):
    def run(*a, **k):
        raise exc
    monkeypatch.setattr(g.subprocess, "run", run)
    got = g._run_child_check(tmp_path, "d")
    assert len(got) == 1 and needle in got[0]


def test_child_mode_prints_tagged_result_and_exit_3(monkeypatch, capsys):
    nonce = "ab" * 16
    monkeypatch.setattr(g.sys, "stdin", io.StringIO(nonce + "\n"))
    monkeypatch.setattr(g, "check_public_default", lambda root: ["e1"])
    assert g.main(["--child-check", "c3"]) == g.CHILD_OK_RC == 3
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [f"VERA-GATE-CHILD-RESULT nonce={nonce} errors={json.dumps(['e1'])}"]


def test_child_mode_systemexit_gives_no_result_line(monkeypatch, capsys):
    monkeypatch.setattr(g.sys, "stdin", io.StringIO("ab" * 16 + "\n"))

    def candidate_exit(root):
        raise SystemExit(0)
    monkeypatch.setattr(g, "check_public_default", candidate_exit)
    assert g.main(["--child-check", "c3"]) == 1
    out = capsys.readouterr().out
    assert "VERA-GATE-CHILD-RESULT" not in out and "RESULT: FAIL" in out


def test_child_mode_rejects_bad_nonce(monkeypatch, capsys):
    monkeypatch.setattr(g.sys, "stdin", io.StringIO("not-a-nonce\n"))
    monkeypatch.setattr(g, "check_public_default", lambda root: (_ for _ in ()).throw(AssertionError("ran")))
    assert g.main(["--child-check", "c3"]) == 2
    assert "VERA-GATE-CHILD-RESULT" not in capsys.readouterr().out


# ------------------------------------------------------------------ G-3: suite cannot self-certify or race (a2)

def test_suite_writing_evidence_and_sidecar_fails(gate_env, evdir, monkeypatch, capsys):
    target = evdir / f"catalog-evidence-{CAND[:12]}.txt"                  # absent before the suite

    def forging_suite(root):
        _bound(evdir)                                                     # candidate code forges both files
        return []
    monkeypatch.setattr(g, "check_tests", forging_suite)
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(target)]) == 1
    out = capsys.readouterr().out
    assert "pre-suite snapshot mismatch" in out and f"FAIL  {A2}" in out


@pytest.mark.parametrize("change", ["evidence", "sidecar", "delete_sidecar", "sidecar_to_fifo"])
def test_suite_changing_evidence_fails(gate_env, evdir, monkeypatch, capsys, change):
    ev = _bound(evdir)
    sc = ev.with_name(ev.name + ".sidecar")

    def suite(root):
        if change == "evidence":
            ev.write_text(_body(count="checked 9 (table, column) pairs; 0 missing"), encoding="utf-8")
        elif change == "sidecar":
            _write_sidecar(ev, run_end_utc="2026-10-17T16:51:15Z")
        elif change == "delete_sidecar":
            sc.unlink()
        else:
            sc.unlink()
            os.mkfifo(sc)
        return []
    monkeypatch.setattr(g, "check_tests", suite)
    gate_env[:] = [(CAND, True), (CAND, True)]
    assert g.main(["--catalog-evidence", str(ev)]) == 1
    assert "pre-suite snapshot mismatch" in capsys.readouterr().out


def test_change_during_checker_read_is_caught_by_bracket(repo, evdir, monkeypatch):
    ev = _bound(evdir)
    snap = g._evidence_snapshot(ev)
    real = g.catalog_evidence_outcome

    def racing(path, root, cand):
        r = real(path, root, cand)
        _write_sidecar(ev, run_end_utc="2026-10-17T16:51:15Z")           # swapped after the checker read
        return r
    monkeypatch.setattr(g, "catalog_evidence_outcome", racing)
    out = g.check_live_catalog_evidence(ev, repo, CAND, "", snap)
    assert out == ["evidence or sidecar changed during the test suite (pre-suite snapshot mismatch)"]


def test_unchanged_snapshot_passes(repo, evdir):
    ev = _bound(evdir)
    assert g.check_live_catalog_evidence(ev, repo, CAND, "", g._evidence_snapshot(ev)) == []


def _suite_fake(seen, rc=0, raise_timeout=False):
    def run(cmd, **kw):
        seen.update(kw)
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        n = kw["env"]["VERA_GATE_NONCE"]
        return subprocess.CompletedProcess(cmd, rc, stdout=f"1 passed\nVERA-ISOLATION-EVIDENCE netns=isolated nonce={n}\n",
                                           stderr="")
    return run


class _FakePopen:
    """Stands in for subprocess.Popen in _run_session tests (no process is created)."""
    instances = []

    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.pid, self.returncode, self.events = cmd, kw, 4242, None, []
        self.stdin = io.BytesIO() if kw.get("stdin") == subprocess.PIPE else None
        if self.stdin is not None:
            self.stdin.close = lambda: self.events.append("stdin-closed")
        kw["stdout"].write(b"child out\n")
        kw["stderr"].write(b"child err\n")
        _FakePopen.instances.append(self)

    def wait(self):
        self.events.append("wait")
        self.returncode = 0
        return 0


@pytest.mark.parametrize("timeout_hit", [False, True])
def test_run_session_kills_group_before_reaping(monkeypatch, tmp_path, timeout_hit):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))            # temp output files stay under tmp_path
    _FakePopen.instances.clear()
    monkeypatch.setattr(g.subprocess, "Popen", _FakePopen)
    states = iter([None, None] + [object()]) if not timeout_hit else iter(lambda: None, 1)
    monkeypatch.setattr(g.os, "waitid", lambda *a: next(states))
    monkeypatch.setattr(g, "WAIT_POLL_S", 0)
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: _FakePopen.instances[-1].events.append(("killpg", pid, sig)))
    if timeout_hit:
        with pytest.raises(subprocess.TimeoutExpired):
            g._run_session(["x"], "/", 0.01, {"A": "1"}, input="n\n")
    else:
        r = g._run_session(["x"], "/", 30, {"A": "1"}, input="n\n")
        assert (r.returncode, r.stdout, r.stderr) == (0, "child out\n", "child err\n")
    fp = _FakePopen.instances[-1]
    assert fp.kw["start_new_session"] is True and fp.kw["env"] == {"A": "1"} and "preexec_fn" not in fp.kw
    assert fp.events == ["stdin-closed", ("killpg", 4242, signal.SIGKILL), "wait"]   # kill BEFORE reap (S-2)
    assert fp.stdin.getvalue() == b"n\n"


def test_run_uses_session_runner_unless_run_is_replaced(monkeypatch):
    calls = []
    monkeypatch.setattr(g, "_run_session", lambda *a: calls.append("session") or "S")
    monkeypatch.setattr(g, "_STDLIB_RUN", g.subprocess.run)                # simulate an unpatched interpreter
    assert g._run(["x"], "/", 1, {}) == "S" and calls == ["session"]
    monkeypatch.setattr(g.subprocess, "run", lambda cmd, **kw: calls.append(sorted(kw)) or "R")
    assert g._run(["x"], "/", 1, {}, input="i") == "R"
    assert calls[-1] == ["capture_output", "cwd", "env", "input", "text", "timeout"]


def test_session_capture_files_use_the_given_tmpdir(monkeypatch, tmp_path):
    """INT-1 M-3: the suite's unnamed capture files are created in the per-run dir that check_tests passes (never the
    system /tmp under `env -i`), and _run forwards tmpdir to the session runner. No process is created."""
    import tempfile
    real, dirs = tempfile.TemporaryFile, []
    monkeypatch.setattr(tempfile, "TemporaryFile", lambda *a, **kw: dirs.append(kw.get("dir")) or real(*a, **kw))
    _FakePopen.instances.clear()
    monkeypatch.setattr(g.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(g.os, "waitid", lambda *a: object())
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    r = g._run_session(["x"], "/", 30, {"A": "1"}, None, str(run_dir))
    assert r.stdout == "child out\n" and dirs == [str(run_dir), str(run_dir)]
    seen = []
    monkeypatch.setattr(g, "_run_session", lambda *a: seen.append(a) or "S")
    monkeypatch.setattr(g, "_STDLIB_RUN", g.subprocess.run)                # simulate an unpatched interpreter
    assert g._run(["x"], "/", 1, {}, tmpdir="T") == "S" and seen[-1][-1] == "T"


def _no_inherited_configs(monkeypatch):
    """INT-1 M-4: check_tests now refuses before spawning (RB-4 S-1) when the tracked root pytest.ini is missing or any
    other pytest config is found in tests/, the root or an ANCESTOR. A tmp_path repo has no tracked ini, and under the
    gate the suite's tmp_path sits below <root>/tmp/release-gate, so <root>/pytest.ini is an ancestor. S-1 is covered by
    the RB-4 tests; these tests exercise the D-B5-1 runner and env, so the config scan is faked (no file is probed)."""
    monkeypatch.setattr(g, "inherited_pytest_configs", lambda root: [])


def test_suite_timeout_reported(repo, monkeypatch):
    seen = {}
    _no_inherited_configs(monkeypatch)
    monkeypatch.setattr(g.subprocess, "run", _suite_fake(seen, raise_timeout=True))
    assert g.check_tests(repo) == [f"test suite exceeded {g.PYTEST_TIMEOUT_S}s"]


def test_suite_timeout_removes_tmpdir_only_after_group_kill_and_reap(repo, monkeypatch):
    """INT-1 r2 G-B: on the REAL session path (fake Popen, no process), a suite timeout kills the group, reaps the
    leader, and only then removes the per-run TMPDIR under <repo>/tmp/release-gate."""
    _no_inherited_configs(monkeypatch)
    _FakePopen.instances.clear()
    monkeypatch.setattr(g, "_STDLIB_RUN", g.subprocess.run)                # unpatched interpreter: _run_session path
    monkeypatch.setattr(g.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(g.os, "waitid", lambda *a: None)                   # the leader never exits
    monkeypatch.setattr(g, "WAIT_POLL_S", 0)
    monkeypatch.setattr(g, "PYTEST_TIMEOUT_S", 0.01)
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: _FakePopen.instances[-1].events.append(("killpg", pid, sig)))
    real_rmtree = shutil.rmtree

    def rmtree(path, ignore_errors=False):
        _FakePopen.instances[-1].events.append(("rmtree", str(path)))
        real_rmtree(path, ignore_errors=ignore_errors)
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    assert g.check_tests(repo) == [f"test suite exceeded {g.PYTEST_TIMEOUT_S}s"]
    fp = _FakePopen.instances[-1]
    tmpdir = fp.kw["env"]["TMPDIR"]
    assert Path(tmpdir).parent == repo / "tmp" / "release-gate"
    assert fp.events == [("killpg", 4242, signal.SIGKILL), "wait", ("rmtree", tmpdir)]   # kill, reap, THEN remove
    assert not Path(tmpdir).exists()


def test_kill_group_tolerates_gone_group_and_none(monkeypatch):
    calls = []

    def killpg(pid, sig):
        calls.append(pid)
        raise ProcessLookupError
    monkeypatch.setattr(g.os, "killpg", killpg)
    g._kill_group(None)
    g._kill_group(77)
    assert calls == [77]


def test_child_output_is_prefixed_never_column_zero(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(g.subprocess, "run", _child_runner(
        3, lambda n: f"RESULT: PASS (6 checks, 0 failed)\nVERA-GATE-CHILD-RESULT nonce={n} errors=[]\n", seen))
    assert g._run_child_check(tmp_path, "c3") == []
    out = capsys.readouterr().out.splitlines()
    assert out == ["      child| RESULT: PASS (6 checks, 0 failed)"]


SECRETS_ENV = {"SSH_AUTH_SOCK": "/run/fake/agent.sock", "SSH_AGENT_PID": "1", "EXTERNAL_DB_URL": "fake",
               "INTERNAL_DB_URL": "fake", "VERA_DB_URL_RO": "fake", "DATABASE_URL": "fake", "PGPASSWORD": "fake",
               "PGHOST": "fake", "PG_SERVICE_FILE": "fake"}


def test_suite_child_and_git_envs_drop_agent_socket_and_db_urls(repo, monkeypatch, tmp_path):
    # Fully synthetic environment: no real variable name or value can appear in an assertion message (R-5).
    monkeypatch.setattr(g.os, "environ", dict(SECRETS_ENV, KEEP_ME="1", VERA_TEST_DB_URL="fake", PATH="/fake/bin"))
    seen = {}
    _no_inherited_configs(monkeypatch)
    monkeypatch.setattr(g.subprocess, "run", _suite_fake(seen))
    assert g.check_tests(repo) == []
    child_seen = []
    monkeypatch.setattr(g.subprocess, "run", _child_runner(3, lambda n: f"VERA-GATE-CHILD-RESULT nonce={n} errors=[]\n",
                                                           child_seen))
    assert g._run_child_check(repo, "c3") == []
    for env in (seen["env"], child_seen[0][1]["env"], g._git_env()):
        assert not set(SECRETS_ENV) & set(env) and "VERA_TEST_DB_URL" not in env
    # INT-1 M-1: the suite child is ALLOWLISTED (RB-4 L4-A/S-2): an unlisted name is dropped and PATH is fixed.
    assert "KEEP_ME" not in seen["env"] and seen["env"]["PATH"] == "/usr/bin:/bin"
    # The child-check and git envs stay deny-only (D-B5-1 G-3, unchanged by INT-1 M-5): unlisted names pass through.
    assert child_seen[0][1]["env"].get("KEEP_ME") == "1" and g._git_env().get("KEEP_ME") == "1"


# ------------------------------------------------------------------ Q-1 / G-7: no hang on FIFO, size cap

@pytest.fixture
def alarm():
    old = signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(AssertionError("gate blocked (hang)")))
    signal.alarm(5)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def test_fifo_evidence_is_missing_not_a_hang(repo, evdir, alarm):
    fifo = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    os.mkfifo(fifo)
    o, why = _outcome(fifo, repo)
    assert o == "MISSING"
    assert g._evidence_snapshot(fifo)[0] == ("irregular", "")


def test_oversized_evidence_is_stale(repo, evdir):
    ev = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    ev.write_bytes(b"x" * (g.EVIDENCE_MAX_BYTES + 1))
    assert _outcome(ev, repo) == ("STALE", "evidence file is too large")


def test_symlinked_evidence_refused_before_read(repo, evdir, tmp_path):
    real = tmp_path / "real.txt"
    real.write_text(_body())
    link = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    os.symlink(real, link)
    assert _outcome(link, repo) == ("STALE", "evidence file is a symlink (refused)")


# ------------------------------------------------------------------ G-8: no control characters echoed

def test_printable_strips_ansi_and_controls():
    s = g._printable("\x1b[2K\rRESULT: PASS\x07\n")
    assert "\x1b" not in s and "\r" not in s and "\x07" not in s and "\n" not in s


# ------------------------------------------------------------------ r1 delta: pre-check is load-bearing; fixed git PATH

def test_checker_never_runs_on_suite_modified_files(repo, evdir, monkeypatch):
    ev = _bound(evdir)
    snap = g._evidence_snapshot(ev)
    _write_sidecar(ev, run_end_utc="2026-10-17T16:51:15Z")              # changed after the pre-suite snapshot

    def must_not_run(*a, **k):
        raise AssertionError("checker ran on bytes changed during the suite")
    monkeypatch.setattr(g, "catalog_evidence_outcome", must_not_run)
    assert g.check_live_catalog_evidence(ev, repo, CAND, "", snap) == [
        "evidence or sidecar changed during the test suite (pre-suite snapshot mismatch)"]


def test_git_exe_uses_fixed_search_path(monkeypatch):
    seen = {}
    monkeypatch.setattr(g.shutil, "which", lambda name, path=None: seen.update(name=name, path=path) or "/usr/bin/git")
    assert g._git_exe() == "/usr/bin/git" and seen == {"name": "git", "path": "/usr/bin:/bin"}
