"""Regression tests for scripts/check_catalog_evidence_freshness.py (VERA-DAR B5, owner directive 2026-10-17 §2;
Phase E fixes B5-1..B5-3, vet-E-acceptance-2026-10-17.md §6c).

DB-free, network-free, no subprocess. The checker is imported by file path (importlib.util). Every file (evidence,
sidecar, fake catalog script, symlinks) is fake and written under tmp_path; no real host, URL, user or evidence
content is used. One test exists per rejection reason in the checker (see the map at the end of this module).
"""
from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

_CHECKER = Path(__file__).resolve().parent.parent / "scripts" / "check_catalog_evidence_freshness.py"
_spec = importlib.util.spec_from_file_location("check_catalog_evidence_freshness", _CHECKER)
cef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cef)

CAND = "a" * 40                                   # fake candidate SHA
OTHER = "b" * 40                                  # a different fake SHA
FAKE_SCRIPT_BYTES = b"# fake catalog script for tests; not the real one\n"
SCRIPT = hashlib.sha256(FAKE_SCRIPT_BYTES).hexdigest()   # its hash, used as the expected script hash
NOW = datetime(2026, 10, 17, 18, 0, 0, tzinfo=timezone.utc)
COUNT = "checked 3 (table, column) pairs; 0 missing"
PASS_BODY = f"live_catalog_check script_sha256={SCRIPT}\n{COUNT}\nRESULT: PASS\n"


# ---------------------------------------------------------------- helpers (tmp_path only)

def _ev_name(sha: str = CAND) -> str:
    return f"catalog-evidence-{sha[:12]}.txt"


def _write_evidence(dirpath: Path, sha: str = CAND, body: str | bytes = PASS_BODY, name: str | None = None) -> Path:
    p = dirpath / (name or _ev_name(sha))
    if isinstance(body, bytes):
        p.write_bytes(body)
    else:
        p.write_text(body, encoding="utf-8")
    return p


def _sidecar_lines(ev: Path, sha: str = CAND, evidence_sha256: str | None = None, script: str = SCRIPT,
                   start: str = "2026-10-16T16:59:00Z", end: str = "2026-10-16T17:00:00Z") -> list[str]:
    digest = evidence_sha256 or hashlib.sha256(ev.read_bytes()).hexdigest()
    return [
        "vera-catalog-evidence-sidecar v1",
        f"candidate_sha={sha}",
        f"head_after={sha}",
        f"script_sha256={script}",
        f"evidence_file={ev.name}",
        f"evidence_sha256={digest}",
        "exit_code=0",
        "worktree_clean=1",
        f"run_start_utc={start}",
        f"run_end_utc={end}",
    ]


def _write_sidecar_lines(ev: Path, lines: list[str] | bytes, path: Path | None = None) -> Path:
    sc = path or ev.with_name(ev.name + ".sidecar")
    if isinstance(lines, bytes):
        sc.write_bytes(lines)
    else:
        sc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sc


def _write_sidecar(ev: Path, **kw) -> Path:
    return _write_sidecar_lines(ev, _sidecar_lines(ev, **kw))


def _write_script(dirpath: Path, data: bytes = FAKE_SCRIPT_BYTES) -> Path:
    p = dirpath / "live_catalog_check.py"
    p.write_bytes(data)
    return p


def _check(ev, sidecar=None, cand=CAND, **kw):
    kw.setdefault("expected_script_sha256", SCRIPT)
    return cef.check(cand, ev, sidecar, now=NOW, **kw)


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")


def _replace(lines: list[str], key: str, new: str | None) -> list[str]:
    """Replace the line starting with `key=` by `new` (None removes it)."""
    out = []
    for ln in lines:
        if ln.startswith(key + "="):
            if new is not None:
                out.append(new)
        else:
            out.append(ln)
    return out


# ---------------------------------------------------------------- PASS paths

def test_happy_path_passes(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    assert CAND[:12] in _check(ev)
    sc = _write_script(tmp_path)
    assert cef.main(["--candidate-sha", CAND, "--evidence", str(ev), "--script", str(sc),
                     "--expected-script-sha256", SCRIPT]) == 0


def test_legacy_pass_path_for_fallback_candidate(tmp_path, monkeypatch):
    ev = _write_evidence(tmp_path, name=cef.LEGACY_EVIDENCE_NAME)
    monkeypatch.setattr(cef, "LEGACY_EVIDENCE_SHA256", hashlib.sha256(ev.read_bytes()).hexdigest())
    assert "legacy fallback" in _check(ev, cand=cef.LEGACY_CANDIDATE_SHA)
    sc = _write_script(tmp_path)
    assert cef.main(["--candidate-sha", cef.LEGACY_CANDIDATE_SHA, "--evidence", str(ev), "--script", str(sc),
                     "--expected-script-sha256", SCRIPT]) == 0


# ---------------------------------------------------------------- B5-2: main() always hashes the script

def test_default_script_is_beside_the_checker():
    assert cef.DEFAULT_SCRIPT_PATH.name == "live_catalog_check.py"
    assert cef.DEFAULT_SCRIPT_PATH.parent == _CHECKER.resolve().parent


def test_main_without_script_hashes_default_script(tmp_path, monkeypatch, capsys):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    argv = ["--candidate-sha", CAND, "--evidence", str(ev), "--expected-script-sha256", SCRIPT]
    monkeypatch.setattr(cef, "DEFAULT_SCRIPT_PATH", tmp_path / "absent" / "live_catalog_check.py")
    assert cef.main(argv) == 1                                   # missing default script: FAIL, never a skip
    assert "catalog script missing" in capsys.readouterr().out
    monkeypatch.setattr(cef, "DEFAULT_SCRIPT_PATH", _write_script(tmp_path, b"# different bytes\n"))
    assert cef.main(argv) == 1                                   # wrong bytes: FAIL
    assert "catalog script hash" in capsys.readouterr().out
    monkeypatch.setattr(cef, "DEFAULT_SCRIPT_PATH", _write_script(tmp_path))
    assert cef.main(argv) == 0                                   # matching bytes: PASS


# ---------------------------------------------------------------- argument / script rejections

def test_short_candidate_sha_is_refused(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    with pytest.raises(cef.EvidenceError, match="full 40-hex"):
        _check(ev, cand=CAND[:12])


def test_invalid_expected_script_hash_is_refused(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    with pytest.raises(cef.EvidenceError, match="expected script sha256 is not 64 hex"):
        _check(ev, expected_script_sha256="xyz")


def test_script_bytes_mismatch_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    with pytest.raises(cef.EvidenceError, match="catalog script hash"):
        _check(ev, script=_write_script(tmp_path, b"# tampered\n"))


@pytest.mark.parametrize("argv", [
    [],
    ["--candidate-sha", CAND],
    ["--evidence", "x.txt"],
    ["--candidate-sha", CAND, "--evidence", "x.txt", "--max-age-hours", "not-a-number"],
    ["--candidate-sha", CAND, "--evidence", "x.txt", "--no-such-flag"],
])
def test_usage_error_exits_2(argv):
    assert cef.main(argv) == 2


# ---------------------------------------------------------------- file-level rejections

def test_missing_evidence_file_fails(tmp_path):
    with pytest.raises(cef.EvidenceError, match="evidence file missing"):
        _check(tmp_path / _ev_name())
    assert cef.main(["--candidate-sha", CAND, "--evidence", str(tmp_path / "nope.txt"),
                     "--script", str(_write_script(tmp_path)), "--expected-script-sha256", SCRIPT]) == 1


def test_symlinked_evidence_fails(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = _write_evidence(real_dir)
    link = tmp_path / _ev_name()
    _symlink(link, real)
    _write_sidecar(link)
    with pytest.raises(cef.EvidenceError, match="evidence file is a symlink"):
        _check(link)


def test_symlinked_sidecar_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    real_sc = _write_sidecar_lines(ev, _sidecar_lines(ev), path=tmp_path / "elsewhere.sidecar")
    _symlink(ev.with_name(ev.name + ".sidecar"), real_sc)
    with pytest.raises(cef.EvidenceError, match="sidecar is a symlink"):
        _check(ev)


def test_symlinked_script_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)
    real = _write_script(tmp_path)
    link = tmp_path / "linked_script.py"
    _symlink(link, real)
    with pytest.raises(cef.EvidenceError, match="catalog script is a symlink"):
        _check(ev, script=link)


def test_reused_legacy_evidence_fails_for_new_candidate(tmp_path):
    ev = _write_evidence(tmp_path, name=cef.LEGACY_EVIDENCE_NAME)
    with pytest.raises(cef.EvidenceError, match="legacy"):
        _check(ev)
    assert cef.main(["--candidate-sha", CAND, "--evidence", str(ev), "--script", str(_write_script(tmp_path)),
                     "--expected-script-sha256", SCRIPT]) == 1


def test_legacy_name_for_fallback_requires_recorded_bytes(tmp_path):
    ev = _write_evidence(tmp_path, name=cef.LEGACY_EVIDENCE_NAME)   # fake bytes != recorded C4 hash
    with pytest.raises(cef.EvidenceError, match="legacy evidence sha256"):
        _check(ev, cand=cef.LEGACY_CANDIDATE_SHA)


@pytest.mark.parametrize("name", [
    "catalog-evidence-XYZ.txt",
    "catalog-evidence-aaaa.txt",                    # fewer than 12 hex
    f"catalog-evidence-{'A' * 12}.txt",             # upper case
    f"evidence-{'a' * 12}.txt",
    f"catalog-evidence-{'a' * 12}.log",
])
def test_bad_evidence_filename_fails(tmp_path, name):
    ev = _write_evidence(tmp_path, name=name)
    _write_sidecar(ev)
    with pytest.raises(cef.EvidenceError, match="evidence filename must be"):
        _check(ev)


def test_mismatched_candidate_sha_in_filename_fails(tmp_path):
    ev = _write_evidence(tmp_path, sha=OTHER)          # evidence for a different commit
    _write_sidecar(ev, sha=OTHER)
    with pytest.raises(cef.EvidenceError, match="filename SHA"):
        _check(ev)


def test_reused_evidence_copied_under_new_name_without_sidecar_fails(tmp_path):
    old = _write_evidence(tmp_path, sha=OTHER)
    _write_sidecar(old, sha=OTHER)
    copy = _write_evidence(tmp_path, body=old.read_text(encoding="utf-8"))   # same PASS bytes, new name
    with pytest.raises(cef.EvidenceError, match="sidecar missing"):
        _check(copy)


# ---------------------------------------------------------------- evidence-body rejections

@pytest.mark.parametrize("body, reason", [
    (b"\xff\xfe\x00not utf-8\n", "evidence is not UTF-8"),
    ("", "evidence is empty"),
    ("\n  \n", "evidence is empty"),
    (f"garbage first line\n{COUNT}\nRESULT: PASS\n", "not the live_catalog_check script_sha256 header"),
    (f"live_catalog_check script_sha256={SCRIPT[:40]}\n{COUNT}\nRESULT: PASS\n",
     "not the live_catalog_check script_sha256 header"),
    (PASS_BODY.replace(SCRIPT, "f" * 64), "evidence script hash"),
    (f"live_catalog_check script_sha256={SCRIPT}\n{COUNT}\n", "no PASS line"),
    (f"live_catalog_check script_sha256={SCRIPT}\nchecked 3 (table, column) pairs; 1 missing\nRESULT: FAIL\n",
     "no PASS line"),
    (f"live_catalog_check script_sha256={SCRIPT}\ncannot verify: x\nRESULT: CANNOT_VERIFY\n", "no PASS line"),
    (f"{PASS_BODY}RESULT: FAIL\n", "no PASS line"),
    (f"live_catalog_check script_sha256={SCRIPT}\nRESULT: PASS\n{COUNT}\n", "no PASS line"),   # PASS not last
    (f"live_catalog_check script_sha256={SCRIPT}\nRESULT: PASS\n", "no 'checked N"),           # no count line
    (f"live_catalog_check script_sha256={SCRIPT}\nchecked 3 (table, column) pairs; 2 missing\nRESULT: PASS\n",
     "no 'checked N"),
])
def test_evidence_body_rejections(tmp_path, body, reason):
    ev = _write_evidence(tmp_path, body=body)
    _write_sidecar(ev)
    with pytest.raises(cef.EvidenceError, match=reason):
        _check(ev)


# ---------------------------------------------------------------- sidecar rejections (one line mutated each)

_SIDECAR_CASES = [
    ("header-version", lambda ls: ["vera-catalog-evidence-sidecar v2"] + ls[1:], "sidecar header"),
    ("header-missing", lambda ls: ls[1:], "sidecar header"),
    ("malformed-line", lambda ls: ls + ["no equals sign here"], "malformed line"),
    ("empty-key", lambda ls: ls + ["=value"], "malformed line"),
    ("duplicate-key", lambda ls: ls + ["exit_code=0"], "repeats key exit_code"),
    ("missing-key", lambda ls: _replace(ls, "worktree_clean", None), r"missing field\(s\): worktree_clean"),
    ("empty-value", lambda ls: _replace(ls, "evidence_sha256", "evidence_sha256="), "missing field"),
    ("candidate_sha", lambda ls: _replace(ls, "candidate_sha", f"candidate_sha={OTHER}"), "sidecar candidate_sha"),
    ("head_after", lambda ls: _replace(ls, "head_after", f"head_after={OTHER}"), "sidecar head_after"),
    ("script_sha256", lambda ls: _replace(ls, "script_sha256", "script_sha256=" + "e" * 64), "sidecar script_sha256"),
    ("evidence_sha256", lambda ls: _replace(ls, "evidence_sha256", "evidence_sha256=" + "d" * 64),
     "sidecar evidence_sha256"),
    ("evidence_file", lambda ls: _replace(ls, "evidence_file", f"evidence_file={_ev_name(OTHER)}"),
     "sidecar evidence_file"),
    ("exit_code", lambda ls: _replace(ls, "exit_code", "exit_code=1"), "exit_code is not 0"),
    ("worktree_clean-0", lambda ls: _replace(ls, "worktree_clean", "worktree_clean=0"), "worktree_clean is not 1"),
    ("worktree_clean-yes", lambda ls: _replace(ls, "worktree_clean", "worktree_clean=yes"), "worktree_clean is not 1"),
    ("utc-start", lambda ls: _replace(ls, "run_start_utc", "run_start_utc=2026-10-16 16:59:00"),
     "run_start_utc is not UTC"),
    ("utc-end", lambda ls: _replace(ls, "run_end_utc", "run_end_utc=2026-10-16T17:00:00+00:00"),
     "run_end_utc is not UTC"),
    ("end-before-start", lambda ls: _replace(ls, "run_end_utc", "run_end_utc=2026-10-16T16:00:00Z"),
     "before run_start_utc"),
    ("future-end", lambda ls: _replace(ls, "run_end_utc", "run_end_utc=2026-10-18T00:00:00Z"), "in the future"),
]


@pytest.mark.parametrize("case, mutate, reason", _SIDECAR_CASES, ids=[c[0] for c in _SIDECAR_CASES])
def test_sidecar_rejections(tmp_path, case, mutate, reason):
    ev = _write_evidence(tmp_path)
    _write_sidecar_lines(ev, mutate(_sidecar_lines(ev)))
    with pytest.raises(cef.EvidenceError, match=reason):
        _check(ev)


def test_sidecar_not_utf8_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar_lines(ev, b"\xff\xfe\x00 not utf-8\n")
    with pytest.raises(cef.EvidenceError, match="sidecar is not UTF-8"):
        _check(ev)


def test_explicit_sidecar_path_missing_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev)                                  # default sidecar exists, but the named one does not
    with pytest.raises(cef.EvidenceError, match="sidecar missing"):
        _check(ev, sidecar=tmp_path / "other.sidecar")


def test_stale_evidence_beyond_max_age_fails(tmp_path):
    ev = _write_evidence(tmp_path)
    _write_sidecar(ev, start="2026-10-15T16:59:00Z", end="2026-10-15T17:00:00Z")
    with pytest.raises(cef.EvidenceError, match="stale"):
        _check(ev, max_age_hours=24)
    assert CAND[:12] in _check(ev, max_age_hours=72)   # within the limit (49 h old): PASS


# Rejection reason -> test (B5-3 coverage map; every `raise EvidenceError` in the checker):
#   candidate not 40-hex ............ test_short_candidate_sha_is_refused
#   expected hash not 64-hex ........ test_invalid_expected_script_hash_is_refused
#   script missing (default) ........ test_main_without_script_hashes_default_script
#   script hash mismatch ............ test_script_bytes_mismatch_fails, test_main_without_script_hashes_default_script
#   symlink (evidence/sidecar/script)  test_symlinked_evidence_fails, test_symlinked_sidecar_fails,
#                                      test_symlinked_script_fails
#   evidence missing ................ test_missing_evidence_file_fails
#   legacy for new candidate ........ test_reused_legacy_evidence_fails_for_new_candidate
#   legacy wrong bytes .............. test_legacy_name_for_fallback_requires_recorded_bytes
#   legacy PASS (fallback) .......... test_legacy_pass_path_for_fallback_candidate
#   bad filename .................... test_bad_evidence_filename_fails
#   filename SHA mismatch ........... test_mismatched_candidate_sha_in_filename_fails
#   evidence non-UTF-8 / empty / bad header / script hash / no PASS / no count line
#                                     test_evidence_body_rejections
#   sidecar missing ................. test_reused_evidence_copied_under_new_name_without_sidecar_fails,
#                                     test_explicit_sidecar_path_missing_fails
#   sidecar non-UTF-8 ............... test_sidecar_not_utf8_fails
#   sidecar header / malformed / duplicate / missing key / candidate_sha / head_after / script_sha256 /
#   evidence_sha256 / evidence_file / exit_code / worktree_clean / UTC format / end<start / future end
#                                     test_sidecar_rejections
#   stale (max age) ................. test_stale_evidence_beyond_max_age_fails
#   usage error exit 2 .............. test_usage_error_exits_2
#   not covered: "unreadable (OSError)" on a regular file (needs permission tricks that do not hold under root).
