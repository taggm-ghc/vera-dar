"""K18W WS-3: negative (fail-closed) tests for release_gate (a2) catalog-evidence binding.

Every evidence file, sidecar and fake SHA is synthetic and written under tmp_path. The only files read outside
tmp_path are the gate, the pinned checker and scripts/live_catalog_check.py in this worktree; the latter two are
BYTE-COPIED into tmp_path/repo/scripts (never imported, never run), exactly as tests/test_release_gate_evidence_binding.py
does, so the checker pin and the approved C1 script hash hold. The gate module is loaded by file path (same pattern
as that file; nothing is imported from it).

Hermetic: an autouse fixture replaces os.environ with a synthetic dict and makes subprocess.run/Popen, os.system and
socket creation raise, so no git, no child process, no DB and no network can be reached. The candidate SHA is passed
directly or via a faked _candidate_state.

Existing coverage is listed in EXISTING_COVERAGE (case -> existing node names, not re-implemented here). New cases
are in NEW_CASES. test_matrix_every_case_fails_closed runs one compact builder per case (old and new) so every case
is visibly fail-closed in one place, next to POSITIVE controls that show the negatives are not vacuous.

ProposedHardening: tests that need $S/kraken-K18W/int/ws3/hardening-proposal.diff (listed in PROPOSAL_TESTS).
They are skipped unless the proposal is applied (detected by release_gate._PASS_COUNT_LINE). The matching residual
characterisation (current behaviour) runs only while the proposal is NOT applied.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_gate_k18w_ws3", ROOT / "scripts" / "release_gate.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

CAND = "c" * 40                                    # fake candidate SHA (gate HEAD)
OTHER = "d" * 40                                   # a different fake SHA
COLLIDE = "c" * 12 + "e" * 28                      # shares the 12-hex filename prefix with CAND
LEGACY_SHA = "9c3dd5652644cf79ab9918ef5b5bcdab3497b1bb"  # recorded fallback id (public commit id, not evidence)
A2 = "(a2) live catalog evidence (required, owner-run)"
HEADER = "vera-catalog-evidence-sidecar v1"
REQUIRED_KEYS = ("candidate_sha", "head_after", "script_sha256", "evidence_file", "evidence_sha256", "exit_code",
                 "worktree_clean", "run_start_utc", "run_end_utc")
HARDENED = hasattr(g, "_PASS_COUNT_LINE")          # True only once hardening-proposal.diff is applied

PROPOSAL_TESTS = (
    "tests/test_release_gate_evidence_negative_k18w.py::TestProposedHardening::test_contradictory_body_fails_closed",
    "tests/test_release_gate_evidence_negative_k18w.py::TestProposedHardening::test_well_formed_bodies_still_pass",
    "tests/test_release_gate_evidence_negative_k18w.py::TestProposedHardening::test_contradictory_body_fails_through_offline_checks",
)

# Case -> existing tests that already cover it (B = tests/test_release_gate_evidence_binding.py,
# F = tests/test_catalog_evidence_freshness.py, L = tests/test_live_optin.py). Not duplicated here.
EXISTING_COVERAGE = {
    "missing_evidence": ["B::test_missing_text_names_bound_file_and_sidecar", "B::test_evidence_path_is_a_directory",
                         "B::test_fifo_evidence_is_missing_not_a_hang", "L::test_gate_catalog_evidence_missing_fails",
                         "F::test_missing_evidence_file_fails"],
    "missing_sidecar": ["B::test_identical_content_copied_under_candidate_name_without_sidecar_fails",
                        "B::test_suite_changing_evidence_fails[delete_sidecar]",
                        "F::test_explicit_sidecar_path_missing_fails"],
    "stale_older_sha_or_run": ["B::test_evidence_for_other_sha_fails",
                               "B::test_age_limit_ships_unset_and_boundary_works_when_set",
                               "B::test_suite_changing_evidence_fails", "F::test_stale_evidence_beyond_max_age_fails"],
    "different_candidate_and_legacy": ["B::test_legacy_name_rejected_for_new_candidate",
                                       "B::test_legacy_name_for_fallback_sha_requires_recorded_bytes",
                                       "B::test_file_named_for_candidate_but_sidecar_for_other_sha_fails",
                                       "B::test_main_fails_on_legacy_evidence_for_candidate",
                                       "B::test_candidate_must_be_full_40_hex", "B::test_candidate_none_fails_closed",
                                       "F::test_reused_legacy_evidence_fails_for_new_candidate",
                                       "F::test_mismatched_candidate_sha_in_filename_fails"],
    "inconsistent": ["B::test_evidence_edited_after_sidecar_fails",
                     "B::test_sidecar_metadata_rejections (head_after, script_sha256, evidence_sha256, evidence_file, "
                     "times)", "B::test_identical_content_with_wrong_sidecar_fields_fails"],
    "checker_or_script_hash": ["B::test_checker_weakened_is_refused_by_pin", "B::test_checker_missing_fails_closed",
                               "B::test_checker_symlink_refused", "B::test_repo_script_not_the_approved_c1_hash_fails",
                               "B::test_evidence_for_other_script_version_fails", "B::test_checker_pin_matches_repo_checker",
                               "F::test_script_bytes_mismatch_fails"],
    "incomplete": ["B::test_incomplete_or_failed_inspection_body_never_passes", "B::test_truncated_sidecar_fails_closed",
                   "B::test_sidecar_byte_truncated_mid_line_fails",
                   "B::test_sidecar_metadata_rejections (exit_code, run times dropped)"],
    "status_not_pass": ["L::test_gate_catalog_evidence_mismatch_or_abort_fails",
                        "B::test_incomplete_or_failed_inspection_body_never_passes (RESULT: FAIL)",
                        "B::test_sidecar_metadata_rejections (exit_code 1/2/blank)"],
    "duplicate_or_conflicting_result": ["B::test_incomplete_or_failed_inspection_body_never_passes (PASS twice)",
                                        "L::test_gate_catalog_evidence_mismatch_or_abort_fails (PASS then FAIL)",
                                        "B::test_sidecar_duplicate_key_and_non_utf8_fail"],
    "positive_control": ["B::test_bound_evidence_passes_with_prefix_only_note",
                         "B::test_main_passes_with_bound_evidence_and_stable_clean_head",
                         "L::test_gate_catalog_evidence_match_passes_only_when_bound"],
}


# ------------------------------------------------------------------ hermetic fixtures

@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """tmp_path is created BEFORE the swap (base temp dir resolved from the real env). Synthetic environ (no real variable can be read or echoed) and no process or socket can be created."""
    monkeypatch.setattr(os, "environ", {"PATH": "/fake/bin", "HOME": "/fake/home"})

    def refuse(*a, **k):
        raise AssertionError("process or socket creation attempted in a hermetic K18W test")
    for mod, name in ((subprocess, "run"), (subprocess, "Popen"), (subprocess, "check_output"), (os, "system"),
                      (socket, "socket"), (socket, "create_connection")):
        monkeypatch.setattr(mod, name, refuse)


def _approved_script_sha() -> str:
    return hashlib.sha256((ROOT / g.CATALOG_SCRIPT).read_bytes()).hexdigest()


def _body(script_sha: str | None = None, count: str = "checked 7 (table, column) pairs; 0 missing",
          result: str = "RESULT: PASS") -> str:
    return f"live_catalog_check script_sha256={script_sha or _approved_script_sha()}\n{count}\n{result}\n"


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


def _fields(ev: Path, sha: str = CAND, **over) -> dict[str, str]:
    f = {"candidate_sha": sha, "head_after": sha, "script_sha256": _approved_script_sha(), "evidence_file": ev.name,
         "evidence_sha256": hashlib.sha256(ev.read_bytes()).hexdigest(), "exit_code": "0", "worktree_clean": "1",
         "run_start_utc": "2026-10-17T16:51:13Z", "run_end_utc": "2026-10-17T16:51:14Z"}
    f.update(over)
    return f


def _sidecar(ev: Path, fields: dict[str, str] | None = None, drop: tuple[str, ...] = (), extra: tuple[str, ...] = (),
             **over) -> Path:
    fields = fields if fields is not None else _fields(ev, **over)
    lines = [HEADER] + [f"{k}={v}" for k, v in fields.items() if k not in drop] + list(extra)
    sc = ev.with_name(ev.name + ".sidecar")
    sc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sc


def _bound(evdir: Path, sha: str = CAND, body: str | bytes | None = None, name: str | None = None,
           sidecar: bool = True, **over) -> Path:
    ev = evdir / (name or f"catalog-evidence-{sha[:12]}.txt")
    data = body if body is not None else _body()
    if isinstance(data, bytes):
        ev.write_bytes(data)
    else:
        ev.write_text(data, encoding="utf-8", newline="")
    if sidecar:
        _sidecar(ev, sha=sha, **over)
    return ev


def _offline_a2(repo: Path, ev, monkeypatch, head: str = CAND) -> list[str]:
    """(a2) through run_offline_checks with the suite, git reads and all other checks faked."""
    monkeypatch.setattr(g, "check_tests", lambda root: [])
    for name in ("check_frozen", "check_no_env_in_widgets", "check_no_unsafe_html", "check_public_default"):
        monkeypatch.setattr(g, name, lambda root: [])
    monkeypatch.setattr(g, "_candidate_state", lambda root: (head, True))
    res = g.run_offline_checks(repo, run_tests=True, catalog_evidence=ev, expect_candidate_sha=head,
                               isolate_candidate_code=False)
    return res[A2]


# ------------------------------------------------------------------ case builders: (repo, evdir) -> (evidence, candidate)

def _c_missing_evidence_orphan_sidecar(repo, evdir):
    ev = _bound(evdir)
    ev.unlink()                                   # the sidecar is left behind and must not count
    return ev, CAND


def _c_missing_sidecar(repo, evdir):
    return _bound(evdir, sidecar=False), CAND


def _c_sidecar_is_directory(repo, evdir):
    ev = _bound(evdir, sidecar=False)
    ev.with_name(ev.name + ".sidecar").mkdir()
    return ev, CAND


def _c_stale_renamed_pair_from_older_sha(repo, evdir):
    old = _bound(evdir, sha=OTHER)                # a genuine-looking pair for an older commit ...
    new = evdir / f"catalog-evidence-{CAND[:12]}.txt"
    shutil.move(old, new)                         # ... renamed to the candidate's name, sidecar too
    shutil.move(old.with_name(old.name + ".sidecar"), new.with_name(new.name + ".sidecar"))
    return new, CAND


def _c_stale_older_sha_older_times(repo, evdir):
    ev = _bound(evdir, sha=CAND, candidate_sha=OTHER, head_after=OTHER, run_start_utc="2026-10-01T00:00:00Z",
                run_end_utc="2026-10-01T00:00:05Z")
    return ev, CAND


def _c_prefix_collision_sidecar(repo, evdir):
    return _bound(evdir, sha=COLLIDE), CAND       # filename prefix matches CAND, sidecar is for COLLIDE


def _c_prefix_collision_full_name(repo, evdir):
    return _bound(evdir, sha=COLLIDE, name=f"catalog-evidence-{COLLIDE}.txt"), CAND


def _c_legacy_name_with_consistent_sidecar(repo, evdir):
    ev = _bound(evdir, name="catalog-evidence.txt")   # sidecar claims CAND; legacy name stays bound to 9c3dd56
    return ev, CAND


def _c_legacy_name_naming_other_sha(repo, evdir):
    ev = _bound(evdir, sha=OTHER, name="catalog-evidence.txt")
    return ev, CAND


def _c_sidecar_hash_mismatch(repo, evdir):
    return _bound(evdir, evidence_sha256=hashlib.sha256(b"other").hexdigest()), CAND


def _c_sidecar_script_contradicts_body(repo, evdir):
    return _bound(evdir, script_sha256=hashlib.sha256(b"x").hexdigest()), CAND


def _c_sidecar_exit_contradicts_pass(repo, evdir):
    return _bound(evdir, exit_code="1"), CAND


def _c_checker_one_byte_appended(repo, evdir):
    ev = _bound(evdir)
    with open(repo / g.CATALOG_CHECKER, "ab") as fh:
        fh.write(b"\n")
    return ev, CAND


def _c_script_missing(repo, evdir):
    ev = _bound(evdir)
    (repo / g.CATALOG_SCRIPT).unlink()
    return ev, CAND


def _c_script_symlink(repo, evdir):
    ev = _bound(evdir)
    real = repo.parent / "elsewhere_live_catalog_check.py"
    shutil.move(repo / g.CATALOG_SCRIPT, real)
    os.symlink(real, repo / g.CATALOG_SCRIPT)     # same bytes, but the checker refuses a symlinked script
    return ev, CAND


def _c_truncated_mid_result(repo, evdir):
    return _bound(evdir, body=_body()[:-3]), CAND   # "...RESULT: PA"


def _c_truncated_after_header(repo, evdir):
    return _bound(evdir, body=_body().splitlines(keepends=True)[0]), CAND


def _c_sidecar_drops_candidate_sha(repo, evdir):
    return _bound(evdir, drop=("candidate_sha",)), CAND


def _c_status_cannot_verify(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT: CANNOT_VERIFY")), CAND


def _c_status_lowercase_pass(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT: pass")), CAND


def _c_status_no_space(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT:PASS")), CAND


def _c_status_spelled_out(repo, evdir):
    return _bound(evdir, body=_body(result="PASS")), CAND


def _c_dup_fail_then_pass(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT: FAIL\nRESULT: PASS")), CAND


def _c_dup_indented_second_pass(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT: PASS\n   RESULT: PASS")), CAND


def _c_pass_then_trailing_line(repo, evdir):
    return _bound(evdir, body=_body(result="RESULT: PASS\ntrailing")), CAND


def _c_duplicate_sidecar_key_conflict(repo, evdir):
    ev = _bound(evdir, sidecar=False)
    _sidecar(ev, extra=(f"candidate_sha={OTHER}",))
    return ev, CAND


def _c_candidate_unknown(repo, evdir):
    return _bound(evdir), None


NEW_CASES = {
    "missing_evidence/orphan_sidecar": _c_missing_evidence_orphan_sidecar,
    "missing_sidecar/sidecar_is_directory": _c_sidecar_is_directory,
    "stale/renamed_pair_from_older_sha": _c_stale_renamed_pair_from_older_sha,
    "stale/older_sha_older_times": _c_stale_older_sha_older_times,
    "other_sha/prefix_collision_sidecar": _c_prefix_collision_sidecar,
    "other_sha/prefix_collision_full_40hex_name": _c_prefix_collision_full_name,
    "other_sha/legacy_name_with_consistent_sidecar": _c_legacy_name_with_consistent_sidecar,
    "other_sha/legacy_name_naming_other_sha": _c_legacy_name_naming_other_sha,
    "inconsistent/sidecar_exit_contradicts_pass": _c_sidecar_exit_contradicts_pass,
    "hash/checker_one_byte_appended": _c_checker_one_byte_appended,
    "hash/script_missing": _c_script_missing,
    "hash/script_symlink_same_bytes": _c_script_symlink,
    "incomplete/truncated_mid_result_line": _c_truncated_mid_result,
    "incomplete/truncated_after_header": _c_truncated_after_header,
    "status/cannot_verify_with_valid_sidecar": _c_status_cannot_verify,
    "status/lowercase_pass": _c_status_lowercase_pass,
    "status/no_space": _c_status_no_space,
    "status/bare_PASS_token": _c_status_spelled_out,
    "duplicate/fail_then_pass": _c_dup_fail_then_pass,
    "duplicate/indented_second_pass": _c_dup_indented_second_pass,
    "duplicate/pass_then_trailing_line": _c_pass_then_trailing_line,
}

# One representative builder for each case already covered elsewhere (re-run here only so the matrix is complete).
COVERED_CASES = {
    "missing_sidecar/fresh_named": _c_missing_sidecar,
    "inconsistent/sidecar_hash_mismatch": _c_sidecar_hash_mismatch,
    "inconsistent/sidecar_script_contradicts_body": _c_sidecar_script_contradicts_body,
    "incomplete/sidecar_drops_candidate_sha": _c_sidecar_drops_candidate_sha,
    "duplicate/conflicting_sidecar_key": _c_duplicate_sidecar_key_conflict,
    "other_sha/candidate_unknown": _c_candidate_unknown,
}

MATRIX = {**NEW_CASES, **COVERED_CASES}


# ------------------------------------------------------------------ positive controls (non-vacuity)

class TestPositiveControls:
    def test_well_formed_set_is_accepted(self, repo, evdir):
        ev = _bound(evdir)
        o, note = g.catalog_evidence_outcome(ev, repo, CAND)
        assert o == "PASSED" and CAND[:12] in note and CAND not in note
        assert g.check_live_catalog_evidence(ev, repo, CAND) == []

    def test_well_formed_set_is_accepted_through_offline_checks(self, repo, evdir, monkeypatch):
        assert _offline_a2(repo, _bound(evdir), monkeypatch) == []

    def test_full_40hex_name_and_crlf_are_accepted(self, repo, evdir):
        ev = _bound(evdir, name=f"catalog-evidence-{CAND}.txt", body=_body().replace("\n", "\r\n"))
        assert g.catalog_evidence_outcome(ev, repo, CAND)[0] == "PASSED"


# ------------------------------------------------------------------ the matrix: every case fails closed

class TestFailClosedMatrix:
    @pytest.mark.parametrize("case", sorted(MATRIX))
    def test_matrix_every_case_fails_closed(self, repo, evdir, case):
        ev, cand = MATRIX[case](repo, evdir)
        o, why = g.catalog_evidence_outcome(ev, repo, cand)
        assert o in ("MISSING", "STALE", "FAILED"), (case, o, why)
        assert g.check_live_catalog_evidence(ev, repo, cand) != []

    @pytest.mark.parametrize("case", sorted(NEW_CASES))
    def test_new_cases_fail_through_offline_checks(self, repo, evdir, monkeypatch, case):
        ev, _ = NEW_CASES[case](repo, evdir)
        out = _offline_a2(repo, ev, monkeypatch)
        assert out and all(isinstance(e, str) and e for e in out), (case, out)

    def test_matrix_covers_every_required_dimension(self):
        families = {k.split("/")[0] for k in MATRIX}
        assert {"missing_evidence", "missing_sidecar", "stale", "other_sha", "inconsistent", "hash", "incomplete",
                "status", "duplicate"} <= families
        assert set(EXISTING_COVERAGE) >= {"missing_evidence", "missing_sidecar", "positive_control"}


# ------------------------------------------------------------------ targeted new negatives

class TestTargetedNegatives:
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    @pytest.mark.parametrize("mode", ["dropped", "blank"])
    def test_each_required_sidecar_field_missing_or_blank_fails(self, repo, evdir, key, mode):
        ev = _bound(evdir, sidecar=False)
        if mode == "dropped":
            _sidecar(ev, drop=(key,))
        else:
            _sidecar(ev, **{key: ""})
        o, why = g.catalog_evidence_outcome(ev, repo, CAND)
        assert o == "STALE" and key in why, (o, why)

    def test_renamed_pair_reason_names_the_sidecar_binding(self, repo, evdir):
        ev, cand = _c_stale_renamed_pair_from_older_sha(repo, evdir)
        o, why = g.catalog_evidence_outcome(ev, repo, cand)
        assert o == "STALE" and "sidecar candidate_sha" in why and CAND not in why

    def test_prefix_collision_is_caught_by_full_sha_in_sidecar(self, repo, evdir):
        ev, cand = _c_prefix_collision_sidecar(repo, evdir)
        assert g.catalog_evidence_outcome(ev, repo, cand) == (
            "STALE", f"evidence not bound to candidate: sidecar candidate_sha {COLLIDE[:12]} != candidate {CAND[:12]}")
        assert g.catalog_evidence_outcome(ev, repo, COLLIDE)[0] == "PASSED"   # control: right SHA is accepted

    def test_legacy_name_with_sidecar_for_candidate_is_still_legacy_bound(self, repo, evdir):
        ev, cand = _c_legacy_name_with_consistent_sidecar(repo, evdir)
        o, why = g.catalog_evidence_outcome(ev, repo, cand)
        assert o == "STALE" and "legacy" in why and LEGACY_SHA[:12] in why

    def test_checker_with_one_extra_byte_is_refused_by_pin(self, repo, evdir):
        ev, cand = _c_checker_one_byte_appended(repo, evdir)
        assert g.catalog_evidence_outcome(ev, repo, cand) == ("STALE", "binding checker unavailable (ImportError)")

    def test_missing_catalog_script_is_stale_not_crash(self, repo, evdir):
        ev, cand = _c_script_missing(repo, evdir)
        assert g.catalog_evidence_outcome(ev, repo, cand) == (
            "STALE", "evidence is for a different version of scripts/live_catalog_check.py")

    def test_symlinked_catalog_script_refused_by_checker(self, repo, evdir):
        ev, cand = _c_script_symlink(repo, evdir)
        o, why = g.catalog_evidence_outcome(ev, repo, cand)
        assert o == "STALE" and "symlink" in why

    @pytest.mark.parametrize("result", ["RESULT: CANNOT_VERIFY", "RESULT: pass", "RESULT:PASS", "RESULT: FAIL\nRESULT: PASS",
                                        "RESULT: PASS\n   RESULT: PASS"])
    def test_non_pass_or_conflicting_status_is_failed_even_with_valid_sidecar(self, repo, evdir, result):
        ev = _bound(evdir, body=_body(result=result))
        o, why = g.catalog_evidence_outcome(ev, repo, CAND)
        assert o == "FAILED" and "not a match" in why

    def test_trailing_line_after_pass_is_stale(self, repo, evdir):
        ev, cand = _c_pass_then_trailing_line(repo, evdir)
        assert g.catalog_evidence_outcome(ev, repo, cand)[0] == "STALE"

    def test_orphan_sidecar_without_evidence_is_missing(self, repo, evdir):
        ev, cand = _c_missing_evidence_orphan_sidecar(repo, evdir)
        assert ev.with_name(ev.name + ".sidecar").is_file()
        assert g.catalog_evidence_outcome(ev, repo, cand)[0] == "MISSING"


# ------------------------------------------------------------------ accepted residuals (characterisation, not desired)

class TestResidualCharacterisation:
    """Current behaviour on the record (design.md 9.1). Each PASSED here is a known, accepted residual."""

    def test_residual_DV6_same_sha_old_run_passes_without_age_limit(self, repo, evdir):
        assert g.CATALOG_EVIDENCE_MAX_AGE_HOURS is None
        ev = _bound(evdir, run_start_utc="2020-01-01T00:00:00Z", run_end_utc="2020-01-01T00:00:01Z")
        assert g.catalog_evidence_outcome(ev, repo, CAND)[0] == "PASSED"

    def test_residual_G9_unknown_sidecar_keys_are_ignored(self, repo, evdir):
        ev = _bound(evdir, sidecar=False)
        _sidecar(ev, extra=("status=FAIL", "operator=unknown"))
        assert g.catalog_evidence_outcome(ev, repo, CAND)[0] == "PASSED"

    @pytest.mark.skipif(HARDENED, reason="hardening proposal applied: see ProposedHardening")
    def test_residual_contradictory_body_currently_passes(self, repo, evdir):
        for body in _CONTRADICTORY_BODIES:
            for p in evdir.iterdir():
                p.unlink()
            ev = _bound(evdir, body=body())
            assert g.catalog_evidence_outcome(ev, repo, CAND)[0] == "PASSED"


def _hdr(sha: str | None = None) -> str:
    return f"live_catalog_check script_sha256={sha or _approved_script_sha()}"


_OK_COUNT = "checked 7 (table, column) pairs; 0 missing"
_CONTRADICTORY_BODIES = (
    lambda: f"{_hdr()}\n{_OK_COUNT}\nchecked 9 (table, column) pairs; 2 missing\nRESULT: PASS\n",   # 2 counts
    lambda: f"{_hdr()}\n{_OK_COUNT}\nmissing: public.t.c\nRESULT: PASS\n",                           # missing line
    lambda: f"{_hdr()}\ncannot verify: query failed (X)\n{_OK_COUNT}\nRESULT: PASS\n",               # cannot verify
    lambda: f"{_hdr()}\n{_hdr('0' * 64)}\n{_OK_COUNT}\nRESULT: PASS\n",                              # 2nd header
    lambda: f"{_hdr()}\n{_OK_COUNT}\n{_OK_COUNT}\nRESULT: PASS\n",                                   # duplicated count
)


# ------------------------------------------------------------------ needs hardening-proposal.diff (integrator decides)

@pytest.mark.skipif(not HARDENED, reason="needs kraken-K18W/int/ws3/hardening-proposal.diff (release_gate.py)")
class ProposedHardening:
    """Exact 3-line PASS shape in the gate (pinned checker unchanged). See PROPOSAL_TESTS. Collected as
    TestProposedHardening (alias below), because pytest only collects classes named Test*."""

    @pytest.mark.parametrize("idx", range(len(_CONTRADICTORY_BODIES)))
    def test_contradictory_body_fails_closed(self, repo, evdir, idx):
        ev = _bound(evdir, body=_CONTRADICTORY_BODIES[idx]())
        assert g.catalog_evidence_outcome(ev, repo, CAND) == (
            "STALE", "evidence is not the exact 3-line PASS output of scripts/live_catalog_check.py")

    @pytest.mark.parametrize("variant", ["plain", "crlf", "blank_lines", "indented"])
    def test_well_formed_bodies_still_pass(self, repo, evdir, variant):
        body = {"plain": _body(), "crlf": _body().replace("\n", "\r\n"),
                "blank_lines": "\n" + _body().replace("\n", "\n\n"),
                "indented": "".join("  " + ln + "\n" for ln in _body().splitlines())}[variant]
        ev = _bound(evdir, body=body)
        assert g.catalog_evidence_outcome(ev, repo, CAND)[0] == "PASSED"

    def test_contradictory_body_fails_through_offline_checks(self, repo, evdir, monkeypatch):
        ev = _bound(evdir, body=_CONTRADICTORY_BODIES[1]())
        assert _offline_a2(repo, ev, monkeypatch) == [
            "evidence is not the exact 3-line PASS output of scripts/live_catalog_check.py"]


TestProposedHardening = ProposedHardening   # collection alias (default python_classes = Test*); one node per test
