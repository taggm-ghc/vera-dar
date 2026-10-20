"""scripts/release_preflight.py: read-only preflight. Fake evidence/config under tmp_path; no DB, network or live run.
The release gate subprocess is replaced by an injected fake runner (no subprocess in these tests)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("release_preflight", SCRIPTS / "release_preflight.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

CAND = "c" * 40
FAKE = b"# fake catalog script\n"
FAKE_SHA = hashlib.sha256(FAKE).hexdigest()
BODY = f"live_catalog_check script_sha256={FAKE_SHA}\nchecked 3 (table, column) pairs; 0 missing\nRESULT: PASS\n"


def _root(tmp_path: Path, chain_enabled=False, layers=("L0",), fulldoc=False) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config").mkdir()
    shutil.copy(SCRIPTS / "check_catalog_evidence_freshness.py", tmp_path / "scripts")
    (tmp_path / "scripts" / "live_catalog_check.py").write_bytes(FAKE)
    (tmp_path / "scripts" / "release_gate.py").write_text(
        f'CATALOG_CHECKER_SHA256 = "{rp.sha256_file(SCRIPTS / "check_catalog_evidence_freshness.py")}"\n')
    (tmp_path / "config" / "ask-provider-chain.json").write_text(json.dumps({"corpus": {"enabled": chain_enabled}}))
    (tmp_path / "config" / "discovery_contract.json").write_text(json.dumps({"layers": list(layers)}))
    (tmp_path / "config" / "fulldoc_answer.json").write_text(json.dumps({"fulldoc": {"enabled": fulldoc}}))
    return tmp_path


def _evidence(d: Path, sha=CAND, script=FAKE_SHA, clean="1"):
    ev = d / f"catalog-evidence-{sha[:12]}.txt"
    ev.write_text(BODY)
    sc = d / (ev.name + ".sidecar")
    sc.write_text("\n".join([
        "vera-catalog-evidence-sidecar v1", f"candidate_sha={sha}", f"head_after={sha}", f"script_sha256={script}",
        f"evidence_file={ev.name}", f"evidence_sha256={hashlib.sha256(ev.read_bytes()).hexdigest()}", "exit_code=0",
        f"worktree_clean={clean}", "run_start_utc=2026-10-19T10:00:00Z", "run_end_utc=2026-10-19T10:01:00Z"]) + "\n")
    return ev


# ---- refusals
@pytest.mark.parametrize("flag", ["--live", "--i-confirm-owner-run", "--deploy", "--confirm", "--force", "--push"])
def test_live_or_write_flags_refused(flag):
    assert "refused" in rp.refuse([flag])
    out = []
    assert rp.run([flag], out=out.append) == 2
    assert out[-1].startswith("PREFLIGHT: REFUSED")


def test_unknown_and_valueless_refused():
    assert "unknown" in rp.refuse(["--bogus"])
    assert "needs a value" in rp.refuse(["--evidence"])
    assert rp.refuse(["--with-tests", "--evidence", "x", "--candidate-sha=" + CAND]) is None


# ---- pins
def test_pins(tmp_path):
    root = _root(tmp_path)
    assert rp.g1_catalog_script_pin(root, FAKE_SHA)[0] == rp.PASS
    assert rp.g1_catalog_script_pin(root, "0" * 64)[0] == rp.FAIL
    assert rp.g2_checker_pin(root)[0] == rp.PASS
    (root / "scripts" / "check_catalog_evidence_freshness.py").write_text("# edited\n")
    assert rp.g2_checker_pin(root)[0] == rp.FAIL


# ---- config
def test_persistence_and_fulldoc(tmp_path):
    root = _root(tmp_path)
    assert rp.g4_persistence(root)[0] == rp.PASS
    assert rp.g5_fulldoc(root)[0] == rp.OWNER
    bad = _root(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, chain_enabled=True, layers=("L0", "L2"), fulldoc=True)
    st, note = rp.g4_persistence(bad)
    assert st == rp.FAIL and "corpus.enabled" in note and "layers" in note
    assert rp.g5_fulldoc(bad)[0] == rp.PASS


# ---- candidate
def test_candidate():
    assert rp.g6_candidate(Path("."), CAND, dirty=False, head=CAND)[0] == rp.PASS
    assert rp.g6_candidate(Path("."), CAND, dirty=True, head=CAND)[0] == rp.OWNER
    assert rp.g6_candidate(Path("."), "nothex", dirty=False, head="bad")[0] == rp.FAIL


# ---- evidence binding (fake evidence, fake script hash)
def test_evidence_missing_is_owner_required(tmp_path):
    root = _root(tmp_path)
    assert rp.g7_evidence(root, CAND, None, FAKE_SHA, checker_ok=True)[0] == rp.OWNER
    assert rp.g7_evidence(root, CAND, tmp_path / "absent.txt", FAKE_SHA, checker_ok=True)[0] == rp.OWNER


def test_evidence_fresh_passes(tmp_path):
    root = _root(tmp_path)
    ev = _evidence(tmp_path)
    st, note = rp.g7_evidence(root, CAND, ev, FAKE_SHA, checker_ok=True)
    assert st == rp.PASS and CAND[:12] in note


def test_evidence_for_other_candidate_or_old_script_is_stale(tmp_path):
    root = _root(tmp_path)
    ev = _evidence(tmp_path, sha="d" * 40)
    assert rp.g7_evidence(root, CAND, ev, FAKE_SHA, checker_ok=True)[0] == rp.OWNER          # bound to another commit
    ev2 = _evidence(tmp_path, script="1" * 64)
    assert rp.g7_evidence(root, CAND, ev2, FAKE_SHA, checker_ok=True)[0] == rp.OWNER          # sidecar names another script hash
    ev3 = _evidence(tmp_path)
    assert rp.g7_evidence(root, CAND, ev3, "2" * 64, checker_ok=True)[0] == rp.OWNER          # script drifted from approved pin


def test_evidence_unattested_worktree(tmp_path):
    root = _root(tmp_path)
    ev = _evidence(tmp_path, clean="0")
    assert rp.g7_evidence(root, CAND, ev, FAKE_SHA, checker_ok=True)[0] == rp.OWNER


# ---- gate output parsing and test gate with fake runner
GATE_OK = "VERA-DAR release gate\n  PASS  (b) frozen-file integrity\n  FAIL  (c1) x\nRESULT: FAIL (2 checks, 1 failed)\n"


def test_parse_gate_output():
    assert rp.parse_gate_output(GATE_OK) == {"(b) frozen-file integrity": "PASS", "(c1) x": "FAIL"}
    res = rp.g3_offline_gate(Path("."), "py", lambda c, cwd, t: (1, GATE_OK))
    assert [r[1] for r in res] == [rp.PASS, rp.FAIL]
    assert rp.g3_offline_gate(Path("."), "py", lambda c, cwd, t: (1, "REFUSED"))[0][1] == rp.FAIL


def test_tests_gate_outcomes(tmp_path):
    seen = {}

    def ok(cmd, cwd, t):
        seen["cmd"] = cmd
        return 0, "  PASS  (a) x\nRESULT: PASS (7 checks, 0 failed)\n"
    assert rp.g8_tests(tmp_path, "py", None, CAND, ok)[0] == rp.PASS
    assert "--live" not in seen["cmd"] and "--skip-tests" not in seen["cmd"]
    only_a2 = "  PASS  (a) x\n  FAIL  (a2) live catalog evidence (required, owner-run)\nRESULT: FAIL (2 checks, 1 failed)\n"
    assert rp.g8_tests(tmp_path, "py", None, CAND, lambda c, w, t: (1, only_a2))[0] == rp.OWNER
    both = "  FAIL  (a) test suite\n  FAIL  (a2) e\nRESULT: FAIL (2 checks, 2 failed)\n"
    assert rp.g8_tests(tmp_path, "py", None, CAND, lambda c, w, t: (1, both))[0] == rp.FAIL


# ---- whole run: never invokes the catalog script; prints owner command; stale -> exit 3
def test_run_owner_required_and_no_live_command(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "_git", lambda *a: CAND if a[1] == "rev-parse" else "")  # clean worktree, no subprocess
    root = _root(tmp_path)
    cmds = []

    def runner(cmd, cwd, t):
        cmds.append(cmd)
        return 1, "  PASS  (b) frozen-file integrity\nRESULT: FAIL (1 checks, 1 failed)\n"
    out = []
    rc = rp.run(["--root", str(root), "--candidate-sha", CAND], runner=runner, out=out.append)
    text = "\n".join(out)
    # G1 fails on the fake script (approved real pin), so the run reports FAIL; the point here is the shape
    assert rc == 1 and "--i-confirm-owner-run" in text and "catalog-evidence-" + CAND[:12] in text
    assert all("live_catalog_check" not in " ".join(c) for c in cmds)
    assert "G8 full release gate (tests)" in text and rp.SKIPPED in text


def test_candidate_sha_must_equal_head():
    assert rp.g6_candidate(Path("."), "f" * 40, dirty=False, head=CAND)[0] == rp.FAIL
    assert rp.g6_candidate(Path("."), CAND, dirty=False, head=CAND)[0] == rp.PASS
    assert rp.g6_candidate(Path("."), None, dirty=False, head=CAND)[2] == CAND


def test_g7_not_executed_when_checker_unpinned(tmp_path):
    root = _root(tmp_path)
    ev = _evidence(tmp_path)
    assert rp.g7_evidence(root, CAND, ev, FAKE_SHA, checker_ok=False)[0] == rp.FAIL


def test_with_tests_only_as_flag_not_value():
    assert rp._flag(["--with-tests"], "--with-tests")
    assert not rp._flag(["--evidence", "--with-tests"], "--with-tests")


def test_refusal_runs_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "_git", lambda *a: calls.append(a) or "")
    out = []
    for flag in ("--live", "--deploy", "--bogus", "--i-confirm-owner-run"):
        assert rp.run([flag], runner=lambda *a: calls.append(a) or (0, ""), out=out.append) == 2
    assert calls == []


def test_gate_gets_resolved_evidence_value(tmp_path):
    seen = {}
    rp.g8_tests(tmp_path, "py", Path("rel.txt"), CAND, lambda c, w, t: seen.setdefault("c", c) and (1, ""))
    assert any(a.startswith("--catalog-evidence=/") for a in seen["c"])


def test_swapped_checker_is_not_executed(tmp_path):
    root = _root(tmp_path)
    assert rp._load_checker(root).APPROVED_SCRIPT_SHA256  # pinned bytes load
    marker = tmp_path / "ran.txt"
    (root / "scripts" / "check_catalog_evidence_freshness.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\n")  # swapped after the pin was recorded
    with pytest.raises(ImportError):
        rp._load_checker(root)
    assert not marker.exists()
    assert rp.g7_evidence(root, CAND, _evidence(tmp_path), FAKE_SHA, checker_ok=True)[0] == rp.OWNER
    assert not marker.exists()
