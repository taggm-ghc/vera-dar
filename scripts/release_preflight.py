"""Read-only release preflight (DEPLOY-1, owner instruction 2026-10-19).

Runs ONLY the non-destructive, offline release checks and reports PASS / FAIL / OWNER-REQUIRED / SKIPPED per gate.
It never connects to a database, never uses the network, never runs scripts/live_catalog_check.py, never deploys,
never commits, and writes no file itself (--with-tests runs the suite, which creates per-run dirs under tmp/release-gate). It does not replace scripts/release_gate.py, which stays the owner's
authoritative gate; it only pre-computes what can be known without the owner.

    <venv>/bin/python -I -B scripts/release_preflight.py [--evidence PATH] [--evidence-dir DIR]
                                                          [--candidate-sha 40HEX] [--with-tests] [--root DIR]
(--root: tree to check, default this repo; --candidate-sha must equal that tree's HEAD)

Gates: G1 live_catalog_check.py hash pin; G2 evidence-checker hash pin (as pinned by the release gate);
G3 release_gate.py --skip-tests sub-checks (b, c1, c2, c3: frozen files, widget env keys, unsafe html, public mode);
G4 R-028 persistence config (corpus.enabled false, layers L0 only); G5 fulldoc.enabled state (reported, owner flips it);
G6 candidate commit identity and clean worktree; G7 catalog evidence bound to the candidate and the pinned script hash;
G8 full test-suite gate (only with --with-tests, which runs release_gate.py without --live and without --skip-tests).

Refused (exit 2, nothing run): any flag that could start a live run or a deploy, and any unknown argument.
Exit codes: 0 every gate PASS; 1 at least one FAIL; 2 refused/usage; 3 no FAIL but OWNER-REQUIRED or SKIPPED remain.
Output never contains environment values, hosts or evidence content.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_SCRIPT = "scripts/live_catalog_check.py"
CHECKER = "scripts/check_catalog_evidence_freshness.py"
GATE = "scripts/release_gate.py"
CONFIG_CHAIN = "config/ask-provider-chain.json"
CONFIG_FULLDOC = "config/fulldoc_answer.json"
CONFIG_DISCOVERY = "config/discovery_contract.json"
HP1_SCRIPT_SHA256 = "803e46fee67ab03bc0c9e73f35feba72544bf7c6daa8d064a53a812eb0a2fe98"
PASS, FAIL, OWNER, SKIPPED = "PASS", "FAIL", "OWNER-REQUIRED", "SKIPPED"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_REFUSE_WORDS = ("live", "confirm", "owner-run", "deploy", "push", "commit", "force", "apply", "migrate")
_ALLOWED_VALUE = ("--evidence", "--evidence-dir", "--candidate-sha", "--root")
_ALLOWED_BOOL = ("--with-tests",)
_UNSAFE_ENV_PREFIXES = ("VERA_", "PYTHON", "PYTEST", "BASH_ENV", "ENV", "LD_", "DYLD_", "GIT_")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner_commands(sha: str | None) -> list[str]:
    s12 = (sha or "<candidate-sha>")[:12]
    return [
        "# steps 1-3 run in the candidate worktree (clean, committed HEAD), as the owner:",
        f"<venv>/bin/python -I -B scripts/live_catalog_check.py --i-confirm-owner-run > catalog-evidence-{s12}.txt",
        f"# 2. write catalog-evidence-{s12}.txt.sidecar (key=value format in scripts/check_catalog_evidence_freshness.py)",
        "# 3. then the authoritative gate:",
        f"env -i PATH=/usr/bin:/bin HOME=\"$HOME\" USER=\"$USER\" LANG=C.UTF-8 <venv>/bin/python -I scripts/release_gate.py --catalog-evidence catalog-evidence-{s12}.txt "
        f"--expect-candidate-sha {sha or '<candidate-sha>'}",
    ]


def refuse(argv: list[str]) -> str | None:
    """Pure. A reason string if argv must be refused, else None. Only a fixed allowlist of flags passes."""
    i = 0
    while i < len(argv):
        a = argv[i]
        name = a.split("=", 1)[0]
        low = name.lower()
        if name not in _ALLOWED_VALUE + _ALLOWED_BOOL:
            if any(w in low for w in _REFUSE_WORDS):
                return f"refused: {name} could start a live run, a deploy or a write; owner-only, not available here"
            return f"refused: unknown argument {name}"
        if name in _ALLOWED_VALUE and "=" not in a:
            i += 1
            if i >= len(argv):
                return f"refused: {name} needs a value"
        i += 1
    return None


# ---------------------------------------------------------------- gates (each returns (status, note))

def g1_catalog_script_pin(root: Path, expected: str = HP1_SCRIPT_SHA256) -> tuple[str, str]:
    try:
        actual = sha256_file(root / CATALOG_SCRIPT)
    except OSError as exc:
        return FAIL, f"catalog script unreadable ({type(exc).__name__})"
    if actual != expected:
        return FAIL, f"catalog script sha256 {actual[:12]} != approved {expected[:12]}"
    return PASS, f"catalog script sha256 {actual[:12]} matches the approved HP-1 pin"


def _gate_checker_pin(root: Path) -> str | None:
    m = re.search(r'^CATALOG_CHECKER_SHA256\s*=\s*"([0-9a-f]{64})"', (root / GATE).read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def g2_checker_pin(root: Path) -> tuple[str, str]:
    try:
        want = _gate_checker_pin(root)
        actual = sha256_file(root / CHECKER)
    except OSError as exc:
        return FAIL, f"gate or checker unreadable ({type(exc).__name__})"
    if not want:
        return FAIL, "could not read the checker pin from the release gate"
    if actual != want:
        return FAIL, f"checker sha256 {actual[:12]} != release-gate pin {want[:12]}"
    return PASS, f"evidence checker sha256 {actual[:12]} matches the release-gate pin"


def _load_checker(root: Path):
    """Execute exactly the bytes that were hashed against the release-gate pin (one read, no re-read)."""
    import types
    path = root / CHECKER
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("checker")
    want = _gate_checker_pin(root)
    data = path.read_bytes()
    if not want or hashlib.sha256(data).hexdigest() != want:
        raise ImportError("checker is not the pinned version")
    mod = types.ModuleType("_preflight_checker")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


def _clean_env(environ=None) -> dict:
    env = os.environ if environ is None else environ
    keep = {k: v for k, v in env.items() if not k.startswith(_UNSAFE_ENV_PREFIXES)}
    return keep


def default_runner(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), env=_clean_env(), capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout


_LINE = re.compile(r"^\s{2}(PASS|FAIL)\s+(\(\w+\).*)$")


def parse_gate_output(out: str) -> dict[str, str]:
    """name -> PASS|FAIL from release_gate.py output lines; the RESULT line is checked by the caller."""
    return {m.group(2).strip(): m.group(1) for ln in out.splitlines() if (m := _LINE.match(ln))}


def g3_offline_gate(root: Path, python: str, runner=default_runner) -> list[tuple[str, str, str]]:
    """release_gate.py --skip-tests: its own exit is always 1 (diagnostic mode), so judge the per-check lines."""
    rc, out = runner([python, "-I", "-B", str(root / GATE), "--skip-tests"], root, 300)
    checks = parse_gate_output(out)
    if not checks:
        return [("G3", FAIL, "no per-check lines from release_gate.py --skip-tests")]
    return [(f"G3 {n}", st, "") for n, st in checks.items()]


def g4_persistence(root: Path) -> tuple[str, str]:
    """R-028: public-query persistence off. Static config only; not proof of production behaviour."""
    try:
        chain = json.loads((root / CONFIG_CHAIN).read_text(encoding="utf-8"))
        disc = json.loads((root / CONFIG_DISCOVERY).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return FAIL, f"config unreadable ({type(exc).__name__})"
    errs = []
    if (chain.get("corpus") or {}).get("enabled") is not False:
        errs.append("corpus.enabled is not false")
    if disc.get("layers") != ["L0"]:
        errs.append('discovery layers != ["L0"]')
    if errs:
        return FAIL, "; ".join(errs)
    notes = []
    if (chain.get("corpus") or {}).get("licence_observations_enabled") is True:
        notes.append("corpus.licence_observations_enabled is true (owner to confirm it is inert with corpus off)")
    if (disc.get("discovery_memory") or {}).get("write_on_l0") is True:
        notes.append("discovery_memory.write_on_l0 is true (inert only while layers are L0 only)")
    return PASS, "corpus.enabled=false, layers=[L0]" + ("; note: " + "; ".join(notes) if notes else "")


def g5_fulldoc(root: Path) -> tuple[str, str]:
    try:
        cfg = json.loads((root / CONFIG_FULLDOC).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return FAIL, f"config unreadable ({type(exc).__name__})"
    en = (cfg.get("fulldoc") or {}).get("enabled")
    if en is True:
        return PASS, "fulldoc.enabled=true"
    if en is False:
        return OWNER, "fulldoc.enabled=false (owner decides when to enable the full-document path)"
    return FAIL, "fulldoc.enabled missing or not a boolean"


def _git(root: Path, *args: str) -> str | None:
    import shutil
    exe = shutil.which("git", path="/usr/bin:/bin")
    if not exe:
        return None
    env = _clean_env()
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    try:
        p = subprocess.run([exe, "-c", "core.fsmonitor=false", "-C", str(root), *args], capture_output=True,
                           text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def g6_candidate(root: Path, sha: str | None = None, dirty: bool | None = None,
                 head: str | None = None) -> tuple[str, str, str | None]:
    """HEAD is always read from the tree; a supplied --candidate-sha must equal it."""
    if head is None:
        out = _git(root, "rev-parse", "HEAD")
        head = out.strip().lower() if out else None
    if not head or not _HEX40.match(head):
        return FAIL, "cannot determine the candidate commit", None
    if sha is not None and sha.lower() != head:
        return FAIL, f"--candidate-sha {sha[:12]} != HEAD {head[:12]}", None
    if dirty is None:
        st = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        dirty = bool(st) if st is not None else True
    if dirty:
        return OWNER, f"HEAD {head[:12]} but the worktree has uncommitted changes; the owner commits the candidate first", head
    return PASS, f"candidate {head[:12]}, worktree clean", head


def g7_evidence(root: Path, sha: str | None, evidence: Path | None, expected: str = HP1_SCRIPT_SHA256,
               checker_ok: bool | None = None) -> tuple[str, str]:
    if not sha:
        return OWNER, "no candidate commit; evidence cannot be bound"
    if checker_ok is not True:
        return FAIL, "evidence checker is not the pinned file (G2); not executed"
    if evidence is None or not Path(evidence).exists():
        return OWNER, f"no catalog evidence for {sha[:12]} (owner-run; see the owner commands)"
    try:
        chk = _load_checker(root)
        note = chk.check(sha, Path(evidence), script=root / CATALOG_SCRIPT, expected_script_sha256=expected)
    except Exception as exc:  # EvidenceError or anything else: stale, never a pass
        msg = str(exc) if type(exc).__name__ == "EvidenceError" else f"checker error ({type(exc).__name__})"
        return OWNER, f"evidence stale or unbound: {msg}"
    return PASS, note


def g8_tests(root: Path, python: str, evidence: Path | None, sha: str | None, runner=default_runner) -> tuple[str, str]:
    cmd = [python, "-I", "-B", str(root / GATE)]
    if evidence is not None:
        cmd += [f"--catalog-evidence={Path(evidence).resolve()}"]
    if sha:
        cmd += ["--expect-candidate-sha", sha]
    rc, out = runner(cmd, root, 1500)
    last = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
    if rc == 0 and last and re.fullmatch(r"RESULT: PASS \(\d+ checks, 0 failed\)", last[-1]):
        return PASS, last[-1]
    failed = sorted(n for n, st in parse_gate_output(out).items() if st == "FAIL")
    if failed and all(n.startswith("(a2)") for n in failed):
        return OWNER, "suite passed; only the owner-run catalog evidence gate (a2) is not satisfied"
    return FAIL, (last[-1] if last else "no RESULT line") + (": " + ", ".join(failed) if failed else "")


# ---------------------------------------------------------------- driver

def _flag(argv: list[str], name: str) -> bool:
    """True only if name is a flag position, not the value of a value-taking option."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            return True
        if a in _ALLOWED_VALUE:
            i += 1
        i += 1
    return False


def _opt(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def run(argv: list[str], runner=default_runner, out=print) -> int:
    reason = refuse(argv)
    if reason:
        out(reason)
        out("PREFLIGHT: REFUSED (nothing was run)")
        return 2
    root = Path(_opt(argv, "--root") or ROOT).resolve()
    python = sys.executable
    with_tests = _flag(argv, "--with-tests")
    results: list[tuple[str, str, str]] = []
    results.append(("G1 live_catalog_check.py pin", *g1_catalog_script_pin(root)))
    g2 = g2_checker_pin(root)
    results.append(("G2 evidence checker pin", *g2))
    results += g3_offline_gate(root, python, runner)
    results.append(("G4 R-028 persistence config", *g4_persistence(root)))
    results.append(("G5 fulldoc.enabled", *g5_fulldoc(root)))
    st, note, sha = g6_candidate(root, _opt(argv, "--candidate-sha"))
    if st == FAIL:
        sha = None
    results.append(("G6 candidate commit", st, note))
    ev = _opt(argv, "--evidence")
    if ev is None and sha:
        d = Path(_opt(argv, "--evidence-dir") or root)
        cand = d / f"catalog-evidence-{sha[:12]}.txt"
        ev = str(cand) if cand.exists() else None
    results.append(("G7 catalog evidence bound to candidate", *g7_evidence(root, sha, Path(ev) if ev else None,
                                                                    checker_ok=g2[0] == PASS)))
    if with_tests:
        results.append(("G8 full release gate (tests)", *g8_tests(root, python, Path(ev) if ev else None, sha, runner)))
    else:
        results.append(("G8 full release gate (tests)", SKIPPED, "pass --with-tests, or run the suite via the guarded runner"))
    out("VERA-DAR release preflight (read-only, offline)")
    for name, st, note in results:
        out(f"  {st:<14} {name}" + (f": {note}" if note else ""))
    out("Owner-only (not run here):")
    for ln in owner_commands(sha):
        out("  " + ln)
    statuses = {s for _, s, _ in results}
    if FAIL in statuses:
        out("PREFLIGHT: FAIL")
        return 1
    if statuses <= {PASS}:
        out("PREFLIGHT: PASS (owner still authorizes the production release)")
        return 0
    out("PREFLIGHT: OWNER-REQUIRED (no failures; owner-controlled or skipped gates remain)")
    return 3


def main(argv: list[str] | None = None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except BaseException as exc:  # noqa: BLE001 - fail closed with a RESULT-style line
        print(f"PREFLIGHT: FAIL (aborted: {type(exc).__name__})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
