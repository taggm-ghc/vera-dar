"""Pre-gate freshness check for owner-run live-catalog evidence (VERA-DAR B5, owner directive 2026-10-17 §2).

D-B5-1 fix X' (post-freeze; integration owner-gated under RB-7): `scripts/release_gate.py` (a2) loads this file,
pinned by sha256, and calls check() with the HEAD it reads itself. It may still run BEFORE the gate as a cross-check.
It closes gap G-9: (a2) alone matched only the script hash and the `RESULT: PASS` line, so any old PASS file for the
same script version passed for any candidate. This checker binds the evidence to ONE candidate SHA.

Stdlib only. No database, no network, no subprocess, no git, no environment variables. It reads only the evidence
file, the sidecar and (optionally) the catalog script's bytes. It never prints evidence content (counts, table names
or anything else), only fixed reasons and short hash prefixes.

    python scripts/check_catalog_evidence_freshness.py --candidate-sha <40-hex> \
        --evidence p3m3/kraken/catalog-evidence-<sha12>.txt [--sidecar <path>] [--script scripts/live_catalog_check.py]

Evidence naming: `catalog-evidence-<12..40 hex of the candidate>.txt`. Sidecar default: `<evidence path>.sidecar`,
written at owner-run time (see SIDECAR FORMAT). The legacy `catalog-evidence.txt` is accepted ONLY for the 9c3dd56
fallback candidate and only if its sha256 equals the recorded fallback value.

SIDECAR FORMAT (key=value, one per line, `#` comments allowed; first non-comment line is the header):
    vera-catalog-evidence-sidecar v1
    candidate_sha=<40 hex>          # HEAD of the worktree the owner ran the script in
    head_after=<40 hex>             # HEAD re-read after the run; must equal candidate_sha
    script_sha256=<64 hex>          # sha256 of scripts/live_catalog_check.py at run time
    evidence_file=<basename>        # basename of the evidence file
    evidence_sha256=<64 hex>        # sha256 of the evidence file after the run
    exit_code=0
    worktree_clean=1                # 1 only if `GIT_OPTIONAL_LOCKS=0 git status --porcelain` at X was empty
    run_start_utc=YYYY-MM-DDTHH:MM:SSZ
    run_end_utc=YYYY-MM-DDTHH:MM:SSZ

Script bytes: main() always hashes a catalog script. `--script` defaults to `live_catalog_check.py` beside this
checker (the candidate's own `scripts/live_catalog_check.py` once installed); a missing script is a FAIL, never a skip.
check() keeps `script=None` for library use only.

Exit codes: 0 PASS, 1 FAIL (reason printed), 2 usage error. The last line is `RESULT: PASS` or `RESULT: FAIL`.

LIMITS (record, Phase E B5-4; read before relying on a PASS):
- On the command line the candidate SHA is operator-supplied. It MUST be `$(GIT_OPTIONAL_LOCKS=0 git -C "$W" rev-parse
  HEAD)` of the worktree in which release_gate.py runs; this checker cannot see git. Inside the gate (X') the gate
  supplies HEAD itself, read before and after the suite on a clean tree.
- Any `--expected-script-sha256` override needs a recorded reason (default is the approved C1 hash).
- The sidecar is owner-attested and UNSIGNED. It defeats accidental reuse, renamed copies, SHA mix-ups and edited
  evidence; it does NOT defeat deliberate forgery by anyone with write access to the evidence directory.
- The legacy path (9c3dd56 only) skips the sidecar and max-age checks by design: it is bound to the recorded C4 bytes.
- Before X' this is a procedural pre-gate control only. Defect D-B5-1 stays OPEN until the owner approves the
  guarded release_gate.py change (RB-7) and X' passes its acceptance tests. Editing this file changes the gate's pin.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Recorded constants (owner-approved values; override on the command line only with a recorded reason).
APPROVED_SCRIPT_SHA256 = "803e46fee67ab03bc0c9e73f35feba72544bf7c6daa8d064a53a812eb0a2fe98"  # C1, 9c3dd56
LEGACY_EVIDENCE_NAME = "catalog-evidence.txt"
LEGACY_CANDIDATE_SHA = "9c3dd5652644cf79ab9918ef5b5bcdab3497b1bb"  # fallback RC (HD-DAR-1)
LEGACY_EVIDENCE_SHA256 = "107bcc269b99a5ad46864f662e5cbb02f37499b034ed55078ad276e6533c4777"  # C4, 2026-10-16
SIDECAR_HEADER = "vera-catalog-evidence-sidecar v1"
SIDECAR_SUFFIX = ".sidecar"

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_NAME = re.compile(r"^catalog-evidence-([0-9a-f]{12,40})\.txt$")
_SCRIPT_LINE = re.compile(r"^live_catalog_check script_sha256=([0-9a-f]{64})$")
_COUNT_LINE = re.compile(r"^checked \d+ \(table, column\) pairs; 0 missing$")
_UTC = "%Y-%m-%dT%H:%M:%SZ"
_REQUIRED_KEYS = ("candidate_sha", "head_after", "script_sha256", "evidence_file", "evidence_sha256", "exit_code",
                  "worktree_clean", "run_start_utc", "run_end_utc")
# main() default for --script: the catalog script beside this checker (scripts/ in the candidate worktree).
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parent / "live_catalog_check.py"


class EvidenceError(Exception):
    """A freshness failure; the message is a fixed, content-free reason."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path, what: str) -> bytes:
    if path.is_symlink():
        raise EvidenceError(f"{what} is a symlink (refused)")
    if not path.is_file():
        raise EvidenceError(f"{what} missing")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{what} unreadable ({type(exc).__name__})") from None


def _check_evidence_body(data: bytes, want_script: str) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceError("evidence is not UTF-8 text") from None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise EvidenceError("evidence is empty")
    m = _SCRIPT_LINE.match(lines[0])
    if not m:
        raise EvidenceError("evidence line 1 is not the live_catalog_check script_sha256 header")
    if m.group(1) != want_script:
        raise EvidenceError(f"evidence script hash {m.group(1)[:12]} != expected {want_script[:12]}")
    results = [ln for ln in lines if ln.startswith("RESULT:")]
    if results != ["RESULT: PASS"] or lines[-1] != "RESULT: PASS":
        raise EvidenceError("no PASS line (evidence must end with exactly one 'RESULT: PASS')")
    if not any(_COUNT_LINE.match(ln) for ln in lines[1:-1]):
        raise EvidenceError("no 'checked N (table, column) pairs; 0 missing' line")


def parse_sidecar(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceError("sidecar is not UTF-8 text") from None
    body = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not body or body[0] != SIDECAR_HEADER:
        raise EvidenceError("sidecar header missing or wrong version")
    fields: dict[str, str] = {}
    for ln in body[1:]:
        key, sep, value = ln.partition("=")
        key, value = key.strip(), value.split("#", 1)[0].strip()
        if not sep or not key:
            raise EvidenceError("sidecar has a malformed line")
        if key in fields:
            raise EvidenceError(f"sidecar repeats key {key}")
        fields[key] = value
    missing = [k for k in _REQUIRED_KEYS if not fields.get(k)]
    if missing:
        raise EvidenceError(f"sidecar missing field(s): {', '.join(missing)}")
    return fields


def _parse_utc(value: str, key: str) -> datetime:
    try:
        return datetime.strptime(value, _UTC).replace(tzinfo=timezone.utc)
    except ValueError:
        raise EvidenceError(f"sidecar {key} is not UTC {_UTC}") from None


def check(candidate_sha: str, evidence: str | Path, sidecar: str | Path | None = None,
          script: str | Path | None = None, expected_script_sha256: str = APPROVED_SCRIPT_SHA256,
          max_age_hours: float | None = None, now: datetime | None = None) -> str:
    """Return a short PASS note, or raise EvidenceError with the first failing reason."""
    cand = (candidate_sha or "").strip().lower()
    if not _FULL_SHA.match(cand):
        raise EvidenceError("candidate SHA must be the full 40-hex commit id")
    if not _HEX64.match(expected_script_sha256 or ""):
        raise EvidenceError("expected script sha256 is not 64 hex")
    ev = Path(evidence)
    want_script = expected_script_sha256
    if script is not None:
        actual = sha256_bytes(_read_regular(Path(script), "catalog script"))
        if actual != want_script:
            raise EvidenceError(f"catalog script hash {actual[:12]} != expected {want_script[:12]}")

    data = _read_regular(ev, "evidence file")
    ev_sha = sha256_bytes(data)

    # 1. Legacy file: only the 9c3dd56 fallback, only byte-identical to the recorded C4 evidence.
    if ev.name == LEGACY_EVIDENCE_NAME:
        if cand != LEGACY_CANDIDATE_SHA:
            raise EvidenceError(f"legacy {LEGACY_EVIDENCE_NAME} is bound to {LEGACY_CANDIDATE_SHA[:12]} only; "
                                f"candidate {cand[:12]} needs fresh owner-run evidence")
        if ev_sha != LEGACY_EVIDENCE_SHA256:
            raise EvidenceError(f"legacy evidence sha256 {ev_sha[:12]} != recorded {LEGACY_EVIDENCE_SHA256[:12]}")
        _check_evidence_body(data, want_script)
        return f"legacy fallback evidence for {cand[:12]} (C4 sha256 {ev_sha[:12]})"

    # 2. Filename must carry the candidate SHA.
    m = _EVIDENCE_NAME.match(ev.name)
    if not m:
        raise EvidenceError("evidence filename must be catalog-evidence-<candidate sha, 12-40 hex>.txt")
    if not cand.startswith(m.group(1)):
        raise EvidenceError(f"filename SHA {m.group(1)[:12]} != candidate {cand[:12]}")

    # 3. Evidence body: script header, count line, exactly one final RESULT: PASS.
    _check_evidence_body(data, want_script)

    # 4. Sidecar written at owner-run time binds SHA, script hash and evidence hash.
    sc_path = Path(sidecar) if sidecar is not None else ev.with_name(ev.name + SIDECAR_SUFFIX)
    fields = parse_sidecar(_read_regular(sc_path, "sidecar"))
    if fields["candidate_sha"].lower() != cand:
        raise EvidenceError(f"sidecar candidate_sha {fields['candidate_sha'][:12]} != candidate {cand[:12]}")
    if fields["head_after"].lower() != cand:
        raise EvidenceError("sidecar head_after != candidate (HEAD moved during the run)")
    if fields["script_sha256"].lower() != want_script:
        raise EvidenceError(f"sidecar script_sha256 {fields['script_sha256'][:12]} != expected {want_script[:12]}")
    if fields["evidence_sha256"].lower() != ev_sha:
        raise EvidenceError(f"sidecar evidence_sha256 {fields['evidence_sha256'][:12]} != file {ev_sha[:12]}")
    if fields["evidence_file"] != ev.name:
        raise EvidenceError("sidecar evidence_file != evidence basename")
    if fields["exit_code"] != "0":
        raise EvidenceError("sidecar exit_code is not 0")
    if fields["worktree_clean"] != "1":
        raise EvidenceError("sidecar worktree_clean is not 1 (run was not attested on a clean worktree)")
    start = _parse_utc(fields["run_start_utc"], "run_start_utc")
    end = _parse_utc(fields["run_end_utc"], "run_end_utc")
    if end < start:
        raise EvidenceError("sidecar run_end_utc is before run_start_utc")
    ref = now or datetime.now(timezone.utc)
    if end > ref:
        raise EvidenceError("sidecar run_end_utc is in the future")
    if max_age_hours is not None and (ref - end).total_seconds() > max_age_hours * 3600:
        raise EvidenceError(f"evidence older than {max_age_hours:g} h (stale)")
    return f"evidence bound to {cand[:12]} (sha256 {ev_sha[:12]}, run end {fields['run_end_utc']})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check that catalog evidence is fresh and bound to the candidate SHA.")
    ap.add_argument("--candidate-sha", required=True, help="full 40-hex release-candidate commit")
    ap.add_argument("--evidence", required=True, help="path to the owner-run evidence file")
    ap.add_argument("--sidecar", default=None, help=f"binding sidecar (default: <evidence>{SIDECAR_SUFFIX})")
    ap.add_argument("--script", default=None,
                    help="candidate's scripts/live_catalog_check.py (default: live_catalog_check.py beside this checker)")
    ap.add_argument("--expected-script-sha256", default=APPROVED_SCRIPT_SHA256)
    ap.add_argument("--max-age-hours", type=float, default=None, help="optional: reject evidence older than this")
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    script = a.script if a.script is not None else DEFAULT_SCRIPT_PATH  # always hash the script bytes (B5-2)
    try:
        note = check(a.candidate_sha, a.evidence, a.sidecar, script, a.expected_script_sha256.lower(),
                     a.max_age_hours)
    except EvidenceError as exc:
        print(f"catalog evidence freshness: {exc}")
        print("RESULT: FAIL")
        return 1
    print(f"catalog evidence freshness: {note}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
