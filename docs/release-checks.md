# Release checks: automated and owner-only

Owner instruction DEPLOY-1 (2026-10-19): document and automate the non-destructive parts of the catalog evidence and
release-gate checks; the production release authorization stays with the owner. No hosts, secrets or evidence content
appear here or in the tool output.

## What is automated (`scripts/release_preflight.py`)

Read-only and offline: no database, no network, no file writes by the tool itself (`--with-tests` runs the suite, which creates per-run dirs under `tmp/release-gate`), never runs `live_catalog_check.py`, never deploys.

    <venv>/bin/python -I -B scripts/release_preflight.py [--evidence PATH | --evidence-dir DIR] [--candidate-sha 40HEX] [--with-tests] [--root DIR]

| Gate | Check | Result when not satisfied |
| --- | --- | --- |
| G1 | `scripts/live_catalog_check.py` sha256 equals the approved HP-1 pin `803e46fee67ab03bc0c9e73f35feba72544bf7c6daa8d064a53a812eb0a2fe98` | FAIL |
| G2 | `scripts/check_catalog_evidence_freshness.py` sha256 equals the pin inside `release_gate.py` | FAIL |
| G3 | `release_gate.py --skip-tests` sub-checks: (b) frozen-file hashes, (c1) no env key in widgets, (c2) no unsafe HTML on model text, (c3) public-mode default | FAIL |
| G4 | R-028 persistence config: `corpus.enabled` false and discovery `layers` equal `["L0"]` (static config, not proof of production behaviour) | FAIL |
| G5 | `fulldoc.enabled` state in `config/fulldoc_answer.json` | OWNER-REQUIRED while false (the owner enables it) |
| G6 | candidate = `git rev-parse HEAD`, worktree clean | OWNER-REQUIRED if dirty (the owner commits the candidate) |
| G7 | catalog evidence for the candidate passes the pinned freshness checker (see binding below) | OWNER-REQUIRED if missing or stale |
| G8 | with `--with-tests` only: full `release_gate.py` (suite under OS network isolation, plus a2) with no `--live` | PASS, OWNER-REQUIRED if only (a2) fails, else FAIL; SKIPPED without the flag |

`--root` selects the tree to check (default: this repo). `--candidate-sha`, if given, must equal that tree's HEAD (else G6 FAIL). Git calls use a fixed PATH, no system/global config, fsmonitor off, untracked=all. G7 refuses to run the checker (FAIL) when G2 fails.

Evidence is looked up at `--evidence`, else `<--evidence-dir, default repo root>/catalog-evidence-<sha12>.txt`.

Exit codes: 0 all PASS; 1 any FAIL; 2 refused; 3 no FAIL but OWNER-REQUIRED or SKIPPED remain. Last line is
`PREFLIGHT: PASS|FAIL|OWNER-REQUIRED|REFUSED`.

Refused with exit 2 and nothing run: `--live`, `--i-confirm-owner-run`, `--deploy`, `--confirm`, `--force`, `--push`,
`--commit` and every unknown argument. Only a fixed allowlist of flags is accepted.

Note: a preflight PASS is not the release gate. Only `release_gate.py` ending in `RESULT: PASS (N checks, 0 failed)`
counts, and the owner authorizes the release.

## Owner-only steps (not run by any agent)

Steps 1-3 run in the candidate worktree.

1. Commit the candidate so the worktree is clean; note `SHA=$(git rev-parse HEAD)`.
2. Live catalog check (read-only, connects through the read-only account; owner-run):

       <venv>/bin/python -I -B scripts/live_catalog_check.py --i-confirm-owner-run > catalog-evidence-<sha12>.txt

   Expected content: first line `live_catalog_check script_sha256=<64 hex>`, then
   `checked N (table, column) pairs; 0 missing`, then exactly one final `RESULT: PASS`. Without the confirm flag it
   prints `RESULT: CANNOT_VERIFY` and exit 2 (a refusal, not a pass).
3. Write `catalog-evidence-<sha12>.txt.sidecar` (format in the docstring of `check_catalog_evidence_freshness.py`):
   `candidate_sha`, `head_after`, `script_sha256`, `evidence_file`, `evidence_sha256`, `exit_code=0`,
   `worktree_clean=1`, `run_start_utc`, `run_end_utc`.
4. Authoritative gate (in the candidate worktree; the gate refuses PYTHON*/GIT_*/LD_*/BASH_ENV/ENV in its environment):

       env -i PATH=/usr/bin:/bin HOME="$HOME" USER="$USER" LANG=C.UTF-8 <venv>/bin/python -I scripts/release_gate.py --catalog-evidence catalog-evidence-<sha12>.txt --expect-candidate-sha <SHA>

   Expected last line: `RESULT: PASS (N checks, 0 failed)`.
5. Enabling `fulldoc.enabled`, deploying and any production change: owner decision, outside these tools.

## How fresh evidence binds to the HP-1 script hash

The evidence must carry the hash of the exact catalog script it ran, and that hash must be the HP-1 value
`803e46fe...` (also the checker default `APPROVED_SCRIPT_SHA256`). The binding is checked four ways: the evidence
body's first line names the script hash; the sidecar `script_sha256` equals it; the on-disk
`scripts/live_catalog_check.py` hashes to it; and the sidecar ties the candidate commit (`candidate_sha` and
`head_after` equal the full 40-hex HEAD, filename prefix equal to the SHA) and the evidence file's own sha256.
Any evidence produced by an earlier script version (a different hash), for a different commit, edited after the
run, or from an unclean worktree is stale: release_gate (a2) FAILs, and the preflight shows OWNER-REQUIRED for it. The legacy `catalog-evidence.txt`
is valid only for the 9c3dd56 fallback candidate. Changing `live_catalog_check.py` or the checker changes the pins
and needs a guarded, owner-approved edit to `release_gate.py`.

## Current status (2026-10-19, uncommitted reconciled tree)

G1, G2, G3, G4 PASS; G5 fulldoc.enabled=false and G6 dirty worktree are OWNER-REQUIRED; G7 has no evidence for the
new script hash, so catalog evidence is stale/missing and OWNER-REQUIRED; G8 SKIPPED unless `--with-tests`.
Overall PREFLIGHT: OWNER-REQUIRED (exit 3). G4 also notes `corpus.licence_observations_enabled` and
`discovery_memory.write_on_l0` are true; the owner confirms they are inert with the Demo Day persistence settings.
