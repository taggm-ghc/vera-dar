# Release checks: automated and owner-only

Owner instruction DEPLOY-1 (2026-10-19): document and automate the non-destructive parts of the catalog evidence and
release-gate checks; the production release authorization stays with the owner. No hosts, secrets or evidence content
appear here or in the tool output.

**Status as of 2026-10-30:** this file owns gate status. The release gate has NOT passed for `main`; the latest recorded run, on a candidate ahead of `main`, failed check (a2) (live catalog evidence). Migration 007 and the related change C1 are pending owner decision; neither is done. Sections headed "history" below are dated records, kept as written.

**Update 2026-11-05 (DOCS-1105-REPO).** Nothing above is superseded: the six checks and the (a2) rule are unchanged, and the release gate has still not passed for `main` (no (a2) evidence for any current candidate is on record). New since 2026-10-30, from `work/RELEASE-VERIFY-1104/`:
- Pre-commit rehearsals (candidate built in scratch, offline tests plus static scans, no live behaviour): the earlier results (e.g. `result-20261105T070856Z.md`, repo HEAD 2054f12) were FAIL (acceptance group S-F: 8 failing probe tests; secret/host scan: 1 hit to triage); the latest listed result, `result-20261105T072514Z.md`, records Verdict PASS with the full suite at 9246 passed, 39 skipped, 12 xfailed, and states "Gate (a2) not run (owner evidence)". That result names repo HEAD 35f21fb, an ancestor of the cd81eab this documentation pass reads (one commit later adds the official-sources lane, licence allowlist and digest storage); it was not re-run here. It is a rehearsal, not the release gate.
- Owner-run wrappers (agents do not run them): `work/RELEASE-VERIFY-1104/owner_release_gate.sh` (derives the SHA from `repo/` HEAD, refuses the pre-MVP base, then runs `work/RELPREP-1031/run_release_gate.sh`: clean worktree, hidden prompt for the read-only database URL, one read-only live catalog check for (a2), then the full gate) and `work/RELEASE-VERIFY-1104/owner_deploy_check.sh` (post-deploy `/version` check). Checksums: `work/RELEASE-VERIFY-1104/SHA256SUMS`.
- (a2) and migration 015: `live_catalog_check.py` checks (table, column) pairs; whether the 015 tables are in its pair list was not checked in this pass. Migrations 007/009/013/014 stay unapplied, so any pair belonging to them blocks (a2) for as long as the code requires it.
- The 2026-10-30 sentence that migration 007 and change C1 are "pending owner decision" is unchanged: no decision is on record in this tree.

<!-- TODO(DOCS-1030 D-REL): after the owner disposes the pending C1 decision, re-touch this file (gate status, the (a2) paragraph and the 007 reference) and the README gate sentence. Do not edit before that decision. -->

## The release gate: six checks (`scripts/release_gate.py`, as of 2026-10-29)

Read from the code (`run_offline_checks`, `release_gate.py:631-649`); the module docstring still lists only (a)-(c) plus (d) and omits (a2), so follow this table and the code. The gate must be run isolated (`python -I`); it exits 1 with `RESULT: FAIL (gate not run isolated)` otherwise.

| Check (key printed by the gate) | Function | What it requires |
| --- | --- | --- |
| (a) test suite | `check_tests` | Full suite, with nonce-bound proof of OS-level network isolation |
| (a2) live catalog evidence (required, owner-run) | `check_live_catalog_evidence` | Owner-run output of `scripts/live_catalog_check.py` bound to the candidate commit and passing the pinned checker `scripts/check_catalog_evidence_freshness.py` (see binding below) |
| (b) frozen-file integrity | `check_frozen` | sha256 of the frozen scope set and trace questions match their recorded hashes |
| (c1) no env key in widgets | `check_no_env_in_widgets` | Static check of UI files |
| (c2) no unsafe_allow_html on model text | `check_no_unsafe_html` | Static check of UI files |
| (c3) public mode default | `check_public_default` | Run in an isolated child process |

A full run prints `RESULT: PASS (6 checks, 0 failed)`. Flags: `--catalog-evidence PATH`, `--expect-candidate-sha FULL40`, `--skip-tests` (diagnostics only; forces a failed result) and `--live` (adds a seventh check, (d) scope router layer 1, `check_live_scope`; not part of the normal gate). Only the exact last line `RESULT: PASS (N checks, 0 failed)` counts.

**(a2) has no waiver, exemption or pending-list mechanism in the code** (observation as of 2026-10-29: no such option exists in `release_gate.py`). Missing, stale, edited or wrong-commit evidence, or evidence whose own `RESULT` is `FAIL`, fails the gate. Evidence that reports missing (table, column) pairs therefore blocks the gate until the database is changed by the owner or the schema code stops requiring them; documents cannot override it.

Reference observation (evidence file read, not re-run): the (a2) evidence recorded for candidate b863375 ended `RESULT: FAIL` with 13 of 158 (table, column) pairs missing, all belonging to the unapplied domain-registry migration 007 (research_domains (5 columns), domain_corpus (4) and domain_chunks (4), all from draft migration 007). b863375 is two commits ahead of main d6681ae, so this says nothing about main passing. The decision on 007 is pending the owner (see `db/README.md`).

Owner runbook: the steps are summarised under "Owner-only steps" below; the owner keeps the full runbook outside this repository. Keep `catalog-evidence-<sha12>.txt` and its `.sidecar` outside the repo tree so the worktree stays clean; the steps below summarise it.

## What is automated (`scripts/release_preflight.py`)

Read-only and offline: no database, no network, no file writes by the tool itself (`--with-tests` runs the suite, which creates per-run dirs under `tmp/release-gate`), never runs `live_catalog_check.py`, never deploys.

    <venv>/bin/python -I -B scripts/release_preflight.py [--evidence PATH | --evidence-dir DIR] [--candidate-sha 40HEX] [--with-tests] [--root DIR]

| Gate | Check | Result when not satisfied |
| --- | --- | --- |
| G1 | `scripts/live_catalog_check.py` sha256 equals the approved HP-1 pin `803e46fee67ab03bc0c9e73f35feba72544bf7c6daa8d064a53a812eb0a2fe98` | FAIL |
| G2 | `scripts/check_catalog_evidence_freshness.py` sha256 equals the pin inside `release_gate.py` | FAIL |
| G3 | `release_gate.py --skip-tests` sub-checks: (b) frozen-file hashes, (c1) no env key in widgets, (c2) no unsafe HTML on model text, (c3) public-mode default | FAIL |
| G4 | R-028 persistence config: `corpus.enabled` false and discovery `layers` equal `["L0"]` (static config, not proof of production behaviour) | FAIL |
| G5 | `fulldoc.enabled` state in `config/fulldoc_answer.json` | Shipped as true at 8b16bf3; the corpus source and deploy check are G5b (OWNER-REQUIRED) |
| G5b | deploy-time owner check (not run by the script; `scripts/` is dockerignored): `fulldoc.enabled=true` and a corpus source is configured. The path is OFF unless one is set: a local bundle directory named by `VERA_FULLDOC_BUNDLE_DIR`, or database-backed documents selected by the run id in `VERA_FULLDOC_DB_RUN_ID` (acquired documents are served only beside the DB-backed base corpus, and configuring both sources is refused: the corpus is reported unusable with code R-SOURCE, FACTS a3). The API-key-protected `/health/dar` must report `fulldoc.status` = `ok` (codes and counts only; `not_configured` or `unusable` means the full-document path is inert or fails closed). Config read only; live behaviour not verified | OWNER-REQUIRED |
| G6 | candidate = `git rev-parse HEAD`, worktree clean | OWNER-REQUIRED if dirty (the owner commits the candidate) |
| G7 | catalog evidence for the candidate passes the pinned freshness checker (see binding below) | OWNER-REQUIRED if missing or stale |
| G8 | with `--with-tests` only: full `release_gate.py` (suite under OS network isolation, plus a2) with no `--live` | PASS, OWNER-REQUIRED if only (a2) fails, else FAIL; SKIPPED without the flag. This is the preflight's classification only: the gate itself has no waiver for (a2) |

`--root` selects the tree to check (default: this repo). `--candidate-sha`, if given, must equal that tree's HEAD (else G6 FAIL). Git calls use a fixed PATH, no system/global config, fsmonitor off, untracked=all. G7 refuses to run the checker (FAIL) when G2 fails.

Evidence is looked up at `--evidence`, else `<--evidence-dir, default repo root>/catalog-evidence-<sha12>.txt`.

Exit codes: 0 all PASS; 1 any FAIL; 2 refused; 3 no FAIL but OWNER-REQUIRED or SKIPPED remain. Last line is
`PREFLIGHT: PASS|FAIL|OWNER-REQUIRED|REFUSED`.

Refused with exit 2 and nothing run: `--live`, `--i-confirm-owner-run`, `--deploy`, `--confirm`, `--force`, `--push`,
`--commit` and every unknown argument. Only a fixed allowlist of flags is accepted.

Note: a preflight PASS is not the release gate. Only `release_gate.py` ending in `RESULT: PASS (N checks, 0 failed)`
counts, and the owner authorizes the release.

## Owner-only steps (not run by any agent)

Steps 1-3 run in the candidate worktree. Write the evidence and sidecar files OUTSIDE the repo tree (for example the parent directory of the worktree) and pass that path in step 4: an untracked file in the worktree makes it "not clean" and fails (a2). The redirect targets shown in steps 2-4 are file names only; prefix them with that outside directory.

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
5. Changing `fulldoc.enabled` (true in the shipped config), deploying and any production change: owner decision, outside these tools.

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

## Preflight status record, history (recorded 2026-10-19; HISTORICAL, superseded by the gate status at the top of this file)

Status as of 2026-10-22: the table below was last recorded 2026-10-19 and has not been re-run in this documentation pass (untested, assumed unchanged). The tree has since had commits (HEAD 50767a2 on 2026-10-21 per local git) and uncommitted edits, so G6 and G7 must be re-evaluated for any candidate.

G1, G2, G3, G4 PASS; G5 fulldoc.enabled was false as of 2026-10-19 (shipped true at 8b16bf3; see G5 row above) and G6 dirty worktree are OWNER-REQUIRED; G7 has no evidence for the
new script hash, so catalog evidence is stale/missing and OWNER-REQUIRED; G8 SKIPPED unless `--with-tests`.
Overall PREFLIGHT: OWNER-REQUIRED (exit 3). G4 also notes `corpus.licence_observations_enabled` and
`discovery_memory.write_on_l0` are true; the owner confirms they are inert with the persistence settings of that date (Demo Day, 2026-10-23, has since passed).
