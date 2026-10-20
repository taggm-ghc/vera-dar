# VERA-DAR vera_vjay schema (M1-M7)

*Annotated 2026-10-01 (R7a author, R11a records) and 2026-10-02 (R11a, facts checked against the migrations and code): the STATUS block below is the 2026-09-30 M1 record and is kept; later state is in the "Current state" block that follows it. Pre-edit copies (plain paths in the sibling repo's gitignored `p3m3/history/`): `2026-10-01-cleanup-archive-6/vera__db-README.md`, `2026-10-01-cleanup-archive-10/vera-repo-db-README.md`.*

STATUS (2026-09-30 22:05 PDT): DEPLOYED to production as an in-place upgrade (not a reset).
`001u_vera_upgrade_in_place.sql` + `003_vera_m3_plus_tables.sql` applied; round-trip test 17/17 PASS in 5s.
The 20 pre-existing candidates, 1 question and 1 run are intact (checksum identical before/after;
JSON backup at ~/tmp/vera_vjay_pre_upgrade_backup_2026-09-30.json, mode 600).
`001` + `002` are for FRESH installs only (001 refuses to drop a schema that holds data); on this
database the upgrade replaced them. Differences on the upgraded DB: ids are BIGINT with serial-style
defaults (not IDENTITY); legacy candidates have `gate_a_decision` NULL = "not recorded" and the
"scored => decision" CHECK is NOT VALID (enforced for new/updated rows only). Existing `idx_*` indexes kept.
M2 code writing Gate A scores MUST now also set `gate_a_decision` (fetch/defer/reject).
Deployment order for this DB: done. For a fresh DB: 001, 002, then 003 at M3.

## Current state (2026-10-01)
- Per the 2026-09-30 read-only catalog check (Round 1 summary), the live `vera_vjay` schema has 13 tables:
  migrations 001-004 are applied, not only M1. The "NOT YET DEPLOYED" wording in the 003 and 004 `.sql` headers was stale and was corrected 2026-10-02 (comment-only edits; no SQL statement changed).
- Migration 005 (`005_vera_m6_schema_sync.sql`, B1) columns exist live, verified read-only 2026-10-01 (see the
  last section). The working-tree 005 differs from the committed one: the `evidence_relations.run_id` backfill was
  removed (spans have no `run_id`; the table was empty at apply time). M4 owns setting `run_id` on new rows;
  that ownership is to be confirmed by R1. The 005 header comment was corrected 2026-10-02 to say so (comment-only).
- Three login accounts were applied 2026-10-01 (see "Accounts and password hashing"). The pipeline runners are
  not yet repointed to them; the write-only grants were exercised for `/ask` corpus admission on 2026-10-07 (see
  "Update 2026-10-07" below).
- Migration states, read from the files 2026-10-02: 001 and 002 are fresh-install files (not applied on this database; 001u replaced them); 001u applied; 003 and 004 applied; 005 applied. The 005 grants (UPDATE on `appraisals` and `evidence_relations`, UPDATE on `runs` for the rw group) mean those two tables are **not** append-only by grant; only `sources` and `canonical_sources` are immutable (trigger plus grants).
- Dev and production share this one schema (account distinction only). Nothing under `db/` is committed
  except what B1 committed; commit only after the owner validates.

## Files
| File | Purpose |
|---|---|
| `migrations/001u_vera_upgrade_in_place.sql` | Non-destructive upgrade of the original-shape schema (applied); partly guarded, apply once |
| `migrations/001_vera_schema_init.sql` | Fresh installs only: DROP/CREATE schema; 6 core tables; immutability trigger; roles + grants. Not re-runnable |
| `migrations/002_vera_indices.sql` | Indexes, each with its purpose (plain `CREATE INDEX`, not re-runnable) |
| `migrations/003_vera_m3_plus_tables.sql` | M3-M7 tables (applied to the live schema per the 2026-09-30 catalog check). **Not re-runnable** (plain `CREATE TABLE`) |
| `migrations/004_vera_m4_m5_additions.sql` | M4/M5 additions (`gate_c_decisions`, `claims.detailed_status/issues`, `runs.gate_c_decision/verification_status/final_answer_id`); replaces the deleted standalone M4/M5 file. **Not re-runnable** (plain `CREATE TABLE`/`ADD COLUMN`) |
| `migrations/005_vera_m6_schema_sync.sql` | B1: M6 columns (`runs.question/final_response/evaluated_at`, `appraisals.rubric_version/policy_version`, `evidence_relations.run_id/rationale`) plus UPDATE grants. Re-runnable (IF NOT EXISTS guards); **no `evidence_relations.run_id` backfill** (M4 owns setting it, R1 to confirm); live columns verified 2026-10-01 |
| `accounts/create_vera_accounts.sql` | Not a migration (`apply_migrations.py` does not pick it up). Creates the 3 login accounts; applied 2026-10-01; contains no passwords |
| `apply_migrations.py` | Subcommands `plan` (alias `--dry-run`; never connects), `inspect`, `rehearse` (local only), `apply`, `rollback` (inspect only, never executes). One transaction per file plus a `schema_migrations` row. See "Migration tool" below |
| `migration_guard.py` | Stdlib-only id validation, exact file resolution, SQL lint, offline target classification |
| `test_m1_roundtrip.py` | Round-trip test; always cleans up its rows |

## Migration order
- Fresh database: 001, 002, 003, 004, 005, in that order. This database was upgraded instead: 001u, then 003, 004, 005 (all applied; nothing is pending).
- Idempotency: only 005 (and parts of 001u) can be re-run safely. **Do not re-run 003 or 004** against the live schema; they fail on existing objects. 001 refuses to drop a schema that holds data.
- *Superseded by "Migration tool" below (historical record of the 120ab53 tool):* run any migration only as the admin identity and after a dry run (the OLD `--dry-run` ran the file then rolled back, and the old script imported `db.get_admin_engine` from week-1v2). The hardened tool's `--dry-run` never connects, and it never imports week-1v2 code.
- Manual alternative: `psql -1 -f <file>` (single transaction).

## Migration tool (hardened; added with the post-release migration-hardening change)
- Correction to the order notes above: `--dry-run` no longer executes anything. It is an alias of `plan`, which never connects
  and never reads a password. Execute-then-roll-back is `rehearse`, allowed only on a verified local database
  (`vera_local_*`, loopback/socket, marker table and token). Migration 006 is applied (see "Update 2026-10-12").
- Ids are exactly three digits. Only top-level `migrations/NNN_vera_*.sql` files are reachable; `pending/` and
  `rollback/` never are, and paths, globs, suffixes, symlinks, duplicates and unknown options are refused.
- Production `apply` needs `--target production`, `--owner-approval OD-<id>`, `VERA_MIGRATE_PRODUCTION=I_ACCEPT_OD-<id>`
  in the process environment only, an interactive terminal, a typed database name and id, and the admin password typed
  at a hidden prompt. It is refused if `DB_ADMIN_PASSWORD` is in the environment or any env file the tool can see.
  It is also refused if any DB URL carries a password or any `PG*` variable is set, and unless every file's sha256 is
  given with `--approve-sha NNN=<sha256>` and matches the reviewed bytes (recorded as `approval_ref = OD-<id>@<sha12>`).
  The SQL lint is an advisory pre-check, not a safety certification: the gate is human review of the exact file.
  These are accident and audit controls; the barrier against misuse is the owner's custody of the admin password.
- Bootstrap: `migrations/008_vera_schema_migrations.sql` is the only file accepted while the tracking table is absent.
  `pending/006` stays in `pending/` until the owner-run baseline (catalog-verified, Approval 4) records it.
- Options are per subcommand; any option a subcommand does not take is refused, never ignored. `--env-file` was removed
  (K18W INC-2). `--approve-sha`, when given, is enforced on every subcommand, including `plan` and local runs.
- Until owner-run baseline rows (origin `baseline`/`superseded`, step D-5) exist, only the bootstrap is accepted, even
  when the tracking table exists (K18W INC-2).
- K18W INC-3:
  - The bootstrap is recognised by exact pinned identity: the name `008_vera_schema_migrations.sql` AND its reviewed
    sha256, pinned as `BOOTSTRAP_SHA256` in `migration_guard.py`. A look-alike name, another number or changed bytes is
    not the bootstrap. Any edit to 008 needs a reviewed update of the pin.
  - Each per-file `apply` transaction starts with `SET TRANSACTION READ WRITE`. This is the only write opt-in, for a
    database whose default is read-only. It is issued after every tool-side gate and before the in-transaction lock
    and recheck. `plan`/`--dry-run`, `inspect`, `rehearse` and `rollback` never opt in, so under a read-only default
    `rehearse` fails closed. The lint hard-refuses any migration body that changes the read/write mode itself
    (`SET SESSION CHARACTERISTICS`, `default_transaction_read_only`/`transaction_read_only` in any form including
    `set_config(...)`, `READ WRITE`). It also hard-refuses `COPY ... PROGRAM`, `ALTER SYSTEM`, any `set_config(`, and
    role or authorization changes in any position. These are statement-order controls checked with mocks only; the
    Postgres behaviour (SQLSTATE 25001/25006) is UNVERIFIED.
  - Header warnings such as "Never run via db/apply_migrations.py" are found even when they wrap across comment lines.
- K18W INC-4: the lint also hard-refuses `ALTER ROLE|USER|DATABASE ... [IN DATABASE ...] RESET ALL`, which would
  remove a role- or database-level `default_transaction_read_only=on`, `RESET ALL` anywhere else (including inside
  literals and nested comments, fail-closed), and any reference to `pg_db_role_setting`. Renaming or creating a role
  or database is reported as ADVISORY. The never-via header check also catches a path split across lines or written
  in quotes or backticks. `WITH ... UPDATE` and an UPDATE whose only WHERE is inside a subquery are flagged as
  UPDATE WITHOUT WHERE. The lint is still advisory and string-based: quoting tricks (`U&"..."`, quoted function
  names, concatenated or `format()`-built dynamic SQL, `"a--b"`/`"a/*b"` identifiers, an apostrophe inside a
  quoted identifier, `E''` escapes, comments the strippers do not model) can evade it, and a data-modifying CTE
  (`WITH x AS (UPDATE ...)`) is not flagged. The real enforcement is DB-side (read-only defaults, role
  privileges) plus owner review of the sha-bound bytes.
  - Credential tripwires also check percent-decoded URL values (`?%70assword=`). `KRB5*` and `GSS*` variables are
    refused like `PG*`. Connections send no client certificate (`sslcertmode=disable`), use no GSS encryption
    (`gssencmode=disable`), and set `require_auth` (production `scram-sha-256`, local `none`), so a Kerberos ticket or
    client certificate cannot stand in for the typed password. This needs libpq 16 or later; older libpq fails closed.
    It is UNVERIFIED against the real driver.
  - If COMMIT raises, the record says `OUTCOME UNKNOWN for <file>`; check `schema_migrations` read-only before any retry.
    An output error after a successful commit is reported as FAILED with that file in the committed list.
  - Local `apply` still runs without `--approve-sha` (the binding is optional on local, and enforced when given).
- The tool never executes a rollback file. `rollback <id>` only shows what the file would drop. Rollbacks are owner-run on
  an isolated database (`psql -1 -f`) with a separate same-turn approval, never on production for an applied migration.

## Round-trip test
`.venv/bin/python ../../../ai-eng-bootcamp.vera/db/test_m1_roundtrip.py` (needs a database and credentials) from the week-1v2 directory.
Writes to PRODUCTION (EXTERNAL_DB_URL and INTERNAL_DB_URL are one instance). It creates a question ->
run -> search iteration -> candidate (Gate A 0.85/fetch) -> fetches https://example.com (offline
fallback text if unreachable) -> stores content + sha256 -> joins back and recomputes the hash ->
checks immutability and CHECK constraints -> deletes every test row (marker `M1-ROUNDTRIP-TEST-<uuid>`)
and asserts zero remain. Prints PASS/FAIL per check and total time. Cleanup runs even on failure.
Identity sequences advance; that is not reversible and harmless.

## Credentials
`db.get_admin_engine()` reads `week-1v2/.env.db-accounts` (DB_ADMIN_ROLE, DB_ADMIN_PASSWORD, host, DB_NAME).
Accepted divergence: the admin login is Render's original `vera_vjay_user` (CREATEROLE/CREATEDB);
`vera_vjay_dbadmin` does not exist and Render cannot rename roles. DDL uses it. The database is also
named `vera_vjay`, and the schema is `vera_vjay`; AI-Internship objects live in schema `internship`
and are untouched. Migration 001 creates NOLOGIN group roles `vera_vjay_rw` / `vera_vjay_ro` only (no
passwords). The login accounts the pipeline should use now exist (see "Accounts and password hashing"); the
earlier plan to create a further `vera_api_rw` login is not to be carried out (no more accounts).

## Schema walkthrough (every table serves a gate or an audit need)
- `questions`: governing question + Evidence Requirements Map (JSONB). Audit anchor.
- `runs`: one execution; baseline vs engineered response, `eval_metrics` JSONB (M6).
- `search_iterations`: each query; Gate C re-search = new `iteration_no`, `triggered_by_gate_c`,
  `targets_requirement` (the unmet requirement it addresses). `rank` = priority among planned queries.
- `candidates`: search results. `gate_a_decision` (fetch/defer/reject) is separate from `fetch_status`
  (pending/fetched/failed). Scored rows must carry a decision; rejected rows cannot be fetched.
  `gate_a_uncertainty` preserves "unknown is not zero". Unique per (run, URL).
- `canonical_sources`: stable source identity by URL, so re-fetching never collides.
- `sources`: immutable versions: `content`, `content_hash`, `hash_algorithm`, `version`, `provenance`.
  Unique (canonical, version) and (canonical, hash). App role: SELECT/INSERT only; an UPDATE trigger
  blocks every role including admin. Changed content = new version row.
- 003 (applied; see Current state): `evidence_spans`, `appraisals` (0-5 scores, GRADE-style quality, Gate B decision),
  `evidence_relations`, `reasoning_context`, `answers` (stage, cost, parent), `claims`
  (`verification_status` indexed). UNLOGGED was rejected: it is truncated on crash, wrong for an audit store.

## Assumptions
- `gate_a_score`/uncertainty/relevance/confidence 0-1; appraisal dimension scores 0-5 (NULL = unevaluated);
  `overall_quality` is categorical (high/moderate/low/very_low) as in GRADE.
- `sources.content` holds extracted text (assumed under a few MB per source; TOAST handles it). PDFs/binaries
  would need extracted text here and raw bytes elsewhere.
- sha256 only for now (CHECK-enforced; widen the CHECK to add algorithms).
- IDs are `BIGINT GENERATED ALWAYS AS IDENTITY` (not SERIAL). FKs are ON DELETE RESTRICT; delete leaf-first.
- `evidence_span_ids` on claims is JSONB per the brief, so no FK enforcement.

## Remaining coordination
The standalone `001_m4_m5_vera_vjay.sql` (TEXT run_id, no FKs) no longer exists: it was replaced by
`004_vera_m4_m5_additions.sql` (per the 004 header). Nothing is left to reconcile with it. Still open: R1 to confirm that
M4 owns the removed 005 `evidence_relations.run_id` backfill; the evaluation-design schema additions (`as_of_quarter`,
`run_type`, config fingerprint, per-sub-test rows) are not written (see `VERA-EVAL-BEST-PRACTICES.md`, O7).

## Accounts and password hashing (applied 2026-10-01)
Three logins (created by `db/accounts/create_vera_accounts.sql`, applied 2026-10-01; the script is idempotent): `vera_claude_code_rw` (dev, read/write),
`vera_pipeline_wo` (production, write-only: INSERT/UPDATE plus SELECT on identity/key columns only, no DELETE, and no UPDATE on `sources`/`canonical_sources`), `vera_eval_ro` (production, read-only).
The runners are **not yet repointed** to these accounts (they still use the admin identity), and the write-only grants are untested against a live run:
`vera/m2_runner.py` reads columns outside the granted key columns (`candidates.source_url`, `canonical_sources.canonical_url`, `sources.content_hash`), so `run_m2` probably cannot run under `vera_pipeline_wo` (static reading, not run live). Passwords live in
the gitignored `.env` as `VERA_DB_PASSWORD_<role>`; `.env.example` documents the accounts without secrets.

Best practice is SCRAM-SHA-256 (MD5 is deprecated). Status quo: the Render server's `password_encryption` is `md5` and can't be
changed by us, so always run `SET password_encryption = 'scram-sha-256';` in the same session before `ALTER ROLE ... PASSWORD`
(the three accounts were re-saved this way on 2026-10-01). Accepted divergence: the server default stays md5; the hash type is not
catalog-verified because `pg_authid` is denied to the admin account.

Verification facts (2026-10-01): applied via the admin `vera_vjay_user`; privilege matrix 9/9 as expected; all three logins tested. Only these 3 accounts were created (no more are to be created).

Update 2026-10-07:
- **Write-only grants exercised.** `/ask` corpus admission (`vera/corpus_admission.py`) ran as `vera_pipeline_wo` in a rolled-back transaction. INSERT ... RETURNING works on questions, runs, candidates, canonical_sources and sources, and UPDATE works on candidates. `ON CONFLICT` is refused (it needs SELECT on the conflict columns), so the code uses savepoints and treats a unique violation as "exists".
- **Password rotated.** `vera_pipeline_wo`'s password was set by the admin as a SCRAM verifier.
- **.env repointed.** VERA-DAR's `.env` now points `EXTERNAL_DB_URL`, `INTERNAL_DB_URL` and `VERA_DB_URL_RW` at `vera_claude_code_rw` and `VERA_DB_URL_WO` at `vera_pipeline_wo`. They previously used an AI-Internship account, which is refused on `vera_vjay`.
- **Read-only URL added.** `VERA_DB_URL_RO` (`vera_eval_ro`) was added for the corpus description (`GET /corpus-summary`, item #83). The read-only SQL contract test now uses VERA-DAR's own `EXTERNAL_DB_URL` host, not another project's file.
- **Still open.** `run_m2` under the write-only account (see above) is still unrun. Column grants of `vera_pipeline_wo` are UNTESTED against a live `m2_runner.py` run. Dev and prod share one schema (accepted divergence).

## Live verification and contract test (2026-10-01)
- B1 (migration 005, commit 28aae11) and B2 (eval freeze, 9fef487) verified live, read-only: the 005 columns exist (`runs.run_label`, `evaluated_at`, `engineered_score`, `baseline_score`, `question`; `appraisals.rubric_version`/`policy_version`; `evidence_relations.run_id`); the frozen question text equals `questions.question_id=2`. `question_id=7` is a test row pending an approved deletion.
- C4: `tests/test_sql_contract.py` checks 87 (table, column) pairs against `vera_vjay`; all exist. M3's store writes `rubric_version`/`policy_version` to columns. `pytest tests`: 178 passed, 2 skipped, as of 2026-10-01, not re-run since.
- Sources are stored here as immutable rows; no raw external sources in the repo (see `docs/vera-design.md` invariants).

## What the pipeline persists, and what it does not (code read 2026-10-02; live rows not queried)
- Persisted: `gate_c_decisions` (decision and full detail per attempt; thresholds are code constants and are not stored with the rows); `claims.detailed_status` and `issues` (M5 fine-grained status); answers by stage with tokens, cost and latency; relations (`span1_id`, `span2_id`, `relation_type`, `confidence` only; `rationale` and `run_id` are not written by that insert); appraisals with rubric/policy version.
- Not persisted: M2 search cost (returned in memory; appended to the file named by `$VERA_COST_LOG` only if set); the appraisal values and years behind an M5 `weak` label; relations dropped by the 60-cap or as malformed.
- Not created by any production code: `runs` rows (only tests insert them), and `sub_question_ids` (read by Gate C, set by nothing). The pipeline therefore cannot yet run end to end; the plan is in the owner's private planning notes (not published).

## Update 2026-10-12
- **Migration 006** (`migrations/pending/006_vera_licence_observations.sql`, additive `licence_observations` table; destructive rollback in `migrations/rollback/`) was applied 2026-10-09 with the owner's approval (commit f1f044a). The file still sits in `pending/`. The loader is behind `licence_observations_enabled` (code default off; switched on in `config/ask-provider-chain.json` by R1 decision 2026-10-10). With 006, migrations 001u, 003, 004, 005 and 006 are applied.
- **Memory rows** use the existing `claims`, `answers` and `evidence` tables under one fixed memory label; `claims.detailed_status` carries the D6 lifecycle: `memory:g1:pending` (written by `/ask`, not recalled) -> `memory:g1:confirmed` (owner-confirmed, the only recall-eligible state) or `memory:rejected`; `memory:tombstoned` marks forgotten rows and `memory:g1` marks legacy pre-D6 auto-written rows (not recalled, listed for review).
- **Test count.** The 178-passed figure above is stale: `pytest tests` was 965 passed, 3 skipped on 2026-10-12.
- **Hosting.** The Render Postgres was upgraded from the free to a paid plan on 2026-10-12; the earlier 2026-10-17 expiry no longer applies. The database remains the shared production instance.

## Migration 007 (domain registry, DRAFT) role order
`pending/007-DRAFT-domain-registry.sql.draft` fails closed. Required order: (1) roles via `db/accounts/create_vera_accounts.sql`, (2) migrations 001-005, (3) 007. If `vera_vjay_wo` or `vera_vjay_ro` is missing, 007 raises in its first statement, before any DDL (run with `psql -1`), instead of skipping the grants and leaving later writes to fail with "permission denied for table research_domains". `vera_vjay_rw` stays optional (NOTICE, grant skipped). Grants: RO = SELECT; WO = INSERT plus SELECT (domain_id). Still a draft: not executed, pending owner Approval 1.

## Update 2026-10-19 (PLAN-ENV, ENV-C): env files, disabled paths
- Env files now come from `R/config/env` via `vera.env_files` (`.env-intern` and `.env-vera`, read only to fingerprint
  production endpoints, password values dropped). The `WEEK1` search is gone. `.env-local-pg` and `.test-db-marker` are
  named there; the marker is never read.
- D-ENV-5 (owner decision: DEFER): `--target production` is DISABLED (`PRODUCTION_PATH_ENABLED = False`) and refuses with
  `D-ENV-5-pending`. No flag or env var enables it. When enabled by a reviewed code change it also refuses under
  `CLAUDECODE`/`AI_AGENT` and any `PG*`/`KRB5*`/`GSS*` name; the TTY and typed password remain the real human control.
- A loopback `vera_local_*` target is a routing hint only. Every connecting `--target local` op refuses with
  `DISPOSABLE-VERIFY-DISABLED-pending-isolation-review` (no disposable verification exists), and every connecting op
  refuses when no fingerprint source exists (`no_sources`). `plan` and `rollback` never connect and still work.
- After fingerprinting, inherited DB URL, PG*, KRB5*, GSS*, admin, TLS and proxy names are removed from the process env.
- Migrations 006 and 007 are not run by this tool.
