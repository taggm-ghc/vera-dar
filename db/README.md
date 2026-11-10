# VERA-DAR vera_vjay schema (M1-M7)

*Current state is at the top. Dated 2026-09/10 history is kept, not deleted, in "Appendix: history (dated)" at the end of this file.*

**Status as of 2026-11-08 (repo HEAD c8c1daa; DOCS-1108; owner-run apply logs).** This file owns migration status. Applied by the owner to `vera_dar_db` on 2026-11-08: **020** (taxonomy, applied 07:46Z, verified at apply time) and **021** (acquisition queue, applied 16:51Z, verified at apply time; its SQL file is not in this tree). **018** (trace purge and erasure) was applied by the owner on 2026-11-08 (owner-run apply, owner-reported result "018 APPLIED-VERIFIED" at 17:18Z; not independently checked against the database by agents). **022 and 023 are not applied** per project records. A draft file in `migrations/pending/` does not by itself mean unapplied: 015, 018 and 020 drafts remain there after apply. *Superseded (2026-11-08, earlier the same day): "018 is NOT applied"; an owner check at 17:13Z had reported 018 not applied, before the apply.*

**Update 2026-11-09 (DOCS-1109).** Draft files `pending/022-DRAFT-taxonomy-extensible.sql.draft` and `pending/024-DRAFT-chunk-vectors.sql.draft` exist at c3aabb1 (024 creates `tax_chunk_vectors` and the job-run rows for the chunk-taxonomy job and hybrid retrieval; 022 is needed only to write taxonomy links). **Both were applied to `vera_dar_db` by the owner on 2026-11-09** (owner-run scripts after a rehearsal; 022 verified read-only at apply time, 024 reported "applied and verified"; owner-reported, not independently re-checked against the database by agents). Their headers still say NOT EXECUTED because file contents are not changed after application, and the files stay in `pending/` by convention (as for 015, 018, 020). The 2026-11-08 status above is otherwise unchanged. Hybrid retrieval also needs stored vectors from the chunk-taxonomy job.

*Superseded (2026-11-08) by the status above; kept as history:* **Status as of 2026-11-05 (repo HEAD cd81eab; DOCS-1105-REPO).** This file owns migration status. Applied to `vera_dar_db`: 001u, 003, 004, 005, 006, 011, 012 and 015 (015 owner-reported 2026-11-04, verification not recorded). Not applied (DRAFT files in `migrations/pending/`): 007, 009, 013, 014. A 016 licence-allowlist draft exists only as staged work outside this tree (`work/LICENCE-ALLOWLIST-1104/staged/db/migrations/pending/016-DRAFT-licence-allowlist.sql.draft`; not in `repo/db/`, not applied). The 016 drafts are now also tracked in `db/migrations/pending/` (`016-DRAFT-lists-to-db.sql.draft`) and `db/migrations/rollback/` (`016-DRAFT-lists-to-db-rollback.sql.draft`); not applied. Planned order for the not-yet-applied migrations: 013, then 016 (combined lists-to-DB, which supersedes 014 and the separate 016 licence-allowlist draft), then 017 (DRAFT in `pending/` and `rollback/`, not applied). Migrations 008 and 010 have no file in `repo/db/migrations/` (008 is the reserved bootstrap name; a 010 draft exists only under `work/MIG010-DRAFT-1031/sql/`). The listing in `repo/db/migrations/` is checked: top level 001, 001u, 002, 003, 004, 005; `pending/` 006, 007, 009, 011, 012, 013, 014, 015; `rollback/` has a rollback for each of 006, 007, 009, 011-015. The next paragraph is the 2026-10-30 status, kept as history. Migrations 007 (domain registry) and 009 (portal discovery) are not applied; 007 is **pending owner decision**. Sections dated earlier than 2026-10-29 are history. Details from 2026-10-29 (DOCS-1029-W2): Production database `vera_dar_db` (schema `vera_vjay`): migrations 001u, 003, 004, 005 applied earlier; 006, 011 and 012 applied by the owner on 2026-10-29 and verified (see "Migration apply history per database"). Migrations 007 (domain registry) and 009 (portal discovery) are DRAFT and NOT applied; the decision on 007 (apply, revise or keep as draft) is pending the owner. Sections below dated earlier are history and are kept as written unless marked. Files keep their `pending/` locations and `.sql.draft` names; they are not renamed after application. *Annotation 2026-11-08 (DOCS-1108): later commits (for example 6ef1287, which states migration 018 was verified, H4) may have changed this status; this table was not re-verified and is owned by the owner-run migration record.*

## Migration apply history per database (current state; as of 2026-11-04: 015 added)
VERA-DAR uses the database `vera_dar_db`; `vera_vjay` is the schema inside it. The earlier VERA-era database, in which the schema was also named `vera_vjay`, is the historical legacy VERA database and is not VERA-DAR; its column is kept as history.
The files stay where they are (`pending/`, `.sql.draft` names unchanged); this table records where each was applied. Numbers and file contents are not changed after application.

| Migration | File (sha256 prefix) | Historical legacy VERA database (not VERA-DAR) | `vera_dar_db` |
| --- | --- | --- | --- |
| 006 licence observations | `pending/006_vera_licence_observations.sql` (`c7c55a43`) | applied 2026-10-09 (owner-approved, commit f1f044a) | applied 2026-10-29 by the owner, verified (OD-1028(59)) |
| 011 topic registry | `pending/011-DRAFT-topic-registry.sql.draft` (`311b1cdc`) | not applied | applied 2026-10-29 by the owner, verified (OD-1028(60)) |
| 012 classifier categories | `pending/012-DRAFT-classifier-categories.sql.draft` (`f8267e6c`) | not applied | applied 2026-10-29 by the owner, verified (OD-1028(60)) |
| 015 request trace + corpus digest state | `pending/015-DRAFT-request-trace.sql.draft` (`ac6e2559`; rollback `rollback/015-DRAFT-request-trace-rollback.sql.draft`, kept) | not applied | applied 2026-11-04 by the owner via `work/PERF-MISS-1104/trace/apply_migration.sh --apply` (owner reported "completed --apply"; the verify output was not seen by agents: **verification not recorded**). `db_sink.enabled` set true by INTEGRATE-1104; `digest_store.enabled` set true by DEFERRED-DB-1104 (2026-11-04). Both flags are in `config/request_trace.json`; offline-tested only (fake engines), not observed against the live database. Owner scripts: `work/PERF-MISS-1104/trace/` (`apply_migration.sh`, `rollback_migration.sh`, `show_trace.sh`, `pull_trace.sh`, `purge_trace.sh`) |
| 017 topic tags (document/part topic tagging; `toptag_*` tables) | `pending/017-DRAFT-topic-tags.sql.draft` (`d9eb06c4`; rollbacks `rollback/017-DRAFT-topic-tags-rollback.sql.draft` and `rollback/017-DRAFT-topic-tags-purge.sql.draft`) | 017 DRAFT, not applied | not applied; tagging, table mode and ingest enqueue ship off; owner scripts in `work/TOPIC-TAGGING-1104/scripts/` |
| 018 trace purge + erasure (`vera_vjay.purge_expired_trace()`, 30-day retention) | `pending/018-DRAFT-trace-purge-erasure.sql.draft` (rollback `rollback/018-DRAFT-trace-purge-erasure-rollback.sql.draft`) | applied 2026-11-08 (owner-run apply; owner-reported "018 APPLIED-VERIFIED" 17:18Z; not independently checked against the database) | applied by the owner on 2026-11-08. Wiring `vera/retention_purge.py` `retention_purge.enabled` is true since 6ef1287 (2026-11-08, OD-1105-PURGE-ENABLE), so the automatic purge is enabled. *Superseded 2026-11-08: earlier the same day this row read "not applied" and the purge failed closed; shipped false before 6ef1287.* `config/request_trace.json` |

Not applied to `vera_dar_db` (state recorded 2026-10-29, not re-checked against the database since; 013 and 014 added 2026-11-05 from the file headers, which say DRAFT / NOT EXECUTED):

| Migration | File | Status |
| --- | --- | --- |
| 007 domain registry | `pending/007-DRAFT-domain-registry.sql.draft` | DRAFT, not applied; **pending owner decision** (apply / revise / keep draft). Not done |
| 009 portal discovery | `pending/009-DRAFT-portal-discovery.sql.draft` | DRAFT, not applied |
| 013 live ask stats (anonymous aggregate rows; `config/live_stats.json` `enabled` ships false) | `pending/013-DRAFT-vera-live-ask-stats.sql.draft` | DRAFT, not applied, untested against PostgreSQL. 013 DRAFT, reviewed 2026-11-05 (LIVESTATS-013-REVIEW-1104), not applied. Owner script `work/NONASK-1103/wpL/owner/apply_live_stats.sh` (dry run by default; see `work/NONASK-1103/wpL/owner/INVOCATION.md`) |
| 014 source registries (`entity_registry`, `reporting_portal`; owner-curated, SELECT-only for app roles) | `pending/014-DRAFT-source-registries.sql.draft` | DRAFT, not applied, untested against PostgreSQL. Superseded (2026-11-05, coordinator plan MIG-NUMBERING-1105) by the planned combined lists-to-DB migration 016; do not apply 014. The seed/fallback files `config/entity_registry.json` and `config/reporting_portals.json` are what the code reads |

This file owns migration status; other documents link here. Rollbacks in `rollback/` were not used. The runtime switches for 011 and 012 (`read_from_db`, `registration_enabled`, the topic judge) ship off.

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
| `test_m1_roundtrip.py` | Round-trip test; always cleans up its rows. Observation as of 2026-10-29: not present in `repo/db/` (the "Round-trip test" section below is history) |

## Migration order
- Fresh database: 001, 002, 003, 004, 005, in that order. This database was upgraded instead: 001u, then 003, 004, 005 (all applied as of 2026-10-01). As of 2026-10-29 also applied: 006, 011, 012; as of 2026-11-04 also 015 (owner-reported; see the apply-history table); not applied: 007, 009, 013 and 014 (DRAFT).
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
- Bootstrap: `migrations/008_vera_schema_migrations.sql` is the only file accepted while the tracking table is absent. Observation as of 2026-10-29: no file of that name exists in `repo/db/migrations/` (listing: 001, 001u, 002-005, `pending/`, `rollback/`), and `pending/` held 006, 007, 009, 011, 012 only (as of 2026-11-05 it also holds 013, 014 and 015); the 008 bootstrap and the `schema_migrations` table are therefore not present in this tree (unverified whether the table exists in the database). The bootstrap and baseline text in this section describes the tool's design and is historical until such a file exists.
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

## Migration 015: request trace and corpus digest state (applied 2026-11-04, owner-reported)
Three tables in `vera_vjay`, created by `pending/015-DRAFT-request-trace.sql.draft` (read from the file 2026-11-05):
- `request_trace_requests`: one row per request (`request_id`, optional `client_request_id`, `endpoint`, `started_at`, the visitor `question` text (max 4000 characters, owner-only: the application accounts have no SELECT on it), `outcome_branch`, `message_code`, event/drop counts, `elapsed_ms`).
- `request_trace_events`: ordered events per request (`request_id`, `seq`, `stage`, `code`, `counts`, `detail` JSONB). `detail` keys come only from the allowlist in `config/request_trace.json` `key_allowlist` (Deny All); question text is never an event field.
- `corpus_digest_state`: the latest public corpus description (`cache_key`, `digest` JSONB, `digest_sha256`); the newest `digest_id` wins. Written by `vera/digest_store.py` when `digest_store.enabled`; the digest content is built and checked in code (claim-level checker, approved fact phrases, code-built coverage) before it is stored.
- Append-only trigger: UPDATE always raises; DELETE is allowed only for rows older than 30 days (the owner purge window, `work/PERF-MISS-1104/trace/purge_trace.sh`). The question text is therefore kept 30 days, owner-only. The app writes with the write-only account (INSERT only); the read-only group has no SELECT on the trace tables, so the optional DB view for the owner route (`db_sink.owner_route_db_view`, shipped false) needs an owner-read login: `work/DEFERRED-DB-1104/grant_trace_read.sh` (undo: `work/DEFERRED-DB-1104/revoke_trace_read.sh`; neither has been run, as far as agents know).
- Rollback (owner-only, destructive, never expected to run): `rollback/015-DRAFT-request-trace-rollback.sql.draft` via `work/PERF-MISS-1104/trace/rollback_migration.sh`.
- Level: code and flags exist; offline tests pass with fake engines (`tests/test_integrate_1104_db_flags.py`); live rows in these tables have not been inspected by agents.

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
`run_type`, config fingerprint, per-sub-test rows) are not written (planned, not tracked in this repository).

## Accounts and password hashing (applied 2026-10-01)
Three logins (created by `db/accounts/create_vera_accounts.sql`, applied 2026-10-01; the script is idempotent): `vera_claude_code_rw` (dev, read/write),
`vera_pipeline_wo` (production, write-only: INSERT/UPDATE plus SELECT on identity/key columns only, no DELETE, and no UPDATE on `sources`/`canonical_sources`), `vera_eval_ro` (production, read-only).
The runners are **not yet repointed** to these accounts (they still use the admin identity), and the write-only grants are untested against a live run:
`vera/m2_runner.py` reads columns outside the granted key columns (`candidates.source_url`, `canonical_sources.canonical_url`, `sources.content_hash`), so `run_m2` probably cannot run under `vera_pipeline_wo` (static reading, not run live). Passwords live in
the gitignored `.env` as `VERA_DB_PASSWORD_<role>`; `.env.example` documents the accounts without secrets.

Best practice is SCRAM-SHA-256: run `SET password_encryption = 'scram-sha-256';` in the same session before `ALTER ROLE ... PASSWORD`.
The stored hash type is not catalog-verified, because `pg_authid` is denied to the admin account.

Verification facts (2026-10-01): applied via the admin login; privilege matrix 9/9 as expected; all three logins tested. Only these 3 accounts were created (no more are to be created).

Update 2026-10-07:
- **Write-only grants exercised.** `/ask` corpus admission (`vera/corpus_admission.py`) ran as `vera_pipeline_wo` in a rolled-back transaction. INSERT ... RETURNING works on questions, runs, candidates, canonical_sources and sources, and UPDATE works on candidates. `ON CONFLICT` is refused (it needs SELECT on the conflict columns), so the code uses savepoints and treats a unique violation as "exists".
- **Password rotated.** `vera_pipeline_wo`'s password was set by the admin as a SCRAM verifier.
- **.env repointed.** VERA-DAR's `.env` now points `EXTERNAL_DB_URL`, `INTERNAL_DB_URL` and `VERA_DB_URL_RW` at `vera_claude_code_rw` and `VERA_DB_URL_WO` at `vera_pipeline_wo`. They previously used an account that is refused on `vera_vjay`.
- **Read-only URL added.** `VERA_DB_URL_RO` (`vera_eval_ro`) was added for the corpus description (`GET /corpus-summary`, item #83). The read-only SQL contract test now uses VERA-DAR's own `EXTERNAL_DB_URL` host, not another project's file.
- **Still open.** `run_m2` under the write-only account (see above) is still unrun. Column grants of `vera_pipeline_wo` are UNTESTED against a live `m2_runner.py` run. Dev and prod share one schema (accepted divergence).

## What the pipeline persists, and what it does not (code read 2026-10-02; live rows not queried)
- Persisted: `gate_c_decisions` (decision and full detail per attempt; thresholds are code constants and are not stored with the rows); `claims.detailed_status` and `issues` (M5 fine-grained status); answers by stage with tokens, cost and latency; relations (`span1_id`, `span2_id`, `relation_type`, `confidence` only; `rationale` and `run_id` are not written by that insert); appraisals with rubric/policy version.
- Not persisted: M2 search cost (returned in memory; appended to the file named by `$VERA_COST_LOG` only if set); the appraisal values and years behind an M5 `weak` label; relations dropped by the 60-cap or as malformed.
- Not created by any production code: `runs` rows (only tests insert them), and `sub_question_ids` (read by Gate C, set by nothing). The pipeline therefore could not run end to end as of that date (history; not re-assessed).

## Migration 007 (domain registry, DRAFT) role order
`pending/007-DRAFT-domain-registry.sql.draft` fails closed. Required order: (1) roles via `db/accounts/create_vera_accounts.sql`, (2) migrations 001-005, (3) 007. If `vera_vjay_wo` or `vera_vjay_ro` is missing, 007 raises in its first statement, before any DDL (run with `psql -1`), instead of skipping the grants and leaving later writes to fail with "permission denied for table research_domains". `vera_vjay_rw` stays optional (NOTICE, grant skipped). Grants: RO = SELECT; WO = INSERT plus SELECT (domain_id). Still a draft as of 2026-10-29: not executed; the owner's decision on 007 is pending (the topic registry went to 011 instead). The release gate's live catalog check (a2) reports (table, column) pairs missing for the domain-registry schema (`research_domains.*`, `domain_chunks.asserted_by`) while 007 stays unapplied (see `docs/release-checks.md`).

## Update 2026-10-19 (PLAN-ENV, ENV-C): env files, disabled paths
- Env files now come from the project-root `config/env` directory (written `R/config/env` in the original note) via `vera.env_files` (`.env-intern` and `.env-vera`, read only to fingerprint
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

## Appendix: history (dated; HISTORICAL, kept as written, superseded by the sections above where they differ)

Everything below is a dated record from 2026-09-30 to 2026-10-12 and is not a current-status source.

STATUS (2026-09-30 22:05 PDT): DEPLOYED to production as an in-place upgrade (not a reset).
`001u_vera_upgrade_in_place.sql` + `003_vera_m3_plus_tables.sql` applied; round-trip test 17/17 PASS in 5s.
The 20 pre-existing candidates, 1 question and 1 run are intact (checksum identical before/after;
a local JSON backup (not published)).
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

## Update 2026-10-12
- **Migration 006** (`migrations/pending/006_vera_licence_observations.sql`, additive `licence_observations` table; destructive rollback in `migrations/rollback/`) was applied per database: to the historical legacy VERA database (not VERA-DAR) on 2026-10-09 with the owner's approval (commit f1f044a); and to `vera_dar_db` by the owner on 2026-10-29 (sha256 c7c55a437bbc0c5521f6853be94b86ab81d47e027056ab65c1c3a8fee0bd3cae), verified 5/5 (owner decisions OD-1028(59), recorded outside this repository). The file still sits in `pending/`. The loader is behind `licence_observations_enabled` (code default off; switched on in `config/ask-provider-chain.json` by R1 decision 2026-10-10). With 006, migrations 001u, 003, 004, 005 and 006 are applied.
- **Memory rows** use the existing `claims`, `answers` and `evidence` tables under one fixed memory label; `claims.detailed_status` carries the D6 lifecycle: `memory:g1:pending` (written by `/ask`, not recalled) -> `memory:g1:confirmed` (owner-confirmed, the only recall-eligible state) or `memory:rejected`; `memory:tombstoned` marks forgotten rows and `memory:g1` marks legacy pre-D6 auto-written rows (not recalled, listed for review).
- **Test count.** The 178-passed figure above is stale: `pytest tests` was 965 passed, 3 skipped on 2026-10-12, both historical and not authoritative. No count is stored in the repo; run the offline command in `docs/development.md` and record the revision and counts with the result.
- **Hosting.** The hosting database plan was upgraded (2026-10-12); the earlier 2026-10-17 expiry no longer applies. The database remains the shared production instance.

## Live verification and contract test (2026-10-01)
- B1 (migration 005, commit 28aae11) and B2 (eval freeze, 9fef487) verified live, read-only: the 005 columns exist (`runs.run_label`, `evaluated_at`, `engineered_score`, `baseline_score`, `question`; `appraisals.rubric_version`/`policy_version`; `evidence_relations.run_id`); the frozen question text equals `questions.question_id=2`. `question_id=7` is a test row pending an approved deletion.
- C4: `tests/test_sql_contract.py` checks 87 (table, column) pairs against `vera_vjay`; all exist. M3's store writes `rubric_version`/`policy_version` to columns. `pytest tests`: 178 passed, 2 skipped, as of 2026-10-01, not re-run since.
- Sources are stored here as immutable rows; no raw external sources in the repo (see `docs/vera-design.md` invariants).

## Round-trip test (historical; script not present in repo/db as of 2026-10-29)
Historical invocation, from a different (course-service) tree; the script is not in this repository. Do not run it: the configured database is production.
Writes to PRODUCTION (EXTERNAL_DB_URL and INTERNAL_DB_URL are one instance). It creates a question ->
run -> search iteration -> candidate (Gate A 0.85/fetch) -> fetches https://example.com (offline
fallback text if unreachable) -> stores content + sha256 -> joins back and recomputes the hash ->
checks immutability and CHECK constraints -> deletes every test row (marker `M1-ROUNDTRIP-TEST-<uuid>`)
and asserts zero remain. Prints PASS/FAIL per check and total time. Cleanup runs even on failure.
Identity sequences advance; that is not reversible and harmless.

## Credentials
(Historical, course-service era; the current tool reads env files via `vera.env_files`, see Update 2026-10-19.) `db.get_admin_engine()` reads `week-1v2/.env.db-accounts` (DB_ADMIN_ROLE, DB_ADMIN_PASSWORD, host, DB_NAME).
Accepted divergence: the admin login is the one provided by the hosting platform (CREATEROLE/CREATEDB), which cannot be renamed. DDL uses it.
The schema is `vera_vjay`. Migration 001 creates NOLOGIN group roles `vera_vjay_rw` / `vera_vjay_ro` only (no
passwords). The login accounts the pipeline should use now exist (see "Accounts and password hashing"); the
earlier plan to create a further `vera_api_rw` login is not to be carried out (no more accounts).

