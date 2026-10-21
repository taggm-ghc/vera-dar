-- VERA-DAR login accounts + write-only group. Idempotent. Run as the admin (vera_vjay_user) via psql.
-- NOT a migration (apply_migrations.py does not pick it up). Contains NO passwords: set them after
-- creation with  \password <role>  in psql so they never reach a file, history or log.
--
-- Same instance, same schema (vera_vjay): "dev" vs "production" is an account distinction only; the
-- dev account touches the same rows as production (see ai_internship_local_db_is_production).
--
--   vera_claude_code_rw  login, in vera_vjay_rw  dev: Claude Code / local runs, tests, eval
--   vera_pipeline_wo     login, in vera_vjay_wo  production: the pipeline's writes; no read of prior data
--   vera_eval_ro         login, in vera_vjay_ro  production: M6 eval, reporting, backups, inspection; SELECT only
--
-- "wo" = INSERT/UPDATE only, plus SELECT on identity/key columns (Postgres needs SELECT for
-- INSERT ... RETURNING and ON CONFLICT DO UPDATE; vera/m2_runner.py uses both). No SELECT on content
-- columns (content, snippet, rationale, claims text...), no DELETE, no DDL.
BEGIN;

DO $r$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_wo') THEN CREATE ROLE vera_vjay_wo NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_claude_code_rw') THEN CREATE ROLE vera_claude_code_rw LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_pipeline_wo') THEN CREATE ROLE vera_pipeline_wo LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro') THEN CREATE ROLE vera_vjay_ro NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_eval_ro') THEN CREATE ROLE vera_eval_ro LOGIN; END IF;
END
$r$;

GRANT vera_vjay_rw TO vera_claude_code_rw;
GRANT vera_vjay_wo TO vera_pipeline_wo;
GRANT vera_vjay_ro TO vera_eval_ro;
GRANT CONNECT ON DATABASE vera_vjay TO vera_vjay_rw;
GRANT CONNECT ON DATABASE vera_vjay TO vera_vjay_ro;
GRANT USAGE ON SCHEMA vera_vjay TO vera_vjay_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA vera_vjay TO vera_vjay_ro;

GRANT CONNECT ON DATABASE vera_vjay TO vera_vjay_wo;
GRANT USAGE ON SCHEMA vera_vjay TO vera_vjay_wo;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA vera_vjay TO vera_vjay_wo;

-- INSERT everywhere the rw group inserts (migrations 001/003/004).
GRANT INSERT ON vera_vjay.questions, vera_vjay.runs, vera_vjay.search_iterations, vera_vjay.candidates,
  vera_vjay.canonical_sources, vera_vjay.sources, vera_vjay.evidence_spans, vera_vjay.appraisals,
  vera_vjay.evidence_relations, vera_vjay.reasoning_context, vera_vjay.answers, vera_vjay.claims,
  vera_vjay.gate_c_decisions TO vera_vjay_wo;

-- UPDATE only where rw has it and the row is mutable. sources/canonical_sources stay immutable.
GRANT UPDATE ON vera_vjay.runs, vera_vjay.search_iterations, vera_vjay.candidates, vera_vjay.questions,
  vera_vjay.appraisals, vera_vjay.evidence_relations, vera_vjay.reasoning_context, vera_vjay.answers,
  vera_vjay.claims TO vera_vjay_wo;

-- Key-column SELECT only (RETURNING / ON CONFLICT / run scoping). Add columns here if a live test shows
-- a "permission denied for column" error; never grant table-wide SELECT to this group.
GRANT SELECT (question_id) ON vera_vjay.questions TO vera_vjay_wo;
GRANT SELECT (run_id, question_id) ON vera_vjay.runs TO vera_vjay_wo;
GRANT SELECT (iteration_id, run_id) ON vera_vjay.search_iterations TO vera_vjay_wo;
GRANT SELECT (candidate_id, run_id) ON vera_vjay.candidates TO vera_vjay_wo;
GRANT SELECT (canonical_source_id) ON vera_vjay.canonical_sources TO vera_vjay_wo;
GRANT SELECT (source_id, candidate_id) ON vera_vjay.sources TO vera_vjay_wo;
GRANT SELECT (span_id) ON vera_vjay.evidence_spans TO vera_vjay_wo;
GRANT SELECT (appraisal_id) ON vera_vjay.appraisals TO vera_vjay_wo;
GRANT SELECT (relation_id) ON vera_vjay.evidence_relations TO vera_vjay_wo;
GRANT SELECT (context_id) ON vera_vjay.reasoning_context TO vera_vjay_wo;
GRANT SELECT (answer_id) ON vera_vjay.answers TO vera_vjay_wo;
GRANT SELECT (claim_id) ON vera_vjay.claims TO vera_vjay_wo;
GRANT SELECT (decision_id) ON vera_vjay.gate_c_decisions TO vera_vjay_wo;

COMMIT;

-- Next, in psql (prompts for the password; nothing is echoed):
--   \password vera_claude_code_rw
--   \password vera_pipeline_wo
--   \password vera_eval_ro
--
-- Verify (read-only):
--   SELECT rolname, rolcanlogin, rolcreaterole, rolcreatedb, rolsuper FROM pg_roles WHERE rolname LIKE 'vera%';
--   SELECT r.rolname AS member, g.rolname AS grp FROM pg_auth_members m
--     JOIN pg_roles r ON r.oid = m.member JOIN pg_roles g ON g.oid = m.roleid WHERE r.rolname LIKE 'vera%';
--   SELECT has_table_privilege('vera_pipeline_wo','vera_vjay.sources','SELECT') AS sel_expect_f,
--          has_table_privilege('vera_pipeline_wo','vera_vjay.sources','INSERT') AS ins_expect_t,
--          has_table_privilege('vera_pipeline_wo','vera_vjay.sources','UPDATE') AS upd_expect_f,
--          has_table_privilege('vera_pipeline_wo','vera_vjay.runs','DELETE')    AS del_expect_f,
--          has_table_privilege('vera_claude_code_rw','vera_vjay.runs','UPDATE') AS rw_upd_expect_t,
--          has_table_privilege('vera_claude_code_rw','vera_vjay.sources','DELETE') AS rw_del_expect_f,
--          has_table_privilege('vera_eval_ro','vera_vjay.sources','SELECT') AS ro_sel_expect_t,
--          has_table_privilege('vera_eval_ro','vera_vjay.runs','INSERT')    AS ro_ins_expect_f,
--          has_table_privilege('vera_eval_ro','vera_vjay.runs','UPDATE')    AS ro_upd_expect_f;
