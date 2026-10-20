-- 005_vera_m6_schema_sync.sql -- M6 judge schema synchronization
-- Adds missing columns required by M6 adapter queries + run_id filters
-- Depends on 001 + 002 + 003 + 004. APPLIED to the live schema (columns verified read-only 2026-10-01;
-- see db/README.md "Current state"). The statements are idempotent (IF NOT EXISTS guards).
--
-- Addresses schema blockers:
--   1. runs.question and runs.final_response missing (M6 tries SELECT question, final_response)
--   2. runs.evaluated_at missing (M6 tries UPDATE ... evaluated_at = now())
--   3. appraisals.rubric_version and appraisals.policy_version missing (M6 needs them for audit trail)
--   4. evidence_relations.run_id missing (M6 needs WHERE run_id filter to prevent data leaks); column + FK + index
--      only, NO backfill (see below)
--   5. evidence_relations.rationale missing (M6 tries SELECT rationale)
--
-- Backfill logic:
--   - runs.question: populated from questions table via foreign key
--   - runs.final_response: copy from engineered_response (the synthesized answer)
--   - runs.evaluated_at: leave NULL (M6 will set on first eval)
--   - evidence_relations.run_id: NOT backfilled. The earlier join chain was removed: evidence_spans are
--     source-level (no run_id) and the table was empty at apply time (verified 2026-10-01). New rows get
--     run_id from the M4 runner (ownership to be confirmed by R1).
--   - evidence_relations.rationale: leave NULL (optional audit detail)
--   - appraisals.rubric_version, policy_version: set to defaults (sourced from corpus at eval time)

-- 1. Add question and final_response to runs table
ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS question TEXT;
COMMENT ON COLUMN vera_vjay.runs.question IS 'Question text for this run (copied from questions.research_question)';

ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS final_response TEXT;
COMMENT ON COLUMN vera_vjay.runs.final_response IS 'Final synthesized answer (same as engineered_response for M6 eval)';

ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMPTZ;
COMMENT ON COLUMN vera_vjay.runs.evaluated_at IS 'Timestamp when M6 evaluation completed (set by M6 runner)';

-- 2. Backfill runs.question from questions table
UPDATE vera_vjay.runs r
SET question = q.research_question
FROM vera_vjay.questions q
WHERE r.question_id = q.question_id AND r.question IS NULL;

-- 3. Backfill runs.final_response from engineered_response (the actual answer)
UPDATE vera_vjay.runs
SET final_response = engineered_response
WHERE final_response IS NULL AND engineered_response IS NOT NULL;

-- 4. Add rubric_version and policy_version to appraisals
ALTER TABLE vera_vjay.appraisals ADD COLUMN IF NOT EXISTS rubric_version TEXT DEFAULT '1.0';
COMMENT ON COLUMN vera_vjay.appraisals.rubric_version IS 'Version of evaluation rubric used (from corpus at eval time)';

ALTER TABLE vera_vjay.appraisals ADD COLUMN IF NOT EXISTS policy_version TEXT DEFAULT '1.0';
COMMENT ON COLUMN vera_vjay.appraisals.policy_version IS 'Version of gate B policy used (from corpus at eval time)';

-- 5. Add run_id to evidence_relations for filtering and FK integrity
ALTER TABLE vera_vjay.evidence_relations ADD COLUMN IF NOT EXISTS run_id BIGINT;
COMMENT ON COLUMN vera_vjay.evidence_relations.run_id IS 'Run this relation belongs to (backfilled via span FK chain)';

-- No run_id backfill: evidence_spans are source-level (no run_id) and evidence_relations is empty
-- at apply time (verified 2026-10-01). Existing rows, if any appear later, stay NULL until the M4 runner sets run_id.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'evidence_relations_run_id_fk') THEN
    ALTER TABLE vera_vjay.evidence_relations
      ADD CONSTRAINT evidence_relations_run_id_fk FOREIGN KEY (run_id)
      REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT;
  END IF;
END $$;
CREATE INDEX IF NOT EXISTS evidence_relations_run_id_idx ON vera_vjay.evidence_relations (run_id);

-- 6. Add rationale column to evidence_relations (optional, for audit trail detail)
ALTER TABLE vera_vjay.evidence_relations ADD COLUMN IF NOT EXISTS rationale TEXT;
COMMENT ON COLUMN vera_vjay.evidence_relations.rationale IS 'Optional: scoring rationale or justification for this relation';

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON vera_vjay.runs TO vera_vjay_rw;
GRANT UPDATE ON vera_vjay.appraisals TO vera_vjay_rw;
GRANT UPDATE ON vera_vjay.evidence_relations TO vera_vjay_rw;
