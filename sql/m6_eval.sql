-- M6 evaluation storage. ADDITIVE and idempotent. Does NOT create vera_vjay.runs:
-- that table belongs to M1. If it does not exist this is a no-op and m6_runner
-- fails verbosely instead of inventing the table.
-- Column types follow M1 (001_vera_schema_init.sql): baseline_response TEXT,
-- eval_metrics JSONB. M6 never writes engineered_response (M5 owns final_response).
-- Not applied to production by M6 (verified 2026-09-30: no vera_vjay objects).
DO $$
BEGIN
  IF to_regclass('vera_vjay.runs') IS NOT NULL THEN
    ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS baseline_response text;
    ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS eval_metrics jsonb;
    ALTER TABLE vera_vjay.runs ADD COLUMN IF NOT EXISTS evaluated_at timestamptz;
  ELSE
    RAISE NOTICE 'vera_vjay.runs missing: M1 schema not applied yet; nothing changed.';
  END IF;
END $$;
