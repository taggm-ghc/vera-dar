-- 004_vera_m4_m5_additions.sql -- M4/M5 additions on top of 001 + 003 (which own runs, evidence_relations,
-- reasoning_context, answers, claims). APPLIED to the live vera_vjay schema (see db/README.md
-- "Current state"); applied after 003 via db/apply_migrations.py. Not idempotent: do not re-run. Replaces the earlier standalone 001_m4_m5_vera_vjay.sql (deleted; it conflicted).

-- Gate C decision per attempt: the audit trail for the bounded search-again loop (M7).
CREATE TABLE vera_vjay.gate_c_decisions (
  decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id      BIGINT NOT NULL REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT,
  attempt     INTEGER NOT NULL CHECK (attempt >= 0),
  decision    TEXT NOT NULL CHECK (decision IN ('adequate','search_again','insufficient')),
  detail      JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX gate_c_decisions_run_id_idx ON vera_vjay.gate_c_decisions (run_id);

-- Claim verification detail (fine-grained status + reasons). 003's verification_status CHECK is coarser
-- (unverified/supported/partially_supported/unsupported/contradicted); the pipeline maps onto it and keeps
-- the precise status (supported/overreach/weak/contested/...) and issue list here.
ALTER TABLE vera_vjay.claims ADD COLUMN detailed_status TEXT;
ALTER TABLE vera_vjay.claims ADD COLUMN issues JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Run-level outcome of M4/M5 (final text goes in runs.engineered_response, which M6 reads).
ALTER TABLE vera_vjay.runs ADD COLUMN gate_c_decision TEXT;
ALTER TABLE vera_vjay.runs ADD COLUMN verification_status TEXT;
ALTER TABLE vera_vjay.runs ADD COLUMN final_answer_id BIGINT REFERENCES vera_vjay.answers (answer_id) ON DELETE RESTRICT;

GRANT SELECT, INSERT ON vera_vjay.gate_c_decisions TO vera_vjay_rw;
