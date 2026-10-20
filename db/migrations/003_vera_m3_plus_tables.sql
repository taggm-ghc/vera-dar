-- 003_vera_m3_plus_tables.sql  --  M3-M7 tables. APPLIED to the live vera_vjay schema
-- (2026-09-30, via db/apply_migrations.py as the admin identity; see db/README.md "Current state").
-- Depends on 001 + 002. Do not re-run against the live schema (the CREATE TABLEs are not idempotent).
--
-- HISTORY: a standalone db/migrations/001_m4_m5_vera_vjay.sql once overlapped this file (TEXT run_id, no FKs).
-- It was deleted; its content was reconciled into this design and 004.

-- Evidence spans: exact locations inside an immutable source version.
CREATE TABLE vera_vjay.evidence_spans (
  span_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_id       BIGINT NOT NULL REFERENCES vera_vjay.sources (source_id) ON DELETE RESTRICT,
  text            TEXT NOT NULL,
  start_index     INTEGER NOT NULL CHECK (start_index >= 0),
  end_index       INTEGER NOT NULL,
  relevance_score NUMERIC(3,2) CHECK (relevance_score BETWEEN 0 AND 1),
  evidence_type   TEXT,  -- e.g. rct, observational, telemetry, survey, vendor_report
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT evidence_spans_range CHECK (end_index > start_index),
  UNIQUE (source_id, start_index, end_index)
);
CREATE INDEX evidence_spans_source_id_idx ON vera_vjay.evidence_spans (source_id);

-- Gate B appraisal (GRADE/CASP-aligned). Scores 0-5; NULL = unevaluated
-- (design: unknown is distinct from zero). Risk dimensions: higher = more
-- concern. applicability_score is the exception: higher = better fit.
CREATE TABLE vera_vjay.appraisals (
  appraisal_id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_id                BIGINT NOT NULL REFERENCES vera_vjay.sources (source_id) ON DELETE RESTRICT,
  bias_score               NUMERIC(2,1) CHECK (bias_score BETWEEN 0 AND 5),               -- 0 none .. 5 severe concern
  inconsistency_score      NUMERIC(2,1) CHECK (inconsistency_score BETWEEN 0 AND 5),
  indirectness_score       NUMERIC(2,1) CHECK (indirectness_score BETWEEN 0 AND 5),
  imprecision_score        NUMERIC(2,1) CHECK (imprecision_score BETWEEN 0 AND 5),
  publication_bias_score   NUMERIC(2,1) CHECK (publication_bias_score BETWEEN 0 AND 5),
  applicability_score      NUMERIC(2,1) CHECK (applicability_score BETWEEN 0 AND 5),      -- 0 poor fit .. 5 direct fit
  overall_quality          TEXT CHECK (overall_quality IN ('high','moderate','low','very_low')),
  gate_b_decision          TEXT CHECK (gate_b_decision IN ('admit','qualify','reject','quarantine','excise-span-and-admit')),
  rationale                TEXT,
  created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT appraisals_decision_has_rationale CHECK (gate_b_decision IS NULL OR rationale IS NOT NULL)
);
CREATE INDEX appraisals_source_id_idx ON vera_vjay.appraisals (source_id);
CREATE INDEX appraisals_gate_b_decision_idx ON vera_vjay.appraisals (gate_b_decision);

CREATE TABLE vera_vjay.evidence_relations (
  relation_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  span1_id      BIGINT NOT NULL REFERENCES vera_vjay.evidence_spans (span_id) ON DELETE RESTRICT,
  span2_id      BIGINT NOT NULL REFERENCES vera_vjay.evidence_spans (span_id) ON DELETE RESTRICT,
  relation_type TEXT NOT NULL CHECK (relation_type IN ('supports','contradicts','qualifies','derives_from','duplicates')),
  confidence    NUMERIC(3,2) CHECK (confidence BETWEEN 0 AND 1),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (span1_id <> span2_id)
);
CREATE INDEX evidence_relations_span1_idx ON vera_vjay.evidence_relations (span1_id);
CREATE INDEX evidence_relations_span2_idx ON vera_vjay.evidence_relations (span2_id);

CREATE TABLE vera_vjay.reasoning_context (
  context_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id              BIGINT NOT NULL REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT,
  constructed_context JSONB NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX reasoning_context_run_id_idx ON vera_vjay.reasoning_context (run_id);

CREATE TABLE vera_vjay.answers (
  answer_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id           BIGINT NOT NULL REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT,
  stage            TEXT NOT NULL CHECK (stage IN ('baseline','draft','revised','final')),
  response_text    TEXT NOT NULL,
  model            TEXT,
  tokens           INTEGER CHECK (tokens >= 0),
  cost             NUMERIC(12,6) CHECK (cost >= 0),
  latency          NUMERIC(10,3) CHECK (latency >= 0),  -- seconds
  parent_answer_id BIGINT REFERENCES vera_vjay.answers (answer_id) ON DELETE RESTRICT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX answers_run_id_idx ON vera_vjay.answers (run_id);
CREATE INDEX answers_parent_idx ON vera_vjay.answers (parent_answer_id);

-- evidence_span_ids is JSONB per the brief (array of span_id). Trade-off: no FK
-- enforcement; the claim verifier must validate ids. A claim_evidence join
-- table would be stricter if M5 wants referential integrity.
CREATE TABLE vera_vjay.claims (
  claim_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  answer_id           BIGINT NOT NULL REFERENCES vera_vjay.answers (answer_id) ON DELETE RESTRICT,
  claim_text          TEXT NOT NULL,
  evidence_span_ids   JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(evidence_span_ids) = 'array'),
  verification_status TEXT NOT NULL DEFAULT 'unverified'
                      CHECK (verification_status IN ('unverified','supported','partially_supported','unsupported','contradicted')),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX claims_answer_id_idx ON vera_vjay.claims (answer_id);
CREATE INDEX claims_verification_status_idx ON vera_vjay.claims (verification_status);

GRANT SELECT, INSERT, UPDATE, DELETE ON vera_vjay.reasoning_context, vera_vjay.answers, vera_vjay.claims TO vera_vjay_rw;
-- evidence/appraisal/relations are append-only for the app
GRANT SELECT, INSERT ON vera_vjay.evidence_spans, vera_vjay.appraisals, vera_vjay.evidence_relations TO vera_vjay_rw;
