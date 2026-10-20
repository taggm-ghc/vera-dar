-- 001u_vera_upgrade_in_place.sql  --  NON-DESTRUCTIVE upgrade of the ORIGINAL-shape
-- vera_vjay schema (INTEGER serial ids, no canonical sources) to the M1 design.
-- Use INSTEAD OF 001 + 002 when vera_vjay already holds data. Apply 003 afterwards.
-- Single transaction; any failure rolls everything back. Data is preserved.
--
-- Differences from a fresh 001 install (intentional, to avoid breaking running code):
--   * ids stay SERIAL-style (nextval default) but are widened to BIGINT, instead of IDENTITY;
--   * legacy candidates keep gate_a_decision = NULL ("not recorded"): their 0.62 scores
--     were produced before decisions existed and inventing 'fetch'/'defer' would falsify
--     the audit trail. The decision-required rule is therefore NOT VALID (enforced for
--     every new/updated row, not checked against legacy rows).

DO $pre$
BEGIN
  IF to_regclass('vera_vjay.sources') IS NULL OR to_regclass('vera_vjay.candidates') IS NULL THEN
    RAISE EXCEPTION 'original-shape tables not found; use 001 for a fresh install';
  END IF;
  IF to_regclass('vera_vjay.canonical_sources') IS NOT NULL THEN
    RAISE EXCEPTION 'canonical_sources exists: already upgraded';
  END IF;
  -- sources cannot be backfilled (no stored content, no canonical identity)
  IF (SELECT count(*) FROM vera_vjay.sources) > 0 THEN
    RAISE EXCEPTION 'sources has rows; upgrade needs a manual backfill plan (content is not stored in the legacy shape)';
  END IF;
END
$pre$;

-- 1. widen ids to BIGINT (tables and owned sequences)
DO $w$
DECLARE r record;
BEGIN
  FOR r IN SELECT c.table_name, c.column_name FROM information_schema.columns c
           WHERE c.table_schema='vera_vjay' AND c.data_type='integer'
             AND c.column_name IN ('question_id','run_id','candidate_id','source_id','iteration_id') LOOP
    EXECUTE format('ALTER TABLE vera_vjay.%I ALTER COLUMN %I TYPE BIGINT', r.table_name, r.column_name);
  END LOOP;
  FOR r IN SELECT s.relname FROM pg_class s WHERE s.relnamespace='vera_vjay'::regnamespace AND s.relkind='S' LOOP
    EXECUTE format('ALTER SEQUENCE vera_vjay.%I AS BIGINT', r.relname);
  END LOOP;
END
$w$;

-- 2. questions
ALTER TABLE vera_vjay.questions
  ADD CONSTRAINT questions_research_question_nonblank CHECK (length(btrim(research_question)) > 0);

-- 3. runs: nothing structural (FK to questions already schema-qualified, NO ACTION = restrict)

-- 4. search_iterations
ALTER TABLE vera_vjay.search_iterations
  ADD COLUMN iteration_no INTEGER NOT NULL DEFAULT 1 CHECK (iteration_no >= 1),
  ADD COLUMN targets_requirement TEXT,
  ADD COLUMN triggered_by_gate_c BOOLEAN NOT NULL DEFAULT false,
  ADD CONSTRAINT search_iterations_rank_check CHECK (rank >= 1),
  ADD CONSTRAINT search_iterations_results_count_check CHECK (results_count >= 0),
  ADD CONSTRAINT search_iterations_query_nonblank CHECK (length(btrim(search_query)) > 0),
  ADD CONSTRAINT search_iterations_run_iter_query_key UNIQUE (run_id, iteration_no, search_query);

-- 5. candidates
UPDATE vera_vjay.candidates SET fetch_status = 'pending' WHERE fetch_status IS NULL;
ALTER TABLE vera_vjay.candidates
  ADD COLUMN search_iteration_id BIGINT REFERENCES vera_vjay.search_iterations (iteration_id) ON DELETE RESTRICT,
  ADD COLUMN search_rank INTEGER CHECK (search_rank >= 1),
  ADD COLUMN gate_a_uncertainty NUMERIC(3,2) CHECK (gate_a_uncertainty BETWEEN 0 AND 1),
  ADD COLUMN gate_a_decision TEXT CHECK (gate_a_decision IN ('fetch','defer','reject')),
  ADD COLUMN fetch_error TEXT,
  ALTER COLUMN fetch_status SET DEFAULT 'pending',
  ALTER COLUMN fetch_status SET NOT NULL,
  ADD CONSTRAINT candidates_source_url_nonblank CHECK (length(btrim(source_url)) > 0),
  ADD CONSTRAINT candidates_rejected_not_fetched CHECK (NOT (gate_a_decision = 'reject' AND fetch_status = 'fetched')),
  ADD CONSTRAINT candidates_run_source_url_key UNIQUE (run_id, source_url);
ALTER TABLE vera_vjay.candidates
  ADD CONSTRAINT candidates_scored_has_decision CHECK (gate_a_score IS NULL OR gate_a_decision IS NOT NULL) NOT VALID;

-- 6. canonical_sources + sources
CREATE TABLE vera_vjay.canonical_sources (
  canonical_source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  canonical_url       TEXT NOT NULL UNIQUE CHECK (length(btrim(canonical_url)) > 0),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE vera_vjay.sources DROP CONSTRAINT sources_content_hash_key;   -- the re-run-breaking UNIQUE(content_hash)
ALTER TABLE vera_vjay.sources ALTER COLUMN version DROP DEFAULT;
ALTER TABLE vera_vjay.sources ALTER COLUMN version TYPE INTEGER USING 1;   -- table verified empty above
ALTER TABLE vera_vjay.sources
  ALTER COLUMN version SET DEFAULT 1,
  ALTER COLUMN version SET NOT NULL,
  ALTER COLUMN provenance SET DEFAULT '{}'::jsonb,
  ALTER COLUMN provenance SET NOT NULL,
  ALTER COLUMN created_at SET NOT NULL,
  ADD COLUMN canonical_source_id BIGINT NOT NULL REFERENCES vera_vjay.canonical_sources (canonical_source_id) ON DELETE RESTRICT,
  ADD COLUMN content TEXT NOT NULL,
  ADD COLUMN hash_algorithm TEXT NOT NULL DEFAULT 'sha256' CHECK (hash_algorithm IN ('sha256')),
  ADD CONSTRAINT sources_version_check CHECK (version >= 1),
  ADD CONSTRAINT sources_hash_format CHECK (content_hash ~ '^[0-9a-f]{64}$'),
  ADD CONSTRAINT sources_canonical_version_key UNIQUE (canonical_source_id, version),
  ADD CONSTRAINT sources_canonical_hash_key UNIQUE (canonical_source_id, content_hash);
ALTER TABLE vera_vjay.sources DROP CONSTRAINT IF EXISTS sources_candidate_id_fkey;
ALTER TABLE vera_vjay.sources ADD CONSTRAINT sources_candidate_id_fkey
  FOREIGN KEY (candidate_id) REFERENCES vera_vjay.candidates (candidate_id) ON DELETE RESTRICT;

-- 7. immutability triggers
CREATE FUNCTION vera_vjay.forbid_update() RETURNS trigger LANGUAGE plpgsql AS $f$
BEGIN
  RAISE EXCEPTION '%.% rows are immutable (UPDATE blocked); insert a new version instead', TG_TABLE_SCHEMA, TG_TABLE_NAME;
END
$f$;
CREATE TRIGGER sources_immutable BEFORE UPDATE ON vera_vjay.sources
  FOR EACH ROW EXECUTE FUNCTION vera_vjay.forbid_update();
CREATE TRIGGER canonical_sources_immutable BEFORE UPDATE ON vera_vjay.canonical_sources
  FOR EACH ROW EXECUTE FUNCTION vera_vjay.forbid_update();

-- 8. indexes (existing idx_* kept: runs.question_id, candidates.run_id, candidates.gate_a_score,
--    sources.candidate_id, sources.content_hash, search_iterations.run_id). Adding only what is missing.
CREATE INDEX candidates_search_iteration_idx ON vera_vjay.candidates (search_iteration_id);
CREATE INDEX candidates_run_gate_a_score_idx ON vera_vjay.candidates (run_id, gate_a_score DESC NULLS LAST);
CREATE INDEX candidates_fetch_pending_idx ON vera_vjay.candidates (run_id, gate_a_score DESC NULLS LAST)
  WHERE fetch_status = 'pending' AND gate_a_decision = 'fetch';
CREATE INDEX candidates_gate_a_decision_idx ON vera_vjay.candidates (run_id, gate_a_decision);

-- 9. roles + grants (same as 001)
DO $roles$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_rw') THEN CREATE ROLE vera_vjay_rw NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro') THEN CREATE ROLE vera_vjay_ro NOLOGIN; END IF;
END
$roles$;
REVOKE ALL ON SCHEMA vera_vjay FROM PUBLIC;
GRANT USAGE ON SCHEMA vera_vjay TO vera_vjay_rw, vera_vjay_ro;
GRANT SELECT, INSERT, UPDATE, DELETE ON vera_vjay.questions, vera_vjay.runs,
      vera_vjay.search_iterations, vera_vjay.candidates TO vera_vjay_rw;
GRANT SELECT, INSERT ON vera_vjay.canonical_sources, vera_vjay.sources TO vera_vjay_rw;
GRANT SELECT ON ALL TABLES IN SCHEMA vera_vjay TO vera_vjay_ro;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA vera_vjay TO vera_vjay_rw;
ALTER DEFAULT PRIVILEGES IN SCHEMA vera_vjay GRANT SELECT ON TABLES TO vera_vjay_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA vera_vjay GRANT USAGE ON SEQUENCES TO vera_vjay_rw;
