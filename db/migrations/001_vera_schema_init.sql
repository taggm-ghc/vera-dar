-- 001_vera_schema_init.sql  --  VERA M1: persistence schema (core M1/M2 tables)
--
-- Target : Postgres 18, database `vera_vjay` (shared with AI-Internship, whose
--          objects live in schema `internship`; this file never touches it).
--          NOTE: `vera_vjay` is BOTH the database name and the schema name here.
-- Run as : admin role (DDL). Applied inside ONE transaction by
--          db/apply_migrations.py (or `psql -1 -f` for manual use).
--
-- Fixes vs the original VERA-CAPSTONE-PLAN.md schema (Opus findings):
--   1. ENUM(...) is MySQL      -> TEXT + CHECK.
--   2. FKs lacked schema       -> every REFERENCES is vera_vjay.<table>.
--   3. content_hash UNIQUE     -> canonical_sources (identity) + sources
--                                 (immutable versions: hash, version, algorithm).
--   4. no Gate A decision      -> candidates.gate_a_decision (fetch/defer/reject)
--                                 kept separate from candidates.fetch_status.
--   5. no query tracking       -> search_iterations (+ candidates.search_iteration_id).
--   6. content not stored      -> sources.content, so hashes are re-verifiable.
--
-- Reset guard: the DROP below refuses to run if the schema already holds data,
-- unless the session sets  vera.allow_reset = 'on'.  Without the guard, a
-- re-run of this "init" file would silently destroy audit records.

DO $guard$
DECLARE n bigint;
BEGIN
  IF to_regclass('vera_vjay.questions') IS NOT NULL THEN
    EXECUTE 'SELECT count(*) FROM vera_vjay.questions' INTO n;
    IF n > 0 AND coalesce(current_setting('vera.allow_reset', true), 'off') <> 'on' THEN
      RAISE EXCEPTION 'vera_vjay.questions holds % rows; refusing to DROP SCHEMA. SET vera.allow_reset = ''on'' to override.', n;
    END IF;
  END IF;
END
$guard$;

DROP SCHEMA IF EXISTS vera_vjay CASCADE;
CREATE SCHEMA vera_vjay;
COMMENT ON SCHEMA vera_vjay IS 'VERA canonical registry: provenance, gate decisions, audit trail.';

-- ---------------------------------------------------------------- questions
-- Governing research question + Evidence Requirements Map (design: "Evidence
-- Requirements Map"). One row per question; many runs may share it.
CREATE TABLE vera_vjay.questions (
  question_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  research_question         TEXT        NOT NULL CHECK (length(btrim(research_question)) > 0),
  governing_context         TEXT,
  evidence_requirements_map JSONB,   -- subquestions, claim types, temporal scope, counterevidence needs
  created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------- runs
-- One end-to-end execution against a question. Holds the direct-baseline and
-- engineered answers and the comparative eval result (M6). Per-stage answers
-- (draft/revised/baseline with cost) live in M4+ `answers`.
CREATE TABLE vera_vjay.runs (
  run_id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  question_id        BIGINT NOT NULL REFERENCES vera_vjay.questions (question_id) ON DELETE RESTRICT,
  baseline_response  TEXT,
  engineered_response TEXT,
  eval_metrics       JSONB,   -- relevance, grounding, reasoning_integrity, auditability (0-5), cost, latency
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -------------------------------------------------------- search_iterations
-- Every query issued in a run. Gate C "search-again" creates a NEW iteration
-- tied to the unmet Evidence Requirement (design: Gate C), never a generic search.
CREATE TABLE vera_vjay.search_iterations (
  iteration_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id                BIGINT NOT NULL REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT,
  iteration_no          INTEGER NOT NULL DEFAULT 1 CHECK (iteration_no >= 1),  -- 1 = initial search; >1 = Gate C re-search
  search_query          TEXT    NOT NULL CHECK (length(btrim(search_query)) > 0),
  rank                  INTEGER CHECK (rank >= 1),   -- priority of this query among those planned in the iteration
  results_count         INTEGER CHECK (results_count >= 0),
  targets_requirement   TEXT,   -- key into questions.evidence_requirements_map this query tries to satisfy (NULL = initial/broad)
  triggered_by_gate_c   BOOLEAN NOT NULL DEFAULT false,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, iteration_no, search_query)
);

-- --------------------------------------------------------------- candidates
-- Search results + Gate A (acquisition priority) outcome.
--   gate_a_decision = what Gate A DECIDED (fetch/defer/reject; NULL = not yet scored)
--   fetch_status    = what actually HAPPENED to acquisition (pending/fetched/failed)
-- Design: Gate A prioritises recall; 'defer' exists so uncertain candidates are
-- not permanently rejected, and gate_a_uncertainty preserves "how little is known".
CREATE TABLE vera_vjay.candidates (
  candidate_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id               BIGINT NOT NULL REFERENCES vera_vjay.runs (run_id) ON DELETE RESTRICT,
  search_iteration_id  BIGINT REFERENCES vera_vjay.search_iterations (iteration_id) ON DELETE RESTRICT,
  source_url           TEXT NOT NULL CHECK (length(btrim(source_url)) > 0),
  title                TEXT,
  snippet              TEXT,
  search_rank          INTEGER CHECK (search_rank >= 1),   -- position in the engine's result list
  gate_a_score         NUMERIC(3,2) CHECK (gate_a_score BETWEEN 0 AND 1),
  gate_a_uncertainty   NUMERIC(3,2) CHECK (gate_a_uncertainty BETWEEN 0 AND 1),
  gate_a_decision      TEXT CHECK (gate_a_decision IN ('fetch','defer','reject')),
  gate_a_rationale     TEXT,
  fetch_status         TEXT NOT NULL DEFAULT 'pending' CHECK (fetch_status IN ('pending','fetched','failed')),
  fetch_error          TEXT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- a scored candidate must carry a decision, and a rejected one must not be fetched
  CONSTRAINT candidates_scored_has_decision CHECK (gate_a_score IS NULL OR gate_a_decision IS NOT NULL),
  CONSTRAINT candidates_rejected_not_fetched CHECK (NOT (gate_a_decision = 'reject' AND fetch_status = 'fetched')),
  -- a URL is a candidate once per run
  UNIQUE (run_id, source_url)
);

-- -------------------------------------------------------- canonical_sources
-- Stable identity of a source across runs/re-fetches (design: "canonical
-- source identity"). Solves the old `content_hash UNIQUE` re-run failure:
-- re-fetching the same URL adds a VERSION, it does not collide.
CREATE TABLE vera_vjay.canonical_sources (
  canonical_source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  canonical_url       TEXT NOT NULL UNIQUE CHECK (length(btrim(canonical_url)) > 0),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------ sources
-- IMMUTABLE acquired versions. App role has INSERT/SELECT only; a trigger
-- additionally blocks UPDATE for every role (see below).
-- Same content re-fetched under the same canonical source is rejected by the
-- UNIQUE (canonical_source_id, content_hash) -> callers reuse the existing
-- version; changed content gets version+1 under the same canonical source.
CREATE TABLE vera_vjay.sources (
  source_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  canonical_source_id BIGINT NOT NULL REFERENCES vera_vjay.canonical_sources (canonical_source_id) ON DELETE RESTRICT,
  candidate_id        BIGINT NOT NULL REFERENCES vera_vjay.candidates (candidate_id) ON DELETE RESTRICT,
  version             INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  content             TEXT NOT NULL,                     -- extracted text as hashed (assumption: <= a few MB; see README)
  content_hash        TEXT NOT NULL,
  hash_algorithm      TEXT NOT NULL DEFAULT 'sha256' CHECK (hash_algorithm IN ('sha256')),
  provenance          JSONB NOT NULL DEFAULT '{}'::jsonb,  -- url, fetched_at, content_type, http_status, fetcher, ...
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT sources_hash_format CHECK (content_hash ~ '^[0-9a-f]{64}$'),
  UNIQUE (canonical_source_id, version),
  UNIQUE (canonical_source_id, content_hash)
);

CREATE FUNCTION vera_vjay.forbid_update() RETURNS trigger LANGUAGE plpgsql AS $f$
BEGIN
  RAISE EXCEPTION '%.% rows are immutable (UPDATE blocked); insert a new version instead', TG_TABLE_SCHEMA, TG_TABLE_NAME;
END
$f$;
CREATE TRIGGER sources_immutable BEFORE UPDATE ON vera_vjay.sources
  FOR EACH ROW EXECUTE FUNCTION vera_vjay.forbid_update();
CREATE TRIGGER canonical_sources_immutable BEFORE UPDATE ON vera_vjay.canonical_sources
  FOR EACH ROW EXECUTE FUNCTION vera_vjay.forbid_update();

-- ------------------------------------------------------------------- roles
-- Group roles carry the privileges; login accounts named for the agent/process
-- that executes the transactions are added to a group by the admin later
-- (see README). No login role is created here, so no password exists here.
DO $roles$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_rw') THEN CREATE ROLE vera_vjay_rw NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro') THEN CREATE ROLE vera_vjay_ro NOLOGIN; END IF;
END
$roles$;

REVOKE ALL ON SCHEMA vera_vjay FROM PUBLIC;
GRANT USAGE ON SCHEMA vera_vjay TO vera_vjay_rw, vera_vjay_ro;
-- data DML for mutable tables
GRANT SELECT, INSERT, UPDATE, DELETE ON vera_vjay.questions, vera_vjay.runs,
      vera_vjay.search_iterations, vera_vjay.candidates TO vera_vjay_rw;
-- append-only for source identity/versions: NO UPDATE, NO DELETE
GRANT SELECT, INSERT ON vera_vjay.canonical_sources, vera_vjay.sources TO vera_vjay_rw;
GRANT SELECT ON ALL TABLES IN SCHEMA vera_vjay TO vera_vjay_ro;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA vera_vjay TO vera_vjay_rw;
-- future M3+ tables: ro gets SELECT automatically; rw grants are explicit per migration
ALTER DEFAULT PRIVILEGES IN SCHEMA vera_vjay GRANT SELECT ON TABLES TO vera_vjay_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA vera_vjay GRANT USAGE ON SEQUENCES TO vera_vjay_rw;

-- M3+ tables (evidence_spans, appraisals, evidence_relations, reasoning_context,
-- answers, claims) are designed in 003_vera_m3_plus_tables.sql and are NOT
-- created here. (UNLOGGED was rejected: unlogged tables are truncated after a
-- crash, which is unacceptable for an audit store.)
