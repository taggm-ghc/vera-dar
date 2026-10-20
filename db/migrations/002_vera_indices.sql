-- 002_vera_indices.sql  --  performance indices for the M1/M2 tables.
-- PK and UNIQUE constraints already create their own indexes (not repeated).
-- Postgres does NOT index FK columns automatically, hence most of these.

-- runs: "all runs of a question" (question -> run walk, eval comparison).
CREATE INDEX runs_question_id_idx ON vera_vjay.runs (question_id);

-- search_iterations: reconstruct a run's search history in order for Gate C.
-- (UNIQUE (run_id, iteration_no, search_query) already serves run_id lookups.)

-- candidates: everything for a run (FK + primary access path).
CREATE INDEX candidates_run_id_idx ON vera_vjay.candidates (run_id);

-- candidates: which candidates came from which query (FK; audit "why was this found").
CREATE INDEX candidates_search_iteration_idx ON vera_vjay.candidates (search_iteration_id);

-- candidates: Gate A review within a run, highest score first (fetch queue / defer review).
CREATE INDEX candidates_run_gate_a_score_idx ON vera_vjay.candidates (run_id, gate_a_score DESC NULLS LAST);

-- candidates: the fetcher's work queue (only unresolved rows -> tiny partial index).
CREATE INDEX candidates_fetch_pending_idx ON vera_vjay.candidates (run_id, gate_a_score DESC NULLS LAST)
  WHERE fetch_status = 'pending' AND gate_a_decision = 'fetch';

-- candidates: audit queries by Gate A decision (e.g. all 'reject's to check false negatives).
CREATE INDEX candidates_gate_a_decision_idx ON vera_vjay.candidates (run_id, gate_a_decision);

-- sources: candidate -> its acquired version(s) (FK).
CREATE INDEX sources_candidate_id_idx ON vera_vjay.sources (candidate_id);

-- sources: find any version with a given hash across all canonical sources
-- (cross-source duplicate / copied-content detection; Independence in Gate C).
CREATE INDEX sources_content_hash_idx ON vera_vjay.sources (content_hash);

-- Note: verification_status (claims) and appraisal indices are created in
-- 003 alongside those tables.
