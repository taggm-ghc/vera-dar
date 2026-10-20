-- 006_vera_licence_observations.sql
-- Append-only licence observations (p3m3 item #97, H-23; plan section 9). Additive and idempotent; ADMIN-RUN ONLY,
-- after R1's same-turn approval. No text, title or snippet columns. Systems: vera-admission (+ backfill, rederived).
-- Writers get INSERT only; readers get SELECT only. UPDATE/DELETE/TRUNCATE are revoked AND blocked by trigger.
-- Rollback (destructive, separate approval): the matching *_rollback.sql file.

CREATE SCHEMA IF NOT EXISTS vera_vjay;

CREATE TABLE IF NOT EXISTS vera_vjay.licence_observations (
    obs_id uuid PRIMARY KEY,                                   -- client-generated so outbox replay is idempotent
    observed_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    source_system text NOT NULL CONSTRAINT licence_observations_source_system_chk
        CHECK (source_system IN ('course-gateway','vera-admission','verify-only','backfill','rederived')),
    portal text NOT NULL DEFAULT '',
    portal_basis text,
    host text NOT NULL DEFAULT '' CONSTRAINT licence_observations_host_chk
        CHECK (host !~ '[/@[:space:]]'),
    url_sha256 char(64) NOT NULL,
    url text CONSTRAINT licence_observations_url_chk
        CHECK (url IS NULL OR decision IN ('allow','label')),
    doc_ref text,
    licence_scope text CONSTRAINT licence_observations_scope_chk
        CHECK (licence_scope IS NULL OR licence_scope IN ('metadata','content')),
    governing boolean,
    spdx text,
    family text,
    nc boolean,
    nd boolean,
    confidence numeric(4,3),
    detected_via text,
    decision text NOT NULL CONSTRAINT licence_observations_decision_chk
        CHECK (decision IN ('allow','label','hold','reject')),
    native_decision text,
    reason_code text,
    config_version text,
    run_id text,
    candidate_id text
);

CREATE INDEX IF NOT EXISTS licence_observations_portal_observed_idx ON vera_vjay.licence_observations (portal, observed_at);
CREATE INDEX IF NOT EXISTS licence_observations_url_sha256_idx ON vera_vjay.licence_observations (url_sha256);

CREATE OR REPLACE FUNCTION vera_vjay.licence_observations_forbid_mutation() RETURNS trigger LANGUAGE plpgsql AS $f$
BEGIN
  RAISE EXCEPTION 'licence_observations is append-only (% not allowed)', TG_OP;
END $f$;

DROP TRIGGER IF EXISTS licence_observations_append_only ON vera_vjay.licence_observations;
CREATE TRIGGER licence_observations_append_only BEFORE UPDATE OR DELETE ON vera_vjay.licence_observations
  FOR EACH ROW EXECUTE FUNCTION vera_vjay.licence_observations_forbid_mutation();
DROP TRIGGER IF EXISTS licence_observations_no_truncate ON vera_vjay.licence_observations;
CREATE TRIGGER licence_observations_no_truncate BEFORE TRUNCATE ON vera_vjay.licence_observations
  FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.licence_observations_forbid_mutation();

REVOKE UPDATE, DELETE, TRUNCATE ON vera_vjay.licence_observations FROM PUBLIC;
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_wo') THEN
    REVOKE UPDATE, DELETE, TRUNCATE ON vera_vjay.licence_observations FROM vera_vjay_wo;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_rw') THEN
    REVOKE UPDATE, DELETE, TRUNCATE ON vera_vjay.licence_observations FROM vera_vjay_rw;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro') THEN
    REVOKE UPDATE, DELETE, TRUNCATE ON vera_vjay.licence_observations FROM vera_vjay_ro;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_wo') THEN
    GRANT INSERT ON vera_vjay.licence_observations TO vera_vjay_wo;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_rw') THEN
    GRANT INSERT ON vera_vjay.licence_observations TO vera_vjay_rw;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vera_vjay_ro') THEN
    GRANT SELECT ON vera_vjay.licence_observations TO vera_vjay_ro;
  END IF;
END $$;
