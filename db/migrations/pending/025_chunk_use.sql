-- 025_chunk_use.sql
-- Filename deliberately contains no "_vera_" token, so apply_migrations.py's `{prefix}_vera_*.sql` glob never picks it up (as 007/011/012/017/020/022/024).
-- CHUNK-USAGE-1110 (WP11 of work/CHUNK-VALUE-1110/plan.md, section 6b) DRAFT. NOT EXECUTED. UNTESTED against a real PostgreSQL.
-- APPLYING IT NEEDS OWNER-APPROVED DDL in that turn (AGENTS.md section 7), admin-run only, rehearsal first
-- (work/CHUNK-USAGE-MIG025-1110/scripts/rehearse_025.sh in vera_dar_rehearsal), never via db/apply_migrations.py.
-- Purpose (rev. 2026-11-10, owner: reduce an association's weight, then soft-remove it, never purge the chunk): a per-chunk, per-association match-to-use counter ("a chunk that is often matched but rarely used is a signal", owner 2026-11-10).
--   chunk_use: one row per stored retrieval chunk = (evidence_spans partition span_id, chunk_start, chunk_end in document offsets).
--   matched_count  times the chunk cleared the question's relevance rules (a retrieval candidate that survived admission scoring)
--   selected_count times it was admitted as evidence for the writer
--   used_count     times it was cited in a delivered answer
--   first_seen_day / last_seen_day  UTC dates of the first and last observation.
-- PRIVACY: counts and dates only. No question text, run id, session, user, or time of day is stored (the plan's privacy rule).
-- Written by the API rw login through best-effort batched upserts (vera/chunk_usage.py); no other writer. FK ON DELETE RESTRICT: the
-- known-bad purge deletes these rows first. App roles get no DELETE (nothing in the app removes a counter).
-- REQUIRES: PostgreSQL 16 or newer; vera_vjay.evidence_spans (003) with BIGINT PK span_id; group roles vera_vjay_rw/_ro/_wo and the app logins.
--   INDEPENDENT of 020, 022 and 024.
-- Grants (default deny; rw and ro only): rw SELECT + column INSERT + column UPDATE as listed below (SELECT is needed because the upsert
--   reads the existing counter); ro SELECT (audit); vera_vjay_wo and PUBLIC get nothing.
-- RUN WITH: psql -X -v ON_ERROR_STOP=1 [-v app_rw_login=.. -v app_ro_login=..] -f <this file>   (own BEGIN/COMMIT).
\if :{?app_rw_login}
\else
  \set app_rw_login vera_claude_code_rw
\endif
\if :{?app_ro_login}
\else
  \set app_ro_login vera_eval_ro
\endif
BEGIN;
SET LOCAL lock_timeout = '5s';   -- the FK briefly locks evidence_spans; never queue behind ingest for long
SELECT set_config('vera025.rw', :'app_rw_login', true), set_config('vera025.ro', :'app_ro_login', true);

-- Preflight 1: PG16+; evidence_spans keyed by BIGINT span_id; nothing this file creates exists (strictly additive).
DO $$
BEGIN
  IF pg_catalog.current_setting('server_version_num')::int < 160000 THEN
    RAISE EXCEPTION '025 requires PostgreSQL 16 or newer.';
  END IF;
  IF pg_catalog.to_regclass('vera_vjay.evidence_spans') IS NULL
     OR NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint c JOIN pg_catalog.pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
                     WHERE c.conrelid = 'vera_vjay.evidence_spans'::pg_catalog.regclass AND c.contype IN ('p','u')
                       AND pg_catalog.array_length(c.conkey, 1) = 1 AND a.attname = 'span_id' AND a.atttypid = 'pg_catalog.int8'::pg_catalog.regtype) THEN
    RAISE EXCEPTION '025 requires vera_vjay.evidence_spans keyed by BIGINT span_id (migration 003).';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace ns ON ns.oid = c.relnamespace
              WHERE ns.nspname = 'vera_vjay' AND c.relname LIKE 'chunk\_use%') THEN
    RAISE EXCEPTION '025 already applied or name collision (chunk_use).';
  END IF;
END $$;

-- Preflight 2 (as 020/022/024): roles exist; logins inherit their groups; the account running this file is not an app role nor reachable from one.
DO $$
DECLARE g text; l text; pr text[][] := ARRAY[['vera_vjay_rw', current_setting('vera025.rw')], ['vera_vjay_ro', current_setting('vera025.ro')]];
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'vera_vjay_wo') THEN
    RAISE EXCEPTION '025 requires group role vera_vjay_wo (db/accounts/create_vera_accounts.sql first).';
  END IF;
  FOR i IN 1 .. 2 LOOP
    g := pr[i][1]; l := pr[i][2];
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = g) THEN
      RAISE EXCEPTION '025 requires group role % (db/accounts/create_vera_accounts.sql first).', g;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = l) THEN
      RAISE EXCEPTION '025 requires login role % (override with -v app_*_login=...).', l;
    END IF;
    IF NOT pg_catalog.pg_has_role(l, g, 'USAGE') THEN
      RAISE EXCEPTION '025: login role % does not inherit group role %.', l, g;
    END IF;
  END LOOP;
  IF current_user = ANY (ARRAY['vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera025.rw'), current_setting('vera025.ro')])
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles r ON r.oid = m.roleid JOIN pg_catalog.pg_roles u ON u.oid = m.member
                 WHERE r.rolname IN ('vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera025.rw'), current_setting('vera025.ro'))
                   AND (m.inherit_option OR m.set_option)
                   AND u.rolname IN (current_user::pg_catalog.text, session_user::pg_catalog.text)) THEN
    RAISE EXCEPTION '025 must not run as, or as a member of, an app group or login role: the table owner is unrestricted.';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles a
              WHERE a.rolname = ANY (ARRAY['vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera025.rw'), current_setting('vera025.ro')])
                AND (pg_catalog.pg_has_role(a.rolname, current_user, 'MEMBER') OR pg_catalog.pg_has_role(a.rolname, session_user, 'MEMBER'))) THEN
    RAISE EXCEPTION '025: an app group or login role can reach the account running this file; it would reach the table owner.';
  END IF;
END $$;

-- 1. chunk_use. One row per (chunk, node_id): node_id 0 is the per-chunk TOTAL row (asks without routing and the chunk's overall
--    counts); node_id > 0 is one routing association (the taxonomy/domain node through which the chunk matched). node_id carries NO
--    foreign key: no tax_* table is touched. 10 pk/fk/check constraints: pk, span_fk, range_chk, counts_chk, days_chk, node_chk,
--    init_chk, weight_chk, assoc_chk, msu_chk. Index chunk_use_assoc_idx (partial: decayed or removed associations, the router's read set).
--    initial_weight = the baseline similarity of the association (the taxonomy job's chunk-link score, kind 'cosine' = 1 - cosine
--    distance clamped to [-1, 1] and rounded to 6 places, supplied by the caller; 1.0 when none is known). weight starts equal to it,
--    never exceeds it, and decays multiplicatively; a use restores it toward initial_weight. removal is judged on weight (by default
--    relative to initial_weight, an application setting). weight (<= initial_weight) decays multiplicatively and removed_at is set SOFT when it falls below the removal threshold; both are
--    reversible by a use (restore) or by an owner UPDATE. A chunk row or a chunk is NEVER deleted by this mechanism: the app roles
--    have no DELETE and the association mark never touches evidence_spans, so the chunk stays globally searchable.
-- ---------------------------------------------------------------------------------------------------------------------------------
CREATE TABLE vera_vjay.chunk_use (
    span_id           BIGINT NOT NULL CONSTRAINT chunk_use_span_fk REFERENCES vera_vjay.evidence_spans (span_id) ON DELETE RESTRICT,
    chunk_start       INTEGER NOT NULL,
    chunk_end         INTEGER NOT NULL,
    node_id           BIGINT NOT NULL DEFAULT 0,
    matched_count     BIGINT NOT NULL DEFAULT 0,
    selected_count    BIGINT NOT NULL DEFAULT 0,
    used_count        BIGINT NOT NULL DEFAULT 0,
    matched_since_use BIGINT NOT NULL DEFAULT 0,
    initial_weight    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    weight            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    removed_at        TIMESTAMPTZ,
    first_seen_day    DATE NOT NULL DEFAULT (now() AT TIME ZONE 'utc')::date,
    last_seen_day     DATE NOT NULL DEFAULT (now() AT TIME ZONE 'utc')::date,
    CONSTRAINT chunk_use_pk PRIMARY KEY (span_id, chunk_start, chunk_end, node_id),
    CONSTRAINT chunk_use_range_chk CHECK (chunk_start >= 0 AND chunk_end > chunk_start),
    CONSTRAINT chunk_use_counts_chk CHECK (matched_count >= 0 AND selected_count >= 0 AND used_count >= 0),
    CONSTRAINT chunk_use_days_chk CHECK (last_seen_day >= first_seen_day),
    CONSTRAINT chunk_use_node_chk CHECK (node_id >= 0),
    CONSTRAINT chunk_use_init_chk CHECK (initial_weight > 0.0 AND initial_weight <= 1.0),
    CONSTRAINT chunk_use_weight_chk CHECK (weight >= 0.0 AND weight <= initial_weight),
    CONSTRAINT chunk_use_assoc_chk CHECK (node_id > 0 OR (initial_weight = 1.0 AND weight = 1.0 AND removed_at IS NULL)),
    CONSTRAINT chunk_use_msu_chk CHECK (matched_since_use >= 0)
);
CREATE INDEX chunk_use_assoc_idx ON vera_vjay.chunk_use (node_id, span_id) WHERE node_id > 0 AND (weight < initial_weight OR removed_at IS NOT NULL);

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 2. Grants: default deny, then exact grants to the rw and ro groups only.
-- ---------------------------------------------------------------------------------------------------------------------------------
REVOKE ALL ON vera_vjay.chunk_use FROM PUBLIC, vera_vjay_wo, vera_vjay_rw, vera_vjay_ro;
DO $$ BEGIN
  EXECUTE pg_catalog.format('REVOKE ALL ON vera_vjay.chunk_use FROM %I, %I', current_setting('vera025.rw'), current_setting('vera025.ro'));
END $$;
GRANT SELECT ON vera_vjay.chunk_use TO vera_vjay_rw, vera_vjay_ro;
GRANT INSERT (span_id, chunk_start, chunk_end, node_id, matched_count, selected_count, used_count, matched_since_use, initial_weight, weight, removed_at, first_seen_day, last_seen_day)
  ON vera_vjay.chunk_use TO vera_vjay_rw;
GRANT UPDATE (matched_count, selected_count, used_count, matched_since_use, initial_weight, weight, removed_at, last_seen_day) ON vera_vjay.chunk_use TO vera_vjay_rw;

-- Final privilege check against the actual logins (inside the transaction: a RAISE rolls everything back).
DO $$
DECLARE
  rw text := current_setting('vera025.rw'); ro text := current_setting('vera025.ro');
  t text := 'vera_vjay.chunk_use'; r text; p text; c text;
  ins text[] := ARRAY['span_id','chunk_start','chunk_end','node_id','matched_count','selected_count','used_count','matched_since_use','initial_weight','weight','removed_at','first_seen_day','last_seen_day'];
  upd text[] := ARRAY['matched_count','selected_count','used_count','matched_since_use','initial_weight','weight','removed_at','last_seen_day'];
BEGIN
  FOREACH r IN ARRAY ARRAY[rw, ro] LOOP
    IF NOT pg_catalog.has_table_privilege(r, t, 'SELECT') THEN RAISE EXCEPTION '025 grant check failed: login % needs SELECT on %.', r, t; END IF;
    FOREACH p IN ARRAY ARRAY['INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'] LOOP
      IF pg_catalog.has_table_privilege(r, t, p) THEN RAISE EXCEPTION '025 grant check failed: login % holds table-wide % on %.', r, p, t; END IF;
    END LOOP;
  END LOOP;
  FOR c IN SELECT a.attname FROM pg_catalog.pg_attribute a WHERE a.attrelid = t::pg_catalog.regclass AND a.attnum > 0 AND NOT a.attisdropped LOOP
    IF pg_catalog.has_column_privilege(rw, t, c, 'INSERT') <> (c = ANY (ins)) THEN
      RAISE EXCEPTION '025 grant check failed: column INSERT for login % is wrong on %.%.', rw, t, c;
    END IF;
    IF pg_catalog.has_column_privilege(rw, t, c, 'UPDATE') <> (c = ANY (upd)) THEN
      RAISE EXCEPTION '025 grant check failed: column UPDATE for login % is wrong on %.%.', rw, t, c;
    END IF;
    IF pg_catalog.has_column_privilege(ro, t, c, 'INSERT') OR pg_catalog.has_column_privilege(ro, t, c, 'UPDATE') THEN
      RAISE EXCEPTION '025 grant check failed: ro login holds a column write privilege on %.%.', t, c;
    END IF;
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class k JOIN pg_catalog.pg_namespace ns ON ns.oid = k.relnamespace, pg_catalog.aclexplode(k.relacl) x
              WHERE ns.nspname = 'vera_vjay' AND k.relname = 'chunk_use'
                AND (x.grantee = 0 OR x.grantee = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = 'vera_vjay_wo'))) THEN
    RAISE EXCEPTION '025 grant check failed: PUBLIC or vera_vjay_wo holds a privilege on chunk_use.';
  END IF;
END $$;

COMMIT;
