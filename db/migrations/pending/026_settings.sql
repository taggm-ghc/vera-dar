-- 026_settings.sql
-- Filename deliberately contains no "_vera_" token, so apply_migrations.py's `{prefix}_vera_*.sql` glob never picks it up (as 020/022/024/025).
-- SETTINGS-W2-1110 (work/SETTINGS-TABLE-1110/plan.md sections 2, 9.1, 9.8) DRAFT. NOT EXECUTED. UNTESTED against a real PostgreSQL.
-- APPLYING IT NEEDS OWNER-APPROVED DDL in that turn (AGENTS.md section 7), admin-run only, rehearsal first
-- (work/SETTINGS-MIG026-1110/scripts/rehearse_026.sh in vera_dar_rehearsal), never via db/apply_migrations.py.
-- Purpose (owner 2026-11-10: "config settings should be in a variables table", "in the db"): the DB layer of the settings table.
--   set_keys           ALLOWLIST KEY REGISTRY (plan 9.1). One row per tunable the reviewed manifest lists (TUNABLE_DB only): type, bounds,
--                      allowed values, one-way direction against the shipped file value. Insert-only. A key absent here can never get a value.
--   set_settings       versioned values. FK to set_keys; a trigger enforces type, bounds, allowed values and the ONE-WAY CAP against the
--                      file value (lower_only: value <= file value; raise_only: value >= file value); at most ONE active version per key
--                      (partial unique index); a new version retires the previous one in the same transaction.
--   set_settings_hist  append-only history (trigger refuses UPDATE, DELETE and TRUNCATE).
--   set_deny_lists     registry of DENY-ONLY lists (plan 9.8): value pattern, size cap. Structurally deny-only: kind is constrained to 'deny',
--                      the name must be deny_*, and no table here has an allow/permit/effect column.
--   set_deny           entries: owner, reason, created_at, expires_at all NOT NULL (default now()+30 days, hard maximum 90 days), value
--                      checked against the list pattern (an allow-shaped value such as '+x', '!x', 'allow:x', '*' or '*.com' cannot match),
--                      wildcard '*.' entries need wildcard_confirmed, per-list cap on unexpired entries. Renewal = a NEW entry; an entry
--                      can only be shortened (early expiry). No DELETE.
--   set_deny_hist      append-only history of denies.
--   set_deny_active    view (security_invoker): unexpired denies only. Consumers read this, so an expired deny is ignored.
-- The loader computes effective = baseline_allow AND NOT active_deny (deny wins); allow-lists stay in reviewed files.
-- WRITERS: none among the app roles. rw and ro get SELECT ONLY on all seven relations. Every write comes from the admin login through the
--   owner-run settings scripts (W3c), so a compromised public app cannot retune itself. No new DB account (10-01 "no more DB accounts").
-- SEED: generated from work/SETTINGS-TABLE-1110/manifest/manifest.json by work/SETTINGS-MIG026-1110/scripts/gen_seed_026.py (hash-pinned in the
--   block below and in SHA256SUMS). set_settings itself is NOT seeded: the shipped file value stays the baseline until the owner writes a version.
-- REQUIRES: PostgreSQL 16 or newer (security_invoker views need 15; same floor as 024/025); schema vera_vjay; group roles vera_vjay_rw/_ro/_wo and the app logins.
--   INDEPENDENT of 020, 022, 024 and 025. Strictly additive: no existing table is altered.
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
SET LOCAL lock_timeout = '5s';
SET LOCAL standard_conforming_strings = on;
SELECT set_config('vera026.rw', :'app_rw_login', true), set_config('vera026.ro', :'app_ro_login', true);

-- Preflight 1: PG16+; schema exists; nothing this file creates exists.
DO $$
BEGIN
  IF pg_catalog.current_setting('server_version_num')::int < 160000 THEN
    RAISE EXCEPTION '026 requires PostgreSQL 16 or newer.';
  END IF;
  IF pg_catalog.to_regnamespace('vera_vjay') IS NULL THEN
    RAISE EXCEPTION '026 requires schema vera_vjay.';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace ns ON ns.oid = c.relnamespace
              WHERE ns.nspname = 'vera_vjay' AND c.relname LIKE 'set\_%')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace ns ON ns.oid = p.pronamespace
                 WHERE ns.nspname = 'vera_vjay' AND p.proname LIKE 'set\_%') THEN
    RAISE EXCEPTION '026 already applied or name collision (a vera_vjay relation or function named set_*).';
  END IF;
END $$;

-- Preflight 2 (as 020/022/024/025): roles exist; logins inherit their groups; the account running this file is not an app role nor reachable from one.
DO $$
DECLARE g text; l text; pr text[][] := ARRAY[['vera_vjay_rw', current_setting('vera026.rw')], ['vera_vjay_ro', current_setting('vera026.ro')]];
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'vera_vjay_wo') THEN
    RAISE EXCEPTION '026 requires group role vera_vjay_wo (db/accounts/create_vera_accounts.sql first).';
  END IF;
  FOR i IN 1 .. 2 LOOP
    g := pr[i][1]; l := pr[i][2];
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = g) THEN
      RAISE EXCEPTION '026 requires group role % (db/accounts/create_vera_accounts.sql first).', g;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = l) THEN
      RAISE EXCEPTION '026 requires login role % (override with -v app_*_login=...).', l;
    END IF;
    IF NOT pg_catalog.pg_has_role(l, g, 'USAGE') THEN
      RAISE EXCEPTION '026: login role % does not inherit group role %.', l, g;
    END IF;
  END LOOP;
  IF current_user = ANY (ARRAY['vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera026.rw'), current_setting('vera026.ro')])
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles r ON r.oid = m.roleid JOIN pg_catalog.pg_roles u ON u.oid = m.member
                 WHERE r.rolname IN ('vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera026.rw'), current_setting('vera026.ro'))
                   AND (m.inherit_option OR m.set_option)
                   AND u.rolname IN (current_user::pg_catalog.text, session_user::pg_catalog.text)) THEN
    RAISE EXCEPTION '026 must not run as, or as a member of, an app group or login role: the table owner is unrestricted.';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles a
              WHERE a.rolname = ANY (ARRAY['vera_vjay_rw','vera_vjay_ro','vera_vjay_wo', current_setting('vera026.rw'), current_setting('vera026.ro')])
                AND (pg_catalog.pg_has_role(a.rolname, current_user, 'MEMBER') OR pg_catalog.pg_has_role(a.rolname, session_user, 'MEMBER'))) THEN
    RAISE EXCEPTION '026: an app group or login role can reach the account running this file; it would reach the table owner.';
  END IF;
END $$;

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 1. Tables. Every constraint is named (the kit counts them).
-- ---------------------------------------------------------------------------------------------------------------------------------
CREATE TABLE vera_vjay.set_keys (
    key                text NOT NULL,
    module             text NOT NULL,
    value_type         text NOT NULL,
    min_value          numeric,
    max_value          numeric,
    allowed            jsonb,
    one_way            text NOT NULL DEFAULT 'none',
    file_value         jsonb NOT NULL,
    description        text NOT NULL,
    registered_in_seed text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT set_keys_pk PRIMARY KEY (key),
    CONSTRAINT set_keys_key_chk CHECK (key ~ '^[a-z0-9_-]+(\.[a-z0-9_-]+)+$' AND length(key) <= 160),
    CONSTRAINT set_keys_module_chk CHECK (module ~ '^[a-z0-9_-]+$'),
    CONSTRAINT set_keys_type_chk CHECK (value_type IN ('int', 'float', 'bool', 'enum', 'string')),
    CONSTRAINT set_keys_oneway_chk CHECK (one_way IN ('none', 'lower_only', 'raise_only')),
    CONSTRAINT set_keys_bounds_chk CHECK (
        (value_type IN ('int', 'float') AND min_value IS NOT NULL AND max_value IS NOT NULL AND min_value <= max_value AND allowed IS NULL)
        OR (value_type NOT IN ('int', 'float') AND min_value IS NULL AND max_value IS NULL)),
    CONSTRAINT set_keys_allowed_chk CHECK (
        (allowed IS NULL OR (jsonb_typeof(allowed) = 'array' AND jsonb_array_length(allowed) BETWEEN 1 AND 64))
        AND (value_type <> 'enum' OR allowed IS NOT NULL)
        AND (value_type <> 'bool' OR allowed IS NULL OR allowed <@ '[true, false]'::jsonb)),
    CONSTRAINT set_keys_file_chk CHECK (
        CASE value_type
            WHEN 'int' THEN CASE WHEN jsonb_typeof(file_value) = 'number'
                                 THEN (file_value #>> '{}')::numeric BETWEEN min_value AND max_value
                                      AND (file_value #>> '{}')::numeric = trunc((file_value #>> '{}')::numeric)
                                 ELSE jsonb_typeof(file_value) = 'null' END
            WHEN 'float' THEN CASE WHEN jsonb_typeof(file_value) = 'number'
                                   THEN (file_value #>> '{}')::numeric BETWEEN min_value AND max_value
                                   ELSE jsonb_typeof(file_value) = 'null' END
            WHEN 'bool' THEN jsonb_typeof(file_value) IN ('boolean', 'null')
            ELSE jsonb_typeof(file_value) IN ('string', 'null') END),
    CONSTRAINT set_keys_oneway_num_chk CHECK (
        one_way = 'none'
        OR (value_type IN ('int', 'float') AND jsonb_typeof(file_value) = 'number'
            AND CASE one_way WHEN 'lower_only' THEN max_value <= (file_value #>> '{}')::numeric
                             ELSE min_value >= (file_value #>> '{}')::numeric END)),
    CONSTRAINT set_keys_seed_chk CHECK (registered_in_seed ~ '^[0-9a-f]{64}$')
);

CREATE TABLE vera_vjay.set_settings (
    setting_id bigserial NOT NULL,
    key        text NOT NULL,
    version    integer,
    value      jsonb NOT NULL,
    state      text NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL DEFAULT now(),
    created_by text NOT NULL,
    reason     text NOT NULL,
    retired_at timestamptz,
    CONSTRAINT set_settings_pk PRIMARY KEY (setting_id),
    CONSTRAINT set_settings_key_fk FOREIGN KEY (key) REFERENCES vera_vjay.set_keys (key) ON DELETE RESTRICT,
    CONSTRAINT set_settings_key_version_uq UNIQUE (key, version),
    CONSTRAINT set_settings_version_chk CHECK (version >= 1),
    CONSTRAINT set_settings_state_chk CHECK (state IN ('active', 'retired')),
    CONSTRAINT set_settings_retired_chk CHECK ((state = 'retired') = (retired_at IS NOT NULL)),
    CONSTRAINT set_settings_by_chk CHECK (length(btrim(created_by)) BETWEEN 1 AND 80),
    CONSTRAINT set_settings_reason_chk CHECK (length(btrim(reason)) BETWEEN 3 AND 500)
);
CREATE UNIQUE INDEX set_settings_active_uq ON vera_vjay.set_settings (key) WHERE state = 'active';

CREATE TABLE vera_vjay.set_settings_hist (
    hist_id    bigserial NOT NULL,
    setting_id bigint NOT NULL,
    key        text NOT NULL,
    version    integer NOT NULL,
    op         text NOT NULL,
    old_state  text,
    new_state  text NOT NULL,
    value      jsonb NOT NULL,
    label      text NOT NULL,
    actor      text NOT NULL DEFAULT session_user,
    at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT set_settings_hist_pk PRIMARY KEY (hist_id),
    CONSTRAINT set_settings_hist_op_chk CHECK (op IN ('insert', 'retire'))
);

CREATE TABLE vera_vjay.set_deny_lists (
    list_name          text NOT NULL,
    kind               text NOT NULL DEFAULT 'deny',
    value_pattern      text NOT NULL,
    max_entries        integer NOT NULL DEFAULT 200,
    description        text NOT NULL,
    registered_in_seed text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT set_deny_lists_pk PRIMARY KEY (list_name),
    CONSTRAINT set_deny_lists_name_chk CHECK (list_name ~ '^[a-z0-9_-]+(\.[a-z0-9_-]+)*\.deny_[a-z0-9_]+$' AND length(list_name) <= 160),
    CONSTRAINT set_deny_lists_kind_chk CHECK (kind = 'deny'),
    CONSTRAINT set_deny_lists_cap_chk CHECK (max_entries BETWEEN 1 AND 1000),
    CONSTRAINT set_deny_lists_pattern_chk CHECK (length(value_pattern) BETWEEN 4 AND 300 AND value_pattern LIKE '^%$' AND '' !~ value_pattern),
    CONSTRAINT set_deny_lists_seed_chk CHECK (registered_in_seed ~ '^[0-9a-f]{64}$')
);

CREATE TABLE vera_vjay.set_deny (
    deny_id            bigserial NOT NULL,
    list_name          text NOT NULL,
    value              text NOT NULL,
    wildcard_confirmed boolean NOT NULL DEFAULT false,
    owner              text NOT NULL,
    reason             text NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    expires_at         timestamptz NOT NULL DEFAULT (now() + interval '30 days'),
    CONSTRAINT set_deny_pk PRIMARY KEY (deny_id),
    CONSTRAINT set_deny_list_fk FOREIGN KEY (list_name) REFERENCES vera_vjay.set_deny_lists (list_name) ON DELETE RESTRICT,
    CONSTRAINT set_deny_value_chk CHECK (length(value) BETWEEN 1 AND 253 AND value = lower(value) AND value !~ '[[:space:]]'),
    CONSTRAINT set_deny_wildcard_chk CHECK (value NOT LIKE '*%' OR wildcard_confirmed),
    CONSTRAINT set_deny_owner_chk CHECK (length(btrim(owner)) BETWEEN 1 AND 80),
    CONSTRAINT set_deny_reason_chk CHECK (length(btrim(reason)) BETWEEN 10 AND 500),
    CONSTRAINT set_deny_expiry_chk CHECK (expires_at > created_at AND expires_at <= created_at + interval '90 days')
);
CREATE INDEX set_deny_list_exp_idx ON vera_vjay.set_deny (list_name, expires_at);

CREATE TABLE vera_vjay.set_deny_hist (
    hist_id    bigserial NOT NULL,
    deny_id    bigint NOT NULL,
    list_name  text NOT NULL,
    value      text NOT NULL,
    op         text NOT NULL,
    owner      text NOT NULL,
    reason     text NOT NULL,
    expires_at timestamptz NOT NULL,
    actor      text NOT NULL DEFAULT session_user,
    at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT set_deny_hist_pk PRIMARY KEY (hist_id),
    CONSTRAINT set_deny_hist_op_chk CHECK (op IN ('add', 'expire'))
);

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 2. Functions (5) and triggers (16). search_path is pinned; every reference is schema-qualified.
-- ---------------------------------------------------------------------------------------------------------------------------------
CREATE FUNCTION vera_vjay.set_block_fn() RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
  RAISE EXCEPTION 'vera_vjay.%: % is not allowed (insert-only or append-only table)', TG_TABLE_NAME, TG_OP USING ERRCODE = 'insufficient_privilege';
END $$;

CREATE FUNCTION vera_vjay.set_settings_guard_fn() RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE k vera_vjay.set_keys%ROWTYPE; vt text; n numeric; fv numeric; nxt integer;
BEGIN
  IF TG_OP = 'UPDATE' THEN
    IF ROW(NEW.setting_id, NEW.key, NEW.version, NEW.value, NEW.created_at, NEW.created_by, NEW.reason)
       IS DISTINCT FROM ROW(OLD.setting_id, OLD.key, OLD.version, OLD.value, OLD.created_at, OLD.created_by, OLD.reason) THEN
      RAISE EXCEPTION 'set_settings: a stored version is immutable; only its state may move active -> retired';
    END IF;
    IF NOT (OLD.state = 'active' AND NEW.state = 'retired') THEN
      RAISE EXCEPTION 'set_settings: state may only move active -> retired (got % -> %)', OLD.state, NEW.state;
    END IF;
    NEW.retired_at := coalesce(NEW.retired_at, now());
    RETURN NEW;
  END IF;
  SELECT * INTO k FROM vera_vjay.set_keys WHERE key = NEW.key;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'set_settings: key % is not in the allowlist registry set_keys (deny by default)', NEW.key;
  END IF;
  IF NEW.state <> 'active' THEN
    RAISE EXCEPTION 'set_settings: a new version is inserted as active';
  END IF;
  vt := jsonb_typeof(NEW.value);
  IF k.value_type = 'int' THEN
    IF vt <> 'number' OR (NEW.value #>> '{}')::numeric <> trunc((NEW.value #>> '{}')::numeric) THEN
      RAISE EXCEPTION 'set_settings: key % needs an integer (got %)', NEW.key, NEW.value;
    END IF;
  ELSIF k.value_type = 'float' THEN
    IF vt <> 'number' THEN RAISE EXCEPTION 'set_settings: key % needs a number (got %)', NEW.key, NEW.value; END IF;
  ELSIF k.value_type = 'bool' THEN
    IF vt <> 'boolean' THEN RAISE EXCEPTION 'set_settings: key % needs a boolean (got %)', NEW.key, NEW.value; END IF;
  ELSIF vt <> 'string' THEN
    RAISE EXCEPTION 'set_settings: key % needs a string (got %)', NEW.key, NEW.value;
  END IF;
  IF vt = 'number' THEN
    n := (NEW.value #>> '{}')::numeric;
    IF k.one_way <> 'none' THEN
      fv := (k.file_value #>> '{}')::numeric;
      IF k.one_way = 'lower_only' AND n > fv THEN
        RAISE EXCEPTION 'set_settings: key % is lower_only; % exceeds the file value % (one-way cap)', NEW.key, n, fv;
      END IF;
      IF k.one_way = 'raise_only' AND n < fv THEN
        RAISE EXCEPTION 'set_settings: key % is raise_only; % is below the file value % (one-way cap)', NEW.key, n, fv;
      END IF;
    END IF;
    IF n < k.min_value OR n > k.max_value THEN
      RAISE EXCEPTION 'set_settings: key % value % is outside the bounds [%, %]', NEW.key, n, k.min_value, k.max_value;
    END IF;
  END IF;
  IF k.allowed IS NOT NULL AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements(k.allowed) e WHERE e = NEW.value) THEN
    RAISE EXCEPTION 'set_settings: key % value % is not among the allowed values %', NEW.key, NEW.value, k.allowed;
  END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended(NEW.key, 26));
  SELECT coalesce(max(version), 0) + 1 INTO nxt FROM vera_vjay.set_settings WHERE key = NEW.key;
  IF NEW.version IS NULL THEN NEW.version := nxt;
  ELSIF NEW.version <> nxt THEN RAISE EXCEPTION 'set_settings: key % next version is % (got %)', NEW.key, nxt, NEW.version;
  END IF;
  UPDATE vera_vjay.set_settings SET state = 'retired', retired_at = now() WHERE key = NEW.key AND state = 'active';
  RETURN NEW;
END $$;

CREATE FUNCTION vera_vjay.set_settings_hist_fn() RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
  INSERT INTO vera_vjay.set_settings_hist (setting_id, key, version, op, old_state, new_state, value, label)
  VALUES (NEW.setting_id, NEW.key, NEW.version, CASE TG_OP WHEN 'INSERT' THEN 'insert' ELSE 'retire' END,
          CASE WHEN TG_OP = 'UPDATE' THEN OLD.state END, NEW.state, NEW.value, NEW.created_by);
  RETURN NULL;
END $$;

CREATE FUNCTION vera_vjay.set_deny_guard_fn() RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
DECLARE l vera_vjay.set_deny_lists%ROWTYPE; cnt integer;
BEGIN
  IF TG_OP = 'UPDATE' THEN
    IF ROW(NEW.deny_id, NEW.list_name, NEW.value, NEW.wildcard_confirmed, NEW.owner, NEW.reason, NEW.created_at)
       IS DISTINCT FROM ROW(OLD.deny_id, OLD.list_name, OLD.value, OLD.wildcard_confirmed, OLD.owner, OLD.reason, OLD.created_at) THEN
      RAISE EXCEPTION 'set_deny: an entry is immutable; only expires_at may change';
    END IF;
    IF NEW.expires_at >= OLD.expires_at THEN
      RAISE EXCEPTION 'set_deny: expires_at can only be shortened; renew by adding a new entry';
    END IF;
    RETURN NEW;
  END IF;
  SELECT * INTO l FROM vera_vjay.set_deny_lists WHERE list_name = NEW.list_name FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'set_deny: list % is not registered in set_deny_lists', NEW.list_name; END IF;
  IF NEW.value !~ l.value_pattern THEN
    RAISE EXCEPTION 'set_deny: value % does not match the pattern of deny list % (deny-shaped values only)', NEW.value, NEW.list_name;
  END IF;
  IF abs(extract(epoch FROM (now() - NEW.created_at))) > 60 THEN
    RAISE EXCEPTION 'set_deny: created_at must be the current time';
  END IF;
  SELECT count(*) INTO cnt FROM vera_vjay.set_deny WHERE list_name = NEW.list_name AND expires_at > now();
  IF cnt >= l.max_entries THEN
    RAISE EXCEPTION 'set_deny: list % already holds % unexpired entries (cap %)', NEW.list_name, cnt, l.max_entries;
  END IF;
  RETURN NEW;
END $$;

CREATE FUNCTION vera_vjay.set_deny_hist_fn() RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$
BEGIN
  INSERT INTO vera_vjay.set_deny_hist (deny_id, list_name, value, op, owner, reason, expires_at)
  VALUES (NEW.deny_id, NEW.list_name, NEW.value, CASE TG_OP WHEN 'INSERT' THEN 'add' ELSE 'expire' END, NEW.owner, NEW.reason, NEW.expires_at);
  RETURN NULL;
END $$;

CREATE TRIGGER set_keys_block_trg BEFORE UPDATE OR DELETE ON vera_vjay.set_keys FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_keys_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_keys FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_lists_block_trg BEFORE UPDATE OR DELETE ON vera_vjay.set_deny_lists FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_lists_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_deny_lists FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_settings_guard_trg BEFORE INSERT OR UPDATE ON vera_vjay.set_settings FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_settings_guard_fn();
CREATE TRIGGER set_settings_hist_trg AFTER INSERT OR UPDATE ON vera_vjay.set_settings FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_settings_hist_fn();
CREATE TRIGGER set_settings_block_trg BEFORE DELETE ON vera_vjay.set_settings FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_settings_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_settings FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_settings_hist_block_trg BEFORE UPDATE OR DELETE ON vera_vjay.set_settings_hist FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_settings_hist_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_settings_hist FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_guard_trg BEFORE INSERT OR UPDATE ON vera_vjay.set_deny FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_deny_guard_fn();
CREATE TRIGGER set_deny_hist_trg AFTER INSERT OR UPDATE ON vera_vjay.set_deny FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_deny_hist_fn();
CREATE TRIGGER set_deny_block_trg BEFORE DELETE ON vera_vjay.set_deny FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_deny FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_hist_block_trg BEFORE UPDATE OR DELETE ON vera_vjay.set_deny_hist FOR EACH ROW EXECUTE FUNCTION vera_vjay.set_block_fn();
CREATE TRIGGER set_deny_hist_trunc_trg BEFORE TRUNCATE ON vera_vjay.set_deny_hist FOR EACH STATEMENT EXECUTE FUNCTION vera_vjay.set_block_fn();

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 3. View: the only thing a consumer should read for denies. Expired entries are not listed.
-- ---------------------------------------------------------------------------------------------------------------------------------
CREATE VIEW vera_vjay.set_deny_active WITH (security_invoker = true) AS
    SELECT deny_id, list_name, value, wildcard_confirmed, owner, reason, created_at, expires_at
    FROM vera_vjay.set_deny
    WHERE expires_at > now();

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 4. Seed (generated; do not edit by hand).
-- ---------------------------------------------------------------------------------------------------------------------------------
-- BEGIN GENERATED SEED (gen_seed_026.py; manifest sha256 211617211f635e175aba00281febbc2b56514357a845e10389fa43d2152e9008; registry file sha256 fc8b5e12073a827db0205401f0a9fdb1c1abb3d7de4cb3b33176024b594677ff; set_keys 167; set_deny_lists 2; registry keys not seeded 0)
SELECT set_config('vera026.seed_sha', '211617211f635e175aba00281febbc2b56514357a845e10389fa43d2152e9008', true), set_config('vera026.seed_keys', '167', true), set_config('vera026.seed_deny_lists', '2', true);
INSERT INTO vera_vjay.set_keys (key, module, value_type, min_value, max_value, allowed, one_way, file_value, description, registered_in_seed) VALUES
  ('ask-provider-chain.corpus_description.subject_areas.chunk_chars', 'ask-provider-chain', 'int', 100, 8000, NULL, 'none', '1200'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('ask-provider-chain.corpus_description.subject_areas.chunks_per_doc', 'ask-provider-chain', 'int', 1, 200, NULL, 'none', '16'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('ask-provider-chain.corpus_description.subject_areas.min_chunk_votes', 'ask-provider-chain', 'int', 1, 50, NULL, 'none', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.cache_ttl_seconds', 'auto_acquire', 'int', 0, 3600, NULL, 'none', '300'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.batch_size', 'auto_acquire', 'int', 1, 20, NULL, 'none', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_attempts', 'auto_acquire', 'int', 1, 5, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_calls_per_registration', 'auto_acquire', 'int', 1, 100, NULL, 'none', '12'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_completion_tokens', 'auto_acquire', 'int', 16, 2048, NULL, 'none', '256'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_judgments_per_day', 'auto_acquire', 'int', 1, 400, NULL, 'lower_only', '400'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_text_chars', 'auto_acquire', 'int', 40, 1200, NULL, 'none', '600'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.max_tokens_per_registration', 'auto_acquire', 'int', 100, 200000, NULL, 'none', '12000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('auto_acquire.micro_topic.registry_table.judge.timeout_s', 'auto_acquire', 'int', 1, 60, NULL, 'none', '15'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.cloudflare.daily_neuron_budget', 'chunk_taxo', 'int', 0, 8000, NULL, 'lower_only', '8000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.cloudflare.reserve_neurons', 'chunk_taxo', 'int', 5000, 1000000000, NULL, 'raise_only', '5000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.max_abstain_share', 'chunk_taxo', 'float', 0.0, 1.0, NULL, 'none', '0.3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.max_null_floor_spread', 'chunk_taxo', 'float', 0.0, 1.0, NULL, 'none', '0.1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_center_match_rate', 'chunk_taxo', 'float', 0.0, 1.0, NULL, 'none', '0.7'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_coverage_domain', 'chunk_taxo', 'float', 0.0, 1.0, NULL, 'none', '0.8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_coverage_subtopic', 'chunk_taxo', 'float', 0.0, 1.0, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_silhouette_domain', 'chunk_taxo', 'float', -1.0, 1.0, NULL, 'none', '0.0'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_silhouette_subtopic', 'chunk_taxo', 'float', -1.0, 1.0, NULL, 'none', '0.1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_silhouette_topic', 'chunk_taxo', 'float', -1.0, 1.0, NULL, 'none', '0.05'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.display_gate.min_stability_ari', 'chunk_taxo', 'float', -1.0, 1.0, NULL, 'none', '0.6'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.domain_range.max_count', 'chunk_taxo', 'int', 1, 1000, NULL, 'none', '8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.domain_range.max_frac', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.domain_range.min_count', 'chunk_taxo', 'int', 1, 100, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.gain_epsilon', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.005'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.identity_jaccard', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.k_max', 'chunk_taxo', 'int', 1, 200, NULL, 'none', '40'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.k_min', 'chunk_taxo', 'int', 2, 100, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.k_step', 'chunk_taxo', 'int', 1, 20, NULL, 'none', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.kmeans_max_iter', 'chunk_taxo', 'int', 1, 1000, NULL, 'none', '50'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.label_terms', 'chunk_taxo', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.min_cluster_size', 'chunk_taxo', 'int', 1, 1000, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.min_support', 'chunk_taxo', 'int', 1, 1000, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.null_q_dist', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.05'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.null_q_margin', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.25'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.patience', 'chunk_taxo', 'int', 1, 20, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.stability_match_cos', 'chunk_taxo', 'float', -1, 1, NULL, 'none', '0.9'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.term_count', 'chunk_taxo', 'int', 1, 100, NULL, 'none', '8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.term_max_df', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.term_min_len', 'chunk_taxo', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.topic_range.max_count', 'chunk_taxo', 'int', 1, 1000, NULL, 'none', '20'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.topic_range.max_frac', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.topic_range.min_count', 'chunk_taxo', 'int', 1, 100, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.vote_min_lead', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_taxo.params.vote_min_share', 'chunk_taxo', 'float', 0, 1, NULL, 'none', '0.4'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_topic_terms.min_distinct_terms', 'chunk_topic_terms', 'int', 1, 20, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_topic_terms.min_lead_ratio', 'chunk_topic_terms', 'float', 1.0, 100.0, NULL, 'none', '1.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_topic_terms.min_token_len', 'chunk_topic_terms', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('chunk_topic_terms.min_vote_share', 'chunk_topic_terms', 'float', 0.0, 1.0, NULL, 'none', '0.25'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('corpus_frequent_subjects.max_phrases', 'corpus_frequent_subjects', 'int', 1, 30, NULL, 'none', '8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('corpus_frequent_subjects.max_words', 'corpus_frequent_subjects', 'int', 2, 6, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('corpus_frequent_subjects.min_docs', 'corpus_frequent_subjects', 'int', 2, 1000, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('corpus_frequent_subjects.min_words', 'corpus_frequent_subjects', 'int', 2, 6, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('corpus_frequent_subjects.title_limit', 'corpus_frequent_subjects', 'int', 1, 20000, NULL, 'none', '2000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.chunks_per_partition', 'hybrid_retrieval', 'int', 1, 100, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.cooldown_s', 'hybrid_retrieval', 'int', 60, 86400, NULL, 'raise_only', '60'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.index_refresh_s', 'hybrid_retrieval', 'int', 10, 10000000, NULL, 'none', '3600'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.max_index_mb', 'hybrid_retrieval', 'int', 1, 400, NULL, 'none', '96'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.page_ids', 'hybrid_retrieval', 'int', 1, 5000, NULL, 'none', '256'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.query_cache_max', 'hybrid_retrieval', 'int', 1, 4096, NULL, 'none', '32'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.query_cache_ttl_s', 'hybrid_retrieval', 'int', 0, 86400, NULL, 'none', '120'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.cache_max', 'hybrid_retrieval', 'int', 0, 100000, NULL, 'none', '2048'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.cooldown_s', 'hybrid_retrieval', 'int', 60, 86400, NULL, 'raise_only', '60'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.max_chars', 'hybrid_retrieval', 'int', 50, 20000, NULL, 'none', '2000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.max_passages', 'hybrid_retrieval', 'int', 1, 64, NULL, 'none', '48'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.reserve_slots', 'hybrid_retrieval', 'int', 0, 20, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.timeout_s', 'hybrid_retrieval', 'float', 0.2, 30, NULL, 'none', '3.0'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rerank.top_partitions', 'hybrid_retrieval', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.rrf_k', 'hybrid_retrieval', 'int', 1, 1000, NULL, 'none', '60'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.stem_floor_min_hits', 'hybrid_retrieval', 'int', 1, 20, NULL, 'none', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.stem_floor_sim', 'hybrid_retrieval', 'float', -1.0, 1.0, NULL, 'none', 'null'::jsonb, 'optional numeric currently unset (null = feature off)', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.timeout_s', 'hybrid_retrieval', 'float', 0.2, 30, NULL, 'none', '3.0'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('hybrid_retrieval.vector_k', 'hybrid_retrieval', 'int', 1, 1000, NULL, 'none', 'null'::jsonb, 'optional numeric currently unset (null = feature off)', current_setting('vera026.seed_sha')),
  ('progress.first_event_grace_s', 'progress', 'int', 0, 600, NULL, 'none', '10'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.get_timeout_s', 'progress', 'float', 0.1, 60, NULL, 'none', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.max_count', 'progress', 'int', 1, 1000000000, NULL, 'none', '10000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.max_events_per_poll', 'progress', 'int', 1, 10000, NULL, 'none', '128'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.max_events_per_record', 'progress', 'int', 2, 10000, NULL, 'none', '128'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.max_records', 'progress', 'int', 1, 100000, NULL, 'none', '64'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.poll_interval_s', 'progress', 'float', 0.1, 60, NULL, 'none', '1.0'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.poll_margin_s', 'progress', 'int', 0, 600, NULL, 'none', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.stream.activity_tick_s', 'progress', 'float', 0.01, 60, NULL, 'none', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.stream.client_no_data_s', 'progress', 'int', 1, 3600, NULL, 'none', '60'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.stream.connect_timeout_s', 'progress', 'float', 0.1, 300, NULL, 'none', '10'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.stream.heartbeat_s', 'progress', 'float', 0.01, 300, NULL, 'none', '10'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.stream.orphan_s', 'progress', 'int', 1, 86400, NULL, 'none', '120'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('progress.ttl_s', 'progress', 'int', 1, 86400, NULL, 'none', '600'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.breaker_failures', 'rate_limits', 'int', 1, 3, NULL, 'lower_only', '3'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.breaker_open_max_s', 'rate_limits', 'float', 900.0, 86400.0, NULL, 'raise_only', '900.0'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.breaker_open_s', 'rate_limits', 'float', 30.0, 86400.0, NULL, 'raise_only', '30.0'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.decrease_factor', 'rate_limits', 'float', 0.01, 0.5, NULL, 'lower_only', '0.5'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.half_open_max', 'rate_limits', 'int', 1, 1, NULL, 'lower_only', '1'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.increase_frac', 'rate_limits', 'float', 0.0, 0.05, NULL, 'lower_only', '0.05'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.jitter_frac', 'rate_limits', 'float', 0.0, 0.2, NULL, 'lower_only', '0.2'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.max_wait_s', 'rate_limits', 'float', 0.0, 2.0, NULL, 'lower_only', '2.0'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.min_rate_factor', 'rate_limits', 'float', 0.001, 0.05, NULL, 'lower_only', '0.05'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('rate_limits.defaults.poll_s', 'rate_limits', 'float', 0.001, 0.05, NULL, 'lower_only', '0.05'::jsonb, 'internal pacing/breaker parameter, clamped by the per-portal ceilings in the file', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.cache_max', 'reference_portals', 'int', 0, 256, NULL, 'lower_only', '256'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.cache_ttl_s', 'reference_portals', 'int', 0, 86400, NULL, 'lower_only', '86400'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.cooldown_min_s', 'reference_portals', 'int', 5, 50, NULL, 'raise_only', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.max_query_terms', 'reference_portals', 'int', 1, 6, NULL, 'lower_only', '6'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.max_refs_per_seed', 'reference_portals', 'int', 1, 30, NULL, 'lower_only', '30'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.max_response_bytes', 'reference_portals', 'int', 1, 4000000, NULL, 'lower_only', '4000000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.inspire_refs.size', 'reference_portals', 'int', 1, 3, NULL, 'lower_only', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.cache_max', 'reference_portals', 'int', 0, 512, NULL, 'lower_only', '512'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.cache_ttl_s', 'reference_portals', 'int', 0, 86400, NULL, 'lower_only', '86400'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.max_calls_per_question', 'reference_portals', 'int', 1, 1, NULL, 'lower_only', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.max_seeds', 'reference_portals', 'int', 1, 3, NULL, 'lower_only', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.or_seeds', 'reference_portals', 'bool', NULL, NULL, '[true, false]'::jsonb, 'none', 'true'::jsonb, 'behaviour toggle (no access/licence/safety effect found in consumer)', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.openalex_rw.per_page', 'reference_portals', 'int', 0, 50, NULL, 'lower_only', '50'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.adapters.wikipedia_title.max_title_tries', 'reference_portals', 'int', 1, 2, NULL, 'lower_only', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_candidates_per_question', 'reference_portals', 'int', 1, 30, NULL, 'lower_only', '30'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_hops', 'reference_portals', 'int', 1, 2, NULL, 'lower_only', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_hrefs_scanned', 'reference_portals', 'int', 1, 5000, NULL, 'lower_only', '5000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_links_per_page', 'reference_portals', 'int', 1, 50, NULL, 'lower_only', '50'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_page_bytes', 'reference_portals', 'int', 1, 4000000, NULL, 'lower_only', '4000000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_pages_per_host', 'reference_portals', 'int', 1, 1, NULL, 'lower_only', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.max_pages_per_question', 'reference_portals', 'int', 1, 4, NULL, 'lower_only', '4'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.shadow_deadline_s', 'reference_portals', 'int', 1, 30, NULL, 'lower_only', '30'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.tdmrep_max_bytes', 'reference_portals', 'int', 1, 65536, NULL, 'lower_only', '65536'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.caps.timeout_s', 'reference_portals', 'int', 1, 5, NULL, 'lower_only', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.relevance.min_contain_len', 'reference_portals', 'int', 1, 60, NULL, 'none', '6'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.relevance.min_hits', 'reference_portals', 'int', 2, 20, NULL, 'raise_only', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('reference_portals.relevance.stem_len', 'reference_portals', 'int', 1, 50, NULL, 'none', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('scope_router.layer2.malformed_retry', 'scope_router', 'int', 0, 1, NULL, 'none', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.fit_quantile', 'taxo_hier', 'float', 0.5, 0.9999, NULL, 'none', '0.95'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.min_agree', 'taxo_hier', 'float', 0, 1, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.min_token_len', 'taxo_hier', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.min_top_share', 'taxo_hier', 'float', 0, 1, NULL, 'none', '0.25'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.null_chunks', 'taxo_hier', 'int', 50, 20000, NULL, 'none', '2000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.null_replicates', 'taxo_hier', 'int', 1, 20, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.seed', 'taxo_hier', 'int', 0, 2147483648, NULL, 'none', '1108'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxo_hier.params.semantic_weight', 'taxo_hier', 'float', 0.0, 1.0, NULL, 'none', '0.5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.anchor_terms', 'taxonomy_induction', 'int', 5, 120, NULL, 'none', '40'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.chunk_chars', 'taxonomy_induction', 'int', 100, 8000, NULL, 'none', '1200'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.chunks_per_doc', 'taxonomy_induction', 'int', 1, 200, NULL, 'none', '16'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.keyphrases', 'taxonomy_induction', 'int', 3, 40, NULL, 'none', '15'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.max_clusters_new', 'taxonomy_induction', 'int', 0, 200, NULL, 'none', '24'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.max_clusters_per_topic', 'taxonomy_induction', 'int', 0, 50, NULL, 'none', '4'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.max_df_share', 'taxonomy_induction', 'float', 0.01, 1.0, NULL, 'none', '0.25'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.max_iter', 'taxonomy_induction', 'int', 1, 50, NULL, 'none', '8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.max_proposals', 'taxonomy_induction', 'int', 1, 500, NULL, 'none', '60'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.min_chunk_votes', 'taxonomy_induction', 'int', 1, 50, NULL, 'none', '1'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.min_df', 'taxonomy_induction', 'int', 1, 1000, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.min_docs', 'taxonomy_induction', 'int', 1, 1000, NULL, 'none', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.min_similarity', 'taxonomy_induction', 'float', 0.0, 1.0, NULL, 'none', '0.12'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.batch_size', 'taxonomy_induction', 'int', 1, 20, NULL, 'none', '6'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.max_calls', 'taxonomy_induction', 'int', 0, 20, NULL, 'lower_only', '20'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.max_completion_tokens', 'taxonomy_induction', 'int', 64, 4096, NULL, 'none', '900'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.max_rate_limited', 'taxonomy_induction', 'int', 0, 100, NULL, 'none', '10'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.max_tokens', 'taxonomy_induction', 'int', 100, 60000, NULL, 'lower_only', '60000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.naming.timeout_s', 'taxonomy_induction', 'int', 1, 120, NULL, 'none', '20'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.scan_max_documents', 'taxonomy_induction', 'int', 1, 20000, NULL, 'none', '2000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.subtopic_min_chunks', 'taxonomy_induction', 'int', 2, 100000, NULL, 'none', '30'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('taxonomy_induction.terms_per_chunk', 'taxonomy_induction', 'int', 3, 50, NULL, 'none', '12'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.max_articles', 'wiki_ref_portal', 'int', 1, 3, NULL, 'lower_only', '3'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.max_refs', 'wiki_ref_portal', 'int', 1, 20, NULL, 'lower_only', '20'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.max_response_bytes', 'wiki_ref_portal', 'int', 1, 4000000, NULL, 'lower_only', '4000000'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.query.max_query_chars', 'wiki_ref_portal', 'int', 1, 3000, NULL, 'none', '300'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.query.max_terms', 'wiki_ref_portal', 'int', 1, 8, NULL, 'lower_only', '8'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.relevance.max_refs_scanned', 'wiki_ref_portal', 'int', 1, 500, NULL, 'lower_only', '500'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.relevance.min_contain_len', 'wiki_ref_portal', 'int', 1, 60, NULL, 'none', '6'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.relevance.min_hits', 'wiki_ref_portal', 'int', 2, 20, NULL, 'raise_only', '2'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.relevance.stem_len', 'wiki_ref_portal', 'int', 1, 50, NULL, 'none', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha')),
  ('wiki_ref_portal.timeout_s', 'wiki_ref_portal', 'int', 1, 5, NULL, 'lower_only', '5'::jsonb, 'numeric tunable', current_setting('vera026.seed_sha'));
INSERT INTO vera_vjay.set_deny_lists (list_name, value_pattern, max_entries, description, registered_in_seed) VALUES
  ('public_fetch.deny_hosts', '^(\*\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$', 200, 'deny-only host list (deny-only list per lists-consensus.md)', current_setting('vera026.seed_sha')),
  ('source_permissions.deny_hosts', '^(\*\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$', 200, 'deny-only host list (list already mapped in config/list_tables.json (owner 11-04, lists in DB); access-gating list: combine mode per list_tab)', current_setting('vera026.seed_sha'));
-- END GENERATED SEED

DO $$
BEGIN
  IF (SELECT count(*) FROM vera_vjay.set_keys) <> current_setting('vera026.seed_keys')::int
     OR (SELECT count(*) FROM vera_vjay.set_deny_lists) <> current_setting('vera026.seed_deny_lists')::int
     OR (SELECT count(*) FROM vera_vjay.set_settings) <> 0 OR (SELECT count(*) FROM vera_vjay.set_deny) <> 0 THEN
    RAISE EXCEPTION '026 seed check failed: unexpected row counts.';
  END IF;
END $$;

-- ---------------------------------------------------------------------------------------------------------------------------------
-- 5. Grants: default deny, then SELECT ONLY to the rw and ro groups. No app-path writes (see header).
-- ---------------------------------------------------------------------------------------------------------------------------------
REVOKE ALL ON vera_vjay.set_keys, vera_vjay.set_settings, vera_vjay.set_settings_hist, vera_vjay.set_deny_lists, vera_vjay.set_deny,
              vera_vjay.set_deny_hist, vera_vjay.set_deny_active FROM PUBLIC, vera_vjay_wo, vera_vjay_rw, vera_vjay_ro;
REVOKE ALL ON SEQUENCE vera_vjay.set_settings_setting_id_seq, vera_vjay.set_settings_hist_hist_id_seq, vera_vjay.set_deny_deny_id_seq,
                       vera_vjay.set_deny_hist_hist_id_seq FROM PUBLIC, vera_vjay_wo, vera_vjay_rw, vera_vjay_ro;
REVOKE ALL ON FUNCTION vera_vjay.set_block_fn(), vera_vjay.set_settings_guard_fn(), vera_vjay.set_settings_hist_fn(),
                       vera_vjay.set_deny_guard_fn(), vera_vjay.set_deny_hist_fn() FROM PUBLIC;
DO $$ BEGIN
  EXECUTE pg_catalog.format('REVOKE ALL ON vera_vjay.set_keys, vera_vjay.set_settings, vera_vjay.set_settings_hist, vera_vjay.set_deny_lists, vera_vjay.set_deny, vera_vjay.set_deny_hist, vera_vjay.set_deny_active FROM %I, %I',
                            current_setting('vera026.rw'), current_setting('vera026.ro'));
END $$;
GRANT SELECT ON vera_vjay.set_keys, vera_vjay.set_settings, vera_vjay.set_settings_hist, vera_vjay.set_deny_lists, vera_vjay.set_deny,
                vera_vjay.set_deny_hist, vera_vjay.set_deny_active TO vera_vjay_rw, vera_vjay_ro;

-- Final privilege check against the actual logins (inside the transaction: a RAISE rolls everything back).
DO $$
DECLARE
  rw text := current_setting('vera026.rw'); ro text := current_setting('vera026.ro');
  t text; r text; p text; c text;
BEGIN
  FOREACH t IN ARRAY ARRAY['vera_vjay.set_keys', 'vera_vjay.set_settings', 'vera_vjay.set_settings_hist', 'vera_vjay.set_deny_lists',
                           'vera_vjay.set_deny', 'vera_vjay.set_deny_hist', 'vera_vjay.set_deny_active'] LOOP
    FOREACH r IN ARRAY ARRAY[rw, ro, 'vera_vjay_rw', 'vera_vjay_ro'] LOOP
      IF NOT pg_catalog.has_table_privilege(r, t, 'SELECT') THEN RAISE EXCEPTION '026 grant check failed: % needs SELECT on %.', r, t; END IF;
      FOREACH p IN ARRAY ARRAY['INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER'] LOOP
        IF pg_catalog.has_table_privilege(r, t, p) THEN RAISE EXCEPTION '026 grant check failed: % holds % on %.', r, p, t; END IF;
      END LOOP;
    END LOOP;
    FOR c IN SELECT a.attname FROM pg_catalog.pg_attribute a WHERE a.attrelid = t::pg_catalog.regclass AND a.attnum > 0 AND NOT a.attisdropped LOOP
      FOREACH r IN ARRAY ARRAY[rw, ro] LOOP
        IF pg_catalog.has_column_privilege(r, t, c, 'INSERT') OR pg_catalog.has_column_privilege(r, t, c, 'UPDATE') THEN
          RAISE EXCEPTION '026 grant check failed: % holds a column write privilege on %.%.', r, t, c;
        END IF;
      END LOOP;
    END LOOP;
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class k JOIN pg_catalog.pg_namespace ns ON ns.oid = k.relnamespace, pg_catalog.aclexplode(k.relacl) x
              WHERE ns.nspname = 'vera_vjay' AND k.relname LIKE 'set\_%'
                AND (x.grantee = 0 OR x.grantee = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = 'vera_vjay_wo')
                     OR (x.grantee IN (SELECT oid FROM pg_catalog.pg_roles WHERE rolname IN ('vera_vjay_rw', 'vera_vjay_ro', rw, ro))
                         AND x.privilege_type <> 'SELECT'))) THEN
    RAISE EXCEPTION '026 grant check failed: PUBLIC or vera_vjay_wo holds a privilege, or an app role holds more than SELECT, on a new relation.';
  END IF;
END $$;

COMMIT;
