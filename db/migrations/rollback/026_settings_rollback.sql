-- 026_settings rollback. SETTINGS-W2-1110 DRAFT. NOT EXECUTED. OWNER-ONLY, admin-run, never via db/apply_migrations.py.
-- A prepared undo path that is never expected to run (OD-1031-ROLLBACK-KEEP). Drops the seven 026 relations (view, six tables) and the six
-- set_* functions in vera_vjay. Nothing else is touched: no CASCADE; every role and every other table is unchanged.
-- DATA-SAFE: it REFUSES while any row exists BEYOND THE SEED: any set_settings, set_settings_hist, set_deny or set_deny_hist row, or any
-- set_keys / set_deny_lists row not registered by the pinned seed (registered_in_seed <> the seed sha below). To roll back anyway, the owner
-- first exports (rollback_026.sh --export), then runs this file with -v purge_rows=1 (the tables are dropped, which fires no row trigger).
-- RUN WITH: psql -X -v ON_ERROR_STOP=1 [-v purge_rows=1] -f <this file>   (own BEGIN/COMMIT; any error rolls everything back).
\if :{?purge_rows}
\else
  \set purge_rows 0
\endif
BEGIN;
SELECT set_config('vera026rb.purge', :'purge_rows', true);
SET LOCAL lock_timeout = '5s';
SET LOCAL search_path = pg_catalog, pg_temp;
-- BEGIN GENERATED SEED (manifest sha256 211617211f635e175aba00281febbc2b56514357a845e10389fa43d2152e9008)
SELECT set_config('vera026rb.seed_sha', '211617211f635e175aba00281febbc2b56514357a845e10389fa43d2152e9008', true), set_config('vera026rb.seed_keys', '153', true), set_config('vera026rb.seed_deny_lists', '2', true);
-- END GENERATED SEED

DO $$
DECLARE n text; extra bigint := 0; cnt bigint; seedrows bigint;
BEGIN
  FOREACH n IN ARRAY ARRAY['set_keys','set_settings','set_settings_hist','set_deny_lists','set_deny','set_deny_hist'] LOOP
    IF pg_catalog.to_regclass('vera_vjay.' || n) IS NULL THEN
      RAISE EXCEPTION '026 rollback: table % is missing (026 not applied, partly applied, or already rolled back); nothing done.', n;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class c WHERE c.oid = pg_catalog.to_regclass('vera_vjay.' || n)
                    AND pg_catalog.pg_get_userbyid(c.relowner) = current_user) THEN
      RAISE EXCEPTION '026 rollback must run as the owner of the 026 tables.';
    END IF;
  END LOOP;
  FOREACH n IN ARRAY ARRAY['set_deny_hist','set_deny','set_deny_lists','set_settings_hist','set_settings','set_keys'] LOOP
    EXECUTE pg_catalog.format('LOCK TABLE vera_vjay.%I IN ACCESS EXCLUSIVE MODE', n);
    IF n IN ('set_keys', 'set_deny_lists') THEN
      EXECUTE pg_catalog.format('SELECT count(*) FROM vera_vjay.%I WHERE registered_in_seed <> $1', n) INTO cnt USING current_setting('vera026rb.seed_sha');
      EXECUTE pg_catalog.format('SELECT count(*) FROM vera_vjay.%I WHERE registered_in_seed = $1', n) INTO seedrows USING current_setting('vera026rb.seed_sha');
      RAISE NOTICE '026 rollback: % rows beyond the seed in % (seed rows %)', cnt, n, seedrows;
    ELSE
      EXECUTE pg_catalog.format('SELECT count(*) FROM vera_vjay.%I', n) INTO cnt;
      RAISE NOTICE '026 rollback: % rows in % (all beyond the seed)', cnt, n;
    END IF;
    extra := extra + cnt;
  END LOOP;
  IF extra > 0 AND current_setting('vera026rb.purge') <> '1' THEN
    RAISE EXCEPTION '026 rollback refused: % rows exist beyond the seed. Export them, then re-run with -v purge_rows=1.', extra;
  END IF;
  IF extra > 0 THEN RAISE NOTICE '026 rollback: dropping % rows beyond the seed (purge_rows=1)', extra; END IF;
END $$;

DROP VIEW vera_vjay.set_deny_active;
DROP TABLE vera_vjay.set_deny_hist;
DROP TABLE vera_vjay.set_deny;
DROP TABLE vera_vjay.set_deny_lists;
DROP TABLE vera_vjay.set_settings_hist;
DROP TABLE vera_vjay.set_settings;
DROP TABLE vera_vjay.set_keys;
DROP FUNCTION vera_vjay.set_deny_hist_fn();
DROP FUNCTION vera_vjay.set_deny_guard_fn();
DROP FUNCTION vera_vjay.set_settings_hist_fn();
DROP FUNCTION vera_vjay.set_settings_guard_fn();
DROP FUNCTION vera_vjay.set_block_fn();
DROP FUNCTION vera_vjay.set_hist_insert_fn();

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace ns ON ns.oid = c.relnamespace
              WHERE ns.nspname = 'vera_vjay' AND c.relname LIKE 'set\_%')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace ns ON ns.oid = p.pronamespace
                 WHERE ns.nspname = 'vera_vjay' AND p.proname LIKE 'set\_%') THEN
    RAISE EXCEPTION '026 rollback: a 026 object is still present after the drop; rolled back.';
  END IF;
END $$;

COMMIT;
