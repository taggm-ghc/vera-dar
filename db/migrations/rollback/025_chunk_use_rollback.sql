-- 025_chunk_use rollback. CHUNK-USAGE-1110 DRAFT. NOT EXECUTED. OWNER-ONLY, admin-run, never via db/apply_migrations.py.
-- A prepared undo path that is never expected to run (OD-1031-ROLLBACK-KEEP). Drops vera_vjay.chunk_use_assoc and vera_vjay.chunk_use. Nothing else is touched:
-- no CASCADE; evidence_spans, the 020/022/024 objects and every role are unchanged.
-- DATA-SAFE: it REFUSES while any row exists (counters are history that cannot be recomputed). To roll back anyway, the owner first
-- exports (rollback_025.sh --export), then runs this file with -v purge_rows=1, which deletes the rows inside the same transaction.
-- RUN WITH: psql -X -v ON_ERROR_STOP=1 [-v purge_rows=1] -f <this file>   (own BEGIN/COMMIT; any error rolls everything back).
\if :{?purge_rows}
\else
  \set purge_rows 0
\endif
BEGIN;
SELECT set_config('vera025rb.purge', :'purge_rows', true);
SET LOCAL lock_timeout = '5s';

DO $$
DECLARE n text; cnt bigint; total bigint := 0;
BEGIN
  IF pg_catalog.to_regclass('vera_vjay.chunk_use') IS NULL OR pg_catalog.to_regclass('vera_vjay.chunk_use_assoc') IS NULL THEN
    RAISE EXCEPTION '025 rollback: a 025 table is missing (025 not applied, partly applied, or already rolled back); nothing done.';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class c WHERE c.oid = 'vera_vjay.chunk_use'::pg_catalog.regclass
                    AND pg_catalog.pg_get_userbyid(c.relowner) = current_user)
     OR NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class c WHERE c.oid = 'vera_vjay.chunk_use_assoc'::pg_catalog.regclass
                    AND pg_catalog.pg_get_userbyid(c.relowner) = current_user) THEN
    RAISE EXCEPTION '025 rollback must run as the owner of the 025 tables.';
  END IF;
  FOREACH n IN ARRAY ARRAY['chunk_use_assoc','chunk_use'] LOOP
    EXECUTE pg_catalog.format('LOCK TABLE vera_vjay.%I IN ACCESS EXCLUSIVE MODE', n);
    EXECUTE pg_catalog.format('SELECT count(*) FROM vera_vjay.%I', n) INTO cnt;
    RAISE NOTICE '025 rollback: % rows in %', cnt, n;
    total := total + cnt;
  END LOOP;
  IF total > 0 AND current_setting('vera025rb.purge') <> '1' THEN
    RAISE EXCEPTION '025 rollback refused: % rows exist. Export them, then re-run with -v purge_rows=1.', total;
  END IF;
  IF total > 0 THEN
    DELETE FROM vera_vjay.chunk_use_assoc;
    DELETE FROM vera_vjay.chunk_use;
    RAISE NOTICE '025 rollback: purged % rows (purge_rows=1)', total;
  END IF;
END $$;

DROP TABLE vera_vjay.chunk_use_assoc;
DROP TABLE vera_vjay.chunk_use;

DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace ns ON ns.oid = c.relnamespace
              WHERE ns.nspname = 'vera_vjay' AND c.relname LIKE 'chunk\_use%') THEN
    RAISE EXCEPTION '025 rollback: a 025 object is still present after the drop; rolled back.';
  END IF;
  IF pg_catalog.to_regclass('vera_vjay.evidence_spans') IS NULL THEN
    RAISE EXCEPTION '025 rollback: evidence_spans is missing after the drop; rolled back.';
  END IF;
END $$;

COMMIT;
