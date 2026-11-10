-- CHUNK-USAGE-1110: per-chunk total counters (vera_vjay.chunk_use). Run by vera/chunk_usage.py (SQLAlchemy text, executemany) and, byte for byte,
-- by the migration 025 rehearsal through psql with -v span_id=.. chunk_start=.. chunk_end=.. m=.. s=.. u=.. . Numeric parameters only.
INSERT INTO vera_vjay.chunk_use (span_id, chunk_start, chunk_end, matched_count, selected_count, used_count)
VALUES (:span_id, :chunk_start, :chunk_end, :m, :s, :u)
ON CONFLICT (span_id, chunk_start, chunk_end) DO UPDATE SET
  matched_count = vera_vjay.chunk_use.matched_count + EXCLUDED.matched_count,
  selected_count = vera_vjay.chunk_use.selected_count + EXCLUDED.selected_count,
  used_count = vera_vjay.chunk_use.used_count + EXCLUDED.used_count,
  last_seen_day = (now() AT TIME ZONE 'utc')::date;
