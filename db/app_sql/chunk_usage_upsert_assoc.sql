-- CHUNK-USAGE-1110: association counters and weights (vera_vjay.chunk_use_assoc), one row per (partition, node_kind, node_id). Run by
-- vera/chunk_usage.py (SQLAlchemy text, executemany) and, byte for byte, by the migration 025 rehearsal through psql with -v name=value for
-- every parameter below. Numeric parameters only (node_kind is a SMALLINT code, rel is true or false).
-- Rules (vera.chunk_usage.step is the Python statement of the same arithmetic):
--   selected_since_use counts selections without a citation; a use resets it to 0.
--   Once selected_since_use reaches k, each further selection multiplies weight by decay, not below floor * initial_weight.
--   A use moves weight toward initial_weight by the fraction restore per use, never above it.
--   removed_at is set (soft) when weight falls below remove_below (times initial_weight when rel is true) and cleared by a use that lifts
--   weight back to or above it. Nothing is deleted.
INSERT INTO vera_vjay.chunk_use_assoc (span_id, node_kind, node_id, matched_count, selected_count, used_count, selected_since_use, initial_weight, weight)
VALUES (:span_id, :node_kind, :node_id, :m, :s, :u, :ssu0, :init, :init)
ON CONFLICT (span_id, node_kind, node_id) DO UPDATE SET
  matched_count = vera_vjay.chunk_use_assoc.matched_count + EXCLUDED.matched_count,
  selected_count = vera_vjay.chunk_use_assoc.selected_count + EXCLUDED.selected_count,
  used_count = vera_vjay.chunk_use_assoc.used_count + EXCLUDED.used_count,
  selected_since_use = (CASE WHEN EXCLUDED.used_count > 0 THEN 0 ELSE vera_vjay.chunk_use_assoc.selected_since_use + EXCLUDED.selected_count END),
  weight = (CASE
    WHEN EXCLUDED.used_count > 0 THEN LEAST(vera_vjay.chunk_use_assoc.initial_weight, vera_vjay.chunk_use_assoc.weight
      + (vera_vjay.chunk_use_assoc.initial_weight - vera_vjay.chunk_use_assoc.weight) * (1.0 - power(1.0 - CAST(:restore AS double precision), EXCLUDED.used_count)))
    WHEN vera_vjay.chunk_use_assoc.selected_since_use + EXCLUDED.selected_count >= CAST(:k AS bigint)
      THEN GREATEST(CAST(:floor AS double precision) * vera_vjay.chunk_use_assoc.initial_weight, vera_vjay.chunk_use_assoc.weight
        * power(CAST(:decay AS double precision), LEAST(EXCLUDED.selected_count, vera_vjay.chunk_use_assoc.selected_since_use + EXCLUDED.selected_count - CAST(:k AS bigint) + 1)))
    ELSE vera_vjay.chunk_use_assoc.weight END),
  removed_at = (CASE
    WHEN (CASE
      WHEN EXCLUDED.used_count > 0 THEN LEAST(vera_vjay.chunk_use_assoc.initial_weight, vera_vjay.chunk_use_assoc.weight
        + (vera_vjay.chunk_use_assoc.initial_weight - vera_vjay.chunk_use_assoc.weight) * (1.0 - power(1.0 - CAST(:restore AS double precision), EXCLUDED.used_count)))
      WHEN vera_vjay.chunk_use_assoc.selected_since_use + EXCLUDED.selected_count >= CAST(:k AS bigint)
        THEN GREATEST(CAST(:floor AS double precision) * vera_vjay.chunk_use_assoc.initial_weight, vera_vjay.chunk_use_assoc.weight
          * power(CAST(:decay AS double precision), LEAST(EXCLUDED.selected_count, vera_vjay.chunk_use_assoc.selected_since_use + EXCLUDED.selected_count - CAST(:k AS bigint) + 1)))
      ELSE vera_vjay.chunk_use_assoc.weight END)
      < (CASE WHEN CAST(:rel AS boolean) THEN CAST(:thr AS double precision) * vera_vjay.chunk_use_assoc.initial_weight ELSE CAST(:thr AS double precision) END)
      THEN COALESCE(vera_vjay.chunk_use_assoc.removed_at, now())
    WHEN EXCLUDED.used_count > 0 THEN NULL
    ELSE vera_vjay.chunk_use_assoc.removed_at END),
  last_seen_day = (now() AT TIME ZONE 'utc')::date;
