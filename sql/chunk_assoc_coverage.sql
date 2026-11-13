\set ON_ERROR_STOP on
-- WP-C (CHUNK-ASSOC-1113, plan D6): per-chunk association coverage with derived reason classes. READ ONLY.
-- OWNER-RUN ONLY, on the read-only login, through work/WP-C-1113/run_coverage.sh. Agents never run this file.
-- SELECT only: no DML, no DDL, no temp objects. One transaction READ ONLY; the guard below refuses a login that can write.
-- Output: ONE line of JSON (counts, ages in hours, run-row numbers). No span text, titles, URLs or labels are ever selected.
-- The verdict (PENDING-CAL / PASS / FAIL) is NOT computed here: scripts/chunk_assoc_coverage.py reads this JSON, the job-counted
-- exclusions and the owner's limits (OD-J) and prints it, so the query alone cannot pass with zero links.
--
-- Variables (psql -v): cn (classifier name), model (embedder model id), text_chars (job.text_chars: the job embeds left(text, n)).
-- Counting rule (D6): a level counts only through a STRICT link (score_kind NULL, 'cosine' or 'cosine_lvl') of classifier :cn in state
-- 'active' to an ACTIVE entity with an ACTIVE parent chain (same predicate as vera/corpus_taxonomy.py _CL and count_labels Q6).
-- Routing kinds (cosine_r<n>, cosine_soft) never count. cosine_lvl is strict here (count_labels Q5 would call it routing).
-- Domain level: no chunk-domain table exists before WP-D (OD-A), so it is derived through tax_domain_topic ("derived, not stored").
-- Classes (derived at query time, first match wins): covered_all (split stored/derived), partial_depth, no_vector_yet, stale_vector,
-- new_since_build, abstained, unexplained. Job-only verdicts (portal, portal_error, no_source, empty, quarantined) are read from the
-- run row and subtracted by the script.
-- Run-row contract (WP-I writes these numbers; until then the script prints PENDING-CAL): build row = latest state 'finished' row of
-- :cn with numeric metrics max_span_id and scan_end_epoch (a build that reached W4); scan row = latest row of :cn with
-- scan_complete = 1. Counts are read from top-level metrics or metrics->'run'.

\if :{?cn}
\else
  \set cn chunk_taxo_auto
\endif
\if :{?model}
\else
  \set model '@cf/baai/bge-m3'
\endif
\if :{?text_chars}
\else
  \set text_chars 8000
\endif

BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '120s';
SET LOCAL lock_timeout = '5s';
DO $g$ BEGIN
  IF current_database() <> 'vera_dar_db' THEN RAISE EXCEPTION 'not vera_dar_db'; END IF;
  IF current_setting('transaction_read_only') <> 'on' THEN RAISE EXCEPTION 'transaction is not read only'; END IF;
  IF has_table_privilege('vera_vjay.sources', 'INSERT') OR has_table_privilege('vera_vjay.sources', 'UPDATE')
     OR has_table_privilege('vera_vjay.sources', 'DELETE')
     OR has_table_privilege('vera_vjay.tax_job_runs', 'INSERT') OR has_table_privilege('vera_vjay.tax_job_runs', 'UPDATE')
     OR has_table_privilege('vera_vjay.tax_chunk_vectors', 'INSERT') OR has_table_privilege('vera_vjay.tax_chunk_topic', 'INSERT')
     OR has_table_privilege('vera_vjay.tax_chunk_subtopic', 'INSERT') OR has_table_privilege('vera_vjay.tax_chunk_subtopic', 'UPDATE') THEN
    RAISE EXCEPTION 'not the read-only login';
  END IF;
END $g$;

WITH
strict_ct AS (
  SELECT ct.span_id, ct.topic_id
  FROM vera_vjay.tax_chunk_topic ct
  JOIN vera_vjay.tax_topics t ON t.topic_id = ct.topic_id AND t.state = 'active'
  WHERE ct.classifier_name = :'cn' AND ct.state = 'active'
    AND (ct.score_kind IS NULL OR ct.score_kind IN ('cosine', 'cosine_lvl'))
),
strict_cs AS (  -- sub-topic link whose sub-topic, topic-subtopic pair and parent topic are all active; one row per (span, parent topic)
  SELECT cs.span_id, cs.subtopic_id, ts.topic_id
  FROM vera_vjay.tax_chunk_subtopic cs
  JOIN vera_vjay.tax_subtopics sb ON sb.subtopic_id = cs.subtopic_id AND sb.state = 'active'
  JOIN vera_vjay.tax_topic_subtopic ts ON ts.subtopic_id = cs.subtopic_id AND ts.state = 'active'
  JOIN vera_vjay.tax_topics t ON t.topic_id = ts.topic_id AND t.state = 'active'
  WHERE cs.classifier_name = :'cn' AND cs.state = 'active'
    AND (cs.score_kind IS NULL OR cs.score_kind IN ('cosine', 'cosine_lvl'))
),
dom_topic AS (  -- topic -> active domain through an active pair
  SELECT dt.topic_id FROM vera_vjay.tax_domain_topic dt
  JOIN vera_vjay.tax_domains d ON d.domain_id = dt.domain_id AND d.state = 'active'
  WHERE dt.state = 'active' GROUP BY dt.topic_id
),
lvl AS (
  SELECT e.span_id, e.source_id, e.created_at,
         v.span_id IS NOT NULL AS has_vec,
         v.updated_at AS vec_at,
         (v.span_id IS NOT NULL AND v.model = :'model'
            AND v.text_sha256 = encode(sha256(convert_to(left(e.text, :text_chars), 'UTF8')), 'hex')) AS vec_current,
         EXISTS (SELECT 1 FROM strict_cs x WHERE x.span_id = e.span_id) AS sub_ok,
         EXISTS (SELECT 1 FROM strict_ct x WHERE x.span_id = e.span_id) AS top_stored_any,
         EXISTS (SELECT 1 FROM (SELECT span_id, topic_id FROM strict_ct UNION SELECT span_id, topic_id FROM strict_cs) u
                   JOIN dom_topic dm ON dm.topic_id = u.topic_id WHERE u.span_id = e.span_id) AS dom_derived
  FROM vera_vjay.evidence_spans e
  LEFT JOIN vera_vjay.tax_chunk_vectors v ON v.span_id = e.span_id
),
bld AS (  -- latest build that reached W4: a finished row carrying the build snapshot
  SELECT r.run_id, r.finished_at, r.classifier_version,
         (r.metrics->>'max_span_id')::bigint AS max_span_id, (r.metrics->>'scan_end_epoch')::double precision AS scan_end_epoch,
         r.metrics AS m
  FROM vera_vjay.tax_job_runs r
  WHERE r.classifier_name = :'cn' AND r.state = 'finished' AND r.classifier_version NOT IN ('pending', 'inert')
    AND jsonb_typeof(r.metrics->'max_span_id') = 'number' AND jsonb_typeof(r.metrics->'scan_end_epoch') = 'number'
  ORDER BY r.run_id DESC LIMIT 1
),
scn AS (  -- latest row whose scan was complete (its exclusion counts are exact, not lower bounds)
  SELECT r.run_id, r.state, r.finished_at, r.started_at, r.metrics AS m
  FROM vera_vjay.tax_job_runs r
  WHERE r.classifier_name = :'cn' AND r.state IN ('finished', 'deferred')
    AND coalesce(r.metrics->>'scan_complete', r.metrics->'run'->>'scan_complete') = '1'
  ORDER BY r.run_id DESC LIMIT 1
),
cls AS (
  SELECT l.*,
    CASE
      -- full depth = strict sub-topic (parent topic active by construction) and an active domain above it. No chunk-domain table
      -- exists before WP-D, so the domain level is always derived and covered_all_stored stays 0 until a stored domain source is added.
      WHEN l.sub_ok AND l.dom_derived THEN 'covered_all_derived'
      WHEN l.sub_ok OR l.top_stored_any THEN 'partial_depth'
      WHEN NOT l.has_vec THEN 'no_vector_yet'
      WHEN NOT l.vec_current THEN 'stale_vector'
      WHEN (SELECT run_id FROM bld) IS NULL THEN 'unexplained'
      WHEN l.span_id > (SELECT max_span_id FROM bld)
           OR extract(epoch FROM l.vec_at) > (SELECT scan_end_epoch FROM bld) THEN 'new_since_build'
      ELSE 'abstained'
    END AS cls,
    CASE
      WHEN NOT l.has_vec THEN extract(epoch FROM (now() - l.created_at)) / 3600.0
      ELSE extract(epoch FROM (now() - l.vec_at)) / 3600.0
    END AS age_h
  FROM lvl l
),
keys AS (SELECT unnest(ARRAY['skipped_portal', 'skipped_portal_error', 'skipped_no_source', 'skipped_empty', 'quarantined',
                             'excluded_with_vector_portal', 'excluded_with_vector_portal_error', 'excluded_with_vector_no_source',
                             'excluded_with_vector_empty', 'abstained', 'scan_complete']) AS k)
SELECT json_build_object(
  'spans', (SELECT count(*) FROM cls),
  'with_vector', (SELECT count(*) FROM cls WHERE has_vec),
  'sources_total', (SELECT count(DISTINCT source_id) FROM cls),
  'sources_with_covered', (SELECT count(DISTINCT source_id) FROM cls WHERE cls LIKE 'covered_all%'),
  'classes', (SELECT coalesce(json_object_agg(cls, n), '{}'::json) FROM (SELECT cls, count(*) AS n FROM cls GROUP BY cls) c),
  -- the 2000 oldest ages (hours, descending) per waiting class; the script skips the oldest k when k job-excluded spans may be among them
  'ages_desc', (SELECT coalesce(json_object_agg(cls, a), '{}'::json)
                  FROM (SELECT cls, (array_agg(round(age_h::numeric, 2) ORDER BY age_h DESC))[1:2000] AS a FROM cls
                         WHERE cls IN ('no_vector_yet', 'stale_vector', 'new_since_build') GROUP BY cls) g),
  'build', (SELECT json_build_object('run_id', b.run_id, 'age_hours', round((extract(epoch FROM (now() - b.finished_at)) / 3600.0)::numeric, 2),
                                     'max_span_id', b.max_span_id,
                                     'counts', (SELECT coalesce(json_object_agg(k.k, coalesce(b.m->k.k, b.m->'run'->k.k)), '{}'::json)
                                                  FROM keys k WHERE coalesce(b.m->k.k, b.m->'run'->k.k) IS NOT NULL))
              FROM bld b),
  'scan', (SELECT json_build_object('run_id', s.run_id, 'state', s.state,
                                    'age_hours', round((extract(epoch FROM (now() - coalesce(s.finished_at, s.started_at))) / 3600.0)::numeric, 2),
                                    'counts', (SELECT coalesce(json_object_agg(k.k, coalesce(s.m->k.k, s.m->'run'->k.k)), '{}'::json)
                                                 FROM keys k WHERE coalesce(s.m->k.k, s.m->'run'->k.k) IS NOT NULL))
             FROM scn s)
)::text AS coverage_json;
COMMIT;
