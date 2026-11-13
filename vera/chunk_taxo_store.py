"""CHUNK-TAXO-JOB-1109: the ONLY SQL the chunk-taxonomy job can issue (plan v3.1; migrations 020 and 022 drafts).

Every statement is a fixed, parameterised string in the module-level SQL allowlist below; run_sql() refuses any other name.
No row deletion, truncation or schema change exists here. Columns touched are exactly those in the 020 rw grants (INSERT slug,label,description,
origin,origin_ref,state on the entities; INSERT <ids>,assigned_via,classifier_*,score,score_kind on the six junctions; UPDATE
state) and the 022 grants (tax_anchors INSERT topic_id,subtopic_id,anchor_text,core_terms,origin,origin_ref,state; UPDATE state).
Writes happen as the API rw login; entity inserts start 'proposed' and move to 'active' in the same transaction under the 022
guard (session settings vera.tax_actor and vera.tax_batch, set transaction-local with set_config(..., true)).

Retirement is scoped: links only where classifier_name = CLASSIFIER and the version is superseded; entities and anchors only
where origin = 'induction' and origin_ref LIKE 'auto-chunk-taxo:%'. Nothing here logs; callers log codes only.

ARXIV-DEFENSE-1113: one read-only statement, sources_portal (SELECT on vera_vjay.sources and vera_vjay.canonical_sources, granted by
db/migrations/001_vera_schema_init.sql), feeds the job's portal check; the verdict itself is fulltext_loader.portal_verdict, not SQL.
"""
from __future__ import annotations

from types import MappingProxyType

CLASSIFIER = "chunk_taxo_auto"
ORIGIN_REF_PREFIX = "auto-chunk-taxo:"
ORIGIN_REF_LIKE = "auto-chunk-taxo:%"
ACTOR = "auto:chunk_taxo"
RW_GROUP = "vera_vjay_rw"
WO_GROUP = "vera_vjay_wo"

LEVELS = ("domain", "topic", "subtopic")
_ENT = {"domain": ("vera_vjay.tax_domains", "domain_id"), "topic": ("vera_vjay.tax_topics", "topic_id"),
        "subtopic": ("vera_vjay.tax_subtopics", "subtopic_id")}
# link kind -> (table, left id column, right id column)
LINKS = {
    "domain_topic": ("vera_vjay.tax_domain_topic", "domain_id", "topic_id"),
    "topic_subtopic": ("vera_vjay.tax_topic_subtopic", "topic_id", "subtopic_id"),
    "chunk_subtopic": ("vera_vjay.tax_chunk_subtopic", "span_id", "subtopic_id"),
    "chunk_topic": ("vera_vjay.tax_chunk_topic", "span_id", "topic_id"),
    "source_topic": ("vera_vjay.tax_source_topic", "source_id", "topic_id"),
    "source_domain": ("vera_vjay.tax_source_domain", "source_id", "domain_id"),
    # WP-D-1113 (migration 027 draft, OD-A): chunk -> domain, derived from the chunk's topic links; written by vera/chunk_domain_writer.py
    # only when the table exists (probe_chunk_domain). Not in CHUNK_LINKS: it has no link_select and is never read as a prior.
    "chunk_domain": ("vera_vjay.tax_chunk_domain", "span_id", "domain_id"),
}
HIERARCHY = ("domain_topic", "topic_subtopic")
CHUNK_LINKS = ("chunk_subtopic", "chunk_topic")
SOURCE_LINKS = ("source_topic", "source_domain")
_ANCHOR_COL = {"topic": "topic_id", "subtopic": "subtopic_id"}


def _build() -> dict[str, str]:
    s: dict[str, str] = {}
    s["probe_account"] = (
        "SELECT current_user::text, "
        "pg_catalog.pg_has_role(current_user, '" + RW_GROUP + "', 'MEMBER'), "
        "(EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '" + WO_GROUP + "') "
        "AND pg_catalog.pg_has_role(current_user, '" + WO_GROUP + "', 'MEMBER'))")
    s["probe_schema"] = ("SELECT pg_catalog.to_regclass('vera_vjay.tax_topics') IS NOT NULL, "
                         "pg_catalog.to_regclass('vera_vjay.tax_anchors') IS NOT NULL, "
                         "(pg_catalog.to_regclass('vera_vjay.tax_chunk_vectors') IS NOT NULL "
                         "AND pg_catalog.to_regclass('vera_vjay.tax_job_runs') IS NOT NULL)")
    # WP-D-1113 (plan A4): true only when the table exists AND this login holds column INSERT on each written column and UPDATE on state
    # (column functions, never has_table_privilege: the grants are column-level, K6). Absent table or a missing privilege reads as false.
    _dom_cols = ("span_id", "domain_id", "assigned_via", "classifier_name", "classifier_version", "score", "score_kind")
    _dom_oid = "pg_catalog.to_regclass('vera_vjay.tax_chunk_domain')"
    s["probe_chunk_domain"] = ("SELECT COALESCE(" + " AND ".join(
        [f"pg_catalog.has_column_privilege(current_user, {_dom_oid}, '{c}', 'INSERT')" for c in _dom_cols]
        + [f"pg_catalog.has_column_privilege(current_user, {_dom_oid}, 'state', 'UPDATE')"]) + ", false)")
    # 024 (draft CHUNK-TAXO-MIG024-1109 bus contract #6): vectors are cached per span keyed by the text sha256; one run row per finished run.
    s["vectors_select"] = ("SELECT span_id, model, text_sha256, vec FROM vera_vjay.tax_chunk_vectors WHERE span_id = ANY(:ids)")
    s["vectors_upsert"] = ("INSERT INTO vera_vjay.tax_chunk_vectors (span_id, model, dim, text_sha256, vec) "
                           "VALUES (:span_id, :model, :dim, :sha, CAST(:vec AS real[])) ON CONFLICT (span_id) DO UPDATE SET "
                           "model = EXCLUDED.model, dim = EXCLUDED.dim, text_sha256 = EXCLUDED.text_sha256, vec = EXCLUDED.vec, updated_at = now()")
    # Run rows (024 + C6 revised contract, OD-1109-DEMB-3 b): a 'running' row is inserted at pass start (classifier_version
    # 'pending', finished_at NULL, display_ok NULL) and neurons_used is updated after each embed batch, so the daily neuron
    # counter survives a restart. The finished row (state 'finished') carries the version, metrics and display_ok and has
    # neurons_used 0 (the running row holds the pass's neurons; SUM per utc_day never double counts).
    s["job_run_start"] = ("INSERT INTO vera_vjay.tax_job_runs (classifier_name, classifier_version, started_at, metrics, reasons, utc_day, "
                          f"neurons_used, state) VALUES ('{CLASSIFIER}', 'pending', now(), CAST('{{}}' AS jsonb), CAST('{{}}' AS text[]), "
                          "timezone('utc', now())::date, 0, 'running') RETURNING run_id")
    s["job_run_neurons"] = ("UPDATE vera_vjay.tax_job_runs SET neurons_used = :neurons WHERE run_id = :run_id AND state = 'running'")
    s["job_run_close"] = ("UPDATE vera_vjay.tax_job_runs SET finished_at = now(), display_ok = false, state = :state, neurons_used = :neurons, "
                          "metrics = CAST(:metrics AS jsonb), reasons = CAST(:reasons AS text[]) WHERE run_id = :run_id AND state = 'running'")
    s["job_run_insert"] = ("INSERT INTO vera_vjay.tax_job_runs (classifier_name, classifier_version, started_at, finished_at, metrics, display_ok, "
                           "reasons, utc_day, neurons_used, state) "
                           f"VALUES ('{CLASSIFIER}', :version, to_timestamp(:started), to_timestamp(:finished), CAST(:metrics AS jsonb), :display_ok, "
                           "CAST(:reasons AS text[]), timezone('utc', now())::date, 0, 'finished')")
    # Self-report row (TAXO-DIAG-FIX-1110): one closed 'failed' row when a pass ends inert; display_ok false satisfies the display/finished checks.
    s["job_run_inert"] = ("INSERT INTO vera_vjay.tax_job_runs (classifier_name, classifier_version, started_at, finished_at, metrics, display_ok, "
                          "reasons, utc_day, neurons_used, state) "
                          f"VALUES ('{CLASSIFIER}', 'inert', now(), now(), CAST('{{}}' AS jsonb), false, "
                          "CAST(:reasons AS text[]), timezone('utc', now())::date, 0, 'failed')")
    s["neurons_today"] = ("SELECT COALESCE(SUM(neurons_used), 0) FROM vera_vjay.tax_job_runs "
                          f"WHERE classifier_name = '{CLASSIFIER}' AND utc_day = timezone('utc', now())::date")
    s["job_run_latest"] = ("SELECT classifier_version, finished_at, metrics, display_ok, reasons FROM vera_vjay.tax_job_runs "
                           f"WHERE classifier_name = '{CLASSIFIER}' AND state = 'finished' AND finished_at IS NOT NULL "
                           "AND classifier_version <> 'pending' ORDER BY finished_at DESC LIMIT 1")
    s["set_ctx"] = ("SELECT pg_catalog.set_config('vera.tax_actor', :actor, true), "
                    "pg_catalog.set_config('vera.tax_batch', :batch, true)")
    s["spans_page"] = ("SELECT span_id, source_id, left(text, :n) FROM vera_vjay.evidence_spans "
                       "WHERE span_id > :after ORDER BY span_id LIMIT :lim")
    # ARXIV-DEFENSE-1113: read-only lookup for the portal check on a page's source ids (columns and tables already read elsewhere;
    # SELECT is granted by db/migrations/001_vera_schema_init.sql). Literal schema name so the SQL contract extractor sees it.
    s["sources_portal"] = ("SELECT s.source_id, cs.canonical_url, s.provenance FROM vera_vjay.sources s "
                           "JOIN vera_vjay.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
                           "WHERE s.source_id = ANY(:ids)")
    for lvl in LEVELS:
        t, idc = _ENT[lvl]
        s[f"nodes_{lvl}"] = f"SELECT slug, {idc}, label, origin, origin_ref, state FROM {t} WHERE state IN ('proposed', 'active')"
        s[f"node_ids_{lvl}"] = f"SELECT slug, {idc}, state FROM {t} WHERE slug = ANY(:slugs)"
        s[f"node_insert_{lvl}"] = (f"INSERT INTO {t} (slug, label, description, origin, origin_ref, state) "
                                   "VALUES (:slug, :label, NULL, :origin, :origin_ref, :state) ON CONFLICT (slug) DO NOTHING")
        s[f"node_activate_{lvl}"] = (f"UPDATE {t} SET state = 'active' WHERE slug = ANY(:slugs) AND state = 'proposed' "
                                     f"AND origin = 'induction' AND origin_ref LIKE '{ORIGIN_REF_LIKE}'")
        s[f"node_retire_{lvl}"] = (f"UPDATE {t} SET state = 'retired' WHERE slug = ANY(:slugs) AND state = 'active' "
                                   f"AND origin = 'induction' AND origin_ref LIKE '{ORIGIN_REF_LIKE}'")
    for lvl, col in _ANCHOR_COL.items():
        t, idc = _ENT[lvl]
        s[f"anchor_insert_{lvl}"] = (f"INSERT INTO vera_vjay.tax_anchors ({col}, anchor_text, core_terms, origin, origin_ref, state) "
                                     f"VALUES (:eid, :text, CAST(:terms AS text[]), 'induction', :origin_ref, 'proposed') ON CONFLICT DO NOTHING")
        s[f"anchor_activate_{lvl}"] = (f"UPDATE vera_vjay.tax_anchors SET state = 'active' WHERE {col} = ANY(:eids) AND state = 'proposed' "
                                       f"AND origin = 'induction' AND origin_ref LIKE '{ORIGIN_REF_LIKE}'")
        s[f"anchor_retire_{lvl}"] = (f"UPDATE vera_vjay.tax_anchors SET state = 'retired' WHERE {col} = ANY(:eids) AND state = 'active' "
                                     f"AND origin = 'induction' AND origin_ref LIKE '{ORIGIN_REF_LIKE}'")
    for kind, (t, a, b) in LINKS.items():
        s[f"link_insert_{kind}"] = (f"INSERT INTO {t} ({a}, {b}, assigned_via, classifier_name, classifier_version, score, score_kind) "
                                    f"VALUES (:a, :b, 'backfill', '{CLASSIFIER}', :version, :score, :score_kind) ON CONFLICT DO NOTHING")
        if kind in HIERARCHY:
            s[f"link_select_{kind}"] = f"SELECT {a}, {b} FROM {t} WHERE classifier_name = '{CLASSIFIER}' AND state = 'active'"
            s[f"link_retire_{kind}"] = (f"UPDATE {t} SET state = 'retired' WHERE classifier_name = '{CLASSIFIER}' "
                                        f"AND state = 'active' AND classifier_version <> :version")
        elif kind in CHUNK_LINKS:
            s[f"link_select_{kind}"] = (f"SELECT {a}, {b} FROM {t} WHERE classifier_name = '{CLASSIFIER}' AND state = 'active' "
                                        f"AND {a} > :after ORDER BY {a}, {b} LIMIT :lim")
            s[f"link_retire_{kind}"] = (f"UPDATE {t} SET state = 'retired' WHERE classifier_name = '{CLASSIFIER}' AND state = 'active' "
                                        f"AND classifier_version <> :version AND {a} = ANY(:ids)")
        else:
            s[f"link_retire_{kind}"] = (f"UPDATE {t} SET state = 'retired' WHERE classifier_name = '{CLASSIFIER}' AND state = 'active' "
                                        f"AND classifier_version <> :version AND {a} = ANY(:ids)")
    return s


SQL = MappingProxyType(_build())
ALLOWED_TEXTS = frozenset(SQL.values())


class SqlRefused(RuntimeError):
    pass


def run_sql(conn, name: str, params=None):
    """Execute allowlisted statement `name`. `params` may be a dict or (executemany) a non-empty list of dicts."""
    stmt = SQL.get(name)
    if stmt is None or stmt not in ALLOWED_TEXTS:
        raise SqlRefused("sql_not_allowlisted")
    from sqlalchemy import text  # noqa: PLC0415
    if params is None:
        return conn.execute(text(stmt))
    if isinstance(params, list) and not params:
        return None
    return conn.execute(text(stmt), params)


def probe_account(conn) -> tuple[bool, str]:
    row = run_sql(conn, "probe_account").first()
    if row is None:
        return False, "account_unknown"
    _user, in_rw, in_wo = row[0], bool(row[1]), bool(row[2])
    if in_wo:
        return False, "account_is_wo"
    if not in_rw:
        return False, "account_not_rw"
    return True, "ok"


def probe_schema(conn) -> tuple[bool, bool, bool]:
    """(020 present, 022 present, 024 present)."""
    row = run_sql(conn, "probe_schema").first()
    if row is None:
        return False, False, False
    return bool(row[0]), bool(row[1]), bool(row[2])


def fetch_vectors(conn, ids: list[int]) -> dict[int, tuple[str, str, "np.ndarray"]]:
    """One page of stored vectors as compact float32 arrays (4 bytes per value, never Python float lists). Callers pass a
    bounded page of ids; the whole set is never decoded at once."""
    import numpy as np  # noqa: PLC0415
    if not ids:
        return {}
    return {int(r[0]): (str(r[1]), str(r[2]), np.asarray(r[3], dtype=np.float32))
            for r in run_sql(conn, "vectors_select", {"ids": list(ids)}).fetchall()}


def fetch_spans(conn, after: int, limit: int, chars: int) -> list[tuple[int, int, str]]:
    rows = run_sql(conn, "spans_page", {"after": int(after), "lim": int(limit), "n": int(chars)}).fetchall()
    return [(int(r[0]), int(r[1]), str(r[2] or "")) for r in rows]


def fetch_source_portal(conn, ids: list[int]) -> list[tuple[int, str | None, object]]:
    """Raw (source_id, canonical_url, provenance) rows for the given source ids; the portal verdict is computed by the caller."""
    if not ids:
        return []
    return [(int(r[0]), None if r[1] is None else str(r[1]), r[2]) for r in run_sql(conn, "sources_portal", {"ids": list(ids)}).fetchall()]


def fetch_nodes(conn, level: str) -> list[dict]:
    out = []
    for r in run_sql(conn, f"nodes_{level}").fetchall():
        out.append({"slug": str(r[0]), "id": int(r[1]), "label": str(r[2]), "origin": str(r[3]),
                    "origin_ref": None if r[4] is None else str(r[4]), "state": str(r[5])})
    return out


def node_ids(conn, level: str, slugs: list[str]) -> dict[str, tuple[int, str]]:
    if not slugs:
        return {}
    return {str(r[0]): (int(r[1]), str(r[2])) for r in run_sql(conn, f"node_ids_{level}", {"slugs": list(slugs)}).fetchall()}


def set_ctx(conn, batch_sha256: str) -> None:
    run_sql(conn, "set_ctx", {"actor": ACTOR, "batch": batch_sha256})


def fetch_hierarchy(conn, kind: str) -> list[tuple[int, int]]:
    return [(int(r[0]), int(r[1])) for r in run_sql(conn, f"link_select_{kind}").fetchall()]


def fetch_chunk_links(conn, kind: str, after: int, limit: int) -> list[tuple[int, int]]:
    return [(int(r[0]), int(r[1])) for r in run_sql(conn, f"link_select_{kind}", {"after": int(after), "lim": int(limit)}).fetchall()]


def run_start(conn) -> int | None:
    row = run_sql(conn, "job_run_start").first()
    return int(row[0]) if row is not None else None


def neurons_today(conn) -> float:
    row = run_sql(conn, "neurons_today").first()
    return float(row[0]) if row is not None and row[0] is not None else 0.0
