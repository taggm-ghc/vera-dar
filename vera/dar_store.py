"""DAR persistence (SQL only; engines are INJECTED; no env, no URL handling, no engine construction).

DarReader: SELECT only (read-only account). DarWriter: INSERT only, each in a savepoint (WoCorpusStore pattern).
No UPDATE/DELETE/TRUNCATE/DDL anywhere. persist() consumes the DarCorpus PERSIST PLAN.

Deviation noted: domain_corpus / domain_chunks inserts carry no RETURNING (the write-only account has no column
SELECT on those tables per the 007 draft); a duplicate is detected from the primary-key IntegrityError.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

SCHEMA = "vera_vjay"
FULLDOC_CHUNKS_KIND = "fulldoc_chunks"  # stored_text_kind of chunk-stored full documents (empty sources.content)


def _sql(sql: str, schema: str) -> str:
    return sql.replace("{S}", schema)


def _year(published, year):
    if isinstance(year, int):
        return year
    try:
        return int(str(published)[:4])
    except (TypeError, ValueError):
        return None


def _json_val(v):
    """Stored provenance JSON value as returned by the driver (parsed, or a JSON string); None when absent."""
    if isinstance(v, str):
        import json
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _lower_ts(ts, seconds: float):
    """ISO timestamp string (or datetime) minus `seconds`, as an ISO string; None when it is not timezone-aware."""
    from datetime import datetime, timedelta
    try:
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return (dt - timedelta(seconds=seconds)).isoformat()


def _partial(reason: str, since: dict) -> dict:
    """A bound was hit: EMPTY payload + partial flag + the prior cursor. The caller must rebuild, never apply."""
    return {"domains": [], "documents": [], "corpus_assoc": [], "chunk_assoc": [], "cursor": dict(since or {}),
            "partial": True, "reason": reason}


def _win(ts_clause: str, alias: str, key_col: str, by_id: bool) -> str:
    """Association time predicate; with an id window, rows of documents inside that window are read whatever their
    timestamp (so a document and its associations are never split by the two windows)."""
    if not ts_clause:
        return ""
    c = ts_clause.replace("{a}", alias)
    if not by_id:
        return c
    sub = ("EXISTS (SELECT 1 FROM {S}.sources w WHERE w.canonical_source_id = " + key_col  # nosec B608 - {S}=module SCHEMA constant; key_col is a literal; values bound
           + " AND w.source_id > :since_source_id_lo)")
    return " AND (" + c.replace(" AND ", "", 1) + " OR " + sub + ")"


class DarReader:
    """SELECT only."""

    def __init__(self, engine_ro, schema: str = SCHEMA):
        self.engine, self.schema = engine_ro, schema

    _conn = None  # set only inside _one_txn (instances are per call; not shared across threads)

    def _rows(self, sql: str, params: dict | None = None) -> list:
        from sqlalchemy import text
        if self._conn is not None:
            return list(self._conn.execute(text(_sql(sql, self.schema)), params or {}).fetchall())
        with self.engine.connect() as c:
            return list(c.execute(text(_sql(sql, self.schema)), params or {}).fetchall())

    @contextmanager
    def _one_txn(self):
        """ALL statements of one snapshot/delta read share ONE connection and transaction. On PostgreSQL it is
        REPEATABLE READ and READ ONLY (a single consistent snapshot: a domain committed between two queries cannot
        be seen by one and missed by the other). On any other dialect (SQLite) or when the options are refused, it
        degrades to the plain single connection (still one transaction for the reads, default isolation)."""
        if self._conn is not None:
            yield
            return
        with self.engine.connect() as c:
            if getattr(getattr(self.engine, "dialect", None), "name", None) == "postgresql":
                try:
                    c = c.execution_options(isolation_level="REPEATABLE READ", postgresql_readonly=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DAR read isolation not applied (%s)", type(exc).__name__)
            self._conn = c
            try:
                yield
            finally:
                self._conn = None

    def domain_id_by_norm(self, name_norm: str):
        rows = self._rows("SELECT domain_id FROM {S}.research_domains WHERE name_norm = :n", {"n": name_norm})
        return rows[0][0] if rows else None

    def span_id_for(self, source_id: int, start: int, end: int):
        rows = self._rows("SELECT span_id FROM {S}.evidence_spans WHERE source_id = :s AND start_index = :a "
                          "AND end_index = :b", {"s": source_id, "a": start, "b": end})
        return rows[0][0] if rows else None

    def canonical_id_for(self, canonical_url: str):
        rows = self._rows("SELECT canonical_source_id FROM {S}.canonical_sources WHERE canonical_url = :u",
                          {"u": canonical_url})
        return rows[0][0] if rows else None

    def source_id_for(self, canonical_source_id: int):
        """Latest source_id (highest version) of a canonical source."""
        rows = self._rows("SELECT source_id FROM {S}.sources WHERE canonical_source_id = :c "
                          "AND coalesce(provenance->>'stored_text_kind', '') <> :fd_kind "  # R5: an empty fulldoc_chunks row is never 'latest'
                          "ORDER BY version DESC LIMIT 1", {"c": canonical_source_id, "fd_kind": FULLDOC_CHUNKS_KIND})
        return rows[0][0] if rows else None

    def domains(self) -> list[dict]:
        rows = self._rows("SELECT domain_id, name, name_norm, description, origin FROM {S}.research_domains "
                          "ORDER BY domain_id")
        return [{"domain_id": r[0], "name": r[1], "name_norm": r[2], "description": r[3], "origin": r[4]}
                for r in rows]

    def corpus_assocs(self) -> list[dict]:
        rows = self._rows("SELECT domain_id, canonical_source_id, asserted_by FROM {S}.domain_corpus "
                          "ORDER BY domain_id, canonical_source_id")
        return [{"domain_id": r[0], "canonical_source_id": r[1], "asserted_by": r[2]} for r in rows]

    def chunk_assocs(self) -> list[dict]:
        rows = self._rows("SELECT domain_id, span_id, asserted_by FROM {S}.domain_chunks ORDER BY domain_id, span_id")
        return [{"domain_id": r[0], "span_id": r[1], "asserted_by": r[2]} for r in rows]

    def corpus_version(self) -> str:
        """Cheap change-detection token: one read-only query -> "max_span_id:chunk_count:max_source_id" plus, when the
        driver returns them, ":corpus_assoc_count:max_domain_id:n_sources:n_spans:n_domains" (the three counts change on EVERY insert, so a late-committing
        document with a LOWER id, which leaves every max unchanged, still changes the token) (so a late document-level label or a new domain also
        changes the token and triggers the incremental refresh)."""
        rows = self._rows("SELECT (SELECT coalesce(max(span_id), 0) FROM {S}.evidence_spans), "
                          "(SELECT count(*) FROM {S}.domain_chunks), "
                          "(SELECT coalesce(max(source_id), 0) FROM {S}.sources), "
                          "(SELECT count(*) FROM {S}.domain_corpus), "
                          "(SELECT coalesce(max(domain_id), 0) FROM {S}.research_domains), "
                          "(SELECT count(*) FROM {S}.sources), (SELECT count(*) FROM {S}.evidence_spans), "
                          "(SELECT count(*) FROM {S}.research_domains)")
        row = tuple(rows[0]) if rows else (0, 0, 0)
        return ":".join(str(x) for x in row)

    def load_into_registry(self, registry) -> int:
        """Optional: register stored domains (by name/description/origin) in an in-memory registry. Returns count."""
        n = 0
        for d in self.domains():
            registry.register(d["name"], d["description"], d["origin"])
            n += 1
        return n

    def load_snapshot(self, full_document_kinds, span_text=None, max_documents=None) -> dict:
        """Snapshot for DarCorpus.restore. Documents = stored full texts whose provenance stored_text_kind is set and
        IN full_document_kinds (fail-closed allow-list, bound parameter; empty = no documents). Placeholder spans (start 0 AND text = span_text) are excluded
        from chunk associations; span_text is a bound parameter. max_documents (optional, from config) bounds the
        number of full texts loaded (bound LIMIT parameter). When the corpus is larger the result is flagged
        {"partial": True, "reason": "snapshot_max_documents", "cursor": None} (the oldest canonical sources are kept).
        Additive key "cursor": the since-dict for load_delta, computed from what was read."""
        return self._collect(full_document_kinds, span_text, max_documents, None)

    def load_delta(self, since: dict | None, full_document_kinds=None, span_text=None, max_documents=None,
                   max_assocs=None, lookback_ids=0, lookback_s=0) -> dict:
        """Same shape as load_snapshot but ONLY rows newer than the cursor `since`
        ({"source_id","span_id","domain_id","assoc_ts"}): documents with source_id > since.source_id (full text),
        domains with domain_id > since.domain_id, corpus/chunk associations with created_at > since.assoc_ts (rows for
        OLD documents included: a later promotion/secondary label is a new row). Same stored_text_kind filter and
        placeholder-span predicate. "cursor" is the advanced since-dict. SELECT only, all values bound parameters.
        Freshness (D-053): lookback_ids / lookback_s LOWER the read thresholds (id > cursor - lookback_ids; created_at >
        cursor_ts - lookback_s) so rows that committed after the cursor passed them are re-read; the cursor itself stays
        monotone and apply_delta ignores rows it already has. Associations of every document inside the id window are
        read whatever their timestamp. Bound (D-053): max_documents counts documents NEWER than the cursor (rows re-read
        only because of the lookback do not count); max_assocs bounds each association query. When a bound is hit the
        result is {"partial": True, "reason": ..., "cursor": <unchanged>} with an EMPTY payload: never a cut_short
        delta presented as complete (the caller must rebuild)."""
        lim = {"max_documents": max_documents, "max_assocs": max_assocs, "lookback_ids": max(int(lookback_ids or 0), 0),
               "lookback_s": max(float(lookback_s or 0), 0.0)}
        return self._collect(full_document_kinds, span_text, None, dict(since or {}), lim)

    def _collect(self, full_document_kinds, span_text, max_documents, since, lim=None) -> dict:
        with self._one_txn():
            return self._collect_in_txn(full_document_kinds, span_text, max_documents, since, lim)

    def _collect_in_txn(self, full_document_kinds, span_text, max_documents, since, lim=None) -> dict:
        kinds = list(full_document_kinds or [])
        delta = since is not None
        since = since or {}
        lim = lim or {}
        lb_i, lb_s = int(lim.get("lookback_ids") or 0), float(lim.get("lookback_s") or 0)
        max_docs, max_as = lim.get("max_documents"), lim.get("max_assocs")
        sid0, did0 = since.get("source_id") or 0, since.get("domain_id") or 0
        ts0 = since.get("assoc_ts")
        sid_lo, did_lo = max(sid0 - lb_i, 0), max(did0 - lb_i, 0)  # lowered READ thresholds; the cursor is not lowered
        ts_lo = ts0
        if delta and lb_s > 0 and ts0 is not None:
            ts_lo = _lower_ts(ts0, lb_s)
            if ts_lo is None:  # naive/unparsable cursor timestamp: the window cannot be applied safely
                return _partial("cursor_ts_unparsable", since)
        if delta:
            drows = self._rows("SELECT domain_id, name, name_norm, description, origin FROM {S}.research_domains "
                               "WHERE domain_id > :since_domain_id ORDER BY domain_id", {"since_domain_id": did_lo})
            dom_list = [{"domain_id": r[0], "name": r[1], "name_norm": r[2], "description": r[3], "origin": r[4]}
                        for r in drows]
        else:
            dom_list = self.domains()
        domains = [{k: d[k] for k in ("name", "name_norm", "description", "origin")} for d in dom_list]
        doc_params = {"kinds": kinds}
        if max_documents is not None:
            doc_params["lim"] = int(max_documents) + 1  # one extra row only detects overflow
        elif delta and max_docs is not None:  # newer-than-cursor docs <= max_docs, plus at most lb_i re-read ones
            doc_params["lim"] = int(max_docs) + lb_i + 1
        if delta:
            doc_params["since_source_id"] = sid_lo
        doc_rows = self._rows(
            "SELECT DISTINCT ON (cs.canonical_source_id) cs.canonical_url, s.source_id, cs.canonical_source_id, "  # nosec B608 - {S}=module SCHEMA constant; literal fragments; values bound
            "s.content, s.provenance->>'url', s.provenance->>'doi', s.provenance->>'title', "
            "s.provenance->>'published', s.provenance->>'source_type', s.provenance->>'retrieved_at', "
            "s.provenance->>'stored_text_kind', s.provenance->'licence', s.provenance->'licence_decision' "
            "FROM {S}.canonical_sources cs JOIN {S}.sources s ON s.canonical_source_id = cs.canonical_source_id "
            "WHERE s.provenance->>'stored_text_kind' IS NOT NULL "
            "AND s.provenance->>'stored_text_kind' = ANY(:kinds) "  # fail-closed allow-list; empty list = no rows
            + ("AND s.source_id > :since_source_id " if delta else "")
            + "ORDER BY cs.canonical_source_id, s.version DESC"
            + (" LIMIT CAST(:lim AS integer)" if "lim" in doc_params else ""), doc_params)
        if delta and max_docs is not None and sum(1 for r in doc_rows if (r[1] or 0) > sid0) > int(max_docs):
            return _partial("delta_max_documents", since)
        cut_short = (not delta) and max_documents is not None and len(doc_rows) > int(max_documents)
        if cut_short:
            doc_rows = doc_rows[: int(max_documents)]
        documents = []
        for r in doc_rows:
            documents.append({"source_key": r[0], "url": r[4] or r[0], "doi": r[5], "title": r[6], "text": r[3],
                              "year": _year(r[7], None), "published": r[7], "source_type": r[8],
                              "retrieved_at": r[9], "stored_text_kind": r[10], "db_source_id": r[1],
                              "db_canonical_source_id": r[2],
                              "canonical_source_id": r[2], "source_id": r[1],
                              "licence": _json_val(r[11]) if len(r) > 11 else None,
                              "licence_decision": _json_val(r[12]) if len(r) > 12 else None,
                              "admission_passed": True})
        ts_clause = " AND {a}.created_at > CAST(:since_assoc_ts AS timestamptz)" if (delta and ts0 is not None) else ""
        ts_params = {"since_assoc_ts": ts_lo} if (delta and ts0 is not None) else {}
        by_id = delta and lb_i > 0  # also read ALL associations of documents inside the id window (see load_delta)
        if by_id:
            ts_params["since_source_id_lo"] = sid_lo
        a_lim = " LIMIT CAST(:lim_assoc AS integer)" if (delta and max_as is not None) else ""
        if a_lim:
            ts_params["lim_assoc"] = int(max_as) + 1
        corpus_rows = self._rows(
            "SELECT d.name_norm, cs.canonical_url, dc.asserted_by, dc.created_at FROM {S}.domain_corpus dc "  # nosec B608 - {S}=module SCHEMA constant; _win/a_lim literal fragments; values bound
            "JOIN {S}.research_domains d ON d.domain_id = dc.domain_id "
            "JOIN {S}.canonical_sources cs ON cs.canonical_source_id = dc.canonical_source_id "
            "WHERE TRUE" + _win(ts_clause, "dc", "dc.canonical_source_id", by_id)
            + " ORDER BY d.domain_id, cs.canonical_source_id" + a_lim, dict(ts_params))
        corpus = [{"name_norm": r[0], "source_key": r[1], "asserted_by": r[2]} for r in corpus_rows]
        chunk_rows = self._rows(
            "SELECT d.name_norm, cs.canonical_url, es.start_index, es.end_index, dk.asserted_by, "  # nosec B608 - {S}=module SCHEMA constant; _win literal fragments; values bound
            "dk.created_at, es.span_id "
            "FROM {S}.domain_chunks dk JOIN {S}.research_domains d ON d.domain_id = dk.domain_id "
            "JOIN {S}.evidence_spans es ON es.span_id = dk.span_id "
            "JOIN {S}.sources s ON s.source_id = es.source_id "
            "AND s.version = (SELECT max(v.version) FROM {S}.sources v "
            "WHERE v.canonical_source_id = s.canonical_source_id "
            "AND coalesce(v.provenance->>'stored_text_kind', '') <> :fd_kind) "  # R5
            "JOIN {S}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
            "WHERE (CAST(:span_text AS text) IS NULL OR NOT (es.start_index = 0 AND es.text = CAST(:span_text AS text)))"
            + _win(ts_clause, "dk", "s.canonical_source_id", by_id)
            + " ORDER BY d.domain_id, dk.span_id" + a_lim, {"span_text": span_text, "fd_kind": FULLDOC_CHUNKS_KIND, **ts_params})
        chunks = [{"name_norm": r[0], "source_key": r[1], "start": r[2], "end": r[3], "asserted_by": r[4]}
                  for r in chunk_rows]
        if a_lim and (len(corpus_rows) > int(max_as) or len(chunk_rows) > int(max_as)):
            return _partial("delta_max_assocs", since)

        if cut_short:  # associations of the documents that were NOT loaded would dangle: filter them out
            keep = {d["source_key"] for d in documents}
            corpus = [a for a in corpus if a["source_key"] in keep]
            chunks = [a for a in chunks if a["source_key"] in keep]

        def _mx(vals, prior):
            vals = [v for v in vals if v is not None] + ([prior] if prior is not None else [])
            if not vals:
                return prior
            try:
                return max(vals)
            except TypeError:
                return max(vals, key=str)

        ts_vals = [r[3] for r in corpus_rows if len(r) > 3] + [r[5] for r in chunk_rows if len(r) > 5]
        ts = _mx([v.isoformat() if hasattr(v, "isoformat") else str(v) for v in ts_vals if v is not None],
                 ts0 if isinstance(ts0, str) else (ts0.isoformat() if hasattr(ts0, "isoformat") else ts0))
        cursor = {"source_id": _mx([r[1] for r in doc_rows], since.get("source_id")),
                  "span_id": _mx([r[6] for r in chunk_rows if len(r) > 6], since.get("span_id")),
                  "domain_id": _mx([d["domain_id"] for d in dom_list], since.get("domain_id")),
                  "assoc_ts": ts}
        out = {"domains": domains, "documents": documents, "corpus_assoc": corpus, "chunk_assoc": chunks,
               "cursor": cursor}
        if cut_short:  # never present a cut_short corpus as complete; no cursor => no deltas, only the TTL rebuild
            out.update(partial=True, reason="snapshot_max_documents", cursor=None)
        return out


class DarWriter:
    """INSERT only, in savepoints."""

    def __init__(self, engine_wo, reader: DarReader | None = None, schema: str = SCHEMA, span_text=None):
        self.engine, self.reader, self.schema, self.span_text = engine_wo, reader, schema, span_text
        self._conn = None

    @contextmanager
    def atomic(self):
        """ONE transaction for every write inside the block: a single engine.begin() connection, each INSERT in a
        savepoint (a duplicate does not abort the transaction), commit once at the end, full rollback on any
        exception (re-raised). A writer without an engine (ConnWriter) is already bound to a caller transaction:
        passthrough. Re-entrant."""
        if self._conn is not None or self.engine is None:
            yield self
            return
        with self.engine.begin() as conn:
            self._conn = conn
            try:
                yield self
            finally:
                self._conn = None

    def _exec(self, conn, sql: str, params: dict):
        from sqlalchemy import text
        return conn.execute(text(_sql(sql, self.schema)), params)

    def _insert_new(self, sql: str, params: dict, returning: bool = True):
        """INSERT in a savepoint. Returns the RETURNING value (or True without RETURNING); None when a unique key
        says the row already exists."""
        if self._conn is not None:
            return self._insert_on(self._conn, sql, params, returning)
        with self.engine.begin() as conn:
            return self._insert_on(conn, sql, params, returning)

    def _insert_on(self, conn, sql: str, params: dict, returning: bool):
        from sqlalchemy.exc import IntegrityError
        sp = conn.begin_nested()
        try:
            res = self._exec(conn, sql, params)
            value = (res.first()[0] if returning else True)
            sp.commit()
            return value
        except IntegrityError:
            sp.rollback()
            return None

    def write_domain(self, name, name_norm, description, origin):
        return self._insert_new("INSERT INTO {S}.research_domains (name, name_norm, description, origin) "
                                "VALUES (:n,:nn,:d,:o) RETURNING domain_id",
                                {"n": name, "nn": name_norm, "d": description or "", "o": origin})

    def write_span(self, source_id, text, start, end):
        v = self._insert_new("INSERT INTO {S}.evidence_spans (source_id, text, start_index, end_index) "
                             "VALUES (:s,:t,:a,:b) RETURNING span_id",
                             {"s": source_id, "t": text, "a": start, "b": end})
        return v

    def write_corpus_assoc(self, domain_id, canonical_source_id, asserted_by) -> bool:
        return bool(self._insert_new("INSERT INTO {S}.domain_corpus (domain_id, canonical_source_id, asserted_by) "
                                     "VALUES (:d,:c,:a)",
                                     {"d": domain_id, "c": canonical_source_id, "a": asserted_by}, returning=False))

    def write_chunk_assoc(self, domain_id, span_id, asserted_by) -> bool:
        return bool(self._insert_new("INSERT INTO {S}.domain_chunks (domain_id, span_id, asserted_by) "
                                     "VALUES (:d,:s,:a)",
                                     {"d": domain_id, "s": span_id, "a": asserted_by}, returning=False))


_PLACEHOLDER = object()


def _is_placeholder(chunk: dict, span_text) -> bool:
    return span_text is not None and chunk.get("start") == 0 and chunk.get("text") == span_text


def persist(plan: dict, writer: DarWriter, span_text=None, ctx: dict | None = None, chunk_text=None) -> dict:
    """Persist a DarCorpus plan. Never raises on duplicates. ctx (caller-held) receives
    {"domains": {mem_id: db_id}, "canonical": {source_key: db_canonical_id}, "spans": {mem_span_id: db_span_id}}.
    chunk_text (optional callable span_id -> str|None) supplies text for promoted spans that have no evidence_spans
    row yet and whose chunk_assoc entry carries no "text".
    ATOMIC (section 3): when the writer owns its engine (DarWriter.atomic) ALL writes of the plan run in ONE
    transaction, committed once; on any exception the whole transaction is rolled back and the result is
    {"status": "failed", "reason": <ExceptionClassName>, "rolled_back": True, counts all zero} (never raises; class
    name only is logged); ctx is updated only after the commit. A writer without atomic() / without an engine (the
    loader's ConnWriter, already inside the loader's transaction) behaves as before, exceptions propagate."""
    if ctx is None:
        ctx = {}
    for k in ("domains", "canonical", "spans"):
        ctx.setdefault(k, {})
    if plan.get("status") != "indexed" or not _has_db_ids(plan):
        return _persist_core(plan, writer, span_text, ctx, chunk_text)
    if not (callable(getattr(writer, "atomic", None)) and getattr(writer, "engine", None) is not None):
        return _persist_core(plan, writer, span_text, ctx, chunk_text)
    local = {k: dict(v) for k, v in ctx.items()}
    try:
        with writer.atomic():
            result = _persist_core(plan, writer, span_text, local, chunk_text)
    except Exception as exc:  # noqa: BLE001  (transaction rolled back by atomic(); class name only)
        logger.warning("DAR persist rolled back (%s)", type(exc).__name__)
        zero = {k: 0 for k in _COUNT_KEYS}
        return {"status": "failed", "reason": type(exc).__name__, "rolled_back": True, **zero}
    for k, v in local.items():
        ctx.setdefault(k, {}).update(v)
    return result


_COUNT_KEYS = ("domains_written", "domains_existing", "spans_written", "spans_existing", "corpus_assoc_written",
               "chunk_assoc_written", "skipped_placeholder", "unresolved", "promoted_skipped", "missing_span_text")


def _has_db_ids(plan: dict) -> bool:
    doc = plan.get("doc") or {}
    sid, csid = doc.get("db_source_id"), doc.get("db_canonical_source_id")
    return isinstance(sid, int) and isinstance(csid, int) and not isinstance(sid, bool) and not isinstance(csid, bool)


def _persist_core(plan: dict, writer: DarWriter, span_text, ctx: dict, chunk_text) -> dict:
    counts = {"domains_written": 0, "domains_existing": 0, "spans_written": 0, "spans_existing": 0,
              "corpus_assoc_written": 0, "chunk_assoc_written": 0, "skipped_placeholder": 0,
              "unresolved": 0, "promoted_skipped": 0, "missing_span_text": 0}
    if plan.get("status") != "indexed":
        return {"status": "skipped", "reason": plan.get("reason") or "not_indexed", **counts}
    doc = plan.get("doc") or {}
    sid, csid = doc.get("db_source_id"), doc.get("db_canonical_source_id")
    if not _has_db_ids(plan):
        return {"status": "skipped", "reason": "no_db_ids", **counts}
    reader = writer.reader
    key = plan["source_key"]
    ctx["canonical"][key] = csid
    for d in plan.get("domains", []):
        dbid = writer.write_domain(d["name"], d["name_norm"], d.get("description", ""), d["origin"])
        if dbid is None:
            counts["domains_existing"] += 1
            dbid = reader.domain_id_by_norm(d["name_norm"]) if reader else None
        else:
            counts["domains_written"] += 1
        if dbid is None:
            counts["unresolved"] += 1
        else:
            ctx["domains"][d["domain_id"]] = dbid
    for a in plan.get("corpus_assoc", []):
        dbd = ctx["domains"].get(a["domain_id"])
        if dbd is None:
            counts["unresolved"] += 1
        elif writer.write_corpus_assoc(dbd, csid, a["asserted_by"]):
            counts["corpus_assoc_written"] += 1
    for ch in plan.get("chunks", []):
        if _is_placeholder(ch, span_text):
            counts["skipped_placeholder"] += 1
            continue
        spid = writer.write_span(sid, ch["text"], ch["start"], ch["end"])
        if spid is None:
            counts["spans_existing"] += 1
            spid = reader.span_id_for(sid, ch["start"], ch["end"]) if reader else None
        else:
            counts["spans_written"] += 1
        if spid is None:
            counts["unresolved"] += 1
            continue
        ctx["spans"][ch["span_id"]] = spid
        for mem in ch.get("domains", []):
            dbd = ctx["domains"].get(mem)
            if dbd is None:
                counts["unresolved"] += 1
            elif writer.write_chunk_assoc(dbd, spid, plan_asserted(plan, mem)):
                counts["chunk_assoc_written"] += 1
    plan_norms = {d["domain_id"]: d["name_norm"] for d in plan.get("domains", [])}

    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception:  # loud via counts, never raise
            return None

    def _domain(a, p):
        dbd = ctx["domains"].get(a["domain_id"])
        if dbd is None and reader is not None:
            norm = a.get("name_norm") or (p.get("domain_norms") or {}).get(a["domain_id"]) \
                or plan_norms.get(a["domain_id"])
            if norm:
                dbd = _safe(reader.domain_id_by_norm, norm)
                if dbd is not None:
                    ctx["domains"][a["domain_id"]] = dbd
        return dbd

    def _canonical(key):
        pcs = ctx["canonical"].get(key)
        if pcs is None and reader is not None and key is not None:
            pcs = _safe(reader.canonical_id_for, key)
            if pcs is not None:
                ctx["canonical"][key] = pcs
        return pcs

    def _plan_source_id(p):
        v = p.get("db_source_id")
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if p.get("source_key") == key:
            return sid
        return None

    def _span(span_id, p, a=None):
        dsp = ctx["spans"].get(span_id)
        if dsp is not None:
            return dsp
        try:
            skey, _, rng = str(span_id).rpartition(":")
            st, _, en = rng.partition("-")
            start, end = int(st), int(en)
        except ValueError:
            return None
        psid = _plan_source_id(p)
        if psid is None and reader is not None:
            pcs = _canonical(p.get("source_key") if p.get("source_key") is not None else skey)
            psid = _safe(reader.source_id_for, pcs) if pcs is not None else None
        if psid is None:
            return None
        dsp = _safe(reader.span_id_for, psid, start, end) if reader is not None else None
        if dsp is None:
            text = (a or {}).get("text")
            if text is None and chunk_text is not None:
                text = _safe(chunk_text, span_id)
            if text is None:
                counts["missing_span_text"] += 1
                return None
            if _is_placeholder({"start": start, "text": text}, span_text):
                counts["skipped_placeholder"] += 1
                counts["promoted_skipped"] += 1
                return _PLACEHOLDER
            dsp = writer.write_span(psid, text, start, end)
            if dsp is None:
                counts["spans_existing"] += 1
                dsp = _safe(reader.span_id_for, psid, start, end) if reader is not None else None
            else:
                counts["spans_written"] += 1
        if dsp is not None:
            ctx["spans"][span_id] = dsp
        return dsp

    for p in plan.get("promoted", []) or []:
        pcs = _canonical(p.get("source_key"))
        for a in p.get("corpus_assoc", []):
            dbd = _domain(a, p)
            if pcs is None or dbd is None:
                counts["promoted_skipped"] += 1
                counts["unresolved"] += 1
            elif writer.write_corpus_assoc(dbd, pcs, a["asserted_by"]):
                counts["corpus_assoc_written"] += 1
        for a in p.get("chunk_assoc", []):
            dbd, dsp = _domain(a, p), _span(a["span_id"], p, a)
            if dsp is _PLACEHOLDER:
                continue
            if dbd is None or dsp is None:
                counts["promoted_skipped"] += 1
                counts["unresolved"] += 1
            elif writer.write_chunk_assoc(dbd, dsp, a["asserted_by"]):
                counts["chunk_assoc_written"] += 1
    if counts["unresolved"]:
        extra = (" missing_span_text=%d: no evidence_spans row and no chunk text supplied" % counts["missing_span_text"]
                 if counts["missing_span_text"] else "")
        return {"status": "partial", "reason": "unresolved=%d (of which promoted_skipped=%d): ids could not be "
                "resolved from ctx or reader.%s" % (counts["unresolved"], counts["promoted_skipped"], extra), **counts}
    return {"status": "persisted", "reason": None, **counts}


def plan_asserted(plan: dict, mem_domain_id) -> str:
    """asserted_by for a chunk association: the doc-level assertion for that domain when present, else the first
    corpus_assoc's, else 'dar'."""
    for a in plan.get("corpus_assoc", []):
        if a["domain_id"] == mem_domain_id:
            return a["asserted_by"]
    for a in plan.get("corpus_assoc", []):
        return a["asserted_by"]
    return "dar"


def smoke_roundtrip(engine_wo, engine_ro, plan, span_text=None) -> dict:
    """persist() through the write-only engine, then read back through the read-only engine."""
    reader = DarReader(engine_ro)
    ctx: dict = {}
    result = persist(plan, DarWriter(engine_wo, reader=reader, span_text=span_text), span_text=span_text, ctx=ctx)
    return {"persist": result, "ctx": ctx, "domains": reader.domains(), "corpus_assocs": reader.corpus_assocs(),
            "chunk_assocs": reader.chunk_assocs()}
