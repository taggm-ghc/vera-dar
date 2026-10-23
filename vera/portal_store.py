"""PORT-P2: discovery-event and attribution store (interface + SQLite implementation for tests and dev).

Production Postgres tables are drafted in db/migrations/pending/009-DRAFT-portal-discovery.sql.draft (NOT applied;
owner-gated). Until the owner applies it, the recorder runs only against this SQLite store (config store.enabled).

Data model (mirrors the draft migration):
  discovery_event      one row per (request, portal) search attempt: status, n_results, latency, cost, search time
  document_identity    one row per document identity (DOI > PMCID > OpenAlex > arXiv > normalized URL)
  candidate_attribution  append-only: event -> document (many portals may credit one document; full history kept)
  document_outcome     append-only outcome events per (request, document) under a versioned definition set; readers
                       use the latest event per (request, document) (document_outcome_latest)
INSERT-only throughout (the production writer account has no UPDATE): no upsert; identity rows use insert-if-absent
(SQLite INSERT OR IGNORE = Postgres ON CONFLICT DO NOTHING, both plain inserts).
No document content is stored; source/span rows are referenced by id only (source_id, canonical_source_id).
Search time (discovery_event.searched_at), acquisition time (document_outcome.acquired_at) and source freshness
(document_outcome.source_date) are separate columns.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from typing import Mapping, Protocol

EVENT_STATUSES = ("completed", "timeout", "quota", "error", "skipped", "not_attempted")
_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#\"'<>]+)", re.I)


def canonical_ref(url) -> str | None:
    """Stable short reference to an acquired document's canonical URL (sha256 prefix), recorded ADDITIVELY next to the
    candidate's URL and ids (URLPUBLIC-1023, PMCIDPUBLIC-1023: those stay public); the recorder matches it to sources."""
    import hashlib
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:24] if url else None


def document_key(ids: Mapping) -> str | None:
    """Document identity key, by precedence DOI > PMCID > OpenAlex > arXiv > URL (a doi.org URL yields its DOI)."""
    doi = str(ids.get("doi") or "").strip()
    url = str(ids.get("url") or ids.get("canonical_url") or "").strip()
    if not doi and url:
        m = _DOI.search(url) if "doi.org/" in url.lower() else None
        doi = m.group(1) if m else ""
    if doi:
        return "doi:" + doi.lower().removeprefix("https://doi.org/").rstrip(".")
    if ids.get("pmcid"):
        return "pmcid:" + str(ids["pmcid"]).upper().removeprefix("PMC")
    if ids.get("openalex_id"):
        return "openalex:" + str(ids["openalex_id"]).rsplit("/", 1)[-1].upper()
    if ids.get("arxiv_id"):
        return "arxiv:" + re.sub(r"v\d+$", "", str(ids["arxiv_id"]).lower())
    if url:
        from vera.portal_registry import normalize_url
        return "url:" + normalize_url(url)
    return None


class DiscoveryStore(Protocol):
    """INSERT-only contract (the production wo account has INSERT, no UPDATE/DELETE): every method appends; a corrected
    outcome is a new add_outcome row and readers use the latest per (request, document)."""

    def add_event(self, *, request_id: str, portal_key: str, status: str, n_results: int = 0, query_hash: str = "",
                  domain: str | None = None, latency_ms: int | None = None, cost_usd: float | None = None,
                  cost_basis: str | None = None, searched_at: float | None = None) -> int: ...

    def add_attribution(self, *, event_id: int, ids: Mapping, rank: int | None = None) -> str | None: ...

    def add_outcome(self, *, request_id: str, doc_key: str, definitions_version: int, **stages) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS discovery_event (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, portal_key TEXT NOT NULL,
  query_hash TEXT NOT NULL DEFAULT '', domain_name_norm TEXT, searched_at REAL NOT NULL, latency_ms INTEGER,
  cost_usd REAL, cost_basis TEXT, n_results INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('completed','timeout','quota','error','skipped','not_attempted')));
CREATE TABLE IF NOT EXISTS document_identity (
  doc_key TEXT PRIMARY KEY, doi TEXT, pmcid TEXT, openalex_id TEXT, arxiv_id TEXT, url TEXT, first_seen_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS candidate_attribution (
  attribution_id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL REFERENCES discovery_event(event_id),
  doc_key TEXT NOT NULL REFERENCES document_identity(doc_key), rank INTEGER, recorded_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS document_outcome (
  request_id TEXT NOT NULL, doc_key TEXT NOT NULL REFERENCES document_identity(doc_key),
  definitions_version INTEGER NOT NULL, relevance_qualified INTEGER, valuable INTEGER,
  acq_attempted INTEGER, acq_succeeded INTEGER, admitted_chunks INTEGER, cited INTEGER,
  freshness_qualified INTEGER, contradictory INTEGER, terminal_reason TEXT,
  source_id INTEGER, canonical_source_id INTEGER, acquired_at TEXT, source_date TEXT, recorded_at REAL NOT NULL,
  outcome_id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE INDEX IF NOT EXISTS document_outcome_req_doc_idx ON document_outcome (request_id, doc_key, outcome_id);
CREATE VIEW IF NOT EXISTS document_outcome_latest AS
  SELECT o.* FROM document_outcome o WHERE o.outcome_id = (SELECT MAX(x.outcome_id) FROM document_outcome x
                                                          WHERE x.request_id = o.request_id AND x.doc_key = o.doc_key);
"""
_STAGES = ("relevance_qualified", "valuable", "acq_attempted", "acq_succeeded", "admitted_chunks", "cited",
           "freshness_qualified", "contradictory", "terminal_reason", "source_id", "canonical_source_id",
           "acquired_at", "source_date")


class SQLiteDiscoveryStore:
    """Thread-safe SQLite store (":memory:" for tests; a file under work/ for dev). Append-only (INSERT only), matching
    the Postgres writer's privileges; aggregation reads the latest outcome per (request, document)."""

    def __init__(self, path: str = ":memory:", clock=time.time, busy_timeout_s: float = 0.2):
        # short busy timeout: a locked database fails fast (the recorder swallows it) instead of the 5 s default
        self._db = sqlite3.connect(path, timeout=float(busy_timeout_s), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._clock = clock

    def _x(self, sql: str, args=()):
        with self._lock:
            return self._db.execute(sql, args)

    def add_event(self, *, request_id, portal_key, status, n_results=0, query_hash="", domain=None, latency_ms=None,
                  cost_usd=None, cost_basis=None, searched_at=None) -> int:
        if status not in EVENT_STATUSES:
            status = "error"
        cur = self._x("INSERT INTO discovery_event (request_id, portal_key, query_hash, domain_name_norm, searched_at,"
                      " latency_ms, cost_usd, cost_basis, n_results, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (request_id, portal_key, query_hash, domain, searched_at or self._clock(), latency_ms, cost_usd,
                       cost_basis, int(n_results or 0), status))
        return int(cur.lastrowid)

    def add_attribution(self, *, event_id, ids, rank=None) -> str | None:
        key = document_key(ids)
        if key is None:
            return None
        now = self._clock()
        self._x("INSERT OR IGNORE INTO document_identity (doc_key, doi, pmcid, openalex_id, arxiv_id, url, first_seen_at)"
                " VALUES (?,?,?,?,?,?,?)", (key, ids.get("doi"), ids.get("pmcid"), ids.get("openalex_id"),
                                            ids.get("arxiv_id"), ids.get("url"), now))
        self._x("INSERT INTO candidate_attribution (event_id, doc_key, rank, recorded_at) VALUES (?,?,?,?)",
                (event_id, key, rank, now))
        return key

    def add_outcome(self, *, request_id, doc_key, definitions_version, **stages) -> None:
        vals = [stages.get(k) for k in _STAGES]
        vals = [int(v) if isinstance(v, bool) else v for v in vals]
        self._x(f"INSERT INTO document_outcome (request_id, doc_key, definitions_version, "
                f"{', '.join(_STAGES)}, recorded_at) VALUES ({', '.join('?' * (len(_STAGES) + 4))})",
                (request_id, doc_key, int(definitions_version), *vals, self._clock()))

    # ------------------------------------------------------------------------------------- reads / aggregation
    def rows(self, sql: str, args=()) -> list:
        cur = self._x(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def attribution_history(self, doc_key: str) -> list:
        return self.rows("SELECT e.request_id, e.portal_key, e.searched_at, a.rank FROM candidate_attribution a "
                         "JOIN discovery_event e USING (event_id) WHERE a.doc_key=? ORDER BY a.attribution_id",
                         (doc_key,))

    def health(self) -> dict:
        """portal_key -> {last_attempt_at, last_success_at, failures_since_success} (skipped/not_attempted excluded)."""
        out = {}
        for r in self.rows("SELECT portal_key, status, searched_at FROM discovery_event "
                           "WHERE status NOT IN ('skipped','not_attempted') ORDER BY searched_at, event_id"):
            h = out.setdefault(r["portal_key"], {"last_attempt_at": None, "last_success_at": None,
                                                 "failures_since_success": 0})
            h["last_attempt_at"] = r["searched_at"]
            if r["status"] == "completed":
                h["last_success_at"], h["failures_since_success"] = r["searched_at"], 0
            else:
                h["failures_since_success"] += 1
        return out

    def portal_tallies(self, domain: str | None = None) -> dict:
        """Per portal: events by status, candidates returned, credited (distinct documents this portal discovered),
        and per-stage counts over the documents it was credited with (each document counted once per portal).
        Optional domain filter on the event's domain."""
        where, args = ("WHERE e.domain_name_norm = ?", (domain,)) if domain else ("", ())
        out: dict = {}
        for r in self.rows(f"SELECT portal_key, status, COUNT(*) n, SUM(n_results) res, SUM(latency_ms) lat, "
                           f"COUNT(latency_ms) nlat, SUM(cost_usd) cost FROM discovery_event e {where} "
                           f"GROUP BY portal_key, status", args):
            t = out.setdefault(r["portal_key"], _empty_tally())
            t["events"][r["status"]] = r["n"]
            t["candidates_returned"] += int(r["res"] or 0)
            t["latency_ms_sum"] += int(r["lat"] or 0)
            t["latency_n"] += int(r["nlat"] or 0)
            t["cost_usd"] += float(r["cost"] or 0.0)
        for r in self.rows(_STAGE_SQL.format(where=where), args):
            t = out.setdefault(r["portal_key"], _empty_tally())
            t.update({k: int(r[k] or 0) for k in _COUNTED})
        return out

    def unique_totals(self, domain: str | None = None) -> dict:
        """Totals with every document counted ONCE however many portals discovered it."""
        where, args = ("WHERE e.domain_name_norm = ?", (domain,)) if domain else ("", ())
        r = self.rows(_STAGE_SQL_UNIQUE.format(where=where), args)[0]
        return {k: int(r[k] or 0) for k in ("documents", *_COUNTED[1:])}

    def domain_tallies(self) -> dict:
        doms = [r["d"] for r in self.rows("SELECT DISTINCT domain_name_norm d FROM discovery_event "
                                          "WHERE domain_name_norm IS NOT NULL")]
        return {d: {"by_portal": self.portal_tallies(d), "unique": self.unique_totals(d)} for d in doms}


_COUNTED = ("deduplicated_documents", "relevance_qualified", "valuable", "acq_attempted", "acq_succeeded",
            "admitted_chunks", "cited", "freshness_qualified", "contradictory", "rejected", "failed")
# per (portal, document): a document credited to a portal in a request joins that request's outcome row
_PAIRS = ("SELECT DISTINCT e.portal_key, a.doc_key, e.request_id FROM candidate_attribution a "
          "JOIN discovery_event e USING (event_id) {where}")
_AGG = ("COUNT(DISTINCT p.doc_key) {first}, "
        "COUNT(DISTINCT CASE WHEN o.relevance_qualified=1 THEN p.doc_key END) relevance_qualified, "
        "COUNT(DISTINCT CASE WHEN o.valuable=1 THEN p.doc_key END) valuable, "
        "COUNT(DISTINCT CASE WHEN o.acq_attempted=1 THEN p.doc_key END) acq_attempted, "
        "COUNT(DISTINCT CASE WHEN o.acq_succeeded=1 THEN p.doc_key END) acq_succeeded, "
        "SUM(o.admitted_chunks) admitted_chunks, "
        "COUNT(DISTINCT CASE WHEN o.cited=1 THEN p.doc_key END) cited, "
        "COUNT(DISTINCT CASE WHEN o.freshness_qualified=1 THEN p.doc_key END) freshness_qualified, "
        "COUNT(DISTINCT CASE WHEN o.contradictory=1 THEN p.doc_key END) contradictory, "
        "COUNT(DISTINCT CASE WHEN o.terminal_reason LIKE '%_rejected' THEN p.doc_key END) rejected, "
        "COUNT(DISTINCT CASE WHEN o.terminal_reason IN ('no_route','fetch_failed','extraction_failed','abstract_only',"
        "'budget_exhausted') THEN p.doc_key END) failed")
_STAGE_SQL = ("SELECT p.portal_key, " + _AGG.format(first="deduplicated_documents") + " FROM (" + _PAIRS +
              ") p LEFT JOIN document_outcome_latest o ON o.request_id=p.request_id AND o.doc_key=p.doc_key "
              "GROUP BY p.portal_key")
# unique totals: collapse portals first, so a document found by several portals in one request counts once
_STAGE_SQL_UNIQUE = ("SELECT " + _AGG.format(first="documents") + " FROM (SELECT DISTINCT doc_key, request_id FROM (" +
                     _PAIRS + ")) p LEFT JOIN document_outcome_latest o ON o.request_id=p.request_id AND o.doc_key=p.doc_key")


def _empty_tally() -> dict:
    return {"events": {}, "candidates_returned": 0, "latency_ms_sum": 0, "latency_n": 0, "cost_usd": 0.0,
            **{k: 0 for k in _COUNTED}}


def rates(t: Mapping) -> dict:
    """Per-portal rates used by selection: attempts (non-skipped events), success_rate, relevant_rate and
    valuable_rate over credited documents (0 when no history)."""
    ev = t.get("events") or {}
    attempts = sum(v for k, v in ev.items() if k not in ("skipped", "not_attempted"))
    docs = int(t.get("deduplicated_documents") or 0)
    return {"attempts": attempts,
            "success_rate": (ev.get("completed", 0) / attempts) if attempts else 0.0,
            "relevant_rate": (t.get("relevance_qualified", 0) / docs) if docs else 0.0,
            "valuable_rate": (t.get("valuable", 0) / docs) if docs else 0.0}


def open_store(cfg: Mapping | None) -> SQLiteDiscoveryStore | None:
    """The configured store, or None (disabled, unknown backend, or no path): the recorder is then a no-op.
    Postgres is deliberately not wired: its tables need the owner-approved 009 migration."""
    s = (cfg or {}).get("store") or {}
    if s.get("enabled") is not True or s.get("backend") != "sqlite" or not s.get("sqlite_path"):
        return None
    return SQLiteDiscoveryStore(str(s["sqlite_path"]), busy_timeout_s=float(s.get("busy_timeout_s", 0.2)))

