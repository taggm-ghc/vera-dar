"""Item #84 Path A: durable memory storage and recall (existing tables only, no DDL, no DELETE).

Stores only claims that passed the claim check and the write gate (vera/memory_gate.py) and whose cited source is
ALREADY in the corpus (else skipped as `source_not_in_corpus`). Privacy (#74): the question, any identity, session
id or preference is never written; the question is only a bound read parameter (and gate input) in memory, and
recall is the same for everyone. Writes use the write-only account (URL in the env var named by config
memory.store.db_url_wo_env), reads the read-only account (memory.store.db_url_ro_env); never the rw account.
Forgetting: TTL and recency decay at read time on claims.created_at; tombstone via claims.detailed_status. There is
no DELETE path here (deletes need R1 approval). Every error is logged host-free and never fails /ask.

D6 (2026-10-10, retires D-029): with confirm_required (default true), /ask writes rows as status_pending; recall reads
only status_live, which only the operator sets (confirm). Legacy auto-written rows (status_legacy_auto) are not
recalled and are listed for review. Operator actions: ro reads the row's state, then wo updates by claim_id.
"""
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

from vera import answer_contract
from vera.corpus_admission import SCHEMA, WoCorpusStore, canonical_url
from vera.memory_gate import (ECHOES_QUESTION, GateResult, SourceRef, _overlap, check_memory_write,
                              normalise_claim)

logger = logging.getLogger("vera")

STAGE_FINAL = "final"
VERIFICATION_SUPPORTED = "supported"
VERIFICATION_UNVERIFIED = "unverified"
SKIP_NOT_IN_CORPUS = "source_not_in_corpus"
SKIP_DUPLICATE = "duplicate"
SKIP_CAP = "cap_reached"
SKIP_DISABLED = "memory_disabled"
SKIP_QUEUE_FULL = "pending_queue_full"
# Operator action outcomes (D6).
CONFIRMED, REJECTED, NOT_FOUND, NOT_ALLOWED = "confirmed", "rejected", "not_found", "transition_not_allowed"
STATE_PENDING, STATE_LEGACY, STATE_LIVE, STATE_REJECTED = "pending", "legacy_auto", "confirmed", "rejected"


def confirm_required(cfg: dict) -> bool:
    """Deny-by-default: only an explicit false turns confirmation off."""
    return cfg.get("confirm_required", True) is not False


def _known_statuses(cfg: dict) -> list:
    return [s for s in (cfg["status_live"], cfg.get("status_pending"), cfg.get("status_rejected"),
                        cfg.get("status_tombstone"), *cfg.get("status_legacy_auto", ())) if s]


def _state_of(status: str, cfg: dict) -> str | None:
    if status == cfg.get("status_pending"):
        return STATE_PENDING
    if status in cfg.get("status_legacy_auto", ()):
        return STATE_LEGACY
    if status == cfg["status_live"]:
        return STATE_LIVE
    if status in (cfg.get("status_rejected"), cfg.get("status_tombstone")):
        return STATE_REJECTED
    return None
_WORD = re.compile(r"[A-Za-z0-9]{3,}")


FULLDOC_CHUNKS_KIND = "fulldoc_chunks"  # stored_text_kind of chunk-stored full documents (empty sources.content)


def _sql(sql: str) -> str:
    return sql.replace("{S}", SCHEMA)


class RoMemoryReader:
    """Read-only lookups: recall, dedupe, source and answer resolution. hide_parameters keeps bound values
    (including the question) out of SQLAlchemy error text."""

    def __init__(self, engine, cfg: dict):
        self.engine, self.cfg = engine, cfg

    def _rows(self, sql: str, params: dict):
        from sqlalchemy import text
        with self.engine.connect() as c:
            return c.execute(text(_sql(sql)), params).fetchall()

    def query_terms(self, question: str) -> str:
        stop = {w.lower() for w in self.cfg["query_stopwords"]}
        short = int(self.cfg["min_query_word_len"])
        words = list(dict.fromkeys(w for w in (x.lower() for x in _WORD.findall(question or ""))
                                   if len(w) >= short and w not in stop))
        return " or ".join(words[: int(self.cfg["max_query_terms"])])

    def recall(self, question: str) -> list[dict]:
        q = self.query_terms(question)
        if not q:
            return []
        terms = q.split(" or ")
        rows = self._rows(
            "SELECT claim_text, created_at, title, url FROM ("
            "SELECT DISTINCT ON (c.claim_id) c.claim_text, c.created_at, s.provenance ->> 'title' AS title, "
            "COALESCE(s.provenance ->> 'url', cs.canonical_url) AS url, "
            "ts_rank(to_tsvector('english', c.claim_text), websearch_to_tsquery('english', :q)) "
            "* power(0.5, extract(epoch FROM now() - c.created_at) / 86400.0 / :hl) AS score "
            "FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
            "JOIN LATERAL jsonb_array_elements_text(c.evidence_span_ids) AS sp(span_id) ON true "
            "JOIN {S}.evidence_spans e ON e.span_id = sp.span_id::bigint "
            "JOIN {S}.sources s ON s.source_id = e.source_id "
            "JOIN {S}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
            "WHERE a.response_text = :qt AND c.detailed_status = :live "
            "AND c.created_at > now() - make_interval(days => :ttl) "
            "AND to_tsvector('english', c.claim_text) @@ websearch_to_tsquery('english', :q) "
            "AND ts_rank(to_tsvector('english', c.claim_text), websearch_to_tsquery('english', :q)) >= :floor "
            "AND (SELECT count(*) FROM unnest(CAST(:terms AS text[])) AS tm(w) "
            "WHERE to_tsvector('english', c.claim_text) @@ plainto_tsquery('english', tm.w)) >= :minterms "
            "ORDER BY c.claim_id) t ORDER BY score DESC LIMIT :k",
            {"qt": self.cfg["question_text"], "live": self.cfg["status_live"], "ttl": int(self.cfg["ttl_days"]),
             "q": q, "terms": terms, "minterms": int(self.cfg["min_matched_terms"]), "floor": float(self.cfg["min_rank"]), "hl": float(self.cfg["half_life_days"]), "k": int(self.cfg["recall_k"])})
        return [{"claim_text": r[0], "source_title": r[2] or "", "source_url": r[3] or "",
                 "saved_at": r[1].isoformat() if r[1] else ""} for r in rows]

    def recall_discovery(self, question: str) -> list[dict]:
        """K2.I4: live discovery notes (no evidence spans, unverified, note prefix) ranked like `recall`. No JOIN to
        spans or sources; no LIKE (prefix compared by left())."""
        from vera.discovery_memory import safe_url
        prefix = answer_contract.text("note_prefix")
        q = self.query_terms(question)
        if not q or not prefix:
            return []
        terms = q.split(" or ")
        rows = self._rows(
            "SELECT claim_text, created_at FROM ("
            "SELECT c.claim_id, c.claim_text, c.created_at, "
            "ts_rank(to_tsvector('english', c.claim_text), websearch_to_tsquery('english', :q)) "
            "* power(0.5, extract(epoch FROM now() - c.created_at) / 86400.0 / :hl) AS score "
            "FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
            "WHERE a.response_text = :qt AND c.detailed_status = :live "
            "AND jsonb_array_length(c.evidence_span_ids) = 0 AND c.verification_status = :unv "
            "AND left(c.claim_text, length(CAST(:prefix AS text))) = CAST(:prefix AS text) "
            "AND c.created_at > now() - make_interval(days => :ttl) "
            "AND to_tsvector('english', c.claim_text) @@ websearch_to_tsquery('english', :q) "
            "AND ts_rank(to_tsvector('english', c.claim_text), websearch_to_tsquery('english', :q)) >= :floor "
            "AND (SELECT count(*) FROM unnest(CAST(:terms AS text[])) AS tm(w) "
            "WHERE to_tsvector('english', c.claim_text) @@ plainto_tsquery('english', tm.w)) >= :minterms "
            ") t ORDER BY score DESC, claim_id LIMIT :k",
            {"qt": self.cfg["question_text"], "live": self.cfg["status_live"], "unv": VERIFICATION_UNVERIFIED,
             "prefix": prefix, "ttl": int(self.cfg["ttl_days"]), "q": q, "terms": terms,
             "minterms": int(self.cfg["min_matched_terms"]), "floor": float(self.cfg["min_rank"]),
             "hl": float(self.cfg["half_life_days"]), "k": int(self.cfg["recall_k"])})
        out = []
        for r in rows:
            first, _, rest = str(r[0] or "").partition("\n")
            out.append({"kind": "discovery_note", "note": first, "urls": [u for u in rest.split(" ") if safe_url(u)],
                        "saved_at": r[1].isoformat() if r[1] else ""})
        return out

    def source(self, canonical: str):
        """(source_id, content length) of an already-admitted source, or None."""
        rows = self._rows(
            "SELECT s.source_id, length(s.content) FROM {S}.sources s "
            "JOIN {S}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
            "WHERE cs.canonical_url = :u AND coalesce(s.provenance->>'stored_text_kind', '') <> :fd_kind "  # R5: an empty fulldoc_chunks row is never 'latest'
            "ORDER BY s.version DESC LIMIT 1", {"u": canonical, "fd_kind": FULLDOC_CHUNKS_KIND})
        return (rows[0][0], rows[0][1]) if rows else None

    def span(self, source_id: int, end: int):
        rows = self._rows("SELECT span_id FROM {S}.evidence_spans WHERE source_id = :s AND start_index = 0 "
                          "AND end_index = :e LIMIT 1", {"s": source_id, "e": end})
        return rows[0][0] if rows else None

    def answer(self):
        rows = self._rows("SELECT answer_id FROM {S}.answers WHERE response_text = :qt AND model = :m "
                          "ORDER BY answer_id LIMIT 1",
                          {"qt": self.cfg["question_text"], "m": self.cfg["gate_version"]})
        return rows[0][0] if rows else None

    def duplicate(self, claim_text: str, span_ids: list) -> bool:
        """Any known memory state counts (pending, confirmed, rejected, tombstoned, legacy): a rejected claim is
        never re-queued."""
        rows = self._rows("SELECT c.claim_id FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
                          "WHERE a.response_text = :qt AND c.claim_text = :t "
                          "AND c.detailed_status = ANY(CAST(:sts AS text[])) "
                          "AND c.evidence_span_ids @> CAST(:e AS jsonb) LIMIT 1",
                          {"qt": self.cfg["question_text"], "t": claim_text, "sts": _known_statuses(self.cfg),
                           "e": json.dumps(list(span_ids))})
        return bool(rows)

    def pending_count(self) -> int:
        rows = self._rows("SELECT count(*) FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
                          "WHERE a.response_text = :qt AND c.detailed_status = :p",
                          {"qt": self.cfg["question_text"], "p": self.cfg["status_pending"]})
        return int(rows[0][0]) if rows else 0

    def claim_status(self, claim_id: int):
        """(detailed_status, expired) of a memory row (joined to the fixed memory answer), or None if it is not one.
        expired = older than ttl_days, so recall would never return it even if confirmed."""
        rows = self._rows("SELECT c.detailed_status, c.created_at <= now() - make_interval(days => :ttl) "
                          "FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
                          "WHERE a.response_text = :qt AND c.claim_id = :id",
                          {"qt": self.cfg["question_text"], "id": int(claim_id), "ttl": int(self.cfg["ttl_days"])})
        return (rows[0][0], bool(rows[0][1])) if rows else None

    def pending(self) -> list[dict]:
        """Rows awaiting review (pending + legacy auto-written), oldest first, with their cited sources."""
        sts = [self.cfg["status_pending"], *self.cfg.get("status_legacy_auto", ())]
        rows = self._rows(
            "SELECT c.claim_id, c.claim_text, c.created_at, c.detailed_status, "
            "c.created_at <= now() - make_interval(days => :ttl), "
            "COALESCE(jsonb_agg(DISTINCT jsonb_build_object('title', s.provenance ->> 'title', "
            "'url', COALESCE(s.provenance ->> 'url', cs.canonical_url))) FILTER (WHERE s.source_id IS NOT NULL), "
            "'[]'::jsonb) "
            "FROM {S}.claims c JOIN {S}.answers a ON a.answer_id = c.answer_id "
            "LEFT JOIN LATERAL jsonb_array_elements_text(c.evidence_span_ids) AS sp(span_id) ON true "
            "LEFT JOIN {S}.evidence_spans e ON e.span_id = sp.span_id::bigint "
            "LEFT JOIN {S}.sources s ON s.source_id = e.source_id "
            "LEFT JOIN {S}.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
            "WHERE a.response_text = :qt AND c.detailed_status = ANY(CAST(:sts AS text[])) "
            "GROUP BY c.claim_id, c.claim_text, c.created_at, c.detailed_status "
            "ORDER BY c.created_at, c.claim_id LIMIT :k",
            {"qt": self.cfg["question_text"], "sts": sts, "k": int(self.cfg["pending_list_limit"]),
             "ttl": int(self.cfg["ttl_days"])})
        out = []
        for r in rows:
            srcs = r[5] if isinstance(r[5], list) else json.loads(r[5] or "[]")
            out.append({"claim_id": int(r[0]), "claim_text": r[1],
                        "created_at": r[2].isoformat() if r[2] else "", "state": _state_of(r[3], self.cfg),
                        "expired": bool(r[4]),
                        "sources": [{"title": x.get("title") or "", "url": x.get("url") or ""} for x in srcs]})
        return out


class WoMemoryStore(WoCorpusStore):
    """INSERT ... RETURNING in savepoints (helpers inherited from the corpus store). Only INSERT and the
    operator tombstone UPDATE; no DELETE."""

    def __init__(self, engine, cfg: dict):
        super().__init__(engine, cfg["question_text"])
        self.cfg = cfg

    def answer_id(self) -> int:
        with self._lock, self.engine.begin() as c:
            qid = self._one(c, "INSERT INTO {S}.questions (research_question) VALUES (:q) RETURNING question_id",
                            {"q": self.question_text})
            rid = self._one(c, "INSERT INTO {S}.runs (question_id, question) VALUES (:q, :t) RETURNING run_id",
                            {"q": qid, "t": self.question_text})
            return self._one(c, "INSERT INTO {S}.answers (run_id, stage, response_text, model) "
                                "VALUES (:r, :st, :t, :m) RETURNING answer_id",
                             {"r": rid, "st": STAGE_FINAL, "t": self.question_text, "m": self.cfg["gate_version"]})

    def span_id(self, source_id: int, end: int) -> int | None:
        with self.engine.begin() as c:
            return self._insert_new(c, "INSERT INTO {S}.evidence_spans (source_id, text, start_index, end_index) "
                                       "VALUES (:s, :t, 0, :e) RETURNING span_id",
                                    {"s": source_id, "t": self.cfg["span_text"], "e": end})

    def add_claim(self, answer_id: int, claim_text: str, span_ids: list, status: str | None = None) -> int:
        """New rows are pending unless confirmation is explicitly switched off (D6, deny-by-default)."""
        if status is None:
            status = self.cfg["status_pending"] if confirm_required(self.cfg) else self.cfg["status_live"]
        with self.engine.begin() as c:
            return self._one(c, "INSERT INTO {S}.claims (answer_id, claim_text, evidence_span_ids, "
                                "verification_status, detailed_status) "
                                "VALUES (:a, :t, CAST(:e AS jsonb), :v, :d) RETURNING claim_id",
                             {"a": answer_id, "t": claim_text, "e": json.dumps(list(span_ids)),
                              "v": VERIFICATION_SUPPORTED, "d": status})

    def add_discovery_note(self, answer_id: int, claim_text: str) -> int:
        """K2.I4: always a pending, unverified row with NO evidence spans, whatever confirm_required says (OD-K15d:
        subject to validation and owner confirmation). A missing status_pending key means no insert."""
        pending = self.cfg.get("status_pending")
        if not pending:
            raise ValueError("memory.store status_pending is missing; discovery note not written")
        with self.engine.begin() as c:
            return self._one(c, "INSERT INTO {S}.claims (answer_id, claim_text, evidence_span_ids, "
                                "verification_status, detailed_status) "
                                "VALUES (:a, :t, CAST(:e AS jsonb), :v, :d) RETURNING claim_id",
                             {"a": answer_id, "t": claim_text, "e": "[]", "v": VERIFICATION_UNVERIFIED,
                              "d": pending})

    def set_status(self, claim_id: int, status: str):
        """Operator transition (D6). wo has UPDATE on claims but SELECT only on claim_id, so the caller checks the
        current state through the ro reader first."""
        with self.engine.begin() as c:
            return self._one(c, "UPDATE {S}.claims SET detailed_status = :d WHERE claim_id = :id RETURNING claim_id",
                             {"d": status, "id": int(claim_id)})

    def tombstone(self, claim_id: int):
        """Operator use only (not called by the app)."""
        with self.engine.begin() as c:
            return self._one(c, "UPDATE {S}.claims SET detailed_status = :d WHERE claim_id = :id RETURNING claim_id",
                             {"d": self.cfg["status_tombstone"], "id": claim_id})


class MemoryUnavailable(RuntimeError):
    """Operator action requested but the ro or wo store is not configured."""


class MemoryService:
    def __init__(self, cfg: dict, gate_cfg, reader: RoMemoryReader | None, store: WoMemoryStore | None,
                 clock=time.time):
        self.cfg, self.gate_cfg, self.reader, self.store, self.clock = cfg, gate_cfg, reader, store, clock
        self._answer_id, self._day, self._day_n, self._lock = None, None, 0, threading.Lock()

    def recall(self, question: str) -> list[dict]:
        if self.reader is None or not answer_contract.evidence_recall_enabled():  # V3: off unless explicitly true
            return []
        try:
            return self.reader.recall(question)
        except Exception as exc:  # noqa: BLE001  (memory never fails /ask)
            logger.warning("memory recall failed (%s)", type(exc).__name__)
            return []

    def recall_discovery(self, question: str) -> list[dict]:
        """Live discovery notes for a related question; [] without L2 or on any error. Never raises."""
        if self.reader is None or "L2" not in answer_contract.layers():
            return []
        try:
            return self.reader.recall_discovery(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory discovery recall failed (%s)", type(exc).__name__)
            return []

    def write_discovery_note(self, candidates, question: str) -> tuple[int, dict]:
        """(notes written, {skip reason: count}). Never raises. The question is a gate input only."""
        from vera import discovery_memory as dm
        skipped: dict = {}
        if self.reader is None or self.store is None:
            return 0, {SKIP_DISABLED: 1}
        reserved = False
        try:
            if "L2" not in answer_contract.layers() or not answer_contract.write_on_l0():
                return 0, {SKIP_DISABLED: 1}
            if int((answer_contract.load_config().get("discovery_memory") or {}).get("max_per_answer", 0)) < 1:
                return 0, {SKIP_CAP: 1}
            if not self.cfg.get("status_pending"):
                return 0, {SKIP_DISABLED: 1}
            if not self._reserve():
                return 0, {SKIP_CAP: 1}
            reserved = True
            cands = dm.eligible_candidates(candidates, self.gate_cfg)
            note = dm.build_note(cands, self.cfg, gate_cfg=self.gate_cfg)
            if note is None:
                skipped["no_candidates"] = 1
            else:
                sentence = note.partition("\n")[0]
                res = dm.check_discovery_note(sentence, cands, question, self.gate_cfg)
                if res.allowed and _overlap(note, question or "", self.gate_cfg) > self.gate_cfg.max_question_overlap:
                    res = GateResult(False, (ECHOES_QUESTION,))
                if not res.allowed:
                    skipped = {r: 1 for r in res.reasons}
                elif self.reader.pending_count() >= int(self.cfg["max_pending"]):
                    skipped[SKIP_QUEUE_FULL] = 1
                elif self.reader.duplicate(note, []):
                    skipped[SKIP_DUPLICATE] = 1
                else:
                    if self._answer_id is None:
                        self._answer_id = self.reader.answer() or self.store.answer_id()
                    self.store.add_discovery_note(self._answer_id, note)
                    reserved = False
                    return 1, {}
            if reserved:
                self._release()
        except Exception as exc:  # noqa: BLE001
            if reserved:
                self._release()
            skipped = {"error": type(exc).__name__}
            logger.warning("memory discovery note failed (%s)", type(exc).__name__)
        if skipped:
            logger.info("memory discovery note skipped: %s", skipped)
        return 0, skipped

    # --- D6 operator actions (callers enforce the owner key; these raise on DB errors) ---
    def _require_rw(self):
        if self.reader is None or self.store is None:
            raise MemoryUnavailable("memory store not configured (needs both the ro and wo DB URLs)")

    def list_pending(self) -> list[dict]:
        self._require_rw()
        return self.reader.pending()

    def _transition(self, claim_id: int, allowed: tuple, target: str, outcome: str) -> str:
        self._require_rw()
        found = self.reader.claim_status(claim_id)
        state = _state_of(found[0] or "", self.cfg) if found else None
        if state is None:
            return NOT_FOUND
        if state not in allowed or (target == self.cfg["status_live"] and found[1]):
            return NOT_ALLOWED  # also: confirming a row past ttl_days would be a no-op (never recalled)
        self.store.set_status(claim_id, target)
        logger.info("memory operator %s claim %d (from %s)", outcome, int(claim_id), state)
        return outcome

    def confirm(self, claim_id: int) -> str:
        return self._transition(claim_id, (STATE_PENDING, STATE_LEGACY), self.cfg["status_live"], CONFIRMED)

    def reject(self, claim_id: int) -> str:
        return self._transition(claim_id, (STATE_PENDING, STATE_LEGACY, STATE_LIVE), self.cfg["status_rejected"],
                                REJECTED)

    def _reserve(self) -> bool:
        """Check-and-increment the daily cap under one lock acquisition (released if the write is skipped)."""
        day = int(self.clock() // 86400)
        with self._lock:
            if self._day != day:
                self._day, self._day_n = day, 0
            if self._day_n >= int(self.cfg["max_writes_per_day"]):
                return False
            self._day_n += 1
            return True

    def _release(self):
        with self._lock:
            self._day_n = max(0, self._day_n - 1)

    def write_claims(self, records, sources, question: str) -> tuple[int, dict]:
        """(rows written, {skip reason: count}). Never raises."""
        skipped: dict = {}
        written = 0
        if self.reader is None or self.store is None:
            return 0, {SKIP_DISABLED: 1}
        try:
            for rec in records or ():
                if written >= int(self.cfg["max_writes_per_answer"]) or not self._reserve():
                    skipped[SKIP_CAP] = skipped.get(SKIP_CAP, 0) + 1
                    continue
                try:
                    reason = self._write_one(rec, sources, question)
                except Exception:
                    self._release()
                    raise
                if reason is None:
                    written += 1
                else:
                    self._release()
                    skipped[reason] = skipped.get(reason, 0) + 1
        except Exception as exc:  # noqa: BLE001
            skipped["error"] = type(exc).__name__
            logger.warning("memory write failed (%s)", type(exc).__name__)
        if skipped:
            logger.info("memory write skipped: %s", skipped)
        return written, skipped

    def _write_one(self, rec: dict, sources, question: str):
        cited = [sources[n - 1] for n in rec["citations"] if 1 <= n <= len(sources)]
        refs = [SourceRef(provider=s.get("provider") or "", licence_decision=s.get("licence_decision") or "",
                          identifier=s.get("identifier") or "", url=s.get("url") or "") for s in cited]
        res = check_memory_write(self.cfg["fact_type"], rec["sentence"], rec["verdict"], refs, question,
                                 self.gate_cfg)
        if not res.allowed:
            return res.reasons[0]
        text = normalise_claim(rec["sentence"], self.gate_cfg)
        spans = []  # one whole-abstract evidence span per distinct cited source
        for canon in dict.fromkeys(canonical_url(s) for s in cited):
            found = self.reader.source(canon)
            if found is None or not found[1]:
                return SKIP_NOT_IN_CORPUS
            source_id, length = found
            span = self.reader.span(source_id, length) or self.store.span_id(source_id, length) \
                or self.reader.span(source_id, length)
            if span is None:
                return "span_unavailable"
            spans.append(span)
        if self.reader.duplicate(text, spans):
            return SKIP_DUPLICATE
        if confirm_required(self.cfg) and self.reader.pending_count() >= int(self.cfg["max_pending"]):
            return SKIP_QUEUE_FULL
        if self._answer_id is None:
            self._answer_id = self.reader.answer() or self.store.answer_id()
        self.store.add_claim(self._answer_id, text, spans)
        return None


_SERVICE: dict = {}
D6_KEYS = ("status_pending", "status_rejected", "max_pending", "pending_list_limit")


def _check_d6_keys(cfg: dict):
    """Fail at load, not silently at write time, when confirmation is on and a D6 key is missing."""
    missing = [k for k in D6_KEYS if k not in cfg]
    if confirm_required(cfg) and missing:
        raise ValueError(f"memory.store config is missing D6 keys {missing}; memory stays off (owner: R9/R1)")


def memory_from_env(block: dict | None, env=os.environ, engine_factory=None, gate_cfg=None):
    """The process-wide MemoryService, or None when memory is disabled or neither DB URL is set."""
    cfg = (block or {}).get("store")
    if not cfg or not cfg.get("enabled"):
        return None
    if "svc" not in _SERVICE:
        if engine_factory is None:
            from sqlalchemy import create_engine as engine_factory
        ro_url = (env.get(cfg["db_url_ro_env"]) or "").strip()
        wo_url = (env.get(cfg["db_url_wo_env"]) or "").strip()
        if not ro_url:
            return None
        _check_d6_keys(cfg)
        if gate_cfg is None:
            from vera.memory_gate import gate_config_from_dict
            gate_cfg = gate_config_from_dict(block)
        kw = {"pool_pre_ping": True, "hide_parameters": True,
              "connect_args": {"connect_timeout": int(cfg["connect_timeout_s"]),
                               "options": f"-c statement_timeout={int(cfg['statement_timeout_ms'])}"}}
        reader = RoMemoryReader(engine_factory(ro_url, **kw), cfg)
        store = WoMemoryStore(engine_factory(wo_url, **kw), cfg) if wo_url else None
        _SERVICE["svc"] = MemoryService(cfg, gate_cfg, reader, store)
    return _SERVICE["svc"]


def load_memory_block(path=None) -> dict | None:
    from vera.memory_gate import CONFIG_KEY, CONFIG_PATH
    try:
        return json.loads(open(path or CONFIG_PATH).read()).get(CONFIG_KEY)
    except (OSError, ValueError):
        return None
