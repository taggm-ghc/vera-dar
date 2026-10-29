"""Item #79 phase C: admit /ask sources that pass VERA-DAR's gauntlet into the corpus.

Gauntlet, per source with an abstract: Gate A scoring (vera/gate_a.py: query fit, evidentiary value, cost/risk)
then the licence gate (vera/licence_gate.py). Every scored source is recorded as a candidate with its decision
and rationale; only `fetch` decisions that the licence gate allows become corpus sources (metadata plus the
abstract, never full text). Runs after the /ask response, so visitors never wait for it.

Privacy (#74): the visitor's question is used in memory for scoring only. Rows are filed under one fixed
corpus question per process (config `corpus.question_text`); the visitor's text is never written.

Account: the production write-only role (vera_pipeline_wo), URL from the env var named in config
`corpus.db_url_env` (never printed). That role may INSERT and read back ids but not read content, and ON CONFLICT
needs read access to the conflict columns, so every step is a plain INSERT ... RETURNING inside a savepoint; a
unique-key violation means the row already exists (a source already in the corpus is counted as known).
Verified against the live grants on 2026-10-07 in a rolled-back transaction.
"""
import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

from vera.gate_a import GateAConfig, score_candidates
from urllib.parse import urlsplit

from vera.licence_gate import METADATA_KIND, apply_licence_gate

logger = logging.getLogger("vera")

SCHEMA = "vera_vjay"


LICENCE_OBS_INSERT = (
    "INSERT INTO {S}.licence_observations (obs_id, observed_at, source_system, portal, portal_basis, host, url_sha256, "
    "url, doc_ref, licence_scope, governing, spdx, family, nc, nd, confidence, detected_via, decision, native_decision, "
    "reason_code, config_version, run_id, candidate_id) VALUES (:obs_id, :observed_at, 'vera-admission', :portal, "
    ":portal_basis, :host, :url_sha256, :url, :doc_ref, :licence_scope, :governing, :spdx, :family, :nc, :nd, "
    ":confidence, :detected_via, :decision, :native_decision, :reason_code, :config_version, :run_id, :candidate_id)")
_DECISION = {"allow": "allow", "reject": "reject", "defer": "hold"}  # the licence gate's names -> the table's


def _host(url: str) -> str:
    h = (urlsplit(str(url or "")).hostname or "").lower()
    return h if h and not any(ch in h for ch in "/@ \t\n") else ""


def licence_observation_rows(cand: dict, run_id=None, now=None) -> list[dict]:
    """Two rows (metadata, content) from a candidate after the licence gate. Licence fields are None when not determined."""
    url = str(cand.get("url") or "")
    host = _host(url)
    decision = _DECISION.get(cand.get("licence_decision"), "hold")
    lic = cand.get("licence") or {}
    governing = "metadata" if cand.get("stored_text_kind") == METADATA_KIND else "content"
    observed = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"observed_at": observed, "portal": host, "portal_basis": "host", "host": host,
            "url_sha256": hashlib.sha256(url.encode("utf-8", "replace")).hexdigest(),
            "url": url if decision == "allow" else None, "doc_ref": cand.get("doi"), "family": None,
            "nc": None, "nd": None, "confidence": None, "decision": decision,
            "native_decision": cand.get("gate_a_decision"), "reason_code": f"licence_{cand.get('licence_decision') or 'unknown'}",
            "config_version": None, "run_id": None if run_id is None else str(run_id), "candidate_id": None}
    out = []
    for scope in ("metadata", "content"):
        rec = lic.get(scope) or {}
        out.append({**base, "obs_id": str(uuid.uuid4()), "licence_scope": scope, "governing": scope == governing,
                    "spdx": rec.get("id"), "detected_via": rec.get("source")})
    return out


class WoCorpusStore:
    """Write-only persistence: no SELECT on content columns, no UPDATE of immutable rows."""

    def __init__(self, engine, question_text: str):
        self.engine, self.question_text = engine, question_text
        self._run_id, self._lock = None, threading.Lock()

    def _one(self, conn, sql: str, params: dict):
        from sqlalchemy import text
        row = conn.execute(text(sql.replace("{S}", SCHEMA)), params).first()
        return row[0] if row else None

    def _insert_new(self, conn, sql: str, params: dict):
        """INSERT ... RETURNING in a savepoint; None when a unique key says the row already exists."""
        from sqlalchemy.exc import IntegrityError
        sp = conn.begin_nested()
        try:
            value = self._one(conn, sql, params)
            sp.commit()
            return value
        except IntegrityError as exc:
            sp.rollback()
            if getattr(getattr(exc, "orig", None), "pgcode", None) not in (None, "23505"):
                raise  # only a UNIQUE violation means "already exists"; CHECK/FK/NOT NULL failures surface to the caller
            return None

    def run_id(self) -> int:
        with self._lock:
            if self._run_id is None:
                with self.engine.begin() as c:
                    qid = self._one(c, "INSERT INTO {S}.questions (research_question) VALUES (:q) RETURNING question_id",
                                    {"q": self.question_text})
                    self._run_id = self._one(c, "INSERT INTO {S}.runs (question_id, question) VALUES (:q, :t) "
                                                "RETURNING run_id", {"q": qid, "t": self.question_text})
            return self._run_id

    def admit(self, cand: dict) -> str:
        """Record one scored candidate; store it as a source if admitted. Returns the outcome label."""
        rid = self.run_id()
        with self.engine.begin() as c:
            cid = self._insert_new(c, "INSERT INTO {S}.candidates (run_id, source_url, title, snippet, search_rank, "
                                      "gate_a_score, gate_a_uncertainty, gate_a_decision, gate_a_rationale) "
                                      "VALUES (:r,:u,:t,:s,:k,:g,:un,:d,:ra) RETURNING candidate_id",
                            {"r": rid, "u": cand["url"], "t": cand.get("title"), "s": cand.get("snippet"),
                             "k": cand.get("rank"), "g": cand.get("gate_a_score"),
                             "un": cand.get("gate_a_uncertainty_value"), "d": cand.get("gate_a_decision"),
                             "ra": cand.get("gate_a_rationale")})
            if cid is None:
                return "duplicate_in_run"
            if cand.get("gate_a_decision") != "fetch":
                return f"recorded_{cand.get('gate_a_decision') or 'unscored'}"
            csid = self._insert_new(c, "INSERT INTO {S}.canonical_sources (canonical_url) VALUES (:u) "
                                       "RETURNING canonical_source_id",
                             {"u": cand["canonical_url"]})
            if csid is None:
                return "known"
            content = cand["snippet"]
            sid = self._one(c, "INSERT INTO {S}.sources (canonical_source_id, candidate_id, version, content, content_hash, "
                         "provenance) VALUES (:cs,:cid,1,:t,:h,CAST(:p AS jsonb)) RETURNING source_id",
                      {"cs": csid, "cid": cid, "t": content, "h": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                       "p": json.dumps(cand["provenance"])})
            cand["canonical_source_id"], cand["source_id"] = csid, sid  # additive: DB ids for the DAR post-admission hook
            self._one(c, "UPDATE {S}.candidates SET fetch_status='fetched' WHERE candidate_id=:id RETURNING candidate_id",
                      {"id": cid})
        return "admitted"

    def record_licence_observations(self, cand: dict, run_id=None) -> int:
        """Append one licence observation per licence scope (metadata, content) for a decided candidate (item #97 H-23).
        Plain INSERT, no RETURNING, no text columns. Raises on DB failure; the caller counts it and never blocks."""
        rows = licence_observation_rows(cand, run_id)
        with self.engine.begin() as c:
            from sqlalchemy import text
            for r in rows:
                c.execute(text(LICENCE_OBS_INSERT.replace("{S}", SCHEMA)), r)
        return len(rows)


def canonical_url(src: dict) -> str:
    """R3 key for a NEW canonical source (R3NORM-1028: the single normaliser, vera.url_key.canonical_key).
    Lookups must use vera.url_key.lookup_keys so rows stored under the old spelling are still found."""
    from vera.url_key import canonical_key
    return canonical_key({"doi": src.get("doi"), "url": src["url"]})


def is_stored_document_source(src: dict) -> bool:
    """True for a source that came out of the DAR stored corpus (it already carries a full-document stored_text_kind
    by acquisition state). Such a source is never re-admitted: admission would write it as a NEW canonical source,
    mislabelled as abstract/metadata (review HIGH-2)."""
    from vera.answer_contract import ACQ_ACQUIRED, acquisition_state
    return acquisition_state({"stored_text_kind": src.get("stored_text_kind"), "admission_passed": True}) == ACQ_ACQUIRED


def gauntlet(question: str, sources: list[dict], llm_call, cfg: dict) -> list[dict]:
    """Gate A then the licence gate; returns scored candidates (copies), each with a final gate_a_decision."""
    sources = [s for s in sources if not is_stored_document_source(s)]
    cands = [{"url": s["url"], "title": s.get("title"), "snippet": s.get("snippet"), "rank": s.get("n"),
              "licence": s.get("licence"), "stored_text_kind": s.get("stored_text_kind"), "doi": s.get("doi"),
              "published": s.get("published"), "discovered_via": s.get("discovered_via"),
              "source_type": s.get("source_type")} for s in sources if s.get("snippet")]
    g = cfg.get("gate_a") or {}
    scored = score_candidates(question, cands, llm_call, cfg=GateAConfig(
        fetch_threshold=float(g.get("fetch_threshold", GateAConfig.fetch_threshold)),
        reject_threshold=float(g.get("reject_threshold", GateAConfig.reject_threshold)),
        max_fetch=int(g.get("max_fetch", GateAConfig.max_fetch))),
        safe_cost=True)  # FIXR-7DA-1028: production admission fails closed safely (defer), never raises on cost
    return apply_licence_gate(scored)


def dar_post_admit(dar_corpus, cand: dict, outcome: str, counts: dict, dar_persist=None) -> None:
    """DAR post-admission hook (DARINT L4). Flag-gated twice: config `enabled` and `corpus.admission_hook_enabled` must
    both be true on the DarCorpus config, else a strict no-op. Hands the admitted candidate to vera.dar_corpus, which
    accepts FULL documents only by stored provenance (answer_contract.acquisition_state); today admission stores
    abstracts (stored_text_kind 'abstract_metadata'), so every call is recorded as skipped: a known limit, not a bypass.
    Never raises and never affects admission."""
    try:
        ccfg = getattr(dar_corpus, "cfg", None) or {}
        if ccfg.get("enabled") is not True or (ccfg.get("corpus") or {}).get("admission_hook_enabled") is not True:
            return
        from vera.dar_corpus import doc_from_admitted

        res = dar_corpus.ingest_document(doc_from_admitted(cand, outcome))
        key = "dar_" + str(res.get("status"))
        counts[key] = counts.get(key, 0) + 1
        if res.get("status") == "indexed" and dar_persist is not None:
            dar_persist(res)
    except Exception as exc:  # noqa: BLE001
        counts["dar_error"] = counts.get("dar_error", 0) + 1
        logger.warning("DAR post-admission hook failed (%s)", type(exc).__name__)


def admit(question: str, sources: list[dict], llm_call, cfg: dict, store, dar_corpus=None, dar_persist=None) -> dict:
    """Run the gauntlet and record every candidate. Never raises; returns outcome counts (logged).
    dar_corpus / dar_persist are optional and flag-gated (see dar_post_admit); None = unchanged behaviour."""
    counts: dict = {}
    try:
        scored = gauntlet(question, sources, llm_call, cfg)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for c in scored:
            c["canonical_url"] = canonical_url(c)
            c["provenance"] = {"url": c["url"], "doi": c.get("doi"), "title": c.get("title"),
                               "published": c.get("published"), "discovered_via": c.get("discovered_via"),
                               "source_type": c.get("source_type"), "licence": c.get("licence"),
                               "licence_decision": c.get("licence_decision"), "gate_a_score": c.get("gate_a_score"),
                               "stored_text_kind": "abstract_metadata", "admitted_by": "/ask item #79",
                               "retrieved_at": now}
            outcome = store.admit(c)
            counts[outcome] = counts.get(outcome, 0) + 1
            if dar_corpus is not None:
                dar_post_admit(dar_corpus, c, outcome, counts, dar_persist)
            if cfg.get("licence_observations_enabled") is True:  # item #97 H-23; default off; never blocks admission
                try:
                    store.record_licence_observations(c, getattr(store, "_run_id", None))
                    counts["licence_obs_written"] = counts.get("licence_obs_written", 0) + 1
                except Exception as exc:  # noqa: BLE001
                    counts["licence_obs_failed"] = counts.get("licence_obs_failed", 0) + 1
                    logger.warning("licence observation write failed (%s)", type(exc).__name__)
    except Exception as exc:  # noqa: BLE001  (admission never affects the answer; fail verbosely in the log)
        counts["error"] = type(exc).__name__
        logger.warning("/ask corpus admission failed (%s)", type(exc).__name__)
    logger.info("/ask corpus admission: %s", counts)
    return counts


_STORE: dict = {}


def store_from_env(cfg: dict, env=os.environ, engine_factory=None):
    """The process-wide store, or None when corpus admission is disabled or its DB URL is unset."""
    if not cfg or not cfg.get("enabled"):
        return None
    url = (env.get(cfg["db_url_env"]) or "").strip()
    if not url:
        return None
    if "store" not in _STORE:
        if engine_factory is None:
            from sqlalchemy import create_engine as engine_factory
        _STORE["store"] = WoCorpusStore(engine_factory(url, pool_pre_ping=True, hide_parameters=True), cfg["question_text"])
    return _STORE["store"]


def llm_call_for(entry):
    """Gate A's llm_call bound to one /ask chain entry (JSON mode, temperature 0, the entry's extra_body)."""
    def call(messages):
        from vera.ask_service import _get_client
        client = entry.client() or _get_client()
        kw = {"extra_body": entry.extra_body} if entry.extra_body else {}
        r = client.chat.completions.create(model=entry.model, messages=messages, temperature=0,
                                           response_format={"type": "json_object"}, **kw)
        return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
    return call
