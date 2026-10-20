"""M2 orchestration: search -> Gate A score -> threshold decision -> selective fetch -> persist.

Writes the M1 schema (db/migrations/001_vera_schema_init.sql): search_iterations, candidates,
canonical_sources, sources. M3 selects its input with:

    SELECT c.*, s.source_id, s.version, s.content, s.content_hash
    FROM vera_vjay.candidates c JOIN vera_vjay.sources s ON s.candidate_id = c.candidate_id
    WHERE c.run_id = :run AND c.gate_a_decision = 'fetch' AND c.fetch_status = 'fetched';

Known M1 schema gap: sources.candidate_id is the only candidate<->source link and a source row is
immutable, so when a *different* candidate re-fetches byte-identical content already stored under
the same canonical URL (e.g. a later run), no new sources row can be made and that candidate has no
join row. The runner still reports it as admitted with the existing source_id; M3 should look the
content up by canonical_url/content_hash for such rows (or M1 should add candidates.source_id).

Cost: the schema has no cost column, so each iteration's ledger is returned on M2Result.cost_calls
and appended as one JSON line to $VERA_COST_LOG (if set) for M6 to sum.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field

from sqlalchemy import text

from vera.cost_ledger import CostLedger
from vera.licence_gate import apply_licence_gate
from vera.gate_a import GateAConfig, score_candidates
from vera.search_and_fetch import normalize_url, search_question
from vera.selective_fetch import fetch_candidate

SCHEMA = "vera_vjay"
logger = logging.getLogger("vera")


@dataclass
class M2Result:
    admitted: list[dict] = field(default_factory=list)  # candidates with a registered source, ready for M3
    deferred: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    fetch_failed: list[dict] = field(default_factory=list)
    iteration_id: int | None = None
    cost_usd: float = 0.0
    cost_calls: list = field(default_factory=list)


class PostgresStore:
    """SQLAlchemy persistence against the M1 schema (db/migrations/001_vera_schema_init.sql).
    `commit=False` lets a caller (the integration test) keep everything in one transaction
    it later rolls back. Needs vera_vjay_rw: INSERT/UPDATE on candidates and search_iterations,
    INSERT only on canonical_sources and sources (immutable)."""

    def __init__(self, session, schema: str = SCHEMA, commit: bool = True):
        self.s, self.schema, self.commit = session, schema, commit

    def _done(self):
        self.s.commit() if self.commit else self.s.flush()

    def _x(self, sql: str, params: dict):
        return self.s.execute(text(sql.replace("{S}", self.schema)), params)

    def existing_urls(self, run_id) -> set[str]:
        rows = self._x("SELECT source_url FROM {S}.candidates WHERE run_id=:r", {"r": run_id})
        return {normalize_url(r[0]) for r in rows}

    def start_iteration(self, run_id, iteration, query, triggered_by_gate_c=False, targets_requirement=None) -> int:
        # Idempotent on (run_id, iteration_no, search_query): a retried iteration reuses its row.
        row = self._x(
            "INSERT INTO {S}.search_iterations (run_id, iteration_no, search_query, triggered_by_gate_c, targets_requirement) "
            "VALUES (:r,:i,:q,:g,:t) ON CONFLICT (run_id, iteration_no, search_query) "
            "DO UPDATE SET search_query=EXCLUDED.search_query RETURNING iteration_id",
            {"r": run_id, "i": iteration, "q": query, "g": triggered_by_gate_c, "t": targets_requirement}).one()
        self._done()
        return row[0]

    def insert_candidate(self, run_id, iteration_id, c: dict) -> int:
        """Always inserted with fetch_status='pending'; the fetch outcome is recorded by update_candidate."""
        row = self._x(
            "INSERT INTO {S}.candidates (run_id, search_iteration_id, source_url, title, snippet, search_rank, "
            "gate_a_score, gate_a_uncertainty, gate_a_decision, gate_a_rationale) "
            "VALUES (:r,:it,:url,:title,:snip,:rank,:score,:unc,:dec,:rat) RETURNING candidate_id",
            {"r": run_id, "it": iteration_id, "url": c["url"], "title": c.get("title"), "snip": c.get("snippet"),
             "rank": c.get("rank"), "score": c["gate_a_score"], "unc": c["gate_a_uncertainty_value"],
             "dec": c["gate_a_decision"], "rat": c["gate_a_rationale"]}).one()
        self._done()
        return row[0]

    def register_source(self, candidate_id, fr) -> tuple[int, int, bool]:
        """Returns (source_id, version, is_new_row). Identical content under the same canonical URL
        reuses the existing version (schema forbids a duplicate); changed content becomes version N+1."""
        canon = normalize_url(fr.url)
        self._x("INSERT INTO {S}.canonical_sources (canonical_url) VALUES (:u) ON CONFLICT (canonical_url) DO NOTHING",
                {"u": canon})
        cs = self._x("SELECT canonical_source_id FROM {S}.canonical_sources WHERE canonical_url=:u", {"u": canon}).scalar_one()
        hit = self._x("SELECT source_id, version FROM {S}.sources WHERE canonical_source_id=:c AND content_hash=:h",
                      {"c": cs, "h": fr.content_hash}).first()
        if hit:
            self._done()
            return hit[0], hit[1], False
        ver = self._x("SELECT COALESCE(MAX(version),0)+1 FROM {S}.sources WHERE canonical_source_id=:c", {"c": cs}).scalar()
        sid = self._x(
            "INSERT INTO {S}.sources (canonical_source_id, candidate_id, version, content, content_hash, provenance) "
            "VALUES (:c,:cand,:v,:t,:h,CAST(:p AS jsonb)) RETURNING source_id",
            {"c": cs, "cand": candidate_id, "v": ver, "t": fr.content_text, "h": fr.content_hash,
             "p": json.dumps(fr.provenance)}).scalar_one()
        self._done()
        return sid, ver, True

    def update_candidate(self, candidate_id, fetch_status, error=None):
        self._x("UPDATE {S}.candidates SET fetch_status=:st, fetch_error=:e WHERE candidate_id=:id",
                {"st": fetch_status, "e": error, "id": candidate_id})
        self._done()

    def finish_iteration(self, iteration_id, results_count: int):
        self._x("UPDATE {S}.search_iterations SET results_count=:n WHERE iteration_id=:id",
                {"n": results_count, "id": iteration_id})
        self._done()


def _append_cost_log(run_id, iteration, q, ledger: CostLedger) -> None:
    path = os.getenv("VERA_COST_LOG")
    line = {"stage": "m2", "run_id": run_id, "iteration": iteration, "query": q,
            "total_usd": ledger.total_usd(), "unknown_cost_calls": ledger.unknown_cost_calls(),
            "calls": ledger.as_dicts()}
    logger.info("m2 cost: %s", json.dumps({k: line[k] for k in ("run_id", "iteration", "total_usd")}))
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")


def run_m2(
    question: str,
    run_id: int,
    store,
    *,
    query: str | None = None,
    iteration: int = 1,
    triggered_by_gate_c: bool = False,
    targets_requirement: str | None = None,
    num_results: int = 10,
    cfg: GateAConfig | None = None,
    ledger: CostLedger | None = None,
    search_fn=search_question,
    score_fn=score_candidates,
    fetch_fn=fetch_candidate,
    deadline_s: float = 300.0,
    clock=time.monotonic,
) -> M2Result:
    """`query` defaults to `question`; Gate C re-search passes a refined query, iteration+1 and
    triggered_by_gate_c=True. URLs already stored for this run are skipped, so re-search only adds
    new candidates. Search failure raises after cost is logged; it is never treated as "no results".
    Fetches stop at `deadline_s`; remaining would-be fetches are persisted as deferred with the reason.
    """
    ledger = ledger or CostLedger()
    cfg = cfg or GateAConfig()
    q = query or question
    start = clock()
    iteration_id = store.start_iteration(run_id, iteration, q, triggered_by_gate_c, targets_requirement)
    res = M2Result(iteration_id=iteration_id)

    try:
        found = search_fn(q, num_results, ledger=ledger)
    except Exception:
        _append_cost_log(run_id, iteration, q, ledger)
        raise
    store.finish_iteration(iteration_id, len(found))
    seen = store.existing_urls(run_id)
    fresh = [c for c in found if normalize_url(c["url"]) not in seen]
    scored = apply_licence_gate(score_fn(question, fresh, cfg=cfg, ledger=ledger))

    for c in scored:
        if c["gate_a_decision"] == "fetch" and clock() - start > deadline_s:
            c["gate_a_decision"] = "defer"
            c["gate_a_rationale"] += f" Demoted to defer: run deadline {deadline_s}s exceeded before fetch."
        cid = store.insert_candidate(run_id, iteration_id, c)
        c["candidate_id"] = cid
        decision = c["gate_a_decision"]
        if decision == "reject":
            res.rejected.append(c); continue
        if decision == "defer":
            res.deferred.append(c); continue

        fr = fetch_fn(c["url"], ledger=ledger)
        if not fr.ok:
            store.update_candidate(cid, "failed", error=fr.error)
            c["fetch_error"] = fr.error
            res.fetch_failed.append(c)
            continue
        fr.provenance.update(search_query=q, search_provider=c.get("provider"), search_rank=c.get("rank"),
                             run_id=run_id, iteration_no=iteration, gate_a_score=c["gate_a_score"],
                             licence=c.get("licence"), doi=c.get("doi"), published=c.get("published"),
                             publisher=c.get("publisher"), source_type=c.get("source_type"),
                             provider_endpoint=c.get("provider_endpoint"), retrieved_at=c.get("retrieved_at"),
                             stored_text_kind=c.get("stored_text_kind"), licence_decision=c.get("licence_decision"))
        sid, ver, _new = store.register_source(cid, fr)
        store.update_candidate(cid, "fetched")
        c.update(source_id=sid, source_version=ver, content_hash=fr.content_hash)
        res.admitted.append(c)
    res.cost_usd, res.cost_calls = ledger.total_usd(), ledger.as_dicts()
    _append_cost_log(run_id, iteration, q, ledger)
    return res
