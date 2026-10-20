"""Persistence for M4/M5 results in the canonical vera_vjay schema (001/003/004 migrations).
Accepts a SQLAlchemy Engine (each write commits) or an open Connection (caller owns the transaction,
used by the rolled-back integration test). Pipeline vocabulary is mapped onto the schema's at this
boundary; precise statuses are preserved in claims.detailed_status / issues.

Schema is append-only for relations/decisions (rw role has no DELETE), so writes are INSERT-only and
idempotent where it matters (relations)."""
import json
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Connection

REL_TO_DB = {"supports": "supports", "refutes": "contradicts", "qualifies": "qualifies"}
CLAIM_STATUS_TO_DB = {"supported": "supported", "overreach": "partially_supported", "weak": "partially_supported",
                      "contested": "partially_supported", "unsupported": "unsupported", "unverified": "unverified",
                      "revised": "unverified"}


def _j(v):
    return json.dumps(v, default=str)


def _i(v):
    return int(v)


class PipelineStore:
    def __init__(self, engine_or_conn):
        self.target = engine_or_conn

    @contextmanager
    def _tx(self):
        if isinstance(self.target, Connection):
            yield self.target
        else:
            with self.target.begin() as c:
                yield c

    def save_gate_c(self, run_id, attempt, gate):
        with self._tx() as c:
            c.execute(text("insert into vera_vjay.gate_c_decisions (run_id, attempt, decision, detail) "
                           "values (:r,:a,:d, cast(:j as jsonb))"),
                      {"r": _i(run_id), "a": attempt, "d": gate["decision"], "j": _j(gate)})

    def save_relations(self, run_id, graph):
        with self._tx() as c:
            for e in graph["edges"]:
                c.execute(text(
                    "insert into vera_vjay.evidence_relations (span1_id, span2_id, relation_type, confidence) "
                    "select :a,:b,:t,:c where not exists (select 1 from vera_vjay.evidence_relations "
                    "where span1_id=:a and span2_id=:b and relation_type=:t)"),
                    {"a": _i(e["span1_id"]), "b": _i(e["span2_id"]), "t": REL_TO_DB[e["relation_type"]],
                     "c": round(e["confidence"], 2)})

    def save_reasoning_context(self, run_id, payload):
        with self._tx() as c:
            c.execute(text("insert into vera_vjay.reasoning_context (run_id, constructed_context) "
                           "values (:r, cast(:j as jsonb))"), {"r": _i(run_id), "j": _j(payload)})

    def save_answer(self, run_id, stage, response_text, *, model=None, tokens=0, cost=0.0,
                    latency=0.0, parent_answer_id=None) -> int:
        with self._tx() as c:
            return c.execute(text(
                "insert into vera_vjay.answers (run_id, stage, response_text, model, tokens, cost, latency, parent_answer_id) "
                "values (:r,:s,:t,:m,:k,:c,:l,:p) returning answer_id"),
                {"r": _i(run_id), "s": stage, "t": response_text, "m": model, "k": tokens, "c": cost,
                 "l": latency, "p": parent_answer_id}).scalar_one()

    def save_claims(self, answer_id, claims):
        with self._tx() as c:
            for cl in claims:
                st = cl.get("verification_status", "unverified")
                c.execute(text(
                    "insert into vera_vjay.claims (answer_id, claim_text, evidence_span_ids, verification_status, "
                    "detailed_status, issues) values (:a,:t, cast(:e as jsonb), :s, :d, cast(:i as jsonb))"),
                    {"a": answer_id, "t": cl["claim_text"], "e": _j([_i(x) for x in cl.get("evidence_span_ids", [])]),
                     "s": CLAIM_STATUS_TO_DB[st], "d": st, "i": _j(cl.get("issues", []))})

    def finalize_run(self, run_id, final_response, final_answer_id, verification_status, gate_c_decision):
        with self._tx() as c:
            n = c.execute(text(
                "update vera_vjay.runs set engineered_response=:f, final_answer_id=:a, verification_status=:v, "
                "gate_c_decision=:g where run_id=:r"),
                {"r": _i(run_id), "f": final_response, "a": final_answer_id, "v": verification_status,
                 "g": gate_c_decision}).rowcount
            if n != 1:
                raise RuntimeError(f"finalize_run: expected to update exactly 1 vera_vjay.runs row for run_id={run_id}, got {n}")
