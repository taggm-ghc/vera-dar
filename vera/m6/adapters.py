"""Adapters from the real M2-M5 output shapes to the M6 scoring contract, and the DB loader.

Real shapes used (read from the repo 2026-09-30):
  M5  run_m4_m5(): {final_response, claims:[{claim_text, evidence_span_ids, verification_status,
                    issues}], gate_c_trace, reasoning_context, cost_usd_synthesis_revision}
  M3  corpus:      {sources:[{source_id,title,url,content_hash,decision,overall_quality,scores,
                    rationale,spans:[{span_id,text,start_index,end_index,...}]}], excluded:[...],
                    rubric_version, gate_b_policy:{version}}  (M4 flattens to corpus["spans"])
  M4  gate_c dict, relations {nodes, edges:[{span1_id,span2_id,relation_type,confidence}]}
Gate A lives on candidates (M2); pass it in, otherwise auditability cannot reach level 5.
"""
from __future__ import annotations

from pathlib import Path

from vera.eval_config import FREEZE_PATH, RUN_PATH, load_eval_config  # noqa: E402  (FREEZE_PATH re-exported)


def load_freeze(path: Path = FREEZE_PATH, run_path: Path = RUN_PATH) -> dict:
    """The frozen file's content (question, requirements, objections, success_criteria), loaded through the
    one validated loader (vera.eval_config): pin-checked against the run config and validated at load.
    Return shape is unchanged (a dict of the freeze file only), so M7 and load_from_db are unaffected."""
    return dict(load_eval_config(path, run_path).raw)


def build_corpus(m3_corpus: dict, freeze: dict, extra_spans: list[dict] | None = None) -> dict:
    """Scoring corpus: spans/sources keyed by id + frozen requirements/objections."""
    spans, sources = {}, {}
    for bucket in ("sources", "excluded"):
        for s in m3_corpus.get(bucket, []):
            sid = str(s["source_id"])
            sources[sid] = {"url": s.get("url"), "title": s.get("title"),
                            "content_hash": s.get("content_hash"), "version": s.get("version", 1),
                            "gate_b_decision": s.get("decision"), "aliases": s.get("aliases", [])}
            for sp in s.get("spans", []):
                spans[str(sp["span_id"])] = {"text": sp["text"], "source_id": sid,
                                            "start": sp.get("start_index"), "end": sp.get("end_index"),
                                            "evidence_type": sp.get("evidence_type")}
    for sp in list(m3_corpus.get("spans", [])) + list(extra_spans or []):
        k = str(sp["span_id"])
        if k not in spans:
            spans[k] = {"text": sp["text"], "source_id": str(sp.get("source_id", "")),
                        "start": sp.get("start_index"), "end": sp.get("end_index"),
                        "evidence_type": sp.get("evidence_type")}
    return {"requirements": freeze["requirements"], "objections": freeze["objections"],
            "spans": spans, "sources": sources,
            "rubric_version": m3_corpus.get("rubric_version"),
            "policy_version": (m3_corpus.get("gate_b_policy") or {}).get("version")}


def build_engineered(m5: dict, m3_corpus: dict, *, gate_a: list[dict] | None = None,
                     relations: dict | None = None, usage: dict | None = None) -> dict:
    """Map M5's result to the scoring contract. Removed claims are dropped (not in the answer)."""
    claims = []
    for i, c in enumerate(m5["claims"], 1):
        if c.get("verification_status") == "removed":
            continue
        claims.append({"id": f"c{i}", "text": c["claim_text"],
                       "citations": [str(x) for x in c.get("evidence_span_ids", [])],
                       "consequential": True, "verification_status": c.get("verification_status"),
                       "issues": c.get("issues", [])})
    gate_b = []
    rv = m3_corpus.get("rubric_version")
    pv = (m3_corpus.get("gate_b_policy") or {}).get("version")
    for bucket in ("sources", "excluded"):
        for s in m3_corpus.get(bucket, []):
            for sp in s.get("spans", []) or [{"span_id": None}]:
                if sp.get("span_id") is None:
                    continue
                gate_b.append({"span_id": str(sp["span_id"]), "source_id": str(s["source_id"]),
                               "decision": s.get("decision"), "rubric": s.get("scores"),
                               "overall_quality": s.get("overall_quality"),
                               "rationale": s.get("rationale"), "rubric_version": rv, "policy_version": pv})
    u = usage or {}
    return {"response_text": m5["final_response"], "claims": claims,
            "usage": {"api_calls": u.get("api_calls", 0), "tokens": u.get("tokens", 0),
                      "cost_usd": u.get("cost_usd", m5.get("cost_usd_synthesis_revision", 0.0)),
                      "latency_s": u.get("latency_s")} if usage else
            {"api_calls": None, "tokens": None,
             "cost_usd": m5.get("cost_usd_synthesis_revision"), "latency_s": None},
            "audit": {"gate_a": gate_a or [], "gate_b": gate_b,
                      "gate_c": {"trace": m5.get("gate_c_trace"), "decision": m5.get("gate_c_decision")},
                      "relations": (relations or {}).get("edges", []),
                      "verification_status": m5.get("verification_status"),
                      "unsupported_claims": m5.get("unsupported_claims", [])}}


def load_from_db(run_id=None, freeze: dict | None = None) -> dict:
    """Assemble {run_id, question, engineered, corpus} from vera_vjay for one run.
    Raises on anything missing; the caller turns that into a verbose failure.
    Assumed joins (M1 001 + 003 designs): answers.run_id, claims.answer_id,
    evidence_spans.span_id, appraisals.source_id, sources.source_id -> canonical_sources,
    candidates (Gate A). Ids compared as text because M3/M4 used TEXT ids.

    M6 schema requirements (005_vera_m6_schema_sync.sql):
      - runs.question, runs.final_response, runs.evaluated_at (added in 005)
      - appraisals.rubric_version, appraisals.policy_version (added in 005)
      - evidence_relations.run_id, evidence_relations.rationale (added in 005)
    """
    from sqlalchemy import text
    from vera.db import get_session
    freeze = freeze or load_freeze()
    with get_session() as s:
        q = ("SELECT run_id::text AS run_id, question, final_response, final_answer_id, "  # nosec B608 - WHERE chosen from two literals; run_id is a bound :id
             "verification_status, gate_c_decision FROM vera_vjay.runs "
             + ("WHERE run_id::text = :id " if run_id is not None else "WHERE final_response IS NOT NULL ")
             + "ORDER BY created_at DESC LIMIT 1")
        run = s.execute(text(q), {"id": str(run_id)} if run_id is not None else {}).mappings().first()
        if run is None or not run["final_response"]:
            raise RuntimeError("no finalized run (runs.final_response empty): M5 has not completed")
        rid = run["run_id"]
        claims = [dict(r) for r in s.execute(text(
            "SELECT claim_text, evidence_span_ids, verification_status, issues FROM vera_vjay.claims "
            "WHERE answer_id = :a ORDER BY claim_id"), {"a": run["final_answer_id"]}).mappings()]
        answers = [dict(r) for r in s.execute(text(
            "SELECT stage, tokens, cost, latency FROM vera_vjay.answers WHERE run_id = :r"),
            {"r": rid}).mappings()]
        span_ids = sorted({str(x) for c in claims for x in (c["evidence_span_ids"] or [])})
        # CRITICAL: Filter evidence_spans by run_id to prevent cross-run data leaks.
        # Join chain: evidence_spans -> sources -> candidates -> runs
        spans = [dict(r) for r in s.execute(text(
            "SELECT es.span_id::text AS span_id, es.source_id::text AS source_id, es.text, es.start_index, es.end_index, "
            "es.evidence_type FROM vera_vjay.evidence_spans es "
            "JOIN vera_vjay.sources src ON es.source_id = src.source_id "
            "JOIN vera_vjay.candidates cand ON src.candidate_id = cand.candidate_id "
            "WHERE cand.run_id = :r"), {"r": rid}).mappings()]
        # CRITICAL: Filter appraisals by run_id to prevent cross-run data leaks.
        # Join chain: appraisals -> sources -> candidates -> runs
        appr = {str(r["source_id"]): dict(r) for r in s.execute(text(
            "SELECT a.source_id::text AS source_id, a.gate_b_decision, a.overall_quality, a.rationale, a.rubric_version, "
            "a.policy_version FROM vera_vjay.appraisals a "
            "JOIN vera_vjay.sources src ON a.source_id = src.source_id "
            "JOIN vera_vjay.candidates cand ON src.candidate_id = cand.candidate_id "
            "WHERE cand.run_id = :r"), {"r": rid}).mappings()}
        # CRITICAL: Filter sources by run_id to prevent cross-run data leaks.
        # Join chain: sources -> candidates -> runs
        srcs = {str(r["source_id"]): dict(r) for r in s.execute(text(
            "SELECT s.source_id::text AS source_id, s.content_hash, s.version, cs.canonical_url AS url "
            "FROM vera_vjay.sources s "
            "JOIN vera_vjay.canonical_sources cs ON cs.canonical_source_id = s.canonical_source_id "
            "JOIN vera_vjay.candidates cand ON s.candidate_id = cand.candidate_id "
            "WHERE cand.run_id = :r"), {"r": rid}).mappings()}
        gc = [dict(r) for r in s.execute(text(
            "SELECT attempt, decision, detail FROM vera_vjay.gate_c_decisions WHERE run_id = :r "
            "ORDER BY attempt"), {"r": rid}).mappings()]
        # CRITICAL: evidence_relations now has run_id FK; filter to prevent cross-run data leaks
        rel = [dict(r) for r in s.execute(text(
            "SELECT span1_id::text AS span1_id, span2_id::text AS span2_id, relation_type, confidence, "
            "rationale FROM vera_vjay.evidence_relations WHERE run_id = :r"), {"r": rid}).mappings()]
        ga = [dict(r) for r in s.execute(text(
            "SELECT c.candidate_id::text AS candidate_id, c.source_url, c.title, c.gate_a_score, "
            "c.gate_a_decision, c.gate_a_rationale, so.source_id::text AS source_id "
            "FROM vera_vjay.candidates c LEFT JOIN vera_vjay.sources so ON so.candidate_id = c.candidate_id "
            "WHERE c.run_id = :r"), {"r": rid}).mappings()]
    by_src: dict[str, list] = {}
    for sp in spans:
        by_src.setdefault(sp["source_id"], []).append(sp)
    m3 = {"sources": [], "excluded": [], "rubric_version": None, "gate_b_policy": {}}
    for sid, sps in by_src.items():
        a, src = appr.get(sid, {}), srcs.get(sid, {})
        m3["rubric_version"] = m3["rubric_version"] or a.get("rubric_version")
        m3["gate_b_policy"] = m3["gate_b_policy"] or {"version": a.get("policy_version")}
        entry = {"source_id": sid, "url": src.get("url"), "content_hash": src.get("content_hash"),
                 "version": src.get("version"), "decision": a.get("gate_b_decision"),
                 "overall_quality": a.get("overall_quality"), "rationale": a.get("rationale"),
                 "spans": sps}
        m3["excluded" if a.get("gate_b_decision") == "reject" else "sources"].append(entry)
    m5 = {"final_response": run["final_response"],
          "claims": [dict(c, evidence_span_ids=c["evidence_span_ids"] or []) for c in claims],
          "gate_c_trace": [g["decision"] for g in gc], "gate_c_decision": run["gate_c_decision"],
          "verification_status": run["verification_status"]}
    llm = [a for a in answers if a["stage"] in ("draft", "revised")]
    usage = {"api_calls": len(llm), "tokens": sum(a["tokens"] or 0 for a in llm),
             "cost_usd": float(sum(a["cost"] or 0 for a in llm)),
             "latency_s": float(sum(a["latency"] or 0 for a in llm)),
             "complete": False}  # M2-M4 spend/latency are not in the DB; see merge_usage
    gate_a = [{"source_id": g["source_id"], "score": g["gate_a_score"], "decision": g["gate_a_decision"],
               "rationale": g["gate_a_rationale"], "url": g["source_url"]} for g in ga if g["source_id"]]
    engineered = build_engineered(m5, m3, gate_a=gate_a,
                                  relations={"edges": rel}, usage=usage)
    engineered["audit"]["gate_a_all_candidates"] = ga
    engineered["audit"]["gate_c"]["detail"] = [g["detail"] for g in gc]
    engineered["usage_note"] = ("usage covers only synthesis+revision answers recorded in vera_vjay.answers; "
                                "M2-M4 spend (search, Gate A/B/C, relations) must be added from their ledgers.")
    return {"run_id": rid, "question": run["question"], "engineered": engineered,
            "corpus": build_corpus(m3, freeze)}


def merge_usage(engineered: dict, extra: dict) -> None:
    """Add M2-M4 totals (search, Gate A/B/C, relations; from their ledgers) to the engineered
    usage and mark it complete. `extra` = {api_calls, tokens, cost_usd, latency_s}."""
    u = engineered["usage"]
    for k in ("api_calls", "tokens", "cost_usd", "latency_s"):
        u[k] = (u.get(k) or 0) + (extra.get(k) or 0)
    u["complete"] = True
