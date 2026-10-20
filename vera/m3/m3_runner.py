"""M3 orchestrator: M2 sources -> spans -> appraisal -> Gate B -> evidence_corpus.

    python -m vera.m3.m3_runner [--run-id X] [--dry-run] [--max-cost 1.0] [--out corpus.json]

Two passes so inconsistency can be judged against the rest of the corpus:
  1. extract spans for every source;
  2. appraise each source with a digest of the other sources' key spans, apply Gate B, persist.
Bounded loop: max_sources, max_cost_usd. Hitting a limit or any per-source failure stops that
work and is reported in corpus["failures"] / corpus["stopped_early"] -- never silently.
Writes go to vera_vjay.* in the DB unless dry_run=True (production Postgres writes approved
by R1, 2026-09-30).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from vera.m3.appraisal_rubric import RUBRIC_VERSION, RubricEngine
from vera.m3.evidence_extractor import extract_spans_detailed
from vera.m3.gate_b import DEFAULT_POLICY, GateBPolicy, evaluate_source
from vera.m3.llm import CallLog, LLMClient, LLMError

logger = logging.getLogger("vera.m3")

RESEARCH_QUESTION = ("What does published evidence (2023-2026) show about AI coding assistants' effect on "
                     "developer productivity, and why do the findings disagree?")
CORPUS_SCHEMA_VERSION = "m3-evidence-corpus-v1"
DIGEST_SPANS_PER_SOURCE = 2
DIGEST_SPAN_CHARS = 220


def _digest_for(others: list[dict]) -> list[dict]:
    out = []
    for o in others:
        keep = [s for s in o["spans"] if s["evidence_type"] in ("fact", "counterevidence")]
        keep = sorted(keep, key=lambda s: -s["relevance_score"])[:DIGEST_SPANS_PER_SOURCE]
        if keep:
            out.append({"source": o["source"].get("title") or o["source"]["source_id"],
                        "findings": [s["text"][:DIGEST_SPAN_CHARS] for s in keep]})
    return out


def run_m3(sources: list[dict], question: str = RESEARCH_QUESTION, llm: LLMClient | None = None,
           engine=None, dry_run: bool = False, policy: GateBPolicy = DEFAULT_POLICY,
           max_sources: int = 50, max_cost_usd: float = 1.0) -> dict:
    """Run M3 over already-loaded sources and return the evidence_corpus dict (JSON-safe)."""
    if llm is None:
        from vera.m3.llm import OpenAIJSONClient

        llm = OpenAIJSONClient()
    log: CallLog = llm.log
    rubric = RubricEngine(llm)
    failures: list[dict] = []
    stopped_early = None
    persist = engine is not None and not dry_run

    def over_budget() -> bool:
        return log.total_cost_usd >= max_cost_usd

    # Pass 1: extraction
    extracted: list[dict] = []
    for src in sources[:max_sources]:
        if over_budget():
            stopped_early = f"cost limit ${max_cost_usd} reached during extraction (spent ${log.total_cost_usd})"
            break
        sid = src["source_id"]
        try:
            res = extract_spans_detailed(src.get("content", ""), question, llm)
        except LLMError as exc:
            failures.append({"source_id": sid, "stage": "extract", "error": str(exc)})
            continue
        extracted.append({"source": src, "spans": res.spans, "extraction": res})
    if len(sources) > max_sources:
        stopped_early = stopped_early or f"max_sources {max_sources} reached; {len(sources) - max_sources} sources not processed"

    # Pass 2: appraise + Gate B + persist
    included, excluded = [], []
    for item in extracted:
        src, spans, ext = item["source"], item["spans"], item["extraction"]
        sid = src["source_id"]
        if over_budget():
            stopped_early = stopped_early or f"cost limit ${max_cost_usd} reached during appraisal (spent ${log.total_cost_usd})"
            failures.append({"source_id": sid, "stage": "appraise", "error": "skipped: cost limit reached"})
            continue
        others = _digest_for([o for o in extracted if o is not item])
        try:
            gate = evaluate_source(src, question, rubric, spans=spans, other_evidence=others, policy=policy)
        except LLMError as exc:
            failures.append({"source_id": sid, "stage": "appraise", "error": str(exc)})
            continue
        span_ids: list[int | None] = [None] * len(spans)
        if persist:
            try:
                from vera.m3.store import save_source_result

                span_ids = save_source_result(engine, sid, spans, gate)
            except Exception as exc:  # DB failure: report, keep going with in-memory result
                failures.append({"source_id": sid, "stage": "persist", "error": repr(exc)})
        summary = {
            "source_id": sid, "title": src.get("title"), "url": src.get("url"),
            "content_hash": src.get("content_hash"),
            "decision": gate["decision"], "overall_quality": gate["overall_quality"],
            "scores": gate["scores"], "rationale": gate["rationale"], "reasons": gate["reasons"],
            "extraction": {"dropped_quotes": len(ext.dropped), "truncated": ext.truncated},
        }
        if gate["decision"] == "reject":
            excluded.append(summary)
        else:
            summary["qualified"] = gate["decision"] == "qualify"
            summary["spans"] = [dict(s, span_id=sid_) for s, sid_ in zip(spans, span_ids)]
            included.append(summary)

    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "rubric_version": RUBRIC_VERSION,
        "gate_b_policy": gate_policy_dict(policy),
        "persisted": persist,
        "counts": {"sources_in": len(sources), "admit": sum(1 for s in included if not s["qualified"]),
                   "qualify": sum(1 for s in included if s["qualified"]), "reject": len(excluded),
                   "failed": len({f["source_id"] for f in failures})},
        "sources": included,      # admitted + qualified: eligible for M4 context
        "excluded": excluded,     # rejected: kept for auditability, not for reasoning
        "failures": failures,
        "stopped_early": stopped_early,
        "llm_cost": log.summary(),
    }


def gate_policy_dict(p: GateBPolicy) -> dict:
    from dataclasses import asdict

    return asdict(p)


def run_from_db(run_id: int | None = None, dry_run: bool = False, **kw) -> dict:
    from vera.db import get_engine
    from vera.m3.store import load_sources

    engine = get_engine()
    sources = load_sources(engine, run_id=run_id)
    return run_m3(sources, engine=engine, dry_run=dry_run, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--dry-run", action="store_true", help="do not write to the database")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--max-sources", type=int, default=50)
    ap.add_argument("--out", default="-")
    a = ap.parse_args(argv)
    from vera.env_files import load_vera_env

    load_vera_env()  # PLAN-ENV: explicit R/config/env loader
    logging.basicConfig(level=logging.INFO)
    try:
        corpus = run_from_db(a.run_id, a.dry_run, max_cost_usd=a.max_cost, max_sources=a.max_sources)
    except Exception as exc:
        print(f"M3 FAILED: {exc!r}", file=sys.stderr)
        return 1
    text = json.dumps(corpus, indent=2, ensure_ascii=False, default=str)
    if a.out == "-":
        print(text)
    else:
        with open(a.out, "w") as f:
            f.write(text)
    return 0 if not corpus["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
