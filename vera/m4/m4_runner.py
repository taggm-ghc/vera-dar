"""M4 orchestration: Gate C -> (bounded M2 re-search) -> relations -> reasoning context.

`research_fn(missing_evidence: list[dict]) -> dict` is the M2/M3 hook: it must return
{"spans": [...], optional "sub_questions": [...]} to merge into the corpus. The loop is
bounded by max_searches (default 2) plus a hard step counter; failures are logged verbosely
and the pipeline proceeds to the capped decision rather than looping.
"""
import logging
from typing import Callable

from vera.pipeline_llm import LLMCallable
from vera.m4.context_builder import build_context_with_index
from vera.m4.evidence_relations import build_relations
from vera.m4.gate_c import evaluate_adequacy

logger = logging.getLogger("vera")
DEFAULT_MAX_SEARCHES = 2


def _merge(corpus: dict, new: dict) -> int:
    have = {s["span_id"] for s in corpus["spans"]}
    added = [s for s in new.get("spans", []) if s.get("span_id") not in have]
    corpus["spans"] = corpus["spans"] + added
    return len(added)


def run_m4(run_id: str, question: str, evidence_corpus: dict, *,
           research_fn: Callable[[list[dict]], dict] | None = None,
           llm: LLMCallable | None = None, store=None,
           max_searches: int = DEFAULT_MAX_SEARCHES) -> dict:
    corpus = {**evidence_corpus, "run_id": run_id, "question": question,
              "spans": list(evidence_corpus.get("spans", []))}
    searches, trace = 0, []
    for step in range(max_searches + 2):  # hard bound on iterations
        gate = evaluate_adequacy(corpus, question, searches_done=searches, max_searches=max_searches, llm=llm)
        if store:
            store.save_gate_c(run_id, step, gate)
        trace.append(gate["decision"])
        if gate["decision"] != "search_again":
            break
        if research_fn is None:
            logger.warning("run %s: Gate C wants search_again but no research_fn supplied; "
                           "re-evaluating at cap. Missing: %s", run_id, gate["missing_evidence"])
            searches = max_searches
            continue
        try:
            added = _merge(corpus, research_fn(gate["missing_evidence"]))
        except Exception:
            logger.exception("run %s: research_fn failed on re-search %d; stopping re-search", run_id, searches + 1)
            searches = max_searches
            continue
        searches += 1
        logger.info("run %s: re-search %d/%d added %d span(s)", run_id, searches, max_searches, added)
    else:  # pragma: no cover - unreachable by construction
        raise RuntimeError(f"run {run_id}: Gate C loop exceeded its hard step bound; trace={trace}")

    graph = build_relations(corpus, llm=llm)
    if store:
        store.save_relations(run_id, graph)
    context, index = build_context_with_index(corpus, graph, gate)
    if store:
        store.save_reasoning_context(run_id, {"text": context, "index": index, "gate_c": gate,
                                              "gate_c_trace": trace})
    return {"gate_c": gate, "gate_c_trace": trace, "relations": graph, "reasoning_context": context,
            "context_index": index, "evidence_corpus": corpus}
