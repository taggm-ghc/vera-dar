"""Gate C: evidence-set adequacy (coverage, independence, currency, missingness).

Deterministic checks decide coverage/independence/currency. An optional LLM
missingness pass proposes material gaps; those can trigger search_again while
search budget remains but never block synthesis once the cap is reached.
Returns one of: adequate / search_again / insufficient.
"""
import logging

from vera.pipeline_llm import UNTRUSTED_NOTE, LLMCallable, LLMError, call
from vera.m4.common import independence_key, sub_questions

logger = logging.getLogger("vera")

MIN_INDEPENDENT_SOURCES = 2   # a sub-question is 'covered' with >= this many independent groups
MIN_YEAR = 2023               # question scope is 2023-2026 evidence
INSUFFICIENT_BELOW = 0.5      # at cap: < this share of required sub-questions with any evidence -> insufficient
MAX_LLM_GAPS = 3


def _missingness_llm(corpus: dict, sqs: list[dict], llm: LLMCallable | None) -> list[dict]:
    lines = [f"- {s['span_id']} ({s.get('source_type', '?')}, {s.get('year', '?')}): {s['text'][:160]}"
             for s in corpus.get("spans", [])[:30]]
    system = (
        "You audit an evidence set for a research question. List at most 3 MATERIAL missing evidence "
        "types that could change the answer (not nitpicks). " + UNTRUSTED_NOTE +
        ' Reply JSON: {"gaps":[{"sub_question_id":str,"need":str,"evidence_type":str}]}. Empty list if none.'
    )
    user = (f"Question: {corpus.get('question', '')}\nSub-questions: "
            + "; ".join(f"{s['id']}={s['text']}" for s in sqs)
            + "\n<evidence>\n" + "\n".join(lines) + "\n</evidence>")
    try:
        gaps = call(llm, system, user).data.get("gaps", [])
    except LLMError as exc:
        logger.warning("Gate C missingness LLM pass failed (skipped, deterministic result stands): %s", exc)
        return []
    out = []
    for g in gaps[:MAX_LLM_GAPS]:
        if isinstance(g, dict) and g.get("need"):
            out.append({"sub_question_id": g.get("sub_question_id"), "need": str(g["need"]),
                        "evidence_type": str(g.get("evidence_type", "unspecified")),
                        "reason": "llm_missingness"})
    return out


def evaluate_adequacy(evidence_corpus: dict, question: str, *, searches_done: int = 0,
                      max_searches: int = 2, llm: LLMCallable | None = None,
                      use_llm_missingness: bool = False) -> dict:
    corpus = dict(evidence_corpus)
    corpus.setdefault("question", question)
    sqs = sub_questions(corpus)
    spans = corpus.get("spans", [])

    coverage, gaps = {}, []
    for sq in sqs:
        mine = [s for s in spans if sq["id"] in (s.get("sub_question_ids") or [])]
        groups = {independence_key(s) for s in mine}
        years = [s["year"] for s in mine if isinstance(s.get("year"), int)]
        outdated = bool(years) and max(years) < MIN_YEAR
        if not mine:
            status = "missing"
        elif len(groups) < MIN_INDEPENDENT_SOURCES:
            status = "thin"
        elif outdated:
            status = "outdated"
        else:
            status = "covered"
        coverage[sq["id"]] = {"status": status, "span_count": len(mine),
                              "independent_sources": len(groups), "required": sq["required"]}
        if sq["required"] and status != "covered":
            need = {"missing": "no admitted evidence for this sub-question",
                    "thin": f"only {len(groups)} independent source(s); need >= {MIN_INDEPENDENT_SOURCES}",
                    "outdated": f"all evidence predates {MIN_YEAR}"}[status]
            gaps.append({"sub_question_id": sq["id"], "need": need, "evidence_type": sq["text"],
                         "reason": status})

    required = [sq for sq in sqs if sq["required"]]
    with_any = [sq for sq in required if coverage[sq["id"]]["span_count"] > 0]
    coverage_ratio = (sum(1 for sq in required if coverage[sq["id"]]["status"] == "covered") / len(required)
                      if required else 1.0)
    any_ratio = len(with_any) / len(required) if required else 1.0

    llm_gaps = (_missingness_llm(corpus, sqs, llm)
                if (use_llm_missingness or llm is not None) and spans else [])
    budget_left = searches_done < max_searches
    reasons = []

    if (gaps or llm_gaps) and budget_left:
        decision = "search_again"
        reasons.append(f"{len(gaps)} deterministic and {len(llm_gaps)} LLM-proposed gap(s); "
                       f"search {searches_done + 1}/{max_searches} available")
        missing = gaps + llm_gaps
        residual, qualified = [], False
    else:
        missing, residual = [], gaps + llm_gaps
        if not spans or (required and any_ratio < INSUFFICIENT_BELOW):
            decision, qualified = "insufficient", False
            reasons.append(f"only {any_ratio:.0%} of required sub-questions have any evidence at search cap")
        else:
            decision, qualified = "adequate", bool(gaps)
            reasons.append("evidence set adequate" + (" with qualifications (unresolved gaps recorded)" if gaps else ""))

    return {"decision": decision, "qualified": qualified, "coverage": coverage,
            "coverage_ratio": round(coverage_ratio, 3), "missing_evidence": missing,
            "residual_gaps": residual, "searches_done": searches_done, "max_searches": max_searches,
            "reasons": reasons}
