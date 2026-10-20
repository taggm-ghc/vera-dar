"""End-to-end M4 -> M5 orchestration: Gate C/relations/context -> synthesis -> claim verification
-> single revision. Persists draft/revised answers, claims with verification status, and the final
response on vera_vjay.runs. Returns the engineered response for M6 evaluation."""
import logging
from typing import Callable

from vera.pipeline_llm import LLMCallable
from vera.m4.m4_runner import run_m4
from vera.m5.claim_verification import verify_claims
from vera.m5.revision import revise_response_detailed
from vera.m5.synthesis import synthesize_response

logger = logging.getLogger("vera")


def _surviving_claims(claims: list[dict], flagged: list[dict], rev: dict) -> list[dict]:
    """Drop removed claims; replace qualified/toned-down claim text. Evidence pointers are kept."""
    acts = {a.get("claim_index"): a for a in rev["actions"]}
    flagged_idx = {f["claim_index"] for f in flagged}
    out = []
    for i, c in enumerate(claims):
        if i in flagged_idx:
            a = acts.get(i, {})
            if a.get("action") == "removed" or not a:
                continue
            c = {**c, "claim_text": a.get("new_claim_text") or c["claim_text"]}
        out.append({k: v for k, v in c.items() if k not in ("verification_status", "issues")})
    return out


def run_m4_m5(run_id: str, question: str, evidence_corpus: dict, *,
              research_fn: Callable[[list[dict]], dict] | None = None,
              llm: LLMCallable | None = None, store=None, max_searches: int = 2) -> dict:
    m4 = run_m4(run_id, question, evidence_corpus, research_fn=research_fn, llm=llm, store=store,
                max_searches=max_searches)
    corpus = m4["evidence_corpus"]

    draft = synthesize_response(question, m4["reasoning_context"], llm=llm)
    ver = verify_claims(draft, corpus, relations=m4["relations"], llm=llm)
    vresp = ver["verified_response"]

    draft_id = None
    if store:
        draft_id = store.save_answer(run_id, "draft", draft["response_text"], model=draft["model"],
                                     tokens=draft["tokens"], cost=draft["cost_usd"], latency=draft["latency_s"])
        store.save_claims(draft_id, vresp["claims"])

    final_text, final_id, final_claims = draft["response_text"], draft_id, vresp["claims"]
    status, revised, rev = ver["verification_status"], False, None
    if ver["needs_revision"]:
        rev = revise_response_detailed(draft, ver["unsupported_claims"], llm=llm)  # ONE pass, never iterated
        if rev["revised_text"]:
            revised, final_text = True, rev["revised_text"]
            kept = _surviving_claims(vresp["claims"], ver["unsupported_claims"], rev)
            # Verify (do not re-revise) the revised answer so its stored statuses are real, not assumed.
            post = verify_claims({**draft, "response_text": final_text, "claims": kept}, corpus,
                                 relations=m4["relations"], llm=llm)
            final_claims = post["verified_response"]["claims"]
            status = "revised_verified" if not post["needs_revision"] and post["verification_status"] == "verified" \
                else "revised_residual_flags"
            if store:
                final_id = store.save_answer(run_id, "revised", final_text, model=rev["model"], tokens=rev["tokens"],
                                             cost=rev["cost_usd"], latency=rev["latency_s"], parent_answer_id=draft_id)
                store.save_claims(final_id, final_claims)
        else:
            status = "needs_revision_failed"
            logger.error("run %s: revision returned empty text; keeping draft with flagged claims", run_id)

    if store:
        store.finalize_run(run_id, final_text, final_id, status, m4["gate_c"]["decision"])

    cost = draft["cost_usd"] + (rev["cost_usd"] if rev else 0.0)
    return {"run_id": run_id, "final_response": final_text, "final_answer_id": final_id,
            "draft_answer_id": draft_id, "claims": final_claims, "verification_status": status,
            "revised": revised, "unsupported_claims": ver["unsupported_claims"],
            "gate_c_decision": m4["gate_c"]["decision"], "gate_c_trace": m4["gate_c_trace"],
            "reasoning_context": m4["reasoning_context"],
            "cost_usd_synthesis_revision": round(cost, 6)}
