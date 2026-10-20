"""Grounded synthesis: the reasoning context is the ONLY permitted knowledge source.
Every claim carries [E#] labels that map back to span_ids parsed from the context."""
import re

from vera.pipeline_llm import UNTRUSTED_NOTE, LLMCallable, call

LABEL_LINE = re.compile(r"^\[(E\d+)\] span_id=(\S+)", re.M)


def parse_label_map(reasoning_context: str) -> dict[str, str]:
    return {lab: sid for lab, sid in LABEL_LINE.findall(reasoning_context)}


def synthesize_response(question: str, reasoning_context: str, *, llm: LLMCallable | None = None) -> dict:
    label_map = parse_label_map(reasoning_context)
    system = (
        "You write a research answer using ONLY the reasoning context. No outside knowledge, no facts, "
        "numbers or studies that are not in the context. Cite evidence units inline as [E#] after each "
        "claim. Preserve the strength of the evidence: report contradictions (REFUTES) and qualifications "
        "explicitly, do not overstate causal or general claims, name unresolved gaps, and if Gate C says "
        "'insufficient' say plainly that evidence is insufficient and only report what the context supports. "
        + UNTRUSTED_NOTE +
        ' Reply JSON: {"response_text": str (with inline [E#] markers), "claims": [{"claim_text": str '
        '(one atomic assertion), "evidence_labels": ["E1", ...]}]}. Every factual sentence in response_text '
        "must correspond to a claim."
    )
    user = f"Question: {question}\n<context>\n{reasoning_context}\n</context>"
    res = call(llm, system, user)
    claims, invalid = [], []
    for c in res.data.get("claims", []):
        if not isinstance(c, dict) or not str(c.get("claim_text", "")).strip():
            continue
        labs = [str(x).strip("[] ") for x in c.get("evidence_labels", []) if isinstance(x, (str, int))]
        good = [l for l in dict.fromkeys(labs) if l in label_map]
        invalid += [l for l in labs if l not in label_map]
        claims.append({"claim_text": c["claim_text"].strip(), "evidence_labels": good,
                       "evidence_span_ids": [label_map[l] for l in good]})
    return {
        "response_text": str(res.data.get("response_text", "")).strip(),
        "claims": claims,
        "evidence_citations": [{"claim_text": c["claim_text"], "span_ids": c["evidence_span_ids"]} for c in claims],
        "invalid_citations": invalid,
        "model": res.model, "tokens": res.tokens, "cost_usd": res.cost_usd, "latency_s": res.latency_s,
    }
