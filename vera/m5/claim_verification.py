"""Claim verification: the research-informed mitigation for 'legitimacy laundering' (claims that
look supported but rest on weak, outdated, contested or selectively used evidence).

Per claim: (1) deterministic checks - citations resolve to corpus spans, numbers in the claim appear
in the cited text, minimal lexical grounding; (2) LLM entailment check (batched) for scope/strength/
direction over-inference; (3) weakness flags - old or low-appraisal-only support, and cited evidence
that is refuted elsewhere while the response never acknowledges disagreement.

Statuses: supported | overreach | unsupported | weak | contested | unverified.
Fails closed on verbosity, not silently: if the LLM check is unavailable, claims passing the
deterministic checks are 'unverified' (never 'supported') and the result says so.
"""
import copy
import logging
import re

from vera.pipeline_llm import UNTRUSTED_NOTE, LLMCallable, LLMError, call

logger = logging.getLogger("vera")

MIN_YEAR = 2023
MIN_APPRAISAL = 0.4
MIN_OVERLAP = 0.2
MAX_CLAIMS = 40
HEDGES = ("however", "but ", "although", "whereas", "in contrast", "disagree", "conflict", "mixed",
          "contradict", "other studies", "not all", "differ")
STOP = set("that this with from have been were their which about also such than more into over under "
           "while these those other using used when what they them will would could should".split())
ACTION = {"unsupported": "remove", "overreach": "tone_down", "weak": "qualify", "contested": "qualify"}
NUM = re.compile(r"\d+(?:\.\d+)?")


def _words(t: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP}


def _deterministic(claim: dict, spans: dict) -> list[str]:
    cited = [spans[str(i)] for i in claim.get("evidence_span_ids", []) if str(i) in spans]
    if not cited:
        return ["no_valid_citation"]
    ctext = " ".join(s["text"] for s in cited)
    issues = []
    missing_nums = [n for n in NUM.findall(claim["claim_text"]) if n not in ctext]
    if missing_nums:
        issues.append(f"numbers_not_in_cited_evidence:{','.join(missing_nums)}")
    cw = _words(claim["claim_text"])
    if cw and len(cw & _words(ctext)) / len(cw) < MIN_OVERLAP:
        issues.append("no_lexical_grounding")
    return issues


def _llm_check(claims: list[dict], spans: dict, llm) -> dict[int, dict]:
    items = []
    for i, c in enumerate(claims):
        ev = "\n".join(f"  ({sid}) {spans[sid]['text'][:500]}" for sid in map(str, c.get("evidence_span_ids", [])) if sid in spans)
        items.append(f"#{i}: {c['claim_text']}\n  cited evidence:\n{ev or '  (none)'}")
    system = (
        "You are a strict fact-checker. For each claim decide whether the CITED evidence alone entails it. "
        "'supported' = entailed in direction, magnitude, population and scope. 'partial' = evidence points "
        "the same way but the claim overstates (stronger, broader, causal, or more general than shown). "
        "'unsupported' = not entailed or contradicted. " + UNTRUSTED_NOTE +
        ' Reply JSON: {"verdicts":[{"index":int,"verdict":"supported|partial|unsupported","issue":"short"}]}'
    )
    res = call(llm, system, "<evidence>\n" + "\n\n".join(items) + "\n</evidence>")
    out = {}
    for v in res.data.get("verdicts", []):
        try:
            out[int(v["index"])] = {"verdict": str(v["verdict"]).lower(), "issue": str(v.get("issue", ""))}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def verify_claims(response: dict, evidence_corpus: dict, *, relations: dict | None = None,
                  llm: LLMCallable | None = None, use_llm: bool = True) -> dict:
    spans = {str(s["span_id"]): s for s in evidence_corpus.get("spans", [])}
    verified = copy.deepcopy(response)
    claims = verified.get("claims", [])[:MAX_CLAIMS]
    verified["claims"] = claims
    text_l = verified.get("response_text", "").lower()
    acknowledges = any(h in text_l for h in HEDGES)
    refuted = set()
    for e in (relations or {}).get("edges", []):
        if e["relation_type"] == "refutes" and e["confidence"] >= 0.5:
            refuted.update((str(e["span1_id"]), str(e["span2_id"])))

    verdicts, llm_ok = {}, False
    if use_llm and claims:
        try:
            verdicts, llm_ok = _llm_check(claims, spans, llm), True
        except LLMError as exc:
            logger.error("claim verification LLM check FAILED (claims will be 'unverified'): %s", exc)

    flagged = []
    for i, c in enumerate(claims):
        issues = _deterministic(c, spans)
        status = "unsupported" if issues else None
        v = verdicts.get(i)
        if v and v["verdict"] == "unsupported":
            status, issues = "unsupported", issues + [f"llm:{v['issue']}"]
        elif v and v["verdict"] == "partial" and status is None:
            status, issues = "overreach", issues + [f"llm:{v['issue']}"]
        if status is None:
            cited = [spans[str(i2)] for i2 in c["evidence_span_ids"] if str(i2) in spans]
            old = all(isinstance(s.get("year"), int) and s["year"] < MIN_YEAR for s in cited)
            low = all(isinstance(s.get("appraisal", {}).get("overall"), (int, float))
                      and s["appraisal"]["overall"] < MIN_APPRAISAL for s in cited)
            if old or low:
                status, issues = "weak", ["only_outdated_evidence" if old else "only_low_appraisal_evidence"]
            elif refuted & {str(x) for x in c["evidence_span_ids"]} and not acknowledges:
                status, issues = "contested", ["cited_evidence_is_contradicted_elsewhere_but_response_acknowledges_no_disagreement"]
        if status is None:
            status = "supported" if llm_ok and i in verdicts else "unverified"
            if status == "unverified":
                issues = ["llm_verification_unavailable" if not llm_ok else "no_llm_verdict_for_claim"]
        c["verification_status"], c["issues"] = status, issues
        if status in ACTION:
            flagged.append({"claim_index": i, "claim_text": c["claim_text"], "status": status, "issues": issues,
                            "evidence_span_ids": c.get("evidence_span_ids", []), "recommended_action": ACTION[status]})

    n = len(claims)
    counts = {s: sum(1 for c in claims if c["verification_status"] == s)
              for s in ("supported", "overreach", "unsupported", "weak", "contested", "unverified")}
    if not n:
        overall = "no_claims"
    elif flagged:
        overall = "needs_revision"
    elif counts["unverified"]:
        overall = "unverified"
    else:
        overall = "verified"
    return {"verified_response": verified, "unsupported_claims": flagged, "needs_revision": bool(flagged),
            "verification_status": overall, "summary": {"total": n, **counts}}
