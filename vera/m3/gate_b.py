"""Gate B -- evidence admission (admit / qualify / reject).

Policy (version gate-b-v1), evaluated in this order; first match wins:

  REJECT   no evidence spans extracted
           OR overall_quality is None (nothing assessable)
           OR overall_quality < 2.5
           OR risk_of_bias == 1 (very serious)       -- cannot be trusted at all
           OR indirectness == 1 (does not address the question)
  ADMIT    overall_quality >= 3.0
           AND risk_of_bias, indirectness, applicability each KNOWN and >= 3
           AND known_weight >= 0.7
  QUALIFY  everything else (2.5 <= overall < 3.0, or a critical domain is
           2 or unknown, or too little of the rubric was assessable). Qualified
           evidence may inform reasoning but must carry its stated limitations.

Inconsistency alone never rejects: on a contested question disagreement between
sources is the finding, not a defect (see appraisal_rubric). Unknown is distinct
from low: an unknown critical domain blocks ADMIT but is not itself grounds to
REJECT. `quarantine`/`excise-span-and-admit` (design's full set) are out of
scope for the MVP, which requires only admit/qualify/reject.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from vera.m3.appraisal_rubric import Appraisal, RubricEngine

POLICY_VERSION = "gate-b-v1"
CRITICAL = ("risk_of_bias", "indirectness", "applicability")


@dataclass(frozen=True)
class GateBPolicy:
    admit_min: float = 3.0
    reject_below: float = 2.5
    critical_admit_min: int = 3
    admit_min_known_weight: float = 0.7
    version: str = POLICY_VERSION


DEFAULT_POLICY = GateBPolicy()


def decide(appraisal: Appraisal, n_spans: int, policy: GateBPolicy = DEFAULT_POLICY) -> tuple[str, list[str]]:
    """Pure decision function. Returns (decision, reasons)."""
    q = appraisal.overall_quality
    if n_spans == 0:
        return "reject", ["no evidence spans could be extracted from the source"]
    if q is None:
        return "reject", ["no rubric domain could be assessed"]
    if q < policy.reject_below:
        return "reject", [f"overall_quality {q} < reject threshold {policy.reject_below}"]
    if appraisal.score("risk_of_bias") == 1:
        return "reject", ["risk_of_bias = 1 (very serious): methods cannot support reliance"]
    if appraisal.score("indirectness") == 1:
        return "reject", ["indirectness = 1 (very serious): source does not address the research question"]

    blockers = []
    if q < policy.admit_min:
        blockers.append(f"overall_quality {q} < admit threshold {policy.admit_min}")
    for d in CRITICAL:
        s = appraisal.score(d)
        if s is None:
            blockers.append(f"{d} unknown")
        elif s < policy.critical_admit_min:
            blockers.append(f"{d} = {s} < {policy.critical_admit_min}")
    if appraisal.known_weight < policy.admit_min_known_weight:
        blockers.append(f"only {appraisal.known_weight:.0%} of rubric weight assessable (< {policy.admit_min_known_weight:.0%})")
    if blockers:
        return "qualify", blockers
    return "admit", [f"overall_quality {q} >= {policy.admit_min}; risk_of_bias, indirectness, applicability all >= {policy.critical_admit_min}"]


def evaluate_source(source: dict, question: str, rubric: RubricEngine, spans: list[dict] | None = None,
                    other_evidence: list[dict] | None = None, policy: GateBPolicy = DEFAULT_POLICY) -> dict:
    """Appraise one source and apply Gate B.

    `spans` defaults to source["spans"]. The returned dict is self-contained for the
    M7 demo: decision, reasons, per-domain scores + rationales, thresholds, versions.
    """
    spans = spans if spans is not None else source.get("spans", [])
    appraisal = rubric.appraise(source, question, spans, other_evidence)
    decision, reasons = decide(appraisal, len(spans), policy)
    rationale = f"Gate B {decision.upper()}: " + "; ".join(reasons) + ". Appraisal: " + appraisal.rationale_text()
    return {
        "source_id": source.get("source_id"),
        "decision": decision,
        "reasons": reasons,
        "overall_quality": appraisal.overall_quality,
        "scores": {d: appraisal.score(d) for d in appraisal.domains},
        "rationale": rationale,
        "appraisal": appraisal.to_dict(),
        "policy": asdict(policy),
    }
