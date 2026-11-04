"""NONASK-1103 E2: Gate C, the deterministic adequacy gate. Pure, stdlib only, no model call, no new threshold.

Requirements are the writer-declared parts only (fulldoc_answer.parse_parts, existing max_parts). A part is supported
when it has at least one shown, checked supported claim (the existing D8 rule). The record returned is separate from the
claim records (it holds no claim text or ids) and decides nothing about which sentences are shown.

  finalize  every declared part supported and no responsiveness-partial (identical to answer_completeness "complete")
  qualify   at least one part supported, but some part unsupported or the shown claims are responsiveness-partial
  abstain   no part supported (no answer)
"""
from __future__ import annotations

FINALIZE, QUALIFY, ABSTAIN = "finalize", "qualify", "abstain"


def _one(declared, supported, labels, resp_partial) -> dict:
    """declared: iterable of declared part keys; supported: iterable of keys with >= 1 shown supported claim;
    labels: optional {key: short text}; resp_partial: the existing fulldoc_answer._resp_partial result."""
    keys = list(dict.fromkeys(declared or ()))
    sup = set(supported or ())
    lab = labels or {}
    rows = [{"part": k, "label": str(lab.get(k, "")), "supported": k in sup} for k in keys]
    n_sup = sum(1 for r in rows if r["supported"])
    all_sup = bool(keys) and n_sup == len(keys)
    if n_sup == 0:
        verdict = ABSTAIN
    elif all_sup and not resp_partial:
        verdict = FINALIZE
    else:
        verdict = QUALIFY
    return {"verdict": verdict, "all_declared_parts_supported": all_sup, "resp_partial": bool(resp_partial),
            "parts": rows}


def evaluate(declared, supported, labels=None, resp_partial: bool = False, subqueries=None) -> dict:
    """As before for the original question. `subqueries` (optional, from vera.subquery parts) is an iterable of
    {"label": str, "declared": [...], "supported": [...], "resp_partial": bool}; each is judged by the same rule
    (coverage of declared parts, no new threshold). With 2+ sub-queries the overall verdict is over the sub-queries:
    finalize all finalize, abstain none supported, else qualify; each sub-query reports an honest `not_found`.
    With 0 or 1 sub-query the record is identical to today's (no "subqueries" key)."""
    rec = _one(declared, supported, labels, resp_partial)
    subs = [s for s in (subqueries or ()) if isinstance(s, dict)]
    if len(subs) < 2:
        return rec
    out = []
    for s in subs:
        r = _one(s.get("declared") or (), s.get("supported") or (), s.get("labels"), bool(s.get("resp_partial")))
        out.append({"label": str(s.get("label", "")), "verdict": r["verdict"], "not_found": r["verdict"] == ABSTAIN,
                    "all_declared_parts_supported": r["all_declared_parts_supported"], "parts": r["parts"]})
    n_found = sum(1 for o in out if not o["not_found"])
    if n_found == 0:
        verdict = ABSTAIN
    elif all(o["verdict"] == FINALIZE for o in out) and rec["verdict"] != QUALIFY:
        verdict = FINALIZE
    else:
        verdict = QUALIFY
    rec.update(verdict=verdict, subqueries=out, all_declared_parts_supported=all(o["all_declared_parts_supported"] for o in out))
    return rec


def completeness(record: dict):
    """The answer_contract.answer_status completeness signal: 'complete' for finalize, 'partial' for qualify, else None."""
    v = (record or {}).get("verdict")
    return "complete" if v == FINALIZE else "partial" if v == QUALIFY else None
