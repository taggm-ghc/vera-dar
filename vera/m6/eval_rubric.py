"""M6 five-dimension rubric scorers.

Data contract (what M5/M3 must hand to M6; verified absent at time of writing,
so tests use fixtures):

answer (engineered):
  {"response_text": str,
   "claims": [{"id", "text", "citations": [span_id, ...], "consequential": bool}],
   "usage": {"api_calls", "tokens", "cost_usd", "latency_s"},
   "audit": {"gate_a": [{"source_id", "score", "rationale", "selected"}],
             "gate_b": [{"span_id", "decision", "rubric": {...}, "rationale",
                         "rubric_version", "model"}],
             "gate_c": {...}, "relations": [...]}}
answer (baseline): {"response_text", "usage"}  -- free text only.
corpus (frozen):
  {"requirements": [{"id","text"}], "objections": [{"id","text"}],
   "spans": {span_id: {"text","source_id","start","end"}},
   "sources": {source_id: {"url","title","content_hash","version","aliases":[...]}}}

Deterministic where possible (coverage math, fabricated-id detection, audit
trace, cost ratio); the LLM judge is used only for semantic labels.
Every scorer returns {"score": 1-5, "pass_metrics": {...}, "rationale": str, ...}.
"""
from __future__ import annotations

import json
import random
import re

from vera.eval_config import requirements_hash16
from vera.m6.judge import Judge

# Pass thresholds are NOT defined here: they are read from config/vera_eval_freeze.json (success_criteria)
# and config/vera_eval_run.json (fabrication rule) through vera.eval_config (goal #2, one source).
# Why: spec says the auditability sample is 3-5 claims; 5 is the upper bound and the documented default.
# Recorded in the run fingerprint (scoring.audit_k).
AUDIT_K = 5
CLAIM_LABELS = ("supported", "partial", "unsupported", "uncited", "fabricated-citation")
OBJ_LABELS = ("ignored", "mentioned", "addressed")


def freeze_hash(corpus: dict) -> str:
    """Hash of the frozen requirement/objection lists, recorded in every report."""
    return requirements_hash16(corpus)  # one formula, owned by vera.eval_config (value unchanged)


# ---------------------------------------------------------------- Relevance
def relevance_level(cov: float) -> int:
    return 1 if cov < 0.20 else 2 if cov < 0.40 else 3 if cov < 0.60 else 4 if cov < 0.80 else 5


def score_relevance(answer: dict, corpus: dict, judge: Judge) -> dict:
    reqs = corpus["requirements"]
    if not reqs:
        raise ValueError("Relevance needs a frozen, non-empty requirement list.")
    prompt = (
        "Score how well the ANSWER covers each evidence requirement. "
        "Use 0 (not covered), 0.5 (partly covered), 1 (fully covered). Judge content only.\n"
        'Reply JSON: {"scores": {"<req_id>": {"score": 0|0.5|1, "rationale": "..."}}}\n\n'
        f"REQUIREMENTS:\n{json.dumps(reqs)}\n\nANSWER:\n{answer['response_text']}")
    raw = judge.judge_json("relevance", prompt).get("scores", {})
    items, total = [], 0.0
    for r in reqs:
        s = raw.get(r["id"], {})
        v = s.get("score", 0)
        if v not in (0, 0.5, 1):
            v = 0
        total += v
        items.append({"id": r["id"], "score": v, "rationale": s.get("rationale", "(judge gave none; scored 0)")})
    cov = total / len(reqs)
    return {"score": relevance_level(cov), "coverage": cov, "items": items,
            "rationale": f"{total}/{len(reqs)} requirement points = {cov:.0%} coverage."}


# ---------------------------------------------------------------- Grounding
def grounding_level(rate: float, fabricated: int) -> int:
    lvl = 1 if rate < 0.40 else 2 if rate < 0.60 else 3 if rate < 0.80 else 4 if rate < 0.90 else 5
    return min(lvl, 2) if fabricated else lvl  # fabricated citation = hard-fail cap


_CITE_RE = re.compile(r"https?://\S+|doi:\S+|10\.\d{4,}/\S+|\[[^\]]+\]|\([A-Z][A-Za-z\-]+(?: et al\.)?,? \d{4}\)")


def _resolve_baseline_ref(ref: str, corpus: dict) -> str | None:
    ref_l = ref.lower().rstrip(".,;)")
    for sid, s in corpus.get("sources", {}).items():
        keys = [s.get("url", ""), s.get("title", "")] + list(s.get("aliases", []))
        if any(k and (k.lower() in ref_l or ref_l in k.lower()) for k in keys):
            return sid
    return None


def _extract_claims(answer: dict, judge: Judge) -> list[dict]:
    if answer.get("claims"):
        return answer["claims"]
    raw = judge.judge_json(
        "claim-split",
        'Split the text into atomic factual claims (one checkable assertion each). '
        'Copy any citation/URL/author-year reference attached to a claim into "refs". '
        'Reply JSON: {"claims": [{"text": "...", "refs": ["..."]}]}\n\nTEXT:\n'
        + answer["response_text"]).get("claims", [])
    return [{"id": f"c{i+1}", "text": c["text"], "refs": c.get("refs", [])} for i, c in enumerate(raw)]


def score_grounding(answer: dict, corpus: dict, judge: Judge) -> dict:
    claims = _extract_claims(answer, judge)
    if not claims:
        return {"score": 1, "traceable_rate": 0.0, "fabricated": 0, "claims": [],
                "rationale": "No claims could be extracted; nothing is traceable."}
    structured = bool(answer.get("claims"))
    prelabelled, to_judge = {}, []
    for c in claims:
        if structured:
            cites = c.get("citations", [])
            if not cites:
                prelabelled[c["id"]] = ("uncited", "No citation attached.", [])
                continue
            missing = [x for x in cites if x not in corpus["spans"]]
            if missing:  # deterministic: id not in corpus => fabricated
                prelabelled[c["id"]] = ("fabricated-citation",
                                        f"Cited span id(s) not in corpus: {missing}", cites)
                continue
            to_judge.append((c, [corpus["spans"][x]["text"] for x in cites]))
        else:
            refs = c.get("refs", [])
            if not refs:
                prelabelled[c["id"]] = ("uncited", "No reference given in the text.", [])
                continue
            sids = [_resolve_baseline_ref(r, corpus) for r in refs]
            texts = [sp["text"] for sp in corpus["spans"].values() if sp["source_id"] in sids]
            if not texts:  # unverifiable != proven fabricated; counted unsupported, tallied apart
                prelabelled[c["id"]] = ("unsupported",
                                        "Reference not in frozen corpus (unverifiable, not proven fabricated).", refs)
                continue
            to_judge.append((c, texts))
    verdict = {}
    if to_judge:
        raw = judge.judge_json(
            "claim-verify",
            "For each claim, decide whether the CITED TEXTS support it: supported, partial, or "
            "unsupported. Use only the cited texts, not outside knowledge.\n"
            'Reply JSON: {"verdicts": {"<claim_id>": {"label": "...", "rationale": "..."}}}\n\n'
            + json.dumps([{"id": c["id"], "claim": c["text"], "cited_texts": t} for c, t in to_judge])
        ).get("verdicts", {})
        for c, _ in to_judge:
            v = raw.get(c["id"], {})
            lab = v.get("label", "unsupported")
            verdict[c["id"]] = (lab if lab in ("supported", "partial", "unsupported") else "unsupported",
                                v.get("rationale", "(judge gave none)"), c.get("citations", c.get("refs", [])))
    out, counts = [], {k: 0 for k in CLAIM_LABELS}
    for c in claims:
        lab, why, cites = prelabelled.get(c["id"]) or verdict[c["id"]]
        counts[lab] += 1
        out.append({"id": c["id"], "text": c["text"], "label": lab, "rationale": why, "citations": cites})
    n = len(claims)
    rate = counts["supported"] / n  # strict: partial is NOT counted as traceable
    fab = counts["fabricated-citation"]
    return {"score": grounding_level(rate, fab), "traceable_rate": rate,
            "partial_rate": counts["partial"] / n, "fabricated": fab,
            "label_counts": counts, "claims": out,
            "rationale": f"{counts['supported']}/{n} claims supported ({rate:.0%}, strict: partial not counted); "
                         f"{fab} fabricated citation(s)."}


# -------------------------------------------------------- Reasoning integrity
def reasoning_level(labels: list[str], overclaims: int, uncertainty: bool, contradicts: bool) -> int:
    k = len(labels)
    addressed = sum(l == "addressed" for l in labels)
    mentioned = sum(l in ("mentioned", "addressed") for l in labels)
    if contradicts or mentioned == 0:
        return 1
    if addressed == k and uncertainty and overclaims == 0:
        return 5
    if addressed / k >= 0.75 and overclaims <= 1:
        return 4
    if mentioned / k >= 0.50 and overclaims <= 2:
        return 3
    return 2


def score_reasoning(answer: dict, corpus: dict, judge: Judge) -> dict:
    objs = corpus["objections"]
    if len(objs) < 4:
        raise ValueError(f"Reasoning integrity needs K>=4 frozen objections; got {len(objs)}.")
    raw = judge.judge_json(
        "reasoning",
        "For each OBJECTION, label the ANSWER: ignored (not raised), mentioned (raised but not "
        "engaged with evidence), addressed (engaged with evidence or limits stated). Also count "
        "overclaims (conclusions stronger than the evidence, e.g. causal or universal claims from "
        "limited data), say whether uncertainty is explicitly attached to conclusions, and whether "
        "the answer contradicts the evidence.\n"
        'Reply JSON: {"objections": {"<id>": {"label": "...", "rationale": "..."}}, '
        '"overclaims": int, "overclaim_examples": ["..."], "uncertainty_attached": bool, '
        '"contradicts_evidence": bool}\n\n'
        f"OBJECTIONS:\n{json.dumps(objs)}\n\nANSWER:\n{answer['response_text']}")
    items, labels = [], []
    for o in objs:
        v = raw.get("objections", {}).get(o["id"], {})
        lab = v.get("label", "ignored")
        lab = lab if lab in OBJ_LABELS else "ignored"
        labels.append(lab)
        items.append({"id": o["id"], "label": lab, "rationale": v.get("rationale", "(none)")})
    oc = int(raw.get("overclaims", 0))
    unc, contra = bool(raw.get("uncertainty_attached", False)), bool(raw.get("contradicts_evidence", False))
    addressed = labels.count("addressed")
    return {"score": reasoning_level(labels, oc, unc, contra), "addressed": addressed,
            "mentioned": labels.count("mentioned"), "overclaims": oc, "uncertainty_attached": unc,
            "contradicts_evidence": contra, "overclaim_examples": raw.get("overclaim_examples", []),
            "items": items,
            "rationale": f"{addressed}/{len(objs)} objections addressed, {labels.count('mentioned')} mentioned; "
                         f"{oc} overclaim(s); uncertainty attached: {unc}."}


# ---------------------------------------------------------------- Cost / latency
def cost_level(ratio: float | None) -> int:
    if ratio is None or ratio > 2:
        return 1
    return 2 if ratio > 1 else 3 if ratio > 0.5 else 4 if ratio > 0.25 else 5


def score_cost_latency(answer: dict, budget: dict) -> dict:
    u = answer.get("usage") or {}
    cb, lb = budget.get("cost_usd"), budget.get("latency_s")
    if not cb or not lb or "cost_usd" not in u or "latency_s" not in u:
        return {"score": 1, "usage": u, "budget": budget, "ratio": None,
                "rationale": "No budget limit or no usage data recorded => score 1."}
    if u.get("complete") is False:
        return {"score": 1, "usage": u, "budget": budget, "ratio": None, "unverified": True,
                "rationale": "Engineered usage is INCOMPLETE (M2-M4 spend/latency not supplied), so "
                             "'within budget' cannot be verified; scored 1 rather than assumed."}
    cr, lr = u["cost_usd"] / cb, u["latency_s"] / lb
    ratio = max(cr, lr)  # the binding constraint decides
    return {"score": cost_level(ratio), "usage": u, "budget": budget, "ratio": round(ratio, 4),
            "cost_ratio": round(cr, 4), "latency_ratio": round(lr, 4),
            "rationale": f"${u['cost_usd']:.4f} of ${cb:.2f} ({cr:.0%}); {u['latency_s']:.0f}s of "
                         f"{lb:.0f}s ({lr:.0%}); binding ratio {ratio:.0%}."}


# ---------------------------------------------------------------- Auditability
def _trace_claim(claim: dict, answer: dict, corpus: dict) -> dict:
    audit = answer.get("audit") or {}
    gb = {g["span_id"]: g for g in audit.get("gate_b", [])}
    ga = {g["source_id"]: g for g in audit.get("gate_a", [])}
    steps, level = [], 1
    spans = [corpus["spans"].get(s) for s in claim.get("citations", [])]
    spans = [s for s in spans if s]
    srcs = [corpus["sources"].get(s["source_id"]) for s in spans]
    if not claim.get("citations"):
        return {"claim_id": claim["id"], "level": 1, "steps": ["no citations: black box"]}
    level = 2
    steps.append("cited sources listed")
    if all(s and s.get("url") for s in srcs) and srcs:
        level = 3
        steps.append("claim linked to source URL")
    full = bool(spans) and all("start" in s and "end" in s for s in spans)
    hashed = bool(srcs) and all(s and s.get("content_hash") for s in srcs)
    gb_ok = bool(spans) and all(gb.get(c, {}).get("rationale") for c in claim["citations"] if c in corpus["spans"])
    if full and hashed and gb_ok:
        level = 4
        steps.append("claim -> span offsets -> source hash -> Gate B rationale")
        ga_ok = all(ga.get(s["source_id"], {}).get("rationale") for s in spans)
        ver_ok = all(gb[c].get("rubric_version") for c in claim["citations"])  # rubric OR model version suffices
        if ga_ok and ver_ok and all(s.get("version") for s in srcs):
            level = 5
            steps.append("+ Gate A decision, appraisal rubric/model version, source version: rebuildable")
    return {"claim_id": claim["id"], "level": level, "steps": steps}


def score_auditability(answer: dict, corpus: dict, *, seed: int = 0, k: int = AUDIT_K) -> dict:
    claims = answer.get("claims")
    if not claims:
        txt = answer.get("response_text", "")
        has_urls = bool(re.search(r"https?://", txt))
        lvl = 2 if has_urls else 1
        return {"score": lvl, "sampled": [], "seed": seed,
                "rationale": "Free-text answer with no claim->evidence linkage; "
                             + ("sources named but not linked to claims." if has_urls else "black box.")}
    pool = [c for c in claims if c.get("consequential")] or claims
    rng = random.Random(seed)  # seeded: the sample is reproducible and recorded
    sample = rng.sample(pool, min(k, len(pool)))  # k defaults to 5 (spec: 3-5)
    traces = [_trace_claim(c, answer, corpus) for c in sample]
    score = min(t["level"] for t in traces)  # weakest link decides
    return {"score": score, "sampled": traces, "seed": seed, "pool_size": len(pool),
            "rationale": f"Traced {len(traces)} of {len(pool)} consequential claims (seed {seed}); "
                         f"weakest trace level {score}."}
