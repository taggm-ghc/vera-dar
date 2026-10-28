"""CATSTEP-1028 (plan s27): unknown-category -> categorise -> similarity -> topic check.

Owner 2026-10-28: "When the classifier replies "not in scope" but gives no recognised category, VERA should evaluate
the prompt/query/question/statement to describe it as a category, then check existing topics for any that include
that category, etc." and "VERA should only ever refuse forbidden/restricted/non-sanctioned topics" and "if a category
is deemed too semantically similar to such, then those should also be logged refusals".

Runs ONLY for a well-formed layer-2 in_scope:false reply whose category is missing or not in the closed enum (plan
s26.1 P10b), after the existing single retry, with a clean raw reply and a clean layer 1. Every other verdict
(malformed, duplicate-key, non-ASCII, category_conflict, refusal_token, refusal-named, layer-1 hit) never reaches it.

Outcomes: "anchor" / "field" (an existing topic includes the category), "unassigned" (no topic: an in-process topic
PROPOSAL, divergence DV-CATSTEP-1), "refuse" (similar to a forbidden or restricted category, or the LLM step named a
refusal category; logged with a closed reason token, divergence DV-CATSTEP-2). Invalid LLM-step output (malformed,
duplicate or non-ASCII key, chat error) is nullified, not treated as misuse, and falls back to the previous safe
decline (reason token "nullified:*"). No topic is refused for its subject or for being absent from the registry. Never stores or logs the raw question: only class
tokens, a qhash, score buckets and keys built from FIXED vocabularies (the topic registry and the refusal anchors).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import unicodedata
from dataclasses import dataclass, field

_LOG = logging.getLogger("vera")
REASON_OK = "llm_out:catstep"
# Decision reasons set by ask_service._catstep. Both prefixes never trigger acquisition (acq_config.trigger_for_decline)
# and never qualify for a reading list or research (those need reason "llm_out:").
REFUSE_PREFIX = "deny:catstep:"  # a refusal: similar to a forbidden/restricted category, or a refusal category named
NULLIFIED_PREFIX = "classifier_malformed:catstep:"  # invalid LLM-step output: the previous safe decline


def decision_reason(out) -> str:
    """The fixed decision reason for a refused or nullified outcome (closed tokens only)."""
    r = str(getattr(out, "reason", "") or "refuse:catstep_error")
    return (NULLIFIED_PREFIX + r.split(":", 1)[1]) if r.startswith("nullified:") else REFUSE_PREFIX + r.split(":", 1)[-1]
UNASSIGNED = "unassigned"
MAX_KEY_TERMS = 8

_PROPOSALS: dict = {}
_PROPOSALS_DROPPED = [0]
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Outcome:
    status: str  # "anchor" | "field" | "unassigned" | "refuse" (a refusal, or the safe fallback after invalid output)
    key: str = UNASSIGNED
    reason: str = ""  # closed token for a refusal, e.g. "refuse:similar_to:harmful"
    refuse_category: str = ""  # a REFUSE_CATEGORIES value, when the refusal maps to one
    tokens: int = 0
    cost: float = 0.0
    calls: int = 0
    log: dict = field(default_factory=dict)


def step_config(cfg=None) -> dict:
    """`unknown_category_step` from config/scope_router.json ({} when missing or not an object)."""
    from vera import scope_router as sr
    s = (cfg or sr.load_config()).raw.get("unknown_category_step")
    return s if isinstance(s, dict) else {}


def enabled(cfg=None) -> bool:
    """Kill switch: only an explicit boolean true runs the step (false restores the D-P10b behaviour)."""
    try:
        return step_config(cfg).get("enabled") is True
    except Exception:  # noqa: BLE001
        return False


def applies(decision, question: str, cfg=None) -> bool:
    """plan s27.1: the exact trigger. Fails closed (False = the previous decline path)."""
    try:
        from vera import scope_router as sr
        return (enabled(cfg) and decision.accepted is False and decision.layer == "layer2_llm"
                and str(decision.reason).startswith("llm_out:") and decision.category == ""
                and getattr(decision, "raw_refusal_named", True) is False and sr.intent_clean(question))
    except Exception:  # noqa: BLE001
        return False


def qhash(question: str) -> str:
    n = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(question or "")).casefold()).strip()
    return hashlib.sha256(n.encode("utf-8")).hexdigest()[:16]


def _bucket(score: float) -> str:
    try:
        return "b%.1f" % (math.floor(float(score) * 2) / 2)  # 0.5-wide buckets
    except (TypeError, ValueError):
        return "bnan"


def _micro_cfg() -> dict:
    from vera import acq_config
    return (acq_config.load_config() or {}).get("micro_topic") or {}


def category_key(question: str, reg) -> str:
    """Normalised category key: sorted unique question tokens that are in the registry vocabulary (fixed, so no personal
    text can enter), at most MAX_KEY_TERMS; else q:<qhash>."""
    from vera import micro_topic
    vocab = set(reg.idf) if reg is not None else set()
    terms = sorted({t for t in micro_topic.tokens(question) if t in vocab})[:MAX_KEY_TERMS]
    return "-".join(terms) if terms else "q:" + qhash(question)


class _Refuse(Exception):
    def __init__(self, token, category=""):
        super().__init__(token)
        self.token, self.category = token, category


def similarity(question: str, scfg: dict) -> tuple[str, float, list]:
    """(forbidden id or "", best score, matched fixed-vocabulary terms). BM25 (micro_topic.Registry) over the refusal
    anchors in config. Raises _Refuse("refuse:similarity_unavailable") when it cannot be computed (fail closed)."""
    from vera import micro_topic
    try:
        sim = scfg["similarity"]
        thr = float(sim["threshold"])
        anchors = sim["anchors"]
        if not isinstance(anchors, dict) or not anchors or not math.isfinite(thr) or thr <= 0:
            raise ValueError("bad similarity config")
        reg = micro_topic.Registry({"entries": [{"id": k, "anchor_text": v} for k, v in anchors.items()
                                                if isinstance(v, str) and v.strip()]})
        if len(reg.entries) != len(anchors):
            raise ValueError("bad anchor text")
        topic_vocab = set(micro_topic.load_registry(_micro_cfg()).idf)  # a registered topic's words never refuse
        query = " ".join(t for t in micro_topic.tokens(question) if t not in topic_vocab)
        best = reg.best(query)
    except Exception as exc:  # noqa: BLE001
        raise _Refuse("refuse:similarity_unavailable") from exc
    if best is None:
        raise _Refuse("refuse:similarity_unavailable")
    score, entry = best
    terms = sorted(set(micro_topic.tokens(query)) & set(micro_topic.tokens(entry.get("anchor_text"))))[:MAX_KEY_TERMS]
    return (entry["id"] if score >= thr else ""), float(score), terms


def _llm_field(question, scfg, cfg, chat, model, pricing, fields) -> tuple[str, int, float, int]:
    """Optional LLM step (ships off). Returns (field or "", tokens, cost, calls); raises _Refuse on any attack signal."""
    from vera import scope_router as sr
    l2 = cfg.layer2
    q = sr._neutralise_for_classifier((question or "")[: int(l2["max_question_chars"])])
    vocab = sorted(fields) + ["other"]
    msgs = [{"role": "system", "content": str(scfg["llm_system_prompt"]).replace("{fields}", ", ".join(vocab))},
            {"role": "user", "content": l2["user_template"].format(question=q)}]
    try:
        content, pt, ct = (chat or sr.default_chat)(msgs, model, l2)
    except Exception as exc:  # noqa: BLE001  fail closed: the extra check could not run
        raise _Refuse("nullified:catstep_llm_error") from exc
    cost = 0.0
    if pricing is not None:
        cost = round(pt / 1_000_000 * pricing.input + ct / 1_000_000 * pricing.output, 8)
    obj = sr._strict_loads(content)
    if not isinstance(obj, dict):
        raise _Refuse("nullified:catstep_malformed")
    tok = sr.refusal_token(content)
    if tok:
        raise _Refuse("refuse:catstep_refusal_token", tok)
    if sr._raw_names_refusal(content):
        raise _Refuse("refuse:catstep_refusal_named")
    f = obj.get("field")
    if not isinstance(f, str) or not unicodedata.normalize("NFKC", f).isascii():
        raise _Refuse("nullified:catstep_malformed")
    return (f if f in fields else ""), pt + ct, cost, 1


def _propose(key: str, qh: str, nearest: str, bucket: str, cap: int) -> str:
    """DV-CATSTEP-1: in-process proposal (no DB registry: DDL not authorized; Q7 anonymous users only propose).
    A duplicate key increments a count and never overwrites; over the cap new keys are dropped and counted."""
    with _LOCK:
        rec = _PROPOSALS.get(key)
        if rec is not None:
            rec["count"] += 1
            return "dup"
        if len(_PROPOSALS) >= cap:
            _PROPOSALS_DROPPED[0] += 1
            return "dropped"
        _PROPOSALS[key] = {"count": 1, "qhash": qh, "nearest": nearest, "bucket": bucket}
        return "new"


def proposals() -> dict:
    with _LOCK:
        return {k: dict(v) for k, v in _PROPOSALS.items()}


def reset_for_tests() -> None:
    with _LOCK:
        _PROPOSALS.clear()
        _PROPOSALS_DROPPED[0] = 0


def _emit(out: Outcome) -> Outcome:
    """One structured class-token log line per outcome (never raw text). Never raises."""
    try:
        _LOG.info("catstep " + " ".join(f"{k}={v}" for k, v in out.log.items()))
    except Exception:  # noqa: BLE001
        pass
    return out


def categorise(question: str, cfg=None, chat=None, model: str = "", pricing=None) -> Outcome:
    """plan s27.2-27.4. Order: anchor match (A4 BM25) -> similarity to every forbidden/restricted category (refuse when
    too similar or not computable) -> optional LLM field step (ships off) -> topic key or proposal."""
    from vera import micro_topic
    from vera import scope_router as sr
    cfg = cfg or sr.load_config()
    qh = qhash(question)
    reg = None
    try:
        scfg = step_config(cfg)
        mcfg = _micro_cfg()
        try:
            reg = micro_topic.load_registry(mcfg)
        except (OSError, ValueError):
            reg = None
        ckey = category_key(question, reg)
        # (1) anchor match
        b = reg.best(question) if reg is not None else None
        thr = micro_topic._threshold(mcfg) if b is not None else math.inf
        akey = ""
        if b is not None and b[0] >= thr:
            akey = micro_topic.key_id(micro_topic._entry_key("registry", b[1]["id"], round(b[0], 4), reg, b[1].get("field")))
        # (2) similarity to forbidden / restricted categories (always, also after an anchor match)
        sim_id, sim_score, sim_terms = similarity(question, scfg)
        if sim_id:
            cat = sim_id if sim_id in sr.REFUSE_CATEGORIES else (scfg["similarity"].get("maps_to") or {}).get(sim_id, "")
            cat = cat if cat in sr.REFUSE_CATEGORIES else "harmful"  # unknown mapping: the strictest category
            raise _Refuse("refuse:similar_to:" + sim_id, cat)
        if akey:
            return _emit(Outcome("anchor", akey, log={"outcome": "anchor", "topic": akey, "key": ckey,
                                                      "sim": _bucket(sim_score), "qhash": qh}))
        # (3) optional LLM step (ships off: plan s27.2)
        tokens = calls = 0
        cost = 0.0
        if scfg.get("llm_step") is True:
            fields = {e.get("field") for e in (reg.entries if reg is not None else ()) if isinstance(e.get("field"), str)}
            f, tokens, cost, calls = _llm_field(question, scfg, cfg, chat, model, pricing, fields)
            if f:
                key = "field:" + f
                return _emit(Outcome("field", key, tokens=tokens, cost=cost, calls=calls,
                                     log={"outcome": "field", "topic": key, "key": ckey, "sim": _bucket(sim_score),
                                          "qhash": qh}))
        # (4) new category: proposal only
        cap = int(scfg.get("proposal_cap") or 0)
        nearest = b[1]["id"] if b is not None else "none"
        bucket = _bucket(b[0]) if b is not None else "bnan"
        st = _propose(ckey, qh, nearest, bucket, cap) if cap > 0 else "dropped"
        return _emit(Outcome("unassigned", UNASSIGNED, tokens=tokens, cost=cost, calls=calls,
                             log={"outcome": "proposal", "proposal": st, "key": ckey, "nearest": nearest,
                                  "score": bucket, "sim": _bucket(sim_score), "qhash": qh}))
    except _Refuse as r:  # "nullified:*" = invalid LLM-step output (owner: invalid output, not misuse) -> safe fallback
        return _emit(Outcome("refuse", reason=r.token, refuse_category=r.category,
                             log={"outcome": r.token.split(":", 1)[0], "reason": r.token, "category": r.category or "none",
                                  "key": _safe_key(question, reg), "qhash": qh, **_sim_log(question, r)}))
    except Exception:  # noqa: BLE001  fail closed: any fault refuses (never researches)
        return _emit(Outcome("refuse", reason="refuse:catstep_error", refuse_category="",
                             log={"outcome": "refuse", "reason": "refuse:catstep_error", "qhash": qh}))


def _safe_key(question, reg) -> str:
    try:
        return category_key(question, reg)
    except Exception:  # noqa: BLE001
        return "q:" + qhash(question)


def _sim_log(question, r) -> dict:
    """Score bucket and matched FIXED-vocabulary terms for a similarity refusal (never question text)."""
    if not r.token.startswith("refuse:similar_to:"):
        return {}
    try:
        from vera import scope_router as sr
        _id, score, terms = similarity(question, step_config(sr.load_config()))
        return {"score": _bucket(score), "terms": "-".join(terms) or "none"}
    except Exception:  # noqa: BLE001
        return {"score": "bnan"}
