"""FRESH-1023 integration adapter: evidence sufficiency, the two research triggers, and the per-request research record.

Freshness is computed ONLY by vera.freshness.assess_freshness (clock hook vera.freshness.utcnow); the outcome taxonomy
and user sentence ONLY by vera.research_outcome.summarize_outcome. This module maps the pipeline's own data (admitted
passages, the inline acquisition Outcome.public()) onto those two modules and projects the additive response fields
`freshness`, `research_outcome` and `triggers`. Pure: no I/O.
"""
from __future__ import annotations

from datetime import datetime

from vera import freshness as fz
from vera import research_outcome as ro

TRIGGER_INSUFFICIENT, TRIGGER_STALE, TRIGGER_UNKNOWN = "insufficient", "stale", "freshness_unknown"
_BUDGET_STATUSES = frozenset({"rate", "daily_cap", "deadline", "budget_unknown", "busy"})
_PORTAL_STATUS = {"completed": "completed", "timeout": "timeout", "quota": "quota", "skipped_quota": "quota",
                  "failed": "error", "error": "error", "skipped_budget": "skipped", "skipped": "skipped"}


def cfg_of(acq: dict | None) -> dict | None:
    """config/auto_acquire.json freshness block when enabled with a positive freshness_max_age_hours, else None."""
    f = (acq or {}).get("freshness") if isinstance(acq, dict) else None
    if not isinstance(f, dict) or f.get("enabled") is not True:
        return None
    try:
        return f if float(f.get("freshness_max_age_hours")) > 0 else None
    except (TypeError, ValueError):
        return None


def now() -> datetime:
    return fz.utcnow()  # looked up at call time: the test hook vera.freshness.utcnow applies


def assess(admitted, cfg: dict, at: datetime | None = None, new_ids=()) -> dict:
    """Per-document freshness of the admitted passages (the least fresh passage decides for its document).
    status: fresh | stale | unknown | none (no evidence). FRESH-AMEND-1023: freshness = corpus_added_at (when the chunk
    was added to the corpus) vs the threshold; evidence_recency (recent | older | unknown) = source_authored_at. Both
    dates are reported per document. `_backfill` (internal, never public) lists the stamps to revise to now()."""
    at = at or now()
    adm = [a for a in admitted or () if isinstance(a, dict)]
    rep = fz.assess_freshness(adm, now=at, cfg=cfg)
    new_ids = {str(x) for x in new_ids or ()}
    rank = {"fresh": 0, "stale": 1, "unknown": 2}
    docs: dict = {}
    backfill: list = []
    for a, c in zip(adm, rep.chunks):
        backfill.extend(c.backfill)
        did = str(a.get("document_id"))
        row = {"document_id": a.get("document_id"), "basis": c.basis, "freshness": c.status, "reason": c.reason,
               "corpus_added_at": c.corpus_added_at.isoformat() if c.corpus_added_at else None,
               "corpus_added_basis": c.basis,
               "source_authored_at": c.source_authored_at, "source_authored_precision": c.authored_precision,
               "source_authored_basis": c.authored_basis, "evidence_recency": c.authored_recency,
               "last_checked_at": None if c.last_checked_at is None else str(c.last_checked_at),
               "new_this_request": did in new_ids}
        if did not in docs or rank[c.status] > rank[docs[did]["freshness"]]:
            docs[did] = row
    counts = {"fresh": 0, "stale": 0, "unknown": 0}
    authored = {"recent": 0, "older": 0, "unknown": 0}
    for r in docs.values():
        counts[r["freshness"]] += 1
        authored[r["evidence_recency"]] += 1
    status = "none" if not docs else "unknown" if counts["unknown"] else "stale" if counts["stale"] else "fresh"
    return {"status": status, "max_age_hours": rep.max_age_hours, "checked_at": at.isoformat(), "counts": counts,
            "chunk_counts": dict(rep.counts), "authored_counts": authored, "trigger_basis": "corpus_added_at",
            "documents": list(docs.values()), "_backfill": backfill}


def apply_backfill(fr: dict | None, cfg: dict, at: datetime, acq: dict | None, stamper=None) -> dict | None:
    """Owner refinement 2026-10-23: a missing / invalid corpus_added_at on a chunk used in this request is revised to the
    request's now() (vera.corpus_stamp: in memory at once, persisted by a conditional single-row UPDATE through the WO
    app path when configured). None when nothing needed it. Never raises."""
    stamps = (fr or {}).get("_backfill") or []
    if not stamps:
        return None
    import os

    from vera import acq_config, corpus_stamp
    reason = ""
    try:
        if acq_config.killed(acq, os.environ):  # review D1: the persist step honours the kill switch; memory stamp still applies
            stamper, reason = None, "kill switch"
        elif stamper is None and any(corpus_stamp.persistable(x) for x in stamps):
            stamper, reason = corpus_stamp.default_stamper(acq)
        return corpus_stamp.backfill(stamps, at, cfg, stamper, reason)
    except Exception as exc:  # noqa: BLE001
        return {"count": 0, "chunk_ids": [], "at": at.isoformat(), "persist": {"status": "failed",
                                                                                "reason": type(exc).__name__}}


def _public_fr(fr: dict | None) -> dict | None:
    return None if fr is None else {k: v for k, v in fr.items() if k != "_backfill"}


def sufficient(result, cfg: dict) -> bool:
    """Sufficiency of an ANSWER: the unchanged admission + writer + claim check produced a grounded answer and, when
    sufficiency.partial_is_insufficient is true, it is not labelled partial (a material subquestion unanswered)."""
    if result is None or getattr(result, "result_kind", None) != "answer":
        return False
    ev = getattr(result, "evidence", None)
    if ev is None or not getattr(ev, "admitted", None):
        return False
    cc = getattr(result, "claim_check", None) or {}
    if (cfg.get("sufficiency") or {}).get("partial_is_insufficient") is True and isinstance(cc, dict) \
            and cc.get("answer_completeness") == "partial":
        return False
    return True


def triggers(is_sufficient: bool, fr: dict | None) -> list:
    """Either trigger alone starts research; both may apply."""
    out = [] if is_sufficient else [TRIGGER_INSUFFICIENT]
    c = (fr or {}).get("counts") or {}
    if c.get("stale"):
        out.append(TRIGGER_STALE)
    if c.get("unknown"):
        out.append(TRIGGER_UNKNOWN)
    return out


def _public_cand(c: dict) -> dict:
    """G23 guard: PMC ids never appear in the /ask response (kept in the stored provenance and the server log only)."""
    c.pop("pmcid", None)
    if isinstance(c.get("ids"), dict):
        c["ids"] = {k: v for k, v in c["ids"].items() if k != "pmcid"}
    return c


def _ids(c: dict) -> dict:
    return {k: c.get(k) for k in ro.ID_KEYS if k != "pmcid"}  # G23 guard: no PMC id in the /ask response


def _terminal(c: dict) -> str:
    o = c.get("outcome")
    if o in ("admitted", "reused"):
        return "admitted"
    if o in ("not_evaluated", "not_used"):
        return "budget_exhausted"
    return ro.reason_from_code(c.get("reason"))


def summarize(*, triggers: list, inline: dict | None, used_new: bool, older_available: bool, fr_before: dict | None,
              fr_after: dict | None = None, citations=None, answer_failed: bool = False,
              failed_text: str | None = None, backfill: dict | None = None) -> dict:
    """contract["research"]: the full record (report detail) for one request."""
    inline = inline if isinstance(inline, dict) else None
    status = (inline or {}).get("status")
    raw_portals = [p for p in ((inline or {}).get("portals") or []) if isinstance(p, dict)]
    attempts = [ro.PortalAttempt(str(p.get("portal")), _PORTAL_STATUS.get(p.get("status"), "error"),
                                 int(p.get("results") or 0)) for p in raw_portals]
    cands = [c for c in ((inline or {}).get("candidates") or []) if isinstance(c, dict)]
    outs = [ro.CandidateOutcome(f"c{i + 1}", _ids(c), _terminal(c), str(c.get("reason") or ""))
            for i, c in enumerate(cands)]
    budget = (status in _BUDGET_STATUSES or any(p.get("status") == "skipped_budget" for p in raw_portals)
              or any(o.terminal_reason == "budget_exhausted" for o in outs))
    if not triggers:
        kind, text, complete, detail = "not_needed", "", False, {}
    else:
        res = ro.summarize_outcome(attempts, outs, budget_exhausted=budget, older_evidence_available=older_available,
                                   new_evidence_admitted=used_new)
        kind, text, complete, detail = res.kind, res.user_text, res.complete, res.report_detail
        if answer_failed and failed_text:
            text = f"{failed_text} {text}".strip()
    return {"triggered": bool(triggers), "triggers": list(triggers), "kind": kind, "user_text": text,
            "complete": complete, "inline_status": status, "portals": [dict(p) for p in raw_portals],
            "candidates": [_public_cand(dict(c, candidate_id=o.candidate_id, ids=dict(o.ids),
                                             terminal_reason=o.terminal_reason)) for c, o in zip(cands, outs)],
            "report_detail": detail, "freshness": {"before": _public_fr(fr_before), "after": _public_fr(fr_after)},
            "citations": list(citations or []),
            **({"corpus_added_backfill": backfill} if backfill else {})}


def public_fields(rec: dict | None) -> dict:
    """Additive top-level /ask fields: triggers, freshness, research_outcome. {} when the request has no record."""
    if not isinstance(rec, dict):
        return {}
    fr = rec.get("freshness") or {}
    cur = fr.get("before") or {}  # the evidence the triggers were decided on; the cited result is under "after"
    aft = fr.get("after") or {}
    c = cur.get("counts") or {}
    return {"triggers": list(rec.get("triggers") or []),
            "freshness": {"status": cur.get("status"), "any_stale_or_unknown": bool(c.get("stale") or c.get("unknown")),
                          "counts": dict(c), "max_age_hours": cur.get("max_age_hours"),
                          "checked_at": cur.get("checked_at"), "documents": list(cur.get("documents") or []),
                          "trigger_basis": cur.get("trigger_basis"), "authored_counts": dict(cur.get("authored_counts") or {}),
                          **({"corpus_added_backfill": rec["corpus_added_backfill"]}
                             if rec.get("corpus_added_backfill") else {}),
                          "after": {"counts": dict(aft.get("counts") or {}),
                                    "documents": list(aft.get("documents") or [])} if aft else None},
            "research_outcome": {"kind": rec.get("kind"), "user_text": rec.get("user_text") or "",
                                 "complete": bool(rec.get("complete")), "portals": list(rec.get("portals") or []),
                                 "candidates": list(rec.get("candidates") or []),
                                 "citations": list(rec.get("citations") or [])}}
