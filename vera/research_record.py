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
                  "failed": "error", "error": "error", "skipped_budget": "skipped", "skipped": "skipped",
                  "not_attempted": "skipped"}
# ROUTE-R3: public statuses (contract completed|timeout|quota|error|skipped|not_attempted)
_PUBLIC_STATUS = {**_PORTAL_STATUS, "skipped_budget": "not_attempted", "not_attempted": "not_attempted"}


ANSWER_BASES = ("generated_new_evidence", "generated_existing_only", "abstained", "not_attempted")
# inline statuses that return before any search or fetch ran (ask_service.py:807/811, inline_acquire.py:406)
_NOT_ATTEMPTED_REASON = {None: "no_search_record", "off": "disabled", "busy": "busy", "deadline": "deadline",
                         "budget_unknown": "budget_unknown", "rate": "rate_limit", "daily_cap": "daily_cap"}


# Statuses that prove a search ran. "deadline" is NOT here: the pre-search guard also returns it (ask_service.py:258/811);
# a deadline after a search shows up as candidates or queried portals instead.
_SEARCHED = frozenset({"ok", "none", "search_failed", "unanswered"})


# Reason code -> was an HTTP request sent for the candidate's source? (code, evidence file:line). Decided per code, not per
# diag bucket (R-ACQ-ROBOTS/CYCLE/HOST/FORMAT_UNSUPPORTED land in selected_not_fetched but follow requests).
_FETCH_YES = frozenset({
    "R-ACQ-UNREACHABLE",         # acq_discovered.py:402 (http_get raised), :410/:416 (after response)
    "R-ACQ-FETCH_TIMEOUT",       # :383 (loop top, after >=1 hop in practice), :430 (while streaming a response)
    "R-ACQ-HOST",                # :385/:388 (after a redirect hop; first-URL host refusal is pre-request: overcount), :513
    "R-ACQ-CYCLE",               # :390/:407 (a revisit implies an earlier request)
    "R-ACQ-ROBOTS",              # :392 (robots.txt itself was fetched)
    "R-ACQ-ENCODING",            # :419 (response headers read)
    "R-ACQ-FORMAT_UNSUPPORTED",  # :422, :508 (content-type of a response)
    "R-ACQ-LARGE",               # :427 (streaming body)
    "R-ACQ-CONVERT",             # :521 (conversion of fetched body)
    "R-ACQ-IDENTITY_MISMATCH",   # :524 (fetched document identity)
    "R-ACQ-LICENCE_ABSENT", "R-ACQ-LICENCE_MISMATCH",  # :561-575 (licence read from the fetched document)
    "R-ACQ-INJECT", "R-ACQ-IRRELEVANT", "R-ACQ-FEWSPANS",  # auto_acquire.py:948/958 (fetched text checks)
    "R-ACQ-WRITE",               # store write after fetch
})
# Everything else is NOT counted (no request proven): R-ACQ-NODOI/NOTOA/NOLIC(prefilter)/DUPLICATE/KNOWN/DEFER_CAP
# (pre-fetch filters; auto_acquire.py:159/863, acq_discovered.py:245-255), R-ACQ-RATELIMITED (nothing sent,
# acq_discovered.py:400), R-ACQ-DEADLINE/DOC_BUDGET/RATE/DAILY/SERVECAP/CRAWL_DELAY (stop before request), R-ACQ-REUSE-*
# (stored text), R-ACQ-PREPRINT_*/ARXIV_* (policy holds), R-ACQ-PUBTYPE/R-ABSTRACT (not proven), R-DB*/R-BUNDLE etc.
_FETCH_STATUSES = frozenset({"fetched_rejected", "selected_not_fetched", "acquired_not_admitted", "admitted"})


def _fetch_made(c: dict) -> bool:
    """True when a fetch request was made for this candidate: by outcome for admitted / acquired text (not reused
    stored text), otherwise by terminal reason code (_FETCH_YES). discovered_not_selected and skipped never count."""
    if not isinstance(c, dict):
        return False
    st = c.get("status")
    if st == "admitted":
        return c.get("status_reason") != "reused_stored"
    if st == "acquired_not_admitted":
        return c.get("outcome") != "reuse_rejected"
    if st not in _FETCH_STATUSES:
        return False
    return str(c.get("reason") or "").strip().upper() in _FETCH_YES


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
    """PMCID-PUBLIC-1023 (owner 2026-10-23): the PMCID is a public identifier like the DOI and stays in the response
    (the former G23 strip is retired). Abstract text is never part of a candidate record."""
    return c


def _ids(c: dict) -> dict:
    return {k: c.get(k) for k in ro.ID_KEYS}  # PMCID-PUBLIC-1023: pmcid is public, like doi


def _terminal(c: dict) -> str:
    o = c.get("outcome")
    if o in ("admitted", "reused"):
        return "admitted"
    if o == "not_used":  # P0DTA-DIAG: passed the checks but the per-request document limit was reached first
        return "doc_budget_reached"
    if o == "not_evaluated":  # P0DTA-DIAG: the recorded stop reason, each with its own terminal reason
        k = ro.skip_class(c.get("reason"))
        return k if k and k not in ("fetch_timeout", "crawl_delay") else "budget_exhausted"
    if o == "queued_background":  # CONT-1023: unfinished in this request, handed to the background queue
        return "queued_background"
    return ro.reason_from_code(c.get("reason"))


_SKIP_CAUSE = {"deadline_reached": "deadline_reached", "candidate_cap": "candidate_cap",
               "doc_budget_reached": "doc_budget_reached", "quota": "job_quota", "fetch_timeout": "fetch_timeout",
               "crawl_delay": "crawl_delay"}


def summarize(*, triggers: list, inline: dict | None, used_new: bool, older_available: bool, fr_before: dict | None,
              fr_after: dict | None = None, citations=None, answer_failed: bool = False,
              failed_text: str | None = None, backfill: dict | None = None,
              answer_basis: str | None = None) -> dict:
    """contract["research"]: the full record (report detail) for one request."""
    inline = inline if isinstance(inline, dict) else None
    status = (inline or {}).get("status")
    raw_portals = [p for p in ((inline or {}).get("portals") or []) if isinstance(p, dict)]
    attempts = [ro.PortalAttempt(str(p.get("portal")), _PORTAL_STATUS.get(p.get("status"), "error"),
                                 int(p.get("results") or 0)) for p in raw_portals]
    cands = [c for c in ((inline or {}).get("candidates") or []) if isinstance(c, dict)]
    outs = [ro.CandidateOutcome(f"c{i + 1}", _ids(c), _terminal(c), str(c.get("reason") or ""))
            for i, c in enumerate(cands)]
    # DDGFIX-1023 DF-4: a candidate queued because its source could not be reached (unreachable, rate-limited,
    # transport error) has that cause, not "the search budget ran out"; URL-only ones have the awaiting_route cause.
    unreachable = [o for o in outs if o.terminal_reason == "queued_background" and ro.queued_unreachable(o.detail)]
    # P0DTA-DIAG: specific stop causes. A queued candidate keeps why it was unfinished in .detail.
    extra: list = []
    for o in outs:
        k = ro.skip_class(o.detail) if o.terminal_reason == "queued_background" else \
            (o.terminal_reason if o.terminal_reason in _SKIP_CAUSE else None)
        if k in _SKIP_CAUSE and _SKIP_CAUSE[k] not in extra:
            extra.append(_SKIP_CAUSE[k])
    if status in ("rate", "daily_cap") and "job_quota" not in extra:  # the quota refused the whole run: say so
        extra.append("job_quota")
    if status == "deadline" and "deadline_reached" not in extra:
        extra.append("deadline_reached")
    if status in ("busy", "budget_unknown") and status not in extra:  # each gets its own truthful sentence
        extra.append(status)
    budget = (status in _BUDGET_STATUSES - {"rate", "daily_cap", "deadline", "busy", "budget_unknown"}
              or any(p.get("status") in ("skipped_budget", "not_attempted") for p in raw_portals)
              or any(o.terminal_reason == "budget_exhausted" or (o.terminal_reason == "queued_background"  # CONT-1023
                     and ro.skip_class(o.detail) is None and ro.reason_from_code(o.detail) != "no_route"
                     and o not in unreachable) for o in outs))
    if not triggers:
        kind, text, complete, detail = "not_needed", "", False, {}
    else:
        res = ro.summarize_outcome(attempts, outs, budget_exhausted=budget, older_evidence_available=older_available,
                                   new_evidence_admitted=used_new, source_unreachable=bool(unreachable),
                                   extra_causes=extra)
        kind, text, complete, detail = res.kind, res.user_text, res.complete, res.report_detail
        if answer_failed and failed_text:
            text = f"{failed_text} {text}".strip()
    dc = ro.diag_counts(cands)
    # DDFIX-1023 TEVAL: discovery, acquisition, admission and answer generation are separate aggregate facts.
    # answer: what the answer was actually based on. Conservative: "generated_new_evidence" only when new evidence
    # was both flagged used (used_new) and counted admitted; an explicit answer_basis cannot upgrade beyond that.
    if answer_basis in ANSWER_BASES:
        answer_state = answer_basis
    elif answer_failed or not triggers:
        answer_state = "generated_existing_only" if older_available else "abstained"
    elif used_new:
        answer_state = "generated_new_evidence"
    else:
        answer_state = "generated_existing_only" if older_available else "abstained"
    if answer_state == "generated_new_evidence" and not (used_new and int(dc.get("admitted") or 0) > 0):
        answer_state = "generated_existing_only" if older_available else "abstained"
    # Four distinct stages. search: a search actually ran (candidates, a portal queried, or a status in
    # inline_acquire.SEARCHED = ok|none|deadline|search_failed|unanswered, inline_acquire.py:51). Pre-search guards
    # (off/busy/budget_unknown/rate/daily_cap with nothing found) mean not_run.
    search_ran = bool(cands) or status in _SEARCHED or any(
        p.get("status") in ("completed", "timeout", "error", "failed") for p in raw_portals)
    search_reason = None if search_ran else _NOT_ATTEMPTED_REASON.get(status, "no_search_record")
    # acquisition: attempted only when a fetch request was made for at least one candidate (see _fetch_made).
    fetched = sum(1 for c in cands if _fetch_made(c))
    steps = {"discovered": len(cands), "search": "ran" if search_ran else "not_run", "search_reason": search_reason,
             "acquisition": "attempted" if fetched else "not_attempted",
             "acquisition_reason": None if fetched else (search_reason or "no_fetch_made"),
             "fetch_requests": fetched, "admitted": int(dc.get("admitted") or 0), "answer": answer_state}
    return {"triggered": bool(triggers), "triggers": list(triggers), "kind": kind, "steps": steps, "user_text": text,
            "complete": complete, "inline_status": status,
            "portals": [dict(p, status=_PUBLIC_STATUS.get(p.get("status"), "error"),
                             n_results=int(p.get("n_results", p.get("results")) or 0)) for p in raw_portals],
            "candidates": [_public_cand(dict(c, candidate_id=o.candidate_id, ids=dict(o.ids),
                                             terminal_reason=o.terminal_reason)) for c, o in zip(cands, outs)],
            "diag_status_counts": ro.diag_counts(cands),  # P0DTA-DIAG: aggregate per status (UI shows only this)
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
                                 "diag_status_counts": dict(rec.get("diag_status_counts") or {}),
                                 "steps": dict(rec.get("steps") or {}),
                                 "citations": list(rec.get("citations") or [])}}
