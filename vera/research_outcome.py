"""Research-outcome taxonomy: portal attempts, per-candidate terminal reasons,
and an honest end-of-search summary (FRESH-C3).

Pure module: no I/O, no network. Never upgrades a candidate to 'admitted'
from a reason code; 'admitted' is only set explicitly by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PORTAL_STATUSES = ("completed", "timeout", "quota", "error", "skipped")
TERMINAL_REASONS = (
    "no_route", "fetch_failed", "extraction_failed", "licence_rejected",
    "provenance_rejected", "injection_rejected", "relevance_rejected",
    "credibility_rejected", "freshness_rejected", "abstract_only",
    "budget_exhausted", "admitted", "already_in_corpus",
    "queued_background",  # CONT-1023 (additive): handed to the background acquisition queue, outcome pending
    # P0DTA-DIAG (additive): why a candidate was left unfinished, each with its own sentence (see _CAUSE_TEXT)
    "deadline_reached", "candidate_cap", "doc_budget_reached", "quota", "fetch_timeout", "unreachable",
    "duplicate",  # P0DTA-DIAG: an in-request duplicate of another candidate (R-ACQ-DUPLICATE)
    "crawl_delay",  # P0DTA-ARXFIX C: an arXiv crawl-delay wait did not fit the remaining time (queued, not a budget stop)
)
# Unfinished-candidate classes that were budget/limit stops of this request (not source failures).
LIMIT_REASONS = ("budget_exhausted", "deadline_reached", "candidate_cap", "doc_budget_reached")
ID_KEYS = ("doi", "pmcid", "openalex_id", "arxiv_id", "url")
KINDS = ("answered_with_new", "qualified_older_only", "no_valuable_source",
         "incomplete_search", "abstain")

NO_VALUABLE_SENTENCE = "No sufficiently valuable source was found across the portals searched."
BACKGROUND_SENTENCE = ("Further sources found by this search may be acquired in the background on a best-effort basis; "
                       "any that are added must first pass the licence, content and relevance checks.")

# Acquisition problems: a source may exist but could not be used.
_ACQ_FAILURES = ("no_route", "fetch_failed", "extraction_failed", "abstract_only")
# Admission rejections: the source was read and judged not admissible.
_ADMISSION_REJECTIONS = (
    "licence_rejected", "provenance_rejected", "injection_rejected",
    "relevance_rejected", "credibility_rejected", "freshness_rejected",
)

# Every reason code found in vera/ (discovery_panel.REASON_TEXT, inline_acquire,
# auto_acquire, fulldoc_corpus, fulldoc_db) at base 9d0d7643.
_CODE_MAP: dict[str, str] = {
    # no open/usable acquisition route
    "R-ACQ-NODOI": "no_route", "R-ACQ-NOTOA": "no_route",
    "R-ACQ-FORMAT_UNSUPPORTED": "no_route", "R-ACQ-HOST": "no_route",
    "R-ACQ-ROBOTS": "no_route", "R-ACQ-CYCLE": "no_route",
    # fetch/transport failures
    "R-ACQ-UNREACHABLE": "fetch_failed", "R-ACQ-RATELIMITED": "fetch_failed",
    "R-ACQ-ERROR": "fetch_failed",
    "R-ACQ-WRITE": "fetch_failed", "R-ERROR": "fetch_failed",
    "R-DBCONN": "fetch_failed", "R-DBRUN": "fetch_failed",
    # extraction failures
    "R-ACQ-LARGE": "extraction_failed", "R-ACQ-CONVERT": "extraction_failed",
    "R-ACQ-ENCODING": "extraction_failed", "R-DBEMPTY": "extraction_failed",
    "R-DBCHUNK": "extraction_failed", "R-DBPARTIAL": "extraction_failed",
    # licence
    "R-ACQ-NOLIC": "licence_rejected", "R-ACQ-LICENCE_ABSENT": "licence_rejected",
    "R-ACQ-LICENCE_MISMATCH": "licence_rejected",
    "R-ACQ-PREPRINT_LICENCE_UNVERIFIED": "licence_rejected",
    "R-ACQ-PREPRINT_NOT_PERMITTED": "licence_rejected",
    "R-LIC-REJECT": "licence_rejected", "R-LIC-HOLD": "licence_rejected",
    "R-LIC-MISMATCH": "licence_rejected",
    "R-ACQ-ARXIV_LICENCE_HOLD": "licence_rejected",  # ARX-A2 (INTEG-1023): arXiv default/unknown licence held
    "R-ACQ-ARXIV_HTML_POLICY": "licence_rejected",  # ARX-A2 (INTEG-1023): refused at the current gate (D-ARX-1 pending)
    # provenance / identity / integrity
    "R-ACQ-IDENTITY_MISMATCH": "provenance_rejected",
    "R-ACQ-DUPLICATE": "duplicate",
    # already held: not a rejection and not new evidence (the stored passages are used through retrieval)
    "R-ACQ-KNOWN": "already_in_corpus",
    "R-ACQ-REUSE-UNUSABLE": "provenance_rejected",
    "R-BUNDLE": "provenance_rejected", "R-PIN": "provenance_rejected",
    "R-SUMS": "provenance_rejected", "R-DUP": "provenance_rejected",
    "R-SOURCE": "provenance_rejected", "R-DBROW": "provenance_rejected",
    "R-DBHASH": "provenance_rejected", "R-DBDOCPIN": "provenance_rejected",
    # safety
    "R-ACQ-INJECT": "injection_rejected",
    # relevance
    "R-ACQ-IRRELEVANT": "relevance_rejected", "R-ACQ-FEWSPANS": "relevance_rejected",
    "R-ACQ-REUSE-IRRELEVANT": "relevance_rejected",
    # credibility (document type not usable as evidence)
    "R-ACQ-PUBTYPE": "credibility_rejected",
    # abstract only
    "R-ABSTRACT": "abstract_only",
    # budgets / caps
    "R-ACQ-DEFER_CAP": "candidate_cap", "R-ACQ-SERVECAP": "budget_exhausted",
    # P0DTA-DIAG: distinct stop reasons (before: all folded into fetch_failed / budget_exhausted)
    "R-ACQ-DEADLINE": "deadline_reached", "R-ACQ-DOC_BUDGET": "doc_budget_reached",
    "R-ACQ-RATE": "quota", "R-ACQ-DAILY": "quota", "R-ACQ-FETCH_TIMEOUT": "fetch_timeout",
    "R-ACQ-CRAWL_DELAY": "crawl_delay",
}
KNOWN_CODES = tuple(sorted(_CODE_MAP))


def reason_from_code(code: str) -> str:
    """Map an existing reason code to a terminal reason.

    Unknown/empty codes map to 'fetch_failed' (a not-used category); never 'admitted'.
    """
    if not isinstance(code, str):
        return "fetch_failed"
    return _CODE_MAP.get(code.strip().upper(), "fetch_failed")


# P0DTA-DIAG (additive): per-candidate acquisition diagnostics. Every discovered candidate ends in exactly one status.
DIAG_STATUSES = ("discovered_not_selected", "selected_not_fetched", "fetched_rejected", "acquired_not_admitted",
                 "skipped", "admitted")
DIAG_LABELS = {  # plain-language aggregate labels for the UI (no individual sources, no gate names)
    "discovered_not_selected": "found but not selected", "selected_not_fetched": "selected but not retrieved",
    "fetched_rejected": "retrieved but not usable", "acquired_not_admitted": "read but not used",
    "skipped": "not reached (time, limit or cap)", "admitted": "used"}
_DEFER, _KNOWN, _NODOI = "R-ACQ-DEFER_CAP", "R-ACQ-KNOWN", "R-ACQ-NODOI"
_PREFILTER = {"R-ACQ-NOTOA": "ineligible_not_open_access", "R-ACQ-NOLIC": "ineligible_licence_prefilter",
              _NODOI: "ineligible_no_identifier", "R-ACQ-DUPLICATE": "duplicate", _KNOWN: "duplicate_or_in_corpus",
              _DEFER: "ranked_below_cap"}


def public_id(rec: dict) -> str | None:
    """DOI, arXiv id, PMCID or URL (all public: URLPUBLIC-1023 / PMCIDPUBLIC-1023); first one present."""
    for k in ("doi", "arxiv_id", "pmcid", "url", "discovered_url"):
        v = rec.get(k) if isinstance(rec, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


_STAGE_BY_CLASS = {"no_route": "resolve", "already_in_corpus": "resolve", "fetch_failed": "fetch",
                   "fetch_timeout": "fetch", "crawl_delay": "fetch", "extraction_failed": "fetch", "abstract_only": "fetch",
                   "licence_rejected": "licence", "relevance_rejected": "relevance", "provenance_rejected": "admission",
                   "injection_rejected": "admission", "credibility_rejected": "admission",
                   "freshness_rejected": "admission"}


def diag_status(outcome, reason, *, quota: bool = False) -> tuple[str, str, str]:
    """(status, reason_code, stage) for one candidate record from the pipeline's own (outcome, reason). Pure; never
    raises. stage: resolve | fetch | licence | relevance | admission (where the candidate's path ended).
    quota: legacy fallback when no per-candidate skip reason was recorded. Nothing is upgraded to 'admitted' except an
    explicit admitted/reused outcome."""
    c = str(reason or "").strip().upper()
    o = outcome if isinstance(outcome, str) else None
    cls = reason_from_code(c)
    if o in ("admitted", "reused"):
        return "admitted", "admitted" if o == "admitted" else "reused_stored", "admission"
    if o in ("dropped", "deferred", "link_only"):
        return "discovered_not_selected", _PREFILTER.get(c, "ineligible"), "resolve"
    if o == "already_stored":
        return "discovered_not_selected", "duplicate_or_in_corpus", "resolve"
    if cls in ("quota", "doc_budget_reached", "deadline_reached", "candidate_cap") or o in ("not_evaluated", "not_reached", None):
        if cls == "quota" or (quota and cls != "deadline_reached" and cls != "doc_budget_reached"):
            return "skipped", "quota", "resolve"
        if cls == "doc_budget_reached":
            return "skipped", "doc_budget", "fetch"
        return "skipped", "deadline" if cls == "deadline_reached" else "candidate_cap", "fetch"
    if o == "not_used":
        return "acquired_not_admitted", "over_document_budget", "admission"
    stage = _STAGE_BY_CLASS.get(cls, "fetch")
    if c == "R-ACQ-RATELIMITED":  # the shared rate limiter skipped the request (nothing sent): skipped, not a failure.
        # terminal_reason stays fetch_failed (existing contract, DF-4 queues it as unreachable); status_reason is
        # "rate_limited" (the source/portal limiter), distinct from "quota" (this deployment's job/day caps).
        return "skipped", "rate_limited", "fetch"
    if o == "reuse_rejected" or cls == "relevance_rejected":
        return "acquired_not_admitted", cls, "relevance" if cls == "relevance_rejected" else "admission"
    if cls in ("no_route", "already_in_corpus", "crawl_delay"):
        return "selected_not_fetched", cls, stage
    return "fetched_rejected", cls, stage


def skip_class(code) -> str | None:
    """Terminal reason for a code that records why a candidate was left unfinished (else None)."""
    k = reason_from_code(code) if isinstance(code, str) else None
    return k if k in ("deadline_reached", "candidate_cap", "doc_budget_reached", "quota", "fetch_timeout",
                                      "crawl_delay") else None


def annotate_diag(rec: dict, *, quota: bool = False) -> dict:
    """Add public_id / status / status_reason / stage to a candidate record in place (existing keys untouched)."""
    st, why, stage = diag_status(rec.get("outcome"), rec.get("reason"), quota=quota)
    rec["public_id"], rec["status"], rec["status_reason"], rec["stage"] = public_id(rec), st, why, stage
    return rec


def diag_counts(records) -> dict:
    """Aggregate count per status (all six keys present)."""
    out = {k: 0 for k in DIAG_STATUSES}
    for r in records or ():
        if isinstance(r, dict) and r.get("status") in out:
            out[r["status"]] += 1
    return out


@dataclass
class PortalAttempt:
    portal: str
    status: str  # completed | timeout | quota | error | skipped
    n_results: int = 0


@dataclass
class CandidateOutcome:
    candidate_id: str
    ids: dict = field(default_factory=lambda: {k: None for k in ID_KEYS})
    terminal_reason: str = "fetch_failed"
    detail: str = ""


@dataclass
class ResearchOutcome:
    kind: str
    user_text: str
    complete: bool
    report_detail: dict


_CAUSE_TEXT = {
    "timeout": "a portal timed out",
    "quota": "a portal's request limit was reached",
    "error": "a portal had a service failure",
    "budget": "the search budget ran out",
    "awaiting_route": "some sources found have no usable identifier yet",  # CONT-1023: URL-only, queued to follow
    "deadline_reached": "the time limit for this search was reached",
    "candidate_cap": "more candidates were found than one request can check",
    "doc_budget_reached": "the limit on documents read per request was reached",
    "job_quota": "the limit on new searches was reached",
    "busy": "the service was busy with other searches",
    "budget_unknown": "the time needed for a new search could not be determined",
    "fetch_timeout": "a source took too long to respond",
    "crawl_delay": "a source asks crawlers to wait longer than this search could",
    "unreachable": "a source found could not be reached in time",  # DDGFIX-1023 DF-4: queued after a fetch failure
}


def queued_unreachable(code) -> bool:
    """DDGFIX-1023 DF-4: the reason a queued candidate was left unfinished is a fetch/transport failure (the source
    could not be reached), not the request's search budget (deadline, candidate cut, not evaluated)."""
    c = str(code or "").strip().upper()
    return c in ("R-ACQ-UNREACHABLE", "R-ACQ-RATELIMITED", "R-ACQ-ERROR") or c.startswith("R-ACQ-ERR-")


def _awaiting_route(candidates) -> bool:
    """CONT-1023: a URL-only candidate (no identifier yet) was queued to be followed in the background."""
    return any(c.terminal_reason == "queued_background" and reason_from_code(c.detail) == "no_route"
               for c in candidates)


def _causes(attempts, budget_exhausted: bool, awaiting: bool = False, unreachable: bool = False) -> list[str]:
    out: list[str] = []
    for a in attempts:
        if a.status in ("timeout", "quota", "error") and a.status not in out:
            out.append(a.status)
    if budget_exhausted:
        out.append("budget")
    if unreachable:
        out.append("unreachable")
    if awaiting:
        out.append("awaiting_route")
    return out


def _cause_phrase(causes: list[str]) -> str:
    return " and ".join(_CAUSE_TEXT[c] for c in causes)


def _detail(attempts, candidates, kind, complete, causes) -> dict:
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.terminal_reason] = counts.get(c.terminal_reason, 0) + 1
    return {
        "kind": kind,
        "complete": complete,
        "incomplete_causes": causes,
        "portal_attempts": [asdict(a) for a in attempts],
        "portals_attempted": [a.portal for a in attempts if a.status != "skipped"],
        "portals_completed": [a.portal for a in attempts if a.status == "completed"],
        "portals_skipped": [a.portal for a in attempts if a.status == "skipped"],
        "candidates": [asdict(c) for c in candidates],
        "terminal_reason_counts": counts,
        "zero_discoveries": len(candidates) == 0
        and sum(a.n_results for a in attempts) == 0,
    }


def summarize_outcome(attempts, candidates, *, budget_exhausted: bool,
                      older_evidence_available: bool,
                      new_evidence_admitted: bool, source_unreachable: bool = False,
                      extra_causes=()) -> ResearchOutcome:
    attempts, candidates = list(attempts), list(candidates)
    applicable = [a for a in attempts if a.status != "skipped"]
    awaiting = _awaiting_route(candidates)
    extra = [x for x in extra_causes if x in _CAUSE_TEXT]  # P0DTA-DIAG: specific stop causes (own sentences)
    complete = bool(applicable) and all(a.status == "completed" for a in applicable) \
        and not budget_exhausted and not awaiting and not source_unreachable and not extra
    causes = _causes(attempts, budget_exhausted, awaiting, source_unreachable)
    causes += [x for x in extra if x not in causes]
    if not applicable and not budget_exhausted and not extra:
        causes = ["none_searched"]
    admitted = new_evidence_admitted or any(c.terminal_reason == "admitted" for c in candidates)
    why = (" Coverage was incomplete because " + _cause_phrase([c for c in causes if c in _CAUSE_TEXT]) + "."
           if causes and not complete and any(c in _CAUSE_TEXT for c in causes) else "")

    if admitted:
        kind = "answered_with_new"
        text = "This answer uses newly found sources." + why
    elif older_evidence_available and not applicable:  # FRESH-1023 integration: never claim a search that did not run
        kind = "incomplete_search"
        text = ("This answer rests on older sources or sources whose date could not be verified; no search for "
                "newer sources could be run"
                + (" because the search budget ran out." if budget_exhausted
                   else " because " + _cause_phrase(extra) + "." if extra else "."))
    elif older_evidence_available:
        kind = "qualified_older_only"
        text = ("This answer rests on older sources; freshness research found no qualifying "
                "newer evidence." + why)
    elif not complete:
        kind = "incomplete_search"
        if causes and causes != ["none_searched"]:
            text = ("The search was incomplete because "
                    + _cause_phrase([c for c in causes if c in _CAUSE_TEXT])
                    + ", so no conclusion about available evidence can be drawn.")
        else:
            text = "No portal was searched, so no conclusion about available evidence can be drawn."
    else:
        acq = sum(1 for c in candidates if c.terminal_reason in _ACQ_FAILURES)
        if acq:
            kind = "abstain"
            text = (f"Sources were found but their full text could not be acquired "
                    f"({acq} of {len(candidates)}), so there is not enough usable evidence to answer.")
        else:
            kind = "no_valuable_source"
            if candidates:
                text = (NO_VALUABLE_SENTENCE + " Candidates were found but did not meet "
                        "the admission criteria.")
            else:
                text = NO_VALUABLE_SENTENCE + " No candidates were found."
    if any(c.terminal_reason == "queued_background" for c in candidates):  # CONT-1023: no claim they will succeed
        text = f"{text} {BACKGROUND_SENTENCE}".strip()
    return ResearchOutcome(kind=kind, user_text=text, complete=complete,
                           report_detail=_detail(attempts, candidates, kind, complete, causes))
