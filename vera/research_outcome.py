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
)
ID_KEYS = ("doi", "pmcid", "openalex_id", "arxiv_id", "url")
KINDS = ("answered_with_new", "qualified_older_only", "no_valuable_source",
         "incomplete_search", "abstain")

NO_VALUABLE_SENTENCE = "No sufficiently valuable source was found across the portals searched."

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
    "R-ACQ-DEADLINE": "fetch_failed", "R-ACQ-ERROR": "fetch_failed",
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
    # provenance / identity / integrity
    "R-ACQ-IDENTITY_MISMATCH": "provenance_rejected",
    "R-ACQ-DUPLICATE": "provenance_rejected",
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
    "R-ACQ-DEFER_CAP": "budget_exhausted", "R-ACQ-SERVECAP": "budget_exhausted",
}
KNOWN_CODES = tuple(sorted(_CODE_MAP))


def reason_from_code(code: str) -> str:
    """Map an existing reason code to a terminal reason.

    Unknown/empty codes map to 'fetch_failed' (a not-used category); never 'admitted'.
    """
    if not isinstance(code, str):
        return "fetch_failed"
    return _CODE_MAP.get(code.strip().upper(), "fetch_failed")


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
}


def _causes(attempts, budget_exhausted: bool) -> list[str]:
    out: list[str] = []
    for a in attempts:
        if a.status in ("timeout", "quota", "error") and a.status not in out:
            out.append(a.status)
    if budget_exhausted:
        out.append("budget")
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
                      new_evidence_admitted: bool) -> ResearchOutcome:
    attempts, candidates = list(attempts), list(candidates)
    applicable = [a for a in attempts if a.status != "skipped"]
    complete = bool(applicable) and all(a.status == "completed" for a in applicable) \
        and not budget_exhausted
    causes = _causes(attempts, budget_exhausted)
    if not applicable and not budget_exhausted:
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
                "newer sources could be run" + (" because the search budget ran out." if budget_exhausted else "."))
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
    return ResearchOutcome(kind=kind, user_text=text, complete=complete,
                           report_detail=_detail(attempts, candidates, kind, complete, causes))
