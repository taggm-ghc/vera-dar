"""POLISH-1110: retrieve_all carries landing_url through to the listed candidate record (P-1110-11 wording is in test_arch_fix_1110_bd)."""
from vera import grounded_ask as ga

CFG = {"enabled": True, "providers": ["p"], "num_results": 5, "deadline_s": 5, "snippet_max_chars": 300,
       "resolve": {"enabled": False}}


def test_landing_url_survives_retrieve_all():
    rows = [{"title": "Alpha study", "url": "https://example.org/a.pdf", "snippet": "alpha study text",
             "landing_url": "https://example.org/a"},
            {"title": "Alpha beta", "url": "https://example.org/b", "snippet": "alpha beta text"}]
    out, _ = ga.retrieve_all("alpha study", CFG, search=lambda _q: rows)
    by = {s["url"]: s for s in out}
    assert by["https://example.org/a.pdf"]["landing_url"] == "https://example.org/a"
    assert "landing_url" not in by["https://example.org/b"]


def test_seed_exclusion_is_not_a_used_up_allowance():
    from vera import research_outcome as ro, search_providers as sp
    assert sp._own_decision(RuntimeError("rate_limited (seed_excluded; not an attempt)")) == "seed_excluded"
    assert sp._own_decision(RuntimeError("rate_limited (daily_cap; not an attempt)")) == "own_cap"
    res = ro.summarize_outcome([ro.PortalAttempt("duckduckgo", "seed_excluded", 0)], [], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False)
    assert "ackground seeding does not use that search service" in res.user_text
    assert "used up" not in res.user_text


def test_wait_too_long_sentence_is_complete_as_rendered():
    from vera import research_outcome as ro
    c = ro.CandidateOutcome("c1", terminal_reason="queued_background", detail="R-ACQ-RATELIMITED-WAIT_TOO_LONG")
    res = ro.summarize_outcome([ro.PortalAttempt("openalex", "completed", 2)], [c], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False, source_unreachable=True)
    assert ("A source found was not fetched because that site's rate limit required a longer wait than the time "
            "left for this question.") in res.user_text
