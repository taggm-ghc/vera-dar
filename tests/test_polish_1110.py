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


# Replicated from work/SEED-SCI-1110/scripts/seed_run.py CAP_RE (the seeder's cap-notice classifier; a match stops the run and backs off to 00:00 UTC)
import re
_SEED_CAP_RE = re.compile(
    r"its hourly or daily search limit has been reached; no background search was started"
    r"|daily search allowance for [^.\n]{1,160}? had already been used up"
    r"|The current search limit was reached; try again later\.")


def test_seed_ddg_refusal_end_to_end_is_not_a_cap_notice():
    import json
    from vera import rate_limit, request_origin, research_outcome as ro, search_providers as sp
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={})))
    try:
        from vera.eval_config import load_eval_config
        sent = []
        p = sp.DuckDuckGoLiteProvider(load_eval_config().search["duckduckgo_lite"], transport=lambda *a: sent.append(1),
                                      today=lambda: "2026-11-10", sleep=lambda s: None, robots_check=lambda u, ua: True)
        with request_origin.scope("seed"):
            try:
                p.call("coral reef bleaching", 5, 5.0)
                raise AssertionError("seed call to DuckDuckGo was not refused")
            except sp.ProviderUnavailable as e:
                status = sp._own_decision(e)
    finally:
        rate_limit.set_registry_for_tests(None)
    assert status == "seed_excluded" and sent == []
    res = ro.summarize_outcome([ro.PortalAttempt("duckduckgo", status, 0)], [], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False)
    text = res.user_text + json.dumps(res.report_detail)
    assert "ackground seeding does not use that search service" in text
    assert "allowance" not in text and "used up" not in text
    assert not _SEED_CAP_RE.search(res.user_text)
