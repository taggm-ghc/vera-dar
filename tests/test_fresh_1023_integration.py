"""FRESH-1023 integration: both triggers on RELEVANT evidence, trigger independence, and the response projection."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from vera import research_record as rr

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
CFG = {"enabled": True, "freshness_max_age_hours": 24, "source_authored_partial_dates_trusted": True,
       "sufficiency": {"partial_is_insufficient": True}}


def _res(admitted, partial=False):
    return SimpleNamespace(result_kind="answer", evidence=SimpleNamespace(admitted=admitted),
                           claim_check={"answer_completeness": "partial"} if partial else {})


def test_partial_answer_over_relevant_stale_evidence_fires_both_triggers():
    adm = [{"document_id": "d1", "published": "2019", "corpus_added_at": (NOW - timedelta(hours=30)).isoformat()}]
    fr = rr.assess(adm, CFG, NOW)
    assert rr.triggers(rr.sufficient(_res(adm, partial=True), CFG), fr) == ["insufficient", "stale"]


def test_fresh_and_sufficient_fires_nothing_and_authoring_time_is_ignored_for_the_trigger():
    # was: ..._ingest_time_is_ignored. Owner clarification: the corpus-added time is the trigger basis; the authoring date
    # only labels older/recent.
    adm = [{"document_id": "d1", "published": (NOW - timedelta(hours=2)).isoformat(),
            "corpus_added_at": (NOW - timedelta(hours=2)).isoformat()}]
    assert rr.triggers(rr.sufficient(_res(adm), CFG), rr.assess(adm, CFG, NOW)) == []
    old = [{"document_id": "d2", "published": "2019-05-01", "corpus_added_at": NOW.isoformat()}]
    fr = rr.assess(old, CFG, NOW)
    assert fr["status"] == "fresh" and fr["documents"][0]["evidence_recency"] == "older"
    stale = [{"document_id": "d3", "published": NOW.isoformat(), "corpus_added_at": (NOW - timedelta(hours=25)).isoformat()}]
    fr = rr.assess(stale, CFG, NOW)
    assert fr["status"] == "stale" and fr["documents"][0]["evidence_recency"] == "recent"


def test_no_admitted_chunks_fires_insufficient():
    assert rr.triggers(rr.sufficient(_res([]), CFG), rr.assess([], CFG, NOW)) == ["insufficient"]


def test_incomplete_coverage_never_claims_all_portals():
    rec = rr.summarize(triggers=["stale"], inline={"status": "none", "candidates": [],
                                                   "portals": [{"portal": "openalex", "status": "completed"},
                                                               {"portal": "europepmc", "status": "timeout"}]},
                       used_new=False, older_available=False, fr_before=None)
    assert rec["kind"] == "incomplete_search" and rec["complete"] is False
    assert "all portals" not in rec["user_text"].lower()


def test_budget_refusal_is_reported_not_equated_with_no_research():
    rec = rr.summarize(triggers=["stale"], inline={"status": "rate"}, used_new=False, older_available=True,
                       fr_before=None)
    assert rec["kind"] == "incomplete_search" and "limit on new searches" in rec["user_text"]  # P0DTA-DIAG: explicit quota sentence
    assert "found no qualifying" not in rec["user_text"]


def test_public_fields_are_additive_projection():
    rec = rr.summarize(triggers=[], inline=None, used_new=False, older_available=True,
                       fr_before=rr.assess([{"document_id": "d", "published": NOW.isoformat(),
                                             "corpus_added_at": NOW.isoformat()}], CFG, NOW))
    pf = rr.public_fields(rec)
    assert set(pf) == {"triggers", "freshness", "research_outcome"} and pf["triggers"] == []
    assert pf["freshness"]["any_stale_or_unknown"] is False and pf["research_outcome"]["kind"] == "not_needed"
    assert pf["freshness"]["trigger_basis"] == "corpus_added_at" and "_backfill" not in json.dumps(pf)
