"""DARB-GAPS-1023: offline tests closing the low-severity gaps from the DAR-B independent verification.
(1) doaj_id survives the no-abstract (traced original) branch of grounded_ask.retrieve_all.
(2) each uncited-answer guard in fulldoc_answer pinned on its own: the per-part guard (`if not fixed`) and the whole-draft
    guard (`count_valid_citations(...) == 0`, run with per_part OFF on a COPY of the shipped config, the rollback config).
Stubbed model/checker, no network, no DB."""
import copy
import json

import pytest

from tests.test_fulldoc_answer import bundle as ev_bundle
from vera import fulldoc_answer as fa
from vera import fulldoc_corpus, grounded_ask
from vera.ask_service import AskResult

GCFG = {"removed_citation_note": "Note: bad citations removed.",
        "claim_check": {"enabled": True, "timeout_s": 5, "max_claims": 40,
                        "checkers": [{"provider": "t", "model": "m", "api_key_env": "FD_TEST_KEY", "base_url": None}]}}


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


# ------------------------------------------------------------------------------------------------- (1) doaj_id, no-abstract branch
def test_doaj_id_and_other_identifiers_kept_for_traced_original_without_abstract():
    cfg = {"num_results": 5, "snippet_max_chars": 1500, "providers": [], "deadline_s": 10, "resolve": {"enabled": True, "deadline_s": 10, "max_results_to_resolve": 3}}
    rec = {"title": "Traced original", "url": "https://example.org/orig", "published": "2024-01-02", "snippet": "",
           "discovered_via": "reference", "doi": "10.1234/abc", "pmcid": "PMC123", "openalex_id": "https://openalex.org/W9",
           "doaj_id": "doajABC123", "provider": "doaj"}
    out, unused = grounded_ask.retrieve_all("q", cfg, search=lambda q: [rec], resolver=lambda r, rc, d: [dict(r)])  # link-only record, resolved to itself
    assert out == [] and len(unused) == 1                     # shown, never grounded on
    u = unused[0]
    assert u["doaj_id"] == "doajABC123", f"doaj_id dropped on the no-abstract branch; keys={sorted(u)}"
    assert (u["doi"], u["pmcid"], u["openalex_id"], u["provider"]) == ("10.1234/abc", "PMC123", "https://openalex.org/W9", "doaj")


# ------------------------------------------------------------------------------------------------- (2) uncited-answer guards
def _ask(cfg, text):
    def check(chk, msgs):  # a checker that would call everything supported: the guard must fire before it matters
        claims = json.loads(msgs[1]["content"])["claims"]
        return json.dumps({"claims": [{"id": c["id"], "verdict": "supported", "reason": "r"} for c in claims]})
    return fa.answer("What reduces latency?", ev_bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=cfg,
                     complete_fn=lambda *a, **k: AskResult(text, 10, 0.01), check_call=check)


def test_per_part_guard_uncited_parts_fall_back_no_valid_citations():
    """fulldoc_answer._answer_parts `if not fixed`: structured per-part reply whose answers carry no valid citation."""
    cfg = copy.deepcopy(fulldoc_corpus.load_config())
    assert fa.per_part_enabled(cfg)                           # the shipped state
    tok = cfg["fallback_tokens"]["no_valid_citations"]
    text = "Parts:\nP1: What reduces latency?\nAnswers:\nP1: Alpha systems cut latency a lot, with no citation at all."
    o = _ask(cfg, text)
    assert o.result is None and o.fallback == tok
    o = _ask(cfg, text.replace("no citation at all.", "a bad citation [9]."))
    assert o.result is None and o.fallback == tok


def test_whole_draft_guard_uncited_answer_with_per_part_off_config_copy():
    """fulldoc_answer._answer `count_valid_citations(...) == 0`, reached only with per_part OFF (the rollback config),
    driven by a COPY of the shipped config (shipped file untouched)."""
    cfg = copy.deepcopy(fulldoc_corpus.load_config())
    cfg["answer"]["per_part"]["enabled"] = False
    assert not fa.per_part_enabled(cfg) and fulldoc_corpus.load_config()["answer"]["per_part"]["enabled"] is True
    tok = cfg["fallback_tokens"]["no_valid_citations"]
    o = _ask(cfg, "Alpha systems cut latency a lot, with no citation at all.")
    assert o.result is None and o.fallback == tok
    o = _ask(cfg, "Alpha systems cut latency a lot [9].")
    assert o.result is None and o.fallback == tok
    ok = _ask(cfg, "Alpha systems reduce latency by forty percent in field trials [1].")  # control: a valid citation passes the guard
    assert ok.fallback != tok
