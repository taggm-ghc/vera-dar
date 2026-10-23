"""DDFIX-1023/INTEG item 6 (owner 2026-10-23 objected to unrelated records in the reading list): records this request's
inline acquisition marked R-ACQ-IRRELEVANT are not listed while at least one other record remains. OFFLINE; SYNTHETIC
records (DOIs 10.9999/...)."""
from types import SimpleNamespace

from vera import ask_service
from vera import answer_contract as contract
from vera.grounded_ask import drop_irrelevant, irrelevant_marks

IRR = "R-ACQ-IRRELEVANT"
REL = {"n": 1, "title": "Orchestrating multi-agent LLM systems", "url": "https://doi.org/10.9999/rel", "doi": "10.9999/rel"}
OFF = {"n": 2, "title": "The U.S. Dollar Dominance: Origins, Status Quo", "url": "https://doi.org/10.9999/usd",
       "doi": "10.9999/usd"}
OFF2 = {"n": 3, "title": "Pedestrian Archetypes Extension", "url": "https://arxiv.org/abs/2607.16922"}
INLINE = {"status": "none",
          "discovered": [{"url": "https://doi.org/10.9999/usd", "outcome": "dropped", "reason": IRR},
                         {"url": "https://doi.org/10.9999/rel", "outcome": "deferred", "reason": "R-ACQ-DEFER_CAP"}],
          "candidates": [{"ids": {"url": "https://arxiv.org/abs/2607.16922/"}, "reason": IRR}]}


def test_marks_are_read_from_discovered_and_candidates_by_url():
    m = irrelevant_marks(INLINE)
    assert ("url", "https://doi.org/10.9999/usd") in m and ("url", "https://arxiv.org/abs/2607.16922") in m
    assert ("url", "https://doi.org/10.9999/rel") not in m


def test_irrelevant_records_are_dropped_when_a_relevant_one_remains():
    assert drop_irrelevant([REL, OFF, OFF2], INLINE) == [REL]


def test_all_irrelevant_keeps_the_list_unchanged():
    assert drop_irrelevant([OFF, OFF2], INLINE) == [OFF, OFF2]


def test_no_marks_or_no_inline_keeps_everything():
    assert drop_irrelevant([REL, OFF], None) == [REL, OFF]
    assert drop_irrelevant([REL, OFF], {"status": "off"}) == [REL, OFF]


def test_stored_corpus_items_are_never_dropped():
    stored = {**OFF, "span": "x"}
    assert drop_irrelevant([REL, stored], INLINE) == [REL, stored]


def test_user_facing_reading_list_hides_the_irrelevant_record():
    sources = [REL, OFF, OFF2]
    grounding = (lambda **k: sources, {"no_sources_text": "none"})
    dec = SimpleNamespace(tokens_used=0, cost_usd=0.0)
    tok = ask_service._INLINE_OUT.set(INLINE)
    try:
        res = ask_service._reading_list("q", "m", {}, dec, {}, grounding, contract)
    finally:
        ask_service._INLINE_OUT.reset(tok)
    titles = [c["title"] for c in res.candidates]
    assert titles == [REL["title"]]
    assert all(s.get("doi") != "10.9999/usd" for s in (res.sources or ()))
