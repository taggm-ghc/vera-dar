"""DDFIX-1023/INTEG item 5: owner decisions 2026-10-23 "34% with idiom strip" (admission.min_question_coverage) and
"drop it to 34% too" (admission.min_relevance, the per-passage gate). OFFLINE: shipped config/fulldoc_answer.json,
fake corpus (tests/test_fix2_verifier_1022._Corpus); no network, DB or model. Fixture texts are SYNTHETIC."""
import copy

import pytest

from tests.test_fix2_verifier_1022 import CFG, _Corpus, _icfg_n
from vera import fulldoc_admission as fa

Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
# 20 content stems; a passage matching 7 of them covers 0.35 (admitted at 0.34, rejected at the former 0.40)
Q20 = ("How do glaciers, permafrost, wetlands, mangroves, coral reefs, seagrass meadows, peatlands, savannas, tundra, "
       "rainforests, deserts, estuaries, lagoons, fjords, volcanoes and aquifers store carbon?")
SPAN_35 = "Peatlands, wetlands, mangroves, seagrass and estuaries store large amounts of carbon in waterlogged soils."
SPAN_30 = "Peatlands, wetlands, mangroves, seagrass and estuaries hold large amounts of carbon in waterlogged soils."
# the ACCEPT multi-agent fixture vocabulary (tests/test_ddfix_acceptance.py MAS / REL snippets)
ACCEPT_SPAN = ("Multi-agent orchestration frameworks assign agent archetypes such as planner, supervisor, worker and "
               "critic roles; a supervisor delegates subtasks to worker agents through hierarchical coordination.")
DOLLAR = ("The status quo of the international monetary system rests on de facto dollar dominance, with implications "
          "for European best practices in reserve management.")


def _admit(q, title, spans, cfg=None):
    return fa.admit(q, _Corpus(title, spans), copy.deepcopy(cfg or CFG))


def test_shipped_values_are_the_owner_decisions_and_the_stem_floor_is_unchanged():
    adm = CFG["admission"]
    assert adm["min_question_coverage"] == 0.34 and adm["min_relevance"] == 0.34
    assert adm["min_matched_stems"] == 2
    assert "_min_question_coverage_owner_note" in adm and "_min_relevance_owner_note" in adm


def test_coverage_ok_is_the_single_document_rule_at_034():
    icfg, n = _icfg_n()
    q = fa._q_stems(Q20, icfg, n)
    assert len(q) == 20
    hit = set(sorted(q)[:7])
    assert fa.coverage_ok(hit, q, CFG["admission"]["min_question_coverage"])  # 0.35
    assert not fa.coverage_ok(set(sorted(q)[:6]), q, CFG["admission"]["min_question_coverage"])  # 0.30


def test_a_passage_at_35_percent_coverage_is_admitted_and_30_percent_is_not():
    icfg, n = _icfg_n()
    assert fa._matched(Q20, SPAN_35, icfg, n) == (7, 20) and fa._matched(Q20, SPAN_30, icfg, n) == (6, 20)
    assert _admit(Q20, "Blue carbon ecosystems", [SPAN_35]).admitted
    assert not _admit(Q20, "Blue carbon ecosystems", [SPAN_30]).admitted
    old = copy.deepcopy(CFG)
    old["admission"]["min_relevance"] = 0.4  # the former per-passage value rejected the same passage
    assert not _admit(Q20, "Blue carbon ecosystems", [SPAN_35], old).admitted


def test_accept_multi_agent_fixture_is_admitted_with_idiom_strip():
    icfg, n = _icfg_n()
    q = fa._q_stems(Q, icfg, n)
    assert not {"facto", "statu", "quo", "pract"} & q  # idiom strip (REL)
    b = _admit(Q, "SYNTHETIC orchestrator-worker and supervisor patterns for multi-agent LLM orchestration",
               [ACCEPT_SPAN])
    assert b.admitted, b.rejected


@pytest.mark.parametrize("title", ["The U.S. Dollar Dominance: Origins, Status Quo, and Implications for Europe"])
def test_us_dollar_status_quo_text_is_still_rejected(title):
    b = _admit(Q, title, [DOLLAR])
    assert not b.admitted, b.admitted
