"""OD-DARINT-SCOPE (option 1): scope set v2 shape and layer-1 behaviour on it (offline). The four-field scope text
itself was removed by SCOPE-GATE-1104 S1 (2026-11-04)."""
import json
from collections import Counter
from pathlib import Path

import pytest

from vera import scope_router as sr

ROOT = Path(__file__).resolve().parent.parent
RAW = json.loads((ROOT / "config" / "scope_router.json").read_text())
SET = json.loads((ROOT / "config" / "scope_set_v2.json").read_text())
FIELDS = ("ai_software_development", "epidemiology", "climate_science",
          "research_reproducibility_software_practice", "ecology")
CFG = sr.parse_config(RAW, "llm")


def test_no_field_list_in_config_prompt_or_decline_text():
    # SCOPE-GATE-1104 S1 (requirement change; owner 2026-11-04: "there should be no Fixed Scope text"): the four-field
    # scope string and the prompt's IN/OUT field lists are removed; the v2 note is kept as Superseded history
    assert "scope" not in RAW
    for text in (RAW["layer2"]["system_prompt"], RAW["decline_template"]):
        for word in ("epidemiology", "climate science", "reproducibility", "ecology", "AI in other fields", "scope"):
            assert word not in text
    # INTEGRATE-1104 round 2 (orchestrator): the superseded v2 note (it held the old field list) is removed from the
    # shipped config too; its history stays in version control
    assert "scope_v2_note" not in RAW


def test_set_v2_has_at_least_six_per_field_and_ten_negatives():
    c = Counter(i["field"] for i in SET["items"] if i["label"] == "in_scope")
    assert all(c[f] >= 6 for f in FIELDS), c
    assert sum(1 for i in SET["items"] if i["label"] == "out_of_scope") >= 10
    assert all(i["expected_verdict"] in ("accept", "decline") and i["field"] for i in SET["items"])


def test_layer1_known_topic_word_declines_fixed_by_scope_1022():
    """SCOPE-1022 (owner HITL-F1: B): the v2 items recorded as known layer-1 false declines (symptoms, diagnosis,
    poison) are in-scope research questions and must now pass layer 1 (the frozen set's expected_layer1 is history)."""
    known = [i for i in SET["items"] if i.get("expected_layer1")]
    assert known
    for i in known:
        assert sr.layer1(i["text"], CFG)[0] != sr.DECLINE, i["id"]


def test_layer1_never_declines_an_in_scope_v2_item():
    for i in SET["items"]:
        if i["label"] == "in_scope":
            assert sr.layer1(i["text"], CFG)[0] != sr.DECLINE, i["id"]


def test_layer1_never_accepts_an_out_of_scope_v2_item():
    for i in SET["items"]:
        if i["label"] == "out_of_scope":
            assert sr.layer1(i["text"], CFG)[0] != sr.ACCEPT, i["id"]
