"""OD-DARINT-SCOPE (option 1): widened four-field scope text, scope set v2 shape, layer 1 unchanged (offline)."""
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


def test_four_fields_named_in_scope_prompt_and_decline_text():
    for text in (RAW["scope"], RAW["decline_template"], RAW["layer2"]["system_prompt"]):
        for word in ("epidemiology", "climate science", "reproducibility", "ecology"):
            assert word in text
    assert "AI in other fields" in RAW["layer2"]["system_prompt"]  # OUT list unchanged
    assert "scope_v2_note" in RAW and "poison" in RAW["scope_v2_note"] and "symptoms" in RAW["scope_v2_note"]


def test_set_v2_has_at_least_six_per_field_and_ten_negatives():
    c = Counter(i["field"] for i in SET["items"] if i["label"] == "in_scope")
    assert all(c[f] >= 6 for f in FIELDS), c
    assert sum(1 for i in SET["items"] if i["label"] == "out_of_scope") >= 10
    assert all(i["expected_verdict"] in ("accept", "decline") and i["field"] for i in SET["items"])


def test_layer1_deny_categories_unchanged_known_declines():
    known = [i for i in SET["items"] if i.get("expected_layer1")]
    assert known
    for i in known:
        assert sr.layer1(i["text"], CFG)[0] == sr.DECLINE, i["id"]


def test_layer1_never_accepts_an_out_of_scope_v2_item():
    for i in SET["items"]:
        if i["label"] == "out_of_scope":
            assert sr.layer1(i["text"], CFG)[0] != sr.ACCEPT, i["id"]
