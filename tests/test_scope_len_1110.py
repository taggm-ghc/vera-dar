"""SCOPE-LEN-1110: the layer-2 classifier (and catstep) window covers the whole question the API accepts. Offline."""
import json

import pytest

import main
from tests.test_fix1_offtopic import PRICING
from tests.test_multiq_1108 import ON, ask  # noqa: F401  (the fixture)
from vera import ask_service, catstep, scope_router as sr

RAW = json.loads(sr.CONFIG_PATH.read_text())
CFG = sr.parse_config(RAW, "llm")
REAL_ROUTE = sr.route
SENT = "TAILSENTINEL"
SAFE = '{"safe_request": true, "category": "safe_request", "reason": "ok"}'
HARM = '{"safe_request": false, "category": "harmful", "reason": "x"}'


def _sentinel_chat(seen=None):
    def chat(messages, model, l2):
        text = messages[-1]["content"]
        if seen is not None:
            seen.append(messages)
        return (HARM if SENT in text else SAFE), 5, 2
    return chat


def test_window_covers_the_api_limit():
    limit = main.AskRequest.model_fields["question"].metadata[0].max_length
    assert limit == sr.API_QUESTION_MAX_CHARS == 4000
    assert sr.load_config().layer2["max_question_chars"] >= limit


@pytest.mark.parametrize("bad", [2000, 3999, "4000", True, None])
def test_misconfigured_window_fails_at_load(bad):
    raw = json.loads(json.dumps(RAW))
    raw["layer2"]["max_question_chars"] = bad
    with pytest.raises(sr.ScopeConfigError):
        sr.parse_config(raw, "llm")


def test_route_declines_when_window_is_smaller_than_the_question():
    import dataclasses
    small = dataclasses.replace(CFG, layer2={**CFG.layer2, "max_question_chars": 2000})
    seen = []
    d = sr.route("What is photosynthesis? " + "a" * 2500, "m", None, cfg=small, chat=_sentinel_chat(seen))
    assert not d.accepted and d.reason == "deny:classifier_window" and seen == []


def test_marker_at_the_end_of_a_4000_char_question_reaches_classifier_and_catstep():
    q = "a" * (4000 - len(SENT)) + SENT
    assert len(q) == 4000
    assert SENT in sr.classifier_messages(q, CFG)[1]["content"]
    cfg = sr.parse_config(json.loads(json.dumps(RAW)), "llm")
    raw = json.loads(json.dumps(RAW))
    raw["unknown_category_step"]["llm_step"] = True
    seen = []
    scfg = sr.parse_config(raw, "llm")
    catstep.reset_for_tests()
    try:
        catstep.categorise(q, scfg, chat=lambda m, mo, l2: (seen.append(m) or ('{"field": "other"}', 1, 1)), model="m", pricing=PRICING)
    finally:
        catstep.reset_for_tests()
    assert seen and SENT in seen[0][1]["content"]


def test_benign_head_harmful_tail_is_declined():
    q = "How do sourdough starters ferment flour? " + "b" * 1950 + " " + SENT
    assert len(q) > 2000
    seen = []
    d = sr.route(q, "m", None, cfg=CFG, chat=_sentinel_chat(seen))
    assert not d.accepted and d.category == "harmful" and seen


def test_multi_question_prompt_with_second_question_after_2000_chars_is_declined_whole(ask, monkeypatch):
    seen = []
    chat = _sentinel_chat(seen)

    def route(original, *a, **k):
        ask["route"] += 1
        return REAL_ROUTE(original, "m", None, cfg=CFG, chat=chat)
    monkeypatch.setattr(sr, "route", route)
    q = "What is CRISPR? " + "c" * 2100 + " Is it used in agriculture? " + SENT + "?"
    assert len(q) > 2000 and q.index("Is it used") > 2000
    r = ask_service.answer_in_scope(q, "m", PRICING)
    assert ask["route"] == 1 and len(seen) == 1 and SENT in seen[0][1]["content"]
    assert r.result_kind == "decline" and ask["parts"] == [] and "parts" not in (r.contract or {})
