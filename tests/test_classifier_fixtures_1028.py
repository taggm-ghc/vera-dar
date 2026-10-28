"""FIXTURES-1028: synthetic classifier-output strings P001-P008 (owner rule 4: invalid classifier output is nullified,
then retried or safely fallen back from, never treated as user misuse). Offline: the classifier is a stub returning the
fixture string; each row asserts the ACTUAL handling in vera/scope_router.py and vera/catstep.py."""
import pytest

from vera import catstep, scope_router as sr

Q = "How do jazz drummers practise brushes?"  # layer 1 UNSURE, no marker: reaches the classifier

# id -> (classifier_raw, intent as written in the fixture file)
FIXTURES = {
    "P001": ('{"in_scope":false,"category":"off_topic","category":"harmful"}', "nullify_then_safe_retry"),
    "P002": ('{"in_scope":true,"category":"off_topic"', "nullify_then_safe_retry"),
    "P003": ('{"in_scope":false,"category":"novel_subject"}', "unrecognized_category_handling"),
    "P004": ('{"in_scope":false,"reason":"Unfamiliar area"}', "unrecognized_category_handling_if_contract_valid"),
    "P005": ('{"in_scope":false,"category":"harmful"}', "existing_safety_policy"),
    "P006": ("I think this is off-topic.", "nullify_then_safe_retry"),
    "P007": ('{"in_scope":false,"category":"off_t\\u006fpic"}', "decode_then_apply_policy"),
    "P008": ('{"in_scope":false,"category":"off_topic","reason":"The text discusses the historical label harmful"}',
             "existing_refusal_token_policy_check"),
}

# id -> (classifier calls, reason, category, raw_refusal_named)
ACTUAL = {
    # duplicate key: strict decode -> malformed (scope_router.py:212-216, 295-299); raw text names "harmful" so
    # _raw_names_refusal (346-352) skips the Q2 retry (326): one call, fixed decline, not a refusal category
    "P001": (1, "classifier_malformed", "", False),
    "P002": (2, "classifier_malformed", "", False),   # bad JSON: nullified (336-340), retried once (317-326), still malformed
    "P003": (2, "llm_out:", "", False),               # unknown enum: parse_category "" (287-292) -> nullified, retry, then catstep path
    "P004": (2, "llm_out:Unfamiliar area", "", False),  # absent category: same
    "P005": (1, "llm_out:", "harmful", False),        # in_scope=false + refusal category: not nullified; REFUSE_CATEGORIES
    "P006": (2, "classifier_malformed", "", False),   # prose: not JSON -> malformed, retried once
    "P007": (1, "llm_out:", "off_topic", False),      # json decodes the escape to off_topic (ASCII): usable, no retry
    "P008": (1, "llm_out:The text discusses the historical label harmful", "off_topic", False),  # prose reason: no refusal_token
}


def run(raw):
    calls = []

    def chat(messages, model, l2):
        calls.append(1)
        return raw, 10, 5
    d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    return d, len(calls)


@pytest.mark.parametrize("pid", sorted(FIXTURES))
def test_fixture_actual_handling(pid):
    d, n = run(FIXTURES[pid][0])
    calls, reason, cat, named = ACTUAL[pid]
    assert (n, d.llm_calls) == (calls, calls)
    assert d.accepted is False and d.layer == "layer2_llm"
    assert d.reason == reason and d.category == cat and d.raw_refusal_named is named


@pytest.mark.parametrize("pid", ["P001", "P002", "P006"])
def test_invalid_output_is_nullified_never_a_refusal_category(pid):
    d, _ = run(FIXTURES[pid][0])
    assert d.reason == "classifier_malformed" and d.category not in sr.REFUSE_CATEGORIES
    assert not catstep.applies(d, Q)  # the safe fixed decline; the category step never runs on it


@pytest.mark.parametrize("pid", ["P003", "P004"])
def test_unrecognised_or_absent_category_goes_to_catstep(pid):
    d, _ = run(FIXTURES[pid][0])
    assert catstep.applies(d, Q) is True


@pytest.mark.parametrize("pid", ["P005", "P007", "P008"])
def test_other_rows_do_not_reach_catstep(pid):
    d, _ = run(FIXTURES[pid][0])
    assert catstep.applies(d, Q) is False


def test_p005_refusal_category_and_p008_reason_prose_not_refused():
    assert run(FIXTURES["P005"][0])[0].category in sr.REFUSE_CATEGORIES
    assert sr.refusal_token(FIXTURES["P008"][0]) == ""
    assert sr.refusal_token(FIXTURES["P001"][0]) == "harmful" or sr._strict_loads(FIXTURES["P001"][0]) is None
