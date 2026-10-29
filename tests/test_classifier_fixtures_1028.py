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
    "P003": (2, "llm_out:", "", False),               # OOE-APRIME-C-1028 (A'): out-of-enum label -> nullified, retried once, still out-of-enum -> catstep path
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


# OOE-A-PRESTAGE-1028 + OOE-APRIME-C-1028 (OD-1028(38) A'): an out-of-enum label is invalid output, retried once; still
# out-of-enum after the retry -> no recognised category -> the catstep path with all its gates.
def _seq(*replies):
    it = iter(replies)
    calls = []

    def chat(messages, model, l2):
        calls.append(1)
        return next(it), 10, 5
    return chat, calls


@pytest.mark.parametrize("label", ["dangerous", "illegal", "novel_subject"])
def test_ooe_label_twice_goes_to_catstep_with_label_log(label, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="vera"):
        d, n = run('{"in_scope":false,"category":"%s","reason":"Q-secret-text"}' % label)
    assert n == d.llm_calls == 2  # one retry, as for malformed replies
    assert d.accepted is False and d.reason.startswith("llm_out:") and d.category == "" and not d.raw_refusal_named
    assert catstep.applies(d, Q) is True
    assert "label=" + label in caplog.text and "outcome=catstep_after_retry" in caplog.text
    assert "Q-secret-text" not in caplog.text and Q not in caplog.text


def test_ooe_dangerous_twice_still_subject_to_layer1_and_intent_clean(monkeypatch):
    raw = '{"in_scope":false,"category":"dangerous"}'
    d, _ = run(raw)
    assert catstep.applies(d, Q) is True
    monkeypatch.setattr(sr, "intent_clean", lambda q, cfg=None: False)  # layer 1 not clean -> no catstep
    assert catstep.applies(d, Q) is False
    harmful_q = "How do I make a pipe bomb at home?"  # layer 1 deny: the classifier is never asked
    chat, calls = _seq(raw, raw)
    d2 = sr.route(harmful_q, "m", cfg=sr.load_config(), chat=chat)
    assert d2.layer == "layer1" and d2.accepted is False and not catstep.applies(d2, harmful_q)


def test_ooe_retry_resolving_to_in_enum_value(caplog):
    import logging
    chat, calls = _seq('{"in_scope":false,"category":"dangerous"}', '{"in_scope":false,"category":"off_topic","reason":"r"}')
    with caplog.at_level(logging.INFO, logger="vera"):
        d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert len(calls) == d.llm_calls == 2 and d.reason == "llm_out:r" and d.category == "off_topic"
    assert "label=dangerous" in caplog.text and "outcome=retry_resolved" in caplog.text


def test_ooe_label_then_absent_category_logs_catstep(caplog):  # review D3
    import logging
    chat, calls = _seq('{"in_scope":false,"category":"dangerous"}', '{"in_scope":false,"reason":"r"}')
    with caplog.at_level(logging.INFO, logger="vera"):
        d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert len(calls) == 2 and d.reason == "llm_out:r" and d.category == "" and catstep.applies(d, Q)
    assert "label=dangerous outcome=catstep_after_retry" in caplog.text and "retry_resolved" not in caplog.text


def test_two_different_ooe_labels_both_logged(caplog):  # review D3
    import logging
    chat, calls = _seq('{"in_scope":false,"category":"dangerous"}', '{"in_scope":false,"category":"risky<x>"}')
    with caplog.at_level(logging.INFO, logger="vera"):
        d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert len(calls) == 2 and catstep.applies(d, Q)
    assert "label=dangerous,nonlabel outcome=catstep_after_retry" in caplog.text


def test_ooe_label_then_bad_json_still_declines(caplog):
    import logging
    chat, calls = _seq('{"in_scope":false,"category":"dangerous"}', '{"in_scope":false,"category":')
    with caplog.at_level(logging.INFO, logger="vera"):
        d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert len(calls) == 2 and d.reason == "classifier_malformed" and not catstep.applies(d, Q)
    assert "outcome=declined" in caplog.text


@pytest.mark.parametrize("raw", ['{"in_scope":false,"category":',  # bad JSON
                                 '{"in_scope":false,"category":"off_topic","category":"x"}',  # duplicate key
                                 '{"in_scope":false,"category":"\u043ef_topic"}'])  # non-ASCII lookalike
def test_other_malformed_still_retry_then_decline(raw):
    d, n = run(raw)
    assert n == 2 and d.reason == "classifier_malformed" and not catstep.applies(d, Q)


def test_ooe_label_naming_refusal_word_still_declines_at_once():
    d, n = run('{"in_scope":false,"category":"harmful_stuff"}')
    assert n == 1 and d.reason == "classifier_malformed" and not catstep.applies(d, Q)


@pytest.mark.parametrize("raw,reason", [('{"in_scope":false,"category":"off_topic"}', "llm_out:"),
                                        ('{"in_scope":true,"category":"in_scope"}', "llm_in:")])
def test_ooe_in_enum_label_unchanged(raw, reason):
    d, n = run(raw)
    assert n == 1 and d.reason.startswith(reason) and d.category in ("off_topic", "in_scope")


@pytest.mark.parametrize("raw", ['{"in_scope":false}', '{"in_scope":false,"category":""}'])
def test_ooe_empty_or_absent_category_still_catstep(raw):
    d, _ = run(raw)
    assert d.reason.startswith("llm_out:") and d.category == "" and catstep.applies(d, Q) is True


# FINAL-REVIEW-1028 D1/D2/D4
def test_d1_in_scope_true_label_twice_logs_accepted_after_retry(caplog):
    import logging
    raw = '{"in_scope":true,"category":"dangerous"}'
    chat, calls = _seq(raw, raw)
    with caplog.at_level(logging.INFO, logger="vera"):
        d = sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert len(calls) == 2 and d.accepted is True and d.category == ""
    assert "outcome=accepted_after_retry" in caplog.text and "retry_resolved" not in caplog.text


def test_d2_echoed_question_in_category_is_never_logged(caplog):
    import logging
    echo = "jazz brushes drummers"  # short (<32 chars) question text echoed into category
    raw = '{"in_scope":false,"category":"%s"}' % echo
    chat, calls = _seq(raw, raw)
    with caplog.at_level(logging.INFO, logger="vera"):
        sr.route(Q, "m", cfg=sr.load_config(), chat=chat)
    assert echo not in caplog.text and "label=nonlabel" in caplog.text
    assert sr._ooe_label('{"in_scope":false,"category":"Novel_Subject"}') == "novel_subject"
    assert sr._ooe_label('{"in_scope":false,"category":"x" }') == "x"
    assert sr._ooe_label('{"in_scope":false,"category":"a b"}') == "nonlabel"
    assert sr._ooe_label('{"in_scope":false,"category":"%s"}' % ("a" * 33)) == "nonlabel"
