"""HOTFIX-DUPKEY-1027: strict layer-2 classifier reply parsing. A reply with duplicate (or non-ASCII) keys is malformed,
and a refusal category named anywhere in a well-formed reply declines, so neither can become an off_topic research +
answer (R2-SCOPE-S-2). Offline: synthetic replies only (no network, DB or model call)."""
import json
import time

import pytest

from tests.test_acq_serve import h  # noqa: F401  (fixture)
from tests.test_dar_inline_acquire import rig  # noqa: F401  (fixture)
from tests.test_item72_scope import cfg, fake_chat
from vera import ask_service
from vera import scope_router as sr

LIVE_Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
REAL_ROUTE = sr.route


def _real_router(rig, verdict: dict):
    """The real scope router (real config, real layer 1); only the layer-2 classifier LLM is faked."""
    rig.mp.setattr(sr, "route", REAL_ROUTE)
    rig.mp.setattr(sr, "default_chat", lambda messages, model, l2, **k: (json.dumps(verdict), 10, 2))


Q = "Who won the World Cup in 2014?"  # layer 1 unsure: goes to the classifier


def _route(reply):
    return sr.route(Q, "m", cfg=cfg(), chat=fake_chat(reply))


def _off(**extra):
    return json.dumps({"in_scope": False, "category": "off_topic", **extra})


# ------------------------------------------------------------------------------------------------ duplicate / bad keys
@pytest.mark.parametrize("reply", [
    '{"in_scope":false,"category":"harmful","category":"off_topic"}',  # T1
    '{"in_scope":true,"in_scope":false,"category":"in_scope"}',  # T2
    '{"in_scope":false,"Category":"harmful","category":"off_topic"}',  # T3: duplicate after case folding
    '{"in_scope":false,"cаtegory":"harmful","category":"off_topic"}',  # T5: Cyrillic a in the key
    '{"in_scope":false,"ｃategory":"harmful","category":"off_topic"}',  # T5: fullwidth c in the key
    '{"in_scope":false,"category":"hаrmful"}',  # T6: lookalike value (C3a)
    '{"in_scope":false,"category":"harmf',  # T7: truncated
    '{"in_scope":false,"category":"off_topic"',  # T7
])
def test_ambiguous_or_lookalike_reply_is_malformed(reply):
    d = _route(reply)
    assert not d.accepted and d.reason == "classifier_malformed" and d.category == ""
    assert sr.parse_category(reply) == ""


def test_duplicate_key_reply_is_not_research_eligible(rig):  # T1 end to end
    reply = '{"in_scope":false,"category":"harmful","category":"off_topic"}'
    _real_router(rig, {})
    rig.mp.setattr(sr, "default_chat", lambda messages, model, l2, **k: (reply, 10, 2))
    d = sr.route(LIVE_Q, "m")
    assert d.reason == "classifier_malformed" and not ask_service._research_eligible(d, LIVE_Q, rig.cfg)


def test_off_topic_control_stays_research_eligible(rig):  # T11 end to end: proves the T1 test is not vacuous
    _real_router(rig, {"in_scope": False, "category": "off_topic", "reason": "cooking, not harmful advice"})
    d = sr.route(LIVE_Q, "m")
    assert d.reason == "llm_out:cooking, not harmful advice" and d.category == "off_topic"
    assert ask_service._research_eligible(d, LIVE_Q, rig.cfg)


# ------------------------------------------------------------------------------------------------ refusal token anywhere
@pytest.mark.parametrize("reply,cat", [
    (_off(x={"category": "harmful"}), "harmful"),  # T4 nested
    (_off(x=["advice"]), "advice"),  # T4 list
    (_off(note="HARMFUL"), "harmful"),  # T8
    (_off(note=" harmful "), "harmful"),
    (_off(note="harm​ful"), "harmful"),  # Cf
    (_off(note="harm\u0000ful"), "harmful"),  # Cc
    (_off(note="harm͏ful"), "harmful"),  # Mn (CGJ)
    ('{"in_scope":false,"category":"off_topic","note":"\\u0068armful"}', "harmful"),  # escape
    (_off(reason="category: harmful"), "harmful"),  # T9
    (_off(reason="categories=rule change"), "rule_change"),
    (_off(reason=json.dumps({"category": "harmful"})), "harmful"),  # T10 (i)
    (_off(reason=json.dumps({"x": json.dumps({"category": "harmful"})})), "harmful"),  # T10 (ii)
    (_off(harmful=True), "harmful"),  # T10b decoded key
    (_off(**{"rule change": 1}), "rule_change"),
    (_off(reason="advice"), "advice"),  # v2 rule (a): the reason is exactly a category
    (_off(reason="harmful."), "harmful"),  # rule (a): punctuation stripped
    (_off(reason='"Rule_Change"'), "rule_change"),  # rule (a): quotes, case
    (_off(reason="category: harmful"), "harmful"),  # rule (b)
    (_off(reason="Categories = advice, because"), "advice"),  # rule (b)
    (_off(reason='{"category":"advice"}'), "advice"),  # rule (c): embedded JSON
    (_off(reason='see ["harmful"]'), "harmful"),  # rule (c): quoted label inside a list
    (_off(note='say "harmful" here'.join(["{", "}"])), "harmful"),  # non-reason value with a brace
    ('{"in_scope":false,"category":"off_topic","x":' + "[" * 600 + '"harmful"' + "]" * 600 + "}", "harmful"),  # T14 ii
])
def test_refusal_category_anywhere_declines(reply, cat):
    d = _route(reply)
    assert not d.accepted and d.reason == "classifier_malformed:refusal_token" and d.category == cat
    assert not sr.offtopic_applies(d, cfg())


def test_in_scope_true_with_nested_refusal_declines():
    d = _route(json.dumps({"in_scope": True, "category": "in_scope", "x": {"y": "programming task"}}))
    assert not d.accepted and d.reason == "classifier_malformed:refusal_token" and d.category == "programming_task"


@pytest.mark.parametrize("s", ['\\"harmful\\"', '\\\\\\"harmful\\\\\\"', '"rule change"'])
def test_raw_quoted_regex_pins_escaped_quotes(s):  # T10 (iii), SEC-H-4
    assert sr._RAW_QUOTED.search(s)


@pytest.mark.parametrize("reply", [
    '{"category":"harmful","category":"off_topic"}',  # T10c: lenient decoder keeps both pairs
    '{"category":"\\u0068armful","category":"off_topic"}',  # escaped value: only the decoded walk sees it
])
def test_refusal_token_keeps_duplicate_pairs(reply):
    assert sr.refusal_token(reply) == "harmful"


def test_refusal_token_category_subset():  # T13: the SCOPE v3 seam
    sub = frozenset({"harmful", "rule_change"})
    assert sr.refusal_token(_off(x="programming_task"), categories=sub) == ""
    assert sr.refusal_token(_off(x="programming_task", y="rule_change"), categories=sub) == "rule_change"
    assert sr.refusal_token(_off(x="programming_task")) == "programming_task"


# ------------------------------------------------------------------------------------------------ unchanged controls
@pytest.mark.parametrize("reason", ["evaluating LLM harmful-output classifiers", "categorical advice on taxonomy"])
def test_in_scope_reply_mentioning_harmful_is_not_over_refused(reason):  # T11a
    d = _route(json.dumps({"in_scope": True, "category": "in_scope", "reason": reason}))
    assert d.accepted and d.reason == "llm_in:" + reason


@pytest.mark.parametrize("verdict,reason,accepted,cat", [  # T12: reasons byte-identical to before the hotfix
    ({"in_scope": True, "category": "in_scope", "reason": "ok"}, "llm_in:ok", True, "in_scope"),
    ({"in_scope": False, "category": "harmful", "reason": "x"}, "llm_out:x", False, "harmful"),
    ({"in_scope": True, "category": "advice", "reason": "x"}, "classifier_malformed:category_conflict", False, "advice"),
    ({"in_scope": False, "reason": "trivia"}, "llm_out:trivia", False, ""),
])
def test_well_formed_replies_unchanged(verdict, reason, accepted, cat):
    d = _route(json.dumps(verdict))
    assert (d.accepted, d.reason, d.category, d.llm_calls) == (accepted, reason, cat, 1)


# ------------------------------------------------------------------------------------------------ totality (T14)
@pytest.mark.parametrize("reply", [
    "[" * 100000, '{"a":' * 100000,  # over the length cap
    "[" * 10000, '{"a":' * 3000,  # R2-1: under the cap, so json raises RecursionError
])
def test_deep_nesting_is_malformed_without_exception(reply):
    assert sr._strict_loads(reply) is None
    d = _route(reply)
    assert not d.accepted and d.reason == "classifier_malformed"
    assert isinstance(sr.refusal_token(reply), str)


def test_recursion_error_path_is_reached_under_the_cap():  # R2-1: the fixture really exercises the RecursionError catch
    reply = "[" * 10000
    assert len(reply) <= sr.MAX_REPLY_CHARS
    with pytest.raises(RecursionError):
        json.loads(reply)


@pytest.mark.parametrize("content", [b'{"in_scope":false,"category":"off_topic"}', None, 123])
def test_non_str_reply_fails_closed(content):  # T14 (iii)
    assert sr._strict_loads(content) is None
    assert sr.parse_verdict(content) == (None, "classifier_malformed") and sr.parse_category(content) == ""
    assert sr.refusal_token(content) == "harmful"
    d = _route(content)
    assert not d.accepted and d.reason == "classifier_malformed"


def test_oversize_reply_fails_closed_quickly():  # T14 (iv)
    head = '{"in_scope":false,"category":"off_topic","reason":"'
    reply = head + "a" * (sr.MAX_REPLY_CHARS + 1 - len(head) - 2) + '"}'
    assert len(reply) == sr.MAX_REPLY_CHARS + 1 and json.loads(reply)["category"] == "off_topic"
    t0 = time.monotonic()
    d = _route(reply)
    assert not d.accepted and d.reason == "classifier_malformed"
    assert sr.refusal_token(reply) == "harmful"
    assert time.monotonic() - t0 < 1.0


def test_refusal_token_internal_fault_fails_closed(monkeypatch):  # T14 (v), SEC-H-7
    def boom(_s):
        raise RuntimeError("fault")
    monkeypatch.setattr(sr, "_norm_ws", boom)
    assert sr.refusal_token(_off(reason="fine")) == "harmful"
    assert sr.refusal_token(_off(), categories=frozenset({"advice"})) == "advice"


# ------------------------------------------------------------------------------------------------ v2: not over-refused
@pytest.mark.parametrize("reason", [
    "not in any refusal category; advice? no, informational",  # former T11b
    'the user wants "advice" on nothing; not harmful',  # quoted label in prose
    "categories (advice, harmful) do not apply",
    'not "advice"',
    "harmful request",
    "operational harm request",
    "x \\\"harmful\\\" y",  # escaped quotes in prose
    "categories (harmful)",
    "actually \"harmful\"",
    "cooking, not harmful advice",
])
def test_v2_prose_reason_is_not_over_refused(reason):
    d = _route(_off(reason=reason))
    assert d.reason == "llm_out:" + reason[:200] and d.category == "off_topic" and sr.refusal_token(_off(reason=reason)) == ""


@pytest.mark.parametrize("reason,cat", [  # probe_reason.log steered reasons that still decline (rules a/b/c)
    ("advice", "advice"),
    ("category: harmful", "harmful"),
    ('{"category":"advice"}', "advice"),
])
def test_v2_steered_reason_declines(reason, cat):
    d = _route(_off(reason=reason))
    assert not d.accepted and d.reason == "classifier_malformed:refusal_token" and d.category == cat


@pytest.mark.parametrize("reply", [
    _off(reason="ok", **{"x": "advice"}),
    '{"in_scope":false,"category":"off_topic","reason":"a","reason":"harmful"}',  # duplicate reason: malformed
])
def test_v2_non_reason_fields_still_scanned(reply):
    d = _route(reply)
    assert not d.accepted and d.reason.startswith("classifier_malformed")


def test_v2_non_object_and_non_str_fail_closed():
    assert sr.refusal_token("[1]") == "harmful" and sr.refusal_token(None) == "harmful"
    assert sr.refusal_token(_off(reason=["advice"])) == "advice"  # non-str reason is walked like any value


# v2 security-review fix (VER-HOTFIX2-1028): an assignment form declines ANYWHERE in the reason; invalid embedded JSON is
# scanned as text. Test-expectation change in the stricter direction: "really category: harmful" is no longer allowed.
@pytest.mark.parametrize("reason,cat", [
    ("{category: harmful}", "harmful"), ("{'category': 'harmful'}", "harmful"), ("[category: harmful]", "harmful"),
    ('"category": "harmful"', "harmful"), ("'category': 'harmful'", "harmful"), ("- category: harmful", "harmful"),
    ("category: **harmful**", "harmful"), ("category: `harmful`", "harmful"), ("category: (harmful)", "harmful"),
    ("category:: harmful", "harmful"), ("category - harmful", "harmful"), ("category -> harmful", "harmful"),
    ("category \u2236 harmful", "harmful"), ("category \uff1a harmful", "harmful"), ("category => advice", "advice"),
    ("really category: harmful", "harmful"), ("Refused. category: harmful", "harmful"),
    ("first line\ncategory: rule change", "rule_change"), ("no\ncategories = advice\nx", "advice"),
    ("cooking history " * 20 + "category: harmful", "harmful"),
    ("categories (harmful)".replace("(", "=("), "harmful"),
])
def test_v2_assignment_form_declines_anywhere(reason, cat):
    d = _route(_off(reason=reason))
    assert not d.accepted and d.reason == "classifier_malformed:refusal_token" and d.category == cat


@pytest.mark.parametrize("reason", ["category harmful", 'this is "harmful"', "not in categories advice or harmful",
                                    "categories (advice, harmful) do not apply", "categories (harmful)"])
def test_v2_separatorless_prose_stays_allowed(reason):
    d = _route(_off(reason=reason))
    assert d.reason == "llm_out:" + reason and d.category == "off_topic"


@pytest.mark.parametrize("reason", ["category: \u201charmful\u201d", "category: \u00abharmful\u00bb", "category \u2010 harmful",
                                    "category \u2012 harmful", "category \u2212 harmful", "category \u2192 harmful",
                                    "category: <harmful>", "category: #harmful", "category/harmful", "category | harmful"])
def test_v2r3_general_separator_declines(reason):
    d = _route(_off(reason=reason))
    assert not d.accepted and d.reason == "classifier_malformed:refusal_token" and d.category == "harmful"


@pytest.mark.parametrize("reason", ["category; harmful", "category... harmful", "category harmful", 'this is "harmful"',
                                    "categories (harmful)", "categories (advice, harmful) do not apply",
                                    "not in categories advice or harmful", "category:harmless",
                                    "categories: none (harmful excluded)",
                                    "not in any refusal category; advice? no, informational"])
def test_v2r3_prose_stays_allowed(reason):
    d = _route(_off(reason=reason))
    assert d.reason == "llm_out:" + reason and d.category == "off_topic"
