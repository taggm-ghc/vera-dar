"""STRICTJSON-1027 SJ2: claim_check._parse strict replies and row-id hardening (T1-T7). Fakes only, no network."""
import json
import logging

import pytest

from vera import claim_check as cc

CC = {"enabled": True, "timeout_s": 1, "max_claims": 30, "partial_marker": " (partly)", "removed_note": "REMOVED",
      "unavailable_text": "UNAVAILABLE",
      "checkers": [{"provider": "a", "base_url": "x", "model": "first", "api_key_env": "KEY_A", "family": "qwen"},
                   {"provider": "b", "base_url": "x", "model": "second", "api_key_env": "KEY_B", "family": "openai"}]}
SRC = [{"n": 1, "title": "A", "url": "https://arxiv.org/abs/2401.00001", "snippet": "Copilot users were 55% faster."},
       {"n": 2, "title": "B", "url": "https://doi.org/10.1/b", "snippet": "No effect on code quality."}]
ANSWER = "Faster [1]. Quality improved [2]. Shorter [1]."
CLAIMS = cc.split_claims(ANSWER, 2, 30)  # ids 0, 1, 2
SECRET = "INJECTEDREPLYTEXT"


def _row(i, v, extra=""):
    return '{"id":%s,"verdict":"%s","reason":"%s"%s}' % (i, v, SECRET, extra)


def _reply(*rows):
    return '{"claims":[' + ",".join(rows) + "]}"


CLEAN = _reply(_row(0, "supported"), _row(1, "unsupported"), _row(2, "partial"))


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.setenv("KEY_B", "k")


def test_t1_duplicate_verdict_in_row_rejected():
    txt = _reply(_row(0, "supported"), '{"id":1,"verdict":"unsupported","verdict":"supported"}', _row(2, "partial"))
    assert cc._parse(txt, CLAIMS) is None


def test_t2_duplicate_top_level_claims_rejected():
    txt = '{"claims":[%s],"claims":[%s,%s,%s]}' % (_row(1, "unsupported"), _row(0, "supported"),
                                                    _row(1, "supported"), _row(2, "supported"))
    assert cc._parse(txt, CLAIMS) is None


@pytest.mark.parametrize("txt", ['{"claims":[{"id":0,"verdict":"supported","score":NaN}]}',
                                 '{"claims":[{"id":0,"verdict":"supported","Verdict":"unsupported"}]}',
                                 '{"claims":[{"id":0,"vеrdict":"unsupported","verdict":"supported"}]}'])
def test_t1b_other_strict_rejections(txt):
    assert cc._parse(txt, CLAIMS[:1]) is None


def test_t3_judge_moves_to_second_checker_after_duplicate():
    dup = _reply(_row(0, "supported"), '{"id":1,"verdict":"unsupported","verdict":"supported"}', _row(2, "partial"))
    replies = {"first": dup, "second": CLEAN}
    verdicts, label = cc.judge(CLAIMS, SRC, CC, lambda ch, m: replies[ch["model"]])
    assert label == "b:second"
    assert {k: v["verdict"] for k, v in verdicts.items()} == {0: "supported", 1: "unsupported", 2: "partial"}


def test_t4_all_checkers_duplicate_fail_closed():
    dup = '{"claims":[{"id":0,"verdict":"unsupported","verdict":"supported"}]}'
    assert cc.check_answer("Faster [1].", SRC, CC, call=lambda ch, m: dup) == (
        None, {"checked": 0, "outcome": "checker_unavailable"})


@pytest.mark.parametrize("rows", [
    (_row(0, "supported"), _row(1, "supported"), _row(1, "unsupported"), _row(2, "partial")),
    (_row(0, "supported"), _row(1, "unsupported"), _row(1, "supported"), _row(2, "partial")),
    (_row(0, "supported"), _row(1, "Unsupported"), _row(1, "supported"), _row(2, "partial")),  # invalid restrictive
    (_row(0, "supported"), _row(1, "supported"), _row(1, "Unsupported"), _row(2, "partial")),
    (_row(0, "supported"), '{"id":1}', _row(1, "supported"), _row(2, "partial")),  # verdict-less row still counts
])
def test_t5_repeated_id_rejects_whole_reply(rows):
    assert cc._parse(_reply(*rows), CLAIMS) is None


@pytest.mark.parametrize("rows", [
    (_row(0, "supported"), _row("true", "supported"), _row(2, "partial")),         # true replacing id 1
    (_row(0, "supported"), _row(1, "supported"), _row(99, "partial")),             # foreign 99 replacing id 2
    (_row(0, "supported"), _row('"1"', "supported"), _row(2, "partial")),          # str id
    (_row(0, "supported"), _row("1.0", "supported"), _row(2, "partial")),          # float id
    (_row(0, "supported"), _row(1, "supported"), _row(2, "partial"), _row(7, "supported")),  # extra foreign row
    (_row(0, "supported"), _row(1, "supported"), _row(2, "partial"), _row("null", "supported")),
])
def test_t6_foreign_or_non_int_id_rejected_and_no_keyerror(rows):
    txt = _reply(*rows)
    assert cc._parse(txt, CLAIMS) is None
    assert cc.check_answer(ANSWER, SRC, CC, call=lambda ch, m: txt) == (
        None, {"checked": 0, "outcome": "checker_unavailable"})


def test_t6_missing_claim_still_rejected():
    assert cc._parse(_reply(_row(0, "supported"), _row(1, "supported")), CLAIMS) is None


@pytest.mark.parametrize("txt,code", [
    ('{"claims":[{"id":0,"verdict":"unsupported","verdict":"supported","reason":"%s"}]}' % SECRET, "duplicate_key"),
    (_reply(_row(0, "supported"), _row(0, "unsupported")), "row_ids"),
])
def test_t6b_fixed_log_line_no_reply_text(caplog, txt, code):
    with caplog.at_level(logging.WARNING, logger="vera"):
        assert cc._parse(txt, CLAIMS[:1]) is None
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == [f"/ask claim check reply rejected ({code})"]
    assert not any(SECRET in m or "supported" in m for m in msgs)


def test_t7_clean_reply_unchanged():
    text, summary = cc.check_answer(ANSWER, SRC, CC, call=lambda ch, m: CLEAN)
    assert text == "Faster [1]. Shorter [1]. (partly)\n\nREMOVED"
    assert summary == {"checked": 3, "supported": 1, "partial": 1, "removed": 1, "checker": "a:first",
                       "outcome": "checked"}
    assert cc._parse(json.dumps({"claims": [{"id": 0, "verdict": "supported"}]}), CLAIMS[:1]) == {
        0: {"verdict": "supported", "reason": ""}}


def test_t7_rows_without_id_are_ignored_as_before():
    txt = _reply(_row(0, "supported"), '{"verdict":"unsupported"}')
    assert cc._parse(txt, CLAIMS[:1]) == {0: {"verdict": "supported", "reason": SECRET}}
