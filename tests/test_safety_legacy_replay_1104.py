"""INTEGRATE-1104 round 2 SAFETY FIX: replay of the 7 safety cases that HEAD declined and the S2 candidate answered when
the classifier returned the LEGACY reply {"in_scope": false, "category": "off_topic"} (SAFETY-REGRESSION-1104 harness,
scratch/SAFETY-REGRESSION-1104/harness/run_cases.py). Cases are loaded by ID from config/scope_set_v1.json and
config/scope_set_v2.json; no prompt text is written here. Offline: fake classifier replies, no DB or network."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import ask_service
from vera import scope_router as sr

ROOT = Path(__file__).resolve().parent.parent
IDS = (("scope_set_v1", "s14"), ("scope_set_v1", "s37"), ("scope_set_v1", "s45"), ("scope_set_v1", "s46"),
       ("scope_set_v1", "s48"), ("scope_set_v2", "v2-41"), ("scope_set_v2", "v2-47"))
LEGACY_OFF = {"in_scope": False, "category": "off_topic", "reason": "fake"}


def _text(setname: str, cid: str) -> str:
    items = json.loads((ROOT / "config" / f"{setname}.json").read_text())["items"]
    (it,) = [i for i in items if i["id"] == cid]
    assert it["label"] == "out_of_scope"  # each is a safety-labelled case in the frozen set
    return it["text"]


def _chat(reply, calls):
    def chat(messages, model, l2, client=None, extra_body=None):
        calls.append(1)
        return json.dumps(reply), 50, 5
    return chat


@pytest.mark.parametrize("setname,cid", IDS, ids=[c for _, c in IDS])
def test_legacy_offtopic_reply_is_never_a_pass_at_the_router(setname, cid):
    calls = []
    d = sr.route(_text(setname, cid), "m", None, chat=_chat(LEGACY_OFF, calls))
    assert d.accepted is False, cid
    assert d.category not in (sr.SAFE_CATEGORY, "off_topic", "in_scope")


@pytest.mark.parametrize("setname,cid", IDS, ids=[c for _, c in IDS])
def test_legacy_offtopic_reply_declines_end_to_end(setname, cid, monkeypatch):
    calls = []
    monkeypatch.setattr(sr, "default_chat", _chat(LEGACY_OFF, calls))
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: pytest.fail("corpus attempt for a safety case"))
    monkeypatch.setattr(ask_service, "_inline", lambda *a, **k: pytest.fail("acquisition for a safety case"))
    monkeypatch.setattr(ask_service, "_reading_list", lambda *a, **k: pytest.fail("reading list for a safety case"))
    r = ask_service.answer_in_scope(_text(setname, cid), "m", SimpleNamespace(input=0, output=0))
    assert r.result_kind == "decline" and r.answer == sr.decline_text(), cid
    assert r.accepted is False and r.acquire_trigger is None and not r.sources


@pytest.mark.parametrize("reply", [
    {"in_scope": False, "category": "off_topic", "reason": "x"},
    {"in_scope": True, "category": "in_scope", "reason": "x"},
    {"in_scope": True, "reason": "x"},
    {"safe_request": True, "category": "off_topic", "reason": "x"},
    {"safe_request": False, "category": "in_scope", "reason": "x"},
], ids=["legacy_off", "legacy_in", "legacy_key_only", "new_key_alias_off", "new_key_alias_in"])
def test_off_schema_reply_is_retried_once_then_declined(reply):
    calls = []
    d = sr.route("What does the evidence say about urban tree cover and heat?", "m", None, chat=_chat(reply, calls))
    assert d.accepted is False and d.reason.startswith("classifier_malformed")
    assert len(calls) == 2  # the existing malformed retry (layer2.malformed_retry = 1), then the neutral decline


def test_off_schema_then_valid_reply_passes_on_the_retry():
    seq = [{"in_scope": False, "category": "off_topic", "reason": "x"},
           {"safe_request": True, "category": "safe_request", "reason": "x"}]

    def chat(messages, model, l2, client=None, extra_body=None):
        return json.dumps(seq.pop(0)), 50, 5
    d = sr.route("What does the evidence say about urban tree cover and heat?", "m", None, chat=chat)
    assert d.accepted is True and d.category == sr.SAFE_CATEGORY and d.llm_calls == 2


# ---- DENYALL-AUDIT-1104 A1: deny all; the ONLY pass is safe_request:true with category exactly safe_request ----
@pytest.mark.parametrize("reply", [
    {"safe_request": True, "reason": "x"},
    {"safe_request": True, "category": "", "reason": "x"},
    {"safe_request": True, "category": "sports", "reason": "x"},
    {"safe_request": True, "category": None, "reason": "x"},
    {"safe_request": True, "category": "SAFE_REQUEST", "reason": "x"},
], ids=["missing", "empty", "out_of_enum", "null", "case_variant"])
def test_true_verdict_without_the_safe_category_is_retried_then_declined(reply):
    calls = []
    d = sr.route("What does the evidence say about urban tree cover and heat?", "m", None, chat=_chat(reply, calls))
    assert d.accepted is False and d.reason.startswith("classifier_malformed"), d
    assert len(calls) == 2


def test_true_verdict_without_the_safe_category_declines_end_to_end(monkeypatch):
    calls = []
    monkeypatch.setattr(sr, "default_chat", _chat({"safe_request": True, "category": "sports", "reason": "x"}, calls))
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: pytest.fail("corpus attempt without an approved pass"))
    monkeypatch.setattr(ask_service, "_inline", lambda *a, **k: pytest.fail("acquisition without an approved pass"))
    r = ask_service.answer_in_scope("What does the evidence say about urban tree cover and heat?", "m",
                                    SimpleNamespace(input=0, output=0))
    assert r.result_kind == "decline" and r.answer == sr.decline_text()


def test_only_the_exact_exception_passes():
    calls = []
    d = sr.route("What does the evidence say about urban tree cover and heat?", "m", None,
                 chat=_chat({"safe_request": True, "category": "safe_request", "reason": "x"}, calls))
    assert d.accepted is True and d.category == sr.SAFE_CATEGORY and len(calls) == 1


@pytest.mark.parametrize("cat", ["advice", "harmful", "rule_change", "programming_task"])
def test_true_verdict_with_a_refusal_category_is_still_a_refusal(cat):
    d = sr.route("What does the evidence say about urban tree cover and heat?", "m", None,
                 chat=_chat({"safe_request": True, "category": cat, "reason": "x"}, []))
    assert d.accepted is False
