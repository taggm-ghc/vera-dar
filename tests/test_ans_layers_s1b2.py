"""ANS-LAYERS-1101 S1B2: real ask_service._inline site (D3), background note kind (A), non-answer notices stay in
`answer` (B), consistent note order (C), strict flag parsing. Offline: stubs only."""
import json
from types import SimpleNamespace

import pytest

from vera import acq_config, answer_contract as ac, ask_service, fulldoc_corpus, inline_acquire
from vera.ask_service import AskResult


class _O:
    def __init__(self, corpus):
        self.corpus, self.status = corpus, "ok"

    def public(self):
        return {"status": "ok"}


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(inline_acquire, "background_note", lambda acq, o: "BG")
    monkeypatch.setattr(inline_acquire, "answer_note", lambda acq, o: "ACQ")
    monkeypatch.setattr(acq_config, "override_answer_prefix", lambda acq: "W")
    monkeypatch.setattr(ask_service, "_inline_notice", lambda acq, o: "NOTICE")
    from vera import acq_discovered
    monkeypatch.setattr(acq_discovered, "use_discovered", lambda acq: False)
    from vera import provider_exhaustion
    monkeypatch.setattr(provider_exhaustion, "note_work", lambda **k: None)
    state = {"corpus": object()}
    monkeypatch.setattr(inline_acquire, "acquire", lambda q, acq, served=None, **kw: _O(state["corpus"]))
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: AskResult(answer="ANS", tokens_used=1, cost_usd=0.0, result_kind="answer"))
    return state


def _flag(monkeypatch, on):
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: on)


def _call(in_scope):
    return ask_service._inline("q", "m", None, SimpleNamespace(topic_key=""), None, None, object(), in_scope=in_scope)


def test_d3_in_scope_real_site_flag_off_legacy_text(rig, monkeypatch):
    _flag(monkeypatch, False)
    r = _call(True)
    assert r.answer == "ACQ BG ANS" and r.notes == ()


def test_in_scope_flag_on_answer_clean_and_order_matches_off_topic(rig, monkeypatch):
    _flag(monkeypatch, True)
    r = _call(True)
    assert r.answer == "ANS" and [n["kind"] for n in r.notes] == ["acquisition", "background"]
    o = _call(False)  # INTEGRATE-1104 / SCOPE-GATE-1104 S2: no off-topic wording remains; in_scope is ignored
    assert o.answer == "ANS" and [n["kind"] for n in o.notes] == ["acquisition", "background"]
    assert "offtopic" not in (o.contract or {}) and o.accepted == r.accepted  # same as the in-scope call


def test_off_topic_flag_off_legacy_text(rig, monkeypatch):
    _flag(monkeypatch, False)
    assert _call(False).answer == "ACQ BG ANS"  # INTEGRATE-1104 S2: no off-topic warning prefix ("W") any more


def test_background_note_is_separate_kind_on_non_answer(rig, monkeypatch):
    rig["corpus"] = None
    _flag(monkeypatch, True)
    n = _call(False)
    assert str(n) == "NOTICE" and n.bg_note == "BG"
    a, notes = ask_service._nonanswer_note("LIST", (), n)
    assert a == "LIST NOTICE" and notes == ({"kind": "background", "text": "BG"},)
    assert ac.split_notes(notes) == ([], ["BG"])  # a process note, not a limitation
    _flag(monkeypatch, False)
    n = _call(False)
    assert n == "NOTICE BG" and not hasattr(n, "bg_note")
    assert ask_service._nonanswer_note("LIST", (), n) == ("LIST NOTICE BG", ())


@pytest.mark.parametrize("on", [True, False])
def test_nonanswer_notice_stays_in_answer_with_legacy_whitespace(on, monkeypatch):
    _flag(monkeypatch, on)
    assert ask_service._nonanswer_note("LIST", (), "N") == ("LIST N", ())
    assert ask_service._nonanswer_note("", (), "N") == ("N", ())
    assert ask_service._nonanswer_note("LIST", (), None) == ("LIST", ())
    assert ask_service._nonanswer_note("T", (), "N", strip=False) == ("T N", ())
    assert ask_service._nonanswer_note("", (), "N", strip=False) == (" N", ())  # legacy f"{text} {note}"


@pytest.mark.parametrize("val,want", [(True, True), (False, False), ("false", False), ("true", False), (1, False), (None, False)])
def test_flag_parses_only_literal_json_true(tmp_path, val, want):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"answer_notes_separate": val}))
    assert ac.notes_separate(p) is want


def test_flag_missing_key_bad_json_and_non_object_are_off(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{}")
    assert ac.notes_separate(p) is False
    p.write_text("not json")
    assert ac.notes_separate(p) is False
    p.write_text("[true]")
    assert ac.notes_separate(p) is False
