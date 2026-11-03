"""ANS-LAYERS-1101 Stage 1 (backend): process/limitation notes stay out of `answer`; answer_status; rollback flag.
Offline: no DB, no network, no LLM."""
import pytest

from vera import answer_contract as ac
from vera import ask_service, fulldoc_answer
from vera.ask_service import AskResult


@pytest.fixture
def sep_on(monkeypatch):
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: True)


@pytest.fixture
def sep_off(monkeypatch):
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: False)


def _res(answer="Body [1]."):
    return AskResult(answer=answer, tokens_used=1, cost_usd=0.0, result_kind="answer")


def test_shipped_flag_default_on():
    assert ac.notes_separate() is True


def test_notes_separate_unreadable_config_is_off(tmp_path):
    assert ac.notes_separate(tmp_path / "missing.json") is False


def test_add_note_on_keeps_answer_and_records(sep_on):
    a, n = ac.add_note("Body", (), "freshness", "Freshness: x", "suffix", "\n\n")
    assert a == "Body" and n == ({"kind": "freshness", "text": "Freshness: x"},)


def test_add_note_off_concatenates_like_before(sep_off):
    assert ac.add_note("Body", (), "x", "N", "suffix", "\n\n") == ("Body\n\nN", ())
    assert ac.add_note("Body", (), "x", "N", "prefix") == ("N Body", ())
    assert ac.add_note("", (), "x", "N", "suffix", "\n\n") == ("N", ())
    assert ac.add_note("", (), "x", "N", "prefix") == ("N ", ())  # legacy: unconditional separator
    assert ac.add_note("Body", (), "x", "", "prefix") == ("Body", ())


def test_with_research_suffix_on_and_off(monkeypatch):
    monkeypatch.setattr(ask_service, "_record_discovery", lambda *a, **k: None)
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: True)
    r = ask_service._with_research(_res(), {"k": 1}, "Freshness: newly found.")
    assert r.answer == "Body [1]." and r.notes[0]["text"] == "Freshness: newly found."
    assert r.contract["research"] == {"k": 1}
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: False)
    r = ask_service._with_research(_res(), {"k": 1}, "Freshness: newly found.")
    assert r.answer == "Body [1].\n\nFreshness: newly found." and r.notes == ()
    assert ask_service._with_research(_res(), {}, None).answer == "Body [1]."


def test_split_notes_buckets_and_dedupes():
    notes = ({"kind": "partial", "text": "P"}, {"kind": "freshness", "text": "F"}, {"kind": "removed", "text": "R"},
             {"kind": "partial", "text": "P"}, {"kind": "weird", "text": "W"}, {"kind": "removed", "text": ""})
    p, f = ac.split_notes(notes)  # unknown kind fails closed: in neither public list
    assert (p, f) == (["P", "R"], ["F"]) and "W" not in p + f
    assert ac.split_notes(()) == ([], [])


@pytest.mark.parametrize("kind,outcome,comp,off,want", [
    ("answer", "answer", None, False, "unassessed"), ("answer", "answer", "complete", False, "complete"),
    ("answer", "answer", "partial", False, "partial"), ("answer", "answer", "bogus", False, "partial"),
    ("answer", "answer", "complete", True, "partial"), ("answer", "answer", None, True, "partial"),
    ("decline", "decline", None, False, "declined"), ("reading_list", "reading_list", None, False, "not_answered"),
    ("discovery", "insufficient_evidence", None, False, "not_answered"),
    ("unspecified", "insufficient_evidence", None, False, "not_answered"),
    ("partial_answer", "insufficient_evidence", None, False, "partial"),
])
def test_answer_status_table(kind, outcome, comp, off, want):
    assert ac.answer_status(kind, outcome, comp, off) == want


def test_coverage_note_is_a_limitation():
    assert ac.split_notes(({"kind": "coverage", "text": "C"},)) == (["C"], [])


def test_inline_in_scope_flag_off_exact_legacy_order(sep_off):
    """D3: legacy text is 'ACQ BG answer' (acquisition note, then background note, then the answer)."""
    ans, nts = ac.add_note("ANS", (), "background", "BG", "prefix")
    ans, nts = ac.add_note(ans, nts, "acquisition", "ACQ", "prefix")
    assert ans == "ACQ BG ANS" and nts == ()


_REC = [{"id": 0, "sentence": "A [1].", "citations": [1], "verdict": "supported", "part": 1}]
_CC = {"partial_marker": " (p)", "removed_note": "REMOVED-NOTE"}
_CFG = {"answer": {"unchecked_note": "UNCHECKED-NOTE", "quote_max_chars": 200}}


def _checked(notes_out):
    return fulldoc_answer.checked_only("A [1]. B.", _REC, [{"n": 1}], _CC, _CFG, {"removed": True}, notes_out=notes_out)


def test_checked_only_notes_out_on(sep_on):
    out = []
    text, _ = _checked(out)
    assert "REMOVED-NOTE" not in text and "UNCHECKED-NOTE" not in text
    assert ("removed", "REMOVED-NOTE") in out and ("unchecked", "UNCHECKED-NOTE") in out


def test_checked_only_notes_out_off_is_old_behaviour(sep_off):
    out = []
    text, _ = _checked(out)
    assert text.endswith("REMOVED-NOTE\n\nUNCHECKED-NOTE") and out == []
    assert _checked(None)[0] == text


def test_checked_only_without_collector_unchanged(sep_on):
    assert "REMOVED-NOTE" in _checked(None)[0]  # legacy callers (no collector) keep the notes in the text


def test_ask_response_fields_additive():
    main = pytest.importorskip("main")
    f = main.AskResponse.model_fields
    for old in ("answer", "model", "tokens_used", "cost_usd", "sources", "claim_check", "result_kind", "outcome",
                "candidates", "contract", "citations", "evidence_status"):
        assert old in f
    assert f["answer"].annotation is str and f["cost_usd"].annotation is float
    r = main.AskResponse(answer="a", model="m", tokens_used=1, cost_usd=0.0).model_dump()
    assert r["answer_status"] == "not_answered" and r["limitations"] == [] and r["process_notes"] == []
    r = main.AskResponse(answer="a", model="m", tokens_used=1, cost_usd=0.0, answer_status="partial",
                         limitations=["L"], process_notes=["P"]).model_dump()
    assert (r["answer_status"], r["limitations"], r["process_notes"]) == ("partial", ["L"], ["P"])
