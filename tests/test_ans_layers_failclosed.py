"""ANS-LAYERS-1101-FC: fail-closed note kinds, single shortfall line, claim_check removed notes by flag."""
from vera import answer_contract as ac
from vera import claim_check as cc
from vera import ui_evidence as ui


class _St:
    def __init__(self):
        self.out = []

    def caption(self, t):
        self.out.append(t)

    def markdown(self, t):
        self.out.append(t)

    def info(self, t):
        self.out.append(t)


def test_unknown_kind_not_public():
    notes = ({"kind": "weird", "text": "W"}, {"kind": "background", "text": "B"}, {"kind": "partial", "text": "P"})
    assert ac.split_notes(notes) == (["P"], ["B"])
    assert ac.split_notes(({"kind": None, "text": "X"}, {"text": "Y"})) == ([], [])


def _data(limits):
    d = {"contract": {"inline_acquisition": {"acquisition": {"stored": 1, "target": 3, "status": "resource_limited"}}}}
    if limits is not None:
        d.update(answer_status="partial", limitations=limits)
    return d


def test_shortfall_shown_once_when_layered():
    line = ui.acquisition_notice_line(_data(None))
    st = _St()
    ui.render_evidence_scope(st, _data([line]))
    assert line not in st.out
    st2 = _St()
    ui.render_limitations(st2, _data([line]))
    assert [o for o in st2.out if line in o] == ["- " + line]


def test_shortfall_legacy_unchanged():
    line = ui.acquisition_notice_line(_data(None))
    st = _St()
    ui.render_evidence_scope(st, _data(None))
    assert st.out.count(line) == 1
    st = _St()
    ui.render_evidence_scope(st, _data([]))  # layered fields but line not in limitations (flag off): still shown
    assert st.out.count(line) == 1


CFG = {"removed_note": "RN", "partial_marker": " (p)"}
CLAIMS = [{"id": 0, "text": "Bad claim [1].", "cites": [1]}, {"id": 1, "text": "Good claim [1].", "cites": [1]}]
VERDICTS = {0: {"verdict": "unsupported"}, 1: {"verdict": "supported"}}


def _apply(monkeypatch, on):
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: on)
    return cc.apply("Bad claim [1]. Good claim [1].", CLAIMS, VERDICTS, CFG)


def test_claim_check_flag_off_legacy_text(monkeypatch):
    text, counts = _apply(monkeypatch, False)
    assert text == "Good claim [1].\n\nRN" and counts["removed"] == 1


def test_claim_check_flag_on_no_note_in_text(monkeypatch):
    text, counts = _apply(monkeypatch, True)
    assert text == "Good claim [1]." and counts["removed"] == 1
