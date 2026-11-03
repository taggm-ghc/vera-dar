"""NONASK-1103 E5: the evidence trace inside the Ask page's 'Research and verification' expander. Offline AppTest."""
from tests.test_ans_layers_ui_1101 import _layered, _exp, _run_dev, _all
from tests.test_ui_render_extract import _answer
import tests.test_ui_render_extract as base

import pytest


@pytest.fixture(autouse=True)
def _legacy_page(monkeypatch):
    from vera import progress
    monkeypatch.setattr(progress, "stream_enabled", lambda: False)


HEAD = "How VERA-DAR chose the evidence"


def _trace(view="public"):
    return {"version": 1, "view": view, "passages": [{"n": 1, "cited": True, "verdict": "supported", "supports": []}],
            "parts": [], "codes": ["DEVCODE_X"]}


def _body(view="public", **kw):
    b = _layered("complete", **kw)
    b["contract"] = {"trace": _trace(view)}
    return b


def test_trace_in_expander_main_pane_only(monkeypatch):
    at = base._run(monkeypatch, _body())
    ver = _exp(at, "Research and verification")
    assert any(HEAD in m.value for m in ver.markdown)
    assert not any(HEAD in e.value for e in list(at.sidebar.markdown) + list(at.sidebar.caption))
    assert not any(HEAD in v for v in _all(at) if v not in [m.value for m in ver.markdown])


def test_expander_appears_for_trace_alone(monkeypatch):
    b = _layered("complete", process_notes=[], claim_check=None)
    b["contract"] = {"trace": _trace()}
    at = base._run(monkeypatch, b)
    assert any(HEAD in m.value for m in _exp(at, "Research and verification").markdown)


def test_public_mode_hides_dev_view(monkeypatch):
    at = base._run(monkeypatch, _body(view="dev"))
    assert not any(HEAD in v or "DEVCODE_X" in v for v in _all(at))


def test_public_view_has_no_dev_codes(monkeypatch):
    at = base._run(monkeypatch, _body())
    assert not any("DEVCODE_X" in v for v in _all(at))


def test_dev_mode_shows_dev_trace(monkeypatch):
    at = _run_dev(monkeypatch, _body(view="dev"))
    assert any(HEAD in m.value for m in _exp(at, "Research and verification").markdown)


def test_absent_trace_is_noop(monkeypatch):
    at = base._run(monkeypatch, _layered("complete"))
    assert not any(HEAD in v for v in _all(at))
    ver = _exp(at, "Research and verification")
    assert any(c.value.startswith("Citation check:") for c in ver.caption)
