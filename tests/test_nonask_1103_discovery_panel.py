"""NONASK-1103 WP-A: reading-list identifier caption uses plain_url; Handle identifiers are shown as persistent."""
from unittest import mock

import pytest

from vera import discovery_panel as dp

HANDLE = "https://hdl.handle.net/11580/127623"
LANDING = "https://example.org/landing"
PID_TEXT = dp.REASON_TEXT["R-ACQ-PID-NOROUTE"]
NODOI_TEXT = dp.REASON_TEXT["R-ACQ-NODOI"]


def _data(identifier, reason="R-ACQ-NODOI"):
    return {"result_kind": "reading_list", "answer": "x",
            "candidates": [{"title": "Handle Paper", "url": LANDING, "identifier": identifier,
                            "published": "2025", "acquisition": "not_acquired"}],
            "contract": {"inline_acquisition": {"discovered": [{"url": LANDING, "reason": reason}]}}}


def _render(data):
    st = mock.MagicMock()
    dp.render_discovery(st, data)
    return [str(c.args[0]) for name in ("markdown", "caption", "info", "warning") for c in getattr(st, name).call_args_list]


def test_handle_identifier_caption_has_no_backslash():
    out = _render(_data(HANDLE))
    captions = [t for t in out if HANDLE in t]
    assert captions, out
    assert not any("\\" in t for t in out if HANDLE in t)


def test_handle_is_named_persistent_not_missing():
    out = _render(_data(HANDLE))
    assert any(PID_TEXT in t for t in out)
    assert not any("No stable identifier" in t for t in out)


def test_non_handle_nodoi_uses_new_wording():
    out = _render(_data("10.1000/xyz"))
    assert any(NODOI_TEXT in t for t in out)
    assert not any("No stable identifier" in t for t in out)
    assert not any(PID_TEXT in t for t in out)


@pytest.mark.parametrize("value,expected", [
    (HANDLE, True), ("hdl.handle.net/11580/1", True), ("10.1000/xyz", False),
    ("https://evil.example/hdl.handle.net/x", False), (None, False), (12, False)])
def test_is_handle(value, expected):
    assert dp._is_handle(value) is expected
