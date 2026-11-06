"""LANE-LINKS-1106 WP5 test 7: lane links render on answer and non-answer pages, escaped, need-to-know.
Stream path UNTESTED: the offline harness cannot drive the NDJSON worker; by code reading the stream terminal calls
the same render_result with the same body dict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ui_render_extract as t  # noqa: E402

import pytest  # noqa: E402
from vera import progress  # noqa: E402


@pytest.fixture(autouse=True)
def _legacy_page(monkeypatch):  # drive the legacy /ask page; the stream path renders via the same render_result
    from vera import progress
    monkeypatch.setattr(progress, "stream_enabled", lambda: False)


LANES = {"links": [{"lane": "reporting", "label": "News", "title": "Vote *count* <b>x</b> [a](javascript:alert(1))",
                    "url": "https://news.example.org/a", "outlet": "example.org", "published": "2026-11-04",
                    "read": False},
                   {"title": "Bad", "url": "javascript:alert(1)"},
                   {"title": "Read one", "url": "https://wikinews.example/b", "outlet": "Wikinews", "read": True}],
         "n_read": 1, "credits": [{"text": "News index: GDELT Project", "url": "https://www.gdeltproject.org/"}]}


def _texts(at):
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


def _body(kind):
    b = t._answer() if kind == "answer" else t._answer(result_kind="reading_list", answer="")
    b["contract"] = {"inline_acquisition": {"status": "no_new", "lanes": LANES}}
    return b


def _check(monkeypatch, kind):
    at = t._run(monkeypatch, _body(kind))
    txt = "\n".join(_texts(at))
    assert "Current reporting (links only)" in txt
    assert "https://news.example.org/a" in txt and "example.org" in txt and "2026-11-04" in txt
    assert "link only" in txt and " - read" in txt
    assert "GDELT" in txt and "gdeltproject.org" in txt
    assert "](javascript" not in txt and "\\]\\(javascript" in txt and "](https://news.example.org/a)" in txt
    assert "](javascript:alert(1))" not in txt
    assert "<b>" not in txt and "https://wikinews.example/b" in txt
    assert "reporting_lane" not in txt.lower() and "portal" not in txt.lower()


def test_answer_page(monkeypatch):
    _check(monkeypatch, "answer")


def test_non_answer_page(monkeypatch):
    _check(monkeypatch, "other")


def test_absent_lanes_renders_nothing(monkeypatch):
    at = t._run(monkeypatch, t._answer())
    assert "Current reporting" not in "\n".join(_texts(at))


def test_dollar_escaped_and_truthy_read_not_enough(monkeypatch):
    from vera import lane_links_ui as L
    assert L._t("cost $5 $x$") == "cost \\$5 \\$x\\$"
    assert L._http("https://a.example/" + "x" * 2000) == ""
    calls = []

    class S:
        def markdown(self, x): calls.append(x)
        def caption(self, x): calls.append(x)
    L.render_lane_links(S(), {"contract": {"inline_acquisition": {"lanes": {"links": [
        {"title": "T", "url": "https://a.example/", "read": "yes"}]}}}})
    assert any("link only" in c for c in calls) and not any(c.endswith("- read") for c in calls)

