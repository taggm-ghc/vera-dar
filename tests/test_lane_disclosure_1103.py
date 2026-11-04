"""NONASK-1103 SD.1: lane disclosure appears only while a lane flag is on."""
import pytest

from vera import disclosure, official_lane, reporting_lane

SEARCH = ("Search terms derived from your question may also be sent to GDELT (a news index) and Hacker News search "
          "(Algolia).")
PAGES = ("VERA-DAR may read public web pages from vendors and news outlets to answer; these pages are used only for "
         "your answer, and page text is not saved.")
LINK = "https://www.gdeltproject.org/"


def _all():
    return [disclosure.disclosure_text(), disclosure.required_notice_text(), disclosure.short_notice_text(),
            " ".join(t for _h, t in disclosure.disclosure_sections())]


def _set(mp, off, rep):
    mp.setattr(official_lane, "lane_enabled", lambda cfg=None: off)
    mp.setattr(reporting_lane, "lane_enabled", lambda cfg=None: rep)
    mp.setattr(disclosure, "web_read_on", lambda: False)  # A6-W web sentence: tested in test_acq6_web_wiring_1103


def test_flags_off_no_lane_text(monkeypatch):
    _set(monkeypatch, False, False)
    assert not disclosure.lanes_disclosure_on()
    for t in _all():
        for bad in ("GDELT", "Algolia", "page text is not saved", "licence-cleared"):
            assert bad not in t
    assert disclosure.short_notice_text() == disclosure.SHORT_NOTICE
    assert disclosure.third_party_text() == disclosure._THIRD_PARTY
    assert "".join(t for _h, t in disclosure.disclosure_sections()).replace(" ", "") == \
        disclosure.disclosure_text().strip().replace(" ", "")


def test_flag_errors_fail_closed(monkeypatch):
    def boom(cfg=None):
        raise RuntimeError("x")
    monkeypatch.setattr(official_lane, "lane_enabled", boom)
    monkeypatch.setattr(reporting_lane, "lane_enabled", boom)
    assert disclosure.lanes_disclosure_on() is False


def test_reporting_on(monkeypatch):
    _set(monkeypatch, False, True)
    for t in _all()[:3]:
        assert SEARCH in t and PAGES in t
    secs = dict(disclosure.disclosure_sections())
    assert LINK in secs["Your question and what is stored"]
    assert "Hacker News search (Algolia)" in secs["Your question and what is stored"]
    assert "licence-cleared or stored" in " ".join(secs.values())


def test_official_only_has_pages_sentence_not_gdelt(monkeypatch):
    _set(monkeypatch, True, False)
    t = disclosure.disclosure_text()
    assert PAGES in t and "GDELT" not in t and "Algolia" not in t
