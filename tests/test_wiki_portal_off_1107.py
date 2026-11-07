"""WIKI-OFF-1107: the shipped Wikipedia citation portal is OFF. robots.txt disallows /w/ (the portal's api.php path) for
all agents, so the portal must not call the API until a robots-compliant route exists (WIKI-REDESIGN). Offline: the fake
http_get fails the test if it is ever called."""
from __future__ import annotations

from vera import wiki_ref_portal as w


def _no_http(*a, **k):
    raise AssertionError("portal must not call HTTP while the shipped switch is off")


def test_shipped_config_is_disabled():
    c = w.load_config()
    assert c["enabled"] is False
    assert w.enabled(c) is False


def test_discover_returns_disabled_without_http():
    cands, links, counts = w.discover("zorblat gait patterns", http_get=_no_http, url_check=lambda u: (True, "ok"))
    assert cands == [] and links == []
    assert counts["error"] == "disabled"
    assert counts["requests"] == 0
