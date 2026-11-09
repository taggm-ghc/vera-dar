"""H19-FIX-1109 (design T4 tests): the Wikipedia-citation path. Wikipedia is never a permission input: a citation URL
reaches page text only through the web route's gates, where an ND outlet page is never used as a plain web page
(OD-1109-T8C, ND news off) and every other page passes the page-level checks (A7, A8, A10, A11) first. NT-3 news links
stay link-only and are never fetched. SYNTHETIC fixtures; stub fetch; no network, no DB."""
import json

import pytest

from vera import lane_evidence as le, licence_gate as lg, public_fetch as pf, reporting_lane as rl
from vera import web_route as wr, wiki_ref_portal as wp

ND_URL = "https://www.votebeat.org/2026/11/07/dup"
OPEN_URL = "https://blog.example.org/2026/zorblats"
BODY = ("Zorblats are synthetic test organisms used in this fixture. " * 6).strip()
_ORIG = rl.load_config


class Fetch:
    def __init__(self, **extra):
        self.calls, self.extra = [], extra

    def __call__(self, url, hop_gate=None, robots_cache=None):
        refused = hop_gate(url) if hop_gate is not None else None
        if refused:
            raise pf.FetchRefused(refused)
        self.calls.append(url)
        return {"url": url, "content_type": "text/html", "x_robots_tag": "", "content_signal": "", "meta_robots": "",
                "licence_href": "", "text": BODY, **self.extra}


def _read(url, f):
    c = wp.to_url_candidate(url)
    assert c is not None
    return wr.read_web("what are zorblats?", [c], fetch=f, registry=[], deny_list=[])


def test_nt3_news_links_are_link_only_and_never_fetched():
    items = le.wiki_link_items([ND_URL, "https://en.wikipedia.org/wiki/X", "https://www.propublica.org/a"])
    assert [i["url"] for i in items] == [ND_URL]  # Wikipedia host dropped; excluded outlet dropped
    assert items[0]["read"] is False and items[0]["credit"] == le.WIKI_CREDIT


@pytest.mark.parametrize("flag_on", [False, True])
def test_citation_to_nd_outlet_is_never_used_as_web_text(monkeypatch, flag_on):
    if flag_on:  # even with the ND switches forced on, the web route never uses ND outlet text
        def on(*a, **k):
            c = json.loads(json.dumps(_ORIG(*a, **k)))
            c["nc_basis"]["status"] = "SATISFIED"
            return c
        monkeypatch.setattr(rl, "load_config", on)
        monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: True)
    out = _read(ND_URL, Fetch())
    assert out.pages == [] and out.listed and out.listed[0]["read"] is False
    assert any(d["gate"].get("use") == "use_nd_licence" for d in out.dev)


def test_citation_to_open_page_is_used_after_a11_filter_with_licence():
    f = Fetch(licence_href="https://creativecommons.org/licenses/by/4.0/",
              text=BODY + " The minister told Reuters the figures were final.", meta_author="A. Writer")
    out = _read(OPEN_URL, f)
    assert len(out.pages) == 1
    pg = out.pages[0]
    assert "Reuters" not in pg.text and pg.licence_info["label"] == "CC BY 4.0"
    att = le.attribution_for(pg)
    assert att["licence"] == "CC BY 4.0" and att["licence_url"].endswith("/by/4.0/") and "A. Writer" in att["creator"]


@pytest.mark.parametrize("extra,token", [
    ({"text": BODY + " This article was originally published by Another Outlet."}, "use_acc_republished"),
    ({"canonical_url": "https://syndicator.example.com/zorblats"}, "use_acc_republished"),
    ({"licence_href": "https://creativecommons.org/licenses/by/4.0/", "text": BODY + " All Rights Reserved."},
     "use_acc_item_excluded"),
    ({"licence_href": "https://creativecommons.org/licenses/by-nc-nd/4.0/"}, "use_nd_licence"),
    ({"text": "NEW YORK (AP) — " + BODY}, "use_acc_third_party_page"),
])
def test_citation_page_level_checks_make_it_link_only(extra, token):
    out = _read(OPEN_URL, Fetch(**extra))
    assert out.pages == [] and out.listed
    uses = [d["gate"].get("use") for d in out.dev]
    assert token in uses or (token == "use_nd_licence" and any(u and "licence" in u for u in uses)), uses
