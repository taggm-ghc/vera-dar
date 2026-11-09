"""H19-FIX2-1109 (OD-1109-T8C: ND news stays off): the reviewer's re-entry probe at fb9022c
(work/H19-T8-1109/reentry_probe_fb9022c.py), 18 cases, as assertions, plus the closed gaps' variants:
G-a canonical-like signals (og:url, twitter:url, JSON-LD url/mainEntityOfPage, rel=amphtml), G-b/G-d credit lines
("Provided by", "Source:", "Courtesy of", "via", "produced by", "in partnership with", "reprinted from"), G-c documented
outlet aliases (config/reporting_lane.json `aliases`), G-e a no-derivatives licence stated in the page text.
Real public_fetch extractors, vera.page_licence and web_route.read_web (fake fetch). SYNTHETIC fixtures; no network,
no DB."""
import json

import pytest

from vera import page_licence as pl, public_fetch as pf, reporting_lane, web_route as wr

P = ("<p>County officials certified the midterm election results after a hand recount of the ballots.</p>"
     "<p>Turnout reached a record level for a midterm.</p>")


def H(head="", tail=""):
    return f"<html><head>{head}</head><body>{P}{tail}</body></html>"


VB = "https://www.votebeat.org/2026/11/07/x"
PROBE = {  # name: (fetched URL, final URL after redirects, html, expect link-only)
    "amp_cache_with_canonical": ("https://www-votebeat-org.cdn.ampproject.org/c/s/www.votebeat.org/x", None,
                                 H(f'<link rel="canonical" href="{VB}">'), True),
    "amp_subdomain": ("https://amp.votebeat.org/x", None, H(), True),
    "mobile_subdomain": ("https://m.democracynow.org/x", None, H(), True),
    "shortener_redirect_to_ND": ("https://bit.ly/abc", VB, H(), True),
    "shortener_no_redirect_canonical": ("https://t.co/abc", None, H(f'<link rel="canonical" href="{VB}">'), True),
    "og_url_only": ("https://news.example.com/a", None, H(f'<meta property="og:url" content="{VB}">'), True),
    "aggregator_provided_by": ("https://www.msn.com/en-us/news/x", None, H(tail="<p>Provided by Votebeat</p>"), True),
    "aggregator_source_colon": ("https://news.example.com/b", None, H(tail="<p>Source: Common Dreams</p>"), True),
    "cue_name_variant_democracy_now_bang": ("https://news.example.com/c", None, H(
        tail="<p>This story was originally published by Democracy Now!</p>"), True),
    "cue_upper_case": ("https://news.example.com/d", None, H(
        tail="<p>THIS ARTICLE WAS ORIGINALLY PUBLISHED BY KFF HEALTH NEWS.</p>"), True),
    "cue_old_name_KHN": ("https://news.example.com/e", None, H(
        tail="<p>This story was originally published by Kaiser Health News.</p>"), True),
    "produced_by_kff": ("https://news.example.com/f", None, H(
        tail="<p>KFF Health News is a national newsroom. This article was produced by KFF Health News.</p>"), True),
    "republished_conversation": ("https://news.example.com/g", None, H(
        tail="<p>This article is republished from The Conversation under a Creative Commons license.</p>"), True),
    "text_only_cc_by_nc_nd_statement": ("https://blog.example.com/h", None, H(
        tail="<p>This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 "
             "International License.</p>"), True),
    "reuters_canonical": ("https://news.example.com/i", None,
                          H('<link rel="canonical" href="https://www.reuters.com/world/x">'), True),
    # must stay usable
    "oa_paper_doi_canonical": ("https://journals.example.org/p", None,
                               H('<link rel="canonical" href="https://doi.org/10.1/x">'), False),
    "oa_paper_originally_published_in_journal": ("https://journals.example.org/q", None, H(
        tail="<p>This article was originally published in Physical Review D.</p>"), False),
    "plain_page_mentions_votebeat_report": ("https://journals.example.org/r", None, H(
        tail="<p>A Votebeat report in 2024 covered certification delays.</p>"), False),
}
EXTRA = {  # variants of the closed gaps (link-only) and near misses (usable)
    "twitter_url_only": ("https://news.example.com/j", H(f'<meta name="twitter:url" content="{VB}">'), True),
    "jsonld_url": ("https://news.example.com/k", H('<script type="application/ld+json">{"@type":"NewsArticle",'
                                                  f'"url":"{VB}"}}</script>'), True),
    "jsonld_main_entity_in_graph": ("https://news.example.com/l", H(
        '<script type="application/ld+json">{"@graph":[{"@type":"WebPage"},{"@type":"NewsArticle",'
        '"mainEntityOfPage":{"@id":"https://kffhealthnews.org/news/x"}}]}</script>'), True),
    "amphtml_back_reference": ("https://news.example.com/m", H(
        '<link rel="canonical" href="https://news.example.com/m">'
        '<link rel="amphtml" href="https://amp.votebeat.org/x">'), True),
    "own_canonical_but_og_url_nd": ("https://news.example.com/n", H(
        f'<link rel="canonical" href="https://news.example.com/n"><meta property="og:url" content="{VB}">'), True),
    "courtesy_of_outlet": ("https://news.example.com/o", H(tail="<p>Courtesy of ProPublica.</p>"), True),
    "via_outlet": ("https://news.example.com/p", H(tail="<p>Story via Chalkbeat</p>"), True),
    "in_partnership_with": ("https://news.example.com/q", H(
        tail="<p>This story was published in partnership with KFF Health News.</p>"), True),
    "reprinted_from": ("https://news.example.com/r", H(tail="<p>Reprinted from Truthout.</p>"), True),
    "byline_outlet": ("https://news.example.com/s", H(tail="<p>By The Texas Tribune</p>"), True),
    "copyright_outlet": ("https://news.example.com/t", H(tail="<p>© 2026 Common Dreams</p>"), True),
    "credit_old_name": ("https://news.example.com/u", H(tail="<p>Source: Kaiser Health News</p>"), True),
    "cc_by_nc_nd_footer_badge": ("https://blog.example.com/v", H(tail="<footer>CC BY-NC-ND 4.0</footer>"), True),
    "cc_by_nd_text": ("https://blog.example.com/w", H(tail="<p>Licence: CC BY-ND 4.0.</p>"), True),
    "nd_licence_url_as_text": ("https://blog.example.com/x", H(
        tail="<p>See creativecommons.org/licenses/by-nc-nd/4.0 for terms.</p>"), True),
    "noderivs_30_name": ("https://blog.example.com/y", H(
        tail="<p>Content is available under Creative Commons Attribution-NonCommercial-NoDerivs 3.0.</p>"), True),
    # usable
    "plain_mention_kff_newsroom": ("https://journals.example.org/s", H(
        tail="<p>KFF Health News is a national newsroom that covers health policy.</p>"), False),
    "plain_mention_old_name": ("https://journals.example.org/t", H(
        tail="<p>Kaiser Health News reported on the same certification delays in 2019.</p>"), False),
    "credit_to_unrestricted_source": ("https://journals.example.org/u", H(
        tail="<p>Source: U.S. Election Assistance Commission. Image courtesy of NASA. Data via the Census Bureau.</p>"),
        False),
    "own_og_url_same_host": ("https://journals.example.org/v", H(
        '<meta property="og:url" content="https://journals.example.org/v">'), False),
    "og_url_doi": ("https://journals.example.org/w", H('<meta property="og:url" content="https://doi.org/10.1/x">'),
                   False),
    "cc_by_text_statement": ("https://journals.example.org/x", H(
        tail="<p>This work is licensed under a Creative Commons Attribution 4.0 International License.</p>"), False),
    "cc_by_nc_sa_text_statement": ("https://journals.example.org/y", H(
        tail="<p>Licensed under CC BY-NC-SA 4.0.</p>"), False),
}


def _got(url, html):
    return {"text": pf.extract_visible_text(html, "text/html"), "url": url, "content_type": "text/html",
            "licence_href": pf._licence_href(html, url), "canonical_url": pf._canonical_href(html, url),
            "canonical_signals": pf.canonical_signals(html, url),
            "third_party_text": pf.third_party_segments(html, "text/html")}


def _used(url, got):
    c = {"url": url, "title": "t", "snippet": "", "doi": None, "discovered_url": url, "portal_origin": True}
    out = wr.read_web("What happened in the midterm election certification?", [c],
                      fetch=lambda u, _g=got, **k: dict(_g), registry={}, deny_list=[], cfg=wr.load_config())
    return bool(out.pages)


def test_probe_has_18_cases():
    assert len(PROBE) == 18


@pytest.mark.parametrize("name", list(PROBE))
def test_reentry_probe_case(name):
    """The reviewer's probe exactly: the page dict carries only canonical_url (no canonical_signals)."""
    url, final, html, want_block = PROBE[name]
    fin = final or url
    got = _got(fin, html)
    got.pop("canonical_signals")
    assert _used(url, got) is (not want_block), (name, pl.check(fin, dict(got))["reasons"])


@pytest.mark.parametrize("name", [n for n in ("og_url_only", "aggregator_provided_by", "aggregator_source_colon",
                                              "cue_old_name_KHN", "produced_by_kff",
                                              "text_only_cc_by_nc_nd_statement")])
def test_six_fb9022c_gaps_fail_at_page_check(name):
    url, final, html, _ = PROBE[name]
    r = pl.check(url, _got(url, html))
    assert not r["ok"] and r["text"] == "", name
    want = "page_licence_reject" if name.startswith("text_only") else "acc_republished"
    assert want in r["reasons"], (name, r["reasons"])


@pytest.mark.parametrize("name", list(EXTRA))
def test_gap_variants(name):
    url, html, want_block = EXTRA[name]
    got = _got(url, html)
    r = pl.check(url, dict(got))
    assert r["ok"] is (not want_block), (name, r["reasons"])
    assert _used(url, got) is (not want_block), name


def test_canonical_signals_extraction():
    html = ('<html><head><link rel="canonical" href="/self"><meta property="og:url" content="https://a.example/og">'
            '<meta name="twitter:url" content="https://b.example/tw"><link rel="amphtml" href="https://amp.c.example/x">'
            '<script type="application/ld+json">[{"url":"https://d.example/ld","mainEntityOfPage":"https://e.example/m"},'
            ' {"@graph":[{"mainEntityOfPage":{"@id":"https://f.example/g"}}]}]</script>'
            '<script type="application/ld+json">{not json</script></head><body>x</body></html>')
    base = "https://h.example/p"
    assert pf.canonical_signals(html, base) == ["https://h.example/self", "https://a.example/og",
                                                "https://b.example/tw", "https://d.example/ld",
                                                "https://e.example/m", "https://f.example/g",
                                                "https://amp.c.example/x"]
    assert pf._canonical_href(html, base) == "https://h.example/self"  # rel=canonical first
    no_link = html.replace('<link rel="canonical" href="/self">', "")
    assert pf._canonical_href(no_link, base) == "https://a.example/og"  # G-a fallback: og:url
    only_amp = '<html><head><link rel="amphtml" href="https://amp.c.example/x"></head></html>'
    assert pf._canonical_href(only_amp, base) == ""  # an AMP back-reference is not the page's canonical
    assert pf.canonical_signals('<meta property="og:url" content="javascript:alert(1)">', base) == []


def test_aliases_are_config_data_on_restricted_rows():
    cfg = reporting_lane.load_config()
    rows = {o["host"]: o for o in cfg["outlets"]}
    assert "Kaiser Health News" in rows["kffhealthnews.org"]["aliases"]
    for o in cfg["outlets"]:
        if "aliases" in o:
            assert isinstance(o["aliases"], list) and all(isinstance(a, str) and a for a in o["aliases"])
            assert pl._restricted_row(o), o["host"]  # aliases only matter for restricted rows
    labels = pl._restricted_labels(cfg)
    assert {"kaiserhealthnews", "kffhealthnews", "democracynow", "commondreams"} <= labels


def test_alias_removed_from_config_reopens_only_that_name():
    cfg = json.loads(json.dumps(reporting_lane.load_config()))
    for o in cfg["outlets"]:
        o.pop("aliases", None)
    url = "https://news.example.com/e"
    text = "Turnout reached a record level. This story was originally published by Kaiser Health News."
    assert pl.check(url, {"text": text, "url": url}, cfg=cfg)["ok"]  # the alias is data, not code
    assert not pl.check(url, {"text": text, "url": url})["ok"]


def test_broken_attribution_pattern_fails_closed():
    cfg = json.loads(json.dumps(reporting_lane.load_config()))
    cfg["page_checks"]["attribution_cue_patterns"] = ["(unclosed"]
    url = "https://journals.example.org/z"
    r = pl.check(url, {"text": "Hadrons are composite particles.", "url": url}, cfg=cfg)
    assert not r["ok"] and "acc_third_party_config" in r["reasons"]
