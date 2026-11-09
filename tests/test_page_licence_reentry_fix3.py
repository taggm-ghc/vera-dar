"""H19-FIX3-1109 (OD-1109-T8C: ND news stays off): residual leaks L1-L3 and false positives from the 8b8bc33 re-verify
as assertions. L1 a prominent link to an ND article with no visible outlet name; L2 the outlet name only as footer /
standalone link text; L3 "The New York Times" (links_only outlet) via documented aliases. False positives: "via" and a
leading "By" are cues for ND-outlet names only; "Source:" for other restricted outlets needs the name to END the
credit. Real extractors, vera.page_licence and web_route.read_web (fake fetch). SYNTHETIC fixtures; no network, no DB."""
import pytest

from vera import page_licence as pl, public_fetch as pf, web_route as wr

P = ("<p>County officials certified the midterm election results after a hand recount of the ballots.</p>"
     "<p>Turnout reached a record level for a midterm.</p>")


def H(tail="", head=""):
    return f"<html><head>{head}</head><body>{P}{tail}</body></html>"


VB = "https://www.votebeat.org/2026/11/07/some-story"
CASES = {  # name: (url, html, expect link-only)
    "l1_read_the_original": ("https://news.example.com/a", H(f'<p>Read the <a href="{VB}">original story here</a>.</p>'), True),
    "l1_originally_published_at_url": ("https://news.example.com/b", H(
        f'<p>Originally published at <a href="{VB}">{VB}</a></p>'), True),
    "l1_continue_reading": ("https://news.example.com/c", H(
        '<p><a href="https://www.democracynow.org/2026/11/7/x">Continue reading</a></p>'), True),
    "l1_view_original_kff": ("https://news.example.com/d", H(
        '<div><a href="https://kffhealthnews.org/news/x">View original</a></div>'), True),
    "l2_footer_link_votebeat": ("https://news.example.com/e", H('<footer><a href="https://www.votebeat.org/">Votebeat</a></footer>'), True),
    "l2_standalone_link_the_nyt": ("https://news.example.com/f", H(
        '<p><a href="https://www.nytimes.com/2026/11/07/x.html">The New York Times</a></p>'), True),
    "l2_footer_link_kaiser_alias": ("https://news.example.com/g", H(
        '<aside><a href="https://kffhealthnews.org/">Kaiser Health News</a></aside>'), True),
    "l3_nyt_full_name_credit": ("https://news.example.com/h", H("<p>This article originally appeared in The New York Times.</p>"), True),
    "l3_provided_by_nyt": ("https://news.example.com/i", H("<p>Provided by The New York Times</p>"), True),
    "l3_courtesy_washington_post": ("https://news.example.com/j", H("<p>Photo courtesy of The Washington Post.</p>"), True),
    "l3_source_usa_today": ("https://www.example.gov/k", H("<p>Source: USA Today</p>"), True),
    "nd_via_votebeat": ("https://news.example.com/l", H("<p>Via Votebeat</p>"), True),
    "nd_by_line_democracy_now": ("https://news.example.com/m", H("<p>By Democracy Now!</p>"), True),
    "nd_source_common_dreams_analysis": ("https://news.example.com/n", H("<p>Source: Common Dreams analysis</p>"), True),
    # false positives that must be usable
    "fp_source_reuters_analysis": ("https://www.example.gov/o", H(
        "<p>Figure 3. Election spending. Source: Reuters analysis of FEC data.</p>"), False),
    "fp_by_npr_standards": ("https://journals.example.org/p", H("<p>By NPR standards the sample is small.</p>"), False),
    "fp_via_bbc_sounds": ("https://journals.example.org/q", H("<p>Data were accessed via BBC Sounds archive for coding.</p>"), False),
    "fp_source_reuters_inline_sentence": ("https://journals.example.org/r", H("<p>Source: Reuters poll shows a split.</p>"), False),
    "fp_plain_mention_link_in_sentence": ("https://journals.example.org/s", H(
        f'<p>A <a href="{VB}">Votebeat report</a> in 2024 covered certification delays.</p>'), False),
    "fp_plain_mention_nyt_in_sentence": ("https://journals.example.org/t", H(
        '<p>Coverage in <a href="https://www.nytimes.com/x">The New York Times</a> and elsewhere covered the story.</p>'), False),
    "fp_link_to_unrestricted_original": ("https://journals.example.org/u", H(
        '<p>Read the <a href="https://doi.org/10.1/x">original paper</a>.</p>'), False),
    "fp_mention_nyt_text_no_cue": ("https://journals.example.org/v", H("<p>The New York Times published an editorial.</p>"), False),
}


def _got(url, html):
    return {"text": pf.extract_visible_text(html, "text/html"), "url": url, "content_type": "text/html",
            "licence_href": pf._licence_href(html, url), "canonical_url": pf._canonical_href(html, url),
            "canonical_signals": pf.canonical_signals(html, url), "outbound_links": pf.outbound_links(html, url),
            "third_party_text": pf.third_party_segments(html, "text/html")}


def _used(url, got):
    c = {"url": url, "title": "t", "snippet": "", "doi": None, "discovered_url": url, "portal_origin": True}
    out = wr.read_web("What happened in the midterm election certification?", [c],
                      fetch=lambda u, _g=got, **k: dict(_g), registry={}, deny_list=[], cfg=wr.load_config())
    return bool(out.pages)


@pytest.mark.parametrize("name", list(CASES))
def test_fix3_case(name):
    url, html, want_block = CASES[name]
    got = _got(url, html)
    assert _used(url, got) is (not want_block), (name, pl.check(url, dict(got))["reasons"])
    assert pl.check(url, dict(got))["ok"] is (not want_block)


def test_outbound_links_extraction():
    ls = pf.outbound_links(H('<p>See <a href="/rel">rel</a> and <a href="javascript:x">js</a>.</p>'
                             '<footer><a href="https://o.example/">Out</a></footer><p><a href="https://p.example/">Solo</a></p>'),
                           "https://h.example/page")
    assert [l["url"] for l in ls] == ["https://h.example/rel", "https://o.example/", "https://p.example/"]
    assert ls[0]["before"].endswith("See") and not ls[0]["footer"] and not ls[0]["alone"]
    assert ls[1]["footer"] and ls[1]["alone"] and ls[2]["alone"] and not ls[2]["footer"]


def test_page_without_outbound_links_key_still_checked():
    url, html, _ = CASES["fp_plain_mention_link_in_sentence"]
    got = _got(url, html)
    got.pop("outbound_links")
    assert pl.check(url, got)["ok"]
