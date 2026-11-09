"""H19-FIX-1109 re-verify (work/H19-T8-1109/reverify-58c42e9.md F1 to F3): the reviewer's 15 realistic non-ND pages,
run through the real public_fetch extractors and vera.page_licence. Every page is usable (the licence allows it); A11
still drops only third-party passages; the shown licence follows the explicit page licence link first. The original ND
A8/A10 cases stay blocked (tests/test_lane_nd_news_h19.py CASES). SYNTHETIC fixtures; no network, no DB."""
import pytest

from vera import page_licence as pl, public_fetch as pf

P = ("<p>Hadrons are composite particles made of quarks held together by the strong force.</p>"
     "<p>Baryons and mesons are the two main families of hadrons.</p>")
CC_BY = '<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">'
CASES = {  # name: (html, expected shown-licence label or None, text that must be gone or None)
    "plain_article": (f"<html><body><article>{P}</article></body></html>", None, None),
    "oa_ccby_footer_arr": (f"<html><head>{CC_BY}</head><body><article>{P}</article><footer>Copyright © 2026 "
                           "Example Publisher B.V. All rights reserved.</footer></body></html>", "CC BY 4.0", None),
    "oa_no_cc_footer_arr": (f"<html><body><article>{P}</article><footer>© 2026 Example University. All rights "
                            "reserved.</footer></body></html>", "All rights reserved (short quotation)", None),
    "preprint_originally_published_in": ("<html><body><article><p>This article was originally published in Physical "
                                         f"Review D 101, 2020.</p>{P}</article></body></html>", None, None),
    "figure_inline_markup": (f'<html><body><article>{P}<figure><img src="x.png"><figcaption><b>Figure 1.</b> Quark '
                             "content of <i>baryons</i> and mesons.</figcaption></figure><p>More text about the quark "
                             "model and colour charge.</p></article></body></html>", None, "Quark content of"),
    "figcaption_multiline": (f"<html><body><article>{P}<figure><figcaption>Figure 2:\n   The   eightfold\n way "
                             "<span>multiplet</span></figcaption></figure></article></body></html>", None, "eightfold"),
    "cite_in_paragraph": ("<html><body><article><p>As shown by <cite>Gell-Mann (1964)</cite>, hadrons have quark "
                          f"substructure.</p>{P}</article></body></html>", None, "Gell-Mann"),
    "references_cite_list": (f"<html><body><article>{P}<ol><li><cite>Gell-Mann M. A schematic model of baryons and "
                             "mesons. Phys Lett. 1964;8:214.</cite></li><li><cite>Zweig G. CERN report 1964.</cite></li>"
                             "</ol></article></body></html>", None, "Zweig"),
    "q_inline": (f"<html><body><article><p>Feynman called them <q>partons</q> in 1969.</p>{P}</article></body></html>",
                 None, "partons"),
    "blockquote_abstract": ("<html><body><article><blockquote><p>Abstract: We study hadron masses on the lattice.</p>"
                            f"</blockquote>{P}</article></body></html>", None, "lattice"),
    "table_in_figure": (f"<html><body><article>{P}<figure><table><tr><td>proton</td><td>938 MeV</td></tr><tr><td>"
                        "neutron</td><td>940 MeV</td></tr></table></figure></article></body></html>", None, "938 MeV"),
    "canonical_doi_org": ('<html><head><link rel="canonical" href="https://doi.org/10.1000/xyz"></head><body><article>'
                          f"{P}</article></body></html>", None, None),
    "canonical_other_host_mirror": ('<html><head><link rel="canonical" href="https://www.ncbi.nlm.nih.gov/pmc/articles/'
                                    f'PMC1/"></head><body><article>{P}</article></body></html>', None, None),
    "mentions_reuters_in_text": (f"<html><body><article>{P}<p>A Reuters report in 2012 covered the Higgs discovery.</p>"
                                 "</article></body></html>", None, "Reuters"),
    "gov_courtesy_of_image_credit": (f"<html><body><article>{P}<p>Image courtesy of NASA.</p></article></body></html>",
                                     None, "courtesy"),
}
URL = "https://journals.example.org/article/1"


def _check(html):
    text = pf.extract_visible_text(html, "text/html")
    page = {"text": text, "url": URL, "licence_href": pf._licence_href(html, URL),
            "canonical_url": pf._canonical_href(html, URL), "third_party_text": pf.third_party_segments(html, "text/html")}
    return pl.check(URL, page)


def test_fifteen_cases():
    assert len(CASES) == 15


@pytest.mark.parametrize("name", list(CASES))
def test_non_nd_page_is_usable(name):
    html, label, gone = CASES[name]
    r = _check(html)
    assert r["ok"], (name, r["reasons"])
    assert "Hadrons are composite particles" in r["text"]
    assert (r["licence"] or {}).get("label") == label
    if gone:
        assert gone not in r["text"]


def test_non_nd_row_with_explicit_markers_still_applies():
    row = {"host": "journals.example.org", "republished_markers": ["syndicated by wire desk"],
           "exclude_item_markers": ["members only item"]}
    assert _check(f"<html><body>{P}<p>Syndicated by Wire Desk.</p></body></html>") ["ok"]
    r = pl.check(URL, {"text": "Hadrons are composite particles. Syndicated by Wire Desk.", "url": URL}, row=row)
    assert not r["ok"] and r["reasons"] == ["acc_republished"]
    r2 = pl.check(URL, {"text": "Hadrons are composite particles. Members only item.", "url": URL}, row=row)
    assert not r2["ok"] and r2["reasons"] == ["acc_item_excluded"]


# ------------------------------------------------- bf66ade re-check: ND content re-entering through another host
from vera import web_route as wr  # noqa: E402

HP = ("<p>County officials certified the midterm election results after a hand recount of the ballots.</p>"
      "<p>Turnout reached a record level for a midterm.</p>")
HOLES = {  # reviewer's work/H19-T8-1109/hole_probe_bf66ade.py: every case must be link-only
    "nonND_host_canonical_to_votebeat": ("https://news.example.com/a", '<html><head><link rel="canonical" '
                                         f'href="https://www.votebeat.org/2026/11/07/x"></head><body>{HP}</body></html>'),
    "nonND_host_canonical_to_nd_subdomain": ("https://news.example.com/a2", '<html><head><link rel="canonical" '
                                             f'href="https://data.votebeat.org/x"></head><body>{HP}</body></html>'),
    "nonND_host_canonical_to_links_only_row": ("https://news.example.com/a3", '<html><head><link rel="canonical" '
                                               f'href="https://www.nytimes.com/2026/x"></head><body>{HP}</body></html>'),
    "nonND_host_text_originally_published_by_votebeat": (
        "https://news.example.com/b", f"<html><body>{HP}<p>This story was originally published by Votebeat.</p></body></html>"),
    "nonND_host_republished_from_kff_health_news": (
        "https://news.example.com/b2", f"<html><body>{HP}<p>Republished from KFF Health News.</p></body></html>"),
    "nonND_host_republished_from_commondreams_with_ccbyncnd_link": (
        "https://news.example.com/c", '<html><head><link rel="license" href="https://creativecommons.org/licenses/by-nc-nd/'
        f'3.0/"></head><body>{HP}<p>Republished from Common Dreams.</p></body></html>'),
    "nonND_page_explicit_ND_licence_link": ("https://journals.example.org/p", '<html><head><link rel="license" '
                                            'href="https://creativecommons.org/licenses/by-nd/4.0/"></head>'
                                            f"<body>{HP}</body></html>"),
    "nonND_page_cc_by_nc_nd_link": ("https://journals.example.org/q", '<html><head><link rel="license" '
                                    'href="https://creativecommons.org/licenses/by-nc-nd/4.0/"></head>'
                                    f"<body>{HP}</body></html>"),
    "ND_host_page_direct": ("https://www.votebeat.org/2026/11/07/y", f"<html><body>{HP}</body></html>"),
    "ND_subdomain_page": ("https://data.votebeat.org/z", f"<html><body>{HP}</body></html>"),
}


def _got(url, html):
    return {"text": pf.extract_visible_text(html, "text/html"), "url": url, "content_type": "text/html",
            "licence_href": pf._licence_href(html, url), "canonical_url": pf._canonical_href(html, url),
            "third_party_text": pf.third_party_segments(html, "text/html")}


@pytest.mark.parametrize("name", list(HOLES))
def test_nd_reentry_through_other_hosts_is_link_only(name):
    url, html = HOLES[name]
    got = _got(url, html)
    if not name.startswith("ND_"):  # pages on ND hosts are stopped by the web route's outlet rule
        assert not pl.check(url, dict(got))["ok"], name
    c = {"url": url, "title": "t", "snippet": "", "doi": None, "discovered_url": url, "portal_origin": True}
    out = wr.read_web("What happened in the midterm election certification?", [c],
                      fetch=lambda u, _g=got, **k: dict(_g), registry={}, deny_list=[], cfg=wr.load_config())
    assert out.pages == [], name


@pytest.mark.parametrize("text", ["This article was originally published in Physical Review D 101, 2020.",
                                  "Republished from the author's institutional repository.",
                                  "Originally published by Global Voices."])  # allow row: not restricted
def test_generic_republication_of_unrestricted_source_stays_usable(text):
    url = "https://journals.example.org/article/2"
    r = pl.check(url, {"text": "Hadrons are composite particles made of quarks. " + text, "url": url})
    assert r["ok"], r["reasons"]
