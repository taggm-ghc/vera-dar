"""ARX-A2 SYNTHETIC fixtures modelled on arXiv's abs page and LaTeXML HTML structure for record 2605.23989v1.
Only the record identity (id, version, title) is real; every author name, sentence and reference is invented
(tests/fulltext_synth.py sentence generator). No real article text is copied."""
from __future__ import annotations

from tests import acq_support as A
from tests import fulltext_synth

AID, VER = "2605.23989", "v1"
TITLE = ("Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, "
         "and system security")
ABS_URL = f"https://arxiv.org/abs/{AID}{VER}"
HTML_URL = f"https://arxiv.org/html/{AID}{VER}"
DOI = f"10.48550/arxiv.{AID}"
CC_BY = "http://creativecommons.org/licenses/by/4.0/"
CC_BY_ND = "http://creativecommons.org/licenses/by-nd/4.0/"
ARXIV_DEFAULT = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
CHROME = ("NAVCHROME-TOKEN", "SCRIPTCHROME-TOKEN", "FOOTERCHROME-TOKEN", "HEADERCHROME-TOKEN", "BUTTONCHROME-TOKEN",
          "STYLECHROME-TOKEN")


def abs_page(licence: str | None = CC_BY, arxiv_id: str = AID, versions=("v1",)) -> bytes:
    lic = (f'<div class="abs-license"><a href="{licence}" title="Rights to this article" class="has_license">'
           f'licence</a></div>' if licence else "")
    hist = "".join(f"<strong>[{v}]</strong> Thu, 21 May 2026 (12 KB)<br/>" for v in versions)
    return (f'<!DOCTYPE html><html lang="en"><head><title>[{arxiv_id}] synthetic</title>'
            f'<meta name="citation_title" content="{TITLE}" />'
            f'<meta name="citation_author" content="Synthauthor, Alpha" />'
            f'<meta name="citation_author" content="Exampleauthor, Beta" />'
            f'<meta name="citation_date" content="2026/05/21" />'
            f'<meta name="citation_arxiv_id" content="{arxiv_id}" /></head><body>'
            f'<header>HEADERCHROME-TOKEN</header><div id="abs"><h1 class="title mathjax">{TITLE}</h1>'
            f'<blockquote class="abstract mathjax">SYNTHETIC abstract: transmission incidence outbreak pathogen.'
            f'</blockquote>{lic}<div class="submission-history">{hist}</div></div></body></html>').encode()


def latexml_html(paras: int = 16, topic: str = "epi") -> bytes:
    def para(k):
        return f'<p class="ltx_p">{" ".join(fulltext_synth._sentences(topic, 6, k))}</p>'
    half = paras // 2
    return ('<!DOCTYPE html><html lang="en"><head><title>SYNTHETIC LaTeXML page</title>'
            '<style>.x{content:"STYLECHROME-TOKEN"}</style><script>var t="SCRIPTCHROME-TOKEN";</script></head><body>'
            '<header class="desktop_header"><nav>NAVCHROME-TOKEN</nav><button>BUTTONCHROME-TOKEN</button></header>'
            '<div class="ltx_page_main"><div class="ltx_page_content">'
            '<article class="ltx_document ltx_authors_1line">'
            f'<h1 class="ltx_title ltx_title_document">{TITLE}</h1>'
            '<div class="ltx_authors"><span class="ltx_creator ltx_role_author">Alpha Synthauthor</span></div>'
            '<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
            f'<p class="ltx_p">{" ".join(fulltext_synth._sentences(topic, 3, 50))}</p></div>'
            '<section class="ltx_section" id="S1"><h2 class="ltx_title ltx_title_section">'
            '<span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>'
            f'{"".join(para(k) for k in range(half))}'
            '<figure class="ltx_figure" id="S1.F1"><img src="x1.png" alt="" />'
            '<figcaption class="ltx_caption">Figure 1: SYNTHETIC caption.</figcaption></figure></section>'
            '<section class="ltx_section" id="S2"><h2 class="ltx_title ltx_title_section">'
            '<span class="ltx_tag ltx_tag_section">2 </span>Methods</h2>'
            f'{"".join(para(k) for k in range(half, paras))}</section>'
            '<section class="ltx_bibliography" id="bib"><h2 class="ltx_title ltx_title_bibliography">References</h2>'
            '<ul class="ltx_biblist"><li class="ltx_bibitem" id="bib.bib1">Synthetic Ref 1. doi:10.9999/ref.1</li>'
            '</ul></section></article></div>'
            '<footer class="ltx_page_footer">FOOTERCHROME-TOKEN</footer></div></body></html>').encode()


def html_resp(body: bytes, ctype="text/html; charset=utf-8", status=200, headers=None):
    return A.Resp(status=status, body=body, ctype=ctype, headers=headers)


def discovery_src(url=ABS_URL, doi=None):
    return {"provider": "arxiv", "url": url, "doi": doi, "title": TITLE,
            "licence": {"metadata": {"id": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}},
            "stored_text_kind": "abstract_metadata",
            "snippet": "transmission incidence outbreak pathogen", "identifier": f"arXiv:{AID}{VER}"}


def routes(licence=CC_BY, html=None, abs_body=None):
    """URL -> response for the arXiv route (abs + html); everything else falls through to the Router (404)."""
    return {ABS_URL: html_resp(abs_body if abs_body is not None else abs_page(licence)),
            HTML_URL: html if html is not None else html_resp(latexml_html())}
