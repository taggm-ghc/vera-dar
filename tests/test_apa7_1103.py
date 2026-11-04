"""NONASK-1103 C1: APA 7 formatter unit tests (fictitious works; offline)."""
import json
from pathlib import Path

import pytest

from vera import apa7

A = lambda f, g: {"family": f, "given": g}  # noqa: E731
D = lambda *p: {"date-parts": [list(p)]}  # noqa: E731


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(apa7, "_public", lambda u: "10.0.0" not in u)


def J(**kw):
    r = {"type": "article-journal", "title": "Sleep and recall", "author": [A("Rivera", "Lucia"), A("Okafor", "Tunde")],
         "issued": D(2021), "container_title": "Journal of Memory", "volume": 14, "issue": 2, "page": "101-118"}
    r.update(kw)
    return r


def test_journal_doi_pmcid_and_intext():
    s = apa7.reference(J(DOI="10.1234/jm.2021.014", PMCID="PMC7654321"), links=False)
    assert s == ("Rivera, L., & Okafor, T. (2021). Sleep and recall. *Journal of Memory*, *14*(2), 101–118. "
                 "https://doi.org/10.1234/jm.2021.014 PMCID: PMC7654321")
    assert apa7.in_text(J()) == "(Rivera & Okafor, 2021)"
    assert apa7.in_text(J(), narrative=True) == "Rivera and Okafor (2021)"


def test_author_counts():
    three = J(author=[A("Chen", "A"), A("Lee", "B"), A("Kim", "C")])
    assert apa7.in_text(three) == "(Chen et al., 2021)"
    many = J(author=[A(f"F{i:02d}", "X") for i in range(22)])
    s = apa7.reference(many, links=False)
    assert "F18, X., . . . F21, X. (2021)" in s and "&" not in s.split("(2021)")[0]
    assert "F19" not in s and "F20" not in s


def test_joined_author_string_is_one_literal():
    r = J(author="Lin, Wei, Chen, Hao")
    assert apa7.reference(r, links=False).startswith("Lin, Wei, Chen, Hao. (2021)")
    assert "et al" not in apa7.in_text(r)


def test_group_author_not_initialised():
    r = J(author=[{"literal": "Example Cloud"}])
    assert apa7.reference(r, links=False).startswith("Example Cloud. (2021)")


def test_no_author_no_date_never_invented():
    r = {"type": "webpage", "title": "Page title"}
    s = apa7.reference(r, links=False)
    assert s == "*Page title*. (n.d.)."
    assert apa7.in_text({"type": "article-journal", "title": "Using Citations"}) == '("Using Citations," n.d.)'


def test_isbn_checksum():
    ch = {"type": "chapter", "title": "Ch", "author": [A("Armstrong", "D")], "issued": D(2019),
          "container_title": "A companion", "publisher": "Brewer"}
    assert apa7.reference(dict(ch, ISBN="978-0-306-40615-7"), links=False).endswith("ISBN 9780306406157")
    s = apa7.reference_ex(dict(ch, ISBN="9780306406158"))
    assert "ISBN" not in s[0] and "R-CIT-BADID-isbn" in s[1]
    assert apa7.valid_isbn("0306406152") == "0306406152" and apa7.valid_isbn("0306406153") is None


def test_doi_validation_omits_not_repairs():
    for bad in ("10.1234/a)(b", "10.12/x", "not a doi", "10.1234/a b"):
        s, codes = apa7.reference_ex(J(DOI=bad))
        assert "doi.org" not in s and "R-CIT-BADID-doi" in codes, bad


def test_arxiv_patterns():
    assert apa7.valid_arxiv("2401.01234v2") and apa7.valid_arxiv("hep-th/9901001") and apa7.valid_arxiv("math.GT/0309136")
    assert apa7.valid_arxiv("240.1234") is None and apa7.valid_arxiv("2401.123") is None


def test_version_read_first():
    r = J(type="preprint", arxiv="2401.01234v1", DOI="10.1234/vor", read_version="arxiv")
    s = apa7.reference(r, links=False)
    assert s.index("arXiv:2401.01234v1") < s.index("doi.org/10.1234/vor")
    r2 = J(arxiv="2401.01234", DOI="10.1234/vor", foreign_ids=["doi"], read_version="arxiv")
    assert "doi.org" not in apa7.reference(r2, links=False)


def test_precedence_doi_over_handle_and_pmid_only():
    s = apa7.reference(J(DOI="10.1234/x", handle="1234/abc"), links=False)
    assert "doi.org/10.1234/x" in s and "hdl.handle" not in s
    assert apa7.reference(J(PMID="123"), links=False).endswith("PMID: 123")


def test_handle_only_from_url_or_field():
    assert "hdl.handle.net/20.500/ab" in apa7.reference(J(URL="https://hdl.handle.net/20.500/ab"), links=False)
    assert apa7.valid_handle("2024/report") == "2024/report"  # explicit field form
    s = apa7.reference(J(title="See 2024/report"), links=False)
    assert "hdl" not in s


def test_ids_from_url_via_ddg_identifier():
    s = apa7.reference(J(URL="https://arxiv.org/abs/2401.01234v2"), links=False)
    assert s.endswith("arXiv:2401.01234v2")


def test_news_with_and_without_byline():
    n = {"type": "article-newspaper", "title": "Chip export rules change", "container_title": "The Daily Ledger",
         "issued": D(2026, 10, 30), "URL": "https://example.org/a"}
    assert apa7.reference(dict(n, author=[A("Park", "J")]), links=False) == (
        "Park, J. (2026, October 30). Chip export rules change. *The Daily Ledger*. https://example.org/a")
    s = apa7.reference(n, links=False)
    assert s.startswith("Chip export rules change. (2026, October 30). *The Daily Ledger*.")
    assert apa7.in_text(n) == '("Chip export rules change," 2026)'


def test_changing_page_retrieved():
    r = {"type": "document", "title": "Rate limits", "author": [{"literal": "Example Cloud"}],
         "URL": "https://docs.example.com/limits", "changing": True, "accessed": D(2026, 11, 3)}
    assert apa7.reference(r, links=False) == ("Example Cloud. (n.d.). *Rate limits*. "
                                              "Retrieved November 3, 2026, from https://docs.example.com/limits")
    assert apa7.in_text(r) == "(Example Cloud, n.d.)"


def test_domain_never_group_author():
    s = apa7.reference({"type": "webpage", "title": "T", "URL": "https://docs.example.com/x"}, links=False)
    assert not s.startswith("docs.example.com") and "(n.d.)" in s
    assert "site: docs.example.com" not in s


def test_dataset_software():
    d = {"type": "dataset", "title": "Arts trends", "author": [{"literal": "Grantmakers"}], "issued": D(2019), "number": "37337"}
    assert apa7.reference(d, links=False) == "Grantmakers. (2019). *Arts trends* (37337) [Data set]."
    sw = {"type": "software", "title": "Maple", "author": [{"literal": "Maplesoft"}], "issued": D(2019),
          "version": "2.1.0", "publisher": "Cybernet"}
    assert apa7.reference(sw, links=False) == "Maplesoft. (2019). *Maple* (Version 2\\.1\\.0). Cybernet."  # plain() escapes the version


def test_no_id_no_url_ends_at_last_element():
    s = apa7.reference(J(), links=False)
    assert s.endswith("101–118.") and "None" not in s


def test_ab_suffix_and_sorting():
    x = J(title="Alpha study"); y = J(title="Beta study")
    assert apa7.in_text(y, context=[x, y]) == "(Rivera & Okafor, 2021b)"
    lst = apa7.reference_list([y, x], links=False)
    assert "(2021a). Alpha" in lst[0] and "(2021b). Beta" in lst[1]


def test_multi_cite_and_locators():
    al = J(author=[A("Alpha", "A")], issued=D(2020)); ze = J(author=[A("Zeta", "Z")], issued=D(2018))
    assert apa7.in_text([ze, al], ns=[3, 1]) == "(Alpha, 2020; Zeta, 2018) [1, 3]"
    assert apa7.in_text(al, ns=[1, 2, 2]) == "(Alpha, 2020) [1, 2]"


def test_hostile_escaping():
    r = J(author=[{"literal": "[x](javascript:alert(1))"}], title="a *b* _c_ <script>\nx")
    s = apa7.reference(r, links=False)
    assert "[x](" not in s and "\n" not in s and "<script>" not in s and "&lt;script&gt;" in s and "\\*b\\*" in s
    assert "javascript" in s  # shown as inert escaped text


@pytest.mark.parametrize("url", ["data:text/html,x", "http://10.0.0.1/", "http://example.org/", "https://u:p@example.org/",
                                 "https://example.org:8443/", "https://example.org/a\tb", "javascript:alert(1)",
                                 "https://10.0.0.1/"])
def test_source_url_rejected(url):
    assert apa7.safe_source_url(url) is None
    s, codes = apa7.reference_ex({"type": "webpage", "title": "T", "URL": url})
    assert "](" not in s and "R-CIT-BADURL" in codes


def test_safe_source_url_accepts_and_encodes():
    assert apa7.safe_source_url("https://example.org/a_(b)") == "https://example.org/a_%28b%29"


def test_safe_link_hosts_and_encoding():
    assert apa7.safe_link("doi", "10.1234/a%b") == "https://doi.org/10.1234/a%25b"
    assert apa7.safe_link("pmcid", "PMC1").startswith("https://pmc.ncbi.nlm.nih.gov/")
    assert apa7.safe_link("bogus", "x") is None


def test_report_number_cap():
    s, codes = apa7.reference_ex({"type": "report", "title": "R", "number": "9" * 61})
    assert "R-CIT-NUMLEN" in codes and "Report No." not in s
    assert "Report No. 42" in apa7.reference({"type": "report", "title": "R", "number": "42"})


def test_characterise_and_caption():
    assert apa7.characterise({"type": "article-newspaper", "container_title": "Ledger", "issued": D(2026, 10, 30)}) \
        == "news article · Ledger · 2026, October 30"
    assert apa7.characterise({"title": "x"}) == "Source · n.d."
    assert apa7.HOUSE_CAPTION == "APA 7 (VERA-DAR house variant: identifiers appended)"


def test_style_flag(tmp_path):
    assert apa7.citation_style() == "apa7"
    p = tmp_path / "s.json"; p.write_text(json.dumps({"citation_style": "legacy"}))
    assert apa7.citation_style(p) == "legacy"
    assert apa7.citation_style(tmp_path / "missing.json") == "apa7"


def test_preprint_has_no_label():
    r = J(type="preprint", title="Sleep", container_title="", arxiv="2401.01234", issued=D(2024), author=[A("Rivera", "L")])
    assert apa7.reference(r, links=False) == "Rivera, L. (2024). *Sleep*. arXiv. arXiv:2401.01234"
    assert "[Preprint]" not in apa7.reference(r)


def _visible(md):
    import re
    from markdown_it import MarkdownIt
    from vera.ui_safety import safe_markdown
    html = MarkdownIt("commonmark").render(safe_markdown(md))
    import html as _h
    return _h.unescape(re.sub(r"<[^>]+>", "", html))


def test_escapes_render_without_visible_backslashes():
    r = {"type": "software", "title": "Maple v2.1", "author": [{"literal": "Maplesoft"}], "issued": D(2019),
         "version": "2.1.0", "publisher": "Cybernet", "DOI": "10.1234/jm.2021.014"}
    vis = _visible(apa7.reference(r))
    assert "\\" not in vis and "(Version 2.1.0)" in vis and "Maple v2.1" in vis
    vis2 = _visible(apa7.reference(J(DOI="10.1234/jm.2021.014")))
    assert "\\" not in vis2 and "101–118" in vis2


def test_angle_brackets_render_visibly_once():
    vis = _visible(apa7.reference(J(title="a<b>c")))
    assert "a<b>c" in vis and "\\" not in vis
