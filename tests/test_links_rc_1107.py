"""LINKS-RC-1107: wiki-citation link relevance, archive de-dup, news-intent-only merge, lane headings, type labels.
Fakes only: no network, no DB."""
from vera import lane_evidence as le
from vera import references_ui as ru
from vera import wiki_ref_portal as wp

CFG = {"enabled": True, "api_url": "https://en.wikipedia.org/w/api.php", "allowed_hosts": ["en.wikipedia.org"],
       "user_agent": "t", "max_articles": 3, "max_refs": 20, "max_links": 10, "timeout_s": 3,
       "budget": {"max_requests_per_question": 4, "max_response_bytes": 4000000},
       "query": {"max_terms": 8, "max_query_chars": 100, "strip_terms": ["what", "is"]},
       "rank": {"stem_len": 5, "use_citing_sentence": True, "max_context_chars": 400}, "skip_disambiguation": True}


def _doc(text):
    return {"batchcomplete": True, "query": {"pages": [{"pageid": 1, "index": 1, "revisions": [
        {"slots": {"main": {"content": text}}}]}]}}


class _R:
    status_code, headers = 200, {"Content-Type": "application/json"}

    def __init__(self, d):
        import json
        self._b = json.dumps(d).encode()

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


def _discover(text, q="What is nuclear stellar genesis?"):
    get = lambda *a, **k: _R(_doc(text))  # noqa: E731
    return wp.discover(q, CFG, http_get=get, url_check=lambda u: True)[1]


def test_offtopic_article_links_dropped():
    txt = ("Bingley is a town in Yorkshire.<ref>https://www.nytimes.com/2001/a.html</ref> "
           "The church was rebuilt.<ref>https://bingley.church/history</ref>")
    assert _discover(txt) == []


def test_relevant_citing_sentence_keeps_link():
    txt = ("Stellar nucleosynthesis builds nuclear species in stars.<ref>https://astro.example.org/nuc</ref> "
           "Unrelated tower.<ref>https://bingley.church/x</ref>")
    assert _discover(txt) == ["https://astro.example.org/nuc"]


def test_template_outside_ref_has_no_score_so_no_link():
    assert _discover("Text. {{cite web|url=https://astro.example.org/nuc|title=x}}") == []


def test_min_link_score_invalid_falls_back():
    for bad in (0, -1, 2, "x", None):
        assert wp.link_min_score({"min_link_score": bad}) == wp.DEFAULT_LINK_MIN_SCORE
    assert wp.link_min_score({"min_link_score": 0.25}) == 0.25


def test_archive_urls_unwrapped_and_deduped(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda u, cfg=None: "links_only")
    items = le.wiki_link_items([
        "https://web.archive.org/web/20200101000000/https://example.com/a",
        "https://web.archive.org/web/20210202000000/https://example.com/a/",
        "https://example.com/a", "https://web.archive.org/web/2020/https://example.com/b"])
    assert [i["outlet"] for i in items] == ["example.com", "example.com"]
    assert len(items) == 2


def test_archive_of_excluded_outlet_dropped(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda u, cfg=None: "exclude" if le.reporting_lane._domain(u) == "bad.com" else "links_only")
    assert le.wiki_link_items(["https://web.archive.org/web/2020/https://bad.com/a"]) == []


def test_seen_dedupes_across_archive_form(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda u, cfg=None: "links_only")
    assert le.wiki_link_items(["https://web.archive.org/web/2020/https://example.com/a"], seen={"https://example.com/a"}) == []


def test_with_web_merges_wiki_only_on_news_intent(monkeypatch):
    from vera import inline_acquire as ia
    wp._LINK_SINK.clear()
    wp.record_links("What is nuclear stellar genesis?", ["https://example.com/a"])
    assert ia.with_web(None, None, "What is nuclear stellar genesis?") is None
    wp.record_links("latest news on the election", ["https://example.com/a"])
    out = ia.with_web(None, None, "latest news on the election")
    assert out is not None and [l["url"] for l in out.links] == ["https://example.com/a"]


def test_ui_headings_split_by_lane():
    from vera import lane_links_ui as L

    class St:
        def __init__(self):
            self.out = []
        markdown = lambda self, s: self.out.append(s)  # noqa: E731
        caption = markdown
    st = St()
    L.render_lane_links(st, {"contract": {"inline_acquisition": {"lanes": {"links": [
        {"lane": "reporting", "title": "N", "url": "https://n.example.org/a"},
        {"lane": "web", "title": "W", "url": "https://w.example.org/a"}]}}}})
    txt = "\n".join(st.out)
    assert txt.index("Current reporting") < txt.index("[N]") < txt.index("Related pages") < txt.index("[W]")


def test_arxiv_record_with_journal_doi_not_preprint():
    base = {"provider": "arxiv", "url": "https://arxiv.org/abs/2301.00001", "title": "T"}
    assert ru.csl_from_source({**base, "doi": "10.1016/j.adt.2023.101635"})["type"] == "article-journal"
    assert ru.csl_from_source({**base, "doi": "10.1007/978-981-15-8818-1_115-2"})["type"] == "chapter"
    assert ru.csl_from_source({**base, "doi": "10.48550/arxiv.2301.00001"})["type"] == "preprint"
    assert ru.csl_from_source(base)["type"] == "preprint"
    assert ru.csl_from_source({**base, "source_type": "preprint", "doi": "10.1016/x"})["type"] == "preprint"


import pytest  # noqa: E402


@pytest.mark.parametrize("u", [
    "https://web.archive.org/web/2020/bad.com/a",
    "https://web.archive.org/web/2020*/https://bad.com/a",
    "https://web.archive.org/web/20200101id_/https%3A%2F%2Fbad.com%2Fa",
    "https://web.archive.org:443/web/2020/https://bad.com:443/a",
    "https://archive.org/web/2020/https://bad.com/a",
    "https://wayback.archive.org/web/2020/https://bad.com/a",
    "https://web.archive.org/web/2020/https://web.archive.org/web/2019/https://bad.com/a"])
def test_archive_forms_see_real_host(monkeypatch, u):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict",
                        lambda x, cfg=None: "exclude" if le.reporting_lane._domain(x) == "bad.com" else "links_only")
    assert le.wiki_link_items([u]) == []
    assert le._unwrap_archive(u) is not None and "bad.com" in le._unwrap_archive(u)


@pytest.mark.parametrize("u", ["https://archive.ph/AbCd1", "https://archive.today/x", "https://archive.is/x",
                               "https://web.archive.org/save/https://x.com/a", "https://web.archive.org/web/2020/not a url",
                               "https://web.archive.org/web/2020/ftp://x.com/a"])
def test_unrecoverable_archive_dropped(monkeypatch, u):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda x, cfg=None: "links_only")
    assert le.wiki_link_items([u]) == []


def test_plain_url_unchanged_by_unwrap():
    assert le._unwrap_archive("https://example.com/a") == "https://example.com/a"


@pytest.mark.parametrize("doi", ["10.31235/osf.io/abc", "10.31223/X5AB", "10.36227/techrxiv.1", "10.22541/au.1"])
def test_more_preprint_servers(doi):
    assert ru.csl_from_source({"provider": "arxiv", "doi": doi})["type"] == "preprint"


def test_book_and_chapter_dois():
    b = {"provider": "arxiv"}
    assert ru.csl_from_source({**b, "doi": "10.1016/B978-0-12-381234-5.00001-2"})["type"] == "chapter"
    assert ru.csl_from_source({**b, "doi": "10.1007/978-3-030-12345-6"})["type"] == "book"
    assert ru.csl_from_source({**b, "doi": "10.1007/978-981-15-8818-1_115-2"})["type"] == "chapter"
    assert ru.csl_from_source({**b, "doi": "10.1016/j.adt.2023.101635"})["type"] == "article-journal"


@pytest.mark.parametrize("u", [
    "https://web.archive.org/web/2020/https://bad.com\\@example.com/a",
    "https://web.archive.org/web/2020/https://bad.com%5C@example.com/a",
    "https://web.archive.org/web/2020/https://169.254.169.254/latest",
    "https://web.archive.org/web/2020/https://127.0.0.1/a",
    "https://web.archive.org/web/2020/https://localhost/a"])
def test_unsafe_inner_url_dropped(monkeypatch, u):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda x, cfg=None: "links_only")
    assert le.wiki_link_items([u]) == []


def test_mixed_list_drops_mirror_keeps_plain(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda x, cfg=None: "links_only")
    out = le.wiki_link_items(["https://archive.ph/x", "https://example.com/a"])
    assert [i["url"] for i in out] == ["https://example.com/a"]
