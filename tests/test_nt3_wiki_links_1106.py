"""NT-3 tests (fakes only; no network, no DB). Run from repo/ after applying nt3.patch: python -m pytest <this file>."""
from vera import lane_evidence as le
from vera import wiki_ref_portal as wp


def setup_function(_):
    wp._LINK_SINK.clear()


def test_portal_search_no_longer_records_links(monkeypatch):
    # WIKI-REDESIGN-1107 (owner 2026-11-07): no "via Wikipedia citations" reading list; followed sources go to the
    # normal lanes as candidates. The record/take API stays for lane_evidence.merge_wiki_links.
    monkeypatch.setattr(wp, "discover", lambda *a, **k: ([], ["https://apnews.com/x", "https://npr.org/y"], {}))
    assert wp.make_portal_search({})("midterms?", 5, 3.0) == []
    assert wp.take_links("  Midterms? ") == []
    wp.record_links("midterms?", ["https://apnews.com/x"])
    assert wp.take_links("  Midterms? ") == ["https://apnews.com/x"] and wp.take_links("midterms?") == []


def test_sink_bounded():
    for i in range(wp._LINK_SINK_MAX + 10):
        wp.record_links(f"q{i}", ["https://a.com/1"])
    assert len(wp._LINK_SINK) <= wp._LINK_SINK_MAX


def test_items_link_only_and_credit():
    items = le.wiki_link_items(["https://www.apnews.com/a", "https://www.apnews.com/a", "ftp://x/y", ""])
    assert len(items) == 1
    it = items[0]
    assert it["read"] is False and it["lane"] == "reporting" and it["outlet"] == "apnews.com"
    assert it["credit"] == "via Wikipedia citations" and it["published"] == "date unknown"
    assert "wiki" not in (it["title"] + it["outlet"]).lower() and "wikipedia.org" not in it["url"]


def test_excluded_outlet_dropped(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda u, cfg=None: "exclude" if "bad.com" in u else "links_only")
    assert [i["url"] for i in le.wiki_link_items(["https://bad.com/a", "https://ok.com/a"])] == ["https://ok.com/a"]


def test_allow_outlet_still_never_read(monkeypatch):
    monkeypatch.setattr(le.reporting_lane, "outlet_verdict", lambda u, cfg=None: "allow")
    assert all(i["read"] is False for i in le.wiki_link_items(["https://apnews.com/a"]))


def test_merge_into_lanes_and_public():
    wp.record_links("q", ["https://apnews.com/a", "https://dup.com/z"])
    base = le.LaneResult(links=[{"lane": "reporting", "url": "https://dup.com/z", "read": False}])
    out = le.merge_wiki_links(base, "q")
    assert [l["url"] for l in out.links] == ["https://dup.com/z", "https://apnews.com/a"]
    assert out.pages == [] and len(base.links) == 1  # no pages (not evidence); input untouched
    pub = out.public()
    assert pub["n_read"] == 0 and pub["links"][1]["credit"] == "via Wikipedia citations"


def test_merge_none_and_empty():
    assert le.merge_wiki_links(None, "nothing") is None
    r = le.LaneResult()
    assert le.merge_wiki_links(r, "nothing") is r


def test_merge_from_none_lanes():
    wp.record_links("q", ["https://apnews.com/a"])
    assert len(le.merge_wiki_links(None, "q").links) == 1


def test_with_web_merges_and_survives_no_web():
    from vera import inline_acquire as ia
    wp.record_links("latest news q", ["https://apnews.com/a"])  # LINKS-RC-1107: merged only for news-intent questions
    out = ia.with_web(None, None, "latest news q")
    assert out is not None and out.links[0]["read"] is False
    assert ia.with_web(None, None) is None  # old 2-arg call unchanged


def test_ui_renders_credit_link_only():
    from vera import lane_links_ui as L
    calls = []

    class S:
        def markdown(self, x): calls.append(x)
        def caption(self, x): calls.append(x)
    item = le.wiki_link_items(["https://apnews.com/a"])[0]
    L.render_lane_links(S(), {"contract": {"inline_acquisition": {"lanes": le.LaneResult(links=[item]).public()}}})
    assert any("link only" in c and "via Wikipedia citations" in c for c in calls)
    assert not any(c.endswith("- read") for c in calls)


def test_wiki_items_never_count_as_examined():
    # D3 / OD-1106-MIDTERMS-D1D4: link-only items carry read False and add no pages; the merged result reads 0 pages
    wp.record_links("q2", ["https://apnews.com/a"])
    out = le.merge_wiki_links(None, "q2")
    assert out.pages == [] and out.public()["n_read"] == 0
    assert all(l["read"] is False for l in out.links)


def test_record_links_concurrent_and_fail_quiet(monkeypatch):
    import threading
    errs = []

    def work(n):
        try:
            for i in range(200):
                wp.record_links(f"q{n}-{i}", ["https://a.com/1"])
                wp.take_links(f"q{n}-{i - 1}")
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(wp._LINK_SINK) <= wp._LINK_SINK_MAX

    class Boom(dict):
        def __setitem__(self, k, v): raise RuntimeError("x")
    monkeypatch.setattr(wp, "_LINK_SINK", Boom())
    wp.record_links("q", ["https://a.com/1"])  # must not raise
