"""OD-1106-HITL H5 (a) (owner 2026-11-06): official lane on with FEC/Census/EAC rows; VOA allow for VOA-original items only;
AP, Reuters, NPR, PBS link-only; The Conversation and ProPublica excluded; congress.gov and state election sites absent.
Offline: reads config only; the lane fetch is a fake."""
import json
from pathlib import Path

from vera import lane_evidence as le, official_lane as ol, reporting_lane as rl

ROOT = Path(__file__).resolve().parent.parent
REG = json.loads((ROOT / "config" / "entity_registry.json").read_text(encoding="utf-8"))["entities"]
RLC = json.loads((ROOT / "config" / "reporting_lane.json").read_text(encoding="utf-8"))
H5 = {"FEC": "fec.gov", "Census Bureau": "census.gov", "EAC": "eac.gov"}


def _rows():
    return {r["entity"]: r for r in REG if r.get("state") == "active"}


def test_official_lane_flag_stays_off_pending_checklist():
    cfg = json.loads((ROOT / "config" / "subquery.json").read_text(encoding="utf-8"))["official_lane"]
    assert cfg["enabled"] is False and cfg["site_search"] is False


def test_fec_census_rows_allow_listed_static_paths_only():
    rows = _rows()
    for name, host in (("FEC", "fec.gov"), ("Census Bureau", "census.gov")):
        r = rows[name]
        assert ol.verdict(r) == "allow" and r["terms_evidence"] and r["volatile"] is False
        assert r["domains"] == ["www." + host]
        assert r["verification"]["status"] != "verified" and "OD-1106-HITL" in r["verification"]["decision"]
        for u in r["docs_entry_urls"]:
            assert ol.page_ok(u, r)
        for bad in ("/data/", "/api/", "/search/", "/login", "/data/committee/C001/", "/search-results.html",
                    "/quickfacts/x"):
            assert not ol.page_ok(f"https://www.{host}{bad}", r), (name, bad)
        assert not ol.page_ok("https://evil.example/", r)
    fec = rows["FEC"]
    for ok in ("/introduction-campaign-finance/x/", "/help-candidates-and-committees/a/", "/data/elections/house/",
               "/data/raising-bythenumbers/"):
        assert ol.page_ok("https://www.fec.gov" + ok, fec), ok
    assert not ol.page_ok("https://www.fec.gov/data/elections/house/?year=2026", fec)  # never a query string
    assert not ol.page_ok("https://www.fec.gov/data/receipts/individual-contributions/", fec)  # contributor data
    cen = rows["Census Bureau"]
    for ok in ("/topics/public-sector/voting", "/library/stories/2026/x.html", "/library/publications/y.html",
               "/library/visualizations/z/", "/data/tables/time-series/demo/voting-and-registration/p20.html"):
        assert ol.page_ok("https://www.census.gov" + ok, cen), ok


def test_eac_row_present_but_disabled():
    from vera import subquery
    eac = next(r for r in REG if r["entity"] == "EAC")
    assert eac["state"] != "active" and eac["terms_unverified"] is True and eac["docs_entry_urls"] == []
    assert ol.verdict(eac) == "links_only"
    assert "EAC" not in {r["entity"] for r in subquery.load_entity_registry()}
    assert "EAC" not in _rows()


def test_fec_crawl_delay_10s_not_sent_early(monkeypatch):
    fec = _rows()["FEC"]
    assert fec["crawl_delay_s"] == 10
    now = [1000.0]
    monkeypatch.setattr(ol, "_clock", lambda: now[0])
    ol._LAST_FETCH.clear()
    calls = []
    row = dict(fec, docs_entry_urls=["https://www.fec.gov/introduction-campaign-finance/a/",
                                     "https://www.fec.gov/introduction-campaign-finance/b/"])
    lane = ol.OfficialLane({"enabled": True}, registry=[row], fetch=lambda u: calls.append(u) or {"text": "x", "url": u})
    out = lane.run(["FEC"])
    reasons = [l.get("reason") for l in out["links"]]
    assert "crawl_delay" in reasons
    first = [c for c in calls if "introduction" in c]
    assert len(first) == 1  # second page inside the 10 s window is link only
    now[0] += 10.5
    out2 = ol.OfficialLane({"enabled": True}, registry=[dict(row, docs_entry_urls=[row["docs_entry_urls"][1]])],
                           fetch=lambda u: calls.append(u) or {"text": "x", "url": u}).run(["FEC"])
    assert out2["links"][0].get("reason") != "crawl_delay"
    ol._LAST_FETCH.clear()


def test_congress_gov_and_state_sites_absent_from_registry():
    hosts = {d for r in REG for d in r.get("domains") or []}
    assert not any(h == "congress.gov" or h.endswith(".congress.gov") for h in hosts)
    assert not any(h.endswith(".gov") and h not in {"www.fec.gov", "www.census.gov", "www.eac.gov"} for h in hosts)


def _v(url):
    return rl.outlet_verdict(url, RLC)


def test_outlet_verdicts():
    assert _v("https://www.voanews.com/a/x/1.html") == "allow"
    assert _v("https://globalvoices.org/x") == "allow" and _v("https://en.wikinews.org/wiki/X") == "allow"
    for u in ("https://www.apnews.com/a", "https://www.reuters.com/a", "https://www.npr.org/a", "https://www.pbs.org/newshour/a"):
        assert _v(u) == "links_only", u
    assert _v("https://www.propublica.org/a") == "exclude" and _v("https://theconversation.com/a") == "exclude"
    assert _v("https://www.congress.gov/bill/1") == "links_only"  # unlisted: link only, never read
    assert _v("https://sos.state.example.gov/") == "links_only"


def test_exclude_drops_items_and_link_only_never_read():
    lane = rl.ReportingLane(RLC)
    items = [{"url": "https://www.propublica.org/a", "title": "t"}, {"url": "https://www.reuters.com/a", "title": "t"},
             {"url": "https://www.voanews.com/a/x/1.html", "title": "t"}]
    out = {it["url"]: it["read_ok"] for it in lane._apply_outlet_policy(items)}
    assert out == {"https://www.reuters.com/a": False, "https://www.voanews.com/a/x/1.html": True}


class _Rep:
    def __init__(self, url):
        self.cfg = RLC
        self.url = url

    def search(self, text, ents=()):
        return {"items": [{"title": "T", "url": self.url, "published": "2026-11-01", "outlet": "voanews.com",
                           "read_ok": True}], "attribution": None, "terms": []}


def _collect(monkeypatch, url, text):
    monkeypatch.setattr(rl, "lane_enabled", lambda cfg=None: True)
    monkeypatch.setattr(rl, "outlet_verdict", lambda u, cfg=None: "allow")  # hop check; the path/wire rules are real
    calls = []

    def fetch(u):
        calls.append(u)
        return {"text": text, "url": u, "content_type": "text/html"}
    res = le.collect("What is the latest news on the 2026 midterm elections?", reporting=_Rep(url), fetch=fetch)
    return res, calls


VOA = "https://www.voanews.com/a/midterms/7001.html"


def test_voa_original_with_byline_is_read(monkeypatch):
    res, calls = _collect(monkeypatch, VOA, "By VOA News. Voters across the country cast ballots.")
    assert calls and len(res.pages) == 1 and res.links[0]["read"] is True


def test_voa_no_byline_is_link_only(monkeypatch):
    res, calls = _collect(monkeypatch, VOA, "Voters across the country cast ballots.")
    assert res.pages == [] and res.links[0]["read"] is False


def test_voa_wire_text_is_link_only(monkeypatch):
    for t in ("By VOA News. WASHINGTON (AP) -- votes.", "By VOA News. Reuters contributed.",
              "By VOA News. AFP reported.", "By VOA News. Jane Doe contributed to this report."):
        res, _ = _collect(monkeypatch, VOA, t)
        assert res.pages == [] and res.links[0]["read"] is False, t


def test_voa_paths_other_than_article_never_fetched(monkeypatch):
    for u in ("https://www.voanews.com/p/5338.html", "https://www.voanews.com/z/1234", "https://www.voanews.com/tv/x",
              "https://www.voanews.com/radio/x", "https://www.voanews.com/embed/player/1.html",
              "https://www.voanews.com/a/x/1.html?x=1", "https://www.voanews.com/a/x/comments/1.html",
              "https://www.voanews.com/a/x/y/1.html"):
        res, calls = _collect(monkeypatch, u, "By VOA News. text")
        assert calls == [] and res.pages == [], u


def test_wikinews_paths_and_licence_attribution(monkeypatch):
    row = next(o for o in RLC["outlets"] if o["host"] == "en.wikinews.org")
    assert row["terms_verdict"] == "allow" and row["no_current_news"] is True
    assert "4.0" in row["licence_note"] and "2.5" in row["licence_note"]
    rules = le._outlet_read_rules("https://en.wikinews.org/wiki/X")
    for ok in ("https://en.wikinews.org/wiki/Some_story", "https://en.wikinews.org/wiki/Story_(2026)"):
        assert le._path_ok(ok, rules), ok
    for bad in ("https://en.wikinews.org/wiki/Special:Search", "https://en.wikinews.org/wiki/special:Search",
                "https://en.wikinews.org/wiki/talk:X", "https://en.wikinews.org/wiki/Talk:X",
                "https://en.wikinews.org/wiki/Special%3ASearch", "https://en.wikinews.org/wiki/Category:Politics",
                "https://en.wikinews.org/wiki/Template:X", "https://en.wikinews.org/wiki/User_talk:Bob",
                "https://en.wikinews.org/wiki/User:Bob", "https://en.wikinews.org/wiki/File:X.jpg",
                "https://en.wikinews.org/wiki/Story?action=raw", "https://en.wikinews.org/w/index.php",
                "https://en.wikinews.org/wiki/", "https://en.wikinews.org/"):
        assert not le._path_ok(bad, rules), bad
    assert "creativecommons.org/licenses/by/2.5" in row["licence_note"]
    from vera.official_lane import LanePage
    pg = LanePage("reporting", "", "T", "https://en.wikinews.org/wiki/Some_story", outlet="en.wikinews.org")
    assert "CC BY 4.0" in le.attribution_for(pg)["licence"] and "2.5" in le.attribution_for(pg)["licence"]


def test_malformed_rules_fail_closed(monkeypatch):
    for row in ({"path_prefixes": []}, {"path_regex": "("}, {"require_any_markers": []}):
        monkeypatch.setattr(rl, "outlet_row", lambda url, cfg=None, r=row: dict(r, host="x"))
        rules = le._outlet_read_rules("https://x.example/a")
        assert not le._path_ok("https://x.example/a", rules) or not le._text_ok("anything", rules)
    monkeypatch.setattr(rl, "outlet_row", lambda url, cfg=None: {"host": "x"})
    rules = le._outlet_read_rules("https://x.example/a")
    assert le._path_ok("https://x.example/a", rules) and le._text_ok("anything", rules)


def test_voa_wire_markers_whitespace_and_case_insensitive():
    rules = le._outlet_read_rules(VOA)
    assert "the associated press" in rules["wire"] and "ap photo" in rules["wire"]
    for t in ("By VOA News. Associated\nPress reported.", "By VOA News. Associated  Press reported.",
              "By VOA News. The\tASSOCIATED \n PRESS said.", "By VOA News. (AP  Photo/x)", "By VOA News. AP\nPhoto"):
        assert not le._text_ok(t, rules), t
    assert not le._text_ok("By\nVOA   News is missing nothing but wire: Reuters", rules)
    assert le._text_ok("By\nVOA  News. Our own reporting.", rules)  # byline match also whitespace-insensitive


def test_wiki_link_items_drop_wikipedia_and_wikimedia_hosts():
    urls = ["https://en.wikipedia.org/wiki/X", "https://de.wikipedia.org/wiki/Y", "https://commons.wikimedia.org/f",
            "https://www.wikidata.org/wiki/Q1", "https://upload.wikimedia.org/a.jpg", "http://en.wikipedia.org/x",
            "https://en.wikinews.org/wiki/Z", "https://ok.example/a"]
    got = [i["url"] for i in le.wiki_link_items(urls)]
    assert "https://ok.example/a" in got and "https://en.wikinews.org/wiki/Z" in got
    assert not any("wikipedia" in u or "wikimedia" in u or "wikidata" in u for u in got)
    assert {"wikipedia.org", "wikimedia.org"} <= set(RLC["wiki_link_drop_hosts"])


def test_path_regex_rejects_trailing_newline():
    rules = le._outlet_read_rules("https://en.wikinews.org/wiki/X")
    assert le._path_ok("https://en.wikinews.org/wiki/X", rules)
    assert not le._path_ok("https://en.wikinews.org/wiki/X\n", rules)
    assert not le._path_ok("https://www.voanews.com/a/x/1.html\n", le._outlet_read_rules(VOA))


class _Raw:
    """A source that bypasses ReportingLane.search and its outlet policy (every item tagged read_ok=True)."""
    def __init__(self, urls):
        self.cfg = RLC
        self.urls = urls

    def search(self, text, ents=()):
        return {"items": [{"title": "T", "url": u, "published": "2026-11-01", "outlet": "x", "read_ok": True}
                          for u in self.urls], "attribution": None, "terms": []}


def test_collect_drops_excluded_outlets_and_keeps_links_only_link_only(monkeypatch):
    monkeypatch.setattr(rl, "lane_enabled", lambda cfg=None: True)
    calls = []

    def fetch(u):
        calls.append(u)
        return {"text": "By VOA News. body", "url": u, "content_type": "text/html"}
    urls = ["https://www.propublica.org/article/x", "https://theconversation.com/y-1",
            "https://www.reuters.com/world/z", VOA]
    res = le.collect("What is the latest news on the 2026 midterm elections?", reporting=_Raw(urls), fetch=fetch)
    shown = {lk["url"]: lk["read"] for lk in res.links}
    assert "https://www.propublica.org/article/x" not in shown and "https://theconversation.com/y-1" not in shown
    assert shown["https://www.reuters.com/world/z"] is False  # links_only: link only even though tagged read_ok
    assert not any("propublica" in c or "theconversation" in c or "reuters" in c for c in calls)
