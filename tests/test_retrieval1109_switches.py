"""RETRIEVAL-1109 H-2 / H-3: the shipped switch states and both states of each (on/off), offline fakes only.
H-2: GOV-PD mode on; official lane on for FEC and Census Bureau ONLY (fetch_entities). H-3: reference portal on with the
wikipedia_title adapter only. SYNTHETIC fixtures (DOIs 10.9999/..., invented sentences). No socket, DB or model."""
from __future__ import annotations

import copy
import json
import pathlib
from html import escape
from urllib.parse import urlencode

import pytest

from vera import gov_public_domain as g
from vera import official_lane as ol
from vera import reference_portal as rp
from vera import subquery

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cfg(name):
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ H-2 GOV-PD
def test_gov_pd_ships_on_and_off_is_inert():
    c = _cfg("us_federal_pd.json")
    assert c["mode"] == "on" and g.mode(c) == "on"
    off = {**c, "mode": "off"}
    assert g.mode(off) == "off"
    assert g.gate_allow({"url": "https://www.fec.gov/x", "text": "plain"}, off) is None
    cand = {"url": "https://www.fec.gov/x"}
    g.annotate_candidate(cand, "plain text", "", off)
    assert "gov_pd" not in cand  # off: nothing computed, no decision


# ------------------------------------------------------------------------------------------------ H-2 official lane
class Stub:
    def __init__(self):
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return {"text": "Official page on midterm elections.", "url": url, "content_type": "text/html"}


@pytest.fixture(autouse=True)
def _reset():
    ol._LAST_FETCH.clear()
    rp.reset_for_tests()
    yield
    ol._LAST_FETCH.clear()
    rp.reset_for_tests()


def test_official_lane_ships_on_fec_census_only_site_search_off():
    c = ol.load_config()
    assert c["enabled"] is True and c["site_search"] is False and c["fetch_entities"] == ["FEC", "Census Bureau"]
    allow = {r["entity"] for r in subquery.load_entity_registry() if r.get("terms_verdict") == "allow"}
    assert {"FEC", "Census Bureau"} <= allow and len(allow) > 2  # the narrowing is what keeps the others unread


def test_midterms_fixture_reads_fec_and_census_only_others_links_only():
    f = Stub()
    lane = ol.OfficialLane(ol.load_config(), registry=subquery.load_entity_registry(), fetch=f)
    out = lane.run(["FEC", "Census Bureau", "Kafka", "Slack", "Google Cloud", "AWS"])
    by = {lk["entity"]: lk for lk in out["links"]}
    assert set(f.calls) <= {"https://www.fec.gov/", "https://www.census.gov/topics/public-sector/voting"}
    assert by["FEC"]["read"] is True and by["Census Bureau"]["read"] is True
    for n in ("Kafka", "Slack", "Google Cloud", "AWS"):
        assert by[n]["read"] is False and by[n]["reason"]
    assert {p.entity for p in out["pages"]} == {"FEC", "Census Bureau"}


def test_official_lane_off_state_reads_nothing():
    f = Stub()
    out = ol.OfficialLane({**ol.load_config(), "enabled": False}, registry=subquery.load_entity_registry(),
                          fetch=f).run(["FEC", "Census Bureau"])
    assert out == {"links": [], "pages": []} and f.calls == []


def test_fetch_entities_absent_means_no_narrowing_and_odd_value_reads_nothing():
    reg = subquery.load_entity_registry()
    f = Stub()
    ol.OfficialLane({"enabled": True}, registry=reg, fetch=f).run(["Kafka"])
    assert f.calls == ["https://kafka.apache.org/"]
    for bad in ("FEC", None, {"FEC": 1}):
        f2 = Stub()
        out = ol.OfficialLane({"enabled": True, "fetch_entities": bad}, registry=reg, fetch=f2).run(["FEC"])
        assert f2.calls == [] and out["pages"] == []


# ------------------------------------------------------------------------------------------------ H-3 reference portal
WIKI = "https://en.wikipedia.org/wiki/Hadrons"
WIKI_SING = "https://en.wikipedia.org/wiki/Hadron"
Q = "What are hadrons?"


def coins(atitle, *ids):
    pairs = [("ctx_ver", "Z39.88-2004"), ("rft.atitle", atitle)] + [("rft_id", i) for i in ids]
    return f'<span class="Z3988" title="{escape(urlencode(pairs))}"></span>'


PAGE = ("<html><body><p>Hadron prose that must never be read. SYNTH-PROSE-CANARY</p>"
        + coins("Hadron structure and hadrons", "info:doi/10.9999/synth.hadron.1")
        + coins("Weather report", "info:doi/10.9999/synth.weather.1")
        + '<a href="https://www.example-news.org/story">news link</a>'
        + '<a href="/wiki/Quark">nav</a></body></html>')


class Resp:
    def __init__(self, status=200, body=b""):
        self.status_code, self.headers = status, {"Content-Type": "text/html; charset=utf-8"}
        self._b = body.encode() if isinstance(body, str) else body

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]

    def close(self):
        pass


class Net:
    def __init__(self, pages, robots=True):
        self.pages, self.calls, self.portals, self.robots_seen, self.robots = pages, [], [], [], robots

    def get(self, url, **kw):
        self.calls.append(url)
        return self.pages.get(url) or Resp(404)

    def robots_ok(self, url, ua):
        self.robots_seen.append(url)
        return self.robots

    def deps(self):
        return rp.Deps(http_get=self.get, url_check=lambda u: (True, ""), robots_ok=self.robots_ok,
                       no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True, crawl_ok=lambda u, ua: True,
                       limited=lambda portal, fn: (self.portals.append(portal), fn)[1],
                       openalex_lookup=lambda s, n, t: [])


def _run(c, net, base=None):
    return rp.collect(Q, base=base, ddg_raw=[{"url": "https://other.example.org/p", "title": "t"}], cfg=c,
                      deps=net.deps(), hosts=rp.load_hosts(c), clock=lambda: 0.0)


def test_portal_ships_on_with_wikipedia_title_only_and_host_row_permits():
    c = rp.validate(_cfg("reference_portals.json"))
    assert c["mode"] == "on"
    assert [k for k, v in c["adapters"].items() if v.get("enabled") is True] == ["wikipedia_title"]
    row = next(r for r in _cfg("portal_hosts.json")["entries"] if r["host"] == "en.wikipedia.org")
    assert row["status"] == "active" and row["terms"] == "permit" and row["rate_portal"] == "wikipedia"
    old = _cfg("wiki_ref_portal.json")
    assert old["enabled"] is False  # the older DDG site: route stays off


def test_hadrons_fixture_resolves_title_and_keeps_only_coins_refs_with_portal_and_robots():
    c = rp.validate(_cfg("reference_portals.json"))
    net = Net({WIKI: Resp(404), WIKI_SING: Resp(200, PAGE)})
    cands, counts = _run(c, net)
    assert net.calls == [WIKI, WIKI_SING]  # as-is title first, then the naive singular "Hadron"
    assert net.portals and set(net.portals) == {"wikipedia"}
    assert WIKI in net.robots_seen or WIKI_SING in net.robots_seen
    assert not any("other.example.org" in u for u in net.calls)  # generic_page is off: no DDG result page read
    dois = {str(x.get("doi")).lower() for x in cands}
    assert "10.9999/synth.hadron.1" in dois and counts["candidates"] == len(cands) >= 1
    assert "SYNTH-PROSE-CANARY" not in json.dumps(cands, default=str) + json.dumps(counts)
    assert not any("example-news.org" in json.dumps(x, default=str) for x in cands)  # news href is not a candidate


def test_hadrons_robots_disallow_means_no_request():
    c = rp.validate(_cfg("reference_portals.json"))
    net = Net({WIKI: Resp(200, PAGE)}, robots=False)
    cands, _counts = _run(c, net)
    assert net.calls == [] and cands == []


def test_portal_off_state_sends_nothing_and_wikipedia_adapter_off_reads_no_wiki():
    c = copy.deepcopy(rp.validate(_cfg("reference_portals.json")))
    net = Net({WIKI: Resp(200, PAGE)})
    c["mode"] = "off"
    assert _run(c, net)[0] == [] and net.calls == []
    c["mode"] = "on"
    c["adapters"]["wikipedia_title"]["enabled"] = False
    assert _run(c, net)[0] == [] and net.calls == []


def test_rate_portal_wikipedia_is_one_per_second():
    p = json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))["portals"]["wikipedia"] \
        if "portals" in json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8")) else \
        json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))["wikipedia"]
    assert p["rate_per_s"] == 1.0 and p["burst"] == 1 and p["max_concurrency"] == 1 and p["hosts"] == ["en.wikipedia.org"]
