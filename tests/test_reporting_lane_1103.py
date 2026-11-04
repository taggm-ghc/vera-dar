"""NONASK-1103 S7: offline tests for vera/reporting_lane.py and vera/query_terms.py. No network."""
import json
from pathlib import Path

from vera import reporting_lane as rl
from vera.query_terms import build_query, extract_terms

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "Can John Smith at john.smith@example.com or +1 (415) 555-0199 account AB1234567 see latest Kubernetes outage reports?"


def cfg(**kw):
    c = rl.load_config(ROOT / "config" / "reporting_lane.json")
    c.update(enabled=True, **kw)
    return c


class Fake:
    def __init__(self, *responses):
        self.r = list(responses)
        self.urls, self.headers = [], []

    def __call__(self, url, headers, timeout):
        self.urls.append(url)
        self.headers.append(headers)
        r = self.r.pop(0) if self.r else (429, {}, b"")
        if isinstance(r, Exception):
            raise r
        return r


GD = json.dumps({"articles": [
    {"url": "https://www.bbc.com/a", "title": "Old", "seendate": "20261001T000000Z", "domain": "bbc.com"},
    {"url": "https://x.org/b", "title": "New", "seendate": "20261102T101500Z"},
    {"url": "https://y.org/c"}, {"title": "no url"}, "junk", {"url": "javascript:alert(1)"}]}).encode()
HN = json.dumps({"hits": [{"title": "H", "url": "https://z.com/p", "created_at_i": 1793000000, "objectID": "9"},
                          {"title": "Ask", "objectID": "10", "created_at": "2026-10-01T00:00:00Z"}]}).encode()


def lane(fake, **kw):
    sl = []
    L = rl.ReportingLane(cfg(**kw), http_get=fake, sleep=sl.append, rand=lambda: 0.5)
    return L, sl


def test_flag_ships_off_and_no_call():
    assert rl.load_config(ROOT / "config" / "reporting_lane.json")["enabled"] is False
    f = Fake()
    L = rl.ReportingLane(rl.load_config(ROOT / "config" / "reporting_lane.json"), http_get=f)
    assert L.search("kubernetes outage")["items"] == [] and f.urls == []


def test_user_sentence_and_pii_never_in_request():
    f = Fake((200, {}, GD))
    L, _ = lane(f)
    L.search(SENTENCE, known_entities=["Kubernetes"])
    assert len(f.urls) == 1
    u = f.urls[0].lower()
    for bad in ("john", "smith", "example.com", "555", "ab1234567", "can%20", "see%20latest", "outage%20reports%3f"):
        assert bad not in u
    assert "kubernetes" in u and SENTENCE.lower() not in u


def test_terms_filter():
    t = extract_terms(SENTENCE, ["Kubernetes"])
    assert "Kubernetes" in t and "outage" in t
    assert not {"John", "Smith"} & set(t) and not any("@" in x or x.startswith("AB1") for x in t)
    assert extract_terms("Anthropic funding", ["Anthropic"]) == ["Anthropic", "funding"]
    assert extract_terms("Contact me at a@b.co") == ["Contact"] or "a@b.co" not in extract_terms("Contact me at a@b.co")
    assert build_query(['a"b', "c-d"]) == "ab c-d"


def test_gdelt_parse_sort_attribution_defensive():
    f = Fake((200, {}, GD))
    L, _ = lane(f)
    r = L.search("kubernetes outage")
    assert [i["title"] for i in r["items"]][:2] == ["New", "Old"]
    assert r["items"][-1]["published"] is None and r["items"][-1]["url"] == "https://y.org/c"
    assert r["items"][0]["outlet"] == "x.org" and r["items"][0]["lane"] == "reporting"
    assert r["attribution"]["url"] == "https://www.gdeltproject.org/"
    assert "mode=artlist" in f.urls[0] and "format=json" in f.urls[0]
    assert rl.parse_gdelt(b"not json") == [] and rl.parse_gdelt({"articles": None}) == []


def test_user_agent_sent():
    f = Fake((200, {}, GD))
    lane(f)[0].search("kubernetes")
    assert "VERA-DAR" in f.headers[0]["User-Agent"]


def test_429_backoff_then_success():
    f = Fake((429, {}, b""), (429, {}, b""), (200, {}, GD))
    L, sl = lane(f)
    assert L.search("kubernetes")["items"]
    assert sl == [60, 120] and len(f.urls) == 3


def test_circuit_open_after_3_retries_then_hn_fallback_and_skip():
    f = Fake(*[(429, {}, b"")] * 4, (200, {}, HN))
    t = [0.0]
    sl = []
    L = rl.ReportingLane(cfg(), http_get=f, sleep=sl.append, clock=lambda: t[0], wall=lambda: 1_793_500_000, rand=lambda: 0.5)
    r = L.search("kubernetes")
    assert sl == [60, 120, 240] and r["attribution"] is None
    assert [i["outlet"] for i in r["items"]][0] == "z.com" and "hn.algolia.com" in f.urls[-1]
    assert "tags=story" in f.urls[-1] and "numericFilters=created_at_i%3E" in f.urls[-1]
    assert not L.available("gdelt")
    n = len(f.urls)
    f.r.append((200, {}, HN))
    L.search("kubernetes")
    assert not any("gdeltproject" in u for u in f.urls[n:])  # circuit holds
    t[0] = 901.0
    assert L.available("gdelt")


def test_timeout_and_5xx_retry_and_retry_after_and_jitter():
    f = Fake(TimeoutError(), (503, {}, b""), (429, {"Retry-After": "300"}, b""), (200, {}, GD))
    L, sl = lane(f)
    L._rand = lambda: 1.0  # +20%
    assert L.search("kubernetes")["items"]
    assert sl == [72.0, 144.0, 360.0]


def test_both_down_returns_empty():
    f = Fake(*[(429, {}, b"")] * 8)
    L, _ = lane(f)
    r = L.search("kubernetes")
    assert r["items"] == [] and len(f.urls) == 8


def test_hn_item_link_for_text_posts():
    items = rl.parse_hn(HN)
    assert items[1]["url"] == rl.HN_ITEM + "10" and items[0]["published"]


def test_rate_config_loads_with_new_portals():
    from vera import rate_limit
    c = rate_limit.load_config()
    assert "gdelt_doc" in c["portals"] and "hn_algolia" in c["portals"]
