"""NEWS-LANE-1104 W2/W3/W6: lane_evidence routing, read_ok and the redirect hop predicate. Offline: no network."""
import pytest

from vera import lane_evidence as le, public_fetch, reporting_lane as rl, web_route

NEWS_Q = "What make the news about the USA 2026 midterm elections?"


class FakeReporting:
    def __init__(self, items=None):
        self.calls, self.items = [], items or []

    def search(self, text, ents=()):
        self.calls.append(text)
        return {"items": list(self.items), "attribution": None, "terms": []}


class HopFetch:
    """public_fetch.fetch-compatible stub: applies hop_gate to each hop BEFORE recording a 'request' to it."""

    def __init__(self, redirect_to=None, text="Global Voices report on the midterm elections and turnout."):
        self.requests, self.redirect_to, self.text = [], redirect_to, text

    def __call__(self, url, hop_gate=None, robots_cache=None):
        hops = [url] + ([self.redirect_to] if self.redirect_to else [])
        for h in hops:
            if hop_gate is not None and hop_gate(h):
                raise public_fetch.FetchRefused("hop")
            self.requests.append(h)
        return {"text": self.text, "url": hops[-1], "content_type": "text/html"}


@pytest.fixture
def rep_on(monkeypatch):
    monkeypatch.setattr(rl, "lane_enabled", lambda cfg=None: True)
    monkeypatch.setattr(le.official_lane, "lane_enabled", lambda cfg=None: False)


def _item(url, read_ok):
    return {"title": "T", "url": url, "published": "2026-11-03", "outlet": rl._domain(url), "read_ok": read_ok}


def test_w3_news_question_reaches_reporting_lane(rep_on):
    r = FakeReporting()
    le.collect(NEWS_Q, reporting=r, fetch=HopFetch())
    assert len(r.calls) >= 1


def test_w3_non_news_question_still_no_call(rep_on):
    r = FakeReporting()
    le.collect("Explain how backpropagation computes gradients", reporting=r, fetch=HopFetch())
    assert r.calls == []


def test_w2_read_ok_false_is_link_only_never_fetched(rep_on):
    f = HopFetch()
    res = le.collect(NEWS_Q, reporting=FakeReporting([_item("https://www.nytimes.com/a", False)]), fetch=f)
    assert f.requests == [] and res.pages == []
    assert [lk["url"] for lk in res.links] == ["https://www.nytimes.com/a"] and res.links[0]["read"] is False


def test_w2_read_ok_true_allow_outlet_is_fetched(rep_on):
    f = HopFetch()
    res = le.collect(NEWS_Q, reporting=FakeReporting([_item("https://globalvoices.org/2026/11/03/x/", True)]), fetch=f)
    assert f.requests == ["https://globalvoices.org/2026/11/03/x/"]
    assert res.links[0]["read"] is True and len(res.pages) == 1


@pytest.mark.parametrize("target", ["https://www.nytimes.com/2026/x.html", "https://fr.globalvoices.org/x",
                                    "https://169.254.169.254/latest"])
def test_w6_cross_host_redirect_refused_before_request(rep_on, target):
    f = HopFetch(redirect_to=target)
    res = le.collect(NEWS_Q, reporting=FakeReporting([_item("https://globalvoices.org/2026/11/03/x/", True)]), fetch=f)
    assert target not in f.requests and res.pages == [] and res.links[0]["read"] is False


def test_w6_same_host_redirect_allowed(rep_on):
    f = HopFetch(redirect_to="https://globalvoices.org/2026/11/03/x-final/")
    res = le.collect(NEWS_Q, reporting=FakeReporting([_item("https://globalvoices.org/2026/11/03/x/", True)]), fetch=f)
    assert len(res.pages) == 1


def test_w6_gated_fetch_refuses_start_url_without_calling_fetch():
    called = []
    with pytest.raises(public_fetch.FetchRefused):
        web_route.gated_fetch("https://www.nytimes.com/a", fetch=lambda u, **k: called.append(u),
                              hop_allowed=lambda u: False)
    assert called == []


def test_w6_predicate_error_fails_closed():
    seen = {}

    def fetch(url, hop_gate=None, robots_cache=None):
        seen["code"] = hop_gate("https://globalvoices.org/robots.txt")
        return {"text": "", "url": url}

    def boom(u):
        if u.endswith("robots.txt"):
            raise RuntimeError("x")
        return True
    web_route.gated_fetch("https://globalvoices.org/a", fetch=fetch, hop_allowed=boom)
    assert seen["code"] == "read_lane_hop_refused"


def test_reporting_hop_predicate():
    ok = le._reporting_hop("https://globalvoices.org/a", {"read_ok": True})
    assert ok("https://globalvoices.org/b") and not ok("https://www.nytimes.com/b") and not ok("https://evil.example/")
    # NEWS-LANE-1104 R1 (INTEGRATE-1104): an untagged item gets no exemption; every hop needs an allow outlet on the
    # start host (news.example is not an allow outlet), and an allow outlet stays readable untagged
    untagged = le._reporting_hop("https://news.example/a", {})
    assert not untagged("https://news.example/robots.txt") and not untagged("https://other.example/")
    gv = le._reporting_hop("https://globalvoices.org/a", {})
    assert gv("https://globalvoices.org/robots.txt") and not gv("https://www.nytimes.com/b")
