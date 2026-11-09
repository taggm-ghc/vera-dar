"""NEWS-LANE-1104: offline tests for the enabled reporting lane (outlet policy, spacing, API client). No network."""
import json
from pathlib import Path

import pytest

from vera import reporting_lane as rl

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "reporting_lane.json"


def shipped():
    return rl.load_config(CFG_PATH)


class Fake:
    def __init__(self, *responses):
        self.r = list(responses)
        self.urls = []

    def __call__(self, url, headers, timeout):
        self.urls.append(url)
        return self.r.pop(0) if self.r else (429, {}, b"")


ARTS = json.dumps({"articles": [
    {"url": "https://globalvoices.org/2026/11/01/a/", "title": "GV one", "seendate": "20261101T100000Z", "domain": "globalvoices.org"},
    {"url": "https://fr.globalvoices.org/2026/11/02/b/", "title": "GV two", "seendate": "20261102T100000Z"},
    {"url": "https://en.wikinews.org/wiki/Story", "title": "WN", "seendate": "20261103T100000Z"},
    {"url": "https://www.nytimes.com/2026/11/03/us/politics/x.html", "title": "NYT", "seendate": "20261103T110000Z"},
    {"url": "https://unknown-outlet.example/n", "title": "U", "seendate": "20261103T120000Z"},
    {"url": "https://notglobalvoices.org/evil", "title": "Lookalike", "seendate": "20261103T130000Z"},
]}).encode()


def test_shipped_config_enabled_gdelt_only_and_allowlist():
    c = shipped()
    assert c["enabled"] is True and rl.lane_enabled(c)
    assert c["fallback_order"] == ["gdelt"]  # HN Algolia excluded (no first-party terms)
    # OWNER DECISION 2026-11-05 ("News sources: yes, as far as permitted by each source's permissions and license
    # terms"): every outlet is searched and listed; only allow outlets are read (PERF-MISS-1105-FIX2)
    assert c["emit_link_only"] is True and c["outlet_query"] == "all" and c["unlisted_verdict"] == "links_only"
    for o in c["outlets"]:
        assert o["terms_verdict"] in rl.VERDICTS and o["terms_evidence"].startswith("work/")
    assert set(rl.allow_hosts(c)) == {"globalvoices.org", "en.wikinews.org", "voanews.com", "commondreams.org", "democracynow.org", "votebeat.org",
                                  "chalkbeat.org", "kffhealthnews.org"}  # RETRIEVAL-1109: ND rows answer-only after H-1; voanews: OD-1106-HITL H5, VOA-original only


def test_outlet_verdicts_exact_subdomain_lookalike_unlisted():
    c = shipped()
    v = lambda u: rl.outlet_verdict(u, c)  # noqa: E731
    assert v("https://globalvoices.org/x") == "allow" and v("https://WWW.GlobalVoices.org/x") == "allow"
    assert v("https://fr.globalvoices.org/x") == "allow"
    assert v("https://notglobalvoices.org/x") == "links_only"
    assert v("https://globalvoices.org.evil.example/x") == "links_only"
    assert v("https://www.nytimes.com/a") == "links_only" and v("https://edition.cnn.com/a") == "links_only"
    assert v("https://unknown.example/") == "links_only"
    assert v("javascript:alert(1)") == "exclude" and v("") == "exclude"


def test_fail_closed_allow_without_evidence_and_unknown_verdict():
    c = dict(shipped(), outlets=[{"host": "a.example", "terms_verdict": "allow"},
                                 {"host": "b.example", "terms_verdict": "maybe", "terms_evidence": "x"},
                                 {"host": "c.example", "terms_verdict": "exclude", "terms_evidence": "x"}],
             unlisted_verdict="allow")
    assert rl.outlet_verdict("https://a.example/", c) == "links_only"
    assert rl.outlet_verdict("https://b.example/", c) == "links_only"
    assert rl.outlet_verdict("https://c.example/", c) == "exclude"
    assert rl.outlet_verdict("https://d.example/", c) == "links_only"  # unlisted can never become allow


def test_allowlist_query_and_only_allow_items_emitted():
    f = Fake((200, {}, ARTS))  # the pre-2026-11-05 allowlist mode (still supported by config)
    L = rl.ReportingLane(dict(shipped(), emit_link_only=False, outlet_query="allowlist"), http_get=f, sleep=lambda s: None)
    r = L.search("USA 2026 midterm elections news")
    q = f.urls[0]
    assert "domain%3Aglobalvoices.org" in q and "domain%3Aen.wikinews.org" in q and "+OR+" in q
    urls = [i["url"] for i in r["items"]]
    assert all(i["read_ok"] is True for i in r["items"])
    assert "https://en.wikinews.org/wiki/Story" in urls
    assert not any("nytimes" in u or "unknown-outlet" in u or "notglobalvoices" in u for u in urls)
    # globalvoices has crawl_delay_s 10: one item of that host per 10 s is offered for reading
    assert sum("globalvoices.org" in u for u in urls) == 1
    assert r["attribution"]["url"] == "https://www.gdeltproject.org/"


def test_emit_link_only_mode_tags_read_ok_false_and_drops_exclude():
    c = dict(shipped(), emit_link_only=True, outlet_query="all",
             outlets=shipped()["outlets"] + [{"host": "unknown-outlet.example", "terms_verdict": "exclude", "terms_evidence": "x"}])
    f = Fake((200, {}, ARTS))
    r = rl.ReportingLane(c, http_get=f, sleep=lambda s: None).search("midterm elections news")
    assert "domain%3A" not in f.urls[0]
    by = {i["url"]: i["read_ok"] for i in r["items"]}
    assert by["https://www.nytimes.com/2026/11/03/us/politics/x.html"] is False
    assert by["https://notglobalvoices.org/evil"] is False
    assert "https://unknown-outlet.example/n" not in by
    assert by["https://en.wikinews.org/wiki/Story"] is True
    assert sorted(v for u, v in by.items() if "globalvoices.org/20" in u) == [False, True]


def test_crawl_delay_across_requests():
    t = [0.0]
    L = rl.ReportingLane(shipped(), http_get=Fake((200, {}, ARTS), (200, {}, ARTS), (200, {}, ARTS)),
                         sleep=lambda s: None, clock=lambda: t[0])
    # items offered for READING (read_ok); since 2026-11-05 the others are still listed as links (emit_link_only)
    gv = lambda r: sum("globalvoices.org" in i["url"] and i["read_ok"] is True for i in r["items"])  # noqa: E731
    assert gv(L.search("midterm elections news")) == 1
    t[0] = 6.0
    assert gv(L.search("midterm elections news")) == 0  # 6 s later (min interval) < crawl delay 10 s
    t[0] = 12.0
    assert gv(L.search("midterm elections news")) == 1


def test_min_interval_skips_never_sleeps():
    t, sl = [100.0], []
    f = Fake((200, {}, ARTS), (200, {}, ARTS))
    c = shipped()
    c["allow_query"] = {"enabled": False}  # RETRIEVAL-1109: the extra allow query (own slot, waits) is tested separately
    L = rl.ReportingLane(c, http_get=f, sleep=sl.append, clock=lambda: t[0])
    L.search("midterm elections")
    t[0] += 2.0
    assert L.search("midterm elections")["items"] == [] and len(f.urls) == 1  # inside 6 s: skipped
    t[0] += 4.0
    assert L.search("midterm elections")["items"] and len(f.urls) == 2
    assert sl == []


def test_allowlist_mode_with_no_allow_hosts_makes_no_call():
    f = Fake()
    c = dict(shipped(), outlets=[], outlet_query="allowlist")
    r = rl.ReportingLane(c, http_get=f).search("midterm elections")
    assert r["items"] == [] and f.urls == []


def test_default_client_refuses_other_hosts_and_schemes(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    for u in ("http://api.gdeltproject.org/api/v2/doc/doc?q=x", "https://evil.example/api", "https://api.gdeltproject.org:8443/x",
              "https://127.0.0.1/x", "https://api.gdeltproject.org.evil.example/x", "file:///etc/passwd"):
        assert rl._default_http_get(u, {}, (5, 5)) == (0, {}, b"")


class _Resp:
    def __init__(self, ctype, body, status=200):
        self.headers, self._b, self.status = {"Content-Type": ctype}, body, status

    def read(self, n):
        out, self._b = self._b[:n], self._b[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_default_client_no_redirect_size_and_type(monkeypatch):
    import urllib.request
    seen = []

    def fake_open(self, req, timeout=None):
        seen.append(self)
        return state["resp"]
    state = {"resp": _Resp("application/json; charset=utf-8", b'{"articles": []}')}
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
    u = rl.GDELT_URL + "?query=x"
    assert rl._default_http_get(u, {}, (5, 5))[0] == 200
    handlers = [h for h in seen[0].handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    assert handlers and all(h.redirect_request(None, None, 302, "", {}, "https://evil.example/") is None for h in handlers)
    # API-ALLOW-1107: the real client now goes through the gdelt_doc rate portal: an immediate second call is paced
    # (nothing sent); a fresh registry per call below keeps the remaining checks about the client itself.
    from vera import rate_limit
    n = len(seen)
    with pytest.raises(rate_limit.RateLimited):
        rl._default_http_get(u, {}, (5, 5))
    assert len(seen) == n
    rate_limit.set_registry_for_tests(None)
    state["resp"] = _Resp("text/html", b"<html>")
    assert rl._default_http_get(u, {}, (5, 5)) == (0, {}, b"")
    rate_limit.set_registry_for_tests(None)
    state["resp"] = _Resp("application/json", b"x" * (rl.MAX_BODY + 10))
    assert rl._default_http_get(u, {}, (5, 5)) == (0, {}, b"")


def test_rollback_flag_off_means_no_call():
    f = Fake((200, {}, ARTS))
    r = rl.ReportingLane(dict(shipped(), enabled=False), http_get=f).search("midterm elections news")
    assert r["items"] == [] and f.urls == []


def test_about_page_has_separate_news_disclosure_block():
    pages = json.loads((ROOT / "config" / "ui_help_pages.json").read_text(encoding="utf-8"))
    blocks = [b for b in pages["about"]["after_disclosure"] if b.get("heading") == "News reports"]
    assert len(blocks) == 1 and len(blocks[0]["paragraphs"]) == 1
    assert "current events" in blocks[0]["paragraphs"][0] and "GDELT" in blocks[0]["paragraphs"][0]


# --- SEC B1 (work/NEWS-LANE-1104/security.md): no backoff sleeps in the request path; shared state; clamps ---

def test_429_never_sleeps_and_marks_unavailable_with_backoff_steps():
    t, sl = [0.0], []
    f = Fake((429, {}, b""), (429, {}, b""), (429, {}, b""), (429, {}, b""))
    L = rl.ReportingLane(shipped(), http_get=f, sleep=sl.append, clock=lambda: t[0], rand=lambda: 0.5)
    assert L.search("midterm elections")["items"] == [] and sl == [] and len(f.urls) == 1
    assert not L.available("gdelt") and L.search("midterm elections")["items"] == [] and len(f.urls) == 1
    opens = []
    for step in (60, 120, 240):
        t[0] = L._open_until["gdelt"]
        L.search("midterm elections")
        opens.append(L._open_until["gdelt"] - t[0])
    assert opens == [120.0, 240.0, 900.0]  # after max_retries failures: the 15-minute circuit
    assert sl == []


def test_retry_after_is_clamped_to_circuit():
    t = [0.0]
    f = Fake((429, {"Retry-After": "1000000000"}, b""))
    L = rl.ReportingLane(shipped(), http_get=f, sleep=lambda s: None, clock=lambda: t[0], rand=lambda: 0.5)
    L.search("midterm elections")
    assert L._open_until["gdelt"] == 900.0


def test_success_resets_failures():
    t = [0.0]
    L = rl.ReportingLane(shipped(), http_get=Fake((503, {}, b""), (200, {}, ARTS), (503, {}, b"")),
                         sleep=lambda s: None, clock=lambda: t[0], rand=lambda: 0.5)
    L.search("midterm elections")
    t[0] = 61.0
    assert L.search("midterm elections")["items"]
    t[0] = 70.0
    L.search("midterm elections news")
    assert L._open_until["gdelt"] - t[0] == 60.0


def test_production_lanes_share_state_across_instances(monkeypatch):
    monkeypatch.setattr(rl, "_SHARED", rl._new_state())
    calls = []
    monkeypatch.setattr(rl, "_default_http_get", lambda u, h, t: calls.append(u) or (429, {}, b""))
    a, b = rl.ReportingLane(shipped()), rl.ReportingLane(shipped())
    a.search("midterm elections")
    assert not b.available("gdelt")  # a new per-request lane sees the open backoff
    b.search("midterm elections")
    assert len(calls) == 1


def test_no_sleep_anywhere_in_request_path_on_failures():
    sl = []
    f = Fake((429, {"Retry-After": "30"}, b""), (503, {}, b""), TimeoutError())
    t = [0.0]

    def get(u, h, to):
        r = f.r.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    L = rl.ReportingLane(shipped(), http_get=get, sleep=sl.append, clock=lambda: t[0])
    for step in range(3):
        L.search("midterm elections")
        t[0] = L._open_until["gdelt"]
    assert sl == []


def test_default_client_rejects_userinfo_and_bad_port(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    for u in ("https://user:pw@api.gdeltproject.org/api/v2/doc/doc", "https://api.gdeltproject.org:abc/x"):
        assert rl._default_http_get(u, {}, (5, 5)) == (0, {}, b"")


def test_outlet_label_from_url_host_not_gdelt_domain_field():
    body = json.dumps({"articles": [{"url": "https://globalvoices.org/a", "title": "T", "domain": "nytimes.com"}]})
    assert rl.parse_gdelt(body)[0]["outlet"] == "globalvoices.org"


def test_status_zero_opens_backoff_window():  # SEC-RR R3
    t = [0.0]
    L = rl.ReportingLane(shipped(), http_get=Fake((0, {}, b"")), sleep=lambda s: None, clock=lambda: t[0], rand=lambda: 0.5)
    L.search("midterm elections")
    assert not L.available("gdelt") and L._open_until["gdelt"] == 60.0
