"""RETRIEVAL-1109 A5 + H-1: the allow-outlet GDELT query and the ND answer-only state. Offline; no network, DB or env."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vera import lane_evidence as le, licence_gate as lg, news_access as na, reporting_lane as rl

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "reporting_lane.json"


def shipped():
    return rl.load_config(CFG)


class Fake:
    def __init__(self, *bodies):
        self.b, self.urls = list(bodies), []

    def __call__(self, url, headers, timeout):
        self.urls.append(url)
        return 200, {}, (self.b.pop(0) if self.b else b'{"articles": []}')


def arts(*rows):
    return json.dumps({"articles": [{"url": u, "title": t, "seendate": d} for u, t, d in rows]}).encode()


ALL = arts(("https://www.nytimes.com/2026/11/08/x.html", "NYT", "20261108T100000Z"),
           ("https://www.votebeat.org/2026/11/07/dup", "VB dup", "20261107T100000Z"))
ALLOW = arts(("https://www.votebeat.org/2026/11/07/dup", "VB dup", "20261107T100000Z"),
             ("https://globalvoices.org/2026/11/01/a/", "GV", "20261101T100000Z"))


def run(cfg, fake, t=None, streamed=True):
    t = t or [100.0]
    sl = []
    if streamed:
        from vera import stream_ctx
        tok = stream_ctx.CTX.set(object())

    def sleep(s):
        sl.append(s)
        t[0] += s
    L = rl.ReportingLane(cfg, http_get=fake, sleep=sleep, clock=lambda: t[0])
    try:
        return L.search("midterm elections"), sl, L
    finally:
        if streamed:
            stream_ctx.CTX.reset(tok)


def test_shipped_defaults():
    c = shipped()
    assert c["allow_query"]["enabled"] is True and c["outlet_query"] == "all"
    assert "_calibration_note" in c["allow_query"]
    assert c["nc_basis"]["status"] == "SATISFIED" and c["nc_basis"]["ref"] == "NC-BASIS-1"
    assert lg.nd_news_enabled() is True
    by = {o["host"]: o["terms_verdict"] for o in c["outlets"]}
    assert by["propublica.org"] == "exclude" and by["theconversation.com"] == "exclude"  # separate H5 decision


def test_allow_query_second_call_only_allow_domains_own_slot_dedupe():
    f = Fake(ALL, ALLOW)
    res, sl, _ = run(shipped(), f)
    assert len(f.urls) == 2
    q1 = parse_qs(urlparse(f.urls[0]).query)["query"][0]
    q2 = parse_qs(urlparse(f.urls[1]).query)["query"][0]
    assert "domain:" not in q1
    doms = {w.split(":", 1)[1] for w in q2.replace("(", " ").replace(")", " ").split() if w.startswith("domain:")}
    assert doms == set(rl.allow_hosts(shipped())) and "nytimes.com" not in doms
    assert sl == [6.0]  # waited the whole 6 s slot (fake clock does not advance on its own)
    urls = [i["url"] for i in res["items"]]
    assert len(urls) == len(set(urls)) == 3
    by = {i["url"]: i["read_ok"] for i in res["items"]}
    assert by["https://globalvoices.org/2026/11/01/a/"] is True and by["https://www.nytimes.com/2026/11/08/x.html"] is False
    assert res["diag"][0]["allow_query"] == {"status": "called", "reason": "ok", "items": 1, "raw_items": 2}


def test_switch_off_is_exactly_one_call():
    c = shipped()
    c["allow_query"]["enabled"] = False
    f = Fake(ALL, ALLOW)
    res, sl, _ = run(c, f)
    assert len(f.urls) == 1 and sl == [] and "allow_query" not in res["diag"][0]
    assert len(res["items"]) == 2


def test_no_wait_mode_skips_inside_interval_and_circuit_blocks():
    c = shipped()
    c["allow_query"]["wait_for_slot"] = False
    f = Fake(ALL, ALLOW)
    res, sl, _ = run(c, f)
    assert len(f.urls) == 1 and sl == [] and res["diag"][0]["allow_query"]["reason"] == "spacing"
    # 429 on the all call opens the circuit: no allow query
    f = Fake()
    f.__class__ = type("F429", (Fake,), {"__call__": lambda s, u, h, t: (s.urls.append(u), (429, {}, b""))[1]})
    res, sl, _ = run(shipped(), f)
    assert len(f.urls) == 1


def test_allow_query_failure_keeps_all_results():
    class F(Fake):
        def __call__(self, url, headers, timeout):
            self.urls.append(url)
            return (200, {}, ALL) if len(self.urls) == 1 else (500, {}, b"")
    f = F()
    res, _, _ = run(shipped(), f)
    assert len(f.urls) == 2 and len(res["items"]) == 2 and res["diag"][0]["status"] == "called"


# H-1: ND allow rows
ND_URL = "https://www.votebeat.org/2026/11/07/dup"


def test_nd_row_resolves_full_text_when_on_and_link_only_when_off(monkeypatch):
    c = shipped()
    assert na.mode(ND_URL, c) == "full_text" and rl.outlet_verdict(ND_URL, c) == "allow"
    off = json.loads(json.dumps(c))
    off["nc_basis"]["status"] = "UNCONFIRMED"
    assert na.mode(ND_URL, off) == "link_only" and rl.outlet_verdict(ND_URL, off) == "links_only"
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: False)
    assert rl.outlet_verdict(ND_URL, c) == "links_only"


def test_nd_candidate_is_defer_answer_only_never_allow_or_stored():
    cand = {"url": ND_URL, "licence": {"content": {"id": "CC-BY-NC-ND-4.0", "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/"}}}
    d, why, code = lg.licence_status(cand)
    assert (d, code) == ("defer", lg.RC_ND_NEWS) and "not stored" in why
    assert lg.licence_decision(cand)[0] != "allow"
    assert lg.apply_licence_gate([cand])[0]["licence_decision"] == "defer"


class Rep:
    def __init__(self, url=ND_URL):
        self.url = url

    def search(self, text, ents=()):
        return {"items": [{"url": self.url, "title": "VB", "outlet": "votebeat.org", "published": "2026-11-07", "read_ok": True}],
                "attribution": None, "terms": [], "diag": []}


BODY = ("County officials certified the midterm election results this week after a recount of the ballots. "
        "Election officials said the midterm election canvass finished on schedule. ") * 4


class Fetch:
    def __init__(self):
        self.calls = []

    def __call__(self, url, *a, **k):
        self.calls.append(url)
        return {"text": BODY, "url": url, "content_type": "text/html"}


def _run_nd(monkeypatch):
    import vera.memory_store as ms
    stored = []
    for name in dir(ms):
        if name.startswith(("store", "save", "put", "write", "enqueue", "ingest", "upsert", "remember", "add")):
            obj = getattr(ms, name)
            if callable(obj) and not isinstance(obj, type):
                monkeypatch.setattr(ms, name, lambda *a, _n=name, **k: stored.append(_n))
    monkeypatch.setattr(le.reporting_lane, "lane_enabled", lambda cfg=None: True)
    f = Fetch()
    res = le.collect("What is the latest news on the midterm election results today?", reporting=Rep(), fetch=f)
    return res, f, stored


def test_nd_allow_row_is_read_answer_only_never_stored(monkeypatch):
    res, f, stored = _run_nd(monkeypatch)
    assert ND_URL in f.calls and len(res.pages) == 1 and res.links[0]["read"] is True
    cfg = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
    adm, exam = le.admit_pages("midterm election results certified recount", res.pages, cfg)
    assert adm, "ND page must yield admitted answer spans"
    from vera import memory_store as ms
    for a in adm:
        assert a["answer_only"] is True and a["stored"] is False and a["licence_class"] == "nd"
        assert le.is_lane(a) and ms.answer_only(a) is True  # memory_store skips it (R-037)
    for d in exam:
        assert d["stored"] is False and d["answer_only"] is True and d["licence_decision"] == le.LICENCE_DISPOSITION
    assert stored == []  # no store/enqueue/ingest/save call on the memory store


def test_nd_row_link_only_when_switch_off(monkeypatch):
    monkeypatch.setattr(lg, "nd_news_enabled", lambda *a, **k: False)
    res, f, _ = _run_nd(monkeypatch)
    assert ND_URL not in f.calls and res.pages == [] and res.links[0]["read"] is False


def test_not_streamed_never_sleeps_and_skips_extra_query():
    f = Fake(ALL, ALLOW)
    res, sl, _ = run(shipped(), f, streamed=False)
    assert len(f.urls) == 1 and sl == [] and res["diag"][0]["allow_query"]["reason"] == "spacing"
    assert len(res["items"]) == 2
