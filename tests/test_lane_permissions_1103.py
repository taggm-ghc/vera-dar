"""NONASK-1103 I-SL: lanes obey vera.source_permissions (I-A6). Offline: fetch is stubbed, nothing is stored."""
import json
from pathlib import Path

from vera import lane_evidence as le, official_lane as ol, reporting_lane as rl, source_permissions as sp

ROOT = Path(__file__).resolve().parent.parent
Q = "What is the latest news on kubernetes outages?"


def _reg():
    return json.loads((ROOT / "config" / "entity_registry.json").read_text(encoding="utf-8"))["entities"]


class Fetch:
    def __init__(self):
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return {"text": "Page body about the latest outage.", "url": url, "content_type": "text/html"}


class FakeReporting:
    def __init__(self, items):
        self.items, self.cfg = items, {"enabled": True, "outlets": []}

    def search(self, text, ents=()):
        # INTEGRATE-1104 (NEWS-LANE R1): the real ReportingLane tags every item; a fake for a READ case tags
        # read_ok=True (an explicit read_ok=False stays a link-only case)
        return {"items": [{**it, "read_ok": it.get("read_ok", True)} for it in self.items], "attribution": None,
                "terms": []}


def _items(hosts, n=1):
    return [{"title": f"t{i}", "url": f"https://{h}/a{i}", "published": "2026-11-01", "outlet": h}
            for h in hosts for i in range(n)]


def rep_on(mp):
    mp.setattr(rl, "lane_enabled", lambda cfg=None: True)
    # INTEGRATE-1104 (NEWS-LANE R1): every reporting hop needs an "allow" outlet verdict; the read cases run with the
    # fake outlets treated as allow (permission denials still come from source_permissions, unchanged)
    mp.setattr(rl, "outlet_verdict", lambda url, cfg=None: "allow")


def test_forbid_vendors_link_only_google_fetched():
    f = Fetch()
    out = ol.OfficialLane({"enabled": True}, registry=_reg(), fetch=f).run(["AWS", "Nvidia", "Azure", "Google Cloud"])
    by = {lk["entity"]: lk for lk in out["links"]}
    assert f.calls == ["https://docs.cloud.google.com/docs"]
    for n in ("AWS", "Nvidia", "Azure"):
        assert by[n]["read"] is False and by[n]["reason"] == "read_registry_forbid"
    assert by["Google Cloud"]["read"] is True and len(out["pages"]) == 1


def test_derivative_forbid_row_link_only_with_use_reason():
    reg = [{"entity": "D", "state": "active", "domains": ["d.example"], "docs_entry_urls": ["https://d.example/x"],
            "access_verdict": "allow", "derivative_verdict": "forbid"}]
    f = Fetch()
    out = ol.OfficialLane({"enabled": True}, registry=reg, fetch=f).run(["D"])
    assert f.calls == [] and out["pages"] == []
    assert out["links"][0]["read"] is False and out["links"][0]["reason"] == "use_registry_forbid"


def test_deny_listed_outlet_not_fetched_others_read(monkeypatch):
    rep_on(monkeypatch)
    monkeypatch.setattr(sp, "load_config", lambda *a, **k: {"deny_hosts": ["blocked.example"]})
    f = Fetch()
    res = le.collect(Q, reporting=FakeReporting(_items(["blocked.example", "ok.example"])), fetch=f)
    assert f.calls == ["https://ok.example/a0"] and len(res.pages) == 1
    by = {lk["outlet"]: lk for lk in res.links}
    assert by["blocked.example"]["read"] is False and by["ok.example"]["read"] is True


def test_read_cap_holds(monkeypatch):
    rep_on(monkeypatch)
    monkeypatch.setattr(le, "_read_cap", lambda: 2)
    f = Fetch()
    res = le.collect(Q, reporting=FakeReporting(_items(["a.example", "b.example", "c.example"], 2)), fetch=f)
    assert len(f.calls) == 2 and len(res.pages) == 2
    assert len(res.links) == 6 and sum(lk["read"] for lk in res.links) == 2


def test_lane_pages_never_ingested_answer_only(monkeypatch):
    rep_on(monkeypatch)
    cfg = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text(encoding="utf-8"))
    q = "What is the latest news on how kubernetes autoscaling adjusts replica counts using metrics?"
    body = ("Kubernetes autoscaling adjusts replica counts based on observed metrics. "
            "Horizontal pod autoscaling reads metrics and scales deployments automatically. " * 3)
    f = Fetch()
    f.__class__ = type("F2", (Fetch,), {"__call__": lambda self, url: (self.calls.append(url), {"text": body, "url": url})[1]})
    res = le.collect(q, reporting=FakeReporting(_items(["ok.example"])), fetch=f)
    assert res.pages
    admitted, examined = le.admit_pages(q, res.pages, cfg)
    assert admitted and examined
    for s in admitted + examined:
        assert s["stored"] is False and s["answer_only"] is True
