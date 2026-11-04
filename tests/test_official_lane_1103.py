"""NONASK-1103 S4: offline tests for vera/official_lane.py and the lane-flag/registry config. No network, DB or env."""
import json
from pathlib import Path

import pytest

from vera import lane_evidence, official_lane as ol, public_fetch, reporting_lane, subquery

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "cloud AI engineering on AWS, Google Cloud, Nvidia and Azure"


def _reg():
    return json.loads((ROOT / "config" / "entity_registry.json").read_text(encoding="utf-8"))["entities"]


class Stub:
    def __init__(self, text="Vendor docs page about cloud AI engineering.", final=None, exc=None):
        self.calls, self.text, self.final, self.exc = [], text, final, exc

    def __call__(self, url):
        self.calls.append(url)
        if self.exc:
            raise self.exc
        return {"text": self.text, "url": self.final or url, "content_type": "text/html"}


def on(**kw):
    return dict({"enabled": True}, **kw)


# ---- shipped config ---------------------------------------------------------------------------------
def test_shipped_flags_off():
    assert ol.load_config()["enabled"] is False
    assert ol.lane_enabled() is False
    assert reporting_lane.load_config(ROOT / "config" / "reporting_lane.json")["enabled"] is False
    assert json.loads((ROOT / "config" / "reporting_lane.json").read_text())["outlets"] == []


def test_lane_enabled_fails_closed():
    for v in ("true", 1, "yes", None):
        assert ol.lane_enabled({"enabled": v}) is False
    assert ol.lane_enabled({"enabled": True}) is True


def test_registry_seed_facts():
    rows = {r["entity"]: r for r in _reg()}
    g = rows["Google Cloud"]
    assert g["terms_verdict"] == "allow" and g["terms_evidence"]
    assert "docs.cloud.google.com" in g["domains"]
    assert "https://docs.cloud.google.com/docs" in g["docs_entry_urls"]
    for name in ("AWS", "Nvidia", "Azure"):
        assert rows[name]["terms_verdict"] == "links_only"
    for r in rows.values():
        assert r.get("terms_verdict") in ol.VERDICTS


def test_verdict_rules():
    assert ol.verdict({}) == "links_only"
    assert ol.verdict({"terms_verdict": "unknown", "terms_evidence": "x"}) == "links_only"
    assert ol.verdict({"terms_verdict": "allow"}) == "links_only"
    assert ol.verdict({"terms_verdict": "allow", "terms_evidence": ""}) == "links_only"
    assert ol.verdict({"terms_verdict": "bogus", "terms_evidence": "x"}) == "links_only"
    assert ol.verdict({"terms_verdict": "links_only", "terms_evidence": "x"}) == "links_only"
    assert ol.verdict({"terms_verdict": "allow", "terms_evidence": "x"}) == "allow"
    assert ol.verdict(None) == "links_only"


# ---- flags off --------------------------------------------------------------------------------------
def test_flags_off_collect_empty_no_fetch():
    f = Stub()
    res = lane_evidence.collect(QUESTION, fetch=f)  # shipped config: both flags off
    assert res.links == [] and res.pages == [] and f.calls == []
    assert lane_evidence.any_enabled() is False


def test_run_flag_off_no_fetch():
    f = Stub()
    out = ol.OfficialLane({"enabled": False}, registry=subquery.load_entity_registry(), fetch=f).run(["Google Cloud"])
    assert out == {"links": [], "pages": []} and f.calls == []


# ---- example question -------------------------------------------------------------------------------
def test_example_question_one_link_per_entity_fetch_only_allow(monkeypatch):
    monkeypatch.setattr(ol, "load_config", lambda path=None: on())
    plan = subquery.decompose(QUESTION)
    ents = [e for p in plan.parts for e in p.entities]
    assert {"AWS", "Google Cloud", "Nvidia", "Azure"} <= set(ents)
    f = Stub()
    lane = ol.OfficialLane(on(), registry=subquery.load_entity_registry(), fetch=f)
    out = lane.run(ents)
    by = {lk["entity"]: lk for lk in out["links"]}
    assert set(by) == {"AWS", "Google Cloud", "Nvidia", "Azure"} and len(out["links"]) == 4
    assert f.calls == ["https://docs.cloud.google.com/docs"]
    assert by["Google Cloud"]["read"] is True and len(out["pages"]) == 1
    for n in ("AWS", "Nvidia", "Azure"):
        assert by[n]["read"] is False and by[n]["reason"] == "read_registry_forbid"


def test_collect_example_question_through_lane_evidence(monkeypatch):
    monkeypatch.setattr(ol, "load_config", lambda path=None: on())
    f = Stub()
    res = lane_evidence.collect(QUESTION, fetch=f)
    ents = {lk["entity"] for lk in res.links if lk["lane"] == "official"}
    assert ents == {"AWS", "Google Cloud", "Nvidia", "Azure"}
    assert f.calls == ["https://docs.cloud.google.com/docs"]


@pytest.mark.parametrize("q", ["cloud ai engineering trends", "google it", "amazon rainforest engineering", "azure sky"])
def test_lexically_similar_non_entity_questions_zero_links(q, monkeypatch):
    monkeypatch.setattr(ol, "load_config", lambda path=None: on())
    f = Stub()
    res = lane_evidence.collect(q, fetch=f)
    assert [lk for lk in res.links if lk.get("lane") == "official"] == []
    assert f.calls == []


def test_off_registry_final_url_link_only():
    f = Stub(final="https://evil.example.com/docs")
    out = ol.OfficialLane(on(), registry=_reg(), fetch=f).run(["Google Cloud"])
    assert out["pages"] == []
    assert out["links"][0]["reason"] == "off_registry" and out["links"][0]["read"] is False


@pytest.mark.parametrize("reason", ["robots", "private_ip", "status_401", "status_403"])
def test_fetch_refused_link_only(reason):
    f = Stub(exc=public_fetch.FetchRefused(reason))
    out = ol.OfficialLane(on(), registry=_reg(), fetch=f).run(["Google Cloud"])
    assert out["pages"] == [] and len(out["links"]) == 1
    assert out["links"][0]["read"] is False and out["links"][0]["reason"] == reason


def test_unknown_access_rows_readable_forbid_rows_never_fetched():
    reg = [{"entity": "X", "domains": ["x.example"], "docs_entry_urls": ["https://x.example/d"], "state": "active",
            "terms_verdict": "unknown", "terms_evidence": None},
           {"entity": "Y", "domains": ["y.example"], "docs_entry_urls": ["https://y.example/d"], "state": "active"},
           {"entity": "Z", "domains": ["z.example"], "docs_entry_urls": ["https://z.example/d"], "state": "active",
            "terms_verdict": "allow"}]
    reg.append({"entity": "F", "domains": ["f.example"], "docs_entry_urls": ["https://f.example/d"],
                "state": "active", "access_verdict": "forbid"})
    f = Stub()
    out = ol.OfficialLane(on(), registry=reg, fetch=f).run(["X", "Y", "Z", "F", "NoRow"])
    # access_verdict unknown/missing is readable (I-A6); forbid is never fetched
    assert sorted(f.calls) == ["https://x.example/d", "https://y.example/d", "https://z.example/d"]
    assert len(out["links"]) == 4 and len(out["pages"]) == 3
    assert [lk for lk in out["links"] if lk["entity"] == "F"][0]["reason"] == "read_registry_forbid"


def test_inactive_row_gives_nothing():
    reg = [dict(r) for r in _reg()]
    for r in reg:
        if r["entity"] == "Google Cloud":
            r["state"] = "suspended"
    f = Stub()
    assert ol.OfficialLane(on(), registry=reg, fetch=f).run(["Google Cloud"])["links"] == [] and f.calls == []


def test_host_ok_exact_match_only():
    row = {"domains": ["docs.cloud.google.com"]}
    assert ol.host_ok("https://docs.cloud.google.com/x", row)
    assert not ol.host_ok("https://evil.docs.cloud.google.com/x", row)
    assert not ol.host_ok("https://cloud.google.com/x", row)


def test_lanepage_repr_has_no_text():
    pg = ol.LanePage("official", "E", "t", "https://h/x", text="SECRET-BODY-TEXT")
    assert "SECRET-BODY-TEXT" not in repr(pg) and "SECRET-BODY-TEXT" not in str(pg)
