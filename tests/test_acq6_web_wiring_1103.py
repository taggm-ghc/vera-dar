"""NONASK-1103 A6-W wiring: the round-0 web route inside the inline loop. Offline: public_fetch.fetch is a stub (no
network), every acquisition dependency is the rounds-test stub, no database, no env files."""
from __future__ import annotations

import json

from tests import leak_sentinel
from tests.test_acq6_rounds_1103 import H, cand, cfg
from vera import inline_acquire as ia
from vera import lane_evidence, public_fetch, web_route


def web(tag, host):
    return {"title": f"Web page {tag}", "url": f"https://{host}/{tag}", "provider": "ddg", "snippet": "agent patterns"}


def _stub_fetch(mp, text, calls):
    def fetch(url, *a, **k):
        calls.append(url)
        return {"text": text, "url": url, "content_type": "text/html", "headers": {}, "meta_robots": []}
    mp.setattr(public_fetch, "fetch", fetch)


def test_web_pages_count_toward_six_never_stored_cached_or_queued(monkeypatch):
    s = leak_sentinel.new()
    calls = []
    text = (f"{s} Multi-agent orchestration uses an orchestrator agent, planner agents and worker agents; agent "
            "archetypes in multi-agent orchestration include router, critic and supervisor agents. ") * 3
    _stub_fetch(monkeypatch, text, calls)
    pools = [[cand("w0"), cand("w1"), web("a", "blog-one.example"), web("b", "blog-two.example"),
              web("c", "docs.aws.amazon.com"), web("d", "blog-one.example")]]
    h = H(monkeypatch, pools)
    o = h.run(cfg(), stream=True)
    assert not any("aws.amazon.com" in u for u in calls), "a read-restricted host was fetched"
    assert sum("blog-one.example" in u for u in calls) == 1, "more than one fetch per host per round"
    n_web = len(o.web.pages)
    assert n_web == 2
    assert o.admitted + n_web == 4 and len(h.searches) >= 2  # 2 docs + 2 web pages < 6: rounds continue
    assert s not in json.dumps(h.writes, default=str), "web page text reached the store"
    assert o.no_cache and not ia._I.results, "an outcome holding web page text was cached"
    assert s not in json.dumps(o.public(), default=str), "web page text in the public outcome"
    recs = [r for r in (o.candidates or []) if r.get("url") and "example" in str(r.get("url"))]
    assert recs and all(r["outcome"] == "web_answer_only" and r.get("background") == "terminal" for r in recs)
    assert not ia.continue_eligible(cfg(), recs[0])  # N4: never handed to enqueue_candidates


def test_web_pages_reach_target_and_stop_rounds(monkeypatch):
    calls = []
    _stub_fetch(monkeypatch, "Agent archetypes in multi-agent orchestration: orchestrator, worker. " * 5, calls)
    pools = [[cand("x0"), cand("x1"), cand("x2"), cand("x3")] + [web(str(i), f"h{i}.example") for i in range(2)]]
    h = H(monkeypatch, pools)
    o = h.run(cfg(), stream=False)
    assert len(o.web.pages) == 2 and o.acquisition["status"] == "met" and len(h.searches) == 1


def test_kill_switch_off_reads_nothing(monkeypatch):
    calls = []
    _stub_fetch(monkeypatch, "text " * 50, calls)
    h = H(monkeypatch, [[cand("k0"), web("a", "blog.example")]])
    o = h.run(cfg(web_read_enabled=False), stream=True)
    assert calls == [] and o.web is None


def test_with_web_merges_as_lane_web_and_admits(monkeypatch):
    calls = []
    _stub_fetch(monkeypatch, "Agent archetypes in multi-agent orchestration: orchestrator, planner, worker. " * 8, calls)
    res = web_route.read_web("agent archetypes multi-agent orchestration", [web("a", "blog.example")])
    o = ia.Outcome("none")
    o.web = res
    lanes = ia.with_web(o, None)
    assert lanes is not None and len(lanes.pages) == 1 and lanes.links[0]["lane"] == "web"
    assert "text" not in json.dumps(lanes.public())
    from vera import fulldoc_corpus
    adm, ex = lane_evidence.admit_pages("agent archetypes in multi-agent orchestration", lanes.pages,
                                        fulldoc_corpus.load_config())
    assert ex and ex[0]["lane"] == "web" and ex[0]["stored"] is False
    assert all(a["epistemic"] == "web" and a["answer_only"] for a in adm)
    assert lane_evidence.attribution_for(lanes.pages[0])["creator"].startswith("Web source")
    assert ia.with_web(ia.Outcome("none"), None) is None


def test_default_registry_is_consulted_fail_closed():
    """No registry argument = the shipped registry (AWS, Nvidia, Microsoft Learn access forbid), never 'no rows'."""
    from vera import source_permissions as sp
    for u in ("https://docs.aws.amazon.com/x", "https://docs.nvidia.com/x", "https://learn.microsoft.com/azure/x"):
        assert sp.read_gate(u).read is False, u


def test_disclosure_shown_iff_web_route_on(monkeypatch):
    from vera import disclosure
    s = "Public web pages may be read to answer, and are stored only when their licence allows."
    monkeypatch.setattr(disclosure, "web_read_on", lambda: True)
    assert s in disclosure.third_party_text()
    monkeypatch.setattr(disclosure, "web_read_on", lambda: False)
    assert s not in disclosure.third_party_text()


def test_shipped_config_enables_web_route():
    from vera import acq_config
    assert ia.web_read_enabled(acq_config.load_config()) is True
