"""NONASK-1103 I-SL push 2b: lanes on the combined base-plus-inline path, the official-lane DDG site: query (allow
rows only, shared provider, ships off) and the lane-aware passage-count disclosure. Offline: everything is stubbed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vera import ask_service, disclosure, fulldoc_answer, fulldoc_corpus, lane_evidence, official_lane
from vera.fulldoc_evidence import EvidenceBundle
from vera.official_lane import LanePage

Q = "What do the Google Cloud docs say about Vertex AI model tuning for cloud ai engineering?"
TEXT = ("Vertex AI model tuning on Google Cloud lets cloud ai engineering teams tune foundation models with "
        "supervised tuning jobs and evaluate tuned models before deployment. ") * 3

REG = [{"entity": "Google Cloud", "aliases": ["Google Cloud"], "state": "active", "terms_verdict": "allow",
        "terms_evidence": "work/NONASK-1103/entity-hosts-evidence.md", "domains": ["docs.cloud.google.com"],
        "docs_entry_urls": ["https://docs.cloud.google.com/docs"]},
       {"entity": "AWS", "aliases": ["AWS"], "state": "active", "terms_verdict": "links_only",
        "terms_evidence": "x", "access_verdict": "forbid", "domains": ["docs.aws.amazon.com"], "docs_entry_urls": ["https://docs.aws.amazon.com/"]}]


def _lanes():
    return lane_evidence.LaneResult(pages=[LanePage("official", "Google Cloud", "Vertex AI tuning",
                                                    "https://docs.cloud.google.com/docs/vertex", text=TEXT)])


def _base_ev():
    return EvidenceBundle(question=Q, examined_full_documents=({"document_id": 1},),
                          admitted=({"span_id": "s1", "document_id": 1, "text": "base passage", "text_sha256": "h"},),
                          status="admitted")


@pytest.fixture
def combined(monkeypatch):
    seen = {}

    def fake_answer(question, ev, *a, **k):
        seen["ev"] = ev
        return SimpleNamespace(result=None, fallback="stub", tokens_used=0, cost_usd=0.0)
    monkeypatch.setattr(fulldoc_answer, "answer", fake_answer)
    monkeypatch.setattr(ask_service, "_base_warming", lambda: True)
    monkeypatch.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg: None)
    monkeypatch.setattr(ask_service, "_pre_llm", lambda **k: None, raising=False)
    fd = SimpleNamespace(evidence=_base_ev(), tokens_used=0, cost_usd=0.0)
    return seen, fd


def test_combined_path_adds_lane_spans_when_inline_found_nothing(combined):
    seen, fd = combined
    o = SimpleNamespace(corpus=None)
    res, failed = ask_service._combined_answer(Q, "m", None, {}, None, {}, fd, o, lanes=_lanes())
    ev = seen["ev"]
    lane = [a for a in ev.admitted if a.get("answer_only")]
    assert lane and all(a["origin"] == "official" and a["stored"] is False for a in lane)
    assert any(a.get("span_id") == "s1" for a in ev.admitted)  # base passages kept, lane in its own slots


def test_combined_path_without_lanes_is_unchanged(combined):
    seen, fd = combined
    res, failed = ask_service._combined_answer(Q, "m", None, {}, None, {}, fd, SimpleNamespace(corpus=None))
    assert (res, failed) == (None, False) and "ev" not in seen


def test_combined_path_irrelevant_lane_page_changes_nothing(combined):
    seen, fd = combined
    lanes = lane_evidence.LaneResult(pages=[LanePage("official", "Google Cloud", "t", "https://docs.cloud.google.com/x",
                                                     text="Unrelated cooking recipes and garden tips. " * 20)])
    assert ask_service._combined_answer(Q, "m", None, {}, None, {}, fd, SimpleNamespace(corpus=None),
                                        lanes=lanes) == (None, False)


def test_site_search_only_for_allow_rows_and_exact_hosts():
    calls = []

    def site(host, terms):
        calls.append((host, tuple(terms)))
        return [{"url": "https://docs.cloud.google.com/vertex-ai/docs/tuning", "title": "Tuning"},
                {"url": "https://evil.example.com/docs.cloud.google.com", "title": "x"},
                {"url": "http://docs.cloud.google.com/plain", "title": "x"}]
    fetched = []
    ol = official_lane.OfficialLane({"enabled": True}, registry=REG, site_search=site,
                                    fetch=lambda u: fetched.append(u) or {"text": TEXT, "url": u})
    out = ol.run(["Google Cloud", "AWS"], ["vertex", "tuning"])
    assert calls == [("docs.cloud.google.com", ("vertex", "tuning"))]  # one query, allow row only
    urls = [lk["url"] for lk in out["links"]]
    assert "https://docs.cloud.google.com/vertex-ai/docs/tuning" in urls
    assert not any("evil" in u or u.startswith("http://") for u in urls)
    assert "https://docs.aws.amazon.com/" in urls and not any("aws" in u for u in fetched)  # links only


def test_site_search_ships_off_and_never_runs_by_default(monkeypatch):
    assert official_lane.load_config().get("site_search") is False
    monkeypatch.setattr(official_lane, "_ddg_site_search", lambda *a: pytest.fail("site search ran"))
    ol = official_lane.OfficialLane({"enabled": True}, registry=REG, fetch=lambda u: {"text": TEXT, "url": u})
    ol.run(["Google Cloud"], ["vertex"])


def test_ddg_site_search_uses_shared_provider_with_explicit_query(monkeypatch):
    from vera import search_and_fetch, search_providers as sp
    got = {}

    class P:
        name = sp.DDG_NAME
        s = {"max_query_chars": 480}

        def call(self, query, n, timeout_s, *, queries=None):
            got["queries"] = queries
            return "resp"

        def parse(self, resp, n):
            return [{"url": "https://docs.cloud.google.com/a", "title": "a"}]
    monkeypatch.setattr(search_and_fetch, "cached_chain", lambda: ([P()], 1, "m"))
    assert official_lane._ddg_site_search("docs.cloud.google.com", ["vertex", "tuning"])[0]["url"].endswith("/a")
    assert got["queries"] == [{"group": "official", "query": "vertex tuning (site:docs.cloud.google.com)"}]


def test_passage_count_sentence_flag_gated(monkeypatch):
    monkeypatch.setattr(disclosure, "web_read_on", lambda: False)
    monkeypatch.setattr(disclosure, "lanes_disclosure_on", lambda: False)
    assert disclosure.base_fulldoc() == disclosure._BASE_FULLDOC
    monkeypatch.setattr(disclosure, "lanes_disclosure_on", lambda: True)
    t = disclosure.base_fulldoc()
    assert "plus up to six passages from public web pages it reads for that answer" in t
    assert "including any from public pages read for that answer" in t
    assert "".join(disclosure._fulldoc_parts()) == t  # the section slices still join to the text
