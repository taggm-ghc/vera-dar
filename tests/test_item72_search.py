"""Item #72 W3a-c tests (offline, injected transports). R72-d dedupe, R72-e zero-hit relaxation, R72-f labels.
No false-merge rate, relaxation effect or label effect is claimed here (all unverified)."""
import json

import pytest
import requests

import vera.search_and_fetch as sf
import vera.search_providers as sp
from vera.eval_config import load_eval_config
from vera.m4.context_builder import build_reasoning_context
from vera.search_and_fetch import _finalize, rrf_merge, search_question
from vera.search_providers import ArxivProvider, OpenAlexProvider, build_arxiv_query_info, build_openalex_search
from vera.source_labels import NEWS_LABEL, VENDOR_LABEL, source_label
from vera.trace_eval.capture import _format_sources

CFG = load_eval_config().search
QT_SHIPPED = CFG["query_terms"]
QT = {**QT_SHIPPED, "relax_on_zero": True}  # relaxation tests build their own settings with the feature on
Q = "What do controlled experiments show about AI coding assistants and developer productivity?"

T1 = ("The AI Productivity Paradox in Software Engineering: A Systematic Review of Code Quality, Technical Debt, "
      "and Organizational Throughput (2024–2026)")
T2 = T1.replace("Throughput", "Throughout")  # the #71 smoke records differ by this typo
A = {"url": "https://doi.org/10.17605/osf.io/jtqgn", "doi": "10.17605/osf.io/jtqgn", "title": T1,
     "published": "2026-01-01", "provider": "openalex"}
B = {"url": "https://doi.org/10.66977/xsci.2610.0006", "doi": "10.66977/xsci.2610.0006", "title": T2,
     "published": "2026-10-02", "provider": "openalex"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(sp, "arxiv_default_transport", boom)
    monkeypatch.setattr(sp, "openalex_default_transport", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(sf, "_CHAIN_CACHE", {})


# ---------------------------------------------------------------------------------------------- R72-d
def test_known_71_duplicate_merges_with_visible_reason():
    out = _finalize([A, {**B, "provider": "duckduckgo_lite"}], "x", 10)
    assert len(out) == 1 and out[0]["url"] == A["url"]            # higher-ranked kept
    assert out[0]["found_by_providers"] == ["openalex", "duckduckgo_lite"]
    m = out[0]["merged_from"][0]
    assert m["reason"] == "title_family+year" and m["url"] == B["url"] and m["provider"] == "duckduckgo_lite"


def test_same_title_different_year_not_merged():
    a = {"url": "https://x.org/1", "title": T1, "year": 2024}
    b = {"url": "https://y.org/2", "title": T1, "year": 2025}
    assert len(_finalize([a, b], "x", 10)) == 2
    # and the family key never crosses years either
    assert len(_finalize([{**A, "published": "2025-01-01"}, B], "x", 10)) == 2


def test_exact_title_and_year_merges_and_authors_guard():
    a = {"url": "https://x.org/1", "title": T1, "year": 2024, "authors": ["Ada Lovelace"]}
    b = {"url": "https://y.org/2", "title": T1.upper(), "year": 2024, "authors": ["Lovelace, A."]}
    c = {"url": "https://z.org/3", "title": T1, "year": 2024, "authors": ["Grace Hopper"]}
    out = _finalize([a, b, c], "x", 10)
    assert [r["url"] for r in out] == ["https://x.org/1", "https://z.org/3"]
    assert out[0]["merged_from"][0]["reason"] == "title+year"


def test_short_family_prefix_never_merges():
    a = {"url": "https://x.org/1", "title": "Code review: practice A", "year": 2024}
    b = {"url": "https://y.org/2", "title": "Code review: practice B", "year": 2024}
    assert len(_finalize([a, b], "x", 10)) == 2


def test_hard_keys_still_merge_and_rrf_records_merges():
    a = {"url": "https://x.org/1", "doi": "10.1/a", "title": "Some distinct title one here"}
    b = {"url": "https://y.org/2", "doi": "https://doi.org/10.1/A", "title": "Totally other words entirely"}
    assert len(_finalize([a, b], "x", 10)) == 1
    merged = rrf_merge({"p1": [A], "p2": [B]}, 60, 10)
    assert len(merged) == 1 and merged[0]["merged_from"][0]["reason"] == "title_family+year"
    assert merged[0]["found_by_providers"] == ["p1", "p2"]


# ---------------------------------------------------------------------------------------------- R72-e
class R:
    def __init__(self, text):
        self.status_code, self.text, self.headers = 200, text, {}


def oa_text(n):
    return json.dumps({"results": [{"id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/10.1/x{i}",
                                    "title": f"Distinct paper title number {i}", "publication_year": 2024,
                                    "publication_date": "2024-01-01", "type": "article"} for i in range(n)]})


def arxiv_text(n):
    ent = "".join(f"<entry><id>http://arxiv.org/abs/2401.0000{i}v1</id><title>Paper {i} title long enough</title>"
                  f"<summary>s</summary><published>2024-01-01T00:00:00Z</published>"
                  f"<author><name>Ada Lovelace</name></author></entry>" for i in range(n))
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{ent}</feed>'


def run(providers):
    return search_question(Q, 5, providers=providers, mode="fan_out", sleep=lambda s: None, backoff_s=0)


def test_openalex_relaxes_once_on_zero_hits():
    seen = []

    def tx(url, params, headers, timeout):
        seen.append(params["search"])
        return R(oa_text(0 if len(seen) == 1 else 2))
    p = OpenAlexProvider(CFG["openalex"], transport=tx, query_terms=QT)
    res = run([p])
    assert len(seen) == 2 and " AND " in seen[0] and " OR " in seen[1]
    assert seen[1].count('"') == seen[0].count('"')               # same core phrases and extra terms
    assert res.meta["relaxed"] == {"openalex": True} and len(res) == 2
    assert p.last_query["relaxed"] is True and p.relaxed is False


def test_arxiv_relaxes_once_and_not_when_hits_or_disabled():
    seen = []

    def tx(url, params, headers, timeout):
        seen.append(params["search_query"])
        return R(arxiv_text(0 if len(seen) == 1 else 1))
    res = run([ArxivProvider(CFG["arxiv"], transport=tx, query_terms=QT)])
    assert len(seen) == 2 and " OR " in seen[1] and res.meta["relaxed"] == {"arxiv": True}
    # hits on the first call: no retry
    seen.clear()
    res = run([ArxivProvider(CFG["arxiv"], transport=lambda u, p, h, t: (seen.append(1), R(arxiv_text(1)))[1],
                             query_terms=QT)])
    assert len(seen) == 1 and res.meta["relaxed"] == {"arxiv": False}
    # flag off: zero stays zero, one call
    seen.clear()
    off = {**QT, "relax_on_zero": False}
    p = OpenAlexProvider(CFG["openalex"], transport=lambda u, pr, h, t: (seen.append(1), R(oa_text(0)))[1],
                         query_terms=off)
    res = run([p])
    assert len(seen) == 1 and res.meta["relaxed"] == {"openalex": False} and len(res) == 0


def test_relaxed_retry_failure_is_recorded_not_raised_and_budget_counts():
    calls = []

    def tx(url, params, headers, timeout):
        calls.append(1)
        if len(calls) == 2:
            raise requests.ConnectionError("boom")
        return R(oa_text(0))
    p = OpenAlexProvider(CFG["openalex"], transport=tx, query_terms=QT)
    items = sf._attempts(p, Q, 5, ledger=None, max_retries=1, backoff_s=0, deadline_at=1e12, timeout_s=1,
                         sleep=lambda s: None, clock=lambda: 0.0)
    assert items == [] and "network error" in p.last_relax_error and p.relaxed is False
    assert p._used == 2                                           # the retry spent the daily allowance


def test_builders_and_validator():
    base, _ = build_openalex_search(Q, QT)
    relaxed, _ = build_openalex_search(Q, QT, relaxed=True)
    assert " AND " in base and relaxed != base
    sq, _ = build_arxiv_query_info(Q, CFG["arxiv"]["query"], QT, relaxed=True)
    assert " OR " in sq
    assert QT["relax_on_zero"] is True


# ---------------------------------------------------------------------------------------------- R72-f
def test_labels_shared_helper_in_capture_and_m4():
    assert source_label({"source_type": "news"}) == NEWS_LABEL
    assert source_label({"source_class": "vendor_claim", "source_type": "news"}) == VENDOR_LABEL
    assert source_label({"source_type": "rct"}) == ""
    txt = _format_sources([{"n": 1, "title": "T", "url": "u", "snippet": "s", "source_type": "news"}])
    assert NEWS_LABEL + " T" in txt
    corpus = {"run_id": "r", "question": "q", "spans": [
        {"span_id": "s1", "text": "x", "source_title": "A", "source_type": "news", "sub_question_ids": ["sq_controlled"]},
        {"span_id": "s2", "text": "y", "source_title": "B", "source_type": "rct", "source_class": "vendor_claim",
         "sub_question_ids": ["sq_controlled"]},
        {"span_id": "s3", "text": "z", "source_title": "C", "source_type": "rct", "sub_question_ids": ["sq_controlled"]},
        {"span_id": "s4", "text": "w", "source_type": "news"}]}
    ctx = build_reasoning_context(corpus, {})
    assert f"| {NEWS_LABEL} A, news |" in ctx and f"| {VENDOR_LABEL} B, rct |" in ctx and "| C, rct |" in ctx
    assert f"span_id=s4 | {NEWS_LABEL} |" in ctx


def test_shipped_config_has_relaxation_and_doaj_off_and_chain_skips_doaj():
    from pathlib import Path
    from vera.search_and_fetch import build_chain
    assert QT_SHIPPED["relax_on_zero"] is False
    assert CFG["doaj"]["enabled"] is False
    root = Path(__file__).resolve().parent.parent
    assert "doaj" not in [p.name for p in build_chain(CFG, repo_root=root)]
