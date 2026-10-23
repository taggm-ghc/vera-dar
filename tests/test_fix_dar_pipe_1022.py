"""DAR-PIPE-1022 fixes (E5 hand-off check, PMCID passthrough, abstract-less originals wiring). Offline, no network/DB.
The end-to-end behaviour is in work/dar-pipeline-1022/tests/test_dar_pipeline_acceptance.py (FIX_* tests); not duplicated."""
import pytest
from fastapi.testclient import TestClient

from vera import acq_discovered, grounded_ask, inline_acquire


def test_discovery_results_adds_unused_in_scope_and_off_topic():
    def fn(restricted=False):
        return [{"url": "https://a/x", "snippet": "s"}]
    fn.unused = lambda restricted=False: [{"url": f"https://u/{restricted}", "doi": "10.1/x"}]
    assert [s["url"] for s in acq_discovered.discovery_results((fn, {}), restricted=False)] == ["https://a/x", "https://u/False"]
    assert acq_discovered.discovery_results((fn, {}), restricted=True)[-1]["url"] == "https://u/True"


@pytest.mark.parametrize("restricted", [False, True])
def test_unused_failure_never_loses_the_main_results(restricted):
    def fn(restricted=False):
        return [{"url": "https://a/x", "snippet": "s"}]

    def boom(restricted=False):
        raise RuntimeError("boom")
    fn.unused = boom
    assert acq_discovered.discovery_results((fn, {}), restricted=restricted) == [{"url": "https://a/x", "snippet": "s"}]


def test_unused_entry_without_abstract_carries_identifiers_only_internally():
    linked = [{"title": "L", "url": "https://example.org/l", "snippet": ""}]
    orig = [{"title": "O", "url": "https://example.org/o", "snippet": "", "doi": "10.1/abc", "pmcid": "PMC123456",
             "openalex_id": "W1", "discovered_via": "openalex"}]
    cfg = {"num_results": 5, "snippet_max_chars": 1500, "providers": [], "deadline_s": 10,
           "resolve": {"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5}}
    out, unused = grounded_ask.retrieve_all("q", cfg, search=lambda q: linked, resolver=lambda r, c, d: orig)
    assert out == [] and unused[0]["doi"] == "10.1/abc" and unused[0]["pmcid"] == "PMC123456"


def test_pmcid_is_copied_to_the_numbered_source_and_not_to_the_public_candidate():
    cfg = {"num_results": 5, "snippet_max_chars": 1500, "providers": [], "deadline_s": 10, "resolve": {"enabled": False}}
    raw = [{"title": "T", "url": "https://example.org/a", "snippet": "abs", "pmcid": "PMC123456", "provider": "openalex"}]
    out, _ = grounded_ask.retrieve_all("q", cfg, search=lambda q: raw)
    assert out[0]["pmcid"] == "PMC123456"
    assert "pmcid" not in str(grounded_ask.public_candidates(out)).lower()


def test_handoff_requires_inline_acquisition_enabled(monkeypatch):
    """ask_service._inline: use_discovered true but inline.enabled false -> no discovery search is made."""
    from vera import ask_service, fulldoc_corpus
    calls = []
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda c: True)
    monkeypatch.setattr(acq_discovered, "discovery_results", lambda *a, **k: calls.append(1) or [])
    monkeypatch.setattr(inline_acquire, "acquire", lambda *a, **k: type("O", (), {"status": "off", "corpus": None})())
    monkeypatch.setattr(inline_acquire, "notice", lambda acq, st: "NOTICE")
    for enabled, expected in ((False, 0), (True, 1)):
        calls.clear()
        monkeypatch.setattr(inline_acquire, "inline_enabled", lambda acq, env=None, e=enabled: e)
        acq = {"inline": {"use_discovered": True}}
        ask_service._inline("q", "m", None, None, None, (lambda restricted=False: [], {}), acq)
        assert len(calls) == expected


def test_main_exposes_unused_on_sources_fn_and_keeps_main_results(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    srcs = [{"n": 1, "title": "T", "url": "https://example.org/a", "snippet": "abs"}]
    unused = [{"title": "U", "url": "https://example.org/u", "doi": "10.1/u", "discovered_via": "openalex"}]
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg, **k: (srcs, unused))
    seen = {}

    def chain(q, e, g):
        fn = g[0]
        seen["s"], seen["u"] = fn(), fn.unused(False)
        seen["r"], seen["ru"] = fn(restricted=True), fn.unused(True)
        raise RuntimeError("no provider available")
    monkeypatch.setattr(main, "answer_via_chain", chain)
    TestClient(main.app).post("/ask", json={"question": "q?"}, headers={"X-API-Key": "k" * 20})
    assert seen["s"] == srcs and seen["u"] == unused
    assert seen["r"] == srcs and seen["ru"] == unused  # restricted search is stubbed to the same retrieve_all pair
