"""DAR-PIPE-1022-ENABLEREADY: the shipped config enables inline.use_discovered (owner D3/D5, 2026-10-23).
Proves, offline, on the ENABLED shipped file: (1) only that flag changed; (2) an eligible discovered record is acquired;
(3) the kill switch VERA_AUTOACQ_DISABLED stops discovery hand-off, every fetch and every store write; (4) setting
use_discovered back to false restores the reading-list behaviour; (5) main.py serves the discovery hand-off from the
per-request search cache (no extra provider search). Fakes: scope decision, HTTP, store, answer model (tests/ rigs)."""
import pytest
from fastapi.testclient import TestClient

from tests import test_fulldoc_e2e as e2e
from tests.test_acq_discovered import DOI, epmc_web, oa_src
from tests.test_acq_serve import _decide, h  # noqa: F401  (h is a fixture)
from tests.test_dar_inline_acquire import GROUND, OFF_Q, rig  # noqa: F401  (rig is a fixture)
from vera import acq_config, acq_discovered, ask_service


class Disc:
    """The request's discovery search seam (grounding[0]); returns one eligible OA record."""

    def __init__(self, srcs):
        self.srcs, self.calls = srcs, []

    def __call__(self, restricted=False):
        self.calls.append(restricted)
        return list(self.srcs)


def _ask(rig, monkeypatch):
    web, _raw = epmc_web()
    rig.setup(web=web, cands=[])  # the acquisition's own search finds nothing: only the discovered record can be acquired
    spy = []
    real = acq_discovered.discovery_results
    monkeypatch.setattr(acq_discovered, "discovery_results", lambda *a, **k: spy.append(1) or real(*a, **k))
    _decide(monkeypatch, False, "layer2_llm", "llm_out:x", "off_topic")
    ds = Disc([oa_src()])
    r = ask_service.answer_in_scope(OFF_Q, "m", e2e.PRICING, grounding=(ds, GROUND[1]))
    return r, web, spy


def _fetched(web, frag):
    return any(frag in c[0] for c in web.calls)


def test_shipped_config_enables_use_discovered_and_nothing_else():
    c = acq_config.load_config()  # the real file
    assert c["inline"]["use_discovered"] is True  # owner D3/D5 2026-10-23
    assert c["inline"]["landing_route"] is False and c["landing_route"]["publisher_hosts"] == []
    assert c["fetch"]["allowed_hosts"] == ["www.ebi.ac.uk", "eutils.ncbi.nlm.nih.gov"]
    assert c["inline"]["enabled"] is True and c["enabled"] is True


def test_enabled_build_acquires_an_eligible_discovered_record(rig, monkeypatch):
    assert rig.cfg["inline"]["use_discovered"] is True  # the rig's copy of the SHIPPED file, not an override
    r, web, spy = _ask(rig, monkeypatch)
    assert spy == [1]
    assert _fetched(web, "fullTextXML") and len(rig.store.plans) == 1
    assert [e["outcome"] for e in r.contract["inline_acquisition"]["discovered"]] == ["admitted"]


@pytest.mark.parametrize("value", ["1", "true"])
def test_kill_switch_on_enabled_build_stops_discovery_fetch_and_db_writes(rig, monkeypatch, value):
    assert rig.cfg["inline"]["use_discovered"] is True
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", value)
    r, web, spy = _ask(rig, monkeypatch)
    assert spy == []                          # E5: no discovery hand-off
    assert web.calls == []                    # no Europe PMC search, no full-text fetch
    assert rig.store.plans == []              # no store write (the INSERT-only WoStore is never reached)
    assert rig.search_calls == []             # the acquisition's own search is not run either
    assert r.result_kind != "answer" and rig.h.answer_calls == []
    assert "discovered" not in ((r.contract or {}).get("inline_acquisition") or {})


def test_flag_false_restores_the_reading_list_behaviour(rig, monkeypatch):
    rig.cfg["inline"]["use_discovered"] = False  # rollback config value (override on the rig's copy)
    r, web, spy = _ask(rig, monkeypatch)
    assert spy == []
    assert not _fetched(web, DOI) and not _fetched(web, "fullTextXML")
    assert rig.store.plans == [] and rig.h.answer_calls == []
    assert r.result_kind == "reading_list"
    assert "discovered" not in ((r.contract or {}).get("inline_acquisition") or {})


def test_main_serves_the_discovery_handoff_from_the_per_request_cache(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    calls = []
    srcs = [{"n": 1, "title": "T", "url": "https://example.org/a", "snippet": "abs"}]
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg, **k: calls.append(1) or (srcs, []))

    def chain(q, e, g):
        for restricted in (True, True, False, False):  # reading list + hand-off, off-topic and in-scope
            acq_discovered.discovery_results(g, restricted=restricted)
        raise RuntimeError("no provider available")
    monkeypatch.setattr(main, "answer_via_chain", chain)
    TestClient(main.app).post("/ask", json={"question": "q?"}, headers={"X-API-Key": "k" * 20})
    assert len(calls) == 2  # one restricted + one full search per request, however often discovery reads them
