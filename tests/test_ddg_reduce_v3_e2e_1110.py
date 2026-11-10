"""DDG-REDUCE-V3-1110 G1: fallback -> admitted and cited end to end with the SHIPPED switches (its own module: it uses the
ROUTE-R2 `rig` fixture, which the other v3 module does not)."""
import pytest

from vera import ddg_reduce, inline_acquire

# ------------------------------------------------------------------------------------------------------ G1 end to end
from tests.test_route_ddg_first_r2 import (  # noqa: E402,F401  (fixtures: h, rig, clock, no_db_no_socket, world)
    DDG, EPMC, OA, _grounded, _need_portals, _ro, canon, clock, ddg_row, epmc_row, h, no_db_no_socket, oa_rec, rig, world)
from vera import arxiv_fulltext, auto_acquire  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_queue(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def test_g1_fallback_to_admitted_and_cited_with_the_shipped_switches(world):
    """The SHIPPED config (no patch of ddg_reduce.load): the scholarly routes return too little (OpenAlex 1 + Europe PMC 0
    < 3), so DuckDuckGo is called AFTER them as the fallback; its DOI-bearing result reaches the Europe PMC full text, which
    is admitted and cited in the same request (the assertions of route_ddg_first_r2 test_9 other than the call order)."""
    assert ddg_reduce.fallback_only() and ddg_reduce.on("answered_question_no_ddg")
    doi, pmc = "10.9999/v3.g1", "PMC9310101"
    world.doc(doi, pmc)
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{doi}/full")]
    world.p.beh[OA] = [oa_rec(doi, "W93101", pmc)]
    world.p.beh[EPMC] = []
    world.store()
    body = world.ask()
    assert DDG in world.p.log and world.p.log[0] != DDG and world.p.log[-1] == DDG, world.p.log   # fallback, last
    _grounded(body)
    ia = body["contract"]["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["persisted"] is True
    assert len(world.rig.store.plans) == 1 and len(world.rig.h.answer_calls) == 1
    ids = {a["document_id"] for a in body["admitted_evidence"]}
    assert ids and all(i.startswith("acq-") for i in ids) and {c["document_id"] for c in body["citations"]} == ids
    ro = _ro(body)
    assert ro["kind"] == "answered_with_new" and ro["complete"] is True
    assert "ddg" in [canon(p["portal"]) for p in _need_portals(body)]
