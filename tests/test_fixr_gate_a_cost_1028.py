"""FIXR-7DA-1028 (VER-REL7DA-1-1028 #2): an unpriced Gate A cost fails closed SAFELY on the production /ask admission
path (every candidate deferred, class name logged, no exception) and still raises on eval paths. Offline only."""
import json
import logging

import pytest

from vera import corpus_admission as ca, gate_a
from vera.cost_ledger import CostLedger
from vera.generator_client import UnpricedModelError
from tests.test_item79_corpus_admission import CFG, FakeStore, _llm, _src


@pytest.fixture
def unpriced(monkeypatch):
    def boom(pt, ct):
        raise UnpricedModelError("No pricing record for 'x'")
    monkeypatch.setattr(gate_a, "_llm_cost", boom)


def _cands(n=3):
    return [{"url": f"https://example.invalid/{i}", "title": f"t{i}", "snippet": "s"} for i in range(n)]


def test_eval_path_still_raises_on_unpriced_cost(unpriced):
    with pytest.raises(UnpricedModelError):
        gate_a.score_candidates("q", _cands(), _llm([0.9, 0.9, 0.9]))


def test_safe_cost_defers_every_candidate_and_logs_the_class_name(unpriced, caplog):
    with caplog.at_level(logging.WARNING, logger="vera"):
        rows = gate_a.score_candidates("q", _cands(), _llm([0.9, 0.9, 0.9]), safe_cost=True)
    assert [r["gate_a_decision"] for r in rows] == ["defer"] * 3 and all(r["gate_a_score"] is None for r in rows)
    assert "UnpricedModelError" in caplog.text and "No pricing record" not in caplog.text


def test_safe_cost_records_an_unknown_cost_in_the_ledger(unpriced):
    led = CostLedger()
    gate_a.score_candidates("q", _cands(1), _llm([0.9]), ledger=led, safe_cost=True)
    assert led.unknown_cost_calls() == 1 and led.total_usd() == 0.0


def test_production_admission_defers_on_unpriced_cost_without_raising(unpriced, caplog):
    store = FakeStore()
    with caplog.at_level(logging.WARNING, logger="vera"):
        counts = ca.admit("visitor question", [_src(1, "https://arxiv.org/abs/1"), _src(2, "https://arxiv.org/abs/2")],
                          _llm([0.9, 0.9]), CFG, store)
    assert counts == {"recorded_defer": 2} and "error" not in counts
    assert {r["gate_a_decision"] for r in store.rows} == {"defer"}
    assert "UnpricedModelError" in caplog.text
    assert json.dumps(counts)


def test_priced_cost_unchanged_on_both_paths():
    a = gate_a.score_candidates("q", _cands(), _llm([0.9, 0.5, 0.1]))
    b = gate_a.score_candidates("q", _cands(), _llm([0.9, 0.5, 0.1]), safe_cost=True)
    assert [r["gate_a_decision"] for r in a] == [r["gate_a_decision"] for r in b] == ["fetch", "defer", "reject"]
