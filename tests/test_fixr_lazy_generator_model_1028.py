"""FIXR-7DA-1028 (VER-REL7DA-1-1028 #3): generator.model is read at first use, not at import, so the /ask modules
import (and main.py starts) when config/vera_eval_run.json is absent. Fresh copies of the modules are executed under
private names, so the already-imported modules are untouched. Offline only."""
import importlib.util
import sys
from pathlib import Path

import pytest

from vera import generator_client as gc

ROOT = Path(__file__).resolve().parent.parent
LAZY = {"vera/gate_a.py": "MODEL", "vera/pipeline_llm.py": "FALLBACK_MODEL", "vera/m3/llm.py": "DEFAULT_MODEL",
        "vera/m6/baseline_collection.py": "DEFAULT_MODEL"}


def _fresh(rel, monkeypatch):
    name = "fixr_lazy_" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def no_run_config(monkeypatch, tmp_path):
    # the uncached reader against a missing file; the real cached value (shared object identity) is left untouched
    monkeypatch.setattr(gc, "RUN_CONFIG_PATH", tmp_path / "missing-vera_eval_run.json")
    monkeypatch.setattr(gc, "default_generator_model", lambda: gc.generator_model())


@pytest.mark.parametrize("rel,name", sorted(LAZY.items()))
def test_module_imports_without_run_config_and_fails_only_at_first_use(no_run_config, monkeypatch, rel, name):
    m = _fresh(rel, monkeypatch)  # import-time read would raise GeneratorConfigError here
    with pytest.raises(gc.GeneratorConfigError):
        getattr(m, name)


def test_gate_a_production_admission_without_run_config_defers_not_raises(no_run_config, monkeypatch):
    from tests.test_item79_corpus_admission import _llm
    g = _fresh("vera/gate_a.py", monkeypatch)
    rows = g.score_candidates("q", [{"url": "https://example.invalid/a", "title": "t", "snippet": "s"}],
                              _llm([0.9]), safe_cost=True)
    assert [r["gate_a_decision"] for r in rows] == ["defer"]


def test_lazy_value_equals_config_and_is_cached():
    from vera import gate_a, pipeline_llm
    from vera.m3 import llm as m3llm
    from vera.m6 import baseline_collection as bc
    want = gc.generator_model()
    assert gate_a.MODEL == pipeline_llm.FALLBACK_MODEL == m3llm.DEFAULT_MODEL == bc.DEFAULT_MODEL == want
    assert gate_a.MODEL is bc.DEFAULT_MODEL  # one cached object
    assert pipeline_llm.LLMResult({}).model == want


def test_explicitly_set_module_attribute_still_wins(monkeypatch):
    from vera import gate_a
    monkeypatch.setattr(gate_a, "MODEL", "override-model-x", raising=False)
    assert gate_a._model() == "override-model-x"
