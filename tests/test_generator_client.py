"""FREEGEN-1028: config-driven generator, shared client factory, fail-closed cost. No network, no real keys."""
import json

import pytest

from vera import generator_client as gc
from vera.eval_config import EvalConfigError, load_eval_config, family_for as eval_family_for
from vera.pricing.config import ModelPricing, PricingRecord


def _run_cfg(tmp_path, generator):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"generator": generator}), encoding="utf-8")
    return p


def test_defaults_unchanged():
    s = gc.generator_settings()
    assert (s.model, s.base_url, s.api_key_env) == ("gpt-4.1-nano", None, "OPENAI_API_KEY")
    assert s.is_default_provider
    from vera import gate_a, pipeline_llm
    from vera.m3 import llm as m3llm
    from vera.m6 import baseline_collection as bc
    assert gate_a.MODEL == bc.DEFAULT_MODEL == pipeline_llm.FALLBACK_MODEL == m3llm.DEFAULT_MODEL == "gpt-4.1-nano"


def test_default_provider_uses_existing_client(monkeypatch):
    sentinel = object()
    import vera.ask_service as ask
    monkeypatch.setattr(ask, "_get_client", lambda: sentinel)
    assert gc.make_client() is sentinel


class FakeClient:
    def __init__(self, **kw):
        self.kw = kw


def test_config_override_routes_to_base_url_and_model(tmp_path):
    p = _run_cfg(tmp_path, {"model": "free-model-x", "base_url": "https://example.invalid/openai/v1",
                            "api_key_env": "FAKE_GEN_KEY"})
    s = gc.generator_settings(p)
    assert s.model == "free-model-x" and not s.is_default_provider
    c = gc.make_client(s, env={"FAKE_GEN_KEY": "k-test"}, client_cls=FakeClient, timeout=5)
    assert c.kw["base_url"] == "https://example.invalid/openai/v1"
    assert c.kw["api_key"] == "k-test" and c.kw["timeout"] == 5


def test_missing_key_env_names_var_not_value(tmp_path):
    s = gc.generator_settings(_run_cfg(tmp_path, {"model": "m", "base_url": "https://e.invalid/v1",
                                                  "api_key_env": "FAKE_GEN_KEY"}))
    with pytest.raises(gc.GeneratorConfigError, match="FAKE_GEN_KEY"):
        gc.make_client(s, env={}, client_cls=FakeClient)


@pytest.mark.parametrize("g", [{}, {"model": ""}, {"model": "m", "base_url": "http://insecure"},
                               {"model": "m", "api_key_env": ""}])
def test_bad_generator_block_rejected(tmp_path, g):
    with pytest.raises(gc.GeneratorConfigError):
        gc.generator_settings(_run_cfg(tmp_path, g))


def test_unknown_model_cost_raises_everywhere():
    with pytest.raises(gc.UnpricedModelError):
        gc.cost_usd("no-such-model", 10, 10)
    from vera.m6.baseline_collection import _cost
    with pytest.raises(gc.UnpricedModelError):
        _cost("no-such-model", 10, 10)
    from vera import pipeline_llm
    with pytest.raises(gc.UnpricedModelError):
        pipeline_llm.complete_json("s", "u", model="no-such-model")


def test_free_model_with_zero_record_costs_zero(monkeypatch):
    rec = PricingRecord(provider="p", model="free-x", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-10-28T00:00:00", effective_from="2026-10-28")
    monkeypatch.setattr(gc, "load_model_pricing", lambda: ModelPricing(records=[rec]))
    assert gc.cost_usd("free-x", 1000, 500) == 0.0


def test_generator_family_none_or_unknown_raises():
    cfg = load_eval_config()
    with pytest.raises(EvalConfigError, match="fail closed"):
        eval_family_for("unmapped-model", cfg)


def test_generator_family_none_raises_explicitly(monkeypatch):
    from vera.eval_config import FamilyResolution
    from vera.m6 import judge
    monkeypatch.setattr(judge, "family_for", lambda m, c, **k: FamilyResolution(m, None, "map"))
    with pytest.raises(judge.JudgeConfigError, match="no family"):
        judge.generation_family(load_eval_config())
