"""Judge provider selection + measurement. No real API calls (requests.post is mocked)."""
import json

import pytest
import requests

from tests.config_support import add_models, load_pair
from vera.cost_ledger import CostLedger
from vera.eval_config import EvalConfigError
from vera.m6.judge import (JudgeConfigError, JudgeLimits, OpenAICompatJudge,
                           build_default_judge, preflight_judge)

TEST_MODEL = "qwen/qwen3-32b"  # test-only model id, mapped through a temporary config (never the real one)


class _Resp:
    ok, status_code, text = True, 200, ""

    def __init__(self, content='{"score": 4}', tin=100, tout=20):
        self._d = {"choices": [{"message": {"content": content}}],
                   "usage": {"prompt_tokens": tin, "completion_tokens": tout}}

    def json(self):
        return self._d


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("VERA_JUDGE_MODEL", "VERA_JUDGE_GROQ_MODEL", "VERA_JUDGE_OPENAI_MODEL", "VERA_JUDGE_PRICE_IN",
              "VERA_JUDGE_PRICE_OUT", "VERA_JUDGE_ALLOW_SAME_FAMILY", "VERA_JUDGE_FAMILY_OVERRIDE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g-test")
    monkeypatch.setenv("MISTRAL_API_KEY", "m-test")
    monkeypatch.setenv("OPENAI_API_KEY", "o-test")


@pytest.fixture
def cfg(tmp_path):
    """A temporary config whose map knows the test model (qwen family) and a llama model."""
    return load_pair(tmp_path, run_edit=add_models({TEST_MODEL: "qwen", "llama-3.3-70b-versatile": "meta-llama"}))


def test_no_default_fails_verbosely_naming_the_shutdown():
    with pytest.raises(JudgeConfigError, match="mixtral-8x7b-32768.*2025-03-20.*VERA_JUDGE_GROQ_MODEL"):
        build_default_judge()


def test_env_model_recorded_as_env_source(monkeypatch, cfg):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    j = build_default_judge(config=cfg)
    assert isinstance(j, OpenAICompatJudge)
    assert (j.provider, j.model, j.family) == ("groq", TEST_MODEL, "qwen")
    assert (j.model_source, j.family_source) == ("env", "map")


def test_llama_on_groq_is_not_labelled_mistral(monkeypatch, cfg):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "llama-3.3-70b-versatile")
    j = build_default_judge(config=cfg)
    assert j.family == "meta-llama" and j.family_source == "map"


def test_unmapped_model_refused_unless_override_disclosed(monkeypatch, cfg):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "unmapped/model-x")
    with pytest.raises(JudgeConfigError, match="not in model_families.*fail closed"):
        build_default_judge(config=cfg)
    monkeypatch.setenv("VERA_JUDGE_FAMILY_OVERRIDE", "Somefamily")
    j = build_default_judge(config=cfg)
    assert j.family == "somefamily" and j.family_source == "override"


def test_gpt_name_contradicting_the_map_is_refused(monkeypatch, tmp_path):
    cfg = load_pair(tmp_path, run_edit=add_models({"vendor/gpt-fake": "notopenai"}))
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "vendor/gpt-fake")
    with pytest.raises(JudgeConfigError, match="contradicts the id"):
        build_default_judge(config=cfg)


def test_generator_family_derived_from_config_not_constant(monkeypatch, tmp_path):
    """A config whose generator is a non-OpenAI mapped model makes the OpenAI judge acceptable."""
    import json
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_model": "gen-other"}))

    def edit(r):
        r["generator"]["model"] = "gen-other"
        r["model_families"]["gen-other"] = {"family": "other", "source": "https://example.invalid/test-only"}
        r["model_families"]["gpt-test-judge"] = {"family": "openai", "source": "https://example.invalid/test-only"}
    cfg = load_pair(tmp_path, run_edit=edit, model_selection_path=sel)
    monkeypatch.setenv("VERA_JUDGE_MODEL", "openai")
    monkeypatch.setenv("VERA_JUDGE_OPENAI_MODEL", "gpt-test-judge")
    j = build_default_judge(config=cfg)
    assert j.family == "openai" and j.same_family_override is False
    # and with the real config (generator gpt-4.1-nano = openai) the same judge is refused
    with pytest.raises(JudgeConfigError, match="not in model_families"):
        build_default_judge()


def test_openai_judge_refused_by_default_same_family(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_MODEL", "openai")
    monkeypatch.setenv("VERA_JUDGE_OPENAI_MODEL", "gpt-4.1-nano")
    with pytest.raises(JudgeConfigError, match="different model family"):
        build_default_judge()


def test_openai_judge_allowed_only_with_explicit_override_and_disclosed(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_MODEL", "openai")
    monkeypatch.setenv("VERA_JUDGE_ALLOW_SAME_FAMILY", "1")
    monkeypatch.setenv("VERA_JUDGE_OPENAI_MODEL", "gpt-4.1-nano")
    assert build_default_judge().same_family_override is True


def test_groq_hosted_openai_family_model_is_still_refused(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "openai/gpt-oss-20b")
    with pytest.raises(JudgeConfigError, match="different model family"):
        build_default_judge()


def test_unknown_selector_and_missing_key_fail_verbosely(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_MODEL", "bogus")
    with pytest.raises(JudgeConfigError, match="not a known judge provider"):
        build_default_judge()
    monkeypatch.setenv("VERA_JUDGE_MODEL", "anthropic")  # removed (R1: no Anthropic key ever)
    with pytest.raises(JudgeConfigError, match="not a known judge provider") as ei:
        build_default_judge()
    assert "ANTHROPIC" not in str(ei.value).upper().replace("'ANTHROPIC'", "")
    monkeypatch.setenv("VERA_JUDGE_MODEL", "mistral")  # no longer a provider (R1: never worked)
    with pytest.raises(JudgeConfigError, match="not a known judge provider"):
        build_default_judge()
    monkeypatch.setenv("VERA_JUDGE_MODEL", "groq")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(JudgeConfigError, match="No judge model configured"):  # the model error comes first
        build_default_judge()
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "openai/gpt-oss-20b")
    with pytest.raises(JudgeConfigError, match="different model family"):
        build_default_judge()
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    with pytest.raises(JudgeConfigError, match="not in model_families"):  # real config does not map the test model
        build_default_judge()


def test_groq_call_measured_and_ledgered(monkeypatch, cfg):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, auth=headers["Authorization"], model=json["model"], max_tokens=json["max_tokens"])
        return _Resp(tin=1_000_000, tout=500_000)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    ledger = CostLedger()
    j = build_default_judge(JudgeLimits(max_cost_usd=10), ledger=ledger, config=cfg)
    assert j.judge_json("relevance", "prompt") == {"score": 4}
    assert "api.groq.com" in seen["url"] and seen["auth"] == "Bearer g-test"
    from vera.m6.judge import JUDGE_MAX_TOKENS
    assert seen["model"] == TEST_MODEL and seen["max_tokens"] == JUDGE_MAX_TOKENS
    e = ledger.entries[0]
    assert e.kind == "llm_judge" and e.provider == f"groq:{TEST_MODEL}"
    assert e.units["tokens_in"] == 1_000_000 and e.units["tokens_out"] == 500_000
    assert e.units["latency_s"] >= 0
    assert e.cost_usd == pytest.approx(0.24 + 0.12)
    assert j.usage.log[0]["model"] == TEST_MODEL
    assert j.usage.cost_usd == pytest.approx(e.cost_usd)
    assert j.price_source == "assumed_default"


def test_price_env_override_recorded_as_env_source(monkeypatch, cfg):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    monkeypatch.setenv("VERA_JUDGE_PRICE_IN", "1.5")
    j = build_default_judge(config=cfg)
    assert j.price_source == "env" and j.price_in == 1.5


def test_http_error_raises_verbosely_without_retry(monkeypatch, cfg):
    calls = []

    class Bad(_Resp):
        ok, status_code, text = False, 400, "model_decommissioned"

    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or Bad())
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    with pytest.raises(RuntimeError, match="HTTP 400.*model_decommissioned"):
        build_default_judge(config=cfg).judge_json("p", "x")
    assert len(calls) == 1


# ---- A/B comparison fixture (post-MVP): same prompts through two judge models, compare outputs + cost
@pytest.fixture
def ab_compare(monkeypatch, cfg):
    """Returns run(prompts, replies_by_model) -> {model: {outputs, usage, ledger_total}}."""
    def run(prompts, replies_by_model):
        out = {}
        for model, replies in replies_by_model.items():
            monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", model)
            it = iter(replies)
            monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(json.dumps(next(it))))
            ledger = CostLedger()
            j = build_default_judge(ledger=ledger, config=cfg)
            outs = [j.judge_json(f"q{i}", p) for i, p in enumerate(prompts)]
            out[model] = {"outputs": outs, "usage": j.usage.as_dict(),
                          "ledger_total": ledger.total_usd(), "model": j.model}
        return out
    return run


def test_ab_fixture_compares_two_judges(ab_compare):
    r = ab_compare(["a", "b"], {TEST_MODEL: [{"score": 4}, {"score": 3}],
                                "llama-3.3-70b-versatile": [{"score": 4}, {"score": 5}]})
    a, b = r[TEST_MODEL], r["llama-3.3-70b-versatile"]
    assert [x == y for x, y in zip(a["outputs"], b["outputs"])] == [True, False]
    assert a["usage"]["api_calls"] == b["usage"]["api_calls"] == 2
    assert a["model"] != b["model"]


# ---- preflight_judge: listing check before any spend; fake http_get only, never the network
class _Listing:
    def __init__(self, ids=(), ok=True, status=200, text=""):
        self._ids, self.ok, self.status_code, self.text = ids, ok, status, text

    def json(self):
        return {"data": [{"id": i} for i in self._ids]}


@pytest.fixture
def groq_judge(monkeypatch, cfg):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", TEST_MODEL)
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-value-123")
    return build_default_judge(config=cfg)


def test_preflight_ok_returns_record(groq_judge):
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"], timeout=timeout)
        return _Listing([TEST_MODEL, "other"])
    rec = preflight_judge(groq_judge, http_get=get)
    assert rec["status"] == "ok" and rec["model"] == TEST_MODEL and rec["listed_models"] == 2
    assert seen["url"] == "https://api.groq.com/openai/v1/models" and seen["timeout"] == 15


def test_preflight_unlisted_model_raises_naming_the_model(groq_judge):
    with pytest.raises(JudgeConfigError, match="does not list model 'qwen/qwen3-32b'.*Nothing was spent"):
        preflight_judge(groq_judge, http_get=lambda *a, **k: _Listing(["other"]))


def test_preflight_http_error_and_exception_verbose_without_the_key(groq_judge):
    with pytest.raises(JudgeConfigError, match="HTTP 401.*unauthorized") as ei:
        preflight_judge(groq_judge, http_get=lambda *a, **k: _Listing(ok=False, status=401, text="unauthorized"))
    assert "secret-key-value-123" not in str(ei.value)

    def boom(*a, **k):
        raise ConnectionError("boom secret-key-value-123")
    with pytest.raises(JudgeConfigError, match="cannot list groq models.*ConnectionError") as ei:
        preflight_judge(groq_judge, http_get=boom)
    assert "secret-key-value-123" not in str(ei.value)


# ---- F4: override can never turn an openai-provider or gpt-named judge into another family
@pytest.mark.parametrize("model", ["chatgpt-4o-latest", "o3-mini", "gpt-4.1-nano"])
def test_openai_provider_refuses_any_family_override(monkeypatch, model):
    monkeypatch.setenv("VERA_JUDGE_MODEL", "openai")
    monkeypatch.setenv("VERA_JUDGE_OPENAI_MODEL", model)
    monkeypatch.setenv("VERA_JUDGE_FAMILY_OVERRIDE", "qwen")
    with pytest.raises(JudgeConfigError, match="refused for the openai provider"):
        build_default_judge()


@pytest.mark.parametrize("model", ["vendor/chatgpt-x", "vendor/my-gpt-4", "acme/openai-clone"])
def test_override_cannot_contradict_gpt_or_openai_in_the_id(monkeypatch, model):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", model)
    monkeypatch.setenv("VERA_JUDGE_FAMILY_OVERRIDE", "qwen")
    with pytest.raises(JudgeConfigError, match="contradicts the id"):
        build_default_judge()


def test_openai_provider_unmapped_non_gpt_name_without_override_fails_closed(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_MODEL", "openai")
    monkeypatch.setenv("VERA_JUDGE_OPENAI_MODEL", "o3-mini")
    with pytest.raises(JudgeConfigError, match="not in model_families"):
        build_default_judge()


# ---- F12: no Anthropic judge or key anywhere
def test_no_anthropic_judge_or_key_text_in_judge_module(monkeypatch):
    import inspect

    import vera.m6.judge as jm
    assert not hasattr(jm, "AnthropicJudge")
    assert "anthropic" not in inspect.getsource(jm).lower()
