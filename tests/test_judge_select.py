"""S3 judge selection tests. FAKES ONLY: no network, no DB, no provider calls.
`requests.get/post` are patched to raise, so any accidental real call fails the test."""
import json

import pytest

from vera import judge_select as js
from vera.cost_ledger import CostLedger
from vera.m6.judge import JudgeConfigError, JudgeLimits

SECRET = "gsk_SENTINELSECRET1234567890"
ENV = {"GROQ_API_KEY": SECRET, "OPENAI_API_KEY": "sk-openaisentinel123456"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("real network call attempted in a test")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.delenv("VERA_JUDGE_ALLOW_SAME_FAMILY", raising=False)
    monkeypatch.delenv("VERA_JUDGE_GROQ_MODEL", raising=False)


def listing(*ids):
    def http_get(url, headers, timeout):
        assert url == js.LISTING_URL["groq"]
        return {"data": [{"id": i} for i in ids]}
    return http_get


class Transport:
    """Fake chat transport; `fail` model ids raise, `bad` ids return non-JSON."""
    def __init__(self, fail=(), bad=()):
        self.fail, self.bad, self.bodies = set(fail), set(bad), []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(body)
        if body["model"] in self.fail:
            raise RuntimeError(f"HTTP 404: model {body['model']} not found; key {SECRET}")
        text = "not json at all" if body["model"] in self.bad else '{"ok": true}'
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


LIVE = ("qwen/qwen3-32b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct",
        "gemma2-9b-it", "deepseek-chat-v3", "mistral-saba-24b")


# ---- family map -------------------------------------------------------------------------
@pytest.mark.parametrize("mid,fam", [
    ("qwen/qwen3-32b", "qwen"), ("llama-3.3-70b-versatile", "meta-llama"),
    ("moonshotai/kimi-k2-instruct", "moonshot"), ("deepseek-chat-v3", "deepseek"),
    ("gemma2-9b-it", "google"), ("mistral-saba-24b", "mistral"), ("mixtral-8x7b", "mistral")])
def test_pattern_map(mid, fam):
    assert js.family_for(mid)[0] == fam


@pytest.mark.parametrize("mid", ["openai/gpt-oss-120b", "gpt-4.1-nano", "GPT-OSS-20B", "openai/foo"])
def test_openai_names_are_openai_family(mid):
    assert js.family_for(mid)[0] == "openai"


@pytest.mark.parametrize("mid", ["deepseek-r1-distill-llama-70b", "llama-guard-4-12b",
                                 "whisper-large-v3", "playai-tts", "text-embed-1", "groq/compound-mini"])
def test_excluded_before_generic_patterns(mid):
    fam, why = js.family_for(mid)
    assert fam is None and "excluded" in why


def test_unknown_and_empty_and_ambiguous_refused():
    fam, why = js.family_for("zzz-model-9")
    assert fam is None and "unknown model family" in why and "refused" in why
    assert js.family_for("")[0] is None
    assert js.family_for("qwen-llama-merge")[0] is None  # ambiguous


# ---- candidates -------------------------------------------------------------------------
def test_candidates_filter_and_deterministic_order():
    ids = LIVE + ("openai/gpt-oss-120b", "deepseek-r1-distill-llama-70b", "whisper-large-v3", "zzz-9")
    c1, r1 = js.candidates(ENV, http_get=listing(*ids))
    c2, _ = js.candidates(ENV, http_get=listing(*reversed(ids)))
    assert [c.model for c in c1] == [c.model for c in c2]
    assert [c.family for c in c1] == ["qwen", "meta-llama", "moonshot", "deepseek", "google", "mistral"]
    refused = {r.model: r.reason for r in r1}
    assert set(refused) == {"openai/gpt-oss-120b", "deepseek-r1-distill-llama-70b", "whisper-large-v3", "zzz-9"}
    assert "unknown" in refused["zzz-9"]


def test_mistral_provider_not_in_default_pool():
    assert js.JUDGE_PROVIDER_ORDER == ("groq",)
    with pytest.raises(JudgeConfigError, match="not in the judge pool"):
        js.parse_pool("mistral:mistral-small-latest")
    with pytest.raises(JudgeConfigError, match="not in the judge pool"):
        js.parse_pool("openai:gpt-4.1-nano")


def test_pool_restricts_and_orders():
    c, r = js.candidates(ENV, http_get=listing(*LIVE),
                         pool_env="groq:gemma2-9b-it,groq:qwen/qwen3-32b,groq:not-live-model")
    assert [x.model for x in c] == ["gemma2-9b-it", "qwen/qwen3-32b"]
    assert any(x.model == "not-live-model" and "not in the live listing" in x.reason for x in r)


def test_pool_entry_that_is_openai_is_refused():
    c, r = js.candidates(ENV, http_get=listing("openai/gpt-oss-20b", "qwen/qwen3-32b"),
                         pool_env="groq:openai/gpt-oss-20b")
    assert c == [] and any("openai" in x.reason for x in r)


def test_key_missing_refuses_without_calling_listing():
    called = []
    c, r = js.candidates({}, http_get=lambda *a: called.append(1))
    assert c == [] and not called and "key absent" in r[0].reason


def test_empty_listing_error():
    with pytest.raises(JudgeConfigError, match="empty"):
        js.candidates(ENV, http_get=listing())


def test_listing_error_is_verbose_and_scrubbed():
    def bad(url, headers, timeout):
        raise RuntimeError(f"401 for key {SECRET}")
    with pytest.raises(JudgeConfigError) as e:
        js.candidates(ENV, http_get=bad)
    assert "listing failed" in str(e.value) and SECRET not in str(e.value)


def test_listing_bad_shape_and_no_transport():
    with pytest.raises(JudgeConfigError, match="unexpected shape"):
        js.candidates(ENV, http_get=lambda *a: {"nope": 1})
    with pytest.raises(JudgeConfigError, match="No http_get"):
        js.candidates(ENV, http_get=None)


def test_only_openai_live_means_no_candidates_and_select_fails():
    with pytest.raises(JudgeConfigError, match="No candidate judge"):
        js.select_judge(ENV, http_get=listing("openai/gpt-oss-120b"), judge_transport=Transport())


# ---- selection and canary ---------------------------------------------------------------
def test_select_picks_first_candidate_and_records():
    t, led = Transport(), CostLedger()
    sel = js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(), ledger=led, judge_transport=t)
    assert (sel.candidate.model, sel.candidate.family) == ("qwen/qwen3-32b", "qwen")
    assert sel.judge.model == "qwen/qwen3-32b" and sel.judge.family == "qwen"
    assert sel.rule == js.RULE_ID == "preference-order-v1"
    rec = sel.as_record()
    assert rec["provider"] == "groq" and rec["rule"] == "preference-order-v1" and rec["refusals"] == []
    assert "listed_models" not in rec["listing"] and rec["listing"]["listed_count"] == len(LIVE)
    assert json.dumps(rec) and SECRET not in json.dumps(rec)
    assert all("mixtral" not in b["model"] for b in t.bodies)


def test_model_and_family_reach_request_body_and_provenance():
    t = Transport()
    sel = js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(), judge_transport=t)
    j = sel.judge
    assert t.bodies[0]["model"] == "qwen/qwen3-32b" and t.bodies[0]["max_tokens"] == js.CANARY_MAX_TOKENS
    assert t.bodies[0]["response_format"] == {"type": "json_object"}
    j.judge_json("later", "x")
    assert t.bodies[-1]["model"] == "qwen/qwen3-32b" and t.bodies[-1]["max_tokens"] == js.JUDGE_MAX_TOKENS
    # scoring_provenance reads exactly these attributes (comparative_eval.py)
    assert (j.family, j.model, j.provider) == ("qwen", "qwen/qwen3-32b", "groq")
    assert j.same_family_override is False


def test_canary_not_counted_in_m6_budget_but_recorded():
    led = CostLedger()
    sel = js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(max_calls=1), ledger=led,
                          judge_transport=Transport())
    assert sel.judge.usage.calls == 0 and sel.judge.usage.tokens == 0 and sel.judge.usage.log == []
    assert sel.canary_spend["api_calls"] == 1 and sel.canary_spend["tokens"] == 15
    assert [e.kind for e in led.entries] == ["llm_judge"]  # ledger entry stays
    sel.judge.judge_json("m6_first", "x")  # max_calls=1 still has its whole budget
    assert sel.judge.usage.calls == 1


def test_canary_failure_falls_through_to_next_family():
    t = Transport(fail=("qwen/qwen3-32b",))
    sel = js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(), judge_transport=t)
    assert sel.candidate.model == "llama-3.3-70b-versatile"
    assert len(sel.canary_failures) == 1 and "qwen/qwen3-32b" in sel.canary_failures[0]
    assert SECRET not in sel.canary_failures[0]


def test_non_json_canary_falls_through():
    sel = js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(max_parse_retries=0),
                          judge_transport=Transport(bad=("qwen/qwen3-32b",)))
    assert sel.candidate.family == "meta-llama"


def test_all_canaries_fail_verbosely_and_attempts_bounded():
    t = Transport(fail=LIVE)
    with pytest.raises(JudgeConfigError) as e:
        js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(), judge_transport=t)
    assert len(t.bodies) == js.JUDGE_MAX_CANARY_ATTEMPTS == 3
    msg = str(e.value)
    assert "canary failures" in msg and "no run is created" in msg and SECRET not in msg


def test_panel_takes_distinct_families():
    ids = ("qwen/qwen3-32b", "qwen/qwen3-8b", "llama-3.3-70b-versatile")
    sel = js.select_judge(ENV, http_get=listing(*ids), limits=JudgeLimits(), judge_transport=Transport(), n=2)
    assert [j.family for j in sel.judges] == ["qwen", "meta-llama"]
    with pytest.raises(JudgeConfigError, match="size 3"):
        js.select_judge(ENV, http_get=listing(*ids), limits=JudgeLimits(), judge_transport=Transport(), n=3)


def test_pool_env_used_by_select():
    env = dict(ENV, VERA_JUDGE_POOL="groq:gemma2-9b-it")
    sel = js.select_judge(env, http_get=listing(*LIVE), limits=JudgeLimits(), judge_transport=Transport())
    assert sel.candidate.model == "gemma2-9b-it" and sel.pool_source == "VERA_JUDGE_POOL"


# ---- self-grading guard -----------------------------------------------------------------
def test_selected_judge_refuses_openai_family_by_default():
    with pytest.raises(JudgeConfigError, match="generation"):
        js.SelectedJudge("groq", "openai/gpt-oss-20b", "openai", limits=JudgeLimits(),
                         api_key="k", transport=Transport())


@pytest.mark.parametrize("name,val", [
    ("VERA_JUDGE_ALLOW_SAME_FAMILY", "1"), ("VERA_JUDGE_GROQ_MODEL", "some/model-x"),
    ("VERA_JUDGE_PRICE_IN", "9.99"), ("VERA_JUDGE_PRICE_OUT", "8.88")])
def test_selected_judge_refuses_stale_env_overrides(monkeypatch, name, val):
    """S5 F3: process env var set -> refused (name shown, value never shown), even for a non-openai family."""
    monkeypatch.setenv(name, val)
    with pytest.raises(JudgeConfigError) as ei:
        js.SelectedJudge("groq", "qwen/qwen3-32b", "qwen", limits=JudgeLimits(), api_key="k",
                         transport=Transport())
    assert name in str(ei.value) and val not in str(ei.value)


def test_selected_judge_refuses_override_in_passed_env_mapping():
    with pytest.raises(JudgeConfigError, match="VERA_JUDGE_ALLOW_SAME_FAMILY"):
        js.SelectedJudge("groq", "openai/gpt-oss-20b", "openai", limits=JudgeLimits(), api_key="k",
                         transport=Transport(), env={"VERA_JUDGE_ALLOW_SAME_FAMILY": "1"})


def test_same_family_override_never_claimed(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_ALLOW_SAME_FAMILY", "1")
    with pytest.raises(JudgeConfigError):
        js.SelectedJudge("groq", "openai/gpt-oss-20b", "openai", limits=JudgeLimits(), api_key="k",
                         transport=Transport())
    monkeypatch.delenv("VERA_JUDGE_ALLOW_SAME_FAMILY")
    j = js.SelectedJudge("groq", "qwen/qwen3-32b", "qwen", limits=JudgeLimits(), api_key="k",
                         transport=Transport())
    assert j.same_family_override is False


def test_select_from_with_stale_env_override_selects_nothing_and_spends_nothing(monkeypatch):
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "other/model")
    t = Transport()
    with pytest.raises(JudgeConfigError, match="VERA_JUDGE_GROQ_MODEL"):
        js.select_judge(ENV, http_get=listing(*LIVE), limits=JudgeLimits(), judge_transport=t)
    assert t.bodies == []


def test_call_failure_text_scrubs_the_api_key():
    """S5 F6: a key value inside the transport exception must not reach the raised message."""
    key = "plainsecretvalue987654"  # not key-shaped, so only the env-name based replacement can catch it

    def boom(url, headers, body, timeout):
        raise RuntimeError(f"401 for token {key}")
    j = js.SelectedJudge("groq", "qwen/qwen3-32b", "qwen", limits=JudgeLimits(), api_key=key, transport=boom)
    with pytest.raises(RuntimeError) as ei:
        j._call("hi")
    assert key not in str(ei.value) and "***" in str(ei.value)


def test_scrub_removes_values_and_keyish_strings():
    out = js.scrub(f"x {SECRET} y gsk_abcdefghijk postgresql://u:pw@h/db", ENV)
    assert SECRET not in out and "gsk_abcdefghijk" not in out and "u:pw@" not in out


# ---- F9: unknown attribute probes never load config
def test_unknown_attribute_raises_attributeerror_without_loading_config(monkeypatch):
    from vera.eval_config import EvalConfigError

    def bad(*a, **k):
        raise EvalConfigError("bad config")
    monkeypatch.setattr(js, "load_eval_config", bad)
    assert not hasattr(js, "NO_SUCH_THING")
    assert getattr(js, "NO_SUCH_THING", 7) == 7
    with pytest.raises(AttributeError):
        js.NO_SUCH_THING
    with pytest.raises(EvalConfigError):  # known names still fail verbosely on a bad config, no fallback
        js.RULE_ID


# ---- F5: pattern path plus explicit map, strictest wins
@pytest.mark.parametrize("mid", ["gpt-4.1-nano", "chatgpt-4o", "openai/gpt-oss-20b", "x/openai-y",
                                 "deepseek-r1-distill-llama-70b", "llama-guard-4-12b", "zzz-model-9",
                                 "qwen-llama-merge", ""])
def test_pattern_path_still_refuses_everything_the_old_code_refused(mid):
    fam, _ = js.family_for(mid, js._settings())  # settings only: pattern path, no map
    assert fam in (None, "openai")


def test_map_can_only_add_refusals():
    st = js._settings()
    mf = {"qwen/qwen3-32b": {"family": "openai"}, "llama-3.3-70b-versatile": {"family": "qwen"},
          "zzz-model-9": {"family": "qwen"}, "deepseek-chat-v3": {"family": "deepseek"},
          "llama-guard-4-12b": {"family": "meta-llama"}}
    assert js.family_for("qwen/qwen3-32b", st, model_families=mf)[0] == "openai"   # map says openai
    fam, why = js.family_for("llama-3.3-70b-versatile", st, model_families=mf)      # disagreement
    assert fam is None and "conflict" in why
    assert js.family_for("zzz-model-9", st, model_families=mf)[0] is None           # map never loosens
    assert js.family_for("llama-guard-4-12b", st, model_families=mf)[0] is None     # exclusion stands
    assert js.family_for("deepseek-chat-v3", st, model_families=mf)[0] == "deepseek"  # agreement ok


def test_family_for_without_settings_reads_run_config_map():
    assert js.family_for("openai/gpt-oss-120b")[0] == "openai"
    assert js.family_for("qwen/qwen3-32b")[0] == "qwen"
