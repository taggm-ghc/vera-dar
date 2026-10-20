"""The 'cannot diverge' proof for goal #2: thresholds, budget, generator and judge family come from the config
files and from nothing else. Temporary config copies only; no DB, no network, no LLM."""
import ast
import json
import re
from pathlib import Path

import pytest
import requests

from tests.config_support import add_models, load_pair, make_pair
from tests.test_m6_comparative import BASE, BASE_D, _dim, _llm, _run
from tests.test_mvp_preflight import FULL_ENV, LIVE, REPO, Transport, listing
from vera import judge_select as js
from vera import mvp_preflight as pf
from vera.eval_config import EvalConfigError, load_eval_config
from vera.m6.comparative_eval import apply_success_rule, evaluate_both
from vera.m6.eval_rubric import score_cost_latency
from vera.m6.judge import JudgeConfigError, JudgeLimits
from vera.m6.m6_runner import run_m6
from vera.m7.fixture import QUESTION, build_corpus, build_engineered, build_judge

VERA = Path(__file__).resolve().parent.parent / "vera"


# ---------------------------------------------------------------- SD1: no literals left in code
def _py_files():
    return [p for p in VERA.rglob("*.py")]


def test_no_threshold_or_budget_literals_in_vera():
    offenders = []
    for p in _py_files():
        tree = ast.parse(p.read_text(), filename=str(p))
        for n in ast.walk(tree):
            targets = []
            if isinstance(n, ast.Assign):
                targets = [t.id for t in n.targets if isinstance(t, ast.Name)]
            elif isinstance(n, (ast.AnnAssign,)) and isinstance(n.target, ast.Name):
                targets = [n.target.id]
            for t in targets:
                if re.match(r"^PASS_", t) or t in ("DEFAULT_BUDGET", "EXPECTED_FREEZE_HASH"):
                    offenders.append(f"{p.name}:{n.lineno} assigns {t}")
            if isinstance(n, ast.Dict):
                keys = {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant)}
                # a budget LIMIT dict has no usage keys; spend/usage dicts (api_calls/tokens) are not limits
                if {"cost_usd", "latency_s"} <= set(keys) and not ({"api_calls", "tokens"} & set(keys)) and all(
                        isinstance(keys[k], ast.Constant) and isinstance(keys[k].value, (int, float))
                        for k in ("cost_usd", "latency_s")):
                    offenders.append(f"{p.name}:{n.lineno} numeric budget dict literal")
    assert not offenders, offenders


def test_freeze_hash_pin_literal_lives_only_in_config_and_tests():
    for p in _py_files():
        assert "637d247a50217128" not in p.read_text() and "661400d265e2c17b" not in p.read_text(), p.name


def test_min_year_copies_agree():
    """MUST guard for an accepted divergence: three definitions today (fingerprinted), equality enforced."""
    from vera.m3 import appraisal_rubric
    from vera.m4 import gate_c
    from vera.m5 import claim_verification
    assert gate_c.MIN_YEAR == claim_verification.MIN_YEAR == appraisal_rubric.EVIDENCE_WINDOW[0]


# ---------------------------------------------------------------- SD1: each value drives the verdict
# (file, key, new value, dimension, pair_override_for_this_case)
CASES = [
    ("freeze", "relevance_threshold", 0.80, "relevance", {}),
    ("freeze", "relevance_improvement_pp", 50, "relevance", {}),
    ("freeze", "grounding_threshold", 0.99, "grounding", {}),
    ("freeze", "reasoning_threshold", 5, "reasoning_integrity", {}),
    ("freeze", "reasoning_objections_improvement", 4, "reasoning_integrity", {}),
    ("freeze", "auditability_threshold", 5, "auditability", {}),
    ("freeze", "cost_latency_threshold", 4, "cost_latency", {}),
    ("run", "grounding_max_fabricated", 1, "grounding", {"grounding": {"fabricated": 1}}),
]


@pytest.mark.parametrize("which,key,val,dim,override", CASES)
def test_each_criterion_drives_the_verdict_and_the_need_text(tmp_path, which, key, val, dim, override):
    real = load_eval_config()

    def f_edit(f):
        if which == "freeze":
            f["success_criteria"][key] = val

    def r_edit(r):
        if which == "run":
            r["scoring"][key] = val
    perturbed = load_pair(tmp_path, f_edit, r_edit)
    eng = _dim(**override)
    a = apply_success_rule(eng, BASE_D, real.success)["checks"][dim]
    b = apply_success_rule(eng, BASE_D, perturbed.success)["checks"][dim]
    assert a["pass"] != b["pass"], f"{key}={val} did not change the verdict"
    assert a["need"] != b["need"], f"{key}={val} did not change the 'need' text"


def test_default_rule_reads_the_real_config():
    assert apply_success_rule(_dim(), BASE_D)["success"]
    assert "60%" in apply_success_rule(_dim(), BASE_D)["checks"]["relevance"]["need"]
    assert "15 pp" in apply_success_rule(_dim(), BASE_D)["checks"]["relevance"]["need"]


def test_budget_from_config_reaches_the_cost_score(tmp_path):
    half = load_pair(tmp_path, run_edit=lambda r: r["budget"].__setitem__("cost_usd", 0.155))
    corpus = build_corpus()
    ev_real = evaluate_both(build_engineered(corpus), BASE, corpus, QUESTION, judge=build_judge())
    ev_half = evaluate_both(build_engineered(corpus), BASE, corpus, QUESTION, judge=build_judge(), config=half)
    r1 = ev_real["detail"]["engineered"]["cost_latency"]["ratio"]
    r2 = ev_half["detail"]["engineered"]["cost_latency"]["ratio"]
    assert r2 > r1 and ev_half["budget"]["cost_usd"] == 0.155 and ev_half["budget_source"] == "run_config"


def test_generator_model_from_config_reaches_the_runner(tmp_path):
    seen = {}

    def llm(q, m):
        seen["model"] = m
        return _llm(q, m)
    run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=llm, store=False)
    assert seen["model"] == load_eval_config().generator_model
    cfg = load_pair(tmp_path, run_edit=add_models({"gpt-3.5-turbo": "openai"}))
    run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=llm, store=False, model="gpt-3.5-turbo",
           config=cfg)
    assert seen["model"] == "gpt-3.5-turbo"
    # an unmapped override model is refused BEFORE any baseline spend (its family is unknown)
    seen.clear()
    with pytest.raises(EvalConfigError, match="fail closed"):
        run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=llm, store=False, model="unmapped-x")
    assert seen == {}


def test_budget_confirmed_message_quotes_the_configured_budget():
    with pytest.raises(RuntimeError, match=r"configured \$1\.00 / 600 s"):
        run_m6(_run(), budget_confirmed=False, judge=build_judge(), llm_call=_llm, store=False)


def test_runner_raises_verbosely_on_a_bad_config_before_any_spend(tmp_path):
    def bad(f):
        del f["success_criteria"]["relevance_threshold"]
    with pytest.raises(EvalConfigError, match="relevance_threshold"):
        load_pair(tmp_path, bad)
    spent = []
    from vera.m6 import m6_runner
    orig = m6_runner.load_eval_config
    m6_runner.load_eval_config = lambda: (_ for _ in ()).throw(EvalConfigError("bad config sentinel"))
    try:
        with pytest.raises(EvalConfigError, match="bad config sentinel"):
            run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=lambda *a: spent.append(1), store=False)
    finally:
        m6_runner.load_eval_config = orig
    assert spent == []


def test_runner_preflight_called_before_baseline(monkeypatch, tmp_path):
    cfg = load_pair(tmp_path, run_edit=add_models({"qwen/qwen3-32b": "qwen"}))
    monkeypatch.setenv("GROQ_API_KEY", "g-test")
    monkeypatch.setenv("VERA_JUDGE_GROQ_MODEL", "qwen/qwen3-32b")
    llm_calls = []

    class NoList:
        ok, status_code, text = True, 200, ""

        def json(self):
            return {"data": [{"id": "something-else"}]}
    with pytest.raises(JudgeConfigError, match="does not list model"):
        run_m6(_run(), budget_confirmed=True, llm_call=lambda *a: llm_calls.append(1), store=False, config=cfg,
               preflight_http_get=lambda *a, **k: NoList())
    assert llm_calls == [], "a baseline call was made although the judge preflight failed"


def test_runner_records_preflight_skipped_when_judge_supplied():
    rep = run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=_llm, store=False)
    prov = rep["evaluation"]["scoring_provenance"]
    assert prov["preflight"]["status"] == "skipped"
    assert len(rep["evaluation"]["run_fingerprint"]["config_sha256"]) == 64


# ---------------------------------------------------------------- pre-flight now reads the pin from config
def _run_pf(**kw):
    base = dict(question_id=2, env=FULL_ENV, repo_root=REPO, limits=JudgeLimits(), judge_transport=Transport(),
                http_get=listing(*LIVE), db_identity=lambda: pf.EXPECTED_DB_ROLE,
                question_text_for=lambda q: json.loads((REPO / "config" / "vera_eval_freeze.json").read_text())["question"],
                check_ignored=lambda rel: True, generator_model="gpt-4.1-nano", pricing_lookup=lambda m: object())
    base.update(kw)
    return pf.run_preflight(**base)


def test_pf10_passes_on_the_real_config_and_names_the_pin_source():
    rep = _run_pf()
    pf10 = next(c for c in rep.checks if c.id == "PF10")
    assert pf10.ok and "match the pin in the run config" in pf10.detail and "637d247a50217128" in pf10.detail


def _repo_copy(tmp_path, freeze_edit=None, run_edit=None):
    (tmp_path / "config").mkdir()
    fp, rp = make_pair(tmp_path, freeze_edit, run_edit, repin=freeze_edit is None)
    (tmp_path / "config" / "vera_eval_freeze.json").write_bytes(fp.read_bytes())
    (tmp_path / "config" / "vera_eval_run.json").write_bytes(rp.read_bytes())
    (tmp_path / "config" / "model-selection.json").write_bytes((REPO / "config" / "model-selection.json").read_bytes())
    return tmp_path


def test_pf10_fails_on_a_changed_freeze_file_and_stops_canary_spend(tmp_path):
    def edit(f):
        f["objections"][0]["text"] += "!"
    root = _repo_copy(tmp_path, edit)  # pin NOT recomputed (repin False): the changed file must be refused
    t = Transport()
    rep = _run_pf(repo_root=root, judge_transport=t)
    pf10 = next(c for c in rep.checks if c.id == "PF10")
    assert not pf10.ok and ("immutable" in pf10.detail or "pinned" in pf10.detail)
    assert t.bodies == []


def test_pf10_fails_on_invalid_run_config_and_dependents_fail_verbosely(tmp_path):
    root = _repo_copy(tmp_path, run_edit=lambda r: r.pop("budget"))
    rep = _run_pf(repo_root=root)
    bad = {c.id for c in rep.checks if not c.ok}
    assert {"PF10", "PF2", "PF3", "PF4"} <= bad and "budget: missing" in next(c for c in rep.checks if c.id == "PF10").detail


# ---------------------------------------------------------------- judge selection is config-driven (D2 wiring)
def test_judge_select_constants_resolve_from_the_run_config_lazily():
    cfg = load_eval_config()
    assert js.JUDGE_PROVIDER_ORDER == cfg.judge.providers and js.RULE_ID == cfg.judge.rule
    assert js.JUDGE_MAX_CANARY_ATTEMPTS == cfg.judge.max_canary_attempts
    assert js.CANARY_MAX_TOKENS == cfg.judge.canary_max_tokens
    assert js.FORBIDDEN_JUDGE_ENV == cfg.judge.forbidden_env_overrides
    assert js.LISTING_URL == {"groq": "https://api.groq.com/openai/v1/models"}
    with pytest.raises(AttributeError):
        js.NOT_A_THING


def test_pool_preference_order_and_attempt_limit_come_from_the_config(tmp_path, monkeypatch):
    def edit(r):
        r["judge"]["selection"]["family_preference"] = ["meta-llama", "qwen"]
        r["judge"]["selection"]["max_canary_attempts"] = 1
        r["judge"]["selection"]["rule"] = "preference-order-test"
    cfg = load_pair(tmp_path, run_edit=edit)
    ids = ("qwen/qwen3-32b", "llama-3.3-70b-versatile")
    env = {"GROQ_API_KEY": "gsk_test_key_value_123456"}
    monkeypatch.delenv("VERA_JUDGE_ALLOW_SAME_FAMILY", raising=False)
    monkeypatch.delenv("VERA_JUDGE_GROQ_MODEL", raising=False)
    t = Transport(fail={"llama-3.3-70b-versatile"})
    with pytest.raises(JudgeConfigError, match=r"limit 1"):  # llama first (preference), fails, 1 attempt only
        js.select_judge(env, http_get=listing(*ids), limits=JudgeLimits(), judge_transport=t, config=cfg)
    assert [b["model"] for b in t.bodies] == ["llama-3.3-70b-versatile"]
    t2 = Transport()
    sel = js.select_judge(env, http_get=listing(*ids), limits=JudgeLimits(), judge_transport=t2, config=cfg)
    assert sel.candidate.model == "llama-3.3-70b-versatile" and sel.rule == "preference-order-test"
    assert (sel.judge.model_source, sel.judge.family_source) == ("explicit", "pattern")


def test_unsupported_pool_provider_is_refused_at_load_naming_it(tmp_path):
    for bad in ("nosuchprovider", "mistral"):
        with pytest.raises(EvalConfigError, match=f"'{bad}' is not a supported judge provider"):
            load_pair(tmp_path, run_edit=lambda r, b=bad: r["judge"]["pool"].__setitem__("providers", ["groq", b]))


def test_supported_providers_match_the_judge_facts_table():
    from vera.eval_config import SUPPORTED_JUDGE_PROVIDERS
    from vera.m6.judge import _PROVIDERS
    assert set(SUPPORTED_JUDGE_PROVIDERS) == set(_PROVIDERS)  # drift guard (judge.py cannot be imported there)


def test_pool_provider_must_be_in_config_and_have_facts(tmp_path):
    import dataclasses
    base = load_pair(tmp_path)  # providers that bypass the loader (a hand-built config) still fail verbosely
    cfg = dataclasses.replace(base, judge=dataclasses.replace(base.judge, providers=("nosuchprovider",)))
    with pytest.raises(JudgeConfigError, match="no provider facts"):
        js.select_judge({"GROQ_API_KEY": "k"}, http_get=listing("qwen/qwen3-32b"), config=cfg)
    with pytest.raises(JudgeConfigError, match="not in the judge pool"):
        js.SelectedJudge("groq", "qwen/qwen3-32b", "qwen", api_key="k", transport=Transport(), config=cfg)


def test_pattern_edit_in_config_changes_the_family_decision(tmp_path):
    cfg = load_pair(tmp_path, run_edit=lambda r: r["judge"]["family_patterns"]["generic"].append(["zzz", "zfam"]))
    assert js.family_for("zzz-model-9")[0] is None  # real config: unknown
    assert js.family_for("zzz-model-9", cfg.judge)[0] == "zfam"


def test_selected_judge_with_config_that_makes_generator_non_openai(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_model": "gen-other"}))

    def edit(r):
        r["generator"]["model"] = "gen-other"
        r["model_families"]["gen-other"] = {"family": "qwen", "source": "https://example.invalid/test-only"}
    cfg = load_pair(tmp_path, run_edit=edit, model_selection_path=sel)
    with pytest.raises(JudgeConfigError, match="equals the generation family"):  # now qwen is the generator's family
        js.SelectedJudge("groq", "qwen/qwen3-32b", "qwen", api_key="k", transport=Transport(), config=cfg)


def test_check_eval_config_script(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_eval_config", VERA.parent / "scripts" / "check_eval_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "eval config OK" in out and "M7 fixture OK" in out and "gsk_" not in out

    def bad(f):
        del f["success_criteria"]["relevance_threshold"], f["success_criteria"]["grounding_threshold"]
    fp, rp = make_pair(tmp_path, bad)
    assert mod.main([str(fp), str(rp)]) == 1
    err = capsys.readouterr().err
    assert "relevance_threshold" in err and "grounding_threshold" in err and "EVAL CONFIG INVALID" in err


# ---------------------------------------------------------------- review F8/F10/F11 (goal #2 fix round)
def test_f8_a_config_file_changed_after_load_does_not_change_the_scoring(tmp_path):
    import json as _json
    fp, rp = make_pair(tmp_path)
    from vera.eval_config import load_eval_config as _load
    cfg = _load(fp, rp)

    def mutate_midrun(q, m):
        d = _json.loads(rp.read_text())
        d["budget"]["cost_usd"] = 0.0001  # an edit mid-run; must be invisible to this run
        d["scoring"]["grounding_max_fabricated"] = 9
        rp.write_text(_json.dumps(d))
        return _llm(q, m)
    rep = run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=mutate_midrun, store=False, config=cfg)
    ev = rep["evaluation"]
    assert ev["budget"]["cost_usd"] == cfg.budget.cost_usd == 1.0
    assert ev["effective_config"]["run_config"]["file_sha256"] == cfg.run_file_sha256
    assert ev["effective_config"]["run_config"]["grounding_max_fabricated"] == 0


def test_f10_preflight_record_is_truthful_when_the_caller_selected_the_judge():
    sel = {"provider": "groq", "model": "qwen/qwen3-32b", "family": "qwen", "family_source": "pattern",
           "rule": "preference-order-v1", "pool_source": "default", "panel": [], "candidates": [],
           "canary_spend": {"tokens": 15}, "listing": {"status": "ok", "checked_at": "t"}}
    rep = run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=_llm, store=False, judge_selection=sel)
    pre = rep["evaluation"]["scoring_provenance"]["preflight"]
    assert pre["status"] == "performed_by_caller" and pre["listing_status"] == "ok"
    assert pre["canary_spend"] == {"tokens": 15} and pre["chosen"] == ["groq", "qwen/qwen3-32b"]


def _patch_cli(monkeypatch, captured):
    from vera.m6 import m6_runner
    monkeypatch.setattr(m6_runner, "load_run", lambda rid=None: _run())
    monkeypatch.setattr(js, "default_http_get", listing(*LIVE))
    monkeypatch.setattr(js, "default_judge_transport", Transport())
    monkeypatch.setenv("GROQ_API_KEY", "g-test-key-123456")
    monkeypatch.delenv("VERA_JUDGE_ALLOW_SAME_FAMILY", raising=False)
    monkeypatch.delenv("VERA_JUDGE_GROQ_MODEL", raising=False)

    def fake_run_m6(run, **kw):
        captured.update(kw)
        return {"evaluation": {"outcome": "success", "failed_dimensions": []},
                "cost_summary": {"cost_usd": 0, "api_calls": 0}}
    monkeypatch.setattr(m6_runner, "run_m6", fake_run_m6)
    monkeypatch.setattr(m6_runner, "build_default_judge",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy env judge reached")))
    return m6_runner


def test_f11_cli_routes_through_pool_selection(monkeypatch):
    cap = {}
    m6_runner = _patch_cli(monkeypatch, cap)
    assert m6_runner.main(["--budget-confirmed", "--no-store"]) == 0
    assert cap["judge"].model == "qwen/qwen3-32b" and cap["judge_selection"]["rule"] == "preference-order-v1"
    assert cap["config"] is not None and cap["ledger"] is not None


@pytest.mark.parametrize("name,val", [("VERA_JUDGE_ALLOW_SAME_FAMILY", "1"), ("VERA_JUDGE_PRICE_IN", "0"),
                                      ("VERA_JUDGE_GROQ_MODEL", "x/y")])
def test_f11_cli_refuses_forbidden_env_overrides(monkeypatch, capsys, name, val):
    cap = {}
    m6_runner = _patch_cli(monkeypatch, cap)
    monkeypatch.setenv(name, val)
    assert m6_runner.main(["--budget-confirmed", "--no-store"]) == 2
    assert cap == {} and name in capsys.readouterr().err


def test_f11_cli_refuses_unconfirmed_budget_before_any_listing(monkeypatch, capsys):
    cap = {}
    m6_runner = _patch_cli(monkeypatch, cap)
    monkeypatch.setattr(js, "default_http_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("listed")))
    assert m6_runner.main(["--no-store"]) == 2 and cap == {} and "not confirmed" in capsys.readouterr().err
