"""Run fingerprint tests. Fakes only; git calls are injected or monkeypatched, no network, no DB."""
import copy
import json

import pytest

from tests.config_support import add_models, load_pair
from tests.test_m6_comparative import BASE, QUESTION
from vera.eval_config import load_eval_config
from vera.m6 import fingerprint as FP
from vera.m6.comparative_eval import evaluate_both
from vera.m7.fixture import ScriptedJudge, build_corpus, build_engineered, build_judge

CODE = {"commit": "abc", "dirty": False, "source": "git"}


def _eff(cfg=None, judge=None, **kw):
    cfg = cfg or load_eval_config()
    args = dict(budget={"cost_usd": 1.0, "latency_s": 600.0}, budget_source="run_config", seed=0,
                pricing_record={"input": 0.1, "output": 0.4}, corpus={})
    args.update(kw)
    return FP.effective_run_config(cfg, judge or build_judge(), **args)


def test_deterministic_and_order_independent():
    a, b = _eff(), _eff()
    assert FP.run_fingerprint(a, code_ver=CODE)["config_sha256"] == FP.run_fingerprint(b, code_ver=CODE)["config_sha256"]
    rev = json.loads(json.dumps(a))
    rev = {k: rev[k] for k in reversed(list(rev))}
    rev["scoring"] = {k: rev["scoring"][k] for k in reversed(list(rev["scoring"]))}
    assert FP.run_fingerprint(rev, code_ver=CODE)["config_sha256"] == FP.run_fingerprint(a, code_ver=CODE)["config_sha256"]
    fp = FP.run_fingerprint(a, code_ver=CODE)
    assert len(fp["config_sha256"]) == 64 and fp["canonicalization"] == FP.CANON and fp["schema"] == FP.SCHEMA
    assert fp["code_version"] == CODE and set(fp["components_sha256"]) == set(a)


def test_code_version_is_beside_the_hash_not_in_it():
    a = _eff()
    h1 = FP.run_fingerprint(a, code_ver=CODE)["config_sha256"]
    h2 = FP.run_fingerprint(a, code_ver={"commit": "other", "dirty": True, "source": "git"})["config_sha256"]
    assert h1 == h2


def _mut(path_keys, value):
    def f(e):
        d = e
        for k in path_keys[:-1]:
            d = d[k]
        d[path_keys[-1]] = value
    return f


@pytest.mark.parametrize("name,mutate", [
    ("question", _mut(("freeze", "question"), "other question")),
    ("criterion", _mut(("freeze", "success_criteria", "relevance_min"), 0.7)),
    ("fabrication rule", _mut(("run_config", "grounding_max_fabricated"), 2)),
    ("budget", _mut(("run_config", "budget", "effective", "cost_usd"), 0.5)),
    ("budget source", _mut(("run_config", "budget", "source"), "caller_override")),
    ("judge model", _mut(("judge", "model"), "another-model")),
    ("judge family source", _mut(("judge", "family_source"), "override")),
    ("judge price", _mut(("judge", "price_in"), 9.0)),
    ("generator model", _mut(("generator", "model"), "gpt-x")),
    ("pricing record", _mut(("generator", "pricing_record"), {"input": 9})),
    ("min_year", _mut(("pipeline_as_of_m6_code", "min_year", "gate_c"), 2020)),
    ("sub_questions", _mut(("pipeline_as_of_m6_code", "m4", "sub_questions"), [])),
    ("seed", _mut(("scoring", "seed"), 99)),
    ("scoring module hash", _mut(("scoring", "module_sha256", "vera/m6/eval_rubric.py"), "0" * 64)),
    ("freeze file hash", _mut(("freeze", "file_sha256"), "0" * 64)),
    ("run file hash", _mut(("run_config", "file_sha256"), "0" * 64)),
    ("judge settings", _mut(("run_config", "judge_settings", "max_canary_attempts"), 9)),
])
def test_each_component_change_changes_the_hash(name, mutate):
    base = _eff()
    changed = copy.deepcopy(base)
    mutate(changed)
    assert FP.run_fingerprint(changed, code_ver=CODE)["config_sha256"] != \
        FP.run_fingerprint(base, code_ver=CODE)["config_sha256"], name


def test_run_config_edit_changes_fingerprint_via_its_file_hash(tmp_path):
    other = load_pair(tmp_path, run_edit=lambda r: r["budget"].__setitem__("cost_usd", 0.5))
    assert FP.run_fingerprint(_eff(other, budget={"cost_usd": 0.5, "latency_s": 600.0}), code_ver=CODE)["config_sha256"] != \
        FP.run_fingerprint(_eff(), code_ver=CODE)["config_sha256"]


def test_selection_record_stable_part_only_and_changed_pool_changes_hash():
    rec = {"provider": "groq", "model": "qwen/qwen3-32b", "family": "qwen", "family_source": "pattern",
           "rule": "preference-order-v1", "pool_source": "default", "candidates": [["groq", "qwen/qwen3-32b", "qwen"]],
           "panel": [], "listing": {"checked_at": "t1"}, "canary_spend": {"tokens": 5}, "refusals": [["a"]]}
    noisy = dict(rec, listing={"checked_at": "t2"}, canary_spend={"tokens": 99}, refusals=[["b"]])
    h = lambda r: FP.run_fingerprint(_eff(judge_selection=r), code_ver=CODE)["config_sha256"]  # noqa: E731
    assert h(rec) == h(noisy)  # volatile parts are not hashed
    # F3: the live candidate list is recorded beside the hash, not in it
    assert h(rec) == h(dict(rec, candidates=[["groq", "llama", "meta-llama"], ["groq", "qwen/qwen3-32b", "qwen"]]))
    assert h(rec) != h(dict(rec, model="llama-3.3-70b-versatile", family="meta-llama"))  # a different judge differs
    assert h(rec) != h(dict(rec, pool_source="VERA_JUDGE_POOL")) and h(rec) != h(dict(rec, rule="other-rule"))
    assert h(rec) != h(None)


def test_recompute_from_stored_effective_config():
    ev = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(),
                       code_ver=CODE)
    fp, eff = ev["run_fingerprint"], json.loads(json.dumps(ev["effective_config"], default=str))
    assert FP.run_fingerprint(eff, code_ver=CODE)["config_sha256"] == fp["config_sha256"]
    assert len(fp["config_sha256"]) == 64 and ev["frozen_requirements_hash"] == "637d247a50217128"
    assert eff["run_config"]["file_sha256"] == load_eval_config().run_file_sha256
    assert eff["freeze"]["file_sha256"] == load_eval_config().freeze_file_sha256


def test_old_hash_field_is_kept_and_new_field_is_separate():
    ev = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(),
                       code_ver=CODE)
    assert len(ev["frozen_requirements_hash"]) == 16 and ev["frozen_requirements_hash"] != ev["run_fingerprint"]["config_sha256"][:16]


def test_nan_rejected_verbosely():
    e = _eff()
    e["scoring"]["seed"] = float("nan")
    with pytest.raises(ValueError, match="not canonically serialisable"):
        FP.run_fingerprint(e, code_ver=CODE)


def test_code_version_unknown_is_recorded_not_silent():
    def boom(*a):
        raise FileNotFoundError("no git")
    assert FP.code_version(run=boom) == {"commit": None, "dirty": None, "source": "unknown (FileNotFoundError)"}
    assert FP.code_version(run=lambda *a: "x")["source"] == "git"


def test_budget_override_is_recorded_as_caller_override():
    ev = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(),
                       budget={"cost_usd": 0.5, "latency_s": 600.0}, code_ver=CODE)
    assert ev["budget_source"] == "caller_override" and ev["budget"]["cost_usd"] == 0.5
    ev2 = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(), code_ver=CODE)
    assert ev2["budget_source"] == "run_config" and ev2["budget"] == {"cost_usd": 1.0, "latency_s": 600.0}


def test_stub_judge_without_attributes_still_fingerprints():
    class Bare:
        family, model = "stubfam", "stub"
    e = _eff(judge=Bare())
    assert e["judge"]["provider"] is None and e["judge"]["family"] == "stubfam"
    assert isinstance(ScriptedJudge, type)


# ---------------------------------------------------------------- review F1/F2/F3/F15 (goal #2 fix round)
M4_USED = {"max_searches": 0, "sub_questions": [{"id": f"r{i}", "text": f"req {i}", "required": True} for i in range(8)]}
LIMITS_USED = {"num_results": None, "max_fetch": None, "max_sources": None, "cap_usd": 1.0, "cap_s": 600.0}


def _h(**kw):
    return FP.run_fingerprint(_eff(**kw), code_ver=CODE)["config_sha256"]


def test_f1_fingerprint_records_the_values_the_run_used():
    eff = _eff(run_inputs={"m4": M4_USED, "stage_limits": LIMITS_USED})
    m4 = eff["pipeline_as_of_m6_code"]["m4"]
    assert m4["max_searches"] == 0 and len(m4["sub_questions"]) == 8  # not the defaults (2 and 4)
    assert m4["values_source"].startswith("run")
    default = _eff()["pipeline_as_of_m6_code"]["m4"]
    assert default["max_searches"] != 0 and len(default["sub_questions"]) == 4
    assert "defaults" in default["values_source"]  # never labelled as effective when the runner did not supply


def test_f1_used_m4_values_change_the_hash():
    base = _h(run_inputs={"m4": M4_USED, "stage_limits": LIMITS_USED})
    assert base != _h(run_inputs={"m4": dict(M4_USED, max_searches=2), "stage_limits": LIMITS_USED})
    assert base != _h(run_inputs={"m4": dict(M4_USED, sub_questions=M4_USED["sub_questions"][:7]),
                                  "stage_limits": LIMITS_USED})
    assert base != _h()  # supplied values vs code defaults


def test_f1_incomplete_m4_used_fails_verbosely():
    with pytest.raises(ValueError, match="max_searches.*sub_questions"):
        _eff(run_inputs={"m4": {"max_searches": 0}})


@pytest.mark.parametrize("key,value", [("num_results", 5), ("max_fetch", 3), ("max_sources", 4),
                                       ("cap_usd", 0.5), ("cap_s", 300.0)])
def test_f2_each_stage_limit_changes_the_hash(key, value):
    base = _h(run_inputs={"m4": M4_USED, "stage_limits": LIMITS_USED})
    assert base != _h(run_inputs={"m4": M4_USED, "stage_limits": dict(LIMITS_USED, **{key: value})})


def test_f2_scored_question_reviewer_overrides_and_model_change_the_hash(tmp_path):
    base = _h()
    assert base != _h(scored_question="a different question was scored")
    assert base != _h(reviewer_overrides={"relevance": {"engineered": 5, "reason": "x"}})
    assert _h(reviewer_overrides={"relevance": {"engineered": 5}}) != _h(reviewer_overrides={"relevance": {"engineered": 4}})
    cfg = load_pair(tmp_path, run_edit=add_models({"gpt-4.1-mini": "openai"}))
    assert _h(cfg=cfg) != _h(cfg=cfg, generator_model="gpt-4.1-mini")  # baseline model override is hashed
    e = _eff(cfg=cfg, generator_model="gpt-4.1-mini")
    assert e["generator"]["model"] == "gpt-4.1-mini" and e["generator"]["configured_model"] == "gpt-4.1-nano"


def test_f2_evaluate_both_hashes_overrides_used_by_the_run(tmp_path):
    cfg = load_pair(tmp_path, run_edit=add_models({"gpt-4.1-mini": "openai"}))
    corpus = build_corpus()
    ev = lambda **kw: evaluate_both(build_engineered(corpus), BASE, corpus, QUESTION, judge=build_judge(),  # noqa: E731
                                    code_ver=CODE, config=cfg, **kw)["run_fingerprint"]["config_sha256"]
    base = ev()
    assert base != ev(generator_model="gpt-4.1-mini")
    assert base != ev(reviewer_overrides={"auditability": {"engineered": 5, "reason": "r"}})
    assert base != ev(run_inputs={"m4": M4_USED, "stage_limits": LIMITS_USED})
    other_q = evaluate_both(build_engineered(corpus), BASE, corpus, QUESTION + " (changed)", judge=build_judge(),
                            code_ver=CODE, config=cfg)["run_fingerprint"]["config_sha256"]
    assert base != other_q


def test_f3_same_judge_different_candidate_list_same_hash_different_judge_differs():
    rec = {"provider": "groq", "model": "qwen/qwen3-32b", "family": "qwen", "family_source": "pattern",
           "rule": "preference-order-v1", "pool_source": "default", "panel": [["groq", "qwen/qwen3-32b", "qwen"]],
           "candidates": [["groq", "qwen/qwen3-32b", "qwen"], ["groq", "a", "b"]]}
    shrunk = dict(rec, candidates=[["groq", "qwen/qwen3-32b", "qwen"]])
    grown = dict(rec, candidates=rec["candidates"] + [["groq", "new-model-id", "google"]])
    assert _h(judge_selection=rec) == _h(judge_selection=shrunk) == _h(judge_selection=grown)
    assert _h(judge_selection=rec) != _h(judge_selection=dict(rec, model="other/model"))
    assert "candidates" not in FP.selection_for_fingerprint(rec)
    # recorded beside the hash: evaluate_both keeps the full record in scoring_provenance
    ev = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(),
                       code_ver=CODE, judge_selection=rec)
    assert ev["scoring_provenance"]["judge_selection"]["candidates"] == rec["candidates"]


def test_f15_empty_budget_is_labelled_run_config():
    for b in ({}, None):
        ev = evaluate_both(build_engineered(build_corpus()), BASE, build_corpus(), QUESTION, judge=build_judge(),
                           budget=b, code_ver=CODE)
        assert ev["budget_source"] == "run_config" and ev["budget"] == {"cost_usd": 1.0, "latency_s": 600.0}
