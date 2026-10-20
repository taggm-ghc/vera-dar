"""Loader tests (goal #2). Temporary copies only: the real freeze file and run config are never written."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.config_support import add_models, load_pair, make_pair, real_freeze, real_run
from vera import eval_config as ec
from vera.eval_config import EvalConfigError, family_for, load_eval_config
from vera.m6 import eval_rubric as R

# An INDEPENDENT record of the immutable freeze file's hash (SD5): changing the freeze file needs the pin in
# config/vera_eval_run.json AND this literal to change, two visible edits, never an accident.
FREEZE_SHA256_LITERAL = "661400d265e2c17b1f68700f1ae36fd15e2bf4d007947ebdf3eb00f4d8027b49"
FREEZE_HASH16_LITERAL = "637d247a50217128"


def test_loads_real_files():
    c = load_eval_config()
    f = real_freeze()["success_criteria"]
    assert c.success.relevance_min == f["relevance_threshold"] == 0.6
    assert c.success.relevance_delta == pytest.approx(0.15) and c.success.relevance_delta == f["relevance_improvement_pp"] / 100
    assert (c.success.grounding_min, c.success.reasoning_min, c.success.reasoning_delta) == (0.9, 4, 2)
    assert (c.success.audit_min, c.success.cost_min, c.success.grounding_max_fabricated) == (4, 3, 0)
    r = real_run()
    assert (c.budget.cost_usd, c.budget.latency_s) == (r["budget"]["cost_usd"], r["budget"]["latency_s"])
    assert c.generator_model == "gpt-4.1-nano"
    # the OpenAI-family entries the judge guard tests rely on
    assert c.model_families["gpt-4.1-nano"]["family"] == c.model_families["openai/gpt-oss-20b"]["family"] == "openai"
    assert c.question == real_freeze()["question"] and len(c.requirements) == 8 and len(c.objections) == 5
    assert c.judge.providers == ("groq",) and c.run_config_version == "3"


def test_freeze_file_is_not_merged_with_run_config():
    c = load_eval_config()
    assert set(c.raw) == set(real_freeze())
    assert "budget" not in c.raw and "success_criteria" in c.raw and "scoring" in c.run_raw


def test_requirements_hash_unchanged_and_old_formula_delegates():
    c = load_eval_config()
    assert c.requirements_hash16 == FREEZE_HASH16_LITERAL
    assert R.freeze_hash({"requirements": list(c.requirements), "objections": list(c.objections)}) == FREEZE_HASH16_LITERAL
    legacy = hashlib.sha256(json.dumps({"requirements": real_freeze()["requirements"],
                                        "objections": real_freeze()["objections"]},
                                       sort_keys=True).encode()).hexdigest()[:16]
    assert legacy == FREEZE_HASH16_LITERAL  # the original formula, written out here, still agrees


def test_freeze_file_unchanged():
    """SD5: the real freeze file is immutable in this incarnation. Raw hash AND newline-normalised hash."""
    raw = ec.FREEZE_PATH.read_bytes()
    assert ec.file_sha256(raw) == FREEZE_SHA256_LITERAL
    # today the file has no CR/BOM, so raw == normalised; if this fails with a CRLF checkout the NORMALISED
    # assertion above still holds, which is exactly why the pin is newline-normalised.
    assert b"\r" not in raw
    assert real_run()["freeze_pin"]["file_sha256"] == FREEZE_SHA256_LITERAL
    assert real_run()["freeze_pin"]["requirements_hash16"] == FREEZE_HASH16_LITERAL


def test_pin_hash_ignores_crlf_checkout_but_not_content(tmp_path):
    raw = ec.FREEZE_PATH.read_bytes()
    assert ec.file_sha256(raw.replace(b"\n", b"\r\n")) == FREEZE_SHA256_LITERAL
    assert ec.file_sha256(b"\xef\xbb\xbf" + raw) == FREEZE_SHA256_LITERAL
    assert ec.file_sha256(raw + b" ") != FREEZE_SHA256_LITERAL
    # and the loader accepts a CRLF copy of the freeze file against the real pin
    fp = tmp_path / "freeze_crlf.json"
    fp.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert load_eval_config(fp, ec.RUN_PATH).freeze_file_sha256 == FREEZE_SHA256_LITERAL


def test_loader_refuses_changed_freeze_naming_both_hashes(tmp_path):
    fp = tmp_path / "freeze.json"
    fp.write_bytes(ec.FREEZE_PATH.read_bytes() + b"\n ")  # one extra byte of whitespace
    with pytest.raises(EvalConfigError) as ei:
        load_eval_config(fp, ec.RUN_PATH)
    msg = str(ei.value)
    assert FREEZE_SHA256_LITERAL in msg and ec.file_sha256(fp.read_bytes()) in msg and "immutable" in msg


def test_loader_refuses_changed_requirements_even_if_file_pin_matches(tmp_path):
    def edit(f):
        f["requirements"][0]["text"] = "tampered"
    fp, rp = make_pair(tmp_path, edit)  # pin recomputed for the file ...
    r = json.loads(rp.read_text())
    r["freeze_pin"]["requirements_hash16"] = FREEZE_HASH16_LITERAL  # ... but the requirements hash stays real
    rp.write_text(json.dumps(r))
    with pytest.raises(EvalConfigError, match=f"requirements hash.*{FREEZE_HASH16_LITERAL}"):
        load_eval_config(fp, rp)


def test_missing_keys_list_all_problems_at_once(tmp_path):
    def edit(f):
        del f["success_criteria"]["grounding_threshold"], f["success_criteria"]["auditability_threshold"]
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, edit)
    m = str(ei.value)
    assert "grounding_threshold: missing or null" in m and "auditability_threshold: missing or null" in m
    assert "2 problem(s)" in m and "nothing was scored" in m


@pytest.mark.parametrize("bad,expect", [(None, "missing or null"), (True, "not a number"), ("0.6", "not a number"),
                                        (float("nan"), "not finite")])
def test_null_bool_string_nan_rejected(tmp_path, bad, expect):
    # NaN cannot be written as strict JSON by json.dumps(allow_nan=False) but Python's loader accepts the token
    def edit(f):
        f["success_criteria"]["relevance_threshold"] = bad
    with pytest.raises(EvalConfigError, match=expect):
        load_pair(tmp_path, edit)


@pytest.mark.parametrize("key,val", [("grounding_threshold", 1.5), ("reasoning_threshold", 6), ("reasoning_threshold", 4.5),
                                     ("relevance_improvement_pp", -1), ("cost_latency_threshold", 0)])
def test_out_of_range_or_fractional_rejected(tmp_path, key, val):
    def edit(f):
        f["success_criteria"][key] = val
    with pytest.raises(EvalConfigError, match=key):
        load_pair(tmp_path, edit)


def test_unknown_criteria_key_rejected(tmp_path):
    def edit(f):
        f["success_criteria"]["relevance_treshold"] = 0.6
    with pytest.raises(EvalConfigError, match="relevance_treshold: unknown key"):
        load_pair(tmp_path, edit)


def test_freeze_status_and_lists_enforced(tmp_path):
    def edit(f):
        f["status"] = "DRAFT"
        f["objections"] = f["objections"][:3]
        f["requirements"].append(dict(f["requirements"][0]))  # duplicate id
        f["question"] = ""
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, edit)
    m = str(ei.value)
    assert "status='DRAFT'" in m and "objections: need a list of >= 4" in m and "duplicate ids" in m and "question" in m


def test_run_config_missing_keys_listed_and_schema_version(tmp_path):
    def edit(r):
        del r["budget"], r["generator"]
        r["schema_version"] = 2
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, run_edit=edit)
    m = str(ei.value)
    assert "budget: missing" in m and "generator.model: missing" in m and "schema_version=2" in m


@pytest.mark.parametrize("cost,lat", [(0, 600), (-1, 600), (1, 0), (float("inf"), 600), (True, 600), ("1", 600)])
def test_budget_must_be_positive_finite_numbers(tmp_path, cost, lat):
    def edit(r):
        r["budget"]["cost_usd"], r["budget"]["latency_s"] = cost, lat
    with pytest.raises(EvalConfigError, match="budget"):
        load_pair(tmp_path, run_edit=edit)


def test_success_criteria_come_from_freeze_not_run(tmp_path):
    def edit(r):
        r["success_criteria"] = {"relevance_threshold": 0.1}
    with pytest.raises(EvalConfigError, match="unknown top-level key 'success_criteria'"):
        load_pair(tmp_path, run_edit=edit)


def test_scoring_parameter_validated(tmp_path):
    def edit(r):
        r["scoring"]["grounding_max_fabricated"] = -1
        r["scoring"]["typo_key"] = 1
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, run_edit=edit)
    assert "grounding_max_fabricated" in str(ei.value) and "typo_key: unknown key" in str(ei.value)


def test_generator_must_match_model_selection_and_be_mapped(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_model": "something-else"}))
    with pytest.raises(EvalConfigError, match="selected_model='something-else'"):
        load_pair(tmp_path, model_selection_path=sel)
    with pytest.raises(EvalConfigError, match="cannot read selected_model"):
        load_pair(tmp_path, model_selection_path=tmp_path / "missing.json")  # strict: a missing file is an error

    def edit(r):
        del r["model_families"]["gpt-4.1-nano"]
    with pytest.raises(EvalConfigError, match="not in model_families"):
        load_pair(tmp_path, run_edit=edit)


def test_family_entry_requires_family_and_source(tmp_path):
    def edit(r):
        r["model_families"]["x/y"] = {"family": "z"}
        r["model_families"]["a/b"] = {"source": "http://s"}
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, run_edit=edit)
    assert "['x/y'].source" in str(ei.value) and "['a/b'].family" in str(ei.value)


def test_family_for_map_override_and_fail_closed(tmp_path):
    c = load_pair(tmp_path, run_edit=add_models({"m/one": "fam1"}))
    assert family_for("m/one", c).source == "map" and family_for("m/one", c).family == "fam1"
    r = family_for("unknown/x", c, override=" Other ")
    assert (r.family, r.source) == ("other", "override")
    with pytest.raises(EvalConfigError, match="fail closed"):
        family_for("unknown/x", c)


def test_judge_section_validated(tmp_path):
    def edit(r):
        r["judge"]["pool"]["providers"] = []
        r["judge"]["selection"]["max_canary_attempts"] = 0
        r["judge"]["family_patterns"]["generic"] = [["qwen"]]
        r["judge"]["bogus"] = 1
    with pytest.raises(EvalConfigError) as ei:
        load_pair(tmp_path, run_edit=edit)
    m = str(ei.value)
    for needle in ("judge.pool.providers", "max_canary_attempts", "family_patterns.generic", "judge.bogus"):
        assert needle in m


def test_unreadable_or_non_object_files_fail_verbosely(tmp_path):
    with pytest.raises(EvalConfigError, match="cannot read/parse the freeze file"):
        load_eval_config(tmp_path / "nope.json", ec.RUN_PATH)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(EvalConfigError, match="cannot read/parse the run config"):
        load_eval_config(ec.FREEZE_PATH, bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[]")
    with pytest.raises(EvalConfigError, match="must be a JSON object"):
        load_eval_config(ec.FREEZE_PATH, arr)


def test_run_status_must_be_versioned_and_version_present(tmp_path):
    def edit(r):
        r["status"] = "FROZEN"
        r["run_config_version"] = ""
    with pytest.raises(EvalConfigError, match="(?s)status='FROZEN'.*run_config_version"):
        load_pair(tmp_path, run_edit=edit)


def test_ask_service_does_not_import_eval_config():
    """FMEA 10: a bad eval file must never be able to take down /ask."""
    code = "import sys, vera.ask_service; print('vera.eval_config' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent),
                         env={"PATH": "/usr/bin", "EXTERNAL_DB_URL": "postgresql://db-disabled.invalid/none",
                              "OPENAI_API_KEY": "", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
    assert out.stdout.strip() == "False", out.stderr[-500:]


# ---------------------------------------------------------------- review F6: code-side minimum safety set
@pytest.mark.parametrize("edit,needle", [
    (lambda r: r["judge"].__setitem__("forbidden_env_overrides", []), "VERA_JUDGE_ALLOW_SAME_FAMILY"),
    (lambda r: r["judge"].__setitem__("forbidden_env_overrides", ["VERA_JUDGE_GROQ_MODEL"]), "VERA_JUDGE_PRICE_IN"),
    (lambda r: r["judge"]["forbidden_env_overrides"].remove("VERA_JUDGE_PRICE_OUT"), "VERA_JUDGE_PRICE_OUT"),
    (lambda r: r["judge"]["family_patterns"].__setitem__("refuse_openai", ["gpt"]), "openai"),
    (lambda r: r["judge"]["family_patterns"].__setitem__("refuse_openai", ["foo"]), "must always contain"),
    (lambda r: r["judge"]["family_patterns"].__setitem__("refuse_openai", []), "must always contain"),
])
def test_f6_minimum_forbidden_env_and_refuse_openai_enforced_at_load(tmp_path, edit, needle):
    with pytest.raises(EvalConfigError, match=needle):
        load_pair(tmp_path, run_edit=edit)


def test_f6_extra_entries_are_allowed_and_real_config_satisfies_the_minimum(tmp_path):
    c = load_pair(tmp_path, run_edit=lambda r: (r["judge"]["forbidden_env_overrides"].append("VERA_EXTRA"),
                                                r["judge"]["family_patterns"]["refuse_openai"].append("o1-")))
    assert "VERA_EXTRA" in c.judge.forbidden_env_overrides and "o1-" in c.judge.refuse_openai_patterns
    real = load_eval_config()
    assert set(ec.REQUIRED_FORBIDDEN_ENV) <= set(real.judge.forbidden_env_overrides)
    assert set(ec.REQUIRED_REFUSE_OPENAI) <= set(real.judge.refuse_openai_patterns)
