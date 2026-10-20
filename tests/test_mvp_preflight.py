"""S3 pre-flight tests. FAKES ONLY: no network, no DB, no provider calls, no .env."""
import json
from pathlib import Path

import pytest

from vera import mvp_preflight as pf
from vera.m6.judge import JudgeLimits
from tests.test_judge_select import ENV as KEYS, LIVE, SECRET, Transport, listing

REPO = Path(__file__).resolve().parent.parent
FULL_ENV = dict(KEYS, VERA_DB_URL_RW="postgresql://u:pw123456@h/db")
FREEZE_Q = json.loads((REPO / "config" / "vera_eval_freeze.json").read_text())["question"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("real network call attempted in a test")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    import vera.db as db
    monkeypatch.setattr(db, "get_engine", boom, raising=False)
    monkeypatch.setattr(db, "get_session", boom, raising=False)


def run(**kw):
    base = dict(question_id=2, env=FULL_ENV, repo_root=REPO, limits=JudgeLimits(),
                judge_transport=Transport(), http_get=listing(*LIVE),
                db_identity=lambda: pf.EXPECTED_DB_ROLE, question_text_for=lambda q: FREEZE_Q,
                check_ignored=lambda rel: True, generator_model="gpt-4.1-nano",
                pricing_lookup=lambda m: object())
    base.update(kw)
    return pf.run_preflight(**base)


def failed(rep):
    return {c.id for c in rep.checks if not c.ok}


def test_constants():
    assert pf.EXPECTED_DB_ROLE == "vera_claude_code_rw" and pf.DB_URL_ENV_RW == "VERA_DB_URL_RW"
    # the interim hardcoded pin is retired: the expected hash is sourced from config/vera_eval_run.json
    assert not hasattr(pf, "EXPECTED_FREEZE_HASH")
    from vera.eval_config import load_eval_config
    assert load_eval_config().requirements_hash16 == "637d247a50217128"


def test_all_green_with_fakes():
    rep = run()
    assert rep.ok, rep.render()
    assert [c.id for c in rep.checks] == [f"PF{i}" for i in range(1, 14)]
    assert rep.selection.candidate.family == "qwen"


@pytest.mark.parametrize("name", list(pf.REQUIRED_ENV_NAMES))
def test_each_missing_env_name_fails_by_name(name):
    env = {k: v for k, v in FULL_ENV.items() if k != name}
    rep = run(env=env, canary=False)
    pf1 = rep.checks[0]
    assert not pf1.ok and name in pf1.detail


def test_secrets_never_in_output():
    rep = run(http_get=lambda *a: (_ for _ in ()).throw(RuntimeError(f"401 {SECRET}")))
    text = rep.render()
    for v in FULL_ENV.values():
        assert v not in text
    assert not rep.ok


def test_judge_failures_map_to_checks():
    rep = run(http_get=listing())  # empty listing
    assert {"PF3", "PF4", "PF6"} <= failed(rep)
    rep = run(http_get=listing("openai/gpt-oss-120b", "zzz-9"))  # nothing eligible
    assert "PF4" in failed(rep) and "PF6" in failed(rep)
    rep = run(judge_transport=Transport(fail=LIVE))  # canary failure
    assert failed(rep) == {"PF6"}
    rep = run(env=dict(FULL_ENV, VERA_JUDGE_POOL="groq:not-live"))  # model not in live listing
    assert "PF4" in failed(rep)
    rep = run(env=dict(FULL_ENV, VERA_JUDGE_POOL="mistral:x"))
    assert "PF2" in failed(rep)


def test_non_gpt_generator_and_missing_pricing_fail():
    assert "PF5" in failed(run(generator_model="mistral-large"))
    assert "PF7" in failed(run(pricing_lookup=lambda m: None))


def test_wrong_role_question_mismatch_not_ignored_fail():
    assert "PF8" in failed(run(db_identity=lambda: "vera_vjay_user"))
    assert "PF9" in failed(run(question_text_for=lambda q: "some other question"))
    assert "PF11" in failed(run(check_ignored=lambda rel: False))


def test_freeze_hash_mismatch_on_temp_copy(tmp_path):
    (tmp_path / "config").mkdir()
    d = json.loads((REPO / "config" / "vera_eval_freeze.json").read_text())
    d["requirements"][0] = dict(d["requirements"][0], text="tampered")
    (tmp_path / "config" / "vera_eval_freeze.json").write_text(json.dumps(d))
    rep = run(repo_root=tmp_path)
    assert "PF10" in failed(rep)
    # real file untouched


def test_no_db_and_no_canary_skips_stated():
    rep = run(require_db=False, canary=False, listing=False,
              http_get=lambda *a: pytest.fail("offline must not list"),
              judge_transport=lambda *a: pytest.fail("offline must not call"),
              db_identity=lambda: pytest.fail("no db"), question_text_for=lambda q: pytest.fail("no db"))
    skipped = {c.id for c in rep.checks if c.skipped}
    assert skipped == {"PF3", "PF4", "PF6", "PF8", "PF9"} and rep.ok and rep.selection is None
    assert "SKIPPED" in rep.render()


def test_canary_off_but_listing_on():
    rep = run(canary=False, judge_transport=lambda *a: pytest.fail("no canary"))
    assert rep.ok and rep.selection is None
    assert next(c for c in rep.checks if c.id == "PF6").skipped


def test_key_absent_listing_not_attempted():
    env = {k: v for k, v in FULL_ENV.items() if k != "GROQ_API_KEY"}
    rep = run(env=env, http_get=lambda *a: pytest.fail("must not list"))
    assert "PF3" in failed(rep)


def test_preflight_fails_before_any_baseline_call():
    """Fake M6 sequence: baseline is only reachable if pre-flight passed (as run_mvp does)."""
    calls = []

    def fake_m6_sequence(report):
        if not report.ok:
            return "stopped"
        calls.append("baseline")
        calls.append("judge_first_call")
        return "ran"

    transport = Transport(fail=LIVE)  # every canary fails
    rep = run(judge_transport=transport)
    assert fake_m6_sequence(rep) == "stopped" and calls == []
    ok = run()
    assert fake_m6_sequence(ok) == "ran" and calls == ["baseline", "judge_first_call"]


def test_cli_no_db_is_offline_and_prints_missing_names(monkeypatch, capsys):
    import vera.mvp_preflight as m
    monkeypatch.setattr(m, "default_http_get", lambda *a: pytest.fail("offline CLI listed"))
    monkeypatch.setattr(m, "default_judge_transport", lambda *a: pytest.fail("offline CLI called"))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: pytest.fail("loaded .env"), raising=False)
    rc = m.main(["--question-id", "2", "--no-db"], env={}, repo_root=REPO)
    out = capsys.readouterr().out
    assert rc == 1
    for n in pf.REQUIRED_ENV_NAMES:
        assert n in out
    assert "SKIPPED" in out and "PREFLIGHT FAILED" in out


def test_cli_no_db_with_all_names_passes_offline(capsys):
    rc = pf.main(["--question-id", "2", "--no-db"], env=FULL_ENV, repo_root=REPO)
    out = capsys.readouterr().out
    # PF11 may fail on a tree where .vera-runs/ is not yet ignored (S4 adds it); nothing else may
    assert all(l.startswith(("PF11", "PREFLIGHT")) or "FAIL" not in l for l in out.splitlines()), out
    for v in FULL_ENV.values():
        assert v not in out


# ---- S5 F2: the canary is last and never spends when any other check failed -------------------
def _counting_transport():
    t = Transport()
    return t


@pytest.mark.parametrize("kw", [
    dict(db_identity=lambda: "vera_vjay_user"),                       # PF8 wrong role
    dict(question_text_for=lambda q: "some other question"),          # PF9 mismatch
    dict(check_ignored=lambda rel: False),                            # PF11
    dict(pricing_lookup=lambda m: None),                              # PF7
    dict(generator_model="mistral-large"),                            # PF5 (and PF12)
    dict(baseline_model="gpt-4.1-mini"),                              # PF12
    dict(env={k: v for k, v in FULL_ENV.items() if k != "VERA_DB_URL_RW"}),  # PF1
])
def test_no_canary_spend_when_any_other_check_fails(kw):
    t = _counting_transport()
    rep = run(judge_transport=t, **kw)
    assert t.bodies == [], "a transport (spend) call happened despite a failed check"
    pf6 = next(c for c in rep.checks if c.id == "PF6")
    assert pf6.skipped and "other check" in pf6.detail
    assert not rep.ok and rep.selection is None


def test_no_canary_spend_on_freeze_hash_mismatch(tmp_path):
    (tmp_path / "config").mkdir()
    d = json.loads((REPO / "config" / "vera_eval_freeze.json").read_text())
    d["requirements"][0] = dict(d["requirements"][0], text="tampered")
    (tmp_path / "config" / "vera_eval_freeze.json").write_text(json.dumps(d))
    t = _counting_transport()
    rep = run(repo_root=tmp_path, judge_transport=t)
    assert t.bodies == [] and "PF10" in failed(rep)


def test_canary_runs_when_all_other_checks_pass_and_is_last():
    t = _counting_transport()
    rep = run(judge_transport=t)
    assert rep.ok and len(t.bodies) >= 1 and rep.selection is not None
    assert [c.id for c in rep.checks] == [f"PF{i}" for i in range(1, 14)]  # shape unchanged


# ---- S5 F7: baseline / Gate A model equals the generator model --------------------------------
def test_baseline_model_constants_equal_generator_by_default():
    from vera.gate_a import MODEL
    from vera.m6.baseline_collection import DEFAULT_MODEL
    assert pf.BASELINE_MODEL is DEFAULT_MODEL and pf.GATE_A_MODEL is MODEL  # imported, not duplicated
    pf12 = next(c for c in run().checks if c.id == "PF12")
    assert pf12.ok


def test_pf12_fails_on_baseline_or_gate_a_mismatch_with_clear_message():
    for kw in (dict(baseline_model="gpt-4.1-mini"), dict(gate_a_model="gpt-4.1-mini")):
        rep = run(canary=False, **kw)
        pf12 = next(c for c in rep.checks if c.id == "PF12")
        assert not pf12.ok and "do not all equal the generator model" in pf12.detail
        assert "gpt-4.1-mini" in pf12.detail and "FIX" in rep.render()


# ---------------------------------------------------------------- review F7/F8 (goal #2 fix round)
def test_f7_unsupported_provider_is_reported_by_pf3_not_raised(monkeypatch):
    import dataclasses
    from vera.eval_config import load_eval_config
    base = load_eval_config()
    bad = dataclasses.replace(base, judge=dataclasses.replace(base.judge, providers=("mistral",)))
    monkeypatch.setattr(pf, "load_eval_config", lambda *a, **k: bad)  # a hand-built config bypassing the loader
    rep = run()  # must not raise
    assert "PF3" in failed(rep) and "mistral" in next(c.detail for c in rep.checks if c.id == "PF3")
    assert rep.selection is None


def test_f8_preflight_uses_the_passed_config_for_the_canary_judge(tmp_path):
    from tests.config_support import load_pair
    cfg = load_pair(tmp_path, run_edit=lambda r: r["judge"]["selection"].__setitem__("rule", "rule-from-passed-config"))
    rep = run(config=cfg)
    assert rep.ok, rep.render()
    assert rep.selection.rule == "rule-from-passed-config"  # not the default config's rule
    assert run().selection.rule != "rule-from-passed-config"


def test_pf13_search_chain_needs_no_key_and_checks_seed(tmp_path):
    rep = run()
    pf13 = next(c for c in rep.checks if c.id == "PF13")
    assert pf13.ok and "no key required" in pf13.detail and "arxiv" in pf13.detail
    from vera.eval_config import load_eval_config
    import dataclasses
    cfg = load_eval_config()
    bad = dict(cfg.search, fallback_seed={"path": "missing-seed.json", "sha256": "0" * 64})
    rep = run(config=dataclasses.replace(cfg, search=bad))
    assert "PF13" in failed(rep) and "not found" in next(c.detail for c in rep.checks if c.id == "PF13")
