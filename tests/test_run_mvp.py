"""S4 runner tests. FAKES ONLY: no network, no DB, no LLM, no .env. Every seam is injected through Deps;
`requests` and the default vera.db engine/session accessors are patched to raise."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vera import pipeline_llm, run_mvp as rm
from vera.judge_select import Candidate, Selection
from vera.m2_runner import M2Result
from vera.m3.llm import CallLog, CallRecord, LLMError
from vera.m6.eval_rubric import score_cost_latency
from vera.eval_config import load_eval_config
from vera.mvp_preflight import Check, EXPECTED_DB_ROLE, PreflightReport
from vera.pipeline_llm import LLMResult
from tests.test_judge_select import LIVE, Transport, listing
from tests.test_mvp_preflight import FREEZE_Q, FULL_ENV

REPO = Path(__file__).resolve().parent.parent
SENTINEL = "postgresql://sentinel_user:sentinel_pw_98765@db.invalid/none"
M2_COST, M3_CALL_COST, M45_CALL_COST = 0.01, 0.002, 0.001
EDGES = [{"span1_id": 101, "span2_id": 102, "relation_type": "supports", "confidence": 0.9}]


@pytest.fixture(autouse=True)
def no_network_no_default_db(monkeypatch):
    import requests
    import vera.db as db

    def boom(*a, **k):
        raise AssertionError("real network / default DB access attempted in a test")
    for obj, names in ((requests, ("get", "post")), (db, ("get_engine", "get_session"))):
        for n in names:
            monkeypatch.setattr(obj, n, boom)
    monkeypatch.setattr(pipeline_llm, "complete_json",
                        lambda system, user, **k: LLMResult(data={}, model="gpt-4.1-nano", tokens=10,
                                                            cost_usd=M45_CALL_COST))


class FakeClock:
    def __init__(self, step=0.1):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class M3Inner:
    def __init__(self, ok=True):
        self.log, self.ok = CallLog(), ok

    def complete_json(self, system, user, purpose):
        self.log.add(CallRecord(purpose, "gpt-4.1-nano", 10, 5, M3_CALL_COST, 0.1, ok=self.ok,
                                error="" if self.ok else "provider failed"))
        return {}


class Rec:
    """Records the calls the fakes receive."""
    def __init__(self):
        self.calls, self.m3_sources, self.questions, self.m6_run, self.m6_kw = [], None, {}, None, None
        self.store_eval, self.created, self.score_fn_called, self.save_relations = [], [], 0, []


def selection():
    c = Candidate("groq", "qwen/qwen3-32b", "qwen", "pattern:qwen")
    return Selection(judge=object(), candidate=c, canary_failures=[], refused=[], rule="preference-order-v1",
                     listing={"listed_models": ["qwen/qwen3-32b"]}, candidates=[c])


def pf_ok(**kw):
    return PreflightReport(True, [Check("PF1", True, "ok")], selection())


def build(tmp_path, rec=None, *, m3_ok=True, m2_cost=M2_COST, m3_calls=1, m45_calls=2, user=EXPECTED_DB_ROLE,
          preflight=None, env=None, m3_boom=None, swallow_gate_a=False, relations=True,
          m2_unknown=False):
    rec = rec or Rec()

    def run_m2(question, run_id, store, *, ledger, search_fn, score_fn, fetch_fn, **kw):
        rec.calls.append("m2")
        rec.questions["m2"] = question
        search_fn(question, 10, ledger=ledger)
        ledger.record(kind="search", provider="arxiv", cost_usd=m2_cost, units={})
        try:
            score_fn(question, [], cfg=None, ledger=ledger)
        except Exception:
            if not swallow_gate_a:
                raise
        fr = fetch_fn("http://x/a", ledger=ledger)
        ledger.record(kind="fetch", provider="http", cost_usd=0.0, units={})
        if m2_unknown:  # as gate_a.py records a provider failure: cost unknown
            ledger.record(kind="llm", provider="groq", cost_usd=0.0, units={}, ok=False, detail="boom")
        cand = {"url": "http://x/a", "title": "A", "source_id": 11, "source_version": 1, "content_hash": "h",
                "gate_a_score": 0.9, "gate_a_decision": "fetch", "gate_a_rationale": "r"}
        assert fr.ok
        return M2Result(admitted=[cand], iteration_id=1)

    def search_fn(q, n, *, ledger):
        return []

    def score_fn(q, cands, *, cfg, ledger):
        rec.score_fn_called += 1
        return []

    def fetch_fn(url, *, ledger):
        from vera.selective_fetch import FetchResult
        return FetchResult(url=url, ok=True, content_text="FETCHED CONTENT", content_hash="h")

    inner = M3Inner(ok=m3_ok)

    def run_m3(sources, question, *, llm, engine, max_cost_usd, **kw):
        rec.calls.append("m3")
        rec.m3_sources, rec.questions["m3"] = sources, question
        if m3_boom:
            raise RuntimeError(m3_boom)
        for _ in range(m3_calls):
            llm.complete_json("s", "u", "extract")
        return {"sources": [{"source_id": 11, "title": "A", "url": "http://x/a", "content_hash": "h",
                             "decision": "admit", "overall_quality": 4, "scores": {}, "rationale": "r",
                             "spans": [{"span_id": 101, "text": "t", "start_index": 0, "end_index": 1,
                                        "evidence_type": "fact", "relevance_score": 0.9}]}],
                "excluded": [], "counts": {"sources_in": 1}, "failures": [], "stopped_early": None,
                "rubric_version": "v", "gate_b_policy": {"version": "p"}}

    def run_m4_m5(run_id, question, corpus, *, store, max_searches):
        rec.calls.append("m45")
        rec.questions["m45"] = question
        rec.questions["m45_run_id"] = run_id
        assert max_searches == rm.MVP_MAX_SEARCHES
        if relations:
            store.save_relations(run_id, {"edges": EDGES})
        for _ in range(m45_calls):
            try:
                pipeline_llm.call(None, "s", "u")
            except LLMError:  # like stage code: only LLMError may be swallowed
                pass
        return {"final_response": "answer", "claims": [{"claim_text": "c", "evidence_span_ids": [101],
                                                        "verification_status": "supported"}],
                "gate_c_trace": ["adequate"], "gate_c_decision": "adequate",
                "verification_status": "verified", "unsupported_claims": []}

    def run_m6(run, **kw):
        rec.calls.append("m6")
        rec.m6_run, rec.m6_kw = run, kw
        rec.cost_latency = score_cost_latency(run["engineered"], {"cost_usd": load_eval_config().budget.cost_usd, "latency_s": load_eval_config().budget.latency_s})
        return {"evaluation": {"outcome": "success", "failed_dimensions": []}, "cost_summary": {"complete": True},
                "baseline": {"response_text": "base"}}

    class Store:
        def save_relations(self, run_id, graph):
            rec.save_relations.append((run_id, graph))

    def create_run(eng, qid, question):
        rec.created.append((qid, question))
        return 4242

    deps = rm.Deps(
        engine_factory=lambda role: ("engine", role), current_user=lambda eng: user,
        question_text=lambda eng, q: FREEZE_Q, create_run=create_run,
        store_eval=lambda eng, rid, report: rec.store_eval.append((rid, report)),
        m2_store_factory=lambda eng: object(), pipeline_store_factory=lambda eng: Store(),
        run_m2=run_m2, search_fn=search_fn, score_fn=score_fn, fetch_fn=fetch_fn,
        m3_client_factory=lambda: inner, run_m3=run_m3, run_m4_m5=run_m4_m5, run_m6=run_m6,
        preflight=preflight or (lambda **kw: pf_ok()), baseline_llm_call=lambda q, m: {},
        judge_transport=Transport(), http_get=listing(*LIVE),
        env=env or dict(FULL_ENV, SECRET_TOKEN=SENTINEL), repo_root=REPO, run_dir=tmp_path / ".vera-runs",
        git_info=lambda: {"head": "abc", "dirty": False}, now=lambda: datetime(2026, 10, 2, tzinfo=timezone.utc),
        clock=FakeClock())
    return deps, rec


@pytest.fixture(autouse=True)
def fake_adapter(monkeypatch):
    def adapt(m3, requirements, **kw):
        pipeline_llm.call(None, "tag", "u")  # one metered tagger call, as the real adapter makes
        return {"spans": [{"span_id": 101}], "sub_questions": [], "tagging": {"coverage": {}}}
    monkeypatch.setattr(rm, "adapt_m3_to_m4", adapt)


def go(deps, stage="L1", run_type=None, **lim):
    run_type = run_type or rm.STAGE_RUN_TYPE[stage]
    return rm.run_mvp(2, stage, run_type, deps=deps, limits=rm.Limits(budget_confirmed=True, **lim))


# ---- happy paths -------------------------------------------------------------------------------
def test_l1_end_to_end(tmp_path, monkeypatch):
    deps, rec = build(tmp_path)
    out = go(deps, "L1")
    assert out.exit_code == 0 and out.state == "closed_ok", out.data.get("error")
    assert rec.calls == ["m2", "m3", "m45", "m6"]
    assert rec.questions["m45_run_id"] == "4242" and isinstance(out.data["run_id"], int)
    assert len(rec.store_eval) == 1 and isinstance(rec.store_eval[0][0], int)
    u = rec.m6_run["engineered"]["usage"]
    assert u["complete"] is True
    assert u["cost_usd"] == pytest.approx(M2_COST + M3_CALL_COST + 2 * M45_CALL_COST + M45_CALL_COST)
    assert rec.cost_latency["score"] != 1
    assert rec.m6_kw["store"] is False and rec.m6_kw["judge"] is not None
    assert rec.m6_kw["llm_call"] is deps.baseline_llm_call
    data = json.loads(out.path.read_text())
    assert data["judge"]["model"] == "qwen/qwen3-32b" and data["judge"]["rule"] == "preference-order-v1"
    assert data["divergences"] and data["account_role"] == EXPECTED_DB_ROLE
    reg = (out.path.parent / rm.REGISTER_NAME).read_text().splitlines()
    assert json.loads(reg[-1])["state"] == "closed_ok"


def test_stage_c_never_stores_eval(tmp_path):
    deps, rec = build(tmp_path)
    out = go(deps, "C")
    assert out.exit_code == 0 and rec.calls[-1] == "m6" and rec.store_eval == []


def test_stage_a_and_b_stop_early(tmp_path):
    deps, rec = build(tmp_path)
    assert go(deps, "A").exit_code == 0 and rec.calls == ["m2", "m3"]
    deps, rec = build(tmp_path)
    assert go(deps, "B").exit_code == 0 and rec.calls == ["m2", "m3", "m45"]


def test_frozen_question_reaches_stages_and_run_row(tmp_path):
    deps, rec = build(tmp_path)
    go(deps, "B")
    assert "–" in FREEZE_Q
    assert rec.questions["m2"] == rec.questions["m3"] == rec.questions["m45"] == FREEZE_Q
    assert rec.created == [(2, FREEZE_Q)]


def test_reused_source_content_comes_from_the_fetch(tmp_path):
    deps, rec = build(tmp_path)
    go(deps, "A")
    assert rec.m3_sources[0]["content"] == "FETCHED CONTENT" and rec.m3_sources[0]["source_id"] == 11


def test_relations_from_save_relations_reach_build_engineered(tmp_path, monkeypatch):
    seen = {}
    real = rm.build_engineered

    def spy(m5, m3, **kw):
        seen.update(kw)
        return real(m5, m3, **kw)
    monkeypatch.setattr(rm, "build_engineered", spy)
    deps, rec = build(tmp_path)
    go(deps, "C")
    assert seen["relations"] == {"edges": EDGES}
    assert rec.save_relations == [("4242", {"edges": EDGES})]  # store still received the graph unaltered


def test_attach_usage_called_and_complete_survives(tmp_path, monkeypatch):
    calls = []
    real = rm.attach_usage

    def spy(eng, usage):
        calls.append(usage)
        return real(eng, usage)
    monkeypatch.setattr(rm, "attach_usage", spy)
    deps, rec = build(tmp_path)
    go(deps, "C")
    assert len(calls) == 1 and rec.m6_run["engineered"]["usage"]["complete"] is True


# ---- budget / usage ------------------------------------------------------------------------------
def test_budget_trip_in_m3_exit_3_no_m45(tmp_path):
    deps, rec = build(tmp_path, m3_calls=5)
    out = go(deps, "C", cap_usd=0.011, cap_s=600.0)
    assert out.exit_code == 3 and out.state == "budget_stop"
    assert "m45" not in rec.calls and "m6" not in rec.calls
    err = out.data["error"]
    assert "stage=m3" in err and "activity=extract" in err and "cap=$0.0110" in err
    assert out.data["failed_stage"] == "m3"


def test_budget_trip_inside_m45_not_swallowed(tmp_path):
    deps, rec = build(tmp_path, m45_calls=5)
    out = go(deps, "C", cap_usd=0.0125, cap_s=600.0)
    assert out.exit_code == 3 and "m6" not in rec.calls and out.data["failed_stage"] == "m45"
    assert "stage=m45" in out.data["error"]


def test_time_cap_trips(tmp_path):
    deps, rec = build(tmp_path)
    out = go(deps, "C", cap_usd=1.0, cap_s=0.25)
    assert out.exit_code == 3 and "time" in out.data["error"]


def test_incomplete_usage_exit_4_before_m6(tmp_path):
    deps, rec = build(tmp_path, m3_ok=False)
    out = go(deps, "C")
    assert out.exit_code == 4 and out.state == "usage_incomplete"
    assert "m6" not in rec.calls and rec.store_eval == []
    assert out.data["usage"]["complete"] is False and out.data["usage"]["incomplete_reasons"]


def test_m2_unknown_cost_stops_before_m3(tmp_path):
    deps, rec = build(tmp_path, m2_unknown=True)
    out = go(deps, "C")
    assert out.exit_code == 4 and out.state == "usage_incomplete" and out.data["failed_stage"] == "m2"
    assert rec.calls == ["m2"] and rec.store_eval == []
    assert "after m2" in out.data["error"] and "m2/" in out.data["error"]


def test_m3_unknown_cost_stops_before_m45(tmp_path):
    deps, rec = build(tmp_path, m3_ok=False)
    out = go(deps, "C")
    assert out.exit_code == 4 and out.data["failed_stage"] == "m3"
    assert rec.calls == ["m2", "m3"] and "after m3" in out.data["error"]


def test_unknown_cost_does_not_stop_early_when_m6_not_planned(tmp_path):
    deps, rec = build(tmp_path, m3_ok=False)
    out = go(deps, "B")  # no M6 in plan: unchanged behaviour
    assert out.exit_code == 0 and rec.calls == ["m2", "m3", "m45"]


def test_complete_usage_proceeds_through_all_stages(tmp_path):
    deps, rec = build(tmp_path)
    assert go(deps, "C").exit_code == 0 and rec.calls == ["m2", "m3", "m45", "m6"]


def test_budget_checked_outside_gate_a_before_score_fn(tmp_path):
    deps, rec = build(tmp_path, m2_cost=5.0)  # search spend alone exceeds the cap
    out = go(deps, "A")
    assert out.exit_code == 3 and rec.score_fn_called == 0 and out.data["failed_stage"] == "m2"
    assert "activity=gate_a" in out.data["error"]


def test_trip_swallowed_by_gate_a_still_stops_run(tmp_path):
    # Gate A swallows every exception from its scorer; the sticky trip must still stop M2, not defer.
    deps, rec = build(tmp_path, m2_cost=5.0, swallow_gate_a=True)
    out = go(deps, "A")
    assert out.exit_code == 3 and "m3" not in rec.calls


# ---- refusals / failures -------------------------------------------------------------------------
def test_preflight_failure_no_create_run_exit_1(tmp_path):
    bad = lambda **kw: PreflightReport(False, [Check("PF8", False, "wrong role", "fix")], None)  # noqa: E731
    deps, rec = build(tmp_path, preflight=bad)
    out = go(deps, "L1")
    assert out.exit_code == 1 and out.state == "preflight_failed" and rec.created == [] and rec.calls == []
    assert out.path.name.startswith("preflight-")


def test_role_mismatch_refused_via_real_preflight(tmp_path):
    from vera.mvp_preflight import run_preflight
    import functools
    pre = functools.partial(run_preflight, check_ignored=lambda rel: True, generator_model="gpt-4.1-nano",
                            pricing_lookup=lambda m: object())
    deps, rec = build(tmp_path, user="postgres", preflight=pre)
    out = go(deps, "L1")
    assert out.exit_code == 1 and rec.created == [] and rec.calls == []
    assert any(c["id"] == "PF8" and not c["ok"] for c in out.data["preflight"])


def test_role_mismatch_belt_and_braces(tmp_path):
    deps, rec = build(tmp_path, user="postgres")  # fake pre-flight passes, identity re-check refuses
    out = go(deps, "L1")
    assert out.exit_code == 1 and rec.created == [] and "role mismatch" in out.data["error"]


def test_run_mvp_refuses_without_budget_confirmed(tmp_path):
    deps, rec = build(tmp_path)
    engines = []
    deps.engine_factory = lambda role: engines.append(role)
    out = rm.run_mvp(2, "L1", "live", deps=deps, limits=rm.Limits())
    assert out.exit_code == 1 and out.state == "preflight_failed" and "budget_confirmed" in out.data["error"]
    assert rec.created == [] and rec.calls == [] and rec.store_eval == [] and engines == []


def test_report_write_failure_does_not_overwrite_closed_ok(tmp_path, monkeypatch):
    deps, rec = build(tmp_path)
    real_open = open

    def flaky_open(file, mode="r", *a, **k):
        if str(file).endswith(rm.REGISTER_NAME) and "a" in mode:
            raise OSError("register disk full")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr("builtins.open", flaky_open)
    with pytest.raises(OSError, match="register disk full"):
        go(deps, "L1")
    monkeypatch.undo()
    report = json.loads((deps.run_dir / "4242.json").read_text())
    assert report["state"] == "closed_ok" and report["exit_code"] == 0 and "error" not in report
    assert len(rec.store_eval) == 1  # the eval was stored; the failed register write did not retry the run


def test_divergences_disclose_keyless_cost_basis_and_seed_fallback():
    assert any("no_published_price_keyless" in d and "assumption" in d for d in rm.DIVERGENCES)
    assert any("seed-list fallback" in d for d in rm.DIVERGENCES)


def test_mismatched_stage_and_run_type_refused(tmp_path):
    deps, rec = build(tmp_path)
    with pytest.raises(rm.RunConfigError):
        rm.run_mvp(2, "L1", "test", deps=deps, limits=rm.Limits())
    with pytest.raises(rm.RunConfigError):
        rm.run_mvp(2, "C", "live", deps=deps, limits=rm.Limits())
    assert rec.created == []


def test_stage_failure_exit_2_secret_scrubbed(tmp_path):
    deps, rec = build(tmp_path, m3_boom=f"connect failed to {SENTINEL}")
    out = go(deps, "C")
    assert out.exit_code == 2 and out.state == "failed" and out.data["failed_stage"] == "m3"
    for p in list(out.path.parent.iterdir()):
        assert "sentinel_pw_98765" not in p.read_text() and SENTINEL not in p.read_text()
    assert "traceback" in out.data


def test_no_default_db_or_network_touched(tmp_path):
    deps, rec = build(tmp_path)
    assert go(deps, "C").exit_code == 0  # autouse fixture patched get_engine/get_session/requests to raise


# ---- CLI: no live without the flag ---------------------------------------------------------------
def _no_deps():
    raise AssertionError("Deps.real / live path must not be built in this mode")


def test_default_mode_is_dry_and_never_builds_deps(capsys):
    rc = rm.main(["--question-id", "2", "--stage", "A", "--run-type", "test"], env=FULL_ENV, deps_factory=_no_deps)
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out and "NOT executed" in out


def test_dry_mode_with_missing_names_fails_cleanly(capsys):
    rc = rm.main(["--question-id", "2", "--stage", "A", "--run-type", "test"], env={}, deps_factory=_no_deps)
    assert rc == 1 and "DRY RUN" in capsys.readouterr().out


def test_live_flag_needs_budget_confirmation(capsys):
    rc = rm.main(["--question-id", "2", "--stage", "A", "--run-type", "test", "--live"], env=FULL_ENV,
                 deps_factory=_no_deps)
    assert rc == 1 and "budget-confirmed" in capsys.readouterr().err


def test_live_refuses_when_preflight_fails(tmp_path):
    bad = lambda **kw: PreflightReport(False, [Check("PF3", False, "no key")], None)  # noqa: E731
    deps, rec = build(tmp_path, preflight=bad)
    rc = rm.main(["--question-id", "2", "--stage", "A", "--run-type", "test", "--live", "--budget-confirmed"],
                 deps_factory=lambda: deps)
    assert rc == 1 and rec.created == [] and rec.calls == []


def test_cli_rejects_stage_run_type_mismatch(capsys):
    rc = rm.main(["--question-id", "2", "--stage", "L1", "--run-type", "test"], deps_factory=_no_deps)
    assert rc == 1 and "requires --run-type live" in capsys.readouterr().err


# ---- vera.db.get_role_engine ---------------------------------------------------------------------
def test_get_role_engine_uses_injected_factory_and_env_name(monkeypatch):
    import vera.db as db
    seen = {}

    def factory(url, **kw):
        seen.update(url=url, kw=kw)
        return "ENGINE"
    assert db.get_role_engine("rw", env={"VERA_DB_URL_RW": "postgresql://u:p@h/d"}, engine_factory=factory) == "ENGINE"
    assert seen["url"] == "postgresql://u:p@h/d"
    with pytest.raises(NotImplementedError, match="goal #3"):
        db.get_role_engine("ro", env={}, engine_factory=factory)
    with pytest.raises(RuntimeError, match="VERA_DB_URL_RW") as ei:
        db.get_role_engine("rw", env={}, engine_factory=factory)
    assert "postgresql" not in str(ei.value)


def test_constants_are_imported_not_redefined():
    from vera import db, mvp_preflight
    assert rm.EXPECTED_DB_ROLE is mvp_preflight.EXPECTED_DB_ROLE and db.DB_URL_ENV_RW is mvp_preflight.DB_URL_ENV_RW


# ---------------------------------------------------------------- review F1/F2/F7/F8 (goal #2 fix round)
def test_f1_f2_m6_receives_the_values_the_run_used(tmp_path, monkeypatch):
    sqs = [{"id": f"r{i}", "text": f"t{i}", "required": True} for i in range(8)]

    def adapt(m3, requirements, **kw):
        pipeline_llm.call(None, "tag", "u")
        return {"spans": [{"span_id": 101}], "sub_questions": sqs, "tagging": {"coverage": {}}}
    monkeypatch.setattr(rm, "adapt_m3_to_m4", adapt)
    deps, rec = build(tmp_path)
    out = go(deps, "L1", num_results=7, max_fetch=3, max_sources=2, cap_usd=0.9)
    assert out.exit_code == 0, out.data.get("error")
    ei = rec.m6_run["effective_inputs"]
    assert ei["m4"] == {"max_searches": rm.MVP_MAX_SEARCHES, "sub_questions": sqs} and ei["m4"]["max_searches"] == 0
    assert ei["stage_limits"] == {"num_results": 7, "max_fetch": 3, "max_sources": 2, "cap_usd": 0.9,
                                  "cap_s": load_eval_config().budget.latency_s}


def test_f8_the_validated_config_is_passed_to_preflight_and_m6(tmp_path):
    seen = {}

    def pf(**kw):
        seen["pf_config"] = kw.get("config")
        return pf_ok()
    deps, rec = build(tmp_path, preflight=pf)
    assert go(deps, "L1").exit_code == 0
    assert seen["pf_config"] is not None and seen["pf_config"] is rec.m6_kw["config"]  # one load, shared


def test_f7_a_raising_preflight_is_reported_not_raised(tmp_path):
    def boom(**kw):
        raise KeyError("mistral")
    deps, rec = build(tmp_path, preflight=boom)
    out = go(deps, "L1")
    assert out.exit_code == rm.EXIT_PREFLIGHT and out.state == rm.STATE_PREFLIGHT_FAILED
    data = json.loads(out.path.read_text())
    assert "pre-flight raised KeyError" in data["error"] and rec.created == []
    assert json.loads((out.path.parent / rm.REGISTER_NAME).read_text().splitlines()[-1])["state"] == "preflight_failed"


def test_pf5_prefix_limitation_is_disclosed():
    assert any("startswith 'gpt'" in d for d in rm.DIVERGENCES)


def test_search_snapshot_hash_in_report_and_fingerprint_input(tmp_path):
    from vera.m6.fingerprint import canonical_json, sha256_hex
    deps, rec = build(tmp_path)
    out = go(deps, "L1")
    assert out.exit_code == 0, out.data.get("error")
    snap = out.data["search"][0]
    assert snap["candidate_list_sha256"] == sha256_hex(canonical_json(snap["candidates"]))
    ei = rec.m6_run["effective_inputs"]["search"][0]
    assert ei["candidate_list_sha256"] == snap["candidate_list_sha256"]


def test_search_candidate_list_changes_the_fingerprint():
    from tests.test_m6_fingerprint import _eff
    from vera.m6 import fingerprint as FP
    a = FP.run_fingerprint(_eff(run_inputs={"search": [{"candidate_list_sha256": "a" * 64}]}), code_ver={})
    b = FP.run_fingerprint(_eff(run_inputs={"search": [{"candidate_list_sha256": "b" * 64}]}), code_ver={})
    assert a["config_sha256"] != b["config_sha256"]
    assert "search" in _eff()["run_config"]  # the configured chain/limits are fingerprinted too


def test_run_and_preflight_need_no_search_key(tmp_path):
    deps, _ = build(tmp_path, env={})  # no search key anywhere in the environment
    assert go(deps, "L1").exit_code == 0
