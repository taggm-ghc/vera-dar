"""CMF-1022-V2 (independent verifier): boundary tests for the corpus-miss inline-retry latency bound.

I1 REVISION (OD-CMF-1022g, shared request deadline): remaining = budget.deadline - monotonic() where the deadline is
fixed once per request (ASK_TIMEOUT_S after its start); need = ask_service.inline_need_s = inline.deadline_s +
POST_INLINE_LLM_CALLS (3) * chain timeout_s + the reading-list tail (2 x grounding deadline_s, 0 when ungrounded, as in
this rig). Rig need = 45 + 3*20 = 105 s (was 85 s: 45 + 2*20 with no tail). The retry is SKIPPED iff remaining < need (strict),
so remaining == need RUNS. 105.1 -> run, 105.0 -> run, 104.9 -> skip. Unreadable terms now skip (budget_unknown). No real sleeps: a fake monotonic
clock is injected into ask_service. Offline; the router, fulldoc, acquire and reading list are stubbed."""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import ask_service, fulldoc_corpus, inline_acquire, public_mode, rate_limit

ROOT = Path(__file__).resolve().parent.parent
T0 = 1000.0
RL_TEXT = "TEXT"


@pytest.fixture
def rig(monkeypatch):
    calls = {"inline": 0, "acquire": 0}
    state = {"elapsed": 0.0, "n": 0}

    def mono():  # first call = request start (t0); every later call = t0 + elapsed
        state["n"] += 1
        return T0 if state["n"] == 1 else T0 + state["elapsed"]
    monkeypatch.setattr(ask_service, "time", SimpleNamespace(monotonic=mono))

    acq = {"inline": {"in_scope": True, "deadline_s": 45}}
    dec = SimpleNamespace(accepted=True, layer="layer2_llm", reason="llm_in:x", tokens_used=3, cost_usd=0.0)
    import vera.scope_router as sr
    monkeypatch.setattr(sr, "route", lambda *a, **k: dec)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: acq)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_acq_on_insufficient", lambda a: False)
    monkeypatch.setattr(inline_acquire, "icfg_of", lambda a: a["inline"])
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    ft = fulldoc_corpus.load_config()["fallback_tokens"]

    def fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=False, ephemeral=None, used=None):
        return (None, ft["insufficient"], 1, 0.0, RL_TEXT)
    monkeypatch.setattr(ask_service, "_fulldoc", fulldoc)

    def inline(*a, **k):
        calls["inline"] += 1
        return "INLINE-NOTE"
    monkeypatch.setattr(ask_service, "_inline", inline)
    monkeypatch.setattr(inline_acquire, "acquire", lambda *a, **k: calls.__setitem__("acquire", calls["acquire"] + 1))

    def reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=None, cost=None):
        return ask_service.AskResult(answer="RL", tokens_used=tokens or 0, cost_usd=cost or 0.0,
                                     result_kind="reading_list", contract={"layers": ["L0"]})
    monkeypatch.setattr(ask_service, "_reading_list", reading_list)

    def run(remaining=None, elapsed=None):
        state["elapsed"] = (public_mode.ASK_TIMEOUT_S - remaining) if elapsed is None else elapsed
        state["n"] = 0
        return ask_service.answer_in_scope("What is X?", "m", SimpleNamespace(input=0, output=0))
    return SimpleNamespace(calls=calls, run=run, acq=acq, monkeypatch=monkeypatch)


def test_threshold_is_derived_from_config_not_hardcoded():
    chain = ask_service._chain_timeout_s()
    deadline_s = float(inline_acquire._b({"inline": {"deadline_s": 45}}, "deadline_s", 45))
    assert (chain, deadline_s, public_mode.ASK_TIMEOUT_S) == (20.0, 45.0, 180)
    assert deadline_s + ask_service.POST_INLINE_LLM_CALLS * chain == 105.0  # ungrounded rig
    from vera.grounded_ask import load_grounding
    gd = load_grounding()["deadline_s"]
    assert ask_service.inline_need_s({"inline": {"deadline_s": 45}}, (None, {})) == 105.0 + ask_service.READING_LIST_SEARCHES * gd == 125.0


@pytest.mark.parametrize("remaining,skipped", [(105.1, False), (105.0, False), (104.9, True)])
def test_boundary_105s(rig, remaining, skipped):
    # elapsed values are exactly representable for 75.0; 74.9/75.1 differ from 105.0 by 0.1, far above float error
    r = rig.run(remaining=remaining)
    assert rig.calls["inline"] == (0 if skipped else 1)
    assert rig.calls["acquire"] == 0  # _inline is stubbed; acquire is never reached directly either way
    if skipped:
        assert r.result_kind == "reading_list" and r.answer == RL_TEXT


def test_exact_boundary_uses_strict_less_than(rig):
    rig.run(elapsed=75.0)  # remaining 105.0 == need 105.0 -> runs
    assert rig.calls["inline"] == 1
    rig.run(elapsed=75.0001)  # remaining just below need -> skipped
    assert rig.calls["inline"] == 1


def test_elapsed_zero_runs_retry(rig):
    r = rig.run(elapsed=0.0)
    assert rig.calls["inline"] == 1 and "INLINE-NOTE" in r.answer


@pytest.mark.parametrize("elapsed", [180.0, 181.0, 500.0])
def test_elapsed_at_or_beyond_180_skips_even_with_negative_remaining(rig, elapsed):
    r = rig.run(elapsed=elapsed)
    assert rig.calls["inline"] == 0 and rig.calls["acquire"] == 0
    assert r.result_kind == "reading_list" and r.answer == RL_TEXT
    assert r.contract["fallback"] == fulldoc_corpus.load_config()["fallback_tokens"]["insufficient"]


def test_skipped_retry_has_no_inline_trace_and_no_acquire_trigger_or_spend_override(rig):
    r = rig.run(remaining=10.0)
    # CR-1: the skip is now reported as inline_acquisition {"status": "deadline"}; no inline_fallback token
    assert "inline_fallback" not in r.contract and r.contract["inline_acquisition"] == {"status": "deadline"}
    assert rig.calls["inline"] == 0


def test_skipped_retry_abstention_path_is_unchanged(rig, monkeypatch):
    """A non-answer token that is not an insufficiency (a decline-style result) never reaches the retry either way."""
    ft = fulldoc_corpus.load_config()["fallback_tokens"]
    monkeypatch.setattr(ask_service, "_fulldoc",
                        lambda *a, **k: (None, "claim_check_disabled", 1, 0.0, RL_TEXT))
    r = rig.run(elapsed=0.0)
    assert rig.calls["inline"] == 0 and r.result_kind == "reading_list" and ft


def test_threshold_follows_config_values(rig, monkeypatch):
    rig.acq["inline"]["deadline_s"] = 10  # need = 10 + 60 = 70
    rig.run(remaining=70.0)
    assert rig.calls["inline"] == 1
    rig.run(remaining=69.9)
    assert rig.calls["inline"] == 1  # unchanged: second run skipped
    monkeypatch.setattr(ask_service, "_chain_timeout_s", lambda: 30.0)  # need = 10 + 90 = 100
    rig.run(remaining=99.9)
    assert rig.calls["inline"] == 1


def test_unreadable_chain_config_skips_with_budget_unknown(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_chain_timeout_s", lambda: None)
    r = rig.run(elapsed=0.0)
    assert rig.calls["inline"] == 0  # I1 revision: fail closed (was fail open)
    assert r.contract["inline_acquisition"] == {"status": "budget_unknown"}


# ---- where the 180 s lives -------------------------------------------------------------------------------------
def test_180s_is_a_client_side_constant_with_no_server_side_enforcement_pin():
    """FINDING pin: ASK_TIMEOUT_S is the UI's requests timeout (public_mode.post_ask). main.py (/ask) and the vera
    package other than public_mode/trace_eval do not reference it, and /ask has no timeout middleware. If this test
    fails, a server-side enforcement was added: replace it with a real termination/demotion test."""
    assert public_mode.ASK_TIMEOUT_S == 180
    main_src = (ROOT / "main.py").read_text()
    assert "ASK_TIMEOUT_S" not in main_src
    assert not re.search(r"wait_for|asyncio\.timeout|add_middleware|BaseHTTPMiddleware|timeout-keep-alive", main_src)
    users = {p.name for p in (ROOT / "vera").glob("*.py") if "ASK_TIMEOUT_S" in p.read_text()}
    assert users == {"public_mode.py", "ask_service.py", "progress_ui.py"}  # PROG-1027: the UI poller imports the constant (client side, plan s3c)


def test_started_inline_run_is_bounded_by_its_own_45s_deadline_scope(monkeypatch):
    seen = {}
    import contextlib

    @contextlib.contextmanager
    def scope(deadline):
        seen["deadline"] = deadline
        yield
    monkeypatch.setattr(rate_limit, "deadline_scope", scope)
    monkeypatch.setattr(inline_acquire, "_run_body", lambda *a, **k: "BODY")
    monkeypatch.setattr(inline_acquire.time, "monotonic", lambda: 500.0)
    out = inline_acquire._run("q", {"inline": {"deadline_s": 45}}, {}, lambda: 0.0, None)
    assert out == "BODY" and seen["deadline"] == 545.0
