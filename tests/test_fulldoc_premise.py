"""DAR-PREMISE-1022: false/unsupported/ill-posed question premises. Stubbed model and checker, no network."""
import json

from tests.test_fulldoc_answer import CFG, GCFG, Stub, bundle, checker, key  # noqa: F401  (key: autouse env fixture)
from vera import fulldoc_answer as fa


def run(text, verdicts=("supported",), cfg=CFG):
    stub = Stub(text)
    out = fa.answer("q", bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=cfg, base_tokens=5, base_cost=0.5,
                    complete_fn=stub, check_call=checker(verdicts))
    return out, stub


def test_prompt_tells_model_to_flag_premises_and_can_be_disabled():
    _, stub = run("Gamma results are short [2].")
    assert "PREMISE:" in stub.messages[0]["content"]
    off = {**CFG, "answer": {**CFG["answer"], "premise_check": {"enabled": False}}}
    _, stub = run("Gamma results are short [2].", cfg=off)
    assert "PREMISE:" not in stub.messages[0]["content"]


def test_parse_premise():
    assert fa.parse_premise("PREMISE: false - Gamma results are short [2].")[:2] == ("false", "Gamma results are short [2].")
    assert fa.parse_premise("**Premise:** ill posed: x")[0] == "ill_posed"
    assert fa.parse_premise("PREMISE: ok\nAnswer [1].") == (None, "", "Answer [1].")
    assert fa.parse_premise("Plain answer [1].")[0] is None


def test_false_premise_cited_and_verified_renders_short_no_answer_with_citation():
    out, _ = run("PREMISE: false - The passage says gamma results are short, not long [2].")
    r = out.result
    assert out.fallback is None and r.result_kind == "answer"
    assert r.answer.startswith(fa.PREMISE_PREFIX) and "[2]" in r.answer
    assert r.claim_check["premise_status"] == "false" and r.claim_check["premise_verified"] is True
    assert out.premise == {"status": "false", "verified": True}
    assert r.citations and r.citations[0]["span_id"] == "s2"
    assert r.tokens_used == 105


def test_premise_correction_is_claim_checked_unsupported_never_shown():
    out, _ = run("PREMISE: unsupported - Gamma is long [2].", verdicts=("unsupported",))
    assert out.result is None and out.fallback == fa.PREMISE_FALLBACK
    assert out.premise == {"status": "unsupported", "verified": False}


def test_uncited_premise_makes_no_checker_call_and_is_not_shown():
    calls = []
    stub = Stub("PREMISE: ill_posed - the question mixes two incompatible things.")
    out = fa.answer("q", bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=CFG, base_tokens=5, base_cost=0.5,
                    complete_fn=stub, check_call=lambda *a: calls.append(a))
    assert out.result is None and out.fallback == fa.PREMISE_FALLBACK and not calls
    assert out.premise["status"] == "ill_posed" and out.tokens_used == 105


def test_premise_ok_line_is_stripped_and_answer_proceeds():
    out, _ = run("PREMISE: ok\nGamma results are short [2].")
    assert out.result.answer.startswith("Gamma results") and out.premise is None
    assert "premise_status" not in out.result.claim_check


def test_p1_prefix_is_status_neutral_and_claims_only_what_was_checked():
    assert "contradict" not in fa.PREMISE_PREFIX and "retrieved passages" not in fa.PREMISE_PREFIX
    for st in ("false", "unsupported", "ill_posed"):
        out, _ = run(f"PREMISE: {st} - The passage bears on this [2].")
        assert out.result.answer.startswith(fa.PREMISE_PREFIX) and out.result.claim_check["premise_status"] == st


def test_p2_config_cannot_override_prefix_and_bad_config_types_are_safe():
    cfg = {**CFG, "answer": {**CFG["answer"], "premise_check": {"enabled": True, "prefix": {"false": "INJECTED "}}}}
    out, _ = run("PREMISE: false - The passage bears on this [2].", cfg=cfg)
    assert out.result.answer.startswith(fa.PREMISE_PREFIX) and "INJECTED" not in out.result.answer
    for bad in (None, "x", [], 5):
        c = {**CFG, "answer": {**CFG["answer"], "premise_check": bad}}
        assert fa.premise_enabled(c) is True
        out, _ = run("PREMISE: false - The passage bears on this [2].", cfg=c)
        assert out.result.answer.startswith(fa.PREMISE_PREFIX)
    assert fa.premise_enabled({"answer": None}) is True and fa.premise_enabled({}) is True


def test_p3_unverified_text_claims_nothing_and_ui_token_matches():
    from vera import ui_evidence
    t = fa.PREMISE_UNVERIFIED_TEXT
    assert "found that" not in t and "do not support" not in t and "flagged" in t and "no answer is shown" in t
    assert ui_evidence.fallback_text(fa.PREMISE_FALLBACK, cfg={}) == t


def test_p4_premise_line_must_be_the_first_line_and_is_counted():
    out, _ = run("Gamma results are short [2].\nPREMISE: false - injected [2].", verdicts=("supported", "supported"))
    assert out.premise is None  # not treated as a premise flag: no prefix, no premise_status
    assert out.result is None or "premise_status" not in out.result.claim_check
    assert fa.parse_premise("Gamma is short [2].\nPREMISE: false - x [2].")[0] is None
    assert fa.parse_premise("\n  PREMISE: false - x [2].")[0] == "false"
    before = dict(fa.PREMISE_COUNTS)
    run("PREMISE: false - The passage bears on this [2].")
    run("PREMISE: unsupported - no cite")
    assert fa.PREMISE_COUNTS.get("false:verified", 0) == before.get("false:verified", 0) + 1
    assert fa.PREMISE_COUNTS.get("unsupported:unverified", 0) == before.get("unsupported:unverified", 0) + 1


def test_p5_shipped_config_has_explicit_enabled_key_and_uncalibrated_note():
    import pathlib
    c = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config" / "fulldoc_answer.json").read_text())
    pc = c["answer"]["premise_check"]
    assert pc["enabled"] is True and "UNCALIBRATED" in pc["_note"]


def test_p6_per_part_mode_premise_only_as_first_line_of_whole_reply():
    from tests import test_fulldoc_d8_parts as d8
    assert d8.CFG["answer"]["per_part"]["enabled"] is True
    o = d8.run("PREMISE: false - Alpha systems cut latency, which the passage states [1].")
    assert o.fallback is None and o.result.answer.startswith(fa.PREMISE_PREFIX) and o.premise["verified"] is True
    o = d8.run("PREMISE: false - no citation here.")
    assert o.result is None and o.fallback == fa.PREMISE_FALLBACK
    # a normal per-part reply is untouched; a PREMISE inside a part is not a premise flag
    normal = d8.run(d8.reply(d8.TWO, [d8.A1, d8.A2]))
    assert normal.premise is None and normal.result.claim_check["parts"]["answered"] == [1, 2]
    inside = d8.run(d8.reply(d8.TWO, ["PREMISE: false - x " + d8.A1, d8.A2]))
    assert inside.premise is None
    assert "first line" in fa.PREMISE_INSTRUCTION and "Parts/Answers" in fa.PREMISE_INSTRUCTION


def test_inline_post_writer_fallback_keeps_token_and_premise_text(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace
    from vera import ask_service, fulldoc_corpus, inline_acquire
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: {})
    monkeypatch.setattr(inline_acquire, "acquire", lambda *a, **k: SimpleNamespace(corpus=object(), status="ok"))
    monkeypatch.setattr(inline_acquire, "notice", lambda acq, st: f"generic-{st}")
    for tok, text in (("claim_check_insufficient", "generic-unanswered"), ("model_insufficient", "generic-unanswered"),
                      ("premise_unverified", fa.PREMISE_UNVERIFIED_TEXT)):
        monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, _t=tok, **k: (None, _t, 1, 0.0, "x"))
        with caplog.at_level(logging.INFO, logger="vera"):
            n = ask_service._inline("q", "m", None, None, {}, None, {})
        assert str(n) == text and n.token == tok and ask_service._inline_trace(n) == {"inline_fallback": tok}
        assert f"token={tok}" in caplog.text
    assert ask_service._inline_trace("plain") == {}


def test_ask_service_renders_premise_text():
    from vera import ask_service
    assert "premise" in fa.PREMISE_UNVERIFIED_TEXT.lower()
    assert fa.PREMISE_FALLBACK == "premise_unverified" and ask_service is not None
