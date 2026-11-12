"""DRAFTVERIFY-IMPL-1111: draft hypothesis, claim stance, gap queries and their hooks. Offline: fake LLM callables only
(no network, no key, no DB, no env files). Every stage ships off; the parity tests prove /ask is unchanged with flags off."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.test_acq6_rounds_1103 import H, Q as ACQ_Q, cand, cfg as acq_cfg
from vera import claim_stance as cs, draft_hypothesis as dh, draft_verify as dv, gap_queries as gq
from vera import inline_acquire as ia, provider_exhaustion as px
from vera import auto_acquire as aa

ROOT = Path(__file__).resolve().parents[1]
ON = {"stages": {"draft_query": "on", "stance": "on", "gap": "on"}}
SHADOW = {"stages": {"draft_query": "shadow", "stance": "shadow", "gap": "shadow"}}


@pytest.fixture(autouse=True)
def _reset():
    dv.set_config_for_tests(None)
    dv.reset()
    px.reset_for_tests()
    yield
    dv.set_config_for_tests(None)
    dv.reset()
    px.reset_for_tests()


def cfg_of(raw=None):
    dv.set_config_for_tests(raw if raw is not None else ON)
    return dv.config()


# ------------------------------------------------------------------------------------------------ config
def test_shipped_config_is_all_off_and_valid():
    raw = json.loads((ROOT / "config/draft_verify.json").read_text())
    assert raw["stages"] == {"draft_query": "off", "stance": "off", "gap": "off"}
    assert "PENDING CAL" in raw["_threshold_note"]
    c = dv.config()
    assert not c.get("refused") and not dv.any_on() and dv.second_query() is None and dv.gap_queries() == []


@pytest.mark.parametrize("bad", [{"stages": {"gap": "yes"}}, {"stages": {"nope": "on"}}, {"k_chunks": True},
                                 {"k_chunks": 0}, {"max_gap_rounds": 99}, {"num_tolerance": float("nan")},
                                 {"draft_model": ""}, {"disconfirm_terms": "x"}, {"max_claims": "8"}, "str", None])
def test_bad_value_refuses_whole_file_not_clamped(bad):
    raw = {"stages": {"draft_query": "on"}}
    if isinstance(bad, dict):
        raw = {**raw, **bad}
        if "stages" in bad:
            raw["stages"] = {**bad["stages"]}
    else:
        raw = bad
    dv.set_config_for_tests(raw)
    c = dv.config()
    assert all(m == "off" for m in c["stages"].values()) and not dv.any_on()


def test_missing_or_unreadable_file_means_off(monkeypatch, tmp_path):
    monkeypatch.setattr(dv, "CONFIG_PATH", tmp_path / "nope.json")
    dv.set_config_for_tests(None)
    assert not dv.any_on() and dv.config().get("refused")


# ------------------------------------------------------------------------------------------------ leak guards
DRAFT = "The Zelmar protocol reduces latency by 37 percent in Hamburg trials of the routing system."
QUESTION = "How does the routing system affect latency?"
CHUNK = "The routing system affects latency in some trials."


def test_leak_check_catches_ngram_number_and_entity_and_passes_clean():
    leaked = "Reports say the Zelmar protocol reduces latency by 37 percent."
    r = dv.leaks([leaked], [DRAFT], [QUESTION, CHUNK])
    assert r["ngram"] and "37" in r["numbers"] and "zelmar" in r["entities"]
    clean = "The routing system affects latency in some trials."
    assert dv.leak_free([clean], [DRAFT], [QUESTION, CHUNK])
    # a number the question or a chunk also holds is not draft-only
    assert dv.leaks(["latency 37 percent"], [DRAFT], [QUESTION, "chunk says 37 percent"])["numbers"] == []
    # a single-token entity leak is caught even with no 5-gram
    assert dv.leaks(["Zelmar"], [DRAFT], [QUESTION])["entities"] == ["zelmar"]


def test_silence_aspect_uses_only_question_or_chunk_terms():
    a = dv.aspect(DRAFT, QUESTION, [CHUNK])
    assert a and "zelmar" not in a and "hamburg" not in a and "37" not in a
    assert dv.aspect("Zelmar Hamburg 37", QUESTION, [CHUNK]) is None  # nothing shared: counted, not shown
    assert dv.silence_text(a).startswith("the sources found do not address ")


def test_terms_query_is_terms_only_and_pii_filtered():
    q = dv.terms_query("Contact bob@example.org or https://x.example/a about 1234567890 routing latency in multi-agent systems.")
    assert q and "@" not in q and "://" not in q and "1234567890" not in q
    assert q != "Contact bob@example.org or https://x.example/a about routing latency in multi-agent systems."
    assert dv.terms_query("") == ""


# ------------------------------------------------------------------------------------------------ stance parse and aggregate
def _reply(n_claims, n_chunks, labels=None, reason="entity differs"):
    labels = labels or {}
    rows = [{"claim": i, "chunk": j, "label": labels.get((i, j), "silent"), "reason": reason}
            for i in range(n_claims) for j in range(n_chunks)]
    return json.dumps({"pairs": rows})


def test_parse_requires_every_pair_exactly_once_and_valid_labels():
    assert cs.parse(_reply(2, 2), 2, 2) is not None
    assert cs.parse(_reply(2, 2), 2, 3) is None  # a pair missing
    good = json.loads(_reply(1, 1))
    for mut in ({"label": "supported"}, {"claim": True}, {"chunk": 5}, {"claim": 1.0}):
        bad = {"pairs": [{**good["pairs"][0], **mut}]}
        assert cs.parse(json.dumps(bad), 1, 1) is None
    dup = {"pairs": good["pairs"] * 2}
    assert cs.parse(json.dumps(dup), 1, 1) is None
    assert cs.parse("not json", 1, 1) is None and cs.parse('{"pairs":3}', 1, 1) is None
    assert cs.parse('{"pairs":[],"pairs":[]}', 0, 0) is None  # duplicate key rejected by strict_json


DOC = {0: "A", 1: "B", 2: "C"}


def agg(rows):
    return cs.aggregate([(j, lab, "r") for j, lab in rows], DOC)


@pytest.mark.parametrize("rows,status", [
    ([(0, "confirms"), (1, "silent")], "confirmed"), ([(0, "supports"), (1, "silent")], "supported"),
    ([(0, "confirms"), (1, "supports")], "confirmed"), ([(0, "silent"), (1, "silent"), (2, "silent")], "silent"),
    ([(0, "rebuts"), (1, "silent")], "rebutted"), ([(0, "rejects"), (1, "rebuts")], "rejected"),
    ([(0, "confirms"), (1, "rejects")], "mixed"), ([(0, "supports"), (1, "rebuts")], "mixed"),
])
def test_aggregate_each_status(rows, status):
    assert agg(rows)["status"] == status


def test_same_document_conflict_is_not_a_cross_document_discrepancy():
    d = {0: "A", 1: "A"}
    a = cs.aggregate([(0, "confirms", ""), (1, "rejects", "")], d)
    assert a["status"] == "confirmed" and a["same_doc_conflict"] and a["contra_docs"] == 0


# ------------------------------------------------------------------------------------------------ triggers
CL = ["Water boils at 100 degrees at sea level", "The protocol was invented in 1950", "Cats purr"]


def mk(status, contra=1, top_silent=True, reason="the boiling temperature differs", ci=0):
    return {"status": status, "support_docs": 0, "contra_docs": contra, "same_doc_conflict": False, "top_silent": top_silent,
            "contra_reasons": [(ci, "rejects", reason)] if status in ("rebutted", "rejected", "mixed") else []}


def test_each_trigger_kind_fires_and_confirmed_supported_never_do():
    c = cfg_of()
    aggs = {0: mk("silent"), 1: mk("rebutted", reason="the invention year differs"), 2: mk("mixed", reason="cats claim differs")}
    t = cs.triggers(aggs, CL, c)
    assert [(x["claim"], x["kind"]) for x in t] == [(1, "rebutted"), (2, "mixed"), (0, "silent")]  # discrepancy before silence
    for st in ("confirmed", "supported"):
        assert cs.triggers({0: mk(st)}, CL, c) == []
    assert cs.triggers({0: mk("rejected", reason="boiling point differs")}, CL, c)[0]["kind"] == "rejected"


def test_silent_trigger_needs_silence_in_the_top_k_chunks():
    assert cs.triggers({0: mk("silent", top_silent=False)}, CL, cfg_of()) == []


def test_min_contra_docs_numeric_tolerance_and_named_element_conditions():
    c = cfg_of({**ON, "min_contra_docs": 2})
    assert cs.triggers({0: mk("rejected", contra=1)}, CL, c) == []
    c = cfg_of()
    chunks = [{"text": "Water boils at 99.5 degrees at sea level."}]
    assert cs.triggers({0: mk("rebutted")}, CL, c, chunks) == []  # 0.5% < tolerance 10%
    chunks = [{"text": "Water boils at 60 degrees at sea level."}]
    assert cs.triggers({0: mk("rebutted")}, CL, c, chunks)  # 40% > tolerance
    chunks = [{"text": "Water boils at 10 degrees at altitude."}]
    assert cs.triggers({0: mk("rebutted", reason="")}, CL, c, chunks) == []  # reason names no element
    assert cs.numeric_difference_ok("no numbers here", "x 5", 0.1) and cs.numeric_difference_ok("100", "1000", 0.1)
    assert cs.numeric_difference_ok("-5 degrees", "5 degrees", 0.1)


def test_trigger_cap_max_triggered_claims():
    c = cfg_of({**ON, "max_triggered_claims": 2})
    aggs = {i: mk("silent") for i in range(6)}
    assert len(cs.triggers(aggs, [f"claim {i}" for i in range(6)], c)) == 2


# ------------------------------------------------------------------------------------------------ judge: Groq-only, fail open
GROQ_A = {"provider": "groq", "base_url": "https://g.example/v1", "model": "qwen/qwen3.8-27b", "api_key_env": "K", "family": "qwen"}
GROQ_B = {"provider": "groq", "base_url": "https://g.example/v1", "model": "openai/gpt-oss-120b", "api_key_env": "K", "family": "gpt-oss"}
OPENAI = {"provider": "openai", "base_url": None, "model": "gpt-4.1-nano", "api_key_env": "OK"}
CHUNKS = [{"text": "alpha", "document_id": "A"}, {"text": "beta", "document_id": "B"}]


def test_groq_filter_never_includes_openai_entry():
    assert cs.groq_checkers({"checkers": [GROQ_A, OPENAI, GROQ_B]}) == [GROQ_A, GROQ_B]
    real = json.loads((ROOT / "config/ask-provider-chain.json").read_text())["grounding"]["claim_check"]
    got = cs.groq_checkers(real)
    assert got and all(c["provider"] == "groq" and c["base_url"] for c in got)


def test_judge_never_calls_openai_even_if_passed_and_flags_same_family():
    seen = []

    def call(ck, msgs):
        seen.append(ck["model"])
        return _reply(1, 2, {(0, 0): "confirms"})
    r = cs.run(["c"], CHUNKS, cfg_of(), checkers=[OPENAI, GROQ_B], call=call)
    assert seen == ["openai/gpt-oss-120b"] and r["same_family"] is True
    r = cs.run(["c"], CHUNKS, cfg_of(), checkers=[GROQ_A], call=call)
    assert r["same_family"] is False and r["aggregates"][0]["status"] == "confirmed"


def test_malformed_reply_falls_to_next_checker_then_none():
    replies = iter(["garbage", _reply(1, 2)])
    r = cs.run(["c"], CHUNKS, cfg_of(), checkers=[GROQ_A, GROQ_B], call=lambda ck, m: next(replies))
    assert r["checker"].endswith("gpt-oss-120b")
    assert cs.run(["c"], CHUNKS, cfg_of(), checkers=[GROQ_A], call=lambda ck, m: "garbage") is None


def test_grace_and_exhaustion_skip_the_stage():
    c = cfg_of()
    tok = px.GRACE.set({"x": 1})
    try:
        assert cs.run(["c"], CHUNKS, c, checkers=[GROQ_A], call=lambda *a: _reply(1, 2)) is None
        assert dv.skip_reason(None, 1, c) == "grace"
        assert dh.make("q", c, call=lambda *a: "{}") is None
    finally:
        px.GRACE.reset(tok)

    class E(Exception):
        status_code = 429
    def boom(ck, m):
        raise E()
    assert cs.run(["c"], CHUNKS, c, checkers=[GROQ_A, GROQ_B], call=boom) is None


def test_budget_shortfall_skips_and_unknown_time_does_not():
    c = cfg_of()
    assert dv.skip_reason(70.0, 20, c) == "budget_short"  # 70 - reserve 60 < 20
    assert dv.skip_reason(100.0, 20, c) is None
    assert dv.skip_reason(float("inf"), 20, c) is None and dv.skip_reason(None, 20, c) is None


def test_exhausted_providers_skip(monkeypatch):
    monkeypatch.setattr(px, "exhausted", lambda: True)
    assert dv.skip_reason(None, 1) == "exhausted"


# ------------------------------------------------------------------------------------------------ draft hypothesis
def test_draft_parse_and_failures():
    c = cfg_of()
    ok = json.dumps({"draft": "Short draft.", "claims": ["a claim", "  ", 3, "b claim"]})
    d = dh.make("q", c, call=lambda m, msgs, t: ok)
    assert d.text == "Short draft." and d.claims == ("a claim", "b claim")
    assert dh.make("q", c, call=lambda *a: "nope") is None
    assert dh.make("q", c, call=lambda *a: json.dumps({"draft": "x", "claims": []})) is None
    assert dh.make("q", c, call=lambda *a: (_ for _ in ()).throw(RuntimeError())) is None
    seen = {}
    dh.make("q", c, call=lambda m, msgs, t: seen.update(model=m) or ok)
    assert seen["model"] == "openai/gpt-oss-20b"
    many = json.dumps({"draft": "x", "claims": [f"c{i}" for i in range(20)]})
    assert len(dh.make("q", c, call=lambda *a: many).claims) == c["max_claims"]


def test_draft_prompt_carries_the_question_only():
    msgs = dh.messages("What is X?")
    assert json.loads(msgs[1]["content"]) == {"question": "What is X?"}


# ------------------------------------------------------------------------------------------------ gap queries
def test_gap_queries_are_terms_only_bounded_and_deduped():
    c = cfg_of()
    claims = ["Photons carry momentum proportional to frequency in vacuum conditions",
              "Photons carry momentum proportional to frequency in vacuum conditions",
              "Chlorophyll absorbs red and blue wavelengths in plant leaves"]
    chunks = [{"text": "Radiation pressure shows momentum transfer from light to surfaces."}]
    trig = [{"claim": 0, "kind": "rejected", "chunk": 0}, {"claim": 1, "kind": "silent", "chunk": None},
            {"claim": 2, "kind": "silent", "chunk": None}]
    qs = gq.build(trig, claims, chunks, "q", c)
    assert qs and len(qs) == len(set(q.lower() for q in qs))
    assert len(qs) <= c["max_gap_queries"]
    for q in qs:
        assert claims[0] not in q and "?" not in q and "://" not in q
    assert len([q for q in qs if "photons" in q.lower() or "momentum" in q.lower()]) <= c["max_queries_per_claim"] + 1
    assert any("criticism" in q for q in qs)  # the disconfirming variant of the discrepancy
    capped = gq.build(trig * 10, claims, chunks, "q", cfg_of({**ON, "max_gap_queries": 2}))
    assert len(capped) <= 2


# ------------------------------------------------------------------------------------------------ gap rounds in _extra_rounds
def _met_pool():
    return [[cand(f"m{i}") for i in range(6)]]


def test_gap_off_met_after_round0_runs_no_extra_search(monkeypatch):
    h = H(monkeypatch, _met_pool() + [[cand("g1")]], endless=True)
    o = h.run(acq_cfg(), stream=True)
    assert len(h.searches) == 1 and o.acquisition["status"] == "met"


def test_gap_on_opens_bounded_gap_rounds_after_six_documents_with_one_job(monkeypatch):
    cfg_of({**ON, "max_gap_rounds": 2})
    dv.GAP_QUERIES.set(["gap alpha query", "gap beta query", "gap gamma query"])
    h = H(monkeypatch, _met_pool() + [[cand("g1")], [cand("g2")], [cand("g3")]], endless=True)
    before = len(aa._S.jobs_hour)
    o = h.run(acq_cfg(), stream=True)
    qs = [s["q"] for s in h.searches]
    assert qs[1:] == ["gap alpha query", "gap beta query"]  # gap queries lead, capped at max_gap_rounds, no generic variant
    assert len(aa._S.jobs_hour) - before == 1  # acq_rounds_one_job: no job beyond the question's single one
    assert o.acquisition["rounds"] == 3


def test_gap_shadow_does_not_change_rounds(monkeypatch):
    cfg_of(SHADOW)
    dv.GAP_QUERIES.set(["gap alpha query"])
    h = H(monkeypatch, _met_pool(), endless=True)
    h.run(acq_cfg(), stream=True)
    assert len(h.searches) == 1


def test_gap_rounds_respect_ceiling_and_generic_variants_follow_when_not_met(monkeypatch):
    cfg_of({**ON, "max_gap_rounds": 1})
    dv.GAP_QUERIES.set(["gap alpha query", "gap beta query"])
    h = H(monkeypatch, [[cand("z0", usable=False)]], endless=True)
    h.run(acq_cfg(), stream=True)
    qs = [s["q"] for s in h.searches]
    assert qs[1] == "gap alpha query" and "gap beta query" not in qs  # only one gap round; generic variants follow


# ------------------------------------------------------------------------------------------------ retrieval hook
def test_second_query_changes_retrieval_only_when_on():
    from tests.test_hybrid_retrieval_1109 import Q, _corpus
    c, _ = _corpus()
    dar = c._evidence_dar()
    base = json.dumps(dar.retrieve(Q, 40, max_per_doc=40), sort_keys=True)
    cfg_of(SHADOW)
    dv.QUERY.set("schedules tasks")
    assert json.dumps(dar.retrieve(Q, 40, max_per_doc=40), sort_keys=True) == base  # shadow: unchanged
    cfg_of({"stages": {"draft_query": "on"}})
    r = dar.retrieve(Q, 40, max_per_doc=40)
    assert r["trace"]["branch_counts"].get("draft", 0) > 0 and json.dumps(r, sort_keys=True) != base
    dv.QUERY.set(None)
    assert json.dumps(dar.retrieve(Q, 40, max_per_doc=40), sort_keys=True) == base


# ------------------------------------------------------------------------------------------------ /ask hooks
class EV:
    def __init__(self, chunks):
        self.admitted = tuple(chunks)


def _stance_call(labels):
    return lambda ck, msgs: _reply(len(json.loads(msgs[1]["content"])["claims"]),
                                   len(json.loads(msgs[1]["content"])["passages"]), labels)


def test_prepare_and_after_answer_shadow_record_but_change_nothing():
    cfg_of(SHADOW)
    dd = json.dumps({"draft": "A short draft.", "claims": ["Water boils at 100 degrees", "Cats purr loudly"]})
    dv.prepare("How hot does water boil?", None, call=lambda m, msgs, t: dd)
    assert dv.DRAFT.get() is not None and dv.QUERY.get() is None and dv.second_query() is None  # shadow: no query
    ev = EV([{"text": "Water boils at 60 degrees here.", "document_id": "A"}])
    dv.after_answer("q", ev, None, call=_stance_call({(0, 0): "rejects"}))
    st = dv.STANCE.get()
    assert st["aggregates"][0]["status"] == "rejected" and st["aggregates"][1]["status"] == "silent"
    assert dv.gap_queries() == []  # shadow gap: planned and counted, never applied


def test_on_mode_sets_second_query_and_gap_queries_from_terms_only():
    cfg_of(ON)
    draft = "Water boils at 100 degrees Celsius at sea level pressure."
    dd = json.dumps({"draft": draft, "claims": ["Water boils at 100 degrees Celsius at sea level pressure", "Photons carry momentum in vacuum"]})
    dv.prepare("How hot does water boil?", None, call=lambda m, msgs, t: dd)
    assert dv.second_query() and dv.second_query() != draft
    ev = EV([{"text": "Water boils at 60 degrees Celsius at sea level pressure.", "document_id": "A"}])
    dv.after_answer("q", ev, None, call=_stance_call({(0, 0): "rejects"}))
    gaps = dv.gap_queries()
    assert gaps and all(q != draft and "?" not in q for q in gaps)


def test_same_family_stance_never_feeds_gap_rounds():
    cfg_of(ON)
    dv.DRAFT.set(dv.Draft("d", ("Photons carry momentum in vacuum",)))
    real = [GROQ_B]
    orig = cs.groq_checkers
    try:
        cs.groq_checkers = lambda g: real
        dv.after_answer("q", EV([{"text": "x", "document_id": "A"}]), None, call=_stance_call({}))
    finally:
        cs.groq_checkers = orig
    assert dv.STANCE.get()["same_family"] is True and dv.gap_queries() == []


def test_hooks_skip_when_exhausted_and_never_raise(monkeypatch):
    cfg_of(ON)
    monkeypatch.setattr(px, "exhausted", lambda: True)
    dv.prepare("q", None, call=lambda *a: pytest.fail("draft call made while exhausted"))
    assert dv.DRAFT.get() is None
    dv.DRAFT.set(dv.Draft("d", ("c",)))
    dv.after_answer("q", EV([{"text": "x", "document_id": "A"}]), None, call=lambda *a: pytest.fail("stance call made"))
    dv.prepare("q", None, call=lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))  # swallowed


def test_trace_carries_no_draft_or_claim_text():
    from vera import request_trace as rt
    cfg_of(SHADOW)
    events = []
    orig = rt.event
    rt.event = lambda stage, code, counts=None, detail=None: events.append((stage, code, counts, detail))
    try:
        secret = "Zelmar protocol invented in Hamburg"
        dd = json.dumps({"draft": secret + ".", "claims": [secret]})
        dv.prepare("q about routing", None, call=lambda *a: dd)
        dv.after_answer("q", EV([{"text": "Routing text.", "document_id": "A"}]), None, call=_stance_call({}))
    finally:
        rt.event = orig
    blob = json.dumps(events, default=str)
    assert events and "Zelmar" not in blob and "Hamburg" not in blob


# ------------------------------------------------------------------------------------------------ flags-off parity (ask_service)
def _ask_fields(r):
    return {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}


def test_ask_service_flags_off_parity_and_no_stage_code_runs(monkeypatch, tmp_path):
    """With every stage off (the shipped state) answer_in_scope never enters a draft_verify stage, and its result equals
    the same call with the hooks' stage modules made unreachable. Also: shadow with a failing draft call equals off."""
    from tests import fulldoc_bundle_synth as synth
    from tests.test_fulldoc_flag import GROUNDING, PRICING, Q, WEB, _no_store
    from vera import ask_service, claim_check, fulldoc_corpus, scope_router
    from vera.ask_service import AskResult

    _no_store(monkeypatch)
    root = synth.write_bundle(tmp_path / "b")
    fcfg = synth.synth_cfg()
    fcfg["fulldoc"]["enabled"] = True
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: fcfg)
    monkeypatch.setenv("VERA_FULLDOC_BUNDLE_DIR", str(root))
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("x [1]", 1, 0.0))
    monkeypatch.setattr(claim_check, "check_answer", lambda *a, **k: (None, {"outcome": "x"}))

    def run():
        fulldoc_corpus.reset_for_tests()
        return ask_service.answer_in_scope(Q, "m", PRICING, grounding=(lambda: [dict(w) for w in WEB], GROUNDING))

    dv.set_config_for_tests(None)  # shipped file: all off
    assert not dv.any_on()
    for name in ("prepare", "after_answer", "skip_reason", "terms_query"):
        monkeypatch.setattr(dv, name, lambda *a, **k: pytest.fail("a draft_verify stage ran with every flag off"))
    monkeypatch.setattr(dh, "make", lambda *a, **k: pytest.fail("draft call with flags off"))
    monkeypatch.setattr(cs, "judge", lambda *a, **k: pytest.fail("stance call with flags off"))
    off = run()
    assert off.evidence is not None  # the answered/fulldoc path really ran
    monkeypatch.undo()  # restore everything, then compare with shadow + a draft call that fails (fail open)
    _no_store(monkeypatch)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: fcfg)
    monkeypatch.setenv("VERA_FULLDOC_BUNDLE_DIR", str(root))
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("x [1]", 1, 0.0))
    monkeypatch.setattr(claim_check, "check_answer", lambda *a, **k: (None, {"outcome": "x"}))
    dv.set_config_for_tests(SHADOW)
    monkeypatch.setattr(dh, "_real_call", lambda cfg: (lambda *a: (_ for _ in ()).throw(RuntimeError("down"))))
    shadow = run()
    dv.set_config_for_tests(None)
    off2 = run()
    assert _ask_fields(shadow) == _ask_fields(off2) == _ask_fields(off)
