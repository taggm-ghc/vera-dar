"""DENYALL-TRACE-1104: deny-all detail gate (string shape + per-stage key allowlist). Offline, no DB, no network."""
import json
import re

import pytest

from vera import request_trace as rt

BASE = json.loads(rt.CONFIG_PATH.read_text(encoding="utf-8"))
AL = BASE["key_allowlist"]


@pytest.fixture(autouse=True)
def _cfg():
    c = json.loads(json.dumps(BASE))
    c["db_sink"]["enabled"] = False
    rt.configure(c)
    yield
    rt.configure(c)


def test_every_stage_has_reason_and_token_keys():
    for st, ent in AL.items():
        if st.startswith("_"):
            continue
        assert ent["reason"].strip() and isinstance(ent["keys"], list)
        assert all(re.fullmatch(r"[a-z][a-z0-9_]{0,39}", k) for k in ent["keys"])


@pytest.mark.parametrize("v", ["hunter2 xyz", "Example.FR", "AKIAIOSFODNN7EXAMPLE", "example.fr", "a" * 65, "", "x/y",
                               "sk-abcdefghijk", "user@host", "http://x", "MixedCase"])
def test_bypass_values_become_placeholder(v):
    d = rt.clean_detail({"provider": v}, stage="search")
    assert d == {"provider": "<str>"}, d


def test_token_values_pass_and_numbers_bools():
    d = rt.clean_detail({"provider": "openalex", "status": "rate_limited", "latency_ms": 812}, stage="search")
    assert d == {"provider": "openalex", "status": "rate_limited", "latency_ms": 812}
    d = rt.clean_detail({"model": "llama-3.3-70b-versatile", "cost_usd": 0.0012}, stage="writer")
    assert d == {"model": "llama-3.3-70b-versatile", "cost_usd": 0.0012}
    assert rt.clean_detail({"accepted": True}, stage="scope") == {"accepted": True}


def test_unknown_keys_dropped_and_counted():
    d = rt.clean_detail({"provider": "x", "question": "what is the password", "secret_note": "z", "extra": 1},
                        stage="search")
    assert d == {"provider": "x", "dropped_keys": 3}


def test_unknown_stage_or_missing_allowlist_drops_all():
    assert rt.clean_detail({"provider": "x"}, stage="nope") == {"dropped_keys": 1}
    assert rt.clean_detail({"provider": "x"}) == {"dropped_keys": 1}
    rt.CFG.pop("key_allowlist")
    assert rt.clean_detail({"provider": "x"}, stage="search") == {"dropped_keys": 1}


def test_question_key_never_allowed():
    for st, ent in AL.items():
        if not st.startswith("_"):
            assert "question" not in ent["keys"] and "text" not in ent["keys"]


def test_nested_walk_same_rules():
    d = rt.clean_detail({"codes": {"no_hits": 2, "Bad Key": 1, "api_key": 3}, "ms": {"search": 40},
                         "status": "Not A Token"}, stage="acquire")
    assert d["codes"] == {"no_hits": 2} and d["ms"] == {"search": 40} and d["status"] == "<str>"
    assert d["dropped_keys"] == 2
    d = rt.clean_detail({"per_claim": [{"id": 0, "verdict": "supported", "cites": [1, 2], "text": "leak"},
                                       {"id": 1, "verdict": "Hunter 2", "cites": []}]}, stage="claim_check")
    assert d["per_claim"][0] == {"id": 0, "verdict": "supported", "cites": [1, 2]}
    assert d["per_claim"][1]["verdict"] == "<str>"
    assert "leak" not in json.dumps(d)


# (stage, code, counts, detail) shapes emitted by the real call sites (read 2026-11-04).
SITES = [
    ("acquire", "done", {"admitted": 1}, {"status": "ok", "codes": {"no_oa": 2}, "ms": {"search": 120, "fetch": 400}}),
    ("acquire", "candidate", None, {"provider": "openalex", "outcome": "stored", "reason": "ok", "diag": "ok"}),
    ("acquire", "outcome", {"stored": 1}, {"status": "ok", "acquisition": {"status": "ok", "stored": 1, "target": 2,
                                           "rounds": 1, "stop": "target"}, "lanes": True}),
    ("acquire", "round", {"round": 1, "latency_ms": 900}, {"status": "ok"}),
    ("acquire", "retry_skipped", None, {"reason": "budget"}),
    ("acquire", "off", None, {"reason": "fulldoc_flag_off"}),
    ("acquire", "answer_fallback", None, {"fallback": "checker_unavailable"}),
    ("search", "provider", {"results": 3}, {"provider": "crossref", "status": "ok", "latency_ms": 55}),
    ("search", "portal", {"results": 0}, {"provider": "core", "status": "quota"}),
    ("search", "seed_fallback", {"results": 2}, {"provider": "seed"}),
    ("premise", "verified", None, {"verdict": "ok"}),
    ("premise", "verdict_flagged", None, {"verdict": "false_premise", "cited": True}),
    ("passages", "sent_to_writer", {"n": 2}, {"span_ids": ["s12", "s40"], "relevance": [0.81, 0.4]}),
    ("passages", "evidence_trace", None, {"cited_span_ids": ["12", "40"]}),
    ("writer", "draft", {"tokens": 900}, {"model": "llama-3.3-70b-versatile", "cost_usd": 0.0}),
    ("writer", "fall", {"records": 0}, {"fallback": "writer_failed"}),
    ("claim_check", "checker_error", None, {"checker": "primary", "exc": "timeouterror"}),
    ("claim_check", "verdicts", {"claims": 1}, {"per_claim": [{"id": 0, "verdict": "supported", "cites": [1]}]}),
    ("claim_check", "checked", {"checked": 1}, {"checker": "primary"}),
    ("scope", "basis", None, {"basis": "classifier", "layer": "layer2_llm", "class": "in_scope", "accepted": True,
                              "refusal": "none"}),
    ("outcome", "decline", None, {"category": "out_of_scope"}),
    ("outcome", "fulldoc_fallback", {"tokens": 5}, {"fallback": "no_admitted"}),
    ("outcome", "reading_list", None, {"branch": "fulldoc_not_configured"}),
    ("outcome", "no_answer", None, {"cause": "insufficient", "fallback": "x", "inline_status": "ok"}),
]


@pytest.mark.parametrize("stage,code,counts,detail", SITES)
def test_real_call_site_events_preserved(stage, code, counts, detail):
    out = rt.clean_detail(detail, stage=stage)
    assert out == detail, (out, detail)
    assert "dropped_keys" not in out


def test_event_end_to_end_and_finish_keys():
    rt.start("secret question text", {"endpoint": "/ask"})
    rt.event("search", "provider", {"results": 1}, {"provider": "Example.FR", "question": "q", "latency_ms": 3})
    rt.finish("answer", "complete")
    v = rt.recent(rt.CURRENT.get().trace_id)
    ev = [e for e in v["events"] if e["stage"] == "search"][0]
    assert ev["detail"] == {"provider": "<str>", "latency_ms": 3, "dropped_keys": 1}
    assert "secret question text" not in json.dumps([e for e in v["events"]])
    assert v["events"][-1]["detail"] == {"message_code": "complete"}
    rt.end_context()


def test_sink_never_raises():
    class Boom:
        def __str__(self):
            raise RuntimeError("x")

    rt.start("q", {})
    rt.event(Boom(), Boom(), counts=Boom(), detail={"provider": Boom(), "status": float("nan"), "codes": Boom()})
    rt.event("search", "provider", detail={"provider": object(), "status": {1: 2}})
    assert rt.clean_detail(Boom(), stage="search") is None
    rt.finish("answer", "complete")
    rt.end_context()


def test_chain_ids_exception_exact_match_and_only_for_id_keys(monkeypatch):
    chain = json.loads(rt.CHAIN_PATH.read_text(encoding="utf-8"))
    mid = chain["providers"][0]["model"]
    assert "/" in mid and mid in rt.CHAIN_IDS
    assert rt.clean_detail({"model": mid}, stage="writer") == {"model": mid}
    assert rt.clean_detail({"model": mid + "x"}, stage="writer") == {"model": "<str>"}
    assert rt.clean_detail({"model": mid.upper()}, stage="writer") == {"model": "<str>"}
    monkeypatch.setitem(rt.CFG, "key_allowlist", {"t": {"keys": ["model", "check_model", "provider", "other"], "reason": "t"}})
    assert rt.clean_detail({"check_model": mid, "provider": "groq", "other": mid}, stage="t") == \
        {"check_model": mid, "provider": "groq", "other": "<str>"}
    assert rt._load_chain_ids(rt.CONFIG_PATH.parent / "missing.json") == frozenset()


@pytest.mark.parametrize("s", ["host=db.example.com password=SENTPW", "PASSWORD: SENTPW x", "api_key=SENTPW", "pwd = SENTPW",
                               "token=SENTPW secret:SENTPW"])
def test_scrub_str_masks_keyword_secrets(s):
    out = rt.scrub_str(s, 200)
    assert "SENTPW" not in out and "<secret>" in out, out
