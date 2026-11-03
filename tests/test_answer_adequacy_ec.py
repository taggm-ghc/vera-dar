"""EC-01..EC-20 table test for vera.answer_adequacy (ANS-LAYERS-1101 S2 J1).

Fixtures are SYNTHETIC unit data, not calibration. Lexicon comes from the real config.
"""
import json
import pathlib

import pytest

from vera import answer_adequacy as aa

_BLK = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config" / "fulldoc_answer.json").read_text())["answer_adequacy"]
# The seed lexicon is loaded from the real shipped config, not copied here.
CFG = {k: v for k, v in _BLK.items() if not k.startswith("_")}
CFG["acq_round_target_new_docs"] = 6  # resolved by the caller from auto_acquire.json
CFG["definitional_max_subject_words"] = 6

DEFERRED = "needs deferred stage2-plan section 2b (scope field on the claim-check call); narrow-scope labelling is not done in this module"


def D(i, title="Power systems RAG subagent pipeline", kind="scholarly", topic="unassigned"):
    return {"document_id": f"d{i}", "title": title, "kind": kind, "micro_topic": topic}


GEN = "What are AI subagents?"
PS = [D(1)]
OK = {"full_texts_read": 1}

# id, question, summary, admitted, route, expected
CASES = [
    pytest.param("EC-01", GEN, OK, PS, None, "partial",
                 marks=pytest.mark.xfail(reason=DEFERRED, strict=True)),
    pytest.param("EC-02", GEN, {"full_texts_read": 1, "candidates_total": 5}, PS, None, "partial",
                 marks=pytest.mark.xfail(reason=DEFERRED, strict=True)),
    ("EC-03", "What retrieval top-k does Section 3 of the paper specify?", OK, PS, None, "adequate"),
    ("EC-04", "What sampling temperature does the paper's evaluation use?", OK, PS, None, "adequate"),
    ("EC-05", "Which dataset does the pipeline's evaluation use?", OK, PS, None, "adequate"),
    ("EC-06", 'Why did the 2019 NIST report retire the term "AI subagent"?',
     {"premise_unverified": True, "full_texts_read": 2}, [D(1), D(2)], None, "correct_no_answer"),
    ("EC-07", "Which power-grid fault type does the RAG subagent pipeline fix?",
     {"premise_unverified": True, "full_texts_read": 1}, PS, None, "correct_no_answer"),
    pytest.param("EC-08",
                 "How does latency of AI subagents compare with single-agent LLM systems?",
                 OK, PS, None, "partial",
                 marks=pytest.mark.xfail(reason="no parts signal: rule cannot see the missing side; needs parts or cc_scope", strict=True)),
    ("EC-08p", "How does latency of AI subagents compare with single-agent LLM systems?",
     {"full_texts_read": 1, "parts": [{"answered": True}, {"answered": False}]}, PS, None, "partial"),
    ("EC-09", "How do AI subagents compare with single-agent systems on accuracy?",
     {"full_texts_read": 2, "parts": [{"answered": True}, {"answered": True}]},
     [D(1, "Subagent accuracy", "scholarly"), D(2, "Single agent accuracy", "scholarly")], None, "adequate"),
    pytest.param("EC-10", "How do I add a subagent to my coding assistant's settings?",
                 OK, PS, None, "inadequate",
                 marks=pytest.mark.xfail(reason="needs router route=general; router is OFF today (route None)", strict=True)),
    ("EC-10r", "How do I add a subagent to my coding assistant's settings?", OK, PS, "general", "inadequate"),
    ("EC-11", "How long should a sourdough starter be fed before baking?",
     {"full_texts_read": 2}, [D(1, "Sourdough feeding", "web"), D(2, "Starter schedule", "web")], "general", "adequate"),
    ("EC-12", "What does the paper's appendix conclude about subagent cost?",
     {"full_texts_read": 0, "candidates_total": 1}, [], None, "inadequate"),
    ("EC-13", GEN, {"full_texts_read": 0, "candidates_total": 4}, [], None, "inadequate"),
    ("EC-14", GEN, {"full_texts_read": 1, "removed_count": 1}, PS, None, "partial"),
    pytest.param("EC-15", GEN, OK, PS, None, "partial",
                 marks=pytest.mark.xfail(reason="wrong-section citation is a grounding concern; no section-location signal", strict=False)),
    pytest.param("EC-16", "What are all the types of AI agents?", OK, PS, None, "partial",
                 marks=pytest.mark.xfail(reason=DEFERRED, strict=True)),
    ("EC-17", GEN, {"full_texts_read": 1, "uncited_dropped": 2}, PS, None, "partial"),
    ("EC-18", "What is the retrieval step order in the RAG subagent pipeline?", OK, PS, None, "adequate"),
    ("EC-19", "Which two agent frameworks does the paper compare?", OK, PS, None, "adequate"),
    pytest.param("EC-20", GEN, {"full_texts_read": 3}, [D(1), D(2, "Grid subagents"), D(3, "Power RAG")], None, "partial",
                 marks=pytest.mark.xfail(reason=DEFERRED, strict=True)),
]


def D2(i, t, kind="web"):
    return {"document_id": f"d{i}", "title": t, "kind": kind}


EXTRA = [
    ("CX1", "What is photosynthesis?", {"full_texts_read": 5},
     [D2(i, t) for i, t in enumerate(["Photosynthesis", "Photosynthesis - Britannica", "Photosynthesis overview", "Photosynthesis in plants", "Light reactions"])]),
    ("CX2", "What is a transformer model?", {"full_texts_read": 8},
     [D2(i, t, "scholarly") for i, t in enumerate(["Attention is all you need", "BERT", "Few-shot learners", "Vision transformer", "Attention survey", "Conformers", "Long-range arena", "Formal algorithms"])]),
    ("CX5", "What is GPT-4's context window?", {"full_texts_read": 1}, [D2(1, "GPT-4 technical report", "scholarly")]),
    ("CX6", "What is the RAG pipeline's step order?", {"full_texts_read": 1}, [D2(1, "RAG pipeline", "scholarly")]),
    ("DEF-1", "Define photosynthesis", {"full_texts_read": 3}, [D2(i, "Photosynthesis") for i in range(3)]),
    ("BROAD-1", "What are the types of renewable energy?", {"full_texts_read": 6}, [D2(i, f"Energy source {i}") for i in range(6)]),
]


@pytest.mark.parametrize("cid,q,s,a", EXTRA, ids=lambda x: x if isinstance(x, str) and x[:2] in ("CX", "DE", "BR") else None)
def test_count_alone_never_partial(cid, q, s, a):
    r = aa.assess(q, s, a, CFG)
    assert r["adequacy"] in ("adequate", "unknown"), (cid, r)
    assert r["adequacy"] == "adequate", (cid, r)
    assert r["scope"] == "unknown"


def test_low_count_gives_user_language_note_only():
    r = aa.assess("What are AI subagents?", {"full_texts_read": 1}, [D(1)], CFG)
    assert r["adequacy"] == "adequate" and r["limitations"] == ["Only 1 full text was read for this question, so the answer may not cover the whole topic."]
    assert "context" not in r["limitations"][0] and "low_coverage" not in r["limitations"][0]


def test_shipped_block_works_as_is():
    blk = {k: v for k, v in _BLK.items() if not k.startswith("_")}
    r = aa.assess("What are AI subagents?", {"full_texts_read": 1}, [D(1)], blk)
    assert r["adequacy"] == "adequate"


@pytest.mark.parametrize("cfg", [None, {}, {"min_docs_general": "config/x.json#y"}])
def test_missing_or_partial_config_unknown_no_exception(cfg):
    r = aa.assess("What are AI subagents?", {"full_texts_read": 1}, [D(1)], cfg)
    assert r["adequacy"] == "unknown"


def test_cx9_docs_without_ids_not_collapsed():
    docs = [{"title": "AI subagents", "kind": "web"} for _ in range(7)]
    r = aa.assess("What are AI subagents?", {"full_texts_read": 7}, docs, CFG)
    assert r["adequacy"] == "adequate" and r["limitations"] == []


def test_cx10_no_floor_in_cfg_no_exception():
    c = {k: v for k, v in CFG.items() if k != "acq_round_target_new_docs"}
    assert aa.assess("What are AI subagents?", {"full_texts_read": 1}, [D(1)], c)["adequacy"] == "adequate"


@pytest.mark.parametrize("cid,question,summary,admitted,route,expected", CASES)
def test_ec_table(cid, question, summary, admitted, route, expected):
    r = aa.assess(question, summary, admitted, CFG, route)
    assert r["adequacy"] == expected, (cid, r)


NEGATIVE = ["EC-03", "EC-04", "EC-05", "EC-11", "EC-18", "EC-19"]


@pytest.mark.parametrize("c", [c for c in CASES if not hasattr(c, "marks") and c[0] in NEGATIVE], ids=lambda c: c[0])
def test_negative_controls_not_partial(c):
    _, q, s, a, route, _ = c
    r = aa.assess(q, s, a, CFG, route)
    assert r["adequacy"] == "adequate" and r["limitations"] == []


def test_limitations_use_no_internal_names():
    # reasons and form are internal-only; limitations must never carry codes
    r = aa.assess("What are AI subagents?", {"full_texts_read": 1, "candidates_total": 4}, [D(1)], CFG)
    assert not any(c in t for t in r["limitations"] for c in r["reasons"])
    banned = ("claim", "gate", "router", "model", "micro", "cc_scope", "S4", "stage")
    for t in aa._LIMITATIONS.values():
        assert not any(b in t.lower() for b in banned), t
