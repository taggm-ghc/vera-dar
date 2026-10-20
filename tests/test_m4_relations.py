from tests.m45_support import FakeLLM, corpus  # noqa: F401
from vera.m4.context_builder import build_reasoning_context
from vera.m4.evidence_relations import build_relations
from vera.m4.gate_c import evaluate_adequacy


def llm_with(rels):
    return FakeLLM(pairwise={"relations": rels})


def test_relation_types_and_contradictions(corpus):
    g = build_relations(corpus, llm=llm_with([
        {"from": "E2", "to": "E1", "type": "refutes", "confidence": 0.9, "rationale": "opposite direction"},
        {"from": "E7", "to": "E1", "type": "qualifies", "confidence": 0.7, "rationale": "task type"},
        {"from": "E3", "to": "E1", "type": "supports", "confidence": 0.6, "rationale": "faster"},
    ]))
    types = {e["relation_type"] for e in g["edges"]}
    assert types == {"supports", "refutes", "qualifies"}
    assert len(g["contradictions"]) == 1 and g["contradictions"][0]["span1_id"] == "s2"
    assert len(g["nodes"]) == 8


def test_invalid_relations_dropped_and_confidence_clamped(corpus):
    g = build_relations(corpus, llm=llm_with([
        {"from": "E1", "to": "E1", "type": "supports", "confidence": 1},       # self edge
        {"from": "E1", "to": "E99", "type": "supports", "confidence": 1},      # unknown id
        {"from": "E1", "to": "E2", "type": "causes", "confidence": 1},         # bad type
        {"from": "E1", "to": "E2", "type": "supports", "confidence": 7},       # clamp
        {"from": "E1", "to": "E2", "type": "supports", "confidence": 0.1},     # duplicate
    ]))
    assert len(g["edges"]) == 1 and g["edges"][0]["confidence"] == 1.0


def test_dependencies_detected_deterministically(corpus):
    corpus["spans"][1]["source_id"] = corpus["spans"][0]["source_id"]
    corpus["spans"][3]["derived_from"] = ["src_s3"]
    g = build_relations(corpus, llm=llm_with([]))
    reasons = {d["reason"] for d in g["dependencies"]}
    assert reasons == {"same_source", "derived_from"}


def test_context_has_labels_pointers_and_is_compact(corpus):
    g = build_relations(corpus, llm=llm_with([
        {"from": "E2", "to": "E1", "type": "refutes", "confidence": 0.9, "rationale": "r"}]))
    ctx = build_reasoning_context(corpus, g, evaluate_adequacy(corpus, "q"))
    assert "[E1] span_id=s1" in ctx and "[E8] span_id=s8" in ctx
    assert "E2 REFUTES E1" in ctx and "contradicted by E2" in ctx
    assert build_reasoning_context(corpus, g) == build_reasoning_context(corpus, g)  # deterministic
    assert len(ctx) < 5000


def test_context_caps_units_and_notes_omission(corpus):
    from tests.m45_support import span
    corpus["spans"] += [span(f"x{i}", "filler", ["sq_field"]) for i in range(40)]
    ctx = build_reasoning_context(corpus, {"edges": []})
    assert "omitted" in ctx and "[E31]" not in ctx
