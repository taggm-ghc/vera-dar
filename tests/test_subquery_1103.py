"""NONASK-1103 S1/S7 unit tests (offline). Unit checks, not a calibration (see config/subquery.json note)."""
import json
import pytest
from vera import subquery as sq, temporality as tp

REG = sq.load_entity_registry()
Q = "How do AWS, Google, Nvidia and Azure compare for cloud ai engineering?"


def plan(q):
    return sq.decompose(q, registry=REG)


def test_example_four_parts_four_hits():
    p = plan(Q)
    assert p.original == Q and p.rule == "entity_list" and len(p.parts) == 4
    assert p.entity_hits() == ["AWS", "Google Cloud", "Nvidia", "Azure"]
    assert all("compare for cloud ai engineering" in x.text for x in p.parts)
    assert p.parts[2].text.startswith("How do Nvidia")


@pytest.mark.parametrize("q,n", [
    ("AWS, Google and Azure pricing models", 3),
    ("What does Nvidia or AWS offer for training?", 2),
    ("Compare GCP / Azure / AWS latency", 3),
    ("Is Azure & AWS cheaper for batch jobs", 2),
    ("Amazon Web Services, Microsoft Azure, Google Cloud and Nvidia GPU offerings", 4),
])
def test_varied_phrasings_split(q, n):
    p = plan(q)
    assert len(p.parts) == n and p.original == q and sq.drift_ok(q, [x.text for x in p.parts])


def test_overflow_merged_not_dropped():
    reg = REG + [{"entity": "Oracle", "aliases": ["Oracle"], "state": "active", "volatile": False}]
    q = "AWS, Google, Nvidia, Azure and Oracle hosting"
    p = sq.decompose(q, registry=reg)
    assert len(p.parts) == 4 and set(p.entity_hits()) == {"AWS", "Google Cloud", "Nvidia", "Azure", "Oracle"}
    assert "Azure and Oracle" in p.parts[-1].text


@pytest.mark.parametrize("q", [
    "pros and cons of cloud", "salt, fat and acid", "what are ai and ml", "research and development costs",
    "what are AI and ML", "cloud, edge and fog computing", "Which is better, Python or Rust?",
    "mechanism of photosynthesis", "",
])
def test_non_entity_controls_not_split(q):
    p = plan(q)
    assert len(p.parts) == 1 and p.rule == "single" and p.original == q


@pytest.mark.parametrize("q", ["google it and see", "Amazon rainforest and Azure sky engineering", "azure sky, google it"])
def test_lexically_similar_controls_no_entity_hits(q):
    p = plan(q)
    assert p.entity_hits() == [] and len(p.parts) == 1


def test_comparison_rules():
    p = plan("Compare Kubernetes with Nomad for scheduling")
    assert p.rule == "comparison" and [x.text for x in p.parts] == ["Kubernetes scheduling", "Nomad scheduling"]
    p = plan("Redis vs Memcached")
    assert p.rule == "comparison" and len(p.parts) == 2
    assert plan("the relationship between diet and cancer").rule == "single"


def test_drift_guard():
    assert not sq.drift_ok("a b", ["a c"]) and not sq.drift_ok("a b", [""]) and sq.drift_ok("a b", ["b"])


def test_registry_only_active_and_alias_collision(tmp_path):
    rows = {"entities": [
        {"entity": "A", "aliases": ["Foo"], "state": "active"}, {"entity": "B", "aliases": ["foo"], "state": "active"},
        {"entity": "C", "aliases": ["Cee"], "state": "draft"}, {"entity": "D", "aliases": ["Dee"], "state": "suspended"},
        {"entity": "E", "aliases": ["Eee"], "state": "active"}]}
    f = tmp_path / "r.json"; f.write_text(json.dumps(rows))
    reg = sq.load_entity_registry(f)
    names = {r["entity"] for r in reg}
    assert names == {"A", "B", "E"}  # C draft, D suspended excluded; colliding alias "foo" dropped from A and B
    assert not sq.match_entities("Foo and foo", reg) and sq.match_entities("Eee", reg)
    shipped = {r["entity"] for r in REG}
    assert "Kubernetes" not in shipped and "Amazon" not in shipped  # draft / suspended seeds


def test_hosts_exact_no_wildcards():
    for r in REG:
        assert r["state"] == "active" and r["domains"] and all("*" not in d and "/" not in d for d in r["domains"])


def test_deterministic_and_flag_off():
    assert plan(Q) == plan(Q)
    cfg = sq.load_config()
    assert cfg["model_fallback"]["enabled"] is False
    called = []
    p = sq.decompose("cloud, edge and fog", registry=REG, decomposer=lambda q: called.append(1) or ["cloud", "edge"])
    assert not called and p.rule == "single"
    cfg2 = dict(cfg, model_fallback={"enabled": True})
    p = sq.decompose("cloud, edge and fog", registry=REG, cfg=cfg2, decomposer=lambda q: ["cloud", "edge"])
    assert p.rule == "model_fallback"
    p = sq.decompose("cloud, edge and fog", registry=REG, cfg=cfg2, decomposer=lambda q: ["invented words"])
    assert p.rule == "single"


def test_part_temporal_flag():
    p = plan(Q)
    assert all(x.temporal for x in p.parts)  # volatile vendors
    p = plan("History of Kubernetes, Nomad and Docker adoption")
    assert not any(x.temporal for x in p.parts)


@pytest.mark.parametrize("q", [
    "What is the current state of fusion research?", "latest advances in protein folding",
    "status quo best practices for password storage", "Who leads the market as of 2026?",
    "What is the state of the art in speech recognition", "How is Brazil doing today", "recent changes to GDPR enforcement",
])
def test_temporality_current(q):
    assert tp.detect(q).current


@pytest.mark.parametrize("q", [
    "mechanism of photosynthesis", "effect of sleep on memory consolidation", "history of the Roman census",
    "what is alternating current", "current density in a wire", "state of matter at low temperature",
    "In 2019 the study was published", "proof of the Pythagorean theorem",
])
def test_temporality_timeless(q):
    assert not tp.detect(q).current


def test_temporality_volatile_entity_and_year():
    assert tp.detect("pricing of X", volatile_entity=True).cues == ("volatile_entity",)
    assert "year" in tp.detect("latest results in 2026").cues
