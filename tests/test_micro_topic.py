"""ACQ S4-0 L2/L2b: read-time micro-topic key. Offline; synthetic text only."""
import copy
import json

import pytest

from vera import acq_config, micro_topic as mt

CFG = copy.deepcopy(acq_config.load_config()["micro_topic"])
EPI = "Transmission and incidence of an infectious disease outbreak: surveillance, cases, deaths and vaccination in a pandemic."
CODE = "Developers using an AI coding assistant for code generation: productivity, bugs and security of generated code."


def test_k4_assignment_best_anchor():
    k = mt.key_for_provenance({"title": "x"}, EPI, CFG)
    assert k["scheme"] == "registry" and k["id"].startswith("epi-") and k["field"] == "epidemiology"
    k2 = mt.key_for_provenance({}, CODE, CFG)
    assert k2["id"].startswith("aicode-") and k2["score"] >= 2.0


def test_below_threshold_unassigned():
    assert mt.key_for_provenance({}, "A short note about medieval pottery glazes and kiln temperatures.", CFG) is None
    assert mt.key_for_provenance({}, "", CFG) is None
    assert mt.key_for_provenance(None, EPI, CFG) is None


def test_stored_key_precedence_and_parent_mapping():
    prov = {"acquisition": {"micro_topic": [{"scheme": "openalex_topic", "id": "T123", "score": 0.9}]}}
    assert mt.key_for_provenance(prov, CODE, CFG) == {"scheme": "openalex_topic", "id": "T123", "score": 0.9}
    prov = {"acquisition": {"micro_topic": [{"scheme": "registry", "id": "eco-biodiversity", "score": 3.0}]}}
    k = mt.key_for_provenance(prov, EPI, CFG)
    assert (k["scheme"], k["id"], k["field"]) == ("registry", "eco-biodiversity", "ecology")
    reg = mt.Registry({"anchor_registry_version": "t", "entries": [
        {"id": "a", "parent_openalex_topic": "T9", "field": "f", "anchor_text": "alpha beta"}]})
    assert mt._entry_key("registry", "a", 1.0, reg)["id"] == "T9"  # parent mapping: no K1/K4 split
    # a provisional K5 stored entry is never read as a key
    prov = {"acquisition": {"micro_topic": [{"scheme": "openalex_topic_provisional", "id": "T5", "score": None}]}}
    assert mt.key_for_provenance(prov, EPI, CFG)["scheme"] == "registry"


def test_deterministic_and_window():
    a = [mt.key_for_provenance({}, EPI, CFG) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    far = "filler " * 1000 + EPI  # beyond text_window_chars: not scored
    assert mt.key_for_provenance({}, far, CFG) is None
    mt.reset_for_tests()
    assert mt.key_for_provenance({}, EPI, CFG) == a[0]


def test_no_write_and_inputs_unchanged():
    prov = {"title": "t", "acquisition": {"marker": "m"}}
    snap = json.dumps(prov, sort_keys=True)
    mt.key_for_provenance(prov, EPI, CFG)
    mt.read_time_key(prov, EPI, CFG)
    assert json.dumps(prov, sort_keys=True) == snap
    import inspect
    src = inspect.getsource(mt)
    for bad in ("open(p, \"w", "requests", "urllib", "psycopg", "logging", ".write("):
        assert bad not in src


def test_no_question_text_in_key():
    q = "SENTINEL-Q-77 how does transmission of an outbreak relate to incidence"
    k = mt.key_for_provenance({}, EPI, CFG)
    assert "SENTINEL" not in json.dumps(k) and q not in json.dumps(k)


def test_k5_disabled():
    assert CFG["k5"]["enabled"] is False
    assert mt.k5_enabled(CFG) is False and mt.k5_enabled({"k5": {"enabled": True}}) is False
    with pytest.raises(RuntimeError):
        mt.k5_key("any question")


def test_counter_and_persisted_count():
    rows = [{"provenance": {}, "content": EPI}, {"provenance": json.dumps({}), "content": EPI},
            {"provenance": {}, "content": CODE}, {"provenance": {}, "content": "pottery glazes"}]
    c = mt.count_by_key(rows, CFG)
    assert sum(c.values()) == 3 and max(c.values()) == 2  # unassigned counts toward nothing
    top = c.most_common(1)[0][0]
    assert mt.persisted_count(rows, top, CFG) == 2 and mt.persisted_count(rows, "registry:none", CFG) == 0


def test_registry_shape():
    reg = mt.load_registry(CFG)
    assert reg.version == "v1-draft-s4-0" and len(reg.entries) >= 10
    for e in reg.entries:
        assert e["parent_openalex_topic"] == "none" and e["status"] == "planner_draft"
        assert 40 <= len(e["anchor_text"].split()) <= 120 and e["anchor_text"].isascii()
