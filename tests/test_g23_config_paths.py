"""GAPTESTS-1023 G4 / G5 / L-E1: real config file load, in-scope path, rollback by file edit. Offline."""
import json

from tests import acq_support as A
from tests._g23_support import REAL_LOAD_CONFIG, SHIPPED_TEXT, http_ask, use_file, variant_text
from tests._gap_support import stored_text
from tests._pipeline_support import (GROUND, PRICING, Q_EPI, DiscSearch, _decide, discovered, disc_entries, epmc_web,  # noqa: F401
                                     h, passages_sent_to_model, rig)
from vera import acq_config, ask_service, fulldoc_corpus


def _insufficient_base(monkeypatch):
    real = ask_service._fulldoc
    tok = fulldoc_corpus.load_config()["fallback_tokens"]["model_insufficient"]
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: (None, tok, 0, 0.0, "x")
                        if (k.get("ephemeral") is None and not k.get("acquired_only")) else real(*a, **k))


def _fetched(rig, frag):
    return [u for u in rig.web.urls if frag in u]


# ---------------------------------------------------------------- G5: the shipped file itself
def test_G5_shipped_file_parsed_by_real_loader_has_discovery_on_landing_off(tmp_path):
    cfg = REAL_LOAD_CONFIG(acq_config.CONFIG_PATH)       # default path, no override anywhere
    assert cfg["enabled"] is True and cfg["inline"]["enabled"] is True
    assert cfg["inline"]["use_discovered"] is True and cfg["inline"]["landing_route"] is False
    assert cfg["inline"]["in_scope"] is True and cfg["kill_switch_env"] == "VERA_AUTOACQ_DISABLED"
    assert cfg["inline"]["discovered_max"] <= cfg["bounds"]["max_candidates_resolved"] + 0


def test_G5_unedited_shipped_file_drives_http_ask_to_a_discovered_answer(rig, monkeypatch, tmp_path):
    """Only the file text (override_layer2 is already true in the shipped file) -> no flag set anywhere in the test."""
    p = use_file(rig, monkeypatch, tmp_path, text=SHIPPED_TEXT)
    assert p.read_text() == SHIPPED_TEXT
    assert json.loads(p.read_text())["inline"]["use_discovered"] is True
    web, _ = epmc_web()
    rig.setup(web=web, cands=[])
    resp, searches = http_ask(rig, monkeypatch, [discovered()])
    body = resp.json()
    assert resp.status_code == 200 and searches
    assert body["result_kind"] == "answer", body.get("contract")
    assert [e["outcome"] for e in body["contract"]["inline_acquisition"]["discovered"]] == ["admitted"]
    assert all(a["document_id"].startswith("acq-") for a in body["admitted_evidence"])


# ---------------------------------------------------------------- G4: in-scope corpus miss on the shipped file
def test_G4_in_scope_corpus_miss_on_shipped_file_answers_from_new_full_text(rig, monkeypatch, tmp_path):
    use_file(rig, monkeypatch, tmp_path, text=SHIPPED_TEXT)
    assert rig.cfg["inline"]["in_scope"] is True and rig.cfg["inline"]["use_discovered"] is True
    web, _ = epmc_web()
    rig.setup(web=web, cands=[])
    _insufficient_base(monkeypatch)
    _decide(rig.mp, True, "layer1", "ok", "")
    ds = DiscSearch([discovered()])
    r = ask_service.answer_in_scope(Q_EPI, "m", PRICING, grounding=(ds, GROUND[1]))
    assert r.result_kind == "answer" and r.evidence is not None and not r.contract.get("offtopic")
    assert ds.calls == [False]
    assert all(a["document_id"].startswith("acq-") for a in r.evidence.admitted)
    assert all(t in stored_text(rig) for t in passages_sent_to_model(rig))
    assert [e["outcome"] for e in disc_entries(r)] == ["admitted"]


# ---------------------------------------------------------------- L-E1: rollback by editing a file copy
def _run_in_scope(rig, monkeypatch, tmp_path, text):
    use_file(rig, monkeypatch, tmp_path, text=text)
    web, _ = epmc_web()
    rig.setup(web=web, cands=[])
    _insufficient_base(monkeypatch)
    _decide(rig.mp, True, "layer1", "ok", "")
    ds = DiscSearch([discovered()])
    return ask_service.answer_in_scope(Q_EPI, "m", PRICING, grounding=(ds, GROUND[1])), ds


def test_LE1_use_discovered_false_in_file_copy_stops_discovered_acquisition(rig, monkeypatch, tmp_path):
    text = variant_text(use_discovered=False)
    assert text != SHIPPED_TEXT and json.loads(text)["inline"]["use_discovered"] is False
    assert json.loads(SHIPPED_TEXT)["inline"]["use_discovered"] is True   # the copy differs only by the edit
    r, ds = _run_in_scope(rig, monkeypatch, tmp_path, text)
    assert rig.cfg["inline"]["use_discovered"] is False
    assert r.result_kind != "answer" and rig.store.plans == []
    assert _fetched(rig, "/fullTextXML") == [] and disc_entries(r) == []


def test_LE1_top_level_enabled_false_in_file_copy_stops_all_inline_acquisition(rig, monkeypatch, tmp_path):
    text = variant_text(enabled=False)
    assert json.loads(text)["enabled"] is False
    r, ds = _run_in_scope(rig, monkeypatch, tmp_path, text)
    assert r.result_kind != "answer" and rig.store.plans == [] and rig.web.urls == []
    assert rig.outcomes == [] or all(o.admitted == 0 for o in rig.outcomes)


def test_LE1_the_same_flow_with_the_unedited_file_does_answer(rig, monkeypatch, tmp_path):
    """Control for the two rollback tests: only the file text differs."""
    r, _ds = _run_in_scope(rig, monkeypatch, tmp_path, SHIPPED_TEXT)
    assert r.result_kind == "answer"
