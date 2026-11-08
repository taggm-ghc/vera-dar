"""PROCFIX-RELKIT-1108: OD-1104-TRACE reopening behind a flag. Shipped OFF: premise_unverified keeps the plain insufficient text."""
import json
from pathlib import Path

from vera import ui_evidence as ue

CONF = Path(__file__).resolve().parent.parent / "config" / "premise_text.json"


def test_shipped_flag_is_off():
    c = json.loads(CONF.read_text(encoding="utf-8"))
    assert c["enabled"] is False  # pins the shipped value; enabling is an owner decision after calibration


def test_off_keeps_old_text():
    assert ue.premise_unverified_text() == ""
    assert ue.fallback_text("premise_unverified", cfg={}) == ue.fallback_text("insufficient_full_document_evidence", cfg={})


def _point(monkeypatch, tmp_path, body):
    p = tmp_path / "pt.json"
    p.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    monkeypatch.setattr(ue, "PREMISE_TEXT_CONFIG", p)


def test_on_shows_distinct_reason_for_premise_token_only(monkeypatch, tmp_path):
    _point(monkeypatch, tmp_path, {"enabled": True, "text": "Could not confirm an assumption."})
    assert ue.fallback_text("premise_unverified") == "Could not confirm an assumption."
    assert ue.fallback_text("model_insufficient") != "Could not confirm an assumption."
    assert "premise_unverified" not in ue.fallback_text("premise_unverified")


def test_bad_config_falls_back(monkeypatch, tmp_path):
    for body in ("{not json", {"enabled": "true", "text": "x"}, {"enabled": True, "text": ""}, {"enabled": True, "text": "{n}"}, {"enabled": 1, "text": "x"}):
        _point(monkeypatch, tmp_path, body)
        assert ue.premise_unverified_text() == ""
    monkeypatch.setattr(ue, "PREMISE_TEXT_CONFIG", tmp_path / "missing.json")
    assert ue.premise_unverified_text() == ""
