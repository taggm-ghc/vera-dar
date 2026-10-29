"""PROG-1027 W1b: config/progress.json and config/progress-stages.json (label coverage and banned-word lint). Offline."""
import json
import re
from pathlib import Path

from vera import progress as p

CFG_DIR = Path(p.__file__).resolve().parent.parent / "config"
LABELS = json.loads((CFG_DIR / "progress-stages.json").read_text(encoding="utf-8"))
TERMINAL_KEYS = {"answered", "partial", "no_answer", "declined", "error", "provider_error", "busy", "expired",
                 "poll_timeout_received", "poll_timeout_unreceived"}
# INTERFACE-CONTRACT s3 banned words plus the v2 list (gate, layer, R-ACQ, DEC-, M#, provider names).
BANNED = [r"scope", r"curated", r"scholarly-only", r"outside what VERA-DAR answers", r"\bgate", r"\blayer", r"R-ACQ", r"DEC-",
          r"\bM\d\b", r"provider", r"groq", r"openai", r"openrouter", r"mistral", r"basis", r"verdict", r"categor", r"eligib"]
PLACEHOLDERS = {"rid", "inspected", "admitted", "found", "checked", "usable", "budget", "stored", "target", "attempt", "minutes"}


def _strings():
    for k in ("sending", "finishing", "progress_unavailable", "reload_caption", "already_running", "background_line"):
        yield k, LABELS[k]
    for st, d in LABELS["stages"].items():
        for sa, t in d.items():
            yield f"stages.{st}.{sa}", t
    for k, t in LABELS["terminals"].items():
        yield f"terminals.{k}", t


def test_progress_config_validates_and_ships_off():
    c = p.load_config(CFG_DIR / "progress.json")
    assert c is not None and c.enabled is False
    raw = json.loads((CFG_DIR / "progress.json").read_text())
    assert "PROVISIONAL" in raw["_calibration"] and "2026-11-10" in raw["_calibration"]


def test_label_map_covers_every_stage_state_pair():
    for stage, states in p.VALID.items():
        assert {s.value for s in states} <= set(LABELS["stages"][stage.value]), stage
    # lifecycle states are mapped for every lifecycle stage too
    for stage in (p.Stage.SCOPE, p.Stage.CORPUS, p.Stage.SEARCH, p.Stage.ACQUIRE, p.Stage.WRITE, p.Stage.CHECK):
        assert {"start", "update", "done", "skipped"} <= set(LABELS["stages"][stage.value])
    assert set(LABELS["stages"]) == {s.value for s in p.Stage}


def test_terminal_keys_complete_and_server_done_is_neutral():
    assert set(LABELS["terminals"]) == TERMINAL_KEYS
    assert set(LABELS["stages"]["done"].values()) == {LABELS["finishing"]}
    assert LABELS["terminals"]["declined"] == "Finished" and LABELS["terminals"]["answered"] == "Finished"


def test_banned_word_lint_and_placeholders():
    for key, text in _strings():
        assert isinstance(text, str) and text.strip(), key
        for pat in BANNED:
            assert not re.search(pat, text, re.I), (key, pat)
        for ph in re.findall(r"\{(\w+)\}", text):
            assert ph in PLACEHOLDERS, (key, ph)
        assert not re.search(r"https?://|\d+\s?%", text), key  # no URLs, no percentages (no fabricated progress)


def test_no_timer_or_estimate_wording_and_background_line_has_no_number():
    for key, text in _strings():
        assert not re.search(r"\b(seconds?|minutes? left|remaining|eta|about \d)\b", text, re.I) or key.startswith("terminals.poll_timeout"), key
    assert not re.search(r"\d|\{", LABELS["background_line"])
    assert "six" not in json.dumps(LABELS).lower()  # inline counts never imply the six-document target


import pytest as _pytest


@_pytest.mark.parametrize("val,off", [("1", True), ("true", True), ("TRUE", True), (" Yes ", True), ("on", True), ("On", True), ("0", False), ("", False), ("no", False)])
def test_kill_switch_accepts_1_true_yes_on_case_insensitive(monkeypatch, val, off):
    from vera import progress
    monkeypatch.setattr(progress, "CFG", progress.CFG._replace(enabled=True) if hasattr(progress.CFG, "_replace") else __import__("dataclasses").replace(progress.CFG, enabled=True))
    monkeypatch.setenv(progress.KILL_SWITCH_ENV, val)
    assert progress.enabled() is (not off)
