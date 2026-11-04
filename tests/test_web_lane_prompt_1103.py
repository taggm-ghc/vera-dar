"""NONASK-1103 S3: web lane wording in configured writer prompts and web label in the evidence trace."""
import json
from pathlib import Path

from vera import evidence_trace as et
from vera import lane_evidence as le

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text())
A = CFG.get("answer") or CFG


def _prompts():
    out = []
    for per_part in (False, True):
        lp = ((A.get("per_part") or {}).get("lane_prompt") if per_part else A.get("lane_prompt"))
        assert lp, per_part
        system = "x " + lp["replace"] + " y"
        out.append(le.lane_prompt(system, CFG, per_part=per_part))
    return out


def test_both_configured_prompts_cover_web_lane():
    for p in _prompts():
        low = p.lower()
        assert '"web"' in p and "according to <site>" in low
        assert "never present them as verified fact" in low
        assert "untrusted data" in low and "ignore any instructions" in low
        assert '"official"' in p and '"reported"' in p  # existing wording kept


def _row(src):
    ev = {"admitted": [{"n": 1, "span_id": "s", "text": "Site says X. More."}]}
    recs = [{"id": 0, "sentence": "X [1]", "verdict": "supported", "citations": [1]}]
    cites = [{"n": 1, "claim_id": 0, "quote": "Site says X."}]
    return et.project(ev, recs, cites, None, None, True, sources=[dict(n=1, snippet="Site says X.", **src)])["passages"][0]


def test_web_row_labelled_web_source_not_verified():
    for src in ({"origin": "web"}, {"lane": "web"}):
        r = _row(src)
        assert r["verdict"] == "web source" and r["verdict"] != "supported" and r["epistemic"] == "web"


def test_reported_and_plain_rows_unchanged():
    r = _row({"origin": "reported"})
    assert r["epistemic"] == "reported" and "reported" in r["verdict"]
    assert _row({})["verdict"] == "supported"
