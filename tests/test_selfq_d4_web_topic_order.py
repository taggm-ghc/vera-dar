"""SELFQ-RCA-1107 D4 r2: web results are ordered by key-concept overlap (generic concepts excluded); zero-overlap results
are demoted, and hidden only when no result overlaps. Data: the SELFQ review probe set. Offline; no network."""
import json
from pathlib import Path

import pytest

from vera import web_route as wr

ROOT = Path(__file__).resolve().parent.parent
_T = json.loads((ROOT / "config" / "web_route.json").read_text())["topic_order"]
ON = {"topic_order": {**_T, "enabled": True}}  # the shipped value is false (pinned below); tests exercise the logic with it on


def _c(title, snippet=""):
    return {"url": f"https://a.example/{"".join(ch if ch.isalnum() else "-" for ch in title)}", "title": title, "snippet": snippet}


# (question, [(title, snippet, relevant)]) from scratch/SELFQ-REVIEW-1107/probe3.py, with the reviewer's relevance call
PROBE = [
    ("What did the Fed decide about interest rates this week?",
     [("Federal Reserve holds rates steady", "Central bank keeps benchmark rate", True),
      ("Powell says inflation cooling", "FOMC statement", True), ("Fed leaves borrowing costs unchanged", "", True),
      ("Stocks rally as Fed pauses", "", True)]),
    ("How does CRISPR gene editing work?",
     [("CRISPR-Cas9 explained", "How the gene-editing tool cuts DNA", True), ("Genome engineering with Cas9", "", True),
      ("Gene editing breakthrough in sickle cell", "", True), ("This blog could help you poop better - The Verge", "", False)]),
    ("What are the effects of the EU AI Act on small businesses?",
     [("EU artificial intelligence regulation: what SMEs need to know", "", True), ("AI Act compliance for startups", "", True),
      ("Brussels finalises rules on AI", "European lawmakers", True), ("SpaceX designed an orbital Vera Rubin", "", False)]),
    ("What caused the 2008 financial crisis?",
     [("Subprime mortgage collapse and Lehman", "", True), ("Great Recession origins", "housing bubble", True),
      ("The 2008 crash explained", "", True)]),
    ("Is the new COVID booster safe for children?",
     [("Pediatric vaccine safety review", "CDC advisers", True), ("Vaccine for kids: what parents should know", "", True),
      ("Coronavirus shots and children", "", True)]),
    ("What is VERA-DAR?", [("About VERA-DAR", "", True), ("Vera Rubin telescope", "", False), ("DAR", "", False)]),
]


@pytest.mark.parametrize("q,rows", PROBE)
def test_probe_relevant_links_never_hidden_and_junk_never_above_them(q, rows):
    cands = [_c(t, s) for t, s, _ in rows]
    out = wr.topic_order(q, cands, ON)
    titles = [c["title"] for c in out]
    rel = [t for t, _, r in rows if r]
    junk = [t for t, _, r in rows if not r]
    assert all(t in titles for t in rel), titles  # paraphrased relevant titles are kept (demoted at most)
    on_topic_rel = [t for t in rel if wr.topic_order(q, [_c(t)], ON)]  # alone, it is not hidden: it overlaps
    for j in junk:
        if j in titles:  # junk is never listed above an on-topic relevant link
            assert all(titles.index(j) > titles.index(t) for t in on_topic_rel), titles


def test_generic_concepts_do_not_make_junk_on_topic():
    q = "Is the new COVID booster safe for children this week?"
    out = wr.topic_order(q, [_c("New week, new work: office tips"), _c("Coronavirus shots and children")], ON)
    assert [c["title"] for c in out] == ["Coronavirus shots and children", "New week, new work: office tips"]


def test_all_zero_pool_is_hidden_vera_dar_case():
    out = wr.topic_order("What is VERA-DAR?", [_c("This blog could help you poop better - The Verge"),
                                                _c("SpaceX designed an orbital Vera Rubin telescope")], ON)
    assert out == []


def _cfg_on():
    cfg = json.loads(json.dumps(wr.load_config()))
    cfg["topic_order"]["enabled"] = True
    return cfg


def test_demoted_link_listed_last_through_read_web():
    # distinct empty-text fetch: both end in listed, in topic order; on-topic first
    r = wr.read_web("What is VERA-DAR?", [_c("Vera Rubin telescope"), _c("About VERA-DAR", "VERA-DAR reading tool")],
                    fetch=lambda *a, **k: {}, cfg=_cfg_on())
    assert [x.get("title") if isinstance(x, dict) else getattr(x, "title", None) for x in r.listed][0].startswith("About")
    assert len(r.dev) == 2 and not any(d["gate"].get("read") == "topic_hidden" for d in r.dev)


def test_hidden_results_leave_a_dev_entry_and_are_never_fetched():
    calls = []
    r = wr.read_web("What is VERA-DAR?", [_c("This blog could help you poop better - The Verge"),
                                          _c("SpaceX designed an orbital Vera Rubin")],
                    fetch=lambda *a, **k: calls.append(a) or {}, cfg=_cfg_on())
    assert calls == [] and not r.listed and not r.pages
    assert [d["gate"]["read"] for d in r.dev] == ["topic_hidden", "topic_hidden"]
    assert all(d["fetched"] is False for d in r.dev)


def test_known_limitation_single_paraphrase_is_hidden():
    # F4: documented, pinned so a later change is deliberate
    out = wr.topic_order("What did the Fed decide about interest rates this week?", [_c("Powell says inflation cooling")], ON)
    assert out == []


def test_off_switch_and_fail_open():
    cands = [_c("Vera Rubin telescope")]
    assert wr.topic_order("What is VERA-DAR?", cands, {"topic_order": {"enabled": False}}) == cands
    assert wr.topic_order("What is VERA-DAR?", [_c("x")], None) == [_c("x")]  # shipped config is off
    assert wr.topic_order("the a of", cands, ON) == cands  # no concepts left: unchanged
    cfg = {"topic_order": {"enabled": True, "generic_concepts": ["vera-dar"]}}
    assert wr.topic_order("What is VERA-DAR?", cands, cfg) == cands  # only generic concepts: unchanged


def test_calibration_note_is_provisional_and_cites_probe_data():
    t = json.loads((ROOT / "config" / "web_route.json").read_text())["topic_order"]
    assert t["enabled"] is False  # shipped OFF (no owner approval)
    assert t["_calibration_note"].startswith("PROVISIONAL")
    assert "probe" in t["_calibration_note"] and "What would change it" in t["_calibration_note"]
