"""STREAM-1031 core config: the live block, its accessor, persist_timeout_s == db_statement_timeout_ms, and TB5 (the
termination bound T_max is finite and every term is non-negative, computed from the loaded config; single-pass variant).
Offline."""
from __future__ import annotations

import json
import math
from pathlib import Path

from vera import acq_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "config" / name).read_text())


def test_live_block_values_note_and_accessor():
    cfg = acq_config.load_config()
    live = cfg["live"]
    assert (live["stall_s"], live["max_reuse_checks_per_prompt"], live["persist_timeout_s"]) == (300, 31, 15)
    n = live["_calibration_note"]
    for w in ("PROVISIONAL", "unmeasured", "OD-1031-CEIL31", "OD-1031-HITL4"):
        assert w in n
    assert acq_config.live(cfg) == {"stall_s": 300, "max_reuse_checks_per_prompt": 31, "persist_timeout_s": 15}
    assert live["max_reuse_checks_per_prompt"] == cfg["bounds"]["max_docs_per_prompt"]  # 31 is the Stage A ceiling


def test_live_accessor_never_returns_an_unbounded_value():
    for bad in (None, "x", 0, -1, True, float("inf")):
        v = acq_config.live({"live": {"stall_s": bad, "max_reuse_checks_per_prompt": bad, "persist_timeout_s": bad}})
        assert v == {"stall_s": 300, "max_reuse_checks_per_prompt": 31, "persist_timeout_s": 15}
    assert acq_config.live(None)["stall_s"] == 300


def test_persist_timeout_equals_db_statement_timeout():
    cfg = acq_config.load_config()
    assert cfg["live"]["persist_timeout_s"] * 1000 == cfg["db_statement_timeout_ms"] == 15000


def _tmax(cfg, rungs, evals, workers):
    live, rl = acq_config.live(cfg), _load("rate_limits.json")["defaults"]
    chain = _load("ask-provider-chain.json")
    s = cfg["search"]
    t_search, w = float(s["timeout_s"]), float(rl["max_wait_s"])
    t_land = float(s["ddg_first"]["landing_timeout_s"])
    calls = int(cfg["bounds"]["max_search_calls_per_job"])
    ar = cfg["arxiv_route"]
    t_doc = max(float(cfg["inline"]["per_doc_timeout_s"]), 2 * float(ar["max_crawl_delay_s"]) + float(cfg["bounds"]["fetch_timeout_s"]))
    waves = math.ceil(evals / workers) + rungs
    t_db = cfg["db_statement_timeout_ms"] / 1000.0
    llm = 1 + 2 * (1 + 4 + 1)  # router + 2 x (writer + max_parts part checks + whole check), per PLAN s2.6(c)
    terms = {
        "pre": float(chain["timeout_s"]) + 60.0 + 2 * 10.0,
        "search": rungs * calls * (t_search + w) + 3 * t_land,
        "evaluation": waves * t_doc,
        "reuse": live["max_reuse_checks_per_prompt"] * t_db,
        "persist": cfg["bounds"]["max_docs_per_prompt"] * live["persist_timeout_s"],
        "llm": llm * len(chain["providers"]) * float(chain["timeout_s"]),
    }
    return terms, sum(terms.values())


def test_TB5_t_max_is_finite_with_every_term_non_negative():
    cfg = acq_config.load_config()
    workers = int(cfg["inline"]["fetch_workers"])
    terms, total = _tmax(cfg, rungs=5, evals=93, workers=workers)  # the full PLAN s2.6 shape (multi-rung ladder)
    assert all(math.isfinite(v) and v >= 0 for v in terms.values()) and math.isfinite(total)
    assert total == 3675.0  # PLAN s2.6(e): about 61 min
    # single pass over one 31-candidate shortlist (this child): strictly smaller
    t1, one = _tmax(cfg, rungs=1, evals=31, workers=workers)
    assert all(math.isfinite(v) and v >= 0 for v in t1.values()) and 0 < one < total
    assert terms["reuse"] == 465.0 and terms["persist"] == 465.0 and terms["evaluation"] == 1450.0
