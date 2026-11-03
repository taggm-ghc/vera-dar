"""NONASK-1103 WP-H: reading-list relevance floor (flag off by default). Offline; synthetic. Calibration note stub:
min_relevance and max_df_ratio are uncalibrated placeholders; calibrate on the WP-E harness (CTQ5 >= 0.8 on holdout) per
records/memory/feedback_threshold_calibration_rule.md. No threshold is enabled."""
import copy
import json
from pathlib import Path

from vera import grounded_ask as ga
from vera.dar_corpus import DarCorpus

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((ROOT / "config" / "domain_routing.json").read_text())
Q = "multi agent orchestration"
MARL = {"url": "https://e.example/marl", "title": "Multi agent reinforcement learning", "snippet": "Many agents learn policies."}
HANDLE = {"url": "https://e.example/handle", "title": "Multi agent orchestration with Handle records",
          "snippet": "Orchestration of agents via persistent identifiers."}
OTHER = {"url": "https://e.example/other", "title": "Multi agent systems", "snippet": "Agent systems overview."}
STORED = {"url": "https://s.example/1", "title": "stored", "snippet": "zzz", "span": "0-5"}


def corp(floor=None):
    cfg = copy.deepcopy(SHIPPED)
    cfg["enabled"] = True
    if floor is not None:
        cfg["corpus"]["reading_list_floor"] = floor
    return DarCorpus(cfg)


def test_shipped_flag_absent_or_off():
    f = SHIPPED["corpus"].get("reading_list_floor")
    assert f is not None and f.get("enabled") is False


def test_flag_off_identity_floor_and_rank_merge():
    ext = [MARL, HANDLE, OTHER]
    for fl in (None, {"enabled": False, "min_relevance": 0.9, "max_df_ratio": 0.1}, {"enabled": True}):
        c = corp(fl)
        assert c.floor_external(Q, ext) is ext  # identity (also: enabled without thresholds fails open)
        assert c.rank_merge(Q, list(ext), [STORED], 5) == corp(None).rank_merge(Q, list(ext), [STORED], 5)


def test_flag_on_drops_marl_keeps_handle_and_stored():
    c = corp({"enabled": True, "min_relevance": 0.3, "max_df_ratio": 0.5})
    out = c.floor_external(Q, [MARL, HANDLE, STORED])
    assert HANDLE in out and STORED in out and MARL not in out


def test_flag_on_rank_merge_drops_below_floor_even_in_head():
    c = corp({"enabled": True, "min_relevance": 0.3, "max_df_ratio": 0.5})
    out = c.rank_merge(Q, [MARL, HANDLE], [STORED], 5)
    assert MARL not in out and HANDLE in out


def test_merge_stored_no_stored_branch_floor_applies_and_off_is_identity():
    ext = [MARL, HANDLE]
    c = corp({"enabled": True, "min_relevance": 0.3, "max_df_ratio": 0.5})
    c.cfg["corpus"]["stored_branch_enabled"] = True
    c.as_sources = lambda q: ([], {})
    assert ga._merge_stored(ext, Q, c, {"num_results": 5}, {}) == [HANDLE]
    c0 = corp(None)
    c0.cfg["corpus"]["stored_branch_enabled"] = True
    c0.as_sources = lambda q: ([], {})
    assert ga._merge_stored(ext, Q, c0, {"num_results": 5}, {}) is ext
