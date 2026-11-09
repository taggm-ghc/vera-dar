"""H19-FIX-1109 (design T6, section 8): page-check and nd_guard trace counters are closed tokens and counts only, and
their detail keys are allowlisted in config/request_trace.json. No URL, host or text in the trace."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import fulldoc_answer as fa, inline_acquire as ia, request_trace

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def file_allowlist(monkeypatch):
    """The shipped config file's allowlist (another test may have replaced the module CFG)."""
    cfg = json.loads((ROOT / "config" / "request_trace.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(request_trace, "CFG", cfg)
    from vera import list_tables
    monkeypatch.setattr(list_tables, "get", lambda key, default=None, *a, **k: default)


def test_lane_trace_carries_page_check_counters(monkeypatch):
    seen = {}
    monkeypatch.setattr(ia.rt, "event", lambda stage, code, counts=None, detail=None: seen.update(c=counts, d=detail))
    checks = {"nd_read": 1, "nd_link_only": 2, "page_link_only": 3, "acc_exc": 3, "a11_drop": 4,
              "acc_exc_first": "acc_republished"}
    ia._trace_lane("What are the 2026 midterm elections results?",
                   SimpleNamespace(diag=[{"portal": "gdelt", "status": "called", "reason": "ok", "items": 2}],
                                   pages=[], links=[], checks=checks))
    for k in ("nd_read", "nd_link_only", "page_link_only", "acc_exc", "a11_drop"):
        assert seen["c"][k] == checks[k]
    assert request_trace.clean_counts(seen["c"])["acc_exc"] == 3
    assert seen["d"]["acc_exc_first"] == "acc_republished"
    kept = request_trace.clean_detail(seen["d"], stage="lane")
    assert kept.get("acc_exc_first") == "acc_republished"


def test_lane_trace_without_checks_adds_nothing(monkeypatch):
    seen = {}
    monkeypatch.setattr(ia.rt, "event", lambda stage, code, counts=None, detail=None: seen.update(c=counts, d=detail))
    ia._trace_lane("What are the 2026 midterm elections results?", SimpleNamespace(diag=[], pages=[], links=[]))
    assert "acc_exc" not in seen["c"] and "acc_exc_first" not in seen["d"]


def test_nd_guard_trace_counts_and_first_rule(monkeypatch):
    seen = []
    monkeypatch.setattr(request_trace, "event", lambda stage, code, counts=None, detail=None:
                        seen.append((stage, code, counts, detail)))
    art = ("The canvass board voted four to one to accept the totals despite objections from two observers. "
           "County officials certified the results on Thursday.")
    src = [{"n": 1, "snippet": art, "canonical_text": art, "licence_class": "nd", "url": "https://v.example/a",
            "title": "T"}]
    kept = [{"id": 0, "sentence": "The canvass board voted four to one to accept the totals despite objections [1].",
             "citations": [1], "verdict": "supported"}]
    out, dropped = fa._nd_guard(kept, src, {}, "q")
    assert dropped == 1 and out == []
    stage, code, counts, detail = seen[-1]
    assert (stage, code) == ("claim_check", "nd_guard") and counts["nd_guard_drop"] == 1 and counts["nd_g1"] == 1
    assert detail == {"nd_guard_rule": "g1"}
    assert request_trace.clean_detail(detail, stage="claim_check") == {"nd_guard_rule": "g1"}
    assert art[:30] not in repr(seen)
