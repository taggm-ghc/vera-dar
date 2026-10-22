"""P0-B1b (demo-1022): a declined (off-topic) non-answer must carry the inline acquisition outcome in its contract, as the
in-scope path does (P0-B1), so the Report states whether a new search was made. Offline, synthetic fixtures only."""
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from tests.test_dar_inline_acquire import rig  # noqa: F401  (fixture)
from vera import ask_service
from vera.m7 import live_report


def _report_status(r):
    data = {"result_kind": r.result_kind, "contract": r.contract, "candidates": list(r.candidates),
            "tokens_used": r.tokens_used, "cost_usd": r.cost_usd}
    return ((live_report.build_live_report(data, "q", 1.0).get("search") or {}).get("inline") or {}).get("status")


def test_declined_with_acquisition_off_is_reported(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: None)
    r = rig.run()
    assert r.result_kind != "answer" and rig.search_calls == []
    assert (r.contract or {}).get("inline_acquisition") == {"status": "off"}
    assert _report_status(r) == live_report.INLINE_STATUS["off"]


def test_declined_inline_search_with_nothing_admitted_is_reported_as_none(rig):
    rig.setup(topic="clim")
    r = rig.run()
    assert r.result_kind != "answer" and rig.search_calls
    assert r.contract["inline_acquisition"]["status"] == "none"
    assert _report_status(r) == live_report.INLINE_STATUS["none"]
