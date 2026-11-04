"""SA-FIX-1031: review fixes for Stage A. Acquisition status causes (search_failed, capped, resource_limited), the UI
notice per cause (counts only), the notice on reading-list non-answers, and the legacy inline_fresh_docs clamp.
Offline fakes; no network or DB."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from vera import acq_config, auto_acquire, inline_acquire, ui_evidence as ue


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _run(flag=None, fail=False, daily=None, n=3):
    dois = [f"10.9999/fx.{i}" for i in range(n)]
    web = A.Web({x: A.epmc_record(x, pmcid=f"PMC76000{i}") for i, x in enumerate(dois)},
                {f"PMC76000{i}": fs.jats(f"fx{i}", "epi", x) for i, x in enumerate(dois)})
    cfg = A.cfg()
    if flag is not None:
        cfg.setdefault("inline", {})["per_prompt_ceiling_enabled"] = flag
    if daily is not None:
        cfg["bounds"]["docs_per_day"] = daily
    d = A.deps([A.cand(x) for x in dois], web)
    if fail:
        def boom(q, k, tm):
            raise RuntimeError("down")
        d.search = boom
    return inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=lambda: 1000.0, deps_factory=lambda a, e: d)


def test_search_failure_is_search_failed_not_resource_limited():
    aq = _run(flag=True, fail=True).public()["acquisition"]
    assert aq["status"] == "search_failed" and aq["stored"] == 0 and aq["target"] == 6


def test_kill_switch_is_capped_not_resource_limited():
    aq = _run(flag=False).public()["acquisition"]  # budget 2 < target 6
    assert aq["status"] == "capped" and aq["stored"] <= 2


def test_spent_daily_budget_is_capped():
    aq = _run(flag=True, daily=0).public()["acquisition"]
    assert aq["status"] == "capped"


def test_resource_limited_only_with_full_budget_and_working_search():
    aq = _run(flag=True).public()["acquisition"]
    assert aq["status"] == "resource_limited" and aq["stored"] < aq["target"]


@pytest.mark.parametrize("status,words", [
    ("resource_limited", "supply was limited"), ("incomplete_deadline", "time limit"),
    ("search_failed", "could not be completed"), ("capped", "limit on new acquisition runs")])
def test_notice_states_each_cause_counts_only(status, words):
    line = ue.acquisition_notice_line({"contract": {"inline_acquisition": {"acquisition": {
        "status": status, "stored": 1, "target": 6}}}})
    assert line.startswith("Found 1 of the 6") and words in line
    for bad in ("http", "openalex", "europe", "duckduckgo", "arxiv", "kill", "switch", "R-ACQ"):
        assert bad not in line.lower()


class _St:
    def __init__(self):
        self.out = []

    def caption(self, t):
        self.out.append(("caption", t))

    def info(self, t):
        self.out.append(("info", t))


def test_notice_shown_on_plain_reading_list_non_answer():
    st = _St()
    ue.render_fallback_reason(st, {"contract": {"inline_acquisition": {"acquisition": {
        "status": "search_failed", "stored": 0, "target": 6}}}})
    assert len(st.out) == 1 and st.out[0][0] == "caption" and "Found 0 of the 6" in st.out[0][1]
    st2 = _St()
    ue.render_fallback_reason(st2, {})
    assert st2.out == []


def test_legacy_override_clamped_to_ceiling():
    cfg = acq_config.load_config()
    cfg["inline"]["per_prompt_ceiling_enabled"] = True
    cfg["bounds"]["inline_fresh_docs"] = 500
    assert acq_config.inline_fresh_docs(cfg) == cfg["bounds"]["max_docs_per_prompt"] == 31
    cfg["bounds"]["inline_fresh_docs"] = 3
    assert acq_config.inline_fresh_docs(cfg) == 3
    cfg["inline"]["per_prompt_ceiling_enabled"] = False
    assert acq_config.inline_fresh_docs(cfg) == 2
