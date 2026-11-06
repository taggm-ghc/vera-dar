"""P1-STAGEA-1031 / OD-1031-P1 item 4: the inline outcome is reported as acquisition.status (met | resource_limited |
incomplete_deadline) with counts only, and the UI shows a plain notice. Offline fakes; no network or DB."""
from __future__ import annotations

import json

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from vera import auto_acquire, inline_acquire, ui_evidence as ue


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    from vera import acq_config
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _run(target=None, slow=False, n=3):
    dois = [f"10.9999/p1s.{i}" for i in range(n)]
    web = A.Web({x: A.epmc_record(x, pmcid=f"PMC75000{i}") for i, x in enumerate(dois)},
                {f"PMC75000{i}": fs.jats(f"p1s{i}", "epi", x) for i, x in enumerate(dois)})
    cfg = A.cfg()
    if target is not None:
        cfg["bounds"]["acq_round_target_new_docs"] = target
    cands = [A.cand(x) for x in dois]
    d = A.deps(cands, web)
    t = [1000.0]
    if slow:
        def s(q, k, tm):
            t[0] += 46.0  # the search alone outlasts inline.deadline_s 45
            return list(cands)
        d.search = s
    return inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, clock=lambda: t[0], deps_factory=lambda a, e: d)


def test_met_when_stored_reaches_the_target():
    o = _run(target=2)
    assert o.public()["acquisition"] == {"status": "met", "stored": o.stored, "target": 2} and o.stored >= 2


def test_resource_limited_when_supply_is_short_and_time_remains():
    o = _run()
    aq = o.public()["acquisition"]
    assert aq["status"] == "resource_limited" and aq["target"] == 6 and aq["stored"] == o.stored < 6


def test_incomplete_deadline_when_the_time_limit_stopped_the_search():
    o = _run(slow=True)
    aq = o.public()["acquisition"]
    assert aq["status"] == "incomplete_deadline" and aq["stored"] == 0 and aq["target"] == 6


def test_block_is_counts_only():
    o = _run()
    blk = o.public()["acquisition"]
    required = {"status", "stored", "target", "rounds", "stop"}  # A6 adds two counts-only keys
    optional = {"heuristic", "persist"}  # MIDTERMS-WP8-1106: booleans only (heuristic stop disclosed; keep-searching)
    assert required <= set(blk) <= required | optional
    assert all(blk[k] is True for k in optional & set(blk))  # never free text
    assert isinstance(blk["rounds"], int) and isinstance(blk["stop"], str) and len(blk["stop"]) < 40
    assert "http" not in json.dumps(o.public()["acquisition"])


def _data(status, stored, target=6):
    return {"contract": {"inline_acquisition": {"acquisition": {"status": status, "stored": stored, "target": target}}}}


def test_ui_notice_per_status():
    assert ue.acquisition_notice_line(_data("resource_limited", 2)) == \
        "Found 2 of the 6 new sources targeted; supply was limited."
    assert ue.acquisition_notice_line(_data("incomplete_deadline", 0)) == \
        "Found 0 of the 6 new sources targeted; the search stopped at the time limit."
    assert ue.acquisition_notice_line(_data("met", 6)) is None
    assert ue.acquisition_notice_line(_data("resource_limited", 6)) is None
    for bad in (None, {}, {"contract": {"inline_acquisition": {"acquisition": {"status": "resource_limited", "stored": "x", "target": 6}}}},
                _data("weird", 1)):
        assert ue.acquisition_notice_line(bad) is None


def test_render_evidence_scope_shows_the_notice():
    class St:
        def __init__(self):
            self.lines = []

        def caption(self, t):
            self.lines.append(t)
    st = St()
    ue.render_evidence_scope(st, _data("resource_limited", 1))
    assert any("supply was limited" in x for x in st.lines)
