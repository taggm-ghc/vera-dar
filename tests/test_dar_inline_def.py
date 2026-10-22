"""INLINE-DEF-1021: defect 3 (outcome cache keyed on question + tried-document set) and defect 4 (docs_per_day
reservation released when the inline run raises). Offline, synthetic fixtures."""
import pytest

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests.test_dar_inline_acquire import direct  # noqa: F401  (fixture)
from vera import auto_acquire, inline_acquire

Q = e2e.Q_EPI


def test_defect3_cached_none_not_reused_when_tried_set_changes(direct):
    cfg = A.cfg()
    d = A.deps([], A.epi_setup()[0])
    calls = []
    inner = d.search
    d.search = lambda q, n, t: calls.append(q) or inner(q, n, t)
    f = lambda a, e: d  # noqa: E731
    a = inline_acquire.acquire(Q, cfg, env=A.ENV_ON, deps_factory=f, served={"k1"})
    assert a.status == "none" and len(calls) == 1
    assert inline_acquire.acquire(Q, cfg, env=A.ENV_ON, deps_factory=f, served={"k1"}) is a and len(calls) == 1
    b = inline_acquire.acquire(Q, cfg, env=A.ENV_ON, deps_factory=f, served={"k1", "k2"})  # new documents tried
    assert b is not a and len(calls) == 2


def test_defect4_reservation_released_when_inline_run_raises(direct, monkeypatch):
    cfg = A.cfg()
    d = A.deps([A.cand("10.9999/acq.epi4")], A.epi_setup()[0])
    before = len(auto_acquire._S.docs_day)

    held = []

    def boom(*a, **k):
        held.append(auto_acquire._S.docs_reserved)  # the reservation is held while the run is in progress
        raise RuntimeError("boom")
    monkeypatch.setattr(auto_acquire, "rank_candidates", boom)
    o = inline_acquire.acquire(Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d)
    assert o.status == "error" and held and held[0] > 0
    assert auto_acquire._S.docs_reserved == 0 and len(auto_acquire._S.docs_day) == before
    assert auto_acquire.docs_budget(cfg, d) == auto_acquire.docs_budget(A.cfg(), d) > 0
