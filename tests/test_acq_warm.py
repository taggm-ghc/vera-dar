"""WED-ACQ2 rebase onto P1.6.1: the startup warm-up (vera/fulldoc_warm.py, single-flight fulldoc_db.cached_load) and an
acquisition job plus the acquired-corpus refresh running AT THE SAME TIME. OFFLINE: synthetic bundle rows behind a gated
fake DB-API connection (the base SELECTs block until released), fake Europe PMC transport, fake WO store. Asserts: ONE
base load (no double load), no deadlock (every thread finishes), the job stores its document, and the acquired corpus
is served afterwards. No network, no DB, no model."""
import threading

import pytest

from tests import acq_support as A
from tests import test_fulldoc_warm as tw
from vera import acq_config, auto_acquire, fulldoc_corpus as fc, fulldoc_db, fulldoc_warm

WAIT = 10.0
ENV = {**tw.ENV_DB}


class Mux:
    """One DB-API connection that routes the base SELECTs to the gated fake and the acquired SELECTs to a fake over the
    plans the job has stored so far."""

    def __init__(self, gate, plans):
        self.b, self.a, self.cur = gate.connect(), A.FakeConn(plans), None

    def cursor(self):
        return self

    def execute(self, sql, params):
        if sql in (fulldoc_db.SQL_ACQ_MAX, fulldoc_db.SQL_ACQ_SOURCES, fulldoc_db.SQL_ACQ_INDEX,
                   fulldoc_db.SQL_ACQ_SOURCES_BY_ID):
            self.cur = self.a
        elif sql == fulldoc_db.SQL_SOURCES:
            self.cur = self.b
        self.cur.execute(sql, params)

    def fetchall(self):
        return self.cur.fetchall()

    def close(self):
        pass


@pytest.fixture()
def world(tw_world):
    yield tw_world


@pytest.fixture()
def tw_world(tmp_path):
    fc.reset_for_tests()
    fulldoc_warm.reset_for_tests()
    auto_acquire.reset_for_tests()
    gen = tw.world.__wrapped__(tmp_path)
    yield next(gen)
    fc.reset_for_tests()
    fulldoc_warm.reset_for_tests()
    auto_acquire.reset_for_tests()


def test_startup_warm_and_acquisition_together_one_load_no_deadlock(world, monkeypatch):
    _root, cfg, rows = world
    gate = tw.Gate(rows)
    plans = []
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda c, env: (lambda: Mux(gate, plans)))
    monkeypatch.setattr(fc, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(acq_config, "db_configured", lambda c, env: True)

    warm = fulldoc_warm.start(cfg, ENV)
    assert warm is not None and gate.entered.wait(WAIT)  # the warm-up holds the base load

    class Store(A.Store):
        def write(self, plan):
            out = super().write(plan)
            plans.append(self.plans[-1])
            return out

    web, _ = A.epi_setup()
    store = Store()
    d = A.deps([A.cand("10.9999/acq.epi4")], web, store=store)
    seen_keys = []

    def corpus_keys():  # as default_deps: the served base corpus (blocks on the warm-up's single flight) + acquired
        out = set()
        for c in (fc.get_corpus(cfg, ENV), fc.get_acquired_corpus(cfg, ENV)):
            dar = getattr(c, "_dar", None)
            if dar is not None:
                out |= set(dar.doc_keys())
        seen_keys.append(len(out))
        return out
    d.corpus_keys = corpus_keys
    d.refresh = lambda: (fc.invalidate_acquired(), fc.refresh_acquired(cfg, ENV, sync=True))
    result = {}
    job = threading.Thread(target=lambda: result.update(auto_acquire.run_job(
        A.job(), A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)), name="test-acq-job", daemon=True)
    job.start()
    ask = threading.Thread(target=lambda: result.update(ask=fc.get_corpus(cfg, ENV)), daemon=True)  # a /ask meanwhile
    ask.start()
    gate.release.set()
    for t in (warm, job, ask):
        t.join(WAIT)
        assert not t.is_alive(), f"{t.name} did not finish (deadlock?)"
    assert gate.n_loads() == 1  # warm-up, job and request shared ONE base load (P161 single-flight kept)
    assert fulldoc_warm.state() == fulldoc_warm.DONE and result["ask"].status == "ok"
    assert result["stored"] == 1 and result["codes"] == {"ACQ-STORED": 1}, result
    assert seen_keys and seen_keys[0] > 0  # the job saw the base corpus' keys (after the warm-up's load)
    acq = fc.get_acquired_corpus(cfg, ENV)
    assert acq is not None and len(acq.acquired) == 1 and acq.acquired[0]["document_id"].startswith("acq-")
    assert gate.n_loads() == 1  # serving the acquired corpus never reloads the base
