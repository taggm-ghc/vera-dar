"""TAXO-MEM-1109: the chunk-taxonomy job holds vectors as float32 blocks (never Python float lists), applies the RSS guard on the
reuse path, before the build and per batch, and releases every buffer when a pass stops early or ends. Offline: in-memory fake
DB (vectors come back as plain float lists per page, like a driver), stub core, no network."""
from __future__ import annotations

import hashlib
import tracemalloc

import numpy as np

from vera import chunk_taxo_job as job
from tests.test_chunk_taxo_job_1109 import ENV, FakeDB, FakeEmb, Res, factory_for

MODEL = "fake"


class MatrixDB(FakeDB):
    """Spans and stored vectors for n rows; vectors live in ONE float32 matrix and are handed out as float lists per page."""

    def __init__(self, n, dim, has=(True, False, True)):
        super().__init__([(i, 100 + i % 7, f"span text {i}") for i in range(1, n + 1)], has=has)
        self.mat = np.random.default_rng(1).random((n, dim), dtype=np.float32)
        self.sha = {i: hashlib.sha256(f"span text {i}".encode()).hexdigest() for i in range(1, n + 1)}

    def h_any(self, name, batch):
        if name == "vectors_select":
            return Res([(i, MODEL, self.sha[i], self.mat[i - 1].tolist()) for i in batch[0]["ids"] if i <= len(self.mat)])
        return super().h_any(name, batch)


class CountCore:
    def __init__(self):
        self.builds, self.shape, self.dtype = 0, None, None

    def build(self, chunks, vecs, params, prior=None):
        self.builds += 1
        self.shape, self.dtype = vecs.shape, vecs.dtype
        return {"version": "v"}

    def quality(self, model, vecs):
        return {}

    def display_gate(self, metrics, gate):
        return True, []


def make(db, rss, core=None):
    cfg = {"enabled": True, "params": {}, "display_gate": {}, "embedder": "x",
           "job": {"page_size": 500, "tick_row_cap": 100000, "rss_soft_mb": 380, "rss_hard_mb": 420}}
    j = job.Job(cfg, core=core or CountCore(), embedder=FakeEmb(), engine_factory=factory_for(db), env=dict(ENV), rss_reader=rss)
    return j


def released(j):
    return j._chunks == [] and j._vecs == [] and j._pending is None


def test_peak_python_memory_5k5_x_1024_float32_blocks():
    db = MatrixDB(5500, 1024)  # fixture allocated before tracing starts (not job-held)
    core = CountCore()
    j = make(db, lambda: 100.0, core)
    tracemalloc.start()
    try:
        out = j.run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out["state"] == "compute_only" and core.builds == 1 and j.rows_embedded == 0  # pure reuse path
    assert core.shape == (5500, 1024) and core.dtype == np.float32
    assert peak <= 60 * 1024 * 1024, f"job-held peak {peak / 1048576:.1f} MB"
    assert released(j)


def test_guard_trips_on_reuse_path_before_any_work():
    db = MatrixDB(1200, 8)
    core = CountCore()
    j = make(db, lambda: 999.0, core)  # all vectors already stored; embedding never runs
    out = j.run()
    assert out["state"] == "partial" and out["stop"] == "rss_hard" and core.builds == 0
    assert released(j)


def test_guard_trips_mid_reuse_scan_and_releases_buffers():
    db = MatrixDB(1200, 8)
    core = CountCore()
    calls = {"n": 0}
    seen = {}

    def rss():
        calls["n"] += 1
        return 100.0 if calls["n"] <= 2 else 999.0  # trip on the 3rd page
    j = make(db, rss, core)
    orig = j._guard

    def spy():
        seen["held"] = len(j._chunks)
        return orig()
    j._guard = spy
    out = j.run()
    assert out["stop"] == "rss_hard" and core.builds == 0 and seen["held"] > 0
    assert released(j)


def test_guard_trips_before_build():
    db = MatrixDB(600, 8)
    core = CountCore()
    holder = {}
    holder["j"] = j = make(db, lambda: 999.0 if holder["j"]._phase == "build" else 100.0, core)
    out = j.run()
    assert out["stop"] == "rss_hard" and core.builds == 0
    assert released(j)


def test_buffers_released_after_complete_pass():
    db = MatrixDB(600, 8)
    j = make(db, lambda: 100.0)
    assert j.run()["state"] == "compute_only"
    assert released(j)
