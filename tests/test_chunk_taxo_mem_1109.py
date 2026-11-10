"""TAXO-MEM-1109 / TAXO-MEM2-1109: the chunk-taxonomy job holds vectors as float32 blocks (never Python float lists), applies the RSS guard on the
reuse path, before the build and per batch, and releases every buffer when a pass stops early or ends. Offline: in-memory fake
DB (vectors come back as plain float lists per page, like a driver), no network. The guard tests use a stub core; the memory
test runs the REAL core (clustering, c-TF-IDF, nearest-center, quality) on 5,500 x 1024 vectors with ~3000-char texts."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import tracemalloc

import numpy as np

from vera import chunk_taxo as real_core
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


def _rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576


class RealTextDB(FakeDB):
    """5,500 spans with ~3000-char texts and clustered 1024-d stored vectors (so the whole pass is the reuse path)."""

    def __init__(self, n=5500, dim=1024, chars=3000, k=40):
        rng = np.random.default_rng(7)
        vocab = ["w%s%05d" % (chr(97 + i % 26), i) for i in range(20000)]
        lab = rng.integers(0, k, n)
        cent = rng.normal(size=(k, dim)).astype(np.float32)
        self.mat = (cent[lab] + 1.2 * rng.normal(size=(n, dim)).astype(np.float32)).astype(np.float32)
        zipf = np.minimum(rng.zipf(1.3, size=n * (chars // 6) + 10), 20000) - 1
        texts, p = [], 0
        for i in range(n):
            nw = chars // 7
            texts.append(" ".join(vocab[(int(z) + 97 * int(lab[i])) % 20000] for z in zipf[p:p + nw])[:chars])
            p += nw
        super().__init__([(i + 1, 100 + i % 50, texts[i]) for i in range(n)], has=(True, False, True))
        self.sha = {i + 1: hashlib.sha256(texts[i].encode()).hexdigest() for i in range(n)}
        self.text_bytes = sum(len(t) for t in texts)

    def h_any(self, name, batch):
        if name == "vectors_select":
            return Res([(i, MODEL, self.sha[i], self.mat[i - 1].tolist()) for i in batch[0]["ids"] if i <= len(self.mat)])
        return super().h_any(name, batch)


def test_real_core_peak_5k5_x_1024_3000_chars(record_property):
    """REAL clustering / build / assign / quality via the real Job: tracemalloc growth <= 80 MB and the pre-build estimate
    covers the measured growth. RSS growth is sampled and recorded (printed with -s, kept as a junit property)."""
    n, dim = 5500, 1024
    db = RealTextDB(n, dim)  # fixture allocated before tracing starts (not job-held)
    cfg = {"enabled": True, "embedder": "x", "display_gate": {},
           "params": {"seeds": [11, 23], "k_max": 16, "null_seeds": [101], "kmeans_max_iter": 20},
           "job": {"page_size": 500, "tick_row_cap": 100000, "rss_soft_mb": 380, "rss_hard_mb": 100000}}
    full = real_core.load_cfg()
    cfg["params"] = dict(full["params"], **cfg["params"])
    cfg["display_gate"] = full["display_gate"]
    j = job.Job(cfg, core=real_core, embedder=FakeEmb(), engine_factory=factory_for(db), env=dict(ENV), rss_reader=_rss_mb)
    stop, peak_rss, base_rss = threading.Event(), [_rss_mb()], _rss_mb()

    def sample():
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], _rss_mb())
            time.sleep(0.01)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    tracemalloc.start()
    try:
        out = j.run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        stop.set()
        th.join()
    grow_mb, rss_mb = peak / 1048576, peak_rss[0] - base_rss
    est_mb = real_core.estimate_build_bytes(n, dim, 16, db.text_bytes) / 1048576
    record_property("tracemalloc_peak_mb", round(grow_mb, 1))
    record_property("rss_growth_mb", round(rss_mb, 1))
    record_property("estimate_mb", round(est_mb, 1))
    print("TAXO-MEM2 real core: tracemalloc peak %.1f MB, RSS growth %.1f MB, build estimate %.1f MB" % (grow_mb, rss_mb, est_mb))
    assert out["state"] == "compute_only" and j.rows_embedded == 0  # pure reuse path
    assert j._pending is None and released(j)
    assert real_core.has_numpy()
    assert grow_mb <= 80, f"job peak {grow_mb:.1f} MB"
    # the estimate is for the build's EXTRA memory; the job-held matrix (22 MB) and texts are already resident before it
    held_mb = n * dim * 4 / 1048576 + db.text_bytes / 1048576
    assert est_mb >= grow_mb - held_mb, f"estimate {est_mb:.1f} MB < measured extra {grow_mb - held_mb:.1f} MB"
    assert j.build_estimate_mb == round(real_core.estimate_build_bytes(n, dim, 16, db.text_bytes) / 1048576, 1)


def test_headroom_check_stops_before_build_when_estimate_does_not_fit():
    db = MatrixDB(600, 64)
    core = CountCore()
    est_mb = real_core.estimate_build_bytes(600, 64, 40, sum(len(f"span text {i}") for i in range(1, 601))) / 1048576
    j = make(db, lambda: 420.0 - est_mb + 1.0, core)  # below the 420 MB hard limit by itself, over it with the estimate
    out = j.run()
    assert out["state"] == "partial" and out["stop"] == "rss_build" and core.builds == 0
    assert j.build_estimate_mb > 0
    assert released(j)
    j2 = make(MatrixDB(600, 64), lambda: 420.0 - est_mb - 1.0, CountCore())  # fits: the build runs
    assert j2.run()["state"] == "compute_only"


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
