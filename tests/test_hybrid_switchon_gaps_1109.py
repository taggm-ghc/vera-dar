"""HYBRID-TESTS-1109: switch-on coverage gaps (a)-(d) from work/HYBRID-1109/fix_review.md item 5. Hybrid forced ON through a fake
embedder; no network, no DB. Every thread is joined with a bound, so no test can hang the suite."""
from __future__ import annotations

import threading
import time

import pytest

from vera import chunk_taxo, chunk_taxo_embed as emb, fulldoc_admission as fa, hybrid_retrieval as hr
from vera.fulldoc_corpus import load_config
from tests.test_hybrid_retrieval_1109 import MODEL, Q, FakeEmbedder, Loader, _corpus, _role, _texts

NAME = "fakegaps"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    hr.reset_for_tests()
    emb.reset_budget_for_tests()
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder=NAME))
    yield
    hr.reset_for_tests()
    emb.unregister(NAME)
    emb.reset_budget_for_tests()


def _on(corpus, table, fake, sync=True, **cfg):
    emb.register(NAME, lambda c: fake)
    hr.set_config_for_tests(dict({"enabled": True}, **cfg))
    hr.set_loader_for_tests(Loader(corpus, table))
    if sync:
        idx, code = hr.ensure_index(corpus, MODEL, sync=True)
        assert code == "ok" and idx is not None


def _run(n, target, bound=10.0):
    """Run `target(i)` in n threads released together; fail (not hang) if any is still alive after `bound` s."""
    gate, out, errs = threading.Barrier(n), [None] * n, []

    def w(i):
        try:
            gate.wait(5)
            out[i] = target(i)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=w, args=(i,), daemon=True) for i in range(n)]
    for t in ts:
        t.start()
    end = time.monotonic() + bound
    for t in ts:
        t.join(max(0.0, end - time.monotonic()))
    assert not any(t.is_alive() for t in ts), "ask thread still running (hang)"
    assert not errs, errs
    return out


def _hybrid_alive():
    return [t for t in threading.enumerate() if t.name == "hybrid-embed" and t.is_alive()]


# ------------------------------------------------------------------------------ (a) second stored-span corpus
def test_a_second_stored_span_corpus_status_ok_and_vector_ranks_reorder():
    c, table = _corpus(spans_b=None)
    # a different partitioning from the first-corpus tests: doc B in four stored partitions (intro + three role paragraphs)
    b = _texts()[1]
    cuts = [0] + [b.index(k) for k in ("Below are", "Worker agents", "Supervisor agents")] + [len(b)]
    c2, table2 = _corpus(spans_b=list(zip(cuts[:-1], cuts[1:])))
    assert len(table2) == len(table) + 2 and c2 is not c
    off = c2.retrieve(Q, 16, True)
    off_ids = [x["span_id"] for x in off]
    fake = FakeEmbedder()
    emb.register(NAME, lambda c: fake)
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 3})
    hr.set_loader_for_tests(Loader(c2, table2))
    assert hr.ensure_index(c2, MODEL, sync=True)[1] == "ok"
    on = c2.retrieve(Q, 16, True)
    assert hr.last_status()["status"] == "ok" and hr.last_status()["vectors"] == len(table2)
    on_ids = [x["span_id"] for x in on]
    assert on_ids != off_ids  # vector ranks changed the result order
    assert any(x.get("vec_sim") is not None for x in on) and all("vec_sim" not in x for x in off)
    b0 = c2._texts["DB"].index("Below are")
    pos = lambda ids, cs: next((i for i, x in enumerate(cs) if _role(x, b0)), None)  # noqa: E731
    p_on, p_off = pos(on_ids, on), pos(off_ids, off)
    assert p_on is not None and (p_off is None or p_on < p_off)  # the 'categories' passage moves up (or in)
    # admission end to end on the second corpus
    adm = fa.admit(Q, c2, load_config())
    assert adm.trace["hybrid"]["status"] == "ok" and any(_role(a, b0) for a in adm.admitted)


# ------------------------------------------------------------------------------ (c) concurrent asks racing embedder creation
N = 8


def _race(factory_fail):
    c, table = _corpus()
    made, fake = [], FakeEmbedder()
    mu = threading.Lock()

    def factory(cfg):
        with mu:
            made.append(1)
        time.sleep(0.3)  # widen the creation window so every ask arrives inside it
        if factory_fail:
            raise RuntimeError("no creds")
        return fake
    emb.register(NAME, factory)
    hr.set_config_for_tests({"enabled": True, "cooldown_s": 60})
    hr.set_loader_for_tests(Loader(c, table))
    assert hr.ensure_index(c, MODEL, sync=True)[1] == "ok"  # the index build does not need the embedder
    res = _run(N, lambda i: (hr.vector_fn(c, Q + f" {i % 2}"), hr.last_status()["status"]))
    return made, res, fake


@pytest.mark.xfail(strict=True, reason="DEFECT hybrid_retrieval._embedder: check and create are in separate lock sections, so N "
                   "concurrent first asks each call load_embedder (N adapters, N credential loads), not one")
def test_c_concurrent_asks_success_create_exactly_one_embedder():
    made, res, fake = _race(False)
    assert len(res) == N and len(made) == 1


def test_c_concurrent_asks_success_all_return_and_embedder_usable():
    made, res, fake = _race(False)
    assert len(res) == N and 1 <= len(made) <= N
    assert all(st == "ok" or st.startswith("cooldown") for _, st in res), res
    assert hr._embedder()[0] is not None


@pytest.mark.xfail(strict=True, reason="DEFECT hybrid_retrieval._embedder: concurrent first asks all retry the failing creation "
                   "(N attempts) because cooldown is only recorded after creation ends")
def test_c_concurrent_asks_failure_create_exactly_one_attempt():
    made, res, _ = _race(True)
    assert len(res) == N and len(made) == 1


def test_c_concurrent_asks_failure_all_return_word_only_then_cooldown():
    made, res, _ = _race(True)
    assert len(res) == N and all(r[0][0] is None and r[1].startswith("embedder:") for r in res), res
    n = len(made)
    hr.vector_fn(_corpus()[0], Q)  # within cooldown: no further creation attempt
    assert len(made) == n


# ------------------------------------------------------------------------------ (d) orphan embed threads after repeated timeouts
def test_d_repeated_embed_timeouts_orphan_threads_bounded_by_cooldown():
    c, table = _corpus()
    release = threading.Event()

    class Hang(FakeEmbedder):
        def embed(self, texts):
            self.calls += 1
            release.wait(15)
            return super().embed(texts)
    fake = Hang()
    _on(c, table, fake, timeout_s=0.2, cooldown_s=60)
    base = len(_hybrid_alive())
    K = 6
    t0 = time.monotonic()
    try:
        for i in range(K):
            fn, _ = hr.vector_fn(c, Q + f" variant {i}")
            assert fn is None  # word-only
            assert hr.last_status()["status"].split(":")[0] in ("embed_error", "cooldown")
        assert time.monotonic() - t0 < 5.0
        assert fake.calls == 1 and len(_hybrid_alive()) - base <= 1  # cooldown honoured: one orphan, not K
    finally:
        release.set()
        for t in _hybrid_alive():
            t.join(5)
    assert len(_hybrid_alive()) == base  # orphans end with the embedder call, not leaked


def test_d_timeouts_after_cooldown_expiry_leave_at_most_one_orphan_per_window():
    c, table = _corpus()
    release = threading.Event()

    class Hang(FakeEmbedder):
        def embed(self, texts):
            self.calls += 1
            release.wait(15)
            return []
    fake = Hang()
    _on(c, table, fake, timeout_s=0.2, cooldown_s=60)
    t = [1000.0]
    hr.set_clock_for_tests(clock=lambda: t[0])
    base = len(_hybrid_alive())
    try:
        for w in range(3):  # three cooldown windows, three asks inside each
            for i in range(3):
                hr.vector_fn(c, Q + f" w{w} i{i}")
            t[0] += 61
        assert fake.calls == 3 and len(_hybrid_alive()) - base <= 3
    finally:
        release.set()
        for x in _hybrid_alive():
            x.join(5)


# ------------------------------------------------------------------------------ (b) threaded index build during asks
def test_b_threaded_index_build_asks_served_word_only_then_vectors():
    c, table = _corpus()
    off = [x["span_id"] for x in c.retrieve(Q, 16, True)]
    gate, started = threading.Event(), threading.Event()

    class Slow(Loader):
        def vectors(self, ids, model):
            started.set()
            gate.wait(8)
            return super().vectors(ids, model)
    fake = FakeEmbedder()
    emb.register(NAME, lambda cfg: fake)
    hr.set_config_for_tests({"enabled": True})
    hr.set_loader_for_tests(Slow(c, table))  # ensure_index NOT forced synchronous: the production thread path
    try:
        t0 = time.monotonic()
        during = c.retrieve(Q, 16, True)
        assert time.monotonic() - t0 < 2.0 and hr.last_status()["status"] == "not_ready"
        assert [x["span_id"] for x in during] == off and all("vec_sim" not in x for x in during)
        assert started.wait(5)
        again = fa.admit(Q, c, load_config())  # still building: still served, word-only
        assert again.trace["hybrid"]["status"] == "not_ready" and fake.calls == 0
    finally:
        gate.set()
    for t in [t for t in threading.enumerate() if t.name == "vera-hybrid-index"]:
        t.join(8)
    hs = c.__dict__["_hybrid_state"]
    assert hs["index"] is not None and hs["building"] is False
    after = c.retrieve(Q, 16, True)
    assert hr.last_status()["status"] == "ok" and any(x.get("vec_sim") is not None for x in after)
