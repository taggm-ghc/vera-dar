"""HYBRID-HARDEN-1109: per-ask total lock-wait budget, locked _INFLIGHT cleanup on a recording timeout, honest docstrings.
No network, no DB. Every thread is joined with a bound; held locks are released in a finally."""
from __future__ import annotations

import threading
import time

import pytest

from vera import chunk_taxo, chunk_taxo_embed as emb, hybrid_retrieval as hr
from tests.test_hybrid_retrieval_1109 import MODEL, Q, FakeEmbedder, Loader, _corpus

NAME = "fakeharden"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    hr.reset_for_tests()
    hr.set_config_for_tests({"enabled": False})
    emb.reset_budget_for_tests()
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder=NAME))
    yield
    hr.reset_for_tests()
    emb.unregister(NAME)
    emb.reset_budget_for_tests()


def _hold(lock, seconds):
    held, release = threading.Event(), threading.Event()

    def holder():
        with lock:
            held.set()
            release.wait(seconds)
    threading.Thread(target=holder, daemon=True).start()
    assert held.wait(5)
    return release


def _in_thread(fn, bound):
    box = {}

    def w():
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e
    t = threading.Thread(target=w, daemon=True)
    t.start()
    t.join(bound)
    return (not t.is_alive()), box


def test_budget_bounds_total_lock_wait_not_each_acquisition():
    hr._begin_ask_budget(0.4)
    release = _hold(hr._LOCK, 10)
    try:
        def three():
            n = 0
            for _ in range(3):
                try:
                    with hr._locked():  # each would wait LOCK_TIMEOUT_S (2 s) without the budget
                        pass
                except TimeoutError:
                    n += 1
            return n
        t0 = time.perf_counter()
        done, box = _in_thread(lambda: (hr._begin_ask_budget(0.4), three())[1], 10)
        took = time.perf_counter() - t0
    finally:
        release.set()
    assert done and box.get("out") == 3 and took < 0.4 + 0.5, took


def test_lock_held_10s_vector_fn_is_word_only_within_budget(monkeypatch):
    c, table = _corpus()
    fake = FakeEmbedder()
    emb.register(NAME, lambda cfg: fake)
    hr.set_config_for_tests({"enabled": True, "lock_budget_s": 0.5})
    hr.set_loader_for_tests(Loader(c, table))
    assert hr.LOCK_TIMEOUT_S >= 2.0
    release = _hold(hr._LOCK, 10)
    try:
        t0 = time.perf_counter()
        done, box = _in_thread(lambda: (hr.vector_fn(c, Q), hr.last_status()), 10)
        took = time.perf_counter() - t0
    finally:
        release.set()
    assert done and "exc" not in box
    (fn, sim_of), st = box["out"]
    assert fn is None and sim_of is None and st["vectors"] == 0
    assert took < 0.5 + 1.0, took  # before: 3 x 2 s per vector_fn call



def test_lock_held_10s_whole_retrieve_is_word_only_within_default_budget(monkeypatch):
    c, table = _corpus()
    lex = [x["span_id"] for x in c.retrieve(Q, 16, True)]  # hybrid off (fixture)
    emb.register(NAME, lambda cfg: FakeEmbedder())
    hr.reset_for_tests()
    hr.set_config_for_tests({"enabled": True})  # default lock_budget_s (3.0)
    hr.set_loader_for_tests(Loader(c, table))
    release = _hold(hr._LOCK, 10)
    try:
        t0 = time.perf_counter()
        done, box = _in_thread(lambda: [x["span_id"] for x in c.retrieve(Q, 16, True)], 15)
        took = time.perf_counter() - t0
    finally:
        release.set()
    assert done and box.get("out") == lex and took < hr.LOCK_BUDGET_S + 1.0, took


def test_config_key_lock_budget_s_default_and_bounds():
    assert hr.config()["lock_budget_s"] == 3.0
    hr.set_config_for_tests({"enabled": True, "lock_budget_s": 1.5})
    assert hr.config()["lock_budget_s"] == 1.5
    for bad in (0, -1, 31, "x", None, float("nan"), True):
        hr.set_config_for_tests({"enabled": True, "lock_budget_s": bad})
        assert hr.config()["lock_budget_s"] == 3.0, bad


def _slow_recording(monkeypatch, hold_s, swap=None):
    """A creator whose result-recording lock wait times out (LOCK_TIMEOUT_S 0.2) because another thread holds _LOCK."""
    monkeypatch.setattr(hr, "LOCK_TIMEOUT_S", 0.2)
    state = {"release": threading.Event()}

    def factory(cfg):
        held = threading.Event()

        def holder():
            with hr._LOCK:
                held.set()
                if swap is not None:
                    hr._INFLIGHT["emb"] = swap  # a takeover installs its own flight while the creator is stuck
                state["release"].wait(hold_s)
        threading.Thread(target=holder, daemon=True).start()
        assert held.wait(5)
        return FakeEmbedder()
    emb.register(NAME, factory)
    hr.set_config_for_tests({"enabled": True})
    return state


def test_inflight_cleanup_after_recording_timeout_is_done_under_the_lock(monkeypatch):
    st = _slow_recording(monkeypatch, 0.35)
    done, box = _in_thread(lambda: hr._embedder(), 10)
    st["release"].set()
    assert done and "exc" not in box
    assert hr._INFLIGHT == {}  # popped under the lock once the holder let go (0.5 s bounded window)


def test_inflight_cleanup_never_removes_a_takeovers_flight(monkeypatch):
    takeover = (threading.Event(), time.monotonic())
    st = _slow_recording(monkeypatch, 0.35, swap=takeover)
    done, box = _in_thread(lambda: hr._embedder(), 10)
    st["release"].set()
    assert done and "exc" not in box
    assert hr._INFLIGHT.get("emb") is takeover  # ownership re-checked under the lock: not ours, not popped


def test_inflight_cleanup_wait_is_bounded_and_leaves_entry_to_stale_rule(monkeypatch):
    st = _slow_recording(monkeypatch, 10)
    t0 = time.perf_counter()
    try:
        done, box = _in_thread(lambda: hr._embedder(), 10)
        took = time.perf_counter() - t0
        assert done and "exc" not in box and took < 0.2 + 0.5 + 1.0, took
        assert "emb" in hr._INFLIGHT  # left for the INFLIGHT_STALE_S rule; waiters time out meanwhile
    finally:
        st["release"].set()  # release the holder at once so the next test's reset is not stalled
        time.sleep(0.05)


def test_h3_successful_embedder_is_cached_when_recording_was_busy(monkeypatch):
    """Recording and the 0.5 s fallback both fail (lock busy); the success is recorded when the ask ends."""
    monkeypatch.setattr(hr, "LOCK_TIMEOUT_S", 0.1)
    st = _slow_recording(monkeypatch, 10)
    monkeypatch.setattr(hr, "LOCK_TIMEOUT_S", 0.1)

    def go():
        hr._begin_ask_budget(0.1)
        e, _ = hr._embedder()
        busy = hr._STATE["emb"] is None and hr._TL.pending_emb is not None
        st["release"].set()
        time.sleep(0.2)  # holder lets go
        hr.end_ask_budget()
        return e, busy, hr._STATE["emb"], hr._wait_left()
    done, box = _in_thread(go, 15)
    st["release"].set()
    assert done and "exc" not in box, box
    e, busy, cached, left = box["out"]
    assert e is not None and busy and cached is e and left is None
    assert hr._INFLIGHT == {}
    assert hr._embedder()[0] is e  # the next ask does not recreate it


def test_h1_retrieve_clears_the_budget_so_later_calls_do_not_inherit_it():
    c, table = _corpus()
    emb.register(NAME, lambda cfg: FakeEmbedder())
    hr.set_config_for_tests({"enabled": True, "lock_budget_s": 0.3})
    hr.set_loader_for_tests(Loader(c, table))
    release = _hold(hr._LOCK, 10)

    def go():
        c.retrieve(Q, 16, True)  # exhausts the budget on the held lock
        return hr._wait_left()
    try:
        done, box = _in_thread(go, 15)
    finally:
        release.set()
    assert done and box.get("out") is None, box


def test_docstrings_do_not_claim_never_raises_and_behaviour_matches():
    assert "never raises" not in (hr.query_vector.__doc__ or "").lower()
    assert "never raises" not in (hr.passage_vectors.__doc__ or "").lower()
    assert "TimeoutError" in hr.query_vector.__doc__ and "TimeoutError" in hr.passage_vectors.__doc__
    release = _hold(hr._LOCK, 10)
    try:
        def go():
            hr._begin_ask_budget(0.2)
            return hr.query_vector(Q)
        done, box = _in_thread(go, 10)
    finally:
        release.set()
    assert done and isinstance(box.get("exc"), TimeoutError)
