"""LIVE-DIAG-1021: the acquisition job's served-key lookup must never wait on the base corpus load."""
from types import SimpleNamespace

from vera import auto_acquire


class _Dar:
    def __init__(self, keys):
        self._k = keys

    def doc_keys(self):
        return set(self._k)


def _fc(base_calls):
    def get_corpus():
        base_calls.append(1)
        return SimpleNamespace(_dar=_Dar({"doi:base"}))
    return SimpleNamespace(get_corpus=get_corpus, get_acquired_corpus=lambda: SimpleNamespace(_dar=_Dar({"doi:acq"})))


def test_skips_base_corpus_while_warming():
    calls = []
    assert auto_acquire.served_doc_keys(_fc(calls), warm_state=lambda: "warming") == {"doi:acq"}
    assert calls == []  # get_corpus (a 60 s single-flight follower wait while loading) is never called


def test_skips_base_when_warm_failed_or_idle():
    for st in ("failed", "idle"):
        calls = []
        assert auto_acquire.served_doc_keys(_fc(calls), warm_state=lambda st=st: st) == {"doi:acq"}
        assert calls == []


def test_reads_base_when_warm_done():
    calls = []
    assert auto_acquire.served_doc_keys(_fc(calls), warm_state=lambda: "done") == {"doi:base", "doi:acq"}
    assert calls == [1]


def test_state_error_fails_to_skip_base_and_none_corpora_ok():
    def boom():
        raise RuntimeError("x")
    fc = SimpleNamespace(get_corpus=lambda: None, get_acquired_corpus=lambda: None)
    assert auto_acquire.served_doc_keys(fc, warm_state=boom) == set()
    assert auto_acquire.served_doc_keys(fc, warm_state=lambda: "done") == set()


def test_default_reads_live_warm_state(monkeypatch):
    from vera import fulldoc_warm
    monkeypatch.setattr(fulldoc_warm, "state", lambda: fulldoc_warm.WARMING)
    calls = []
    assert auto_acquire.served_doc_keys(_fc(calls)) == {"doi:acq"} and calls == []
