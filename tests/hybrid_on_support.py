"""HYBRID-HANG-1109: run existing test modules with hybrid retrieval ON (stage 1 + stage 2) through a fake embedder and an
in-memory vector loader bound to each served corpus (no network, no DB). Used by tests/test_hybrid_on_*_1109.py, which
copy a module's tests and fixtures (rerun_module) and add `hybrid_on` as a module-scoped autouse fixture."""
from __future__ import annotations

import hashlib
import importlib

import pytest

from vera import chunk_taxo, chunk_taxo_embed as emb, hybrid_retrieval as hr

NAME = "fakehybrid-on"


def rerun_module(namespace: dict, modname: str) -> None:
    """Copy every test, fixture and helper of tests.<modname> (underscore-named autouse fixtures included) into the
    calling module's namespace, so pytest collects them again there, under hybrid_on."""
    mod = importlib.import_module("tests." + modname)
    for k, v in vars(mod).items():
        if not (k.startswith("__") and k.endswith("__")):
            namespace.setdefault(k, v)


class Counter:
    def __init__(self, fake):
        self.fake, self.vector_fn_calls, self.statuses = fake, 0, {}


class CorpusLoader:
    """Vectors for exactly the served partitions of one corpus (sha256 of the served text, so none is stale)."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.calls = 0

    def vectors(self, ids, model):
        from tests.test_hybrid_retrieval_1109 import fake_vec
        self.calls += 1
        loc = {sid: (did, s, e) for sid, did, s, e in hr._spans_of(self.corpus)}
        n = hr._text_chars()
        for i in ids:
            if i in loc:
                did, s, e = loc[i]
                txt = self.corpus._texts[did][s:e][:n]
                yield i, hashlib.sha256(txt.encode("utf-8")).hexdigest(), fake_vec(txt)

    def neurons_today(self):
        return 0.0


class _NullLoader:
    def vectors(self, ids, model):
        return iter(())

    def neurons_today(self):
        return 0.0


def _install():
    from tests.test_hybrid_retrieval_1109 import FakeEmbedder
    fake = FakeEmbedder()
    emb.register(NAME, lambda c: fake)
    tc = chunk_taxo.load_cfg()
    saved = {k: getattr(hr, k) for k in ("_taxo_cfg", "build_index", "ensure_index", "vector_fn")}
    orig_build, orig_ensure, orig_vfn = hr.build_index, hr.ensure_index, hr.vector_fn
    counter = Counter(fake)

    def vfn(corpus, question):
        counter.vector_fn_calls += 1
        out = orig_vfn(corpus, question)
        st = (hr.last_status() or {}).get("status")
        counter.statuses[st] = counter.statuses.get(st, 0) + 1
        return out
    hr.vector_fn = vfn
    hr.reset_for_tests()
    hr._taxo_cfg = lambda: dict(tc, embedder=NAME)
    hr.build_index = lambda corpus, model, loader=None: orig_build(corpus, model, CorpusLoader(corpus))
    # Tests are synchronous: build in the caller so the vectors take part (production builds in a thread, never on the
    # request path; tests/test_hybrid_retrieval_1109.py::test_no_request_path_index_build covers that).
    hr.ensure_index = lambda corpus, model, sync=False: orig_ensure(corpus, model, sync=True)
    hr.set_loader_for_tests(_NullLoader())
    hr.set_config_for_tests({"enabled": True, "rerank": {"enabled": True}})
    return counter, saved


def _uninstall(saved):
    for k, v in saved.items():
        setattr(hr, k, v)
    hr.reset_for_tests()
    emb.unregister(NAME)


@pytest.fixture(scope="module", autouse=True)
def hybrid_on():
    counter, saved = _install()
    try:
        yield counter
    finally:
        _uninstall(saved)


@pytest.fixture(autouse=True)
def _hybrid_still_on(hybrid_on):
    assert hr.enabled() and hr.config()["rerank"]["enabled"] is True
    yield
