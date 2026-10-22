"""DAR-INLINE-1021-R2: make_search retries once with the plain OR query when the AND profile returns nothing."""
from vera import auto_acquire
from vera import search_providers as sp


class _P:
    name = "openalex"


def _run(profile, results):
    seen = []

    def search_fn(question, n, **kw):
        seen.append(sp.OPENALEX_PROFILE.get())
        return results.pop(0)

    cfg = {"search": {"providers": ["openalex"], "openalex_profile": profile}}
    out = auto_acquire.make_search(cfg, chain_fn=lambda: [_P()], search_fn=search_fn)("q", 5, 10)
    return out, seen


def test_and_empty_retries_once_with_or():
    out, seen = _run("acq", [[], ["r"]])
    assert out == ["r"] and seen == ["acq", None]
    assert sp.OPENALEX_PROFILE.get() is None


def test_and_nonempty_no_retry():
    out, seen = _run("acq", [["a"], ["x"]])
    assert out == ["a"] and seen == ["acq"]
    assert sp.OPENALEX_PROFILE.get() is None


def test_profile_none_no_retry():
    out, seen = _run(None, [[], ["x"]])
    assert out == [] and seen == [None]
    assert sp.OPENALEX_PROFILE.get() is None


def test_reset_on_exception():
    def boom(*a, **k):
        raise RuntimeError("x")
    cfg = {"search": {"providers": ["openalex"], "openalex_profile": "acq"}}
    try:
        auto_acquire.make_search(cfg, chain_fn=lambda: [_P()], search_fn=boom)("q", 5, 10)
    except RuntimeError:
        pass
    assert sp.OPENALEX_PROFILE.get() is None
