"""HYBRID-HANG-1109: tests/test_accept_d_1023.py re-run with hybrid retrieval ON (fake embedder, stage 1 + stage 2; no network, no DB).
On 0bf22bb (flags on, no embedder credentials) such runs hung in the sandbox: _embedder deadlock, fixed in vera/hybrid_retrieval.py."""
from tests.hybrid_on_support import _hybrid_still_on, hybrid_on, rerun_module  # noqa: F401  (module-scoped autouse: hybrid ON)

rerun_module(globals(), "test_accept_d_1023")


def test_zz_hybrid_was_exercised(hybrid_on):  # noqa: F811
    from vera import hybrid_retrieval as hr
    assert hr.enabled() and hybrid_on.vector_fn_calls > 0, hybrid_on.statuses
    assert hybrid_on.statuses.get("ok", 0) > 0 and hybrid_on.fake.calls > 0, hybrid_on.statuses  # vectors took part
