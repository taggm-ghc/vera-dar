"""TEST-HARDEN-1113 (P-1110-10): regression guard for the env_guard -> fulldoc_e2e order dependence.

Cause (found by bisect: reproduced at d3fe0fe, fixed by a887314 TEST-ISO-1110): a boundary probe in tests/test_env_guard.py
put a sentinel DB URL in the environment and ran /ask, which left the request_trace writer (a process singleton with a
daemon thread that caches its DB engine) alive; every later test's trace flush then hit the DB boundary ("test boundary
violation ... DB access attempt(s) refused"), e.g. the 11 tests of tests/test_fulldoc_e2e.py. The fix is the autouse
fixture _request_trace_writer_per_test in tests/conftest.py. pytest runs the tests of a file in definition order, so the
second test sees whatever the first one left behind. No DB, no network, no thread: the writer is a fake with stop().
"""
from vera import request_trace as rt


class _FakeWriter:
    stopped = []

    def stop(self):
        _FakeWriter.stopped.append(self)


def test_a_leaves_a_writer_behind():
    _FakeWriter.stopped.clear()
    rt._WRITER = _FakeWriter()  # as a probe test that ran /ask under a sentinel URL would leave it
    assert rt._WRITER is not None


def test_b_next_test_starts_without_the_previous_writer():
    assert rt._WRITER is None
    assert len(_FakeWriter.stopped) == 1  # stopped (not just forgotten) at the edge of the previous test
