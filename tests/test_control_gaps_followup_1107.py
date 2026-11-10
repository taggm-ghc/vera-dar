"""CONTROL-GAPS-1107 follow-up: (a) the arXiv cold-tdmrep window at its edge (kills the race-fix revert and a zero
deadline_extension), (b) ask_service.inline_need_s includes that window, (c) the S5 missing-licence count runs once per
process. Fakes only; no network, no DB."""
from __future__ import annotations

import logging
from concurrent.futures import Future
from datetime import datetime, timezone

import pytest

from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc  # noqa: F401  (_direct: autouse)
from tests.test_p0dta_arxiv_ids import ROBOTS, _crawl_run, clk  # noqa: F401  (clk: fixture)
from vera import acq_discovered, arxiv_fulltext, ask_service, inline_acquire
from vera import memory_store as ms

# ARXIV-PORTAL-1110: these tests pin the DORMANT arXiv evidence route (closed in code: arxiv_fulltext.PORTAL_ONLY);
# the fixture reopens it per test. Portal-only behaviour: tests/test_arxiv_portal_1110.py.
pytestmark = pytest.mark.usefixtures("legacy_arxiv_route")


class _SyncPool:
    """Runs each job AT SUBMIT (deterministic): by the time submit() returns, a worker has already cached arxiv.org's
    tdmrep.json, which is exactly the state a late-computed collection window would see."""
    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *a, **k):
        f = Future()
        f.set_result(fn(*a, **k))
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        pass


# ---------------------------------------------------------------------------------------------------- (a)
def test_cold_tdmrep_arxiv_doc_finishing_inside_the_extra_window_is_admitted(clk, monkeypatch):
    """Clock 1000, Crawl-delay 15: tdmrep.json 1015, abs 1030, html 1045. Inline deadline 40 s -> 1040: the document
    finishes AFTER the inline deadline but inside deadline + one Crawl-delay (1055). Admitted only when (1) the
    extension exists (deadline_extension > 0) and (2) the collection window was decided before any fetch started."""
    monkeypatch.setattr(inline_acquire, "ThreadPoolExecutor", _SyncPool)
    o, web, _ = _crawl_run(clk, inline_s=40)
    assert [round(w) for w in clk.waits] == [15, 15, 15]
    assert disc(o)["outcome"] == "admitted", o.public()
    assert X.HTML_URL in [c[0] for c in web.calls]


def test_deadline_extension_is_one_crawl_delay_only_while_cold_and_only_with_the_route():
    acq = cfg_on()
    assert arxiv_fulltext.deadline_extension(acq) == float(acq["arxiv_route"]["max_crawl_delay_s"]) == 15.0
    acq_discovered._TDMREP["arxiv.org"] = (0.0, [])
    assert arxiv_fulltext.deadline_extension(acq) == 0.0
    acq_discovered._TDMREP.clear()
    assert arxiv_fulltext.deadline_extension({"arxiv_route": {"enabled": False}}) == 0.0


# ---------------------------------------------------------------------------------------------------- (b)
def test_inline_need_s_includes_the_cold_tdmrep_window():
    acq = cfg_on()
    cold = ask_service.inline_need_s(acq, None)
    acq_discovered._TDMREP["arxiv.org"] = (0.0, [])
    warm = ask_service.inline_need_s(acq, None)
    assert cold is not None and warm is not None and cold - warm == pytest.approx(15.0)
    off = dict(acq, arxiv_route={"enabled": False})
    acq_discovered._TDMREP.clear()
    assert ask_service.inline_need_s(off, None) == pytest.approx(warm)


def test_inline_need_s_fails_closed_when_the_extension_is_unreadable(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "deadline_extension", lambda acq: 1 / 0)
    assert ask_service.inline_need_s(cfg_on(), None) is None  # the caller skips the retry


# ---------------------------------------------------------------------------------------------------- (c)
CFGM = {"query_stopwords": [], "min_query_word_len": 3, "max_query_terms": 8, "question_text": "qt", "status_live": "live",
        "ttl_days": 30, "min_matched_terms": 1, "min_rank": 0.0, "half_life_days": 30, "recall_k": 3}


def _reader(sqls, count_fails=False):
    r = ms.RoMemoryReader(object(), CFGM)

    def rows(sql, p):
        sqls.append(sql)
        if "count(DISTINCT" in sql:
            if count_fails:
                raise RuntimeError("db")
            return [(4,)]
        return [("c", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "u")]
    r._rows = rows
    return r


def test_missing_licence_count_runs_once_per_process(caplog):
    sqls = []
    r = _reader(sqls)
    with caplog.at_level(logging.INFO, logger="vera"):
        for _ in range(3):
            assert r.recall("quantized inference latency")
    assert sum("count(DISTINCT" in s for s in sqls) == 1
    assert sum("licence_decision missing" in m.getMessage() for m in caplog.records) == 1
    assert sum("count(DISTINCT" in s for s in sqls) == 1 and _reader([]).recall("quantized inference latency")


def test_failed_count_is_retried_on_the_next_recall():
    sqls = []
    r = _reader(sqls, count_fails=True)
    r.recall("quantized inference latency")
    r.recall("quantized inference latency")
    assert sum("count(DISTINCT" in s for s in sqls) == 2  # never cached as done after a failure
