"""REL18-1113 (REL17 verifier T-1): the job-level W4 step `Job._retire_excluded` retires the tax_chunk_domain rows of the
positively excluded spans, after the two chunk_* retire statements, only when the table exists. Offline: a recording connection."""
from __future__ import annotations

from tests.test_chunk_domain_retire_rel17_1113 import RetConn
from vera import chunk_domain_writer as dw
from vera import chunk_taxo_job as job


class JobConn(RetConn):
    """RetConn whose strict-link selects return no rows (fetchall)."""

    def execute(self, stmt, params=None):
        r = super().execute(stmt, params)
        r.fetchall = lambda: []
        return r


def _job(dom, page=2):
    j = job.Job.__new__(job.Job)  # no __init__: only the attributes _retire_excluded reads
    j.page, j._dom = page, dom
    j._excl = {"portal": [5, 3], "empty": [9]}
    return j


def test_retire_excluded_retires_domain_rows_of_excluded_spans():
    c, st = JobConn(True, retire_count=3), dw.new_state()
    _job(st)._retire_excluded(c)
    names = c.names()
    assert names.count("link_retire_spans_chunk_subtopic") == 2 and names.count("link_retire_spans_chunk_topic") == 2  # 3 ids, page 2
    dom = [p for n, p in c.log if n == "link_retire_spans_chunk_domain"]
    assert dom == [{"ids": [3, 5]}, {"ids": [9]}]
    assert names.index("link_retire_spans_chunk_domain") > max(i for i, n in enumerate(names) if n == "link_retire_spans_chunk_topic")
    assert st.retired == 6


def test_retire_excluded_no_domain_statement_when_table_absent_or_disabled():
    c = JobConn(False)
    _job(dw.new_state())._retire_excluded(c)
    assert "link_retire_spans_chunk_domain" not in c.names() and "link_retire_spans_chunk_topic" in c.names()
    c = JobConn(True)
    _job(dw.new_state(False))._retire_excluded(c)
    assert "link_retire_spans_chunk_domain" not in c.names() and "link_retire_spans_chunk_topic" in c.names()
