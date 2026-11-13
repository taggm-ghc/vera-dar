"""REL17-1113 (verifier D-1): W4 cleanup of positively excluded spans also retires their tax_chunk_domain rows, only when the table exists.
Offline: a recording connection; nothing connects to a database."""
from __future__ import annotations

from tests.test_chunk_domain_wpd_1113 import RecConn
from vera import chunk_domain_writer as dw
from vera import chunk_taxo_store as store


def test_statement_is_allowlisted_and_scoped():
    sql = store.SQL["link_retire_spans_chunk_domain"]
    assert "tax_chunk_domain" in sql and "classifier_name = 'chunk_taxo_auto'" in sql and "state = 'active'" in sql
    assert "ANY(:ids)" in sql and "DELETE" not in sql.upper()


class RetConn(RecConn):
    def execute(self, stmt, params=None):
        r = super().execute(stmt, params)
        if self.log[-1][0] == "link_retire_spans_chunk_domain":
            r.rowcount = self.retire_count
        return r


def test_retire_spans_table_present():
    c, st = RetConn(True, retire_count=2), dw.new_state()
    dw.retire_spans(c, st, [1, 2])
    assert c.names() == ["probe_chunk_domain", "link_retire_spans_chunk_domain"]
    assert c.log[1][1] == {"ids": [1, 2]} and st.retired == 2


def test_retire_spans_table_absent_or_disabled_or_empty():
    c = RecConn(False)
    dw.retire_spans(c, dw.new_state(), [1])
    assert c.names() == ["probe_chunk_domain"]
    c = RecConn(True)
    dw.retire_spans(c, dw.new_state(False), [1])
    dw.retire_spans(c, dw.new_state(), [])
    dw.retire_spans(c, None, [1])
    assert c.names() == []
