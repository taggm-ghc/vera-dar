"""D2-FIX-1113 (REL17 verifier D-2): the WP-I incremental path scores its chunk_domain rows and reports the domain metrics.
Offline, WP-I fixtures. Before the fix incremental rows had score and score_kind NULL and the run row had no domain metrics."""
from __future__ import annotations

from tests.test_chunk_taxo_wpi_1113 import AssocCore, VEmb, add_spans, build_db, last_close
from tests.test_chunk_taxo_wpw_1113 import mk
from vera import chunk_domain_writer as dw


class DomCore(AssocCore):
    """AssocCore whose associate also carries the domain key (the real core always does for a strict decision)."""

    def associate(self, model, vec):
        out = super().associate(model, vec)
        if out.get("level") in ("subtopic", "topic"):
            out["domain"] = next(n["key"] for n in model.nodes if n["level"] == "domain")
        return out


def _inc_db():
    core = DomCore()
    db = build_db(core=core)  # table absent: the build writes no domain rows
    db.domain_table = True
    add_spans(db, [13, 14, 15, 16])
    return core, db


def test_d2_incremental_domain_rows_carry_a_score():
    core, db = _inc_db()
    j = mk(db, e=VEmb(), core=core, incremental="on")
    assert j.run()["complete"] and j.status()["state"] == "incremental"
    rows = {k[1]: v for k, v in db.links.items() if k[0] == "chunk_domain" and v["state"] == "active"}
    assert sorted(rows) == [13, 14, 15, 16]
    for v in rows.values():
        assert v["score"] is not None and -1.0 <= v["score"] <= 1.0
        assert v["score_kind"] in ("cosine_lvl", "cosine")


def test_d2_incremental_run_row_has_domain_metrics():
    core, db = _inc_db()
    mk(db, e=VEmb(), core=core, incremental="on").run()
    m, _ = last_close(db)
    assert m["run"]["domain_table_absent"] == 0 and m["run"]["domain_rows_attempted"] == 4


def test_d2_incremental_table_absent_is_reported_and_nothing_written():
    core = DomCore()
    db = build_db(core=core)
    add_spans(db, [13, 14])
    mk(db, e=VEmb(), core=core, incremental="on").run()
    m, _ = last_close(db)
    assert m["run"]["domain_table_absent"] == 1 and not [k for k in db.links if k[0] == "chunk_domain"]


def test_d2_scores_from_centers_rules():
    item = dict(subtopic="S", topic="T", domain="D", level="subtopic", distance=0.1)
    dom = dict(subtopic=None, topic=None, domain="D", level="domain", distance=0.25)
    out = dw.domain_scores_from_centers({"D": [2.0, 0.0]}, [(1, item, [3.0, 4.0]), (2, dom, [1, 0]), (3, item, [0, 0]),
                                                           (4, dict(item, domain="X"), [1, 0])])
    assert out == {1: (0.6, "cosine_lvl"), 2: (0.75, "cosine")}
    assert dw.domain_scores_from_centers(None, None) == {}
