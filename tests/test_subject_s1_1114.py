"""SUBJECT-BUILD-1114 WP-S1: vera/subject.py (T1-T13 of work/SUBJECT-1114/PLAN.md). Fakes only: no database, no network, no model."""
import re
from contextlib import contextmanager

import pytest

from vera import corpus_taxonomy as ctx
from vera import subject as sj

CFG = {"stored_taxonomy": {"enabled": True, "classifier_name": "chunk_taxo_auto"}}
OPEN = {"display_ok": True, "version": "v1"}
D1, D2 = 1, 2
T10, T11 = 10, 11
S100 = 100
LAB = {D1: "energy cluster", D2: "land cluster", T10: "solar, grid, panel", T11: "wind, turbine, blade",
       S100: "solar, grid, panel #2", 101: "grid, storage, battery"}
CIT = {"span_id": "7:0-300", "document_id": "7", "canonical_source_id": "10.1234/abc", "start": 0, "end": 300,
       "merged_span_ids": (501, "7:0-300"), "corpus_chunks": [{"span_id": 501, "start": 0, "end": 200}]}


def R(k, node, span=None, parent=None, score=None, label=None):
    return (k, span, node, parent, score, label if label is not None else LAB.get(node))


HIER = [R("ts", S100, parent=T10), R("dt", T10, parent=D1), R("tn", T10), R("tn", T11), R("dn", D1), R("dn", D2)]


class Res:
    def __init__(self, rows=None, val=None):
        self._r, self._v = rows or [], val

    def all(self):
        return self._r

    def scalar(self):
        return self._v


class Eng:
    def __init__(self, rows, has_domain=False, boom=False):
        self.rows, self.has, self.boom, self.log = rows, has_domain, boom, []

    @contextmanager
    def begin(self):
        if self.boom:
            raise RuntimeError("no db")
        eng = self

        class C:
            def execute(self, stmt, params=None):
                s = str(stmt)
                eng.log.append((s, params))
                if s.startswith("SET LOCAL"):
                    return Res()
                if "to_regclass" in s:
                    return Res(val=eng.has)
                return Res(rows=eng.rows)
        yield C()

    def selects(self):
        return [(s, p) for s, p in self.log if s.startswith("SELECT 'cs'")]


@pytest.fixture(autouse=True)
def _reset():
    sj._PROBE_CACHE.update(version=object(), has=False)


def run(rows, cit=CIT, has_domain=False, cfg=CFG, status=OPEN):
    e = Eng(rows, has_domain)
    return sj.subject_for(cit, engine=e, cfg=cfg, status=status), e


def lv(o):
    return (o["domain"], o["topic"], o["subtopic"], o["basis"], o["reason"])


def test_t1_subtopic_three_levels():
    o, e = run([R("cs", S100, span=501, score=0.9)] + HIER)
    assert lv(o) == ("energy cluster", "solar, grid, panel", "solar, grid, panel #2", "subtopic", None)
    assert o["ids"] == {"domain": D1, "topic": T10, "subtopic": S100}
    assert e.log[0][0].startswith("SET LOCAL statement_timeout = 300")


def test_t2_topic_only():
    o, _ = run([R("ct", T10, span=501, score=0.8)] + HIER)
    assert lv(o) == ("energy cluster", "solar, grid, panel", None, "topic", None)


def test_t3_domain_only_cosine_row():
    o, e = run([R("cd", D1, span=501, score=0.7)] + HIER, has_domain=True)
    assert lv(o) == ("energy cluster", None, None, "domain", None)
    assert "tax_chunk_domain" in e.selects()[0][0]


def test_t3b_cosine_lvl_domain_crosschecks_equal():
    o, _ = run([R("cs", S100, span=501, score=0.9), R("cd", D1, span=501, score=0.5)] + HIER, has_domain=True)
    assert o["domain"] == "energy cluster" and o["counts"]["domain_mismatch"] == 0
    assert "tax_chunk_domain" in str(sj.SQL_WITH_DOMAIN) and not re.search(r"cd\.score_kind", sj.SQL_WITH_DOMAIN)  # never strict-filtered


def test_t4_table_absent_uses_hierarchy_domain_no_error():
    o, e = run([R("ct", T10, span=501)] + HIER, has_domain=False)
    assert o["domain"] == "energy cluster" and o["reason"] is None
    assert "tax_chunk_domain" not in e.selects()[0][0]


def test_t4b_probe_cached_and_rerun_on_version_change():
    e = Eng([R("ct", T10, span=501)] + HIER, has_domain=True)
    probes = lambda: sum(1 for s, _ in e.log if "to_regclass" in s)  # noqa: E731
    sj.subject_for(CIT, engine=e, cfg=CFG, status=OPEN)
    sj.subject_for(CIT, engine=e, cfg=CFG, status=OPEN)
    assert probes() == 1
    sj.subject_for(CIT, engine=e, cfg=CFG, status={"display_ok": True, "version": "v2"})
    assert probes() == 2


def test_t5_strict_and_classifier_in_sql():
    o, e = run([R("ct", T10, span=501)] + HIER, cfg={"stored_taxonomy": {"enabled": True, "classifier_name": "other_cls"}})
    sql, params = e.selects()[0]
    assert params["cn"] == "other_cls" and "cs.classifier_name = :cn" in sql and "ct.classifier_name = :cn" in sql
    assert sj.STRICT_CT in sql and sj.STRICT_CS in sql
    assert params["spans"] == [501] and params["doc"] == -1


def test_t6_retired_rows_filtered_in_every_branch():
    sql = sj.SQL_WITH_DOMAIN
    joins = re.findall(r"JOIN vera_vjay\.tax_(?:topics|subtopics|domains) \w+ ON [^J]*?state = 'active'", sql)
    assert len(joins) == len(re.findall(r"JOIN vera_vjay\.tax_(?:topics|subtopics|domains) ", sql))
    for alias in ("cs", "ct", "cd", "ts", "dt", "st", "sd"):
        assert f"{alias}.state = 'active'" in sql
    assert "tn" in sql and "t.state = 'active'" in sql and "d.state = 'active'" in sql


def test_t7_ties_score_then_lowest_id_order_independent():
    rows = [R("ct", T11, span=501, score=0.8), R("ct", T10, span=501, score=0.8), R("ct", 12, span=502, score=0.6, label="x")] + HIER
    a, _ = run(rows)
    b, _ = run(list(reversed(rows)))
    assert a == b and a["ids"]["topic"] == T10
    rows2 = [R("ct", T10, span=501, score=0.5), R("ct", T11, span=502, score=0.9)] + HIER
    assert run(rows2)[0]["ids"]["topic"] == T11
    rows3 = [R("ct", T11, span=501, score=0.99), R("cs", S100, span=502, score=0.1)] + HIER
    assert run(rows3)[0]["basis"] == "subtopic"  # deepest wins over score


def test_t8_document_vote_only():
    o, e = run([R("vt", T10, score=0.7)] + HIER)
    assert lv(o) == ("energy cluster", "solar, grid, panel", None, "document_vote", None)
    assert e.selects()[0][1]["doc"] == -1  # "10.1234/abc" is not an integer key
    o2, _ = run([R("vd", D2)], cit={"document_id": 7})
    assert lv(o2) == ("land cluster", None, None, "document_vote", None)


def test_t9_nothing():
    o, _ = run([])
    assert lv(o) == (None, None, None, "none", "no_record")


def test_t10_gate_closed_no_db_no_labels():
    bad = Eng([R("ct", T10, span=501)] + HIER, boom=True)
    for cfg, st in ((CFG, {"display_ok": False}), ({"stored_taxonomy": {"enabled": False}}, OPEN), ({}, OPEN),
                    (CFG, {"display_ok": "yes"})):
        o = sj.subject_for(CIT, engine=bad, cfg=cfg, status=st)
        assert lv(o) == (None, None, None, "none", "display_gate_closed")


def test_t11_domain_mismatch_and_multi_parent():
    o, _ = run([R("ct", T10, span=501), R("cd", D2, span=501)] + HIER, has_domain=True)
    assert o["domain"] is None and o["topic"] == "solar, grid, panel" and o["counts"]["domain_mismatch"] == 1
    rows = [R("ct", T10, span=501), R("dt", T10, parent=D2)] + HIER
    o2, _ = run(rows)
    assert o2["ids"]["domain"] == D1 and o2["counts"]["multi_parent"] == 1
    rows3 = [R("cs", S100, span=501), R("ts", S100, parent=9)] + HIER + [R("tn", 9, label="zz")]
    o3, _ = run(rows3)
    assert o3["ids"]["topic"] == 9 and o3["counts"]["multi_parent"] >= 1


def test_t12_id_shapes():
    ids, doc = sj.resolve_ids(CIT)
    assert ids == (501,) and doc is None  # string span id and non-integer canonical key ignored
    merged = {"span_id": "7:0-900", "document_id": 7, "canonical_source_id": None, "start": 0, "end": 900,
              "merged_span_ids": (11, 12, "7:0-300", "7:300-600"),
              "corpus_chunks": [{"span_id": 13, "start": 800, "end": 1000}, {"span_id": 99, "start": 5000, "end": 6000},
                                {"span_id": 98, "start": None, "end": None}]}
    assert sj.resolve_ids(merged) == ((11, 12, 13), 7)
    assert sj.resolve_ids({"document_id": 7, "canonical_source_id": 3})[1] == 3
    o = sj.subject_for({"span_id": "x:1-2", "document_id": "doc-x"}, engine=Eng([]), cfg=CFG, status=OPEN)
    assert lv(o) == (None, None, None, "none", "unmapped")
    # merged members: deepest strict node among members wins
    o2, e = run([R("ct", T11, span=11, score=0.99), R("cs", S100, span=12, score=0.2)] + HIER, cit=merged)
    assert o2["basis"] == "subtopic" and e.selects()[0][1]["spans"] == [11, 12, 13]
    assert e.selects()[0][1]["doc"] == 7


def test_t12b_exception_is_unavailable():
    o = sj.subject_for(CIT, engine=Eng([], boom=True), cfg=CFG, status=OPEN)
    assert lv(o) == (None, None, None, "none", "unavailable")


def test_t13_alias_parity_with_imported_constants():
    assert ctx.STRICT_CT is sj.STRICT_CT and ctx.STRICT_CS is sj.STRICT_CS
    assert sj.STRICT_CT.startswith("(ct.") and sj.STRICT_CS.startswith("(cs.")
    for sql in (sj.SQL_WITH_DOMAIN, sj.SQL_NO_DOMAIN):
        assert "tax_chunk_topic ct" in sql and "tax_chunk_subtopic cs" in sql
        assert sj.STRICT_CT in sql and sj.STRICT_CS in sql
    assert "tax_chunk_domain" not in sj.SQL_NO_DOMAIN
