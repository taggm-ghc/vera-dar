"""WP-D-1113 plan A4 T52-T54: own-cosine domain scores, per-pass probe, kill switch. Offline; uses the job's FakeDB."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tests.test_chunk_taxo_job_1109 import FakeDB, StubCore, make, spans
from vera import chunk_domain_writer as dw


def _model():
    nodes = [{"key": "D", "level": "domain", "center": [1.0, 0.0]}, {"key": "D2", "level": "domain", "center": [0.0, 2.0]},
             {"key": "T", "level": "topic", "center": [1.0, 0.0]}]
    assoc = {
        1: dict(subtopic="S0", topic="T", domain="D", level="subtopic", distance=0.1),
        2: dict(subtopic=None, topic="T", domain="D2", level="topic", distance=0.3),
        3: dict(subtopic=None, topic=None, domain="D", level="domain", distance=0.25),   # domain-only back-off
        4: dict(subtopic=None, topic=None, domain=None, level=None, distance=0.9),       # abstained
        5: dict(subtopic="S0", topic="T", domain="D", level="subtopic", distance=0.1),   # zero vector
        6: dict(subtopic="S0", topic="T", domain="NOCENTER", level="subtopic", distance=0.1),
    }
    return SimpleNamespace(nodes=nodes, assoc=assoc, span_order=[1, 2, 3, 4, 5, 6])


VECS = [[3.0, 4.0], [3.0, 4.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]


def test_t52_scores_equal_a_recomputation_and_domain_only_uses_its_distance():
    sc = dw.domain_scores(_model(), VECS)
    assert sc[1] == (round(3.0 / 5.0, 6), "cosine_lvl")                  # cos((3,4), D center (1,0))
    assert sc[2] == (round(4.0 / 5.0, 6), "cosine_lvl")                  # cos((3,4), D2 center (0,2)); centers need not be unit length
    assert sc[3] == (0.75, "cosine")                                      # domain-only chunk: strict cosine scored from its distance
    assert 4 not in sc and 5 not in sc and 6 not in sc                    # abstained, zero vector, domain without a center: no score (NULL)
    assert all(-1.0 <= s <= 1.0 and math.isfinite(s) for s, _ in sc.values())


def test_t52_garbage_never_raises():
    assert dw.domain_scores(SimpleNamespace(nodes=None, assoc=None, span_order=None), []) == {}
    assert dw.domain_scores(SimpleNamespace(nodes=[{"level": "domain", "key": "D", "center": [1.0]}], assoc={1: dict(level="topic", domain="D")},
                                             span_order=[1]), [[1.0, 2.0]]) == {}                    # dimension mismatch


class CosCore(StubCore):
    """StubCore plus real centers and span_order, so the job computes its own domain scores."""

    def build(self, chunks, vecs, params, prior=None):
        m = super().build(chunks, vecs, params, prior)
        for n in m.nodes:
            n["center"] = [1.0, 0.0]
        m.span_order = sorted(int(c[0]) for c in chunks)
        return m


def _scored_job(db, **kw):
    j = make(db, **kw)
    j.core = CosCore()
    return j


def test_t52_job_stores_the_recomputed_scores_in_the_same_page():
    db = FakeDB(spans())
    db.domain_table = True
    j = _scored_job(db)
    assert j.run()["complete"]
    rows = {k[1]: v for k, v in db.links.items() if k[0] == "chunk_domain"}
    assert sorted(rows) == list(range(1, 11))
    for sid, v in rows.items():
        if sid % 3 == 0:      # "alpha" -> vector (1,0): cosine 1 to the center (1,0)
            assert (v["score"], v["score_kind"]) == (1.0, "cosine_lvl")
        elif sid % 3 == 1:    # "beta" -> (-1,0)
            assert (v["score"], v["score_kind"]) == (-1.0, "cosine_lvl")
        else:                 # "gamma" -> zero vector: the core decided at topic level; no cosine is defined, stored NULL
            assert (v["score"], v["score_kind"]) == (None, None)


def test_t53_table_appears_between_two_passes_probe_is_per_pass():
    db = FakeDB(spans())
    j = make(db)
    assert j.run()["complete"] and not db.active("chunk_domain")
    assert db.names().count("probe_chunk_domain") == 1
    db.domain_table = True
    db.spans = spans(12)                      # new span set: a new version, so the next pass writes again
    assert j.run()["complete"]
    assert [a for a, _ in db.active("chunk_domain")] == list(range(1, 13))
    assert db.names().count("probe_chunk_domain") == 2


@pytest.mark.parametrize("cfgval,writes", [(False, False), (True, True), (None, True), ("false", True), (0, True)])
def test_t54_kill_switch_is_the_literal_false(cfgval, writes):
    db = FakeDB(spans())
    db.domain_table = True
    kw = {} if cfgval is None else {"chunk_domain_write": cfgval}
    j = make(db, **kw)
    assert j.run()["complete"]
    assert bool(db.active("chunk_domain")) is writes
    if not writes:
        assert "probe_chunk_domain" not in db.names()
        import json
        assert json.loads(db.runs[-1]["metrics"])["run"]["domain_write_disabled"] == 1
    assert len(db.active("chunk_subtopic")) + len(db.active("chunk_topic")) == 10   # the other links are unaffected either way


def test_t55_no_ddl_and_no_007_table_in_any_domain_statement():
    db = FakeDB(spans())
    db.domain_table = True
    make(db).run()
    for name, _ in db.log:
        assert "domain_chunks" not in name
    from vera import chunk_taxo_store as store
    for name in ("probe_chunk_domain", "link_insert_chunk_domain", "link_retire_chunk_domain"):
        up = store.SQL[name].upper()
        assert "DOMAIN_CHUNKS" not in up and not any(b in up for b in ("CREATE ", "ALTER ", "DROP ", "GRANT ", "DELETE", "TRUNCATE"))
