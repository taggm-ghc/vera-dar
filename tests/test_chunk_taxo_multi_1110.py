"""TAXO-ROUTE-R1-1110: generous multi-membership links (core rule, job writer, strict display counts)."""
from __future__ import annotations

from vera import chunk_taxo as ct

P = ct._merge(ct.DEFAULT_PARAMS, {}, "params")


def _info(sub_centers=3):
    keys = ["S%d" % i for i in range(sub_centers)]
    sub = {"keys": keys, "centers": [], "counts": [10] * sub_centers, "parent": ["T0", "T0", "T1"][:sub_centers],
           "parent_idx": [0, 0, 1][:sub_centers], "null": {"dist_q": 0.3, "margin_q": 0.1}}
    top = {"keys": ["T0", "T1"], "centers": [], "counts": [20, 10], "parent": ["D", "D"], "parent_idx": [0, 0],
           "null": {"dist_q": 0.3, "margin_q": 0.1}}
    dom = {"keys": ["D"], "centers": [], "counts": [30], "parent": [None], "parent_idx": [], "null": {"dist_q": 0.3, "margin_q": 0.1}}
    return {"subtopic": sub, "topic": top, "domain": dom}


def test_ambiguous_chunk_gets_both_links_with_raw_cosine():
    sims = {"subtopic": [0.80, 0.78, 0.30], "topic": [0.80, 0.30], "domain": [0.8]}
    a = ct._decide(_info(), sims, P)
    assert a["level"] is None or a["subtopic"] is None  # SPLIT: strict rule drops the subtopic
    got = {(lv, k): (s, kind) for lv, k, s, kind in a["multi"]}
    assert got[("subtopic", "S0")] == (0.8, "cosine_r1") and got[("subtopic", "S1")] == (0.78, "cosine_r2")
    assert ("subtopic", "S2") not in got  # 0.30 is outside margin and floor


def test_outlier_gets_only_soft_routing_link():
    sims = {"subtopic": [0.20, 0.10, 0.05], "topic": [0.20, 0.05], "domain": [0.2]}
    a = ct._decide(_info(), sims, P)
    assert a["level"] is None and a["topic"] is None
    assert [(lv, k, kind) for lv, k, s, kind in a["multi"]] == [("subtopic", "S0", "cosine_soft"), ("topic", "T0", "cosine_soft")]
    assert [s for *_x, s, _k in [(0, 0, e[2], e[3]) for e in a["multi"]]] == [0.2, 0.2]


def test_caps_respected_and_strict_result_unchanged():
    info = _info()
    sims = {"subtopic": [0.9, 0.89, 0.88], "topic": [0.9, 0.89], "domain": [0.9]}
    p = ct._merge(P, {"multi": {"max_links": {"subtopic": 2, "topic": 1, "domain": 1}}}, "params")
    a = ct._decide(info, sims, p)
    assert len([m for m in a["multi"] if m[0] == "subtopic"]) <= 2 and len([m for m in a["multi"] if m[0] == "topic"]) <= 1
    off = ct._merge(P, {"multi": {"enabled": False}}, "params")
    b = ct._decide(info, sims, off)
    assert "multi" not in b
    assert {k: v for k, v in a.items() if k != "multi"} == b  # strict fields identical with multi on or off


def test_config_validates():
    import json
    import pathlib
    import pytest
    cfg = ct.load_cfg()
    assert cfg["params"]["multi"]["max_links"] == {"subtopic": 4, "topic": 3, "domain": 3}
    bad = pathlib.Path(__file__).parent  # unused; validate directly
    for patch in ({"max_links": {"subtopic": 0, "topic": 1, "domain": 1}}, {"rel_margin_mult": -1}, {"enabled": "x"}):
        with pytest.raises(ValueError):
            ct._validate_params(ct._merge(P, {"multi": patch}, "params"))
    del json, bad


def test_job_page_writes_all_links_with_cosine_scores_no_duplicates_and_idempotent(monkeypatch):
    from vera import chunk_taxo_job as job
    from vera import chunk_taxo_store as store
    ids = {"subtopic": {"S0": 10, "S1": 11}, "topic": {"T": 20}, "domain": {"D": 30}}
    assoc = {
        1: dict(subtopic="S0", topic="T", domain="D", level="subtopic", distance=0.1, margin=0.5, reason=None,
                multi=[("subtopic", "S0", 0.9, "cosine_r1"), ("subtopic", "S1", 0.77, "cosine_r2"), ("topic", "T", 0.91, "cosine_r1"),
                       ("subtopic", "S1", 0.77, "cosine_r2")]),
        2: dict(subtopic=None, topic=None, domain=None, level=None, distance=0.7, margin=0.0, reason="NO_FIT",
                multi=[("subtopic", "S0", 0.3, "cosine_soft"), ("topic", "T", 0.3, "cosine_soft")]),
    }
    rows = {}
    log = []

    def run_sql(conn, name, params):
        log.append(name)
        if name.startswith("link_insert_"):
            for r in params:
                rows.setdefault((name, r["a"], r["b"]), []).append(r)
    monkeypatch.setattr(store, "run_sql", run_sql)
    j = object.__new__(job.Job)
    for _ in range(2):  # a rerun of the same version re-sends the same rows (ON CONFLICT DO NOTHING in SQL)
        j._write_chunk_page(None, [1, 2], assoc, ids, "v1")
    assert all(len(v) == 2 and v[0] == v[1] for v in rows.values())  # identical rows on rerun, never two different ones
    first = {k: v[0] for k, v in rows.items()}
    assert first[("link_insert_chunk_subtopic", 1, 10)]["score_kind"] == "cosine"
    assert first[("link_insert_chunk_subtopic", 1, 10)]["score"] == 0.9  # raw cosine = 1 - distance
    assert first[("link_insert_chunk_subtopic", 1, 11)] ["score"] == 0.77 and first[("link_insert_chunk_subtopic", 1, 11)]["score_kind"] == "cosine_r2"
    assert first[("link_insert_chunk_topic", 1, 20)]["score_kind"] == "cosine_r1"
    soft = [v for k, v in first.items() if k[1] == 2]
    assert len(soft) == 2 and {v["score_kind"] for v in soft} == {"cosine_soft"} and {v["score"] for v in soft} == {0.3}
    one_pass = [k for k in first if k[1] == 1]
    assert len(one_pass) == 3  # the repeated S1 entry did not make a duplicate (span, node)


def test_display_counts_use_strict_links_only():
    import pytest
    pytest.importorskip('sqlalchemy')
    from sqlalchemy import text
    from tests.test_corpus_taxonomy_wp6_1109 import _engine, _ins, CN
    from vera import corpus_taxonomy as ctx
    e = _engine()
    with e.begin() as c:
        _ins(c, "sources", [(1, 1), (2, 2)])
        _ins(c, "evidence_spans", [(1, 1), (2, 2)])
        _ins(c, "tax_topics", [(1, "a", "A", "active"), (2, "b", "B", "active")])
        _ins(c, "tax_subtopics", [(1, "s", "S", "active")])
        _ins(c, "tax_topic_subtopic", [(1, 1, 1, "active")])
        _ins(c, "tax_chunk_topic", [(1, 1, 1, CN, "active", "cosine")])
        _ins(c, "tax_chunk_topic", [(2, 2, 2, CN, "active", "cosine_r1"), (3, 1, 2, CN, "active", "cosine_soft")])
        _ins(c, "tax_chunk_subtopic", [(1, 2, 1, CN, "active", "cosine_r2")])
    with e.connect() as c:
        rows = c.execute(text(ctx.Q_TOPIC_UNION), {"cn": CN}).fetchall()
        assert {(r[0], r[1]) for r in rows} == {(1, 1)}  # only the strict chunk 1 -> topic 1
        assert c.execute(text(ctx.Q_LABELLED), {"cn": CN}).fetchone()[0] == 1
        assert c.execute(text(ctx.Q_SUBTOPIC_UNION), {"cn": CN}).fetchall() == []
