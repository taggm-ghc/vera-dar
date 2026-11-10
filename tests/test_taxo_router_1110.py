"""TAXO-ROUTE-1110 R3: router in shadow mode. No network, no DB."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from vera import chunk_taxo, chunk_taxo_embed as emb, hybrid_retrieval as hr, taxo_router as tr
from tests.test_hybrid_retrieval_1109 import MODEL, Q, FakeEmbedder, Loader, _corpus

CFG = {"mode": "shadow", "min_routed_scored": 1, "route_rel_margin": 0.10, "route_abs_floor": 0.2, "neighbour_cos": 0.9}


@pytest.fixture(autouse=True)
def _reset():
    tr.set_config_for_tests(None)
    tr.clear_shadow()
    yield
    tr.set_config_for_tests(None)
    tr.clear_shadow()
    hr.reset_for_tests()
    hr.set_config_for_tests({"enabled": False})
    emb.unregister("fakehybrid")
    emb.reset_budget_for_tests()


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Idx:
    def __init__(self, mat, rows):
        self.mat, self.dim, self.rows = mat, mat.shape[1], [("d", i, i + 1) for i in range(len(rows))]
        self.span_ids = [100 + i for i in range(len(rows))]

    def __len__(self):
        return len(self.rows)


def _idx(rows):
    """rows: list of vectors (dim 4). span ids 100+i."""
    mat = np.stack([_unit(*r) for r in rows]).astype(np.float32)
    ix = _Idx(mat, rows)
    return ix


# partitions 0-2: domain X near e0; 3-4: domain Y near e1; 5: domain Z near e2; 6: no link; 7: outlier (linked to Z, far)
ROWS = [(1, .1, 0, 0), (1, .2, 0, 0), (1, 0, .1, 0), (.1, 1, 0, 0), (0, 1, .1, 0), (0, 0, 1, 0), (0, 0, 0, 1), (.3, .3, .3, -1)]
LINKS = [(100, "X"), (101, "X"), (102, "X"), (103, "Y"), (104, "Y"), (105, "Z"), (107, "Z")]
JS = {"version": "v1", "display_ok": True}
PIDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32)  # chunk 8 has no vector


def _router(cfg=None):
    ix = _idx(ROWS)
    r = tr.build(ix, LINKS, dict(tr.DEFAULTS, **(cfg or CFG)), job_state=JS)
    assert r is not None
    return ix, r


def _scores(*chunks):
    return {c: 1.0 for c in chunks}


def test_default_mode_is_off_and_file_is_valid():
    assert tr.config()["mode"] == "off"
    assert tr.mode() == "off"
    assert tr.assoc_hook(1, "X", 0.7) == (0.7, False)  # weight = link cosine, nothing removed


def test_excludes_clearly_unrelated_domain_but_keeps_unlinked_and_novector():
    ix, r = _router()
    d = tr.route(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 1, 3, 5), dict(tr.DEFAULTS, **CFG))
    assert d["route"] == "routed" and d["n_selected"] == 1
    R = d["R"]
    assert R[[0, 1, 2]].all() and not R[[3, 4, 5]].any()
    assert R[6] and R[8]  # unlinked partition and no-vector chunk always in (set U)


def test_cross_domain_question_keeps_both_domains():
    ix, r = _router()
    q = _unit(1, 1, 0, 0)  # between X and Y
    d = tr.route(r, ix, q, PIDX, _scores(0, 3), dict(tr.DEFAULTS, **CFG))
    assert d["route"] == "routed" and d["n_selected"] == 2
    assert d["R"][[0, 1, 2, 3, 4]].all() and not d["R"][5]


def test_neighbour_domains_are_added():
    ix, r = _router(dict(CFG, neighbour_cos=0.5))
    # X center and Y center are near-orthogonal; Z too: neighbours need cos >= 0.5, so none here
    assert not r.neighbour_mask.any()
    rows = [(1, 0, 0, 0), (1, 0, 0, 0), (.8, .6, 0, 0), (0, 0, 1, 0)]
    ix2 = _idx(rows)
    r2 = tr.build(ix2, [(100, "X"), (101, "X"), (102, "Y"), (103, "Z")], dict(tr.DEFAULTS, **dict(CFG, neighbour_cos=0.7)), job_state=JS)
    d = tr.route(r2, ix2, _unit(1, 0, 0, 0), np.array([0, 1, 2, 3], dtype=np.int32), _scores(0, 2),
                 dict(tr.DEFAULTS, **dict(CFG, route_rel_margin=0.01)))
    assert d["route"] == "routed" and d["R"][2] and not d["R"][3]  # Y pulled in as X's neighbour, Z excluded


def test_outlier_only_evidence_is_still_included():
    ix, r = _router()
    # partition 7 is an outlier, soft-linked to Z; a question about Z routes to Z and so includes it
    d = tr.route(r, ix, _unit(0, 0, 1, 0), PIDX, _scores(5, 7), dict(tr.DEFAULTS, **CFG))
    assert d["route"] == "routed" and d["R"][5] and d["R"][7]
    # an outlier with NO link is in U regardless of the question
    ix2 = _idx(ROWS)
    r2 = tr.build(ix2, LINKS[:-1], dict(tr.DEFAULTS, **CFG), job_state=JS)
    d2 = tr.route(r2, ix2, _unit(1, .1, 0, 0), PIDX, _scores(0, 7), dict(tr.DEFAULTS, **CFG))
    assert d2["route"] == "routed" and d2["R"][7]


def test_every_fallback_triggers():
    ix, r = _router()
    cfg = dict(tr.DEFAULTS, **CFG)
    q = _unit(1, .1, 0, 0)
    assert tr.route(None, ix, q, PIDX, _scores(0), cfg)["route"] == "global:router_not_loaded"
    assert tr.route(r, _idx(ROWS[:-1]), q, PIDX, _scores(0), cfg)["route"] == "global:version_mismatch"
    ix_same_len = _idx(ROWS)
    assert tr.route(r, ix_same_len, q, PIDX, _scores(0), cfg)["route"] == "global:version_mismatch"  # other index object
    assert tr.route(r, ix, None, PIDX, _scores(0), cfg)["route"] == "global:no_vector"
    assert tr.route(r, ix, _unit(0, 0, 0, 1), PIDX, _scores(0), dict(cfg, route_abs_floor=0.9))["route"] == "global:no_domain_fits"
    assert tr.route(r, ix, _unit(1, 1, 1, 0), PIDX, _scores(0), dict(cfg, route_rel_margin=2.0))["route"] == "global:no_domain_excluded"
    assert tr.route(r, ix, q, PIDX, _scores(0), dict(cfg, min_routed_scored=40))["route"] == "global:few_routed_scored"
    assert tr.route(r, ix, q, PIDX[:2], {5: 1.0}, "bad")["route"] == "global:error"  # never raises
    # link/job version and display gate (taxonomy job state the links came from)
    r_nov = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG), job_state={"version": None, "display_ok": True})
    assert tr.route(r_nov, ix, q, PIDX, _scores(0), cfg)["route"] == "global:job_version_unknown"
    r_gate = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG), job_state={"version": "v1", "display_ok": False})
    assert tr.route(r_gate, ix, q, PIDX, _scores(0), cfg)["route"] == "global:display_gate_failed"
    r_none = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG))  # no job state at all
    assert tr.route(r_none, ix, q, PIDX, _scores(0), cfg)["route"] == "global:job_version_unknown"


def test_usage_hook_can_soft_remove_an_association(monkeypatch):
    monkeypatch.setattr(tr, "assoc_hook", lambda sid, dk, sim: (sim, sid == 100))
    ix = _idx(ROWS)
    r = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG), job_state=JS)
    assert r.part_mask[0] == 0 and r.part_mask[1] != 0  # partition 0 loses its link; it is then in U (always in)


def test_link_cosine_is_the_initial_weight_and_soft_links_count():
    rows = [(1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]
    ix = _idx(rows)
    # same partition set, different weights on partition 2: the center moves toward the heavier member
    lo = tr.build(ix, [(100, "X", 0.9), (102, "X", 0.01), (101, "Y", 0.9)], dict(tr.DEFAULTS, **CFG), job_state=JS)
    hi = tr.build(ix, [(100, "X", 0.9), (102, "X", 5.0), (101, "Y", 0.9)], dict(tr.DEFAULTS, **CFG), job_state=JS)
    j = lo.domains.index("X")
    assert lo.centers[j][0] > hi.centers[j][0]
    assert lo.part_mask[2] != 0  # a tiny-score link still gives membership


def test_shadow_counts_recall_and_fallback_has_no_recall():
    ix, r = _router()
    tr.set_config_for_tests(CFG)
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 1, 2), [0, 1], 5)
    s = tr.last_shadow()
    assert s["route"] == "routed" and s["recall_candidates"] == 1.0 and s["missed_candidates"] == 0
    assert s["effective"] == "shadow" and s["recall_none_reason"] is None and s["n_unlinked"] == 2  # partition 6, chunk 8
    assert s["n_routed"] < s["n_chunks"] and s["cpu_thread_s"] >= 0 and "question" not in s
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)  # a globally found chunk (3, domain Y) is excluded
    s = tr.last_shadow()
    assert s["missed_candidates"] == 1 and s["recall_candidates"] == 0.5
    # fallback: recall is None with the reason, and the counterfactual is still counted
    tr.set_config_for_tests(dict(CFG, min_routed_scored=40))
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)
    s = tr.last_shadow()
    assert s["route"] == "global:few_routed_scored" and s["recall_candidates"] is None
    assert s["recall_none_reason"] == "global:few_routed_scored" and s["missed_candidates"] == 1
    tr.shadow(None, ix, None, PIDX, _scores(0), [], 5)
    s = tr.last_shadow()
    assert s["route"] == "global:router_not_loaded" and s["recall_candidates"] is None and s["missed_candidates"] is None
    # over budget: recall None, reason recorded
    tr.set_config_for_tests(dict(CFG, time_budget_ms=1))
    import vera.taxo_router as m
    clock = iter([0.0, 1.0])
    real = m.time.thread_time
    m.time.thread_time = lambda: next(clock)
    try:
        tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 1), [0], 5)
    finally:
        m.time.thread_time = real
    s = tr.last_shadow()
    assert s["route"] == "global:over_budget" and s["recall_candidates"] is None
    tr.clear_shadow()
    tr.set_config_for_tests({"mode": "off"})
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0), [], 5)
    assert tr.last_shadow() is None


def test_finish_counts_admitted_outside_routed_set():
    ix, r = _router()
    tr.set_config_for_tests(CFG)
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)
    tr.note_candidate("s0", 0)   # domain X: inside
    tr.note_candidate("s3", 3)   # domain Y: outside
    tr.note_candidate("s8", 8)   # no vector: inside (U)
    rec = tr.finish(["s0", "s3", "s8", "other"])
    assert rec["admitted_total"] == 4 and rec["missed_admitted"] == 1 and rec["admitted_unmapped"] == 1
    tr.clear_shadow()  # next ask starts clean
    assert tr.last_shadow() is None and tr.finish(["s0"]) is None


def test_finish_checks_merged_passages_member_by_member():
    ix, r = _router()
    tr.set_config_for_tests(CFG)
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)
    tr.note_candidate("m0", 0)  # inside
    tr.note_candidate("m1", 1)  # inside
    tr.note_candidate("m3", 3)  # routed out
    rec = tr.finish([{"span_id": "D:0-9", "merged_span_ids": ("m0", "m1")},
                     {"span_id": "D:10-19", "merged_span_ids": ("m1", "m3")},   # one routed-out member
                     {"span_id": "D:20-29", "merged_span_ids": ("zz",)}])      # no known member
    assert rec["admitted_total"] == 3 and rec["missed_admitted"] == 1 and rec["admitted_unmapped"] == 1


def test_gate_and_version_fallbacks_still_record_the_counterfactual():
    ix, _ = _router()
    tr.set_config_for_tests(CFG)
    for js, code in (({"version": "v1", "display_ok": False}, "global:display_gate_failed"),
                     ({"version": None, "display_ok": True}, "global:job_version_unknown")):
        r = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG), job_state=js)
        tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)
        s = tr.last_shadow()
        assert s["route"] == code and s["recall_candidates"] is None and s["missed_candidates"] == 1


def test_thread_safety_shared_router_per_thread_records():
    ix, r = _router()
    tr.set_config_for_tests(CFG)
    seen, errs = {}, []

    def work(n):
        try:
            q = _unit(1, .1, 0, 0) if n % 2 == 0 else _unit(0, 0, 1, 0)
            for _ in range(200):
                tr.shadow(r, ix, q, PIDX, _scores(0, 5), [0, 5], 5)
            seen[n] = tr.last_shadow()
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(8)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert not errs and len(seen) == 8 and all(not t.is_alive() for t in ts)
    assert all(s["route"].startswith(("routed", "global")) for s in seen.values())


# ------------------------------------------------------------------------------------------------ integration
class LinkLoader(Loader):
    state = {"version": "v1", "display_ok": True}
    bump_during_links = False

    def job_state(self):
        return dict(self.state)

    def links(self, ids):
        dom = {"DA": "A", "DB": "B", "DC": "C"}
        for i in ids:
            did = self.t[i][0]
            if did in dom:
                yield i, dom[did], 0.8
        if self.bump_during_links:
            self.state = {"version": "v2", "display_ok": True}


def _setup(monkeypatch, mode):
    c, table = _corpus()
    fake = FakeEmbedder()
    emb.register("fakehybrid", lambda cfg: fake)
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests({"enabled": True})
    tr.set_config_for_tests({"mode": mode, "min_routed_scored": 1, "route_abs_floor": -1.0, "route_rel_margin": 0.0})
    hr.set_loader_for_tests(LinkLoader(c, table))
    idx, code = hr.ensure_index(c, MODEL, sync=True)
    assert code == "ok"
    return c, idx


def test_shadow_never_changes_results_full_dicts_and_records_a_route(monkeypatch):
    c, idx = _setup(monkeypatch, "off")
    assert idx.router is None  # off: nothing built
    off = c.retrieve(Q, 16, True)
    assert tr.last_shadow() is None
    hr.reset_for_tests()
    c2, idx2 = _setup(monkeypatch, "shadow")
    assert idx2.router is not None and idx2.router.nbytes() < 5 * 1024 * 1024
    on = c2.retrieve(Q, 16, True)
    assert on == off  # whole candidate dicts: text, score, vec_sim, order
    s = tr.last_shadow()
    assert s is not None and s["route"] != "" and "recall_candidates" in s and s["effective"] == "shadow"


def test_admission_trace_carries_the_shadow_record_and_it_is_cleared_per_ask(monkeypatch):
    from vera import fulldoc_admission as fa
    from vera.fulldoc_corpus import load_config
    c, idx = _setup(monkeypatch, "off")
    off = fa.admit(Q, c, load_config())
    assert "route_shadow" not in off.trace
    hr.reset_for_tests()
    c2, idx2 = _setup(monkeypatch, "shadow")
    on = fa.admit(Q, c2, load_config())
    rs = on.trace["route_shadow"]
    assert rs["effective"] == "shadow" and rs["admitted_total"] == len(on.admitted) and "missed_admitted" in rs
    assert [a["span_id"] for a in on.admitted] == [a["span_id"] for a in off.admitted]
    assert Q not in repr(rs)
    # a second ask starts clean: the record is the new ask's, not a leftover
    tr.set_config_for_tests({"mode": "off"})
    again = fa.admit(Q, c2, load_config())
    assert "route_shadow" not in again.trace


def test_stale_links_during_build_and_gate_failure_fall_back(monkeypatch):
    c, table = _corpus()
    emb.register("fakehybrid", lambda cfg: FakeEmbedder())
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests({"enabled": True})
    tr.set_config_for_tests({"mode": "shadow", "min_routed_scored": 1})
    ld = LinkLoader(c, table)
    ld.bump_during_links = True
    hr.set_loader_for_tests(ld)
    idx, _ = hr.ensure_index(c, MODEL, sync=True)
    assert idx.router is None  # job version changed while the links were read
    c.retrieve(Q, 16, True)
    assert tr.last_shadow()["route"] == "global:router_not_loaded"
    hr.reset_for_tests()
    c2, table2 = _corpus()
    emb.register("fakehybrid", lambda cfg: FakeEmbedder())
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests({"enabled": True})
    ld2 = LinkLoader(c2, table2)
    ld2.state = {"version": "v1", "display_ok": False}
    hr.set_loader_for_tests(ld2)
    hr.ensure_index(c2, MODEL, sync=True)
    c2.retrieve(Q, 16, True)
    assert tr.last_shadow()["route"] == "global:display_gate_failed"


def test_router_not_loaded_without_links_api_is_global(monkeypatch):
    c, table = _corpus()
    emb.register("fakehybrid", lambda cfg: FakeEmbedder())
    tc = chunk_taxo.load_cfg()
    monkeypatch.setattr(hr, "_taxo_cfg", lambda: dict(tc, embedder="fakehybrid"))
    hr.set_config_for_tests({"enabled": True})
    tr.set_config_for_tests({"mode": "shadow"})
    hr.set_loader_for_tests(Loader(c, table))  # no links()
    hr.ensure_index(c, MODEL, sync=True)
    c.retrieve(Q, 16, True)
    assert tr.last_shadow()["route"] == "global:router_not_loaded"


def test_sql_matches_the_schema_contract():
    """The router SQL names only tables and columns the pending drafts create (sql_contract_support extractor)."""
    from tests import sql_contract_support as sc
    found = {(t, c) for t, c in sc.collect_all() if t in ("tax_topic_subtopic", "tax_chunk_topic", "tax_chunk_subtopic")}
    assert ("tax_topic_subtopic", "topic_id") in found and ("tax_subtopics", "topic_id") not in sc.collect_all()
    assert "tax_subtopics sb ON sb.subtopic_id" in tr.SQL_DOMAIN_LINKS and "LIKE 'cosine%'" in tr.SQL_DOMAIN_LINKS
    assert "'active'" in tr.SQL_DOMAIN_LINKS and tr.SQL_DOMAIN_LINKS.count("state = 'active'") >= 7


def test_offline_global_vs_masked_comparison_harness(monkeypatch):
    """Offline harness (fixtures only): admit each question globally, then again with candidates outside the routed set
    removed, and compare the admitted evidence. Questions: the Ask-check cases (when the file is present), plus synthetic
    cross-domain and outlier-only ones. Synthetic corpus: DA/DC are domain A/C, DB domain B."""
    import json
    from pathlib import Path
    from tests.taxo_router_compare import compare
    c, idx = _setup(monkeypatch, "shadow")
    qs = {"synthetic_cross_domain": "How do worker and supervisor agents relate to orchestration state across agent turns?",
          "synthetic_outlier": "Which agent archetypes are used in multi-agent orchestration?",
          "synthetic_unrelated": "What is the capital of a country nobody mentions?"}
    p = Path(__file__).resolve().parent / "taxo_router_ask_questions.json"
    assert p.is_file(), "tests/taxo_router_ask_questions.json missing"  # fail loudly
    ask = [x["question"] for x in json.loads(p.read_text(encoding="utf-8"))["questions"]]
    assert len(ask) == 19, len(ask)
    rows = compare(c, list(qs.values()) + ask)
    assert len(rows) == len(qs) + len(ask)
    for r in rows:
        assert set(r) >= {"route", "global_admitted", "masked_admitted", "lost", "gained", "missed_admitted"}
        assert r["lost"] == len(set(r["global_admitted"]) - set(r["masked_admitted"]))
    # the harness detects over-narrowing: margin 0 routes to one domain and loses cross-domain evidence
    assert any(r["lost"] > 0 and r["route"] == "routed" for r in rows[:2]), rows[:2]
    assert all(r["missed_admitted"] for r in rows if r["lost"])  # a loss implies admitted evidence outside R
    # generous margin (D2/D3): every domain selected => global fallback, nothing lost
    tr.set_config_for_tests({"mode": "shadow", "min_routed_scored": 1, "route_abs_floor": -1.0, "route_rel_margin": 2.0})
    rows = compare(c, list(qs.values()) + ask)
    assert all(r["lost"] == 0 and r["route"] == "global:no_domain_excluded" for r in rows), rows[:2]
