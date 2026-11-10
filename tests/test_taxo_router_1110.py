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
PIDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int32)  # chunk 8 has no vector


def _router(cfg=None):
    ix = _idx(ROWS)
    r = tr.build(ix, LINKS, dict(tr.DEFAULTS, **(cfg or CFG)))
    assert r is not None
    return ix, r


def _scores(*chunks):
    return {c: 1.0 for c in chunks}


def test_default_mode_is_off_and_file_is_valid():
    assert tr.config()["mode"] == "off"
    assert tr.mode() == "off"
    assert tr.assoc_hook(1, "X") == (1.0, False)


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
    r2 = tr.build(ix2, [(100, "X"), (101, "X"), (102, "Y"), (103, "Z")], dict(tr.DEFAULTS, **dict(CFG, neighbour_cos=0.7)))
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
    r2 = tr.build(ix2, LINKS[:-1], dict(tr.DEFAULTS, **CFG))
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


def test_usage_hook_can_soft_remove_an_association(monkeypatch):
    monkeypatch.setattr(tr, "assoc_hook", lambda sid, dk: (1.0, sid == 100))
    ix = _idx(ROWS)
    r = tr.build(ix, LINKS, dict(tr.DEFAULTS, **CFG))
    assert r.part_mask[0] == 0 and r.part_mask[1] != 0  # partition 0 loses its link; it is then in U (always in)


def test_shadow_records_recall_and_counts_and_off_records_nothing():
    ix, r = _router()
    cfg = dict(CFG)
    tr.set_config_for_tests(cfg)
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 1, 2), [0, 1], 5)
    s = tr.last_shadow()
    assert s["route"] == "routed" and s["recall"] == 1.0 and s["missed"] == 0 and s["global_candidates"] == 3
    assert s["n_routed"] < s["n_chunks"] and s["process_time_s"] >= 0 and "question" not in s
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0, 3), [], 5)  # a globally found chunk (3, domain Y) is excluded
    s = tr.last_shadow()
    assert s["missed"] == 1 and s["recall"] == 0.5
    tr.clear_shadow()
    tr.set_config_for_tests({"mode": "off"})
    tr.shadow(r, ix, _unit(1, .1, 0, 0), PIDX, _scores(0), [], 5)
    assert tr.last_shadow() is None


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
    def links(self, ids):
        dom = {"DA": "A", "DB": "B", "DC": "C"}
        for i in ids:
            did = self.t[i][0]
            if did in dom:
                yield i, dom[did]


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


def test_shadow_never_changes_results_and_records_a_route(monkeypatch):
    c, idx = _setup(monkeypatch, "off")
    assert idx.router is None  # off: nothing built
    off = [x["span_id"] for x in c.retrieve(Q, 16, True)]
    assert tr.last_shadow() is None
    hr.reset_for_tests()
    c2, idx2 = _setup(monkeypatch, "shadow")
    assert idx2.router is not None and idx2.router.nbytes() < 5 * 1024 * 1024
    on = [x["span_id"] for x in c2.retrieve(Q, 16, True)]
    assert on == off
    s = tr.last_shadow()
    assert s is not None and s["route"] != "" and "recall" in s


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
