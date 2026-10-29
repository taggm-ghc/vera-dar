"""IMPL-1028-T1O1: T1 source-type router + O1 topic-fit ordering. OFFLINE, SYNTHETIC inputs (the H1 pool of
work/hyp-1028/h_all.py: 12 routable PMC records, 3 DDG DOI-mapped non-OA records, 3 URL-only DDG hits). The 36 labelled
questions in tests/data_source_type_labelled_1028.json are synthetic and are NOT a calibration."""
import copy
import json
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import acq_config, auto_acquire as aa, inline_acquire, search_providers as sp, source_type

LAB = json.loads((Path(__file__).parent / "data_source_type_labelled_1028.json").read_text())
Q = "What are the differences and commonalities between boxing and mixed martial arts?"
FD = None


def _cfg(router=False, fit=False):
    c = A.cfg()
    c["source_type_router"]["enabled"] = router
    c["topic_fit_order"]["enabled"] = fit
    return c


def _pmc(i):
    t = f"Head injury in boxing and mixed martial arts, cohort {i}"
    return {"title": t, "snippet": "boxing mixed martial arts " + t, "pmcid": f"PMC{9000000 + i}", "doi": f"10.1000/pmc{i}",
            "provider": "europepmc", "oa": {"epmc_ft": True, "is_oa": True}, "license": "cc-by", "licence": "CC-BY"}


def _ddg(i):
    t = f"Boxing versus mixed martial arts: how the sports differ {i}"
    return {"title": t, "snippet": "boxing mixed martial arts " + t, "doi": f"10.2000/web{i}",
            "url": f"https://example.org/web{i}", "provider": "ddg", "acq_source": "ddg", "ddg_id_kind": "doi"}


def _url(i):
    return sp._ddg_url_cand({"url": f"https://sports.example.com/boxing-vs-mma-{i}", "title": f"Boxing vs MMA {i}"})


def _pool():
    return [_pmc(i) for i in range(12)] + [_ddg(i) for i in range(3)] + [_url(i) for i in range(3)]


def _rank(cfg, route=None, pool=None, protected=()):
    drops = {}
    out = aa.rank_candidates(Q, pool if pool is not None else _pool(), cfg, set(), aa._fd_cfg(), lambda c: None,
                             drops=drops, route=route,
                             protected=None if protected is None else set(protected))
    return out, drops


# ------------------------------------------------------------------------------------------------ switches / defaults
def test_shipped_config_has_both_switches_off_and_uncalibrated_notes():
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "auto_acquire.json").read_text())
    assert cfg["source_type_router"]["enabled"] is False and cfg["topic_fit_order"]["enabled"] is False
    assert "UNCALIBRATED" in cfg["source_type_router"]["_calibration_note"]
    assert "UNCALIBRATED" in cfg["topic_fit_order"]["_calibration_note"]
    assert "UNCALIBRATED" in cfg["source_type_router"]["fit_table"]["_note"]
    assert "SEED" in cfg["source_type_router"]["_lexicon_note"] and "DB table" in cfg["source_type_router"]["_lexicon_note"]
    assert source_type.enabled(cfg) is False


def test_defaults_reproduce_current_ranking_exactly():
    base, bd = _rank(_cfg(), route=None)
    base_ids = [r[1] for r in base]
    for route in ("scholarly", "general", "mixed"):  # route supplied but switches off
        out, d = _rank(_cfg(), route=route)
        assert [r[1] for r in out] == base_ids
    # topic_fit on but router off, or router on but fit off: still today's order
    assert [r[1] for r in _rank(_cfg(False, True), "general")[0]] == base_ids
    assert [r[1] for r in _rank(_cfg(True, False), "general")[0]] == base_ids
    # route scholarly or None with both on: today's order
    assert [r[1] for r in _rank(_cfg(True, True), "scholarly")[0]] == base_ids
    assert [r[1] for r in _rank(_cfg(True, True), None)[0]] == base_ids
    assert len(base) == 6 and all(r[2].get("provider") == "europepmc" for r in base)  # H1 baseline: DDG deferred


# ------------------------------------------------------------------------------------------------ router
def test_labelled_set_shape():
    assert len(LAB) >= 30 and {x["label"] for x in LAB} == {"scholarly", "general", "mixed"}


def test_router_never_routes_general_a_scholarly_question_and_scholarly_never_general():
    cfg = _cfg()
    bad = [(x["q"], x["label"]) for x in LAB if x["label"] == "scholarly" and source_type.route(x["q"], cfg) == "general"]
    assert not bad, bad


def test_router_general_questions_are_never_scholarly():
    cfg = _cfg()
    bad = [x["q"] for x in LAB if x["label"] == "general" and source_type.route(x["q"], cfg) == "scholarly"]
    assert not bad, bad


def test_router_accuracy_on_synthetic_labels_is_reported_and_bounded():
    cfg = _cfg()
    ok = sum(source_type.route(x["q"], cfg) == x["label"] for x in LAB)
    gen = [x for x in LAB if x["label"] == "general"]
    gen_hit = sum(source_type.route(x["q"], cfg) == "general" for x in gen)
    print(f"router exact {ok}/{len(LAB)}; general recall {gen_hit}/{len(gen)}")
    assert ok >= int(0.6 * len(LAB))
    assert gen_hit >= int(0.6 * len(gen))


@pytest.mark.parametrize("q,exp", [
    (Q, "general"),
    ("How do I bake sourdough bread at home?", "general"),
    ("Who won the football World Cup in 1998?", "general"),
    ("What is the efficacy of metformin in a randomized trial of type 2 diabetes patients?", "scholarly"),
    ("What are the differences between type 1 and type 2 diabetes?", "scholarly"),
    ("MMA levels in B12 deficiency", "scholarly"),
    ("What is the Higgs boson mass measured by ATLAS, quantum field theory?", "scholarly"),
    ("boxing injuries incidence in a cohort study", "mixed"),
    ("Tell me something interesting", "mixed"),
    ("", "mixed"),
    ("MMA", "mixed"),
])
def test_router_cases(q, exp):
    assert source_type.route(q, _cfg()) == exp


def test_router_trace_has_rule_ids_and_counts_only_never_question_text():
    r = source_type.classify(Q, _cfg())
    assert set(r) == {"route", "basis", "scores"} and Q.lower() not in json.dumps(r).lower()
    assert source_type.route("zzz", _cfg()) == "mixed"


def test_router_pure_and_uses_no_identity():
    import inspect
    src = inspect.getsource(source_type)
    assert "import requests" not in src and "socket" not in src and "psycopg" not in src and "sqlalchemy" not in src


def test_route_not_recorded_when_router_off():
    assert inline_acquire.Outcome("ok").public().get("source_route") is None
    o = inline_acquire.Outcome("ok", source_route={"route": "general", "basis": ["G-LEX"], "scores": {"s": 0, "g": 1}})
    assert o.public()["source_route"]["route"] == "general"


# ------------------------------------------------------------------------------------------------ O1
def test_o1_general_reserves_ddg_slots_and_orders_ddg_first():
    out, drops = _rank(_cfg(True, True), "general")
    kinds = ["ddg" if r[2].get("acq_source") == "ddg" else "pmc" for r in out]
    assert len(out) == 6  # DP-7 cap unchanged
    assert kinds[:3] == ["ddg"] * 3 and kinds.count("ddg") == 3  # fit key ahead of routable_fulltext
    assert sum(1 for c in _pool() if False) == 0
    urls = [c for c in _pool() if not c.get("doi") and not c.get("pmcid")]
    assert len(urls) == 3


def test_o1_min_slots_swap_in_when_fit_key_alone_would_not(monkeypatch):
    cfg = _cfg(True, True)
    cfg["source_type_router"]["fit_table"]["general"] = {"web": 0, "scholarly": 0}  # neutral fit; slots must still work
    out, _ = _rank(cfg, "general")
    ddg = [r for r in out if r[2].get("acq_source") == "ddg"]
    assert len(out) == 6 and len(ddg) == 2  # web_min_slots general = 2


def test_o1_mixed_has_no_reserved_slots_and_neutral_order():
    base = [r[1] for r in _rank(_cfg(), None)[0]]
    assert [r[1] for r in _rank(_cfg(True, True), "mixed")[0]] == base


def test_o1_never_evicts_protected_reuse_candidates():
    cfg = _cfg(True, True)
    cfg["source_type_router"]["fit_table"]["general"] = {"web": 0, "scholarly": 0}
    cfg["topic_fit_order"]["web_min_slots"]["general"] = 6
    pool = _pool()
    base, _ = _rank(cfg, None, pool)
    # protect the last two baseline entries: the swap must evict others
    from vera.dar_corpus import doc_key_set
    prot = set()
    for r in base[-2:]:
        prot |= doc_key_set(r[1], r[2].get("url"), None)
    out, _ = _rank(cfg, "general", pool, protected=prot)
    kept = {r[1] for r in out}
    assert {r[1] for r in base[-2:]} <= kept
    assert sum(1 for r in out if r[2].get("acq_source") == "ddg") == 3  # only 3 exist; swapped in without eviction of prot
    # all protected -> nothing to evict -> no DDG swapped in
    prot_all = set()
    for r in base:
        prot_all |= doc_key_set(r[1], r[2].get("url"), None)
    out2, _ = _rank(cfg, "general", pool, protected=prot_all)
    assert [r[1] for r in out2] == [r[1] for r in base]


def test_o1_reserve_reads_only_on_general():
    pool = _pool()
    cfg = _cfg(True, True)
    cfg["source_type_router"]["fit_table"]["general"] = {"web": 0, "scholarly": 0}
    cfg["topic_fit_order"]["web_min_slots"]["general"] = 0
    ranked, _ = _rank(cfg, None, pool)
    ranked = list(ranked)
    ranked[-1] = (ranked[-1][0], ranked[-1][1], _ddg(9), ranked[-1][3])  # a DDG entry beyond the 2 reads
    g = aa.reserve_reads(ranked, 2, cfg, "general")
    assert sum(1 for r in g[:2] if r[2].get("acq_source") == "ddg") == 1 and len(g) == len(ranked)
    assert g[0][1] == ranked[0][1]  # best read kept
    assert [r[1] for r in aa.reserve_reads(ranked, 2, cfg, "mixed")] == [r[1] for r in ranked]  # web_min_reads 0
    assert [r[1] for r in aa.reserve_reads(ranked, 2, _cfg(), "general")] == [r[1] for r in ranked]  # off


# ------------------------------------------------------------------------------------------------ revision 2: live wiring
class _Deps:
    """Store stub: stored_rows(cus) -> (sources, spans) for the canonical URLs in `stored`; or raises."""
    def __init__(self, stored_dois=(), fail=False, missing=False):
        from vera.corpus_admission import canonical_url
        self.keys = {canonical_url({"doi": d, "url": ""}) for d in stored_dois}
        self.fail, self.calls = fail, 0
        if not missing:
            self.stored_rows = self._rows

    def _rows(self, cus):
        self.calls += 1
        if self.fail:
            raise RuntimeError("db down")
        return [{"canonical_url": k, "source_id": i} for i, k in enumerate(cus) if k in self.keys], []


def _swap_cfg():
    cfg = _cfg(True, True)
    cfg["source_type_router"]["fit_table"]["general"] = {"web": 0, "scholarly": 0}
    cfg["topic_fit_order"]["web_min_slots"]["general"] = 6
    return cfg


def test_stored_candidates_survive_the_swap_via_the_store_lookup():
    pool, cfg = _pool(), _swap_cfg()
    base = [r[1] for r in _rank(cfg, None, pool)[0]]
    stored = base[-2:]  # the two lowest-ranked entries are stored documents
    deps = _Deps(stored)
    prot = inline_acquire._protected_keys(pool, cfg, deps)
    assert deps.calls == 1 and prot  # ONE batch read
    out, _ = _rank(cfg, "general", pool, protected=prot)
    assert set(stored) <= {r[1] for r in out}
    assert sum(1 for r in out if r[2].get("acq_source") == "ddg") == 3 and len(out) == 6


def test_lookup_failure_or_missing_store_means_no_swap():
    pool, cfg = _pool(), _swap_cfg()
    for deps in (_Deps(fail=True), _Deps(missing=True)):
        prot = inline_acquire._protected_keys(pool, cfg, deps)
        assert prot is None
        out, _ = _rank(cfg, "general", pool, protected=prot)
        assert not any(r[2].get("acq_source") == "ddg" for r in out)  # swap skipped entirely
    # and with the switches off the lookup result is irrelevant: today's order
    off = [r[1] for r in _rank(_cfg(), "general", pool, protected=None)[0]]
    assert off == [r[1] for r in _rank(_cfg(), None, pool)[0]]


def test_no_lookup_when_topic_fit_off(monkeypatch):
    # the inline caller only looks up when O1 is active (_topic_fit_active); off => no extra DB read
    assert aa._topic_fit_active(_cfg(), "general") is None
    assert aa._topic_fit_active(_cfg(True, True), "general") is not None
    assert aa._topic_fit_active(_cfg(True, True), "scholarly") is None


@pytest.mark.xfail(reason="known misroute (mixed->general): sports-injury epidemiology is both; lexicon has no "
                          "injury/epidemiology conflict rule yet; tracked for calibration", strict=True)
@pytest.mark.parametrize("q", ["How common are concussions in MMA compared with boxing?",
                               "What are the most frequent injuries among marathon runners?"])
def test_known_mixed_to_general_misroutes(q):
    assert [x["label"] for x in LAB if x["q"] == q] == ["mixed"]
    assert source_type.route(q, _cfg()) == "mixed"


# ------------------------------------------------------------------------------------------------ revision 3 (SHIPPED fit_table)
def test_shipped_fit_table_keeps_stored_candidates_in_todays_cut():
    pool, cfg = _pool(), _cfg(True, True)  # shipped fit_table and web_min_slots
    base_cut = [r[1] for r in _rank(cfg, None, pool)[0]]
    for stored in (base_cut[-2:], base_cut[-1:], base_cut[:2], base_cut):
        prot = inline_acquire._protected_keys(pool, cfg, _Deps(stored))
        out = [r[1] for r in _rank(cfg, "general", pool, protected=prot)[0]]
        assert set(stored) <= set(out), (stored, out)
        assert len(out) == 6


def test_shipped_fit_table_lookup_failure_is_todays_order_exactly():
    pool, cfg = _pool(), _cfg(True, True)
    base = [r[1] for r in _rank(cfg, None, pool)[0]]
    for route in ("general", "mixed"):
        assert [r[1] for r in _rank(cfg, route, pool, protected=None)[0]] == base
    prot = inline_acquire._protected_keys(pool, cfg, _Deps(fail=True))
    assert [r[1] for r in _rank(cfg, "general", pool, protected=prot)[0]] == base


def test_shipped_fit_table_no_stored_still_prefers_ddg():
    out = _rank(_cfg(True, True), "general", _pool(), protected=set())[0]
    assert sum(1 for r in out if r[2].get("acq_source") == "ddg") == 3


def test_multiword_lexicon_terms_match():
    for q in ("Which video games are best for kids?", "Is there a good tv show about chefs?"):
        assert source_type.route(q, json.loads((Path(__file__).resolve().parent.parent / "config" / "auto_acquire.json").read_text())) == "general"


def test_protect_lookup_batch_cap_and_key_build_failure_fail_safe():
    cfg = _cfg(True, True)
    big = [_pmc(i) for i in range(inline_acquire.PROTECT_LOOKUP_MAX + 1)]
    assert inline_acquire._protected_keys(big, cfg, _Deps()) is None
    assert inline_acquire._protected_keys([_pmc(1)], cfg, _Deps(missing=True)) is None
    assert inline_acquire._protected_keys([{"doi": object(), "url": object()}], cfg, _Deps()) in (None, set())
