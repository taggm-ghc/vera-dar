"""Consolidation of near-duplicate derived domains (vera.dar_consolidate + DomainIdentifier.merge_alias).
Synthetic fixtures only (tests/fixtures/dar); no DB, network or env. Uses DomainIdentifier directly, so the plan is
computed by the test and not auto-applied by DarCorpus."""
import copy
import itertools
import json
import random
from pathlib import Path

from vera.dar_consolidate import plan_consolidation
from vera.domain_identify import DomainIdentifier
from vera.domain_registry import InMemoryDomainRegistry

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT.parent / "config" / "domain_routing.json").read_text())
FX = ROOT / "fixtures" / "dar"


def _docs():
    ds = json.loads((FX / "demo9.json").read_text()) + json.loads((FX / "eco1.json").read_text())
    return {d["doc_id"]: d["content"] for d in ds}


def _split_cfg():
    """Identification config where every document founds its own derived domain (raised association thresholds)."""
    ic = copy.deepcopy(CFG["identification"])
    ic.update(assoc_min_sim=0.99, sig_min_shared=999)
    return ic


def _ingest(order, docs, ic):
    reg = InMemoryDomainRegistry()
    ident = DomainIdentifier(reg, ic)
    founded = {}
    for key in order:
        before = {d.domain_id for d in reg.list_domains()}
        ident.add_document(key, docs[key], "t")
        new = [d.domain_id for d in reg.list_domains() if d.domain_id not in before]
        if new:
            assert len(new) == 1
            founded[new[0]] = key
    return reg, ident, founded


def _partition(plan, founded):
    """frozenset of frozensets of founding doc keys per canonical group."""
    groups = {i: {k} for i, k in founded.items()}
    for p in plan:
        groups[p["canonical"]].add(founded[p["alias"]])
        groups.pop(p["alias"], None)
    return {frozenset(g) for g in groups.values()}


DEMO = {frozenset({"D1-1", "D1-2", "D1-3"}), frozenset({"D2-1", "D2-2", "D2-3"}),
        frozenset({"D3-1", "D3-2", "D3-3"}), frozenset({"E-1"})}


def test_near_duplicates_merge_in_every_order_and_subjects_stay_apart():
    docs = _docs()
    ic = _split_cfg()
    keys = sorted(docs)
    seen = set()
    for seed in range(6):
        order = list(keys)
        random.Random(seed).shuffle(order)
        reg, ident, founded = _ingest(order, docs, ic)
        assert len(reg.list_domains()) == 10
        plan = plan_consolidation(ident, reg, ic)
        part = _partition(plan, founded)
        assert part == DEMO, (seed, sorted(map(sorted, part)))  # epi, ecology and the rest stay apart
        for p in plan:
            assert p["canonical"] < p["alias"] and p["shared"] >= ic["consolidate_min_shared"] and p["cos"] >= ic["consolidate_min_cos"]
        assert plan == sorted(plan, key=lambda p: (p["alias"], p["canonical"]))
        for p in plan:
            ident.merge_alias(p["alias"], p["canonical"])
        canon = {d.domain_id for d in reg.list_domains()} - set(ident.aliases)
        assert len(canon) == 4
        assert plan_consolidation(ident, reg, ic) == []  # already consolidated: nothing left to plan
        seen.add(frozenset(frozenset(founded[i] for i in [c] + [a for a, cc in ident.aliases.items() if cc == c]) for c in canon))
    assert seen == {frozenset(DEMO)}


def test_one_topic_every_permutation_gives_one_canonical_domain():
    docs = _docs()
    ic = _split_cfg()
    for perm in itertools.permutations(["D1-1", "D1-2", "D1-3"]):
        reg, ident, founded = _ingest(perm, docs, ic)
        plan = plan_consolidation(ident, reg, ic)
        for p in plan:
            ident.merge_alias(p["alias"], p["canonical"])
        assert {d.domain_id for d in reg.list_domains()} - set(ident.aliases) == {1}  # canonical = lowest id
        assert ident.aliases == {2: 1, 3: 1}


def test_disabled_or_unsplit_plans_nothing():
    docs = _docs()
    ic = _split_cfg()
    reg, ident, _ = _ingest(sorted(docs), docs, ic)
    off = dict(ic, consolidate_enabled=False)
    assert plan_consolidation(ident, reg, off) == []
    assert plan_consolidation(ident, reg, dict(ic, consolidate_enabled="true")) == []  # only literal True enables
    # shipped identification: the same subjects already share domains, nothing to consolidate
    reg2, ident2, _ = _ingest(sorted(docs), docs, copy.deepcopy(CFG["identification"]))
    assert plan_consolidation(ident2, reg2, CFG["identification"]) == []


def test_merge_alias_semantics_idempotent_and_classify_never_returns_alias():
    docs = _docs()
    ic = _split_cfg()
    reg, ident, founded = _ingest(sorted(docs), docs, ic)
    plan = plan_consolidation(ident, reg, ic)
    for p in plan:
        assert ident.merge_alias(p["alias"], p["canonical"]) is True
    snap = {k: dict(v) for k, v in ident._profiles.items()}
    for p in plan:
        assert ident.merge_alias(p["alias"], p["canonical"]) is False  # idempotent
    assert ident._profiles == snap
    assert isinstance(ident.aliases, dict) and ident.aliases is not ident._aliases  # read-only copy
    ident.aliases[999] = 1
    assert 999 not in ident.aliases
    n_rows = len(reg.list_domains())
    assert n_rows == 10  # rows are never deleted
    for key in sorted(docs):
        for kind in ("question", "chunk"):
            for hit in ident.classify(docs[key][:600], kind):
                assert hit["domain_id"] not in ident.aliases
    # the canonical profile now holds the alias term sums
    p0 = plan[0]
    assert all(ident._profiles[p0["canonical"]].get(t, 0) >= w - 1e-12 for t, w in ident._profiles[p0["alias"]].items())
    # a later document about the same subject joins the canonical profile, not the alias
    before = sum(ident._profiles[p0["canonical"]].values())
    alias_before = sum(ident._profiles[p0["alias"]].values())
    ident._add_to_profile(p0["alias"], {"zzterm": 1.0})
    assert sum(ident._profiles[p0["canonical"]].values()) == before + 1.0
    assert sum(ident._profiles[p0["alias"]].values()) == alias_before


def test_real_style_distinct_forecasting_subjects_are_not_merged():
    """epidemic vs ecological forecasting (demo D1 vs E-1): similar vocabulary, different subjects."""
    docs = _docs()
    ic = _split_cfg()
    reg, ident, founded = _ingest(["D1-1", "E-1", "D1-2", "D1-3"], docs, ic)
    part = _partition(plan_consolidation(ident, reg, ic), founded)
    assert part == {frozenset({"D1-1", "D1-2", "D1-3"}), frozenset({"E-1"})}
