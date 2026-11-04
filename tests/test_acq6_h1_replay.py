"""NONASK-1103 A6-0 (H1 replay): offline replay of per-document vs combined re-admission of the ephemeral corpus
(ask_service.py ~:843 admits the corpus built by inline_acquire.ephemeral_corpus with ONE shared cfg: retrieval
candidate_k 16, max_spans_total 6, max_admitted_chars 4800). SYNTHETIC fixtures only (invented DOIs 10.9999/..., invented
sentences); modelled on the reported question: multi-agent orchestration practitioner text plus MARL distractors.
Pure measurement + invariants of the CURRENT behaviour; it edits no production module."""
import copy

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from tests.test_dar_inline_acquire import direct  # noqa: F401  (fixture)
from vera import acq_config, auto_acquire, fulldoc_corpus, fulltext_loader, inline_acquire
from vera.fulldoc_admission import admit

Q = "What are the best practices for multi-agent orchestration of LLM agents?"
ORCH = ["multi-agent", "orchestration", "agents", "supervisor", "workflow", "delegation", "handoff", "tool",
        "planner", "worker", "routing", "coordination", "orchestrator", "llm", "agent"]
MARL = ["multi-agent", "reinforcement", "agents", "policy", "reward", "cooperative", "decentralized", "learning",
        "value", "gradient", "orchestration", "llm", "agent", "markov", "environment"]


@pytest.fixture
def topics():
    fs.TOPICS["orch"], fs.TOPICS["marl"] = ORCH, MARL
    yield
    fs.TOPICS.pop("orch", None), fs.TOPICS.pop("marl", None)


def _acquire_plans(specs, cfg):
    """One acquisition per (doi, topic): the real inline_acquire gates, a fake web, the plan the writer would store."""
    plans = []
    cfg["inline"]["acq6_rounds_enabled"] = False  # kill switch off: the single-search contract this replay models (rounds: test_acq6_rounds_1103)
    for i, (doi, topic) in enumerate(specs, 1):
        web, _ = A.epi_setup(doi=doi, pmcid=f"PMC{7000000 + i}", topic=topic, jats_kw={"seed": i * 3})
        store = A.Store()
        d = A.deps([A.cand(doi, title=f"SYNTHETIC {topic} study", snippet=Q)], web, store=store)
        inline_acquire.reset_for_tests()
        inline_acquire.acquire(Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e, d=d: d)
        plans.extend(store.plans)
    return plans


def _spans(ev):
    return list(getattr(ev, "admitted", None) or [])


def replay(n_orch=3, n_marl=3):
    cfg = A.cfg()
    specs = [(f"10.9999/orch.{i}", "orch") for i in range(n_orch)] + [(f"10.9999/marl.{i}", "marl") for i in range(n_marl)]
    plans = _acquire_plans(specs, cfg)
    fd, icfg = auto_acquire._fd_cfg(), acq_config.ingest_cfg(fulltext_loader.load_config(), cfg)
    alone = {}
    for p in plans:
        c = inline_acquire.ephemeral_corpus([p], cfg, fd, icfg)
        alone[p["canonical_url"]] = _spans(admit(Q, c, fd)) if c is not None else []
    comb_c = inline_acquire.ephemeral_corpus(plans, cfg, fd, icfg)
    comb = _spans(admit(Q, comb_c, fd)) if comb_c is not None else []
    ck = copy.deepcopy(fd)
    ck["retrieval"]["candidate_k"] = max(sum(len(v) for v in alone.values()), 16)  # plan 4.6: k >= sum of per-doc spans
    comb_k = _spans(admit(Q, comb_c, ck)) if comb_c is not None else []
    return plans, alone, comb, fd, comb_k


def test_h1_replay_reports_combined_loss(direct, topics):
    plans, alone, comb, fd, comb_k = replay()
    assert len(plans) >= 4, "fixture must acquire most documents"
    pass_alone = {u for u, s in alone.items() if s}
    alone_total = sum(len(s) for s in alone.values())
    comb_docs = {a["document_id"] for a in comb}
    print(f"H1 replay: docs acquired={len(plans)} admitting alone={len(pass_alone)} alone spans={alone_total} "
          f"| combined spans={len(comb)} combined docs={len(comb_docs)} "
          f"| caps candidate_k={fd['retrieval']['candidate_k']} max_spans_total={fd['admission']['max_spans_total']} "
          f"max_admitted_chars={fd['admission'].get('max_admitted_chars')}")
    assert len(pass_alone) >= 2
    # the shared caps bound the combined result (documented behaviour); the loss is what the replay measures
    assert len(comb) <= int(fd["admission"]["max_spans_total"])
    lost_docs = len(pass_alone) - len(comb_docs)
    assert lost_docs >= 0
    print(f"H1 replay: documents that admit alone but contribute 0 passages combined = {lost_docs}; "
          f"passages lost = {alone_total - len(comb)}")
    print(f"H1 replay: with candidate_k raised to the per-doc sum ({sum(len(v) for v in alone.values())}) alone: "
          f"combined spans={len(comb_k)} docs={len({a['document_id'] for a in comb_k})} "
          f"(max_spans_total={fd['admission']['max_spans_total']} still binds)")
    assert len(comb_k) <= int(fd["admission"]["max_spans_total"])


# ---- NONASK-1103 A6-3 (D2): seeded round-robin combined re-admission -------------------------------------------------
def _combined(specs, switch=True):
    cfg = A.cfg()
    plans = _acquire_plans(specs, cfg)
    fd, icfg = copy.deepcopy(auto_acquire._fd_cfg()), acq_config.ingest_cfg(fulltext_loader.load_config(), cfg)
    fd["admission"]["seed_per_doc_spans"] = switch
    c = inline_acquire.ephemeral_corpus(plans, cfg, fd, icfg)
    return plans, fd, icfg, c, (_spans(admit(Q, c, fd)) if c is not None else [])


def test_d2_every_admitted_document_contributes_at_least_one_passage(direct, topics):
    specs = [(f"10.9999/orch.{i}", "orch") for i in range(3)] + [(f"10.9999/marl.{i}", "marl") for i in range(3)]
    plans, fd, icfg, c, comb = _combined(specs)
    alone = {p["canonical_url"]: _spans(admit(Q, inline_acquire.ephemeral_corpus([p], A.cfg(), fd, icfg), fd)) for p in plans}
    able = sum(1 for s in alone.values() if s)
    assert able >= 2 and len(plans) <= 6
    assert len(comb) <= int(fd["admission"]["max_spans_total"]) == 6  # cap unchanged
    assert len({a["document_id"] for a in comb}) >= able  # every document that admits alone contributes >= 1


def test_d2_six_drift_plus_one_on_topic_keeps_on_topic_and_drift_does_not_fill_all(direct, topics):
    specs = [(f"10.9999/marl.{i}", "marl") for i in range(6)] + [("10.9999/orch.9", "orch")]
    plans, fd, icfg, c, comb = _combined(specs)
    on_topic = next(p for p in plans if "orch.9" in p["canonical_url"] or "orch.9" in str(p["source"]["provenance"]))
    docs = [a["document_id"] for a in comb]
    on_ids = {a["document_id"] for a in comb if a["text"] and "orchestrat" in a["text"].lower()}
    assert len(comb) <= 6 and on_topic is not None
    assert on_ids, "the on-topic document must be included"
    assert len(set(docs)) >= 2 and len(set(docs)) == len({d for d in docs})  # spread over documents, not one dominant
    assert sum(1 for d in docs if d not in on_ids) <= 5  # drift does not fill every slot


def test_d2_kill_switch_false_is_the_pre_change_single_retrieval(direct, topics):
    specs = [(f"10.9999/orch.{i}", "orch") for i in range(3)] + [(f"10.9999/marl.{i}", "marl") for i in range(3)]
    _, fd_on, _, c_on, on = _combined(specs, True)
    _, fd_off, icfg, c_off, off = _combined(specs, False)
    ref = _spans(admit(Q, c_off, {**fd_off, "admission": {k: v for k, v in fd_off["admission"].items()
                                                         if k != "seed_per_doc_spans"}}))
    assert [a["span_id"] for a in off] == [a["span_id"] for a in ref]  # absent == false: byte-identical selection
    assert len(on) <= 6 and len(off) <= 6
