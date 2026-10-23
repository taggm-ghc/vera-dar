"""DDFIX-1023/INTEG D6 (verifier of REL rev2): an unjudged short title (< MIN_JUDGE_WORDS words) is tier 0 only when it
matches a central concept, else tier 1, so short weak or off-topic routable records never sort ahead of relevant
non-routable ones at either cut. Probe adapted from work/ddfix-1023/ver-rel/scripts/p2.py. OFFLINE; SYNTHETIC records."""
import pytest

from tests import acq_support as A
from vera import acq_config, acq_discovered, auto_acquire as aa, search_query

Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
WEAK = [("long weak title", "Pedestrian Archetypes Extension -- More Pedestrian Models for Autonomous Vehicle Safety Testing"),
        ("short weak title (<4 words)", "Pedestrian Archetypes Extension"),
        ("weak, orchestration only", "Container orchestration for cloud workloads"),
        ("short off-topic title", "Status Quo Bias")]


def _epmc(i, title):  # routable full text
    d, p = f"10.9999/w{i}", f"PMC76000{i:02d}"
    return {"url": f"https://doi.org/{d}", "title": title, "snippet": "", "doi": d, "year": 2024, "provider": "europepmc",
            "pmcid": p, "oa": {"is_oa": True, "has_oa_location": True, "licence": None, "pmcid": p, "epmc_ft": True}}


def _rel(i):  # relevant, no OA signal (not routable)
    d = f"10.9999/r{i}"
    return {"provider": "openalex", "doi": d, "url": f"https://doi.org/{d}",
            "title": f"Orchestrating multi-agent LLM systems study {i}: agent roles and coordination", "snippet": ""}


ACQ = A.cfg()
N = int(acq_config.bound(ACQ, "max_candidates_resolved"))


def _pool(weak, nrel):
    return [_epmc(j, weak if j == 0 else f"{weak} {j}") for j in range(3)] + [_rel(i) for i in range(nrel)]


@pytest.mark.parametrize("label,weak", WEAK)
@pytest.mark.parametrize("nrel", [N - 2, N, N + 2])
def test_rank_cut_weak_records_never_displace_relevant_ones(label, weak, nrel):
    ranked = aa.rank_candidates(Q, _pool(weak, nrel), ACQ, set(), aa._fd_cfg(), lambda c: None, drops={})
    dois = [r[1] for r in ranked]
    n_rel = sum(d.startswith("10.9999/r") for d in dois)
    assert n_rel == min(nrel, N), (label, nrel, dois)
    if nrel >= N:
        assert n_rel == len(dois), (label, "a weak record was selected although relevant ones filled every slot", dois)
    else:
        assert dois[:nrel] == [f"10.9999/r{i}" for i in range(nrel)], (label, "relevant records not selected first", dois)


@pytest.mark.parametrize("label,weak", WEAK)
def test_discovered_max_cut_puts_relevant_records_first(label, weak):
    n = acq_discovered.discovered_max(ACQ)
    cands, _held = acq_discovered.discovered_candidates(_pool(weak, n + 2), ACQ, question=Q)
    got = [c.get("doi") for c in cands][:n]
    assert got and all(str(d).startswith("10.9999/r") for d in got), (label, got)


def test_short_texts_are_never_dropped_and_tier0_only_on_a_central_concept():
    kc = search_query.key_concepts(Q)
    t = search_query.relevance_tiers(kc, ["Pedestrian Archetypes Extension", "Status Quo Bias",
                                          "Multi-agent orchestration", _rel(0)["title"]])
    assert t[0] == 1 and t[1] == 1 and t[2] == 0 and t[3] == 0, t
