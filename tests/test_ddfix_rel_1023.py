"""DDFIX-1023/REL: discovery queries keep the question's concepts and drop generic idioms; off-topic candidates (lexical
false positives of the search) go behind relevant ones BEFORE the max_candidates_resolved / discovered_max cuts.

OFFLINE: fixed clock, fake search, fake HTTP (tests/acq_support.py); no network, DB or model. The five false positives
are the exact records of the owner's live reading list for Q (build 9b55c75, 2026-10-23); the relevant records and the
Europe PMC "status quo" hits are SYNTHETIC (DOIs 10.9999/..., fake PMC ids)."""
import json
from pathlib import Path

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from vera import acq_config, acq_discovered, arxiv_fulltext, auto_acquire, inline_acquire, search_providers as sp
from vera import search_query, wiki_ref_portal

ROOT = Path(__file__).resolve().parent.parent
Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
RUN = json.loads((ROOT / "config" / "vera_eval_run.json").read_text())["search"]
QT = RUN["query_terms"]
PROFILE = json.loads((ROOT / "config" / "auto_acquire.json").read_text())["search"]["openalex_profile"]
GENERIC = {"de", "facto", "status", "quo", "best", "practices", "track"}

# the owner's live reading list (verbatim titles and identifiers): all were "Not checked: too many candidates"
LIVE_FP = [
    {"provider": "arxiv", "url": "https://arxiv.org/abs/2512.02349v2", "doi": None,
     "title": "ACM COMPUTE 2025 Best Practices Track Proceedings", "snippet": ""},
    {"provider": "openalex", "url": "https://doi.org/10.1177/01492063241226918", "doi": "10.1177/01492063241226918",
     "title": "The Status of Status Research: A Review of the Types, Functions, Levels, and Audiences", "snippet": ""},
    {"provider": "openalex", "url": "https://doi.org/10.3790/vaw.2025.1457703", "doi": "10.3790/vaw.2025.1457703",
     "title": "The U.S. Dollar Dominance: Origins, Status Quo, and Implications for Europe", "snippet": ""},
    {"provider": "arxiv", "url": "https://arxiv.org/abs/2409.18288v3", "doi": None,
     "title": "The track-length extension fitting algorithm for energy measurement of interacting particles in liquid "
              "argon TPCs", "snippet": ""},
    {"provider": "arxiv", "url": "https://arxiv.org/abs/2607.16922v1", "doi": None,
     "title": "Pedestrian Archetypes Extension -- More Pedestrian Models for Autonomous Vehicle Safety Testing",
     "snippet": ""},
]
REL_OA = [  # SYNTHETIC relevant reading-list records (no OA signal: P0DTA put them behind every routable hit)
    {"provider": "openalex", "doi": "10.9999/rel.oa1", "url": "https://doi.org/10.9999/rel.oa1",
     "title": "Orchestrating multi-agent LLM systems: agent roles and coordination patterns",
     "snippet": "We survey orchestrator, planner, critic and executor agents in multi-agent frameworks."},
    {"provider": "openalex", "doi": "10.9999/rel.oa2", "url": "https://doi.org/10.9999/rel.oa2",
     "title": "Agent archetypes for multi-agent orchestration in practice", "snippet": ""},
]
REL_EPMC = ("10.9999/rel.epmc", "PMC7400001",
            "Role-specialised multi-agent orchestration of large language model agents in clinical workflows")
FP_EPMC = ["Status quo bias in shared decision making", "Best practices for tracking medication adherence",
           "De facto standards in clinical data exchange", "The status of nurse practitioners in rural care",
           "Practice archetypes among general practitioners: a cluster analysis",
           "Quo vadis, evidence-based practice? A status report"]


def _epmc(doi, pmcid, title):
    return {"url": f"https://doi.org/{doi}", "title": title, "snippet": "", "doi": doi, "year": 2024,
            "provider": "europepmc", "pmcid": pmcid,
            "oa": {"is_oa": True, "has_oa_location": True, "licence": None, "pmcid": pmcid, "epmc_ft": True}}


def _search_hits():
    fps = [_epmc(f"10.9999/fp.epmc{i}", f"PMC75000{i:02d}", t) for i, t in enumerate(FP_EPMC)]
    return fps + [_epmc(*REL_EPMC)]  # the relevant hit is LAST in the providers' order


# ---------------------------------------------------------------------------------------------- 1. queries
def test_query_terms_drop_generic_idioms_and_keep_the_concepts():
    qt = search_query.build_query_terms(Q, QT)
    assert not GENERIC & set(qt.extra)
    assert {"multi-agent", "orchestration", "agent", "archetypes"} <= set(qt.extra)
    assert qt.runs == (("agent", "archetypes"), ("multi-agent", "orchestration"))


def test_acquisition_and_query_is_the_compound_concept_not_the_first_four_words():
    assert sp.build_openalex_search(Q, QT, profile=PROFILE) == (
        "agent AND archetypes AND multi-agent AND orchestration", "extra_only+and")
    epmc = sp.build_europepmc_query(Q, QT, PROFILE, "OPEN_ACCESS:y")
    assert epmc == "(agent AND archetypes AND multi-agent AND orchestration) AND OPEN_ACCESS:y"


@pytest.mark.parametrize("q", [
    lambda: sp.build_openalex_search(Q, QT)[0],
    lambda: sp.build_arxiv_query_info(Q, RUN["arxiv"]["query"], QT)[0],
    lambda: sp.build_ddg_queries(Q, QT, RUN["duckduckgo_lite"])[0]["query"].split(" (site:")[0],
    lambda: wiki_ref_portal.build_query(Q, json.loads((ROOT / "config" / "wiki_ref_portal.json").read_text())),
])
def test_no_provider_query_carries_an_isolated_generic_word(q):
    words = set(q().lower().replace("(", " ").replace(")", " ").replace("all:", "").split())
    assert not GENERIC & words, q()
    assert "multi-agent" in words and "orchestration" in words


def test_or_retry_and_reading_list_stay_or_queries():  # D3: broader than the AND query, never a strict AND
    assert sp.build_openalex_search(Q, QT)[0] == "(agent OR archetypes OR multi-agent OR orchestration)"
    assert sp.build_arxiv_query_info(Q, RUN["arxiv"]["query"], QT)[0].startswith(
        "(all:agent OR all:archetypes OR all:multi-agent OR all:orchestration) AND submittedDate:")


def test_questions_without_generic_words_or_compounds_keep_their_previous_query():
    q = "What are the key differences between boxing and mma?"
    assert sp.build_openalex_search(q, QT) == ("(key OR differences OR boxing OR mma)", "extra_only")
    assert search_query.build_query_terms("status quo", QT).extra == ("status", "quo")  # only an idiom: kept


# ---------------------------------------------------------------------------------------------- 2. relevance
def test_live_false_positives_rank_below_relevant_records_and_only_zero_matches_are_off_topic():
    kc = search_query.key_concepts(Q, extra_generic=PROFILE["generic_terms"])
    rel = [r["title"] for r in REL_OA] + [REL_EPMC[2],  # verifier D4: these must never be dropped
           "A taxonomy of LLM agent roles and coordination patterns for software engineering",
           "Coordinating teams of autonomous language-model agents: planner, critic and executor",
           "AutoGen: enabling next-gen LLM applications via multiagent conversation"]
    tiers = search_query.relevance_tiers(kc, [r["title"] for r in LIVE_FP] + rel)
    assert tiers[:5] == [2, 2, 2, 2, 1]  # four match no concept (dropped); "Pedestrian Archetypes" is weak (ordered last)
    assert tiers[5:] == [0] * len(rel)
    assert search_query.relevance_tiers(kc, ["t"]) == [0]  # too short to judge: neutral
    assert search_query.relevance_tiers(kc, [r["title"] for r in LIVE_FP[:4]]) == [0] * 4  # no match in the pool


def test_covid_contact_tracing_record_is_never_dropped():
    q = "What is the track record of contact tracing in controlling COVID-19 outbreaks?"
    kc = search_query.key_concepts(q, extra_generic=PROFILE["generic_terms"])
    pool = ["Controlling COVID-19 outbreaks in nursing homes with contact tracing",
            "Effectiveness of digital contact tracing apps for SARS-CoV-2: a systematic review",
            "Track-length estimation in particle detectors using deep networks"]
    assert search_query.relevance_tiers(kc, pool) == [0, 0, 2]


def test_relevant_candidates_outrank_lexical_false_positives_before_the_cut():
    acq = A.cfg()
    disc, _held = acq_discovered.discovered_candidates(LIVE_FP + REL_OA, {**acq, "inline": {
        **acq.get("inline", {}), "use_discovered": True}}, Q)
    cands = sp.merge_sources(disc, _search_hits())
    drops = {}
    ranked = auto_acquire.rank_candidates(Q, cands, acq, set(), auto_acquire._fd_cfg(), lambda c: None, drops=drops)
    dois = [r[1] for r in ranked]
    rel = {"10.9999/rel.oa1", "10.9999/rel.oa2", REL_EPMC[0]}
    assert set(dois[:3]) == rel, dois
    assert dois[0] == REL_EPMC[0]  # routable full text first among the relevant (P0DTA order kept within the group)
    assert set(dois[3:]) == {"10.48550/arxiv.2607.16922", "10.9999/fp.epmc4"}  # weak matches, behind every relevant
    cut = {c.get("doi") for c in cands if drops.get(id(c)) == acq_config.reason(acq, "defer_cap")}
    assert not rel & cut


def test_off_topic_records_go_last_before_the_discovered_max_cut():
    acq = A.cfg()
    acq.setdefault("inline", {})["discovered_max"] = 2
    cands, held = acq_discovered.discovered_candidates(LIVE_FP + REL_OA, acq, Q)
    assert [c["doi"] for c in cands] == ["10.9999/rel.oa1", "10.9999/rel.oa2"]
    irr = [h for h in held if h["reason"] == acq_config.reason(acq, "irrelevant")]
    assert len(irr) == 4 and all(h["outcome"] == "dropped" for h in irr)  # zero matches: never "too many candidates"
    assert [h["outcome"] for h in held if h not in irr] == ["deferred"]  # the weak "Pedestrian Archetypes" record


def test_off_topic_search_hits_are_dropped_as_irrelevant_before_the_cut():
    acq = A.cfg()
    cands = _search_hits()
    drops = {}
    ranked = auto_acquire.rank_candidates(Q, cands, acq, set(), auto_acquire._fd_cfg(), lambda c: None, drops=drops)
    assert [r[1] for r in ranked] == [REL_EPMC[0], "10.9999/fp.epmc4"]  # relevant first; the weak "archetypes" hit kept
    assert sorted(drops.values()) == [acq_config.reason(acq, "irrelevant")] * (len(FP_EPMC) - 1)


# ---------------------------------------------------------------------------------------------- 3. admission stems
def test_admission_question_stems_share_the_generic_idiom_set():
    from vera import fulldoc_admission as fa
    from tests.test_fix2_verifier_1022 import _icfg_n
    icfg, n = _icfg_n()
    q = fa._q_stems(Q, icfg, n)
    assert not {"facto", "statu", "quo", "pract"} & q and {"agent", "arche", "multi", "orche"} <= q
    assert fa.coverage_ok({"agent", "multi", "orche"}, q, 0.7) and not fa.coverage_ok({"agent"}, q, 0.7)


@pytest.mark.parametrize("title,span,admitted", [
    ("Orchestrating multi-agent LLM systems",
     "Multi-agent orchestration frameworks assign agent archetypes such as planner, critic and executor roles to "
     "coordinate the work of each agent.", True),
    ("The U.S. Dollar Dominance: Origins, Status Quo, and Implications for Europe",
     "The status quo of the international monetary system rests on de facto dollar dominance, with implications for "
     "European best practices in reserve management.", False),
])
def test_admission_admits_the_relevant_text_and_still_rejects_the_off_topic_status_quo_paper(title, span, admitted):
    import copy
    from vera import fulldoc_admission as fa
    from tests.test_fix2_verifier_1022 import CFG, _Corpus
    b = fa.admit(Q, _Corpus(title, [span]), copy.deepcopy(CFG))
    assert bool(b.admitted) is admitted, b.rejected


# ---------------------------------------------------------------------------------------------- 3. inline end to end
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq = A.cfg()
    acq["inline"]["enabled"] = True
    acq["inline"]["use_discovered"] = True
    yield acq
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def test_inline_relevant_sources_are_selected_and_fetched_not_cut_as_too_many(env):
    doi, pmcid, title = REL_EPMC
    oa1 = REL_OA[0]["doi"]
    web = A.Web({doi: A.epmc_record(doi, pmcid=pmcid, title=title), oa1: A.epmc_record(oa1, pmcid="PMC7400002")},
                {pmcid: fs.jats("rel", "epi", doi), "PMC7400002": fs.jats("rel2", "epi", oa1)})
    d = A.deps(_search_hits(), web)
    out = inline_acquire.acquire(Q, env, env=A.ENV_ON, clock=Clock(), deps_factory=lambda a, e: d,
                                 discovered=LIVE_FP + REL_OA)
    pub = out.public()
    by = {c.get("doi"): c for c in pub["candidates"] if c.get("doi")}
    for rd in (doi, oa1, REL_OA[1]["doi"]):  # selected: evaluated (whatever the full-text gates then decide)
        assert by[rd]["status_reason"] != "candidate_cap" and by[rd]["outcome"] not in ("dropped", "not_evaluated"), \
            by[rd]
    assert any(f"/{pmcid}/fullTextXML" in u for u in web.urls)
    assert any("/PMC7400002/fullTextXML" in u for u in web.urls)
    live = {r["url"] for r in LIVE_FP[:4]}  # the four that match no concept of the question
    reached = [e for e in pub["discovered"] if e.get("url") in live and e.get("gate_a", {}).get("decision") == "fetch"]
    assert not reached  # none of the live false positives took a slot from a relevant source


# ---------------------------------------------------------------------------------------------- 4. generalisation
# VER-REL generalisation set (work/ddfix-1023/ver-rel/scripts/gen.py): the acquisition AND query keeps every distinctive
# concept (D1), idioms go but bare "status"/"best"/"practice"/"track" stay where they carry meaning (D2), filler verbs
# ("affect", "drives", "improve", "effective", "interact", "controlling") are never AND terms.
GEN = [
    ("How does HIV status affect tuberculosis treatment outcomes?", "hiv AND status AND tuberculosis AND treatment"),
    ("What is the track record of contact tracing in controlling COVID-19 outbreaks?",
     "contact AND tracing AND covid-19 AND outbreaks"),
    ("What are best practices in code review for AI coding assistants?", '"ai coding assistants" AND "code review"'),
    ("How do AI coding assistants affect developer productivity?",
     '"ai coding assistants" AND "developer productivity" AND (affect)'),
    ("What is the status of measles vaccination coverage in sub-Saharan Africa?",
     "status AND measles AND vaccination AND sub-saharan"),
    ("How does climate change affect coral reef bleaching?", "climate AND change AND coral AND reef"),
    ("What practices improve reproducibility in machine learning research?",
     "practices AND reproducibility AND machine AND learning"),
    ("What drives biodiversity loss in freshwater ecosystems?", "biodiversity AND loss AND freshwater AND ecosystems"),
    ("How effective is contact tracing for Ebola?", "contact AND tracing AND ebola"),
    ("What is the state of the art in long-term sea-level rise projections?",
     "long-term AND sea-level AND rise AND projections"),
    ("What are the best treatments for drug-resistant tuberculosis?", "treatments AND drug-resistant AND tuberculosis"),
    ("How do socioeconomic status and air pollution exposure interact in asthma?",
     "socioeconomic AND air AND pollution AND asthma"),
]
FILLER = {"affect", "drives", "improve", "effective", "interact", "controlling"}
IDIOM_WORDS = {"de", "facto", "quo", "state", "art", "record"}


@pytest.mark.parametrize("q,want", GEN)
def test_generalisation_acquisition_and_query(q, want):
    got = sp.build_openalex_search(q, QT, profile=PROFILE)[0]
    assert got == want
    if '"' not in got:  # no configured core phrase: filler and idiom words are never AND-ed
        assert not (FILLER | IDIOM_WORDS) & set(got.split())


@pytest.mark.parametrize("q,_want", GEN)
def test_generalisation_or_retry_is_broader_and_reading_list_is_not_a_strict_and(q, _want):
    and_q = sp.build_openalex_search(q, QT, profile=PROFILE)[0]
    or_q = sp.build_openalex_search(q, QT)[0]
    ax = sp.build_arxiv_query_info(q, RUN["arxiv"]["query"], QT)[0].split(" AND submittedDate")[0]
    if '"' in and_q:  # configured core phrases: unchanged behaviour
        return
    assert " AND " not in or_q and " AND " not in ax
    assert set(and_q.split(" AND ")) <= set(or_q.strip("()").split(" OR "))
