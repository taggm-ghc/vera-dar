"""RELEVANCE-R1R4-1106: compound weight (R1), framing words in news terms (R2), arXiv AND (R3), colour = color (R4)."""
from vera import fulldoc_corpus, query_terms, query_variants_acq as qv, search_providers as sp, search_query as sq
from vera import fulldoc_admission as fa

QA = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"


def _cfg():
    return sq._default_terms_cfg()


def test_r1_component_not_above_compound():
    kc, w = sq.key_concepts(QA, _cfg())
    assert w["agent"] <= w["multi-agent"] and w["multi-agent"] > w["orchestration"] == w["archetypes"]


def test_r1_variants_of_the_compound_are_one_concept():
    for q in ("multiagent orchestration archetypes", "multi agent orchestration archetypes"):
        kc, _ = sq.key_concepts(q, _cfg())
        assert "multi-agent" in kc and "multiagent" not in kc and "multi" not in kc


def test_r1_marl_not_fully_relevant():
    kc = sq.key_concepts(QA, _cfg())
    marl, ok = sq.relevance_tiers(kc, ["Transformer-based multi-agent deep reinforcement learning considering agents' order of action decisions",
                                       "The Orchestration of Multi-Agent Systems: Architectures, Protocols and Agent Archetypes"])
    assert marl != 0 and ok == 0
    assert sq.relevance_tiers(kc, ["Container orchestration for cloud workloads"] * 2)[0] != 0


def test_r2_framing_gone_from_news_terms():
    t = query_terms.extract_terms(QA)
    for bad in ("facto", "status", "quo", "best", "practices"):
        assert bad not in t, t
    assert "orchestration" in t or "archetypes" in t


def test_r2_plain_status_word_kept():
    assert "status" in query_terms.extract_terms("What is the vaccination status of children?")


def test_r3_arxiv_ands_compound_and_drops_bare_component():
    q = {"min_term_chars": 3, "max_terms": 8, "operator": "OR"}
    body, _ = sp.build_arxiv_query_info(QA, q, _cfg())
    assert 'all:"multi-agent"' in body and " AND " in body
    assert "all:agent " not in body and "all:agent)" not in body and "OR all:multi-agent" not in body


def test_r3_no_bare_one_word_variants():
    for v in qv.variants(QA):
        assert len(v.split()) > 1 or "-" in v, v
    assert not any(v in ("agent", "color") for v in qv.variants("What color is the sky?"))


def test_r4_colour_matches_color():
    assert sq.normalize_spelling("The Colours of the sky; colouring") == "The colors of the sky; coloring"
    kc = sq.key_concepts("What color is the sky?", _cfg())
    assert sq.matched_concepts(kc, "The colour of the sky depends on Rayleigh scattering.") >= {"color", "sky"}
    icfg = fulldoc_corpus.load_config()["admission"]["relevance"]
    n = icfg["stem_len"]
    assert fa._stems("The colour of the sky", icfg, n) == fa._stems("The color of the sky", icfg, n)
    hits, _ = fa._matched("What color is the sky?", "The colour of the sky depends on Rayleigh scattering.", icfg, n)
    assert hits == 2


def test_r1_llm_agent_papers_stay_relevant_marl_demoted():
    kc = sq.key_concepts(QA, _cfg())
    t = sq.relevance_tiers(kc, ["A taxonomy of LLM agent roles and coordination patterns for software engineering",
                                "AutoGen: enabling next-gen LLM applications via multiagent conversation",
                                "Augmenting the action space with conventions to improve multi-agent cooperation in Hanabi",
                                "Multi-agent reinforcement learning for orchestration of traffic signals"])
    assert t[:2] == [0, 0] and t[2] != 0 and t[3] != 0


MARL_CASES = [  # QA F1: (question, an on-topic MARL title that matches >= 2 of the question's concepts)
    ("What are the most effective algorithms for multi-agent reinforcement learning?",
     "QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning algorithms"),
    ("How does MARL handle credit assignment in multi-agent cooperative games?",
     "Credit assignment in cooperative multi-agent reinforcement learning: a survey of MARL games"),
    ("multi-agent policy gradient methods for traffic signal control",
     "Multi-agent policy gradient methods for adaptive traffic signal control"),
    ("How do multi-agent systems coordinate in swarm robotics?",
     "Decentralized multi-agent reinforcement learning to coordinate robot swarms"),
]


def test_r1_marl_questions_keep_marl_papers():  # RL vocabulary in the question or no LLM sense: no demotion
    for q, title in MARL_CASES:
        kc = sq.key_concepts(q, _cfg())
        assert sq.relevance_tiers(kc, [title, title])[0] == 0, (q, kc)


def test_r2_status_quo_cue_kept_for_political_not_for_framing():  # QA F2
    from vera import temporality
    assert not temporality.detect(QA).current
    assert temporality.detect("Will the status quo in the Taiwan Strait hold?").current
    assert temporality.detect("What is the status quo on abortion law in the US?").current


def test_r3_plural_agents_question_keeps_llm_taxonomy_tier0():  # QA F3
    kc = sq.key_concepts("How do LLM multi-agent frameworks like AutoGen orchestrate agents?", _cfg())
    assert sq.relevance_tiers(kc, ["A taxonomy of LLM agent roles and coordination patterns for software engineering"] * 2)[0] == 0


def test_r2_bare_status_quo_best_practices_stays_current_but_framing_question_not_routed():
    from vera import lane_router, temporality
    assert temporality.detect("status quo best practices for password storage").current
    assert not lane_router.reporting_cue(QA, QA)
    assert lane_router.reporting_cue("Will the status quo in the Taiwan Strait hold?", "")


def test_r1_incidental_compound_does_not_invert_order():  # QA F4
    for q, ontopic, generic in (
            ("How can hospitals reduce emergency department wait times, for example with multi-agent scheduling?",
             "Interventions to reduce ED crowding and wait times in hospitals", "Multi-agent systems in logistics"),
            ("What are the effects of microfinance on poverty in a multi-agent economy model?",
             "The impact of microfinance on poverty: evidence from randomized trials", "Multi-agent systems")):
        kc = sq.key_concepts(q, _cfg())
        assert kc[1]["multi-agent"] == 1, (q, kc)
        t = sq.relevance_tiers(kc, [ontopic, generic])
        assert t[0] <= t[1], (q, t)
