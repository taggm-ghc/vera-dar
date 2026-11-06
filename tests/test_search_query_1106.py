"""MIDTERMS-WP3-1106: years kept as concepts and query terms; framing words are filler; news intent makes event nouns
and years central. Offline: no DB, network or model."""
from vera import search_query as sq

Q = "What are implications of the US 2026 Midterm elections?"


def test_year_kept_and_framing_dropped():
    kc, w = sq.key_concepts(Q)
    assert "2026" in kc and "implications" not in kc
    assert w["2026"] < w["elections"] == w["midterm"]  # F1-QUERY-1106: a bare year is not central


def test_part_query_keeps_year():
    assert "2026" in sq.part_query(Q, []).split()


def test_diagnosis_test4_tiers():
    kc = sq.key_concepts(Q)
    t = sq.relevance_tiers(kc, [
        "2026 midterm elections results and analysis of the House races across the country this year",
        "Ethical implications of business decisions in modern corporate governance and management practice",
        "Drawing a Map of Elections: districts and boundaries in several national voting systems today"])
    assert t[0] == 0 and t[1] >= 1


def test_technical_question_regression():
    q = "How do AI coding assistants affect developer productivity?"
    kc, w = sq.key_concepts(q)
    assert "affect" not in kc and "developer productivity" in kc
    assert max(w.values()) == 1  # no news intent: no central concept appears
    t = sq.relevance_tiers((kc, w), [
        "Developer productivity with AI coding assistants: a randomized controlled trial of experienced engineers",
        "Pedestrian archetypes in urban planning and street design for crowded city centres everywhere",
    ])
    assert t[0] == 0 and t[1] == 2


def test_year_in_non_news_question_is_a_term():
    kc, _ = sq.key_concepts("What did the 2019 study on code review find?")
    assert "2019" in kc
