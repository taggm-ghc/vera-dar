"""GOLDEN-XFAIL-1106: mid-term = midterm in admission and query concepts; lexical-resource titles are weak for physical questions."""
from vera import fulldoc_corpus, search_query as sq
from vera import fulldoc_admission as fa

Q = "What are implications of the US 2026 Midterm elections?"
TITLE = "A word association norms dataset for English colour terms"


def _stems(text):
    r = fulldoc_corpus.load_config()["admission"]["relevance"]
    return fa._stems(text, r, r["stem_len"])


def test_midterm_variants_tokenise_alike():
    base = _stems("the midterm elections")
    assert _stems("the mid-term elections") == base
    assert _stems("the mid term elections") == base
    assert _stems("the mid-terms elections") == _stems("the midterms elections")


def test_midterm_variant_passage_admitted_like_midterm():
    r = fulldoc_corpus.load_config()["admission"]["relevance"]
    n = r["stem_len"]
    t = "Pre-election surveys of mid-term elections suggest shifting party support."
    assert fa._matched(Q, t, r, n)[0] == 2
    assert fa._relevance(Q, t, r, n) >= 0.34


def test_midterm_variants_are_one_query_concept():
    for q in ("mid-term elections", "mid term elections", "midterm elections"):
        kc, _ = sq.key_concepts(q, sq._default_terms_cfg())
        assert "midterm" in kc and "mid-term" not in kc and "mid" not in kc


def test_unrelated_hyphenated_compounds_unchanged():
    for q in ("long-term effects of self-driving cars", "peer-review of long-term studies"):
        kc, _ = sq.key_concepts(q, sq._default_terms_cfg())
        assert "midterm" not in kc


def test_lexical_resource_title_demoted_for_physical_question():
    kc = sq.key_concepts("What color is the sky?")
    rel, lex = sq.relevance_tiers(kc, ["Rayleigh scattering and the colour of the daytime sky", TITLE])
    assert rel == 0 and lex == 1


def test_lexical_resource_title_kept_when_question_asks_for_it():
    kc = sq.key_concepts("Which word association norms datasets cover color terms?")
    assert sq.relevance_tiers(kc, [TITLE, "Colour naming across languages and word association norms in English"])[0] == 0
