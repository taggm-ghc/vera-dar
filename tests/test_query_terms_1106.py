"""MIDTERMS-WP2-1106 (RC3) + privacy review REVISE (work/WP2-PRIVACY-REVIEW-1106/review.md).

extract_terms keeps lexicon common nouns and allowlisted acronyms (Deny-All); personal names, unlisted all-caps tokens,
dotted initials and GDELT operators are dropped. Offline: no DB, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vera.query_terms import _default_lexicon, _known_acronyms, extract_terms

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "query_terms.json").read_text())


def _lowered(terms):
    return {t.lower() for t in terms}


# --- diagnosis test 2 and the original regression checks ---

def test_diagnosis_test2_keeps_topic_words():
    t = extract_terms("What are implications of the US 2026 Midterm elections?")
    for w in ("US", "Midterm", "2026", "elections"):
        assert w in t, t


def test_person_name_still_dropped():
    t = extract_terms("What did John Smith say")
    assert "John" not in t and "Smith" not in t, t


def test_name_pair_at_start_still_dropped():
    t = extract_terms("John Smith election remarks")
    assert "John" not in t and "Smith" not in t and "election" in t, t


def test_unknown_capitalised_mid_sentence_dropped():
    t = extract_terms("What did the committee say about Zorblatt yesterday")
    assert "Zorblatt" not in t and "committee" in t, t


def test_registry_entity_still_kept():
    assert extract_terms("Anthropic funding", ["Anthropic"]) == ["Anthropic", "funding"]


def test_acronyms_allowlisted_kept():
    t = extract_terms("UK and EU trade, NATO says")
    assert {"UK", "EU", "NATO"} <= set(t), t


def test_dotted_allowlisted_acronym_normalised():
    assert "US" in extract_terms("Effects of U.S. tariffs")


# --- one test per leak case in the privacy review ---

@pytest.mark.parametrize("question, leaked", [
    ("What did JR Smith say", {"jr", "smith"}),
    ("Did JOHN say anything about the election", {"john"}),
    ("ask MARIA about the budget", {"maria"}),
    ("Did AOC win the vote", {"aoc"}),
    ("What did RFK say about vaccines", {"rfk"}),
    ("What did J.R. Smith say", {"j.r", "jr", "smith"}),
    ("What did J. R. Smith say", {"j", "r", "smith"}),
    ("Was PALIN at the summit", {"palin"}),
    ("JOHN SMITH SAID WHAT ABOUT THE ELECTION", {"john", "smith"}),
])
def test_review_leak_cases_dropped(question, leaked):
    t = extract_terms(question)
    assert not (_lowered(t) & leaked), t


def test_gdelt_operators_never_pass():
    t = extract_terms("IT OR budget AND NOT war")
    assert not (_lowered(t) & {"or", "and", "not", "it"}), t
    assert "budget" in t
    assert "OR" not in extract_terms("x OR y", acronyms=["OR"])  # refused even if allowlisted by mistake


def test_stop_words_in_capitals_dropped():
    t = extract_terms("What IS THE impact of tariffs")
    assert not (_lowered(t) & {"is", "the"}), t


def test_lexicon_word_kept_only_when_listed():
    assert "Midterm" not in extract_terms("Effects of Midterm voting", lexicon=())
    assert "Midterm" in extract_terms("Effects of Midterm voting", lexicon=["midterm"])


def test_lexicon_decoupled_from_reporting_lane():
    lex = _default_lexicon()
    assert lex == {w.lower() for w in CFG["common_capitalised"]}
    assert "strike" not in lex and "poll" not in lex
    assert "midterm" in lex and "elections" in lex
    assert "Strike" not in extract_terms("What did Strike say about the budget")


_NAMES = """
james john robert michael william david richard joseph thomas charles christopher daniel matthew anthony mark donald
steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey ryan jacob gary nicholas eric
jonathan stephen larry justin scott brandon benjamin samuel gregory alexander frank patrick raymond jack dennis jerry
tyler aaron jose adam nathan henry douglas zachary peter kyle noah ethan jeremy walter christian keith roger terry
austin sean gerald carl harold dylan arthur lawrence jordan jesse bryan billy bruce gabriel joe logan albert willie
alan eugene russell vincent philip bobby johnny bradley roy ralph randy louis harry wayne howard victor martin bill
will grant hunter hope rose summer mary patricia jennifer linda elizabeth barbara susan jessica sarah karen lisa nancy
betty margaret sandra ashley kimberly emily donna michelle carol amanda dorothy melissa deborah stephanie rebecca
sharon laura cynthia kathleen amy angela shirley anna brenda pamela emma nicole helen samantha katherine christine
debra rachel carolyn janet catherine maria heather diane ruth julie olivia joyce virginia victoria kelly lauren
christina joan evelyn judith megan andrea cheryl hannah jacqueline martha gloria teresa ann sara madison frances
kathryn janice jean abigail alice judy sophia grace denise amber doris marilyn danielle beverly isabella theresa diana
natalie brittany charlotte marie kayla alexis lori kamala hillary nancy
smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez wilson anderson
taylor moore jackson lee perez thompson white harris sanchez clark ramirez lewis robinson walker young allen king
wright scott torres nguyen hill flores green adams nelson baker hall rivera campbell mitchell carter roberts gomez
phillips evans turner diaz parker cruz edwards collins reyes stewart morris morales murphy cook rogers gutierrez ortiz
morgan cooper peterson bailey reed kelly howard ramos kim cox ward richardson watson brooks chavez wood james bennett
gray mendoza ruiz hughes price alvarez castillo sanders patel myers long ross foster jimenez powell jenkins perry
russell sullivan bell coleman butler henderson barnes gonzales fisher vasquez simmons romero jordan patterson
alexander hamilton graham reynolds griffin wallace moreno west cole hayes bryant herrera gibson ellis tran medina
aguilar stevens murray ford castro marshall owens harrison fernandez mcdonald woods washington kennedy wells vargas
henry chen freeman webb tucker guzman burns crawford olson simpson porter hunter gordon mendez silva shaw snyder mason
dixon munoz hunt hicks holmes palmer wagner black robertson boyd rose stone salazar fox warren mills meyer rice
schmidt garza daniels ferguson nichols stephens soto weaver ryan gardner payne grant dunn kelley spencer hawkins
arnold pierce vazquez hansen peters santos hart bradley knight elliott cunningham duncan armstrong hudson carroll
lane riley andrews alvarado ray delgado berry perkins hoffman johnston matthews pena richards contreras willis
carpenter lawrence sandoval biden trump palin gates shapps carney hicks brennan poll strike court
""".split()


def test_lexicon_and_acronyms_contain_no_common_names():
    names = set(_NAMES)
    assert len(names) >= 200
    lex = _default_lexicon()
    assert lex, "lexicon must load"
    assert not (lex & names), sorted(lex & names)
    assert not ({a.lower() for a in _known_acronyms()} & names)


def test_acronym_allowlist_has_no_personal_initialisms():
    acr = _known_acronyms()
    assert {"US", "UK", "EU", "UN", "NATO", "GDP", "CDC", "FBI", "NASA", "WHO", "IMF"} <= acr
    assert not (acr & {"AOC", "RFK", "JFK", "MTG", "DJT", "JR", "OR", "AND", "NOT"})
    assert all(a.isalpha() and a.isupper() for a in CFG["known_acronyms"])
