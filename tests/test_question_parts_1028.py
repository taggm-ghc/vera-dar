"""Tests for vera.question_parts (synthetic text only)."""
from __future__ import annotations

from vera.question_parts import (DEFAULT_TABLE, Part, concept_parts, facets,
                                 part_fit, table)

Q = "What are key differences and commonalities between boxing and mixed martial arts?"


def _f(name):
    return next(p for p in facets(Q) if p.label == name)


def test_two_facets_in_order():
    assert [p.label for p in facets(Q)] == ["contrast", "common"]


def test_single_facet_not_split():
    assert facets("What are the differences between boxing and MMA?") == []


def test_no_facet():
    assert facets("How does boxing work?") == []


def test_contrast_fit():
    t = "Unlike boxing, MMA allows kicks, whereas boxing does not."
    assert part_fit(t, _f("contrast")) >= 2
    assert part_fit(t, _f("common")) == 0


def test_common_fit():
    assert part_fit("Both sports share a ring", _f("common")) >= 2


def test_weak_cues_are_not_part_evidence():  # REPLAY-1028 b09: "than" promoted an off-topic passage
    b09 = ("Officers trained in mixed martial arts techniques reported fewer use-of-force injuries during arrests "
           "than officers trained in baton tactics.")
    assert part_fit(b09, _f("contrast")) == 0 and part_fit(b09, _f("common")) == 0
    for weak in ("Boxing is more physical than golf", "Boxing, but not MMA", "while training", "also popular",
                 "the same gloves", "However, rules vary"):
        assert part_fit(weak, _f("contrast")) == 0 and part_fit(weak, _f("common")) == 0


def test_word_boundaries():
    c, m = _f("contrast"), _f("common")
    assert part_fit("Press the button", c) == 0
    assert part_fit("Add butter", c) == 0
    assert part_fit("a bothersome fee", m) == 0
    assert part_fit("alsoran", m) == 0


def test_concept_parts():
    ps = concept_parts(["boxing", "mixed martial arts|mma"])
    assert len(ps) == 2
    assert ps[1].label == "mixed martial arts"
    assert part_fit("MMA fighters", ps[1]) == 1
    assert part_fit("Mixed martial arts is popular", ps[1]) == 1
    assert part_fit("MMAX", ps[1]) == 0


def test_concept_inflection():
    p = Part("concept", "boxing", ("boxing",))
    assert part_fit("boxing gyms", p) == 1
    assert part_fit("Boxers train", p) == 0  # prefix 'boxin' only


def test_concept_parts_single_or_dupes():
    assert concept_parts(["boxing"]) == []
    assert concept_parts(["boxing", "Boxing"]) == []


def test_table_override():
    cfg = {"facets": {"cost": {"question": ["cost"], "span": ["price*"]},
                      "time": {"question": ["time"], "span": ["hour*"]}}}
    ps = facets("cost and time?", cfg)
    assert [p.label for p in ps] == ["cost", "time"]
    assert part_fit("The prices vary", ps[0]) == 1


def test_table_default():
    assert table({}) is DEFAULT_TABLE
    assert table(None) is DEFAULT_TABLE
