"""P-1111-07 / P07-FIX-1112: six shipped patterns in config/us_federal_pd.json had polynomial backtracking
(whitespace quantifiers split by optional items, or a MULTILINE `^` followed by a newline-crossing `\\s*`).
The shipped rewrites are linear and give the same search result as the pre-fix patterns (kept below as
literal references, from git 2181844)."""
import copy
import json
import random
import re
import time

import pytest

from vera import gov_public_domain as g

NEWS = r"(?:news|press|times|post|journal|magazine|tribune|gazette|reuters|ap)"
# key, distinguishing substring of the shipped pattern, pre-fix pattern
OLD = {
    "c135": ("copyright_notice_patterns", r"(?:19|20)\d{2}", r"\bcopyright\s*(?:©|\(c\))?\s*(?:19|20)\d{2}"),
    "c138": ("copyright_notice_patterns", r"(?-i:[A-Z])",
             r"\bcopyright(?:ed)?[ \t]*(?::|\bby\b|\bheld by\b|\bowned by\b)?[ \t]*"
             r"(?!(?-i:Office|Act|Law|Laws|Policy|Notice|Information|Status|Protection|Restrictions?|Infringement|"
             r"Accessibility|Privacy)\b)(?-i:[A-Z])"),
    "c139": ("copyright_notice_patterns", r"(?:the[ \t]+)?",
             r"\bcopyright(?:ed)?[ \t]*(?::|\bby\b|\bheld by\b|\bowned by\b)?[ \t]*(?:the[ \t]+)?"
             r"(?!(?:office|offices|act|acts|law|laws|policy|policies|notice|notices|information|status|protection|"
             r"protections|restrictions?|infringement|accessibility|privacy|the|in|of|and|or|to|is|are|was|were|be|"
             r"been|may|might|does|do|did|for|on|under|claims?|holders?|owners?|issues?|questions?|page|statement|"
             r"terms|permissions?|fair|guidance|applies|apply|exists?|law)\b)[a-z]"),
    "t164": ("third_party_text_patterns", "(?:source|by)", r"^\s*(?:source|by)\s*:?\s*[^\n]{0,60}\b" + NEWS + r"\b"),
    "k186": ("contractor_patterns", r"(?:grant|award)", r"\b(?:grant|award)s?\s*(?:no\.?|number|#)?\s*:?\s*[a-z]{0,4}\d[\w-]{2,}"),
    "i191": ("image_credit_patterns", "credit:", r"^\s*credit:\s"),
}
FLAGS = re.IGNORECASE | re.MULTILINE


def _cfg():
    return copy.deepcopy(json.loads(g.CONFIG_PATH.read_text(encoding="utf-8")))


def _shipped(tag):
    key, marker, _old = OLD[tag]
    pats = [p for p in _cfg()["page_override"][key] if marker in p and p not in {o[2] for o in OLD.values()}]
    if tag == "c135":
        pats = [p for p in pats if p.startswith(r"\bcopyright\s*")]
    elif tag in ("c138", "c139"):
        pats = [p for p in pats if p.startswith(r"\bcopyright(?:ed)?[ \t]*") and
                (("(?-i:[A-Z])" in p) == (tag == "c138"))]
    assert len(pats) == 1, (tag, pats)
    return pats[0]


NEW = {t: _shipped(t) for t in OLD}
OLD_RE = {t: re.compile(OLD[t][2], FLAGS) for t in OLD}
NEW_RE = {t: re.compile(NEW[t], FLAGS) for t in OLD}


def test_old_patterns_are_gone_from_shipped_config():
    shipped = {p for v in _cfg()["page_override"].values() if isinstance(v, list) for p in v}
    for tag, (_k, _m, old) in OLD.items():
        assert old not in shipped, tag
        assert NEW[tag] != old


def _timed(rx, text):
    t0 = time.perf_counter()
    rx.search(text)
    return time.perf_counter() - t0


def _inputs(tag, n):
    kw = {"c135": "copyright", "c138": "copyright", "c139": "copyright", "t164": "source", "k186": "grant",
          "i191": "credit"}[tag]
    fills = [" ", "\t", "\n", "\n ", " \t\n", "\r\f\v ", "\u00a0"]
    out = {}
    for i, f in enumerate(fills):
        out[f"kw+fill{i}"] = kw + f * n
        out[f"kw:+fill{i}"] = kw + ":" + f * n
        out[f"fill{i}+kw"] = f * n + kw + " acme"
    out["by+nlsp"] = "by" + "\n " * n
    out["kw_repeated"] = (kw + " ") * n
    out["kw_nl_repeated"] = (kw + "\n") * n
    return out


@pytest.mark.parametrize("tag", sorted(OLD))
def test_shipped_pattern_is_linear_on_adversarial_50k_input(tag):
    worst = 0.0
    for name, text in _inputs(tag, 50_000).items():
        worst = max(worst, min(_timed(NEW_RE[tag], text) for _ in range(2)))
        assert worst < 0.5, (tag, name, worst)


@pytest.mark.parametrize("tag", sorted(OLD))
def test_doubling_input_does_not_quadruple_time(tag):
    """Quadratic growth shows as ~4x per doubling; linear as ~2x. Generous bound against timer noise."""
    for name, text in _inputs(tag, 20_000).items():
        big = _inputs(tag, 40_000)[name]
        a = max(min(_timed(NEW_RE[tag], text) for _ in range(2)), 1e-3)
        b = min(_timed(NEW_RE[tag], big) for _ in range(2))
        assert b < a * 7 + 0.05, (tag, name, a, b)


def test_pre_fix_pattern_was_superlinear():
    """Documents the defect on a bounded input (old pattern about 1 s at 3000 spaces; the fix takes microseconds)."""
    text = "source" + " " * 3000
    assert _timed(OLD_RE["t164"], text) > 10 * max(_timed(NEW_RE["t164"], text), 1e-5)


MATCH = {
    "t164": ["Source: Reuters", "source reuters", "By Staff, The New York Times", "by AP", "BY: the Associated Press",
             "Source: Daily News", "  By John Smith, Washington Post", "\tsource:\tpress release",
             "intro\nSource: ACME News\nmore", "intro\n\n\n   By the Gazette", "Source:\n\nReuters",
             "source\nnews", "by\n\n\nthe times", "x\n\f source: tribune", "x\n\v by magazine",
             "x\n\u00a0 by news", "x\r\n  by Press", "by" + " " * 40 + "ap", "Source: " + "a" * 55 + " news",
             "by\u3000times", "by:ap", "bypass ap", "bypass the news", "Sources: reuters"],
    "c135": ["Copyright 2020", "copyright © 2019", "Copyright (c) 1999", "copyright\n\t2001", "COPYRIGHT©2024"],
    "c138": ["Copyright Elsevier", "copyright: Acme", "Copyright by Wiley", "copyrighted Owned By Smith"],
    "c139": ["copyright acme", "copyright: the acme", "copyright held by smith", "copyrighted the ltd"],
    "k186": ["grant no. R01-123", "Award number: 1234567", "grants # ab1234", "grant: 123456", "award\n\nab1234",
             "grant 1234"],
    "i191": ["Credit: NASA", "  credit: Jane", "x\n\n credit:\tEPA"],
}
NO_MATCH = {
    "t164": ["", "Source: ", "a by-line from the Post", "stand by news", "the source of news",
             "sourcenews", "Source: " + "a" * 80 + " news", "x by news", "x source: reuters", "BY APPLE",
             "by newsletter", "source\n\n\n\n", "by\n" * 5, "source: apple"],
    "c135": ["copyright", "copyright 1899", "copyright © abc", "copy 2020"],
    "c138": ["copyright", "copyright Office", "copyright act", "copyright 2020"],
    "c139": ["copyright", "copyright the", "copyright of acme", "copyright law", "copyright 123"],
    "k186": ["grant", "grant no.", "award xyz", "grants for schools", "grant: a"],
    "i191": ["credit", "credit:", "credits: x", "a credit: x"],
}


@pytest.mark.parametrize("tag", sorted(OLD))
def test_match_and_non_match_corpora_equal_pre_fix_results(tag):
    for s in MATCH[tag] + NO_MATCH[tag] + [m for t in MATCH for m in MATCH[t]] + [m for t in NO_MATCH for m in NO_MATCH[t]]:
        assert bool(OLD_RE[tag].search(s)) == bool(NEW_RE[tag].search(s)), (tag, s)
    for s in MATCH[tag]:
        assert OLD_RE[tag].search(s) and NEW_RE[tag].search(s), (tag, s)
    for s in NO_MATCH[tag]:
        assert not OLD_RE[tag].search(s) and not NEW_RE[tag].search(s), (tag, s)


@pytest.mark.parametrize("tag", sorted(OLD))
def test_differential_fuzz_equals_pre_fix_results(tag):
    rng = random.Random(1112)
    atoms = ["source", "Source", "by", "BY", "credit", "grant", "award", "s", "no.", "number", "#", "copyright",
             "copyrighted", "held by", "the", ":", " ", "  ", "\t", "\n", "\n\n", "\r", "\f", "\v", "\u00a0",
             "reuters", "news", "ap", "post", "x", "acme", "1", "12", "2020", "19", "20", "\u00a9", "(c)", "-",
             "Office", "Act", "a" * 30, ".", "Times", "bypass"]
    for _ in range(6000):
        s = "".join(rng.choice(atoms) for _ in range(rng.randint(0, 14)))
        assert bool(OLD_RE[tag].search(s)) == bool(NEW_RE[tag].search(s)), (tag, s)


def test_page_check_decisions_equal_pre_fix_config():
    cfg_new = _cfg()
    cfg_old = _cfg()
    for tag, (key, _m, old) in OLD.items():
        lst = cfg_old["page_override"][key]
        lst[lst.index(NEW[tag])] = old
    pages = [s for t in MATCH for s in MATCH[t]] + [s for t in NO_MATCH for s in NO_MATCH[t]] + [
        "Air quality basics.\n\n\nSource: Reuters\nLearn more.",
        "Image credit: ACME\nSource: Daily News",
        "Photo\nby Reuters news",
        "Photo by Jane Doe\nAir quality basics.",
        "Image courtesy of Brookhaven National Laboratory\nQuarks are particles.",
        "  Credit: EPA \n\n By Staff Writer, Gazette",
        "Plain federal text about ozone.",
        "This work was funded by grant no. 12345.",
    ]
    for p in pages:
        g.reset_for_tests()
        new = g.page_check(p, "", cfg_new)
        g.reset_for_tests()
        old = g.page_check(p, "", cfg_old)
        assert new == old, repr(p)
    g.reset_for_tests()


@pytest.mark.parametrize("text", [
    "Air quality.\nsource" + " " * 50_000 + "\n" + "\n" * 50_000,
    "copyright" + " " * 50_000,
    "grant" + " " * 50_000,
    "\n" * 100_000 + "credit: x",
    "by\n " * 25_000,
    "copyright\t" * 5_000 + "grant " * 5_000,
])
def test_page_check_is_fast_on_adversarial_pages(text):
    cfg = _cfg()
    g.reset_for_tests()
    t0 = time.perf_counter()
    g.page_check(text, "", cfg)
    elapsed = time.perf_counter() - t0
    g.reset_for_tests()
    assert elapsed < 2.0, elapsed


# a whitespace quantifier, then any run of optional elements, then another whitespace quantifier
_W = r"(?:\\s|\[ \\t\])[*+]"
# optional element: bare `?`, a quantified-optional single char / escape / class (`:?`, `\-?`, `[:,]?`), or an
# optional non-capturing group (CPAT13-FIX-1112: `[ \t]*:?[ \t]*` was quadratic and slipped past a bare-`?` check)
_OPT = r"(?:\?|(?:[^\s()\\\[\]*+?|]|\\.|\[[^\]]*\])\?|\(\?:[^()]*\)\?)"
_ADJACENT = re.compile(_W + r"(?:" + _OPT + r")*" + _W)
_ANCHORED = re.compile(r"\^\\s[*+]")


def _flat(p):
    return re.sub(r"\(\?:\((?:[^()]|\([^()]*\))*\)\\s\*\)\?", "", p)  # drop the factored optional groups


def _unsafe(p):
    return bool(_ANCHORED.search(p) or _ADJACENT.search(_flat(p)))


def test_sibling_guard_flags_the_pre_fix_media_contact_pattern():
    """Non-vacuity (CPAT13-FIX-1112): the guard fails the old `[ \\t]*:?[ \\t]*` form and passes the shipped one."""
    old_form = r"media[ \t]+contacts?[ \t]*:?[ \t]*[^\n]{0,9}?x"
    assert _unsafe(old_form)
    assert _unsafe(r"foo\s*-?\s*bar") and _unsafe(r"foo\s*[:,]?\s+bar") and _unsafe(r"a[ \t]+(?:x)?[ \t]*b")
    assert not _unsafe(r"media[ \t]+contacts?[ \t]*(?::[ \t]*)?[^\n]{0,9}?x")
    assert not _unsafe(r"foo\s*:\s*bar")


def test_no_shipped_page_override_pattern_has_adjacent_unbounded_whitespace_quantifiers():
    """Sibling guard: whitespace quantifiers (`\\s*`, `\\s+`, `[ \\t]*`) must not sit next to each other, or be
    separated only by optional elements (`?`, `:?`, `(?:...)?`), and `^\\s*` / `^\\s+` (newline-crossing after a
    MULTILINE anchor) is banned."""
    n = 0
    for key, lst in _cfg()["page_override"].items():
        if not isinstance(lst, list):
            continue
        for p in lst:
            if not isinstance(p, str):
                continue
            n += 1
            assert not _unsafe(p), (key, p)
    assert n >= 60
