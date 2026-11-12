"""GOVPD-CONTRACTOR-1111 (P-1111-05): JPL/Caltech-authored text on shared NASA hosts must not get the US-government-work label.
Offline: no socket, DB or env. Fixtures are short texts in real page shape. Config-only fix: 7 bounded contractor_patterns."""
import copy
import hashlib
import json
import random
import re
import time

import pytest

from vera import gov_public_domain as g
from vera import licence_gate as lg
from vera import web_route as wr

SCI = "https://science.nasa.gov/mission/perseverance"
WWW = "https://www.nasa.gov/missions/artemis"
NEW_COUNT = 7
SHIPPED_COUNT = 9  # contractor_patterns at origin/main 2181844


def _cfg(mode):
    c = copy.deepcopy(json.loads(g.CONFIG_PATH.read_text(encoding="utf-8")))
    c["mode"] = mode
    return c


@pytest.fixture
def mode(monkeypatch):
    def set_mode(m):
        g.reset_for_tests()
        c = _cfg(m)
        monkeypatch.setattr(g, "load_config", lambda path=None: c)
        return c
    yield set_mode
    g.reset_for_tests()


def cand(url, text):
    return {"url": url, "licence": {"metadata": {}, "content": {}}, "stored_text_kind": "full_text",
            "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


# ---- positives: must fall back (planner 11, reviewer-trial hits 7)
POSITIVE = [
    "JPL is managed for NASA by Caltech in Pasadena, California.",
    "The Jet Propulsion Laboratory, a division of Caltech, builds robots.",
    "News Media Contact\nJane Doe\nJet Propulsion Laboratory, Pasadena, Calif.\n818-555-0100",
    "Dark energy\nby Robin Example\nNASA's Jet Propulsion Laboratory\nDark energy is a mystery.",
    "Written by Jane Doe, Jet Propulsion Laboratory",
    "Operated for the Department of Energy by Lawrence Livermore National Security, LLC",
    "The telescope is operated for NASA by the Space Telescope Science Institute.",
    "Media contact: Jane Doe, Caltech",
    "JPL is a federally funded research and development center.",
    "The mission is managed by the California Institute of Technology for NASA.",
    "By Jane Doe | Jet Propulsion Laboratory",
    "by Jane Doe\nNASA’s Jet Propulsion Laboratory\nText.",
    "Jet Propulsion Laboratory, Pasadena",
    "JPL, Pasadena, Calif.",
    "jpl is managed for nasa by caltech.",
    "JPL, a division of Caltech in Pasadena, manages the mission.",
    "SOFIA was operated for NASA by the Universities Space Research Association.",
    "Argonne National Laboratory is a U.S. Department of Energy laboratory managed by UChicago Argonne, LLC.",
]

# ---- negatives: must stay granted (planner 7 with the image-credit text split, reviewer-trial 7)
NEGATIVE = [
    "The Perseverance rover, a mission managed by JPL, landed in Jezero crater.",
    "NASA's Jet Propulsion Laboratory in Southern California built the instrument. Researchers at Caltech use the data.",
    "Credit: NASA/JPL-Caltech",
    "Image by NASA/JPL-Caltech",
    "Image courtesy of Caltech/IPAC",
    "By Jane Smith\nNASA Headquarters\nThe Mars mission is led by NASA and partners; JPL contributes.",
    "The probe is operated for the public by NASA.",
    "The station is managed for NASA by the agency's Johnson Space Center.",
    "Study the History of Caltech and JPL in this lesson on rocketry.",
    "Credit: NASA/JPL-Caltech/MSSS",
    "NASA/JPL-Caltech/ASU/MSSS",
    "By Jane Doe, NASA\nCredit: NASA/JPL-Caltech",
    "Learn more at https://www.jpl.nasa.gov/missions",
    "Use JPL-Horizons ephemeris data.",
    "By Jane Doe\nNASA Headquarters\nText about Caltech alumni.",
    "NASA relies on FFRDCs such as JPL for some work.",
]

# ---- documented residuals, pinned so a change is noticed (plan "Known residuals" d-f; advisories A1, A2).
# FALL_BACK_BUT_FEDERAL: the label is lost though the page is NASA-authored (accepted, narrowing only).
FALL_BACK_BUT_FEDERAL = [
    "The mission is managed for NASA's Science Mission Directorate by Goddard Space Flight Center.",
    "The program is managed for the agency by Marshall Space Flight Center in Huntsville.",
    "The rover was built at JPL, Pasadena being its home.",
    "Sam Example\nHeadquarters, Washington\n202-555-0143\nLee Example\nJet Propulsion Laboratory, Pasadena, Calif.",
    "By Jane Doe, NASA Headquarters, on the JPL rover",
]
# STILL_GRANTED_MISSES: contractor-shaped text these patterns do not catch (advisories A1, A2; recall unmeasured).
STILL_GRANTED_MISSES = [
    "Media contact: Jane Doe, JPL/Caltech",
    "Contact: Jane Doe, NASA/JPL",
    "By Jane Doe, Caltech/JPL",
    "PASADENA, Calif. -- NASA's Jet Propulsion Laboratory today announced",
    "News Media Contact\nPat Example\n818-555-0142\npat.example@jpl.example.com",
    "Writer: Jane Doe, Jet Propulsion Laboratory",
    "Johns Hopkins APL, Laurel, Md.",
    "APL manages the mission for NASA's Science Mission Directorate.",
    "Media contact: Jane Doe, Space Telescope Science Institute, Baltimore, Md.",
    "By Jane Doe\nSouthwest Research Institute\nText.",
]


def _page(text):
    return "Mission overview.\n" + text + "\nLearn more about the mission."


def test_fixture_count_at_least_48():
    assert len(POSITIVE) + len(NEGATIVE) + len(FALL_BACK_BUT_FEDERAL) + len(STILL_GRANTED_MISSES) >= 48


def test_config_adds_exactly_seven_bounded_patterns():
    pats = json.loads(g.CONFIG_PATH.read_text(encoding="utf-8"))["page_override"]["contractor_patterns"]
    assert len(pats) == SHIPPED_COUNT + NEW_COUNT
    for i, p in enumerate(pats[SHIPPED_COUNT:]):
        re.compile(p, re.I | re.M)
        if i == 2:  # "division of Caltech": \s+ between literal words only, linear (as in the converged plan)
            continue
        assert "\\s+" not in p and "\\s*" not in p and "\\n)*" not in p  # no unbounded whitespace quantifier


@pytest.mark.parametrize("text", POSITIVE)
@pytest.mark.parametrize("url", [SCI, WWW])
def test_contractor_text_falls_back(mode, url, text):
    mode("on")
    for page in (text, _page(text)):
        d = g.decide(url, page)
        assert d["applies"] is False and d["reason_code"] == g.RC_CONTRACTOR and d["licence"] is None, text
        assert lg.licence_status(cand(url, page))[2] != lg.RC_US_FED_PD


@pytest.mark.parametrize("text", NEGATIVE)
def test_federal_text_stays_public_domain(mode, text):
    mode("on")
    d = g.decide(SCI, _page(text))
    assert d["applies"] is True and d["reason_code"] == g.RC_APPLIES, (text, d["reason_code"])


def test_image_credits_still_recorded(mode):
    mode("on")
    d = g.decide(SCI, _page("Image courtesy of NASA/JPL-Caltech"))
    assert d["applies"] is True and d["image_credit"] is True


@pytest.mark.parametrize("text", FALL_BACK_BUT_FEDERAL)
def test_pinned_residual_false_positives(mode, text):
    mode("on")
    d = g.decide(SCI, _page(text))
    assert d["applies"] is False and d["reason_code"] == g.RC_CONTRACTOR, text


@pytest.mark.parametrize("text", STILL_GRANTED_MISSES)
def test_pinned_known_misses(mode, text):
    """Not a claim of correctness: these shapes are still granted (recall unmeasured). A future pattern that catches
    them should flip these on purpose."""
    mode("on")
    d = g.decide(SCI, _page(text))
    assert d["applies"] is True, text


def test_web_lane_returns_no_licence_and_records_code(mode):
    mode("on")
    dev = {}
    assert wr._gov_pd(SCI, POSITIVE[0], {}, dev) is None
    assert dev["gov_pd"] == g.RC_CONTRACTOR
    dev2 = {}
    assert wr._gov_pd(SCI, NEGATIVE[0], {}, dev2)["id"] == "LicenseRef-us-government-work"


# ---- adversarial timing: contractor stage only (the shipped third-party pattern at config line ~164 is P-1111-07)
def _contractor_time(text, cfg):
    rx = g._compiled(cfg, "contractor_patterns")
    t0 = time.perf_counter()
    for r in rx:
        r.search(text)
    return time.perf_counter() - t0


ADVERSARIAL = {
    "200k-newlines": "\n" * 200_000,
    "20k-newlines": "\n" * 20_000,
    "by-then-200k-newlines": "by Jane\n" + "\n" * 200_000,
    "by-then-20k-newlines": "by Jane\n" + "\n" * 20_000,
    "managed-for-spaces": ("managed for " + " " * 2000) * 100,
    "200k-spaces": " " * 200_000,
    "contact-commas": ("Contact: " + "a," * 80 + "\n") * 1200,
    "realistic-200kb": (_page(NEGATIVE[1]) + "\n") * 1500,
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_contractor_patterns_are_fast(mode, name):
    cfg = mode("on")
    text = ADVERSARIAL[name]
    assert len(text) >= 20_000
    assert _contractor_time(text, cfg) <= 0.1, name


# ---- narrowing only: nothing denied at origin/main becomes granted
def test_narrowing_only(mode):
    cfg = mode("on")
    base = copy.deepcopy(cfg)
    base["page_override"]["contractor_patterns"] = cfg["page_override"]["contractor_patterns"][:SHIPPED_COUNT]
    texts = [_page(t) for t in POSITIVE + NEGATIVE + FALL_BACK_BUT_FEDERAL + STILL_GRANTED_MISSES]
    rnd = random.Random(7)
    words = ("NASA JPL Caltech managed operated for by the of division contact media By written Jet Propulsion "
             "Laboratory Pasadena , | - / \n credit image FFRDC LLC Inc university copyright permission").split(" ")
    texts += [" ".join(rnd.choice(words) for _ in range(rnd.randint(1, 30))) for _ in range(3000)]
    changed = 0
    for t in texts:
        before = g.page_check(t, "", base)
        after = g.page_check(t, "", cfg)
        assert not (after[0] and not before[0]), t  # never denied -> granted
        changed += before[0] and not after[0]
    assert changed >= len(POSITIVE) - 3  # the patterns actually bite (non-vacuous)
