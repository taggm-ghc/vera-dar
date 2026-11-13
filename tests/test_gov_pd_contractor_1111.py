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
NEW_COUNT = 12  # 7 (GOVPD-CONTRACTOR-1111) + 5 (CONTRACTOR2-1113)
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
    # CONTRACTOR2-1113: slash-joined affiliations, writer lines, Pasadena dateline, contractor mail domain,
    # "operated by ... under contract", "managed on behalf of ... by". All names are fictional.
    "Media contact: Jane Doe, JPL/Caltech",
    "Contact: Jane Doe, NASA/JPL",
    "By Jane Doe, Caltech/JPL",
    "Writer: Jane Doe, Jet Propulsion Laboratory",
    "Media contact: Jane Doe, Space Telescope Science Institute, Baltimore, Md.",
    "PASADENA, Calif. -- NASA's Jet Propulsion Laboratory today announced a new instrument.",
    "News Media Contact\nPat Example\n818-555-0142\npat.example@jpl.example.com",
    "Questions: pat.example@caltech.example.org",
    "The observatory is operated by Example Research Alliance under contract with the Department of Energy.",
    "The array is run by Example Labs under a prime contract to the agency.",
    "The mission is managed on behalf of NASA by JPL.",
    "The telescope is managed for NASA by the Southwest Research Institute.",
    "The instrument is managed for the agency by jpl.",
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
    # CONTRACTOR2-1113 false-positive guards (genuine federal text that the new patterns must leave alone)
    "The site is operated by the National Park Service under the Organic Act.",
    "The facility is managed by NASA staff. Contract awards are posted separately.",
    "Contact: NASA Headquarters, Washington\nCredit: NASA/JPL-Caltech",
    "PASADENA is a city in California. NASA's JPL operates nearby.",
    "Media contacts are listed on the agency page.",
    "Email questions to press@agency.test.",
    "The telescope is managed for NASA by its Goddard center.",
    "Writer: Jane Doe, NASA Headquarters\nCredit: NASA/JPL-Caltech/MSSS",
    "By Jane Doe, NASA/JPL-Caltech",
    "The rover is run by a NASA team under the Mars program.",
    "Visit caltech.edu or jpl.nasa.gov for more.",
    "By Quill Varnet, NASA. Credit: Caltech/IPAC",
    "Contact: Orla Fenwick. Credit: NASA/JPL-Caltech",
    "By Quill Varnet | NASA/JPL-Caltech/SwRI",
    "Writer: Jane Doe, NASA/ESA/CSA/STScI",
    'By Quill Varnet, NASA. Credit: NASA/JPL',
    'By Quill Varnet, NASA Goddard. Image: Caltech/JPL',
    'Contact: Orla Fenwick. Credit: JPL/Caltech',
    'Writer: Orla Fenwick, NASA Ames. Image credit: NASA/JPL',
    'Writer: Orla Fenwick, NASA Goddard. Credit: NASA, ESA, and STScI',
    'By Quill Varnet, NASA. Image: NASA, ESA, J. Example (STScI)',
    'By Quill Varnet, NASA. Credit: NASA, Johns Hopkins APL, SwRI',
    'Media contact: Orla Fenwick, NASA Headquarters. Photo: NASA/JPL',
    'Writer: Orla Fenwick, NASA. Credit: NASA/Caltech',
    "By Quill Varnet, NASA — Credit: NASA/JPL",
    'Contact: Orla Fenwick | Credit: NASA/JPL',
    'Contact: Orla Fenwick. Credit: NASA/JPL.',
    'By Quill Varnet, NASA. Credit: NASA/JPL, NASA/Goddard',
    'The ship is operated by NOAA under a contract with the shipyard for maintenance.',
    'Launch services were run by the Air Force under a contract signed in 2019.',
    'The program is administered by NASA under contract NNX-EXAMPLE.',
    'The aquarium is operated by the city under contract with NOAA.',
    'The observatory site is run by NSF under contract to the University of Example.',
    'The archive is managed by the Department of Energy under a contract vehicle.',
    'The visitor center is run by the Park Service under a concession contract.',
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
    "Johns Hopkins APL, Laurel, Md.",
    "APL manages the mission for NASA's Science Mission Directorate.",
    "By Jane Doe\nSouthwest Research Institute\nText.",
]


def _page(text):
    return "Mission overview.\n" + text + "\nLearn more about the mission."


def test_fixture_count_at_least_48():
    assert len(POSITIVE) + len(NEGATIVE) + len(FALL_BACK_BUT_FEDERAL) + len(STILL_GRANTED_MISSES) >= 48


def test_config_adds_exactly_twelve_bounded_patterns():
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
# RELFIX-1112: load-independent check. An absolute 0.1 s bound failed under host load, so each pattern is timed on
# the same adversarial shape at size n and GROWTH*n and the ratio is bounded. A linear scan grows about GROWTH-fold
# whatever the host speed; quadratic backtracking grows about GROWTH**2-fold (exponential: far more). Both axes are
# scaled: the number of lines, and the run inside one line (a blow-up within a line is invisible to line counts).
GROWTH = 8
RATIO_BOUND = 24  # 3x headroom over linear (8); a quadratic pattern gives about 64
MIN_SMALL_CPU = 0.005  # repeat the small search until it costs at least 5 ms, so timer noise cannot dominate
CPU_CEILING = 2.0  # backstop only: all patterns once on the full-size text; 20x the old bound, not the real check


def _cpu(rx, text, reps):
    t0 = time.process_time()
    for _ in range(reps):
        rx.search(text)
    return time.process_time() - t0


def _growth_ratio(rx, make, n):
    """CPU-time ratio of rx.search on make(GROWTH*n) over make(n); interleaved best of 3 so drift hits both."""
    small, big = make(n), make(GROWTH * n)
    reps = 1
    while _cpu(rx, small, reps) < MIN_SMALL_CPU and reps < 65536:
        reps *= 2
    ts, tb = [], []
    for _ in range(3):
        ts.append(_cpu(rx, small, reps))
        tb.append(_cpu(rx, big, reps))
    return min(tb) / max(min(ts), 1e-9)


# name -> (prefix, line head, run, inner run count, line tail, lines). Text = prefix + (head + run*inner + tail)*lines;
# every full-size text is >= 20k characters, as before.
ADVERSARIAL = {
    "200k-newlines": ("", "", "\n", 1, "", 200_000),
    "20k-newlines": ("", "", "\n", 1, "", 20_000),
    "by-then-200k-newlines": ("by Jane\n", "", "\n", 1, "", 200_000),
    "by-then-20k-newlines": ("by Jane\n", "", "\n", 1, "", 20_000),
    "managed-for-spaces": ("", "managed for ", " ", 2000, "", 100),
    "200k-spaces": ("", "", " ", 1, "", 200_000),
    "contact-commas": ("", "Contact: ", "a,", 80, "\n", 1200),
    # verifier probes (VERIFY-GOVPD-1111): dense-punctuation bylines and padded "managed for"
    "by-dense-punct": ("", "by ", ",|-–—", 20, "\n", 3000),
    "by-comma-120": ("", "by ", "a,", 120, "\n", 800),
    "by-dash-words": ("", "by ", "x-", 60, " " + "y " * 90 + "\n", 800),
    # CPAT13-FIX-1112: "media contact" + whitespace with no colon was quadratic (two [ \t]* around an optional ":")
    "media-contact-spaces": ("", "media contact", " ", 30000, "", 1),
    "media-contact-tabs": ("", "news media contacts", "\t", 30000, "", 1),
    "media-contact-mixed-ws": ("", "media contact", " \t", 15000, "", 1),
    "managed-padded": ("", "managed     for     the     ", "", 0, "", 7000),
    "managed-padded-3": ("", "managed   for   the   ", "", 0, "", 9000),
    "managed-padded-nbsp3": ("", "managed   for   the   ", "", 0, "", 9000),
    "managed-padded-nbsp1": ("", "managed for the ", "", 0, "", 12000),
    "managed-padded-5": ("", "managed" + " " * 5 + "for" + " " * 5 + "the" + " " * 5, "", 0, "", 7000),
    # CONTRACTOR2-1113: the five added patterns (head-line, Pasadena dateline, mail domain, under-contract, on-behalf)
    "writer-spaces": ("", "writer", " ", 30000, "", 1),
    "contact-colon-spaces": ("", "contact:", " ", 30000, "", 1),
    "contact-long-line": ("", "Contact: ", "x ", 100, "\n", 1500),
    "contact-caltech-slashes": ("", "Contact: ", "caltech/", 40, "\n", 800),
    "by-sep-spaces": ("", "by a,", " ", 30000, "", 1),
    "by-spaces": ("", "by", " ", 30000, "", 1),
    "nbsp-run-undercontract": ("", "managed by x ", "\u00a0", 3000, "", 10),
    "contact-cred-lines": ("", "Contact: ", "cred image ", 40, "\n", 1500),
    "contact-cred-long": ("", "Contact: ", "credit", 10000, "", 1),
    "undercontract-gov-the": ("", "operated by ", "the ", 3000, "", 10),
    "undercontract-gov-spaces": ("", "operated by the", " ", 30000, "", 1),
    "nbsp-run-onbehalf": ("", "managed on behalf of ", "\u00a0", 3000, "", 10),
    "pasadena-spaces": ("", "pasadena, calif", " ", 30000, "", 1),
    "pasadena-lines": ("", "pasadena, calif. -- ", "x ", 30, "\n", 2000),
    "mail-at-signs": ("", "", "@", 1, "", 200_000),
    "mail-dotted": ("", "@", "a.", 20000, "", 1),
    "mail-jpl-repeat": ("", "@jpl.a ", "", 0, "", 20000),
    "undercontract-managed-by": ("", "managed by ", "a ", 3000, "", 10),
    "undercontract-repeat": ("", "run by " + "a" * 30 + " ", "", 0, "", 8000),
    "undercontract-spaces": ("", "operated by x ", " ", 20000, "", 1),
    "onbehalf-spaces": ("", "managed on behalf of ", " ", 2000, "", 100),
    "onbehalf-tokens": ("", "managed for ", "a ", 3000, "", 10),
    "realistic-200kb": ("", _page(NEGATIVE[1]) + "\n", "", 0, "", 1500),
}


def _text(prefix, head, run, inner, tail, lines):
    return prefix + (head + run * inner + tail) * lines


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_contractor_patterns_are_fast(mode, name):
    cfg = mode("on")
    prefix, head, run, inner, tail, lines = ADVERSARIAL[name]
    full = _text(prefix, head, run, inner, tail, lines)
    assert len(full) >= 20_000
    rxs = g._compiled(cfg, "contractor_patterns")
    t0 = time.process_time()
    for rx in rxs:
        rx.search(full)
    assert time.process_time() - t0 <= CPU_CEILING, name
    by_lines = lambda k: _text(prefix, head, run, inner, tail, k)  # noqa: E731
    by_run = lambda k: _text(prefix, head, run, k, tail, 1)  # noqa: E731
    for i, rx in enumerate(rxs):
        r = _growth_ratio(rx, by_lines, max(1, lines // GROWTH))
        assert r <= RATIO_BOUND, (name, "lines", i, round(r, 1))
        if run:  # a run inside the line: grow it to about 8x the full-size line, one line only
            r = _growth_ratio(rx, by_run, max(1, inner))
            assert r <= RATIO_BOUND, (name, "run", i, round(r, 1))


def test_growth_check_catches_quadratic_backtracking():
    """Non-vacuity: the same check fails a known quadratic pattern (\\s+x on a run of spaces with no x), on both axes."""
    quadratic, linear = re.compile(r"\s+x", re.I | re.M), re.compile(r"x", re.I | re.M)
    one_line = lambda k: "by " + " " * k + "\n"  # noqa: E731
    assert _growth_ratio(quadratic, one_line, 500) > RATIO_BOUND
    assert _growth_ratio(linear, one_line, 500) <= RATIO_BOUND


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
