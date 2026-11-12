"""ROBOTS-LABEL-1111 (WP-L) tests 19, 21, 22, 25, 26: every cause suffix renders a true sentence, no internal token leaks,
the substring consumers do not mistake a read failure for a rule, and the stdout cause line (R3) is exact and safe."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from vera import acq_discovered, discovery_panel as dp, inline_acquire, research_outcome, research_record, search_and_fetch
from vera import search_providers

ROOT = Path(__file__).resolve().parent.parent
CODES = json.loads((ROOT / "config" / "auto_acquire.json").read_text())["reason_codes"]
INTERNAL = re.compile(r"R-[A-Z]+-|https?://|\b[a-z0-9-]+\.(?:com|org|ac\.uk|net)\b|_[A-Z]{2,}|STATUS_|CACHED_")
BASES = ("robots_unreadable", "route_unreadable", "noroute_unchecked")
CAUSES = ["TIMEOUT", "NETWORK", "STATUS_5XX", "STATUS_OTHER", "STATUS_401", "STATUS_403", "STATUS_4XX", "CTYPE",
          "REDIRECTS", "SIZE", "ENCODING", "CYCLE", "PARSE", "EVAL"]


def test_t19_new_codes_are_not_transient_and_map_to_no_route():
    cfg = {"reason_codes": CODES}
    for k in ("robots_unreadable", "route_robots", "route_noai", "route_unreadable", "noroute_unchecked", "access_denied",
              "route_denied"):
        assert inline_acquire._transient(cfg, CODES[k]) is False
        assert research_outcome._CODE_MAP[CODES[k]] == "no_route"
        assert research_outcome.reason_from_code(CODES[k] + "-ANY") == (
            "no_route" if k in ("robots_unreadable", "route_noai", "route_unreadable", "noroute_unchecked",
                                "access_denied") else "fetch_failed")


def test_t21_substring_consumers_do_not_mistake_unreadable_for_a_rule():
    unreadable = Exception("R-ACQ-ROBOTS_UNREADABLE-TIMEOUT: x")
    rule = Exception("R-ACQ-ROBOTS: robots.txt of the host disallows the query path")
    assert search_providers._own_decision(unreadable) is None
    assert search_providers._own_decision(rule) == "robots"
    assert search_and_fetch._failure_why(["R-ACQ-ROBOTS_UNREADABLE"]) != "robots"
    assert search_and_fetch._failure_why(["R-ACQ-ROBOTS: x"]) == "robots"


def test_t22_every_reason_code_has_a_label_and_no_label_names_a_host_url_or_code():
    for name, code in CODES.items():
        if code.startswith("R-ACQ-") and name in ("robots_unreadable", "route_robots", "route_noai", "route_unreadable",
                                                  "noroute_unchecked", "access_denied", "route_denied"):
            assert code in dp.REASON_TEXT and not INTERNAL.search(dp.REASON_TEXT[code]), name


@pytest.mark.parametrize("base", BASES)
def test_t25_every_cause_and_cached_variant_renders_base_plus_cause_sentence(base):
    code = CODES[base]
    for cause in CAUSES:
        line = dp.reason_text(f"{code}-{cause}")
        assert line.startswith(dp.REASON_TEXT[code]) and dp.CAUSE_TEXT[cause] in line and not INTERNAL.search(line)
        cached = dp.reason_text(f"{code}-CACHED_{cause}")
        assert cached.startswith(dp.REASON_TEXT[code]) and "reused" in cached and not INTERNAL.search(cached)
        assert dp._reason_line({"outcome": "rejected", "reason": f"{code}-{cause}"}) == line
    assert "within the last hour" in dp.reason_text(f"{code}-CACHED_STATUS_403")  # held for robots_ttl_s (RC-1)
    assert "within the last minute" in dp.reason_text(f"{code}-CACHED_STATUS_5XX")


def test_t25b_unknown_suffix_base_only_two_tuple_cached_and_route_cause():
    code = CODES["robots_unreadable"]
    assert dp.reason_text(f"{code}-ZZZ") == dp.REASON_TEXT[code]
    assert dp.reason_text(f"{code}-CACHED") == dp.REASON_TEXT[code] + " Cause: a failure seen recently, reused."
    assert dp.reason_text(f"{code}-CACHED_ZZZ").startswith(dp.REASON_TEXT[code])
    assert dp.reason_text("R-ACQ-ROBOTS-PAGE_NOAI") == dp.REASON_TEXT["R-ACQ-ROBOTS"]
    assert dp.reason_text("R-ACQ-ROUTE_NOAI-META") == dp.REASON_TEXT["R-ACQ-ROUTE_NOAI"]
    d = dp.reason_text("R-ACQ-ACCESS_DENIED-ROUTE")
    assert d.startswith(dp.REASON_TEXT["R-ACQ-ACCESS_DENIED"]) and "full-text service" in d
    assert dp.reason_text("R-ACQ-ACCESS_DENIED") == dp.REASON_TEXT["R-ACQ-ACCESS_DENIED"]
    assert "robots" not in dp.REASON_TEXT["R-ACQ-ACCESS_DENIED"].lower()  # neutral: no party is named (RC-3)


def _lines(capsys):
    return [x for x in capsys.readouterr().out.splitlines() if x.startswith("robots_cause ")]


def test_t26_cause_line_sanitised_one_per_failure_and_no_path_or_question(capsys):
    acq_discovered.cause_log("www.ebi.ac.uk", "route", "R-ACQ-ROBOTS_UNREADABLE-TIMEOUT", "TIMEOUT", False, 200,
                             "text/html\r\nX: y", 12, 7)
    acq_discovered.cause_log("evil.example\r\nFORGED line", "source", "R-ACQ-X\nY", "T\r", False, "9\n9", "t" * 300, "z", "q")
    out = capsys.readouterr().out
    assert out.count("\n") == 2 and "\r" not in out
    a, b = out.splitlines()
    assert a == ("robots_cause host=www.ebi.ac.uk hop=route code=R-ACQ-ROBOTS_UNREADABLE-TIMEOUT cause=TIMEOUT cached=0 "
                 "status=200 ctype=other bytes=12 ms=7")  # a CR/LF in the content type renders as ctype=other
    assert "host=- " in b and "code=- " in b and "cause=- " in b and "ctype=other" in b and "status=-" in b
    assert "bytes=- ms=-" in b and "FORGED" not in b


def test_t26b_handler_is_attached_once_to_stdout_without_propagating():
    lg = logging.getLogger("vera.robots_cause")
    assert sum(isinstance(h, acq_discovered._StdoutHandler) for h in lg.handlers) == 1 and lg.propagate is False


def test_t26c_one_line_per_fresh_failure_and_one_cached_line_per_ttl(capsys):
    from tests import acq_support as A
    from vera import auto_acquire
    web = A.Web({}, {"PMC1": b"<article/>"}, {"/robots.txt": A.Resp(status=500, body=b"", ctype="text/plain")})
    now = [100.0]
    for _ in range(3):
        with pytest.raises(auto_acquire._Stop):
            auto_acquire._guarded_get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML", A.cfg(),
                                      A.deps([], web), 1000, 5.0, False, clock=lambda: now[0])
        now[0] += 1
    ls = _lines(capsys)
    assert len(ls) == 2 and "cached=0" in ls[0] and "cached=1" in ls[1] and "cause=CACHED_STATUS_5XX" in ls[1]
    assert not any(("PMC1" in x or "/europepmc" in x or "?" in x) for x in ls)
    now[0] += 61  # past the fail TTL: a fresh read, then the next cached hit is logged again
    with pytest.raises(auto_acquire._Stop):
        auto_acquire._guarded_get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML", A.cfg(),
                                  A.deps([], web), 1000, 5.0, False, clock=lambda: now[0])
    assert len(_lines(capsys)) == 1


def test_t26d_research_record_counts_only_requests_made():
    f = research_record._fetch_made
    base = {"status": "selected_not_fetched"}
    assert f({**base, "reason": "R-ACQ-ROBOTS_UNREADABLE-STATUS_5XX"}) is True
    assert f({**base, "reason": "R-ACQ-ROBOTS_UNREADABLE-CACHED_STATUS_5XX"}) is False
    assert f({**base, "reason": "R-ACQ-ACCESS_DENIED"}) is True
    for c in ("R-ACQ-ACCESS_DENIED-ROUTE", "R-ACQ-ROUTE_ROBOTS", "R-ACQ-NOROUTE_UNCHECKED-TIMEOUT",
              "R-ACQ-ROUTE_UNREADABLE-NETWORK", "R-ACQ-ROUTE_NOAI-META", "R-ACQ-ROUTE_DENIED"):
        assert f({**base, "reason": c}) is False, c
