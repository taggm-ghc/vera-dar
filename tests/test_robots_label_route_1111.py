"""ROBOTS-LABEL-1111 (WP-L) route-hop tests 10-13, 15-16, 27: a refusal at VERA-DAR's full-text service hop (Europe PMC on
www.ebi.ac.uk) is not the candidate's own site and says so; when no landing route exists the line says "no cleared route".
Real shapes: the Europe PMC search URL as resolve_oa builds it, robots URL https://www.ebi.ac.uk/robots.txt, the JHEP
DOI. Fake HTTP only; no network, no DB."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from vera import acq_discovered, auto_acquire, discovery_panel, research_outcome, research_record

EBI = "www.ebi.ac.uk"
DOI = "10.1007/jhep07(2024)184"
ROBOTS = "/robots.txt"


def _txt(body="", status=200, ctype="text/plain", headers=None):
    return A.Resp(status=status, body=body.encode(), ctype=ctype, headers=headers)


def _web(over, doi=DOI):
    web, _raw = A.epi_setup(doi=doi, over=over)
    return web


def _eval(web, doi=DOI, cfg=None):
    return auto_acquire.evaluate_candidate(A.job(), 1.0, doi, A.cand(doi), None, cfg or A.cfg(), A.deps([], web), {}, {},
                                           clock=A.clock_at())


def _code(web, cfg=None, doi=DOI):
    with pytest.raises(auto_acquire._Stop) as e:
        _eval(web, doi=doi, cfg=cfg)
    return e.value.code


def _landing_cfg():
    c = A.cfg()
    c["inline"]["use_discovered"] = True
    c["inline"]["landing_route"] = True
    return c


def test_t10_unreadable_without_landing_route_is_noroute_unchecked():
    assert A.cfg()["inline"].get("landing_route") is not True  # the live state this test depends on
    web = _web({ROBOTS: _txt("", 500)})
    code = _code(web)
    assert code == "R-ACQ-NOROUTE_UNCHECKED-STATUS_5XX"
    line = discovery_panel.reason_text(code)
    assert "no cleared route" in line and "robots rules do not permit" not in line and "Cause: a server error." in line
    assert not any("doi.org" in u for u in web.urls)  # the source's own site was never contacted


def test_t11_unreadable_with_landing_route_is_route_unreadable_and_landing_not_attempted():
    web = _web({ROBOTS: _txt("", 500)})
    code = _code(web, cfg=_landing_cfg())
    assert code == "R-ACQ-ROUTE_UNREADABLE-STATUS_5XX" and not any("doi.org" in u for u in web.urls)
    assert "full-text service could not be checked" in discovery_panel.reason_text(code)


def test_t12_route_rules_say_no_is_route_robots():
    web = _web({ROBOTS: _txt("User-agent: *\nDisallow: /europepmc/\n")})
    code = _code(web)
    assert code == "R-ACQ-ROUTE_ROBOTS"
    assert "source's own site was not contacted" in discovery_panel.reason_text(code)


def test_t13_cached_unreadable_entry_on_the_jats_hop_is_a_route_code():
    acq_discovered._ROBOTS[EBI] = (0.5, False, "STATUS_5XX")
    acq_discovered._TDMREP[EBI] = (0.5, [])
    web = _web({})
    code = _code(web)
    assert code == "R-ACQ-NOROUTE_UNCHECKED-CACHED_STATUS_5XX" and "R-ACQ-ROBOTS" not in code.replace("ROBOTS_", "")
    assert research_outcome.reason_from_code(code) == "no_route"
    assert "within the last minute, reused" in discovery_panel.reason_text(code)


def test_t15_candidates_that_end_before_the_hop_keep_their_own_codes():
    acq_discovered._ROBOTS[EBI] = (0.5, False, "STATUS_5XX")
    web = _web({})
    lookup = A.Lookup(known=[DOI])
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.evaluate_candidate(A.job(), 1.0, DOI, A.cand(DOI), None, A.cfg(), A.deps([], web, lookup=lookup),
                                        {}, {}, clock=A.clock_at())
    assert e.value.code == "R-ACQ-KNOWN" or e.value.code == A.cfg()["reason_codes"]["in_corpus"]
    with pytest.raises(auto_acquire._Stop) as e2:  # no DOI and no PMCID: not_oa before any request
        auto_acquire.evaluate_candidate(A.job(), 1.0, "", {"title": "x", "url": "https://example.org/a"}, None, A.cfg(),
                                        A.deps([], web), {}, {}, clock=A.clock_at())
    assert "ROBOTS" not in e2.value.code and "ROUTE" not in e2.value.code
    assert web.urls == []


def test_t16_three_candidates_reach_the_hop_one_robots_request_and_no_source_fetch_counted():
    web = _web({ROBOTS: _txt("", 500)})
    codes = []
    for d in ("10.9999/a1", "10.9999/a2", "10.9999/a3"):
        codes.append(_code(web, doi=d))
    assert all(c.startswith("R-ACQ-NOROUTE_UNCHECKED-") for c in codes)
    assert sum(1 for u in web.urls if u.endswith(ROBOTS)) == 1  # unchanged cache behaviour: one read in the window
    for c in codes:
        assert not research_record._fetch_made({"status": "selected_not_fetched", "reason": c})


def test_t17_cause_line_for_the_route_hop(capsys):
    _code(_web({ROBOTS: _txt("", 500)}))
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if x.startswith("robots_cause ")]
    assert len(lines) == 1
    assert "host=www.ebi.ac.uk hop=route code=R-ACQ-ROBOTS_UNREADABLE-STATUS_5XX cause=STATUS_5XX cached=0 status=500" \
        in lines[0]
    assert "/robots" not in lines[0] and "?" not in lines[0]


@pytest.mark.parametrize("hdr,suffix", [({"X-Robots-Tag": "noai"}, "X_ROBOTS_TAG"),
                                        ({"Content-Signal": "ai-input=no"}, "CONTENT_SIGNAL")])
def test_t27_route_response_no_ai_signal(hdr, suffix):
    web = _web({"/rest/search": A.Resp(body=b"{}", headers=hdr)})
    code = _code(web)
    assert code == f"R-ACQ-ROUTE_NOAI-{suffix}"
    assert "marked its answer as not for AI use" in discovery_panel.reason_text(code)


def test_t27b_page_no_ai_exception_is_unchecked_not_a_rule(monkeypatch):
    from vera import source_permissions as sp

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(sp, "_lower_headers", boom)
    code = _code(_web({}))
    assert code == "R-ACQ-NOROUTE_UNCHECKED-EVAL" and "ROBOTS" not in code.replace("ROBOTS_", "")


def test_t27c_source_hop_signal_is_robots_page_noai():
    stop = acq_discovered.page_signal_stop(A.cfg(), "META", "https://pub.example/a")
    assert stop.code == "R-ACQ-ROBOTS-PAGE_NOAI"
    assert discovery_panel.reason_text(stop.code) == discovery_panel.REASON_TEXT["R-ACQ-ROBOTS"]
    err = acq_discovered.page_signal_stop(A.cfg(), "error", "https://pub.example/a")
    assert err.code == "R-ACQ-ROBOTS_UNREADABLE-EVAL"
