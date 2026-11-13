"""NOROUTE-CODE-1112: one code and one true sentence per cause in auto_acquire.resolve_oa. A Europe PMC miss is
R-ACQ-NOROUTE (VERA-DAR's full-text service has no copy it may use), never R-ACQ-NOTOA, which is kept for an index
record that lists no open-access copy (search_providers.acq_prefilter). Fake http_get; no network, no DB."""
import json
from types import SimpleNamespace

import pytest

from tests import acq_support as A
from tests.test_acq_discovered import (Router, _direct, _pub_is_original, cfg_on, oa_src, run,  # noqa: F401 (fixtures)
                                       step_b_web)
from vera import (acq_config, auto_acquire, discovery_panel as dp, inline_acquire, research_outcome as ro,
                  research_record as rr, search_providers as sp, ui_evidence)

NOROUTE, NOTOA, NODOI = "R-ACQ-NOROUTE", "R-ACQ-NOTOA", "R-ACQ-NODOI"
NOROUTE_TEXT = ("Not read: VERA-DAR's full-text service has no copy it may use, and the source's own site was not "
                "contacted.")
NOTOA_TEXT = "Not read: the index record lists no open-access copy of the full text."
JHEP = "10.1007/jhep07(2024)184"  # CC-BY (SCOAP3); not held by Europe PMC
OED = "10.1093/oed/1833363174"  # a closed reference entry
UNCHECKED, UNREADABLE = "R-ACQ-NOROUTE_UNCHECKED", "R-ACQ-ROUTE_UNREADABLE"


def _searches(web):
    return sum(1 for u in web.urls if "/rest/search" in u)


def _disc(o, i=0):
    return o.public()["discovered"][i]


def _oed_work():
    """An OpenAlex work in the R2 (acquisition) select shape: OA status with no OA location."""
    return {"id": "https://openalex.org/W4400000001", "doi": f"https://doi.org/{OED}",
            "title": "SYNTHETIC outbreak transmission incidence entry", "type": "reference-entry",
            "open_access": {"is_oa": False, "oa_status": "closed", "oa_url": None},
            "best_oa_location": None,
            "locations": [{"is_oa": False, "landing_page_url": f"https://doi.org/{OED}", "license": None,
                           "source": {"id": "https://openalex.org/S4210000001"}}],
            "ids": {"openalex": "https://openalex.org/W4400000001", "doi": f"https://doi.org/{OED}"}}


# ------------------------------------------------------------------------------------------------ (i)-(iii), (vi)
def test_i_cc_by_doi_absent_from_europe_pmc_is_noroute_not_notoa():
    web = Router({}, {})  # Europe PMC search returns an empty resultList
    o, store = run(cfg_on(), [oa_src(doi=JHEP)], web)
    e = _disc(o)
    assert e["outcome"] == "rejected" and e["reason"] == NOROUTE and store.plans == []
    assert _searches(web) == 1 and not any("doi.org" in u for u in web.urls)
    assert dp.reason_text(e["reason"]) == NOROUTE_TEXT
    assert "not openly available" not in dp.reason_text(e["reason"])


def test_ii_europe_pmc_record_not_in_its_oa_subset_is_noroute():
    web = Router({JHEP: A.epmc_record(JHEP, oa="N")}, {})
    o, _store = run(cfg_on(), [oa_src(doi=JHEP)], web)
    assert _disc(o)["reason"] == NOROUTE and _searches(web) == 1


def test_iii_europe_pmc_record_for_another_doi_is_noroute():
    other = json.dumps({"resultList": {"result": [A.epmc_record("10.9999/other.doi")]}}).encode()
    web = Router({}, {}, over={"/rest/search": A.Resp(body=other)})
    o, _store = run(cfg_on(), [oa_src(doi=JHEP)], web)
    assert _disc(o)["reason"] == NOROUTE and not any("fullTextXML" in u for u in web.urls)


def test_vi_oed_doi_discovered_without_index_oa_metadata_is_noroute_never_notoa():
    web = Router({}, {})
    o, _store = run(cfg_on(), [oa_src(doi=OED)], web)  # the /ask R1 shape: no `oa` on the discovery record
    assert _disc(o)["reason"] == NOROUTE and _searches(web) == 1


# ------------------------------------------------------------------------------------------------ (iv)
def test_iv_no_identifier_is_nodoi_with_no_request():
    web = A.Web({}, {})
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.resolve_oa(None, A.cfg(), A.deps([], web))
    assert e.value.code == NODOI and web.urls == []
    with pytest.raises(auto_acquire._Stop) as e2:  # a malformed PMCID is no identifier either
        auto_acquire.resolve_oa(None, A.cfg(), A.deps([], web), pmcid="PMCX1")
    assert e2.value.code == NODOI and web.urls == []


# ------------------------------------------------------------------------------------------------ (v)
def test_v_oed_r2_candidate_with_index_metadata_is_notoa_before_any_request():
    oa = sp.openalex_oa_info(_oed_work())
    assert oa == {"is_oa": False, "has_oa_location": False, "licence": None, "pmcid": None}
    c = A.cand(OED, oa=oa)
    cfg, seen, drops = A.cfg(), [], {}
    web = A.Web({}, {})
    ranked = auto_acquire.rank_candidates(A.Q_EPI, [c], cfg, set(), auto_acquire._fd_cfg(), seen.append, drops=drops)
    assert ranked == [] and drops[id(c)] == NOTOA and seen == [NOTOA] and web.urls == []
    assert dp.reason_text(NOTOA) == NOTOA_TEXT


# ------------------------------------------------------------------------------------------------ (vii), (xi)
def test_vii_landing_route_still_entered_when_europe_pmc_has_no_record():
    web, _raw = step_b_web()
    o, _store = run(cfg_on(step_b=True), [oa_src()], web)
    assert any("doi.org" in u for u, _h, _k in web.calls)  # the trigger is preserved
    assert _disc(o)["outcome"] == "admitted" and _disc(o).get("reason") != NOROUTE


@pytest.mark.parametrize("step_b, code", [(False, UNCHECKED), (True, UNREADABLE)])
def test_xi_unreadable_europe_pmc_answer_is_not_a_route_verdict(step_b, code):
    bad = b'{"resultList": {"result": [{"doi": "x", "doi": "y"}]}}'  # duplicate key: strict JSON reject
    web = A.Web({}, {}, overrides={"/rest/search": A.Resp(body=bad)})
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.resolve_oa(JHEP, cfg_on(step_b=step_b), A.deps([], web))
    assert e.value.code == code
    web2, _raw = step_b_web()
    web2.over["/rest/search"] = A.Resp(body=bad)
    o, _store = run(cfg_on(step_b=step_b), [oa_src()], web2)
    assert _disc(o)["reason"] == code and not any("doi.org" in u for u, _h, _k in web2.calls)  # landing not entered


# ------------------------------------------------------------------------------------------------ (viii)-(x)
def test_viii_parity_config_text_and_outcome_map():
    cfg = A.cfg()
    assert acq_config.reason(cfg, "no_route") == NOROUTE and cfg["reason_codes"]["no_route"] == NOROUTE
    assert dp.REASON_TEXT[NOROUTE] == NOROUTE_TEXT and NOROUTE in ro.KNOWN_CODES
    assert ro.reason_from_code(NOROUTE) == "no_route"


def test_ix_diagnostics_and_no_source_request_counted():
    assert ro.diag_status("rejected", NOROUTE) == ("selected_not_fetched", "no_route", "resolve")
    assert rr._fetch_made({"status": "selected_not_fetched", "reason": NOROUTE}) is False


@pytest.mark.parametrize("codes", [{NOROUTE: 2}, {NOROUTE: 1, "R-ACQ-UNREACHABLE": 1}])
def test_x_supply_status_and_its_clause_are_true(codes):
    o = SimpleNamespace(stored=0, answer_only=0, documents=[], admitted=0)
    assert inline_acquire._supply_status(A.cfg(), codes, o) == "no_route"
    line = ui_evidence.acquisition_notice_line({"contract": {"inline_acquisition": {"acquisition": {
        "status": "no_route", "stored": 0, "target": 6}}}})
    assert line == ("VERA-DAR obtained 0 new full documents for this question (aim: at least 6); at least one source "
                    "found had no full-text route VERA-DAR can use.")
    assert "no identifier" not in line
