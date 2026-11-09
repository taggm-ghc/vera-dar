"""GOV-PD-1108: licence rule us-federal-public-domain (17 U.S.C. 105). Offline: fetch is a stub; no socket, DB or env."""
import copy
import hashlib
import datetime as dt
import json
import re
from pathlib import Path

import pytest

from vera import gov_public_domain as g
from vera import lane_evidence as le
from vera import licence_gate as lg
from vera import public_fetch as pf
from vera import web_route as wr

ROOT = Path(__file__).resolve().parent.parent
DOE = "https://www.energy.gov/science/doe-explainsquarks-and-gluons"
DOE_TEXT = ("DOE Explains...Quarks and Gluons\nImage courtesy of Brookhaven National Laboratory\n"
            "Quarks are elementary particles that make up protons and neutrons, held together by gluons.")
STATE = "https://www.energy.ca.gov/about"   # ca.gov: a STATE .gov domain
UNKNOWN = "https://www.example.org/quarks"


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


def cand(url, text=None, **kw):
    """A candidate after its full text was fetched: text bound to the stored hash (gov_public_domain.candidate_text)."""
    c = {"url": url, "licence": {"metadata": {}, "content": {}}, "stored_text_kind": "full_text", **kw}
    if text is not None:
        c["text"] = text
        c["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return c


# ---------------------------------------------------------------- shipped state and data provenance
def test_ships_shadow_and_licence_is_existing_allowlist_entry():
    c = json.loads(g.CONFIG_PATH.read_text(encoding="utf-8"))
    assert c["mode"] == "shadow"  # orchestrator 2026-11-08: shadow first; 'on' only after the shadow counts are reviewed
    entry, how = lg.match_entry(c["licence_id"])
    assert how == "id" and entry["status"] == "active" and entry["storable"] is True


def test_snapshot_is_cisa_data_not_a_hand_list():
    cfg = json.loads(g.CONFIG_PATH.read_text(encoding="utf-8"))
    snap = json.loads((ROOT / cfg["dataset"]["path"]).read_text(encoding="utf-8"))
    assert snap["source_url"] == cfg["dataset"]["source_url"].replace("http://", "https://")
    assert snap["source_url"].startswith("https://raw.githubusercontent.com/cisagov/dotgov-data/")
    assert snap["licence"] == "CC0-1.0" and re.fullmatch(r"[0-9a-f]{64}", snap["sha256"])
    dt.date.fromisoformat(snap["retrieved_at"][:10])
    assert snap["row_count"] == len(snap["domains"]) > 1000
    assert {r["type"] for r in snap["domains"]} <= set(cfg["dataset"]["federal_types"])
    for r in snap["domains"]:  # reduced columns only: no contact e-mail, city or state ever vendored
        assert set(r) == {"domain", "type", "organization", "suborganization"}
        assert "@" not in json.dumps(r)
    doms = {r["domain"] for r in snap["domains"]}
    assert {"energy.gov", "nasa.gov", "noaa.gov", "nih.gov"} <= doms and "ca.gov" not in doms


# ---------------------------------------------------------------- host rule
def test_cisa_federal_host_is_public_domain(mode):
    cfg = mode("on")
    d = g.decide(DOE, DOE_TEXT, "", cfg)
    assert d["applies"] is True and d["basis"] == "cisa_dotgov" and d["reason_code"] == g.RC_APPLIES
    assert d["licence"]["id"] == "LicenseRef-us-government-work" and d["licence"]["rule"] == "us-federal-public-domain"
    assert d["image_credit"] is True  # an image credit is recorded, not a fallback (text only is used)
    for u in ("https://science.nasa.gov/x", "https://www.noaa.gov/a", "https://www.army.mil/", "https://www.fs.fed.us/"):
        assert g.host_status(u, cfg).federal is True, u


def test_licence_gate_admits_federal_page_in_mode_on(mode):
    mode("on")
    d, why, code = lg.licence_status(cand(DOE, DOE_TEXT))
    assert (d, code) == ("allow", lg.RC_US_FED_PD) and "17 U.S.C. 105" in why


@pytest.mark.parametrize("url", [STATE, "https://www.ca.gov/", "https://www.nyc.gov/x", "https://www.texas.gov/y"])
def test_state_and_local_gov_not_public_domain(mode, url):
    mode("on")
    hs = g.host_status(url)
    assert hs.federal is False and hs.code == g.RC_GOV_NON_FEDERAL
    assert lg.licence_status(cand(url, "Plain state agency text."))[2] == lg.RC_MISSING


def test_unknown_host_unaffected(mode):
    base = lg.licence_status(cand(UNKNOWN, "Some text."))
    for m in ("on", "shadow"):
        mode(m)
        assert lg.licence_status(cand(UNKNOWN, "Some text.")) == base
        assert g.host_status(UNKNOWN).code == g.RC_NOT_FEDERAL
    mode("on")
    lic = cand(UNKNOWN, "x")
    lic["licence"]["content"] = {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
    assert lg.licence_status(lic)[0] == "allow"


@pytest.mark.parametrize("url", ["https://pmc.ncbi.nlm.nih.gov/articles/PMC1/", "https://www.osti.gov/biblio/1",
                                 "https://www.bnl.gov/newsroom/", "https://www.ornl.gov/news", "https://www.lbl.gov/"])
def test_excluded_hosts_and_contractor_labs_fall_back(mode, url):
    mode("on")
    assert g.host_status(url).code == g.RC_EXCLUDED
    assert lg.licence_status(cand(url, "Clean text."))[2] == lg.RC_MISSING


def test_lookalike_hosts_are_not_federal(mode):
    mode("on")
    for u in ("https://energy.gov.evil.example/x", "https://notenergy.gov/x", "https://energygov.com/",
              "https://xn--nrgy-gov.example/", "https://milsite.com/"):
        assert g.host_status(u).federal is False, u


# ---------------------------------------------------------------- page-level override
@pytest.mark.parametrize("text,code", [
    ("Quarks. © 2024 American Physical Society", g.RC_COPYRIGHT),
    ("Copyright 2023 The Regents of the University of California. All rights reserved.", g.RC_COPYRIGHT),
    ("This article originally appeared in Symmetry magazine.", g.RC_THIRD_PARTY),
    ("Text courtesy of the Fermilab Office of Communication.", g.RC_THIRD_PARTY),
    ("Reprinted with permission from AIP Publishing.", g.RC_THIRD_PARTY),
    ("ORNL is managed by UT-Battelle LLC for the US Department of Energy.", g.RC_CONTRACTOR),
    ("This material is based upon work supported by the National Science Foundation.", g.RC_CONTRACTOR),
])
def test_page_with_notice_falls_back(mode, text, code):
    mode("on")
    d = g.decide(DOE, text)
    assert d["applies"] is False and d["reason_code"] == code and d["licence"] is None
    assert lg.licence_status(cand(DOE, text))[2] == lg.RC_MISSING


def test_declared_licence_link_or_url_governs(mode):
    mode("on")
    assert g.decide(DOE, DOE_TEXT, "https://creativecommons.org/licenses/by/4.0/")["reason_code"] == g.RC_DECLARED
    c = cand(DOE, DOE_TEXT)
    c["licence"]["content"] = {"id": "All rights reserved"}
    assert lg.licence_status(c)[0] == "defer"
    c2 = cand(DOE, DOE_TEXT, licence_href="https://example.org/terms")
    assert lg.licence_status(c2)[2] == lg.RC_MISSING


def test_no_page_text_means_unchecked(mode):
    mode("on")
    assert lg.licence_status(cand(DOE))[2] == lg.RC_MISSING
    assert g.decide(DOE, "")["reason_code"] == g.RC_NO_TEXT


def test_annotate_candidate_hook_for_portal(mode):
    mode("off")
    c = {"url": DOE}
    assert g.annotate_candidate(c, DOE_TEXT) is None and "gov_pd" not in c
    mode("on")
    c = {"url": DOE, "licence": {"content": {}}}
    assert g.annotate_candidate(c, DOE_TEXT)["applies"] is True
    assert lg.licence_status(c)[2] == lg.RC_MISSING  # the hook grants nothing: no bound full text on the candidate
    forged = {"url": DOE, "licence": {"content": {}}, "gov_pd": dict(c["gov_pd"], url=DOE)}
    assert lg.licence_status(forged)[2] == lg.RC_MISSING  # a precomputed decision is never trusted (finding 5)
    assert lg.licence_status(cand(DOE, DOE_TEXT))[2] == lg.RC_US_FED_PD


def test_unbound_or_snippet_text_never_allows(mode):
    mode("on")
    c = cand(DOE, DOE_TEXT)
    c["text"] = "A short clean snippet."  # text no longer matches the stored hash
    assert lg.licence_status(c)[2] == lg.RC_MISSING
    c2 = cand(DOE, DOE_TEXT)
    c2.pop("text_sha256")
    assert lg.licence_status(c2)[2] == lg.RC_MISSING
    c3 = cand(DOE, DOE_TEXT, stored_text_kind="snippet")
    assert lg.licence_status(c3)[2] == lg.RC_MISSING
    c4 = {"url": DOE, "licence": {"content": {}}, "page_text": DOE_TEXT, "content": DOE_TEXT, "snippet": DOE_TEXT}
    assert lg.licence_status(c4)[2] == lg.RC_MISSING


def test_metadata_kind_never_uses_rule(mode):
    mode("on")
    c = cand(DOE, DOE_TEXT, stored_text_kind=lg.METADATA_KIND)
    assert lg.licence_status(c)[2] == lg.RC_MISSING


# ---------------------------------------------------------------- modes and staleness
def test_off_and_shadow_change_no_decision(mode):
    mode("off")
    assert lg.licence_status(cand(DOE, DOE_TEXT))[2] == lg.RC_MISSING
    mode("shadow")
    scored = lg.apply_licence_gate([cand(DOE, DOE_TEXT), cand(STATE, "x")])
    assert [c["licence_reason_code"] for c in scored] == [lg.RC_MISSING, lg.RC_MISSING]
    assert scored[0]["gov_pd_shadow"] == g.RC_APPLIES and scored[1]["gov_pd_shadow"] == g.RC_GOV_NON_FEDERAL


def test_stale_snapshot_fails_closed(mode):
    cfg = mode("on")
    assert g.host_status(DOE, cfg, today=dt.date(2099, 1, 1)).code == g.RC_STALE


def test_unreadable_dataset_fails_closed(mode):
    cfg = mode("on")
    cfg["dataset"]["path"] = "config/does-not-exist-gov-pd-1108.json"  # nothing is written anywhere
    g.reset_for_tests()
    assert g.host_status(DOE, cfg).federal is False


# ---------------------------------------------------------------- web lane: access unchanged, licence attached
class Fetch:
    def __init__(self, pages):
        self.calls, self.pages = [], pages

    def __call__(self, url, hop_gate=None, robots_cache=None):
        refused = hop_gate(url) if hop_gate is not None else None
        if refused:
            raise pf.FetchRefused(refused)
        self.calls.append(url)
        r = self.pages[url]
        if isinstance(r, Exception):
            raise r
        return {"url": url, "content_type": "text/html", "x_robots_tag": "", "content_signal": "", "meta_robots": "",
                "licence_href": "", **r}


def _web(cands, f):
    return wr.read_web("what are quarks?", cands, fetch=f, registry=[], deny_list=[])


def _wc(url):
    return {"url": url, "title": "DOE Explains", "snippet": "quarks", "doi": None, "acq_source": "ddg"}


def test_robots_disallow_still_blocks_federal_page(mode, monkeypatch):
    mode("on")
    called = []
    monkeypatch.setattr(g, "decide", lambda *a, **k: called.append(1) or {})
    r = _web([_wc(DOE)], Fetch({DOE: pf.FetchRefused("robots")}))
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_robots_disallow" and called == []


def test_noai_and_tdm_signals_still_block_federal_page(mode):
    mode("on")
    r = _web([_wc(DOE)], Fetch({DOE: {"text": DOE_TEXT, "x_robots_tag": "noai"}}))
    assert r.pages == [] and r.listed


def test_web_lane_attaches_licence_only_in_mode_on(mode):
    mode("off")
    r = _web([_wc(DOE)], Fetch({DOE: {"text": DOE_TEXT}}))
    assert r.pages[0].licence is None and "gov_pd" not in r.dev[0]
    mode("shadow")
    r = _web([_wc(DOE)], Fetch({DOE: {"text": DOE_TEXT}}))
    assert r.pages[0].licence is None and r.dev[0]["gov_pd"] == g.RC_APPLIES
    mode("on")
    r = _web([_wc(DOE), _wc(STATE)], Fetch({DOE: {"text": DOE_TEXT}, STATE: {"text": "State text."}}))
    assert r.pages[0].licence["id"] == "LicenseRef-us-government-work"
    assert r.pages[1].licence is None and r.dev[1]["gov_pd"] == g.RC_GOV_NON_FEDERAL
    assert r.pages[0].answer_only is True and r.pages[0].stored is False  # never stored by this rule
    assert DOE_TEXT not in json.dumps(r.dev)  # dev carries codes only
    assert le._page_licence_id(r.pages[0]) == "LicenseRef-us-government-work"
    assert le._page_licence_id(r.pages[1]) is None


def test_web_lane_copyright_page_falls_back(mode):
    mode("on")
    r = _web([_wc(DOE)], Fetch({DOE: {"text": "Quarks. © 2025 Some Publisher"}}))
    assert r.pages[0].licence is None and r.dev[0]["gov_pd"] == g.RC_COPYRIGHT


# ---------------------------------------------------------------- refresh script
def test_refresh_parse_reduces_columns_and_refuses_short_files():
    import importlib.util
    spec = importlib.util.spec_from_file_location("refresh", ROOT / "scripts" / "refresh_us_gov_domains.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    head = "Domain name,Domain type,Organization name,Suborganization name,City,State,Security contact email\n"
    rows = "".join(f"A{i}.GOV,Federal - Executive,Org {i},,City,ST,CONTACTCOL{i}\n" for i in range(120))
    out = mod.parse(head + rows)
    assert len(out) == 120 and out[0]["domain"] == "a0.gov" and "CONTACTCOL" not in json.dumps(out)
    with pytest.raises(SystemExit):
        mod.parse(head + rows[:200])
    with pytest.raises(SystemExit):
        mod.parse("x,y\n1,2\n")


# ---------------------------------------------------------------- opus review GOV-PD-1108 findings 1-4, 7
def test_every_doe_row_verdict(mode):
    """Finding 1: a DOE row is federal only when its Suborganization names a DOE office or an administration."""
    cfg = mode("on")
    snap = json.loads((ROOT / cfg["dataset"]["path"]).read_text(encoding="utf-8"))
    admitted = {r["domain"] for r in snap["domains"] if r["organization"] == "Department of Energy"
                and g.host_status("https://" + r["domain"] + "/", cfg).federal}
    assert {"energy.gov", "doe.gov", "eia.gov"} <= admitted
    for d in ("bnl.gov", "ornl.gov", "lbl.gov", "anl.gov", "fnal.gov", "nlr.gov", "nrelhub.gov", "driveelectric.gov",
              "nersc.gov", "orau.gov", "srs.gov", "hanford.gov", "rl.gov", "nnss.gov", "unnpp.gov", "cebaf.gov",
              "sandia.gov", "lanl.gov", "llnl.gov", "pppl.gov", "fueleconomy.gov", "isotopes.gov", "nuclear.gov"):
        assert d not in admitted, d
    for r in snap["domains"]:
        if r["domain"] in admitted:
            assert re.search(r"office of|department of energy|administration|project office", r["suborganization"], re.I), r


@pytest.mark.parametrize("host", ["usps.gov", "myusps.gov", "itis.gov", "smithsonian.gov", "lsc.gov", "nga.gov",
                                  "kennedy-center.gov", "presidio.gov", "amtrakoig.gov", "sji.gov", "usip.gov"])
def test_organisation_exclusions(mode, host):
    mode("on")
    hs = g.host_status(f"https://www.{host}/x")
    assert hs.federal is False


@pytest.mark.parametrize("url", ["https://www.regulations.gov/comment/x", "https://docs.house.gov/m/x",
                                 "https://energycommerce.house.gov/h", "https://www.congress.gov/congressional-record/x",
                                 "https://www.supremecourt.gov/DocketPDF/x.pdf", "https://ecf.dcd.uscourts.gov/x",
                                 "https://ntrl.ntis.gov/x", "https://apps.dtic.mil/sti/x", "https://www.jpl.nasa.gov/n",
                                 "https://frederick.cancer.gov/x", "https://medlineplus.gov/ency/article/1.htm",
                                 "https://collections.ushmm.gov/x"])
def test_third_party_repositories_excluded(mode, url):
    mode("on")
    assert g.host_status(url).code == g.RC_EXCLUDED


@pytest.mark.parametrize("text,code", [
    ("Licensed under CC BY-ND 4.0.", g.RC_DECLARED),
    ("This work is available under a Creative Commons licence.", g.RC_DECLARED),
    ("By Jane Doe, Associated Press\nThe storm...", g.RC_THIRD_PARTY),
    ("Source: Reuters\nMarkets fell.", g.RC_THIRD_PARTY),
    ("The following article from The Washington Post was inserted in the Record.", g.RC_THIRD_PARTY),
    ("Content provided by A.D.A.M., Inc.", g.RC_THIRD_PARTY),
    ("WASHINGTON (AP) — Lawmakers said...", g.RC_THIRD_PARTY),
    ("Photo courtesy of NASA. Courtesy of the Associated Press.", g.RC_THIRD_PARTY),
    ("Image courtesy of X. Text courtesy of Y.", g.RC_THIRD_PARTY),
    ("Standard Reference Data © 2024 by the U.S. Secretary of Commerce", g.RC_COPYRIGHT),
])
def test_review_page_probes_fall_back(mode, text, code):
    mode("on")
    d = g.decide(DOE, text)
    assert d["applies"] is False and d["reason_code"] == code


@pytest.mark.parametrize("url", ["https://evil.com\\@energy.gov/", "https://user:@energy.gov/",
                                 "https://energy.gov:8443/x", "https://evil.com:@energy.gov/x"])
def test_ambiguous_urls_are_not_federal(mode, url):
    mode("on")
    assert g.host_status(url).federal is False


# ---------------------------------------------------------------- LICCOND-FIX-1108: third-party and contractor markers
EPA = "https://www.epa.gov/air-quality"


@pytest.mark.parametrize("text,code", [
    ("Copyright Elsevier Inc.", g.RC_COPYRIGHT),
    ("Copyright: American Medical Association.", g.RC_COPYRIGHT),
    ("Reproduced with permission from the publisher.", g.RC_THIRD_PARTY),
    ("Permission to reprint granted by the NEJM.", g.RC_THIRD_PARTY),
    ("Reprinted from JAMA", g.RC_THIRD_PARTY),
    ("Excerpted from The Lancet", g.RC_THIRD_PARTY),
    ("Used under license from the American Cancer Society.", g.RC_THIRD_PARTY),
    ("This article was published in Nature Medicine.", g.RC_THIRD_PARTY),
    ("This report was prepared by Booz Allen Hamilton for the agency.", g.RC_CONTRACTOR),
    ("funded by NIH grant R01-123", g.RC_CONTRACTOR),
    # re-verify round 2 (LICCOND-FIX-1108 v4)
    ("Content licensed from Healthwise, Incorporated.", g.RC_THIRD_PARTY),
    ("Provided courtesy and permission of the Mayo Clinic.", g.RC_THIRD_PARTY),
    ("Text by permission of the author.", g.RC_THIRD_PARTY),
    ("This article is shared through a content partnership with The Washington Post.", g.RC_THIRD_PARTY),
    ("copyright the regents of the university of california", g.RC_COPYRIGHT),
])
def test_third_party_and_contractor_markers_fall_back(mode, text, code):
    mode("on")
    page = "Air quality basics.\n" + text + "\nLearn more about ozone."
    d = g.decide(EPA, page)
    assert d["applies"] is False and d["reason_code"] == code and d["licence"] is None
    assert lg.licence_status(cand(EPA, page))[2] != lg.RC_US_FED_PD


@pytest.mark.parametrize("url,text", [
    (DOE, DOE_TEXT),
    ("https://www.noaa.gov/education/resource-collections/weather-atmosphere/hurricanes",
     "Hurricanes are large, swirling storms. They produce winds of 119 kilometers per hour or higher.\n"
     "Learn more about how NOAA forecasts hurricanes."),
    ("https://www.cdc.gov/clean-hands/about/index.html",
     "Washing your hands is easy, and it is one of the most effective ways to prevent the spread of germs.\n"
     "Follow these five steps every time."),
    (EPA, "Air quality index reports tell you how clean or polluted your air is.\n"
          "This page is not subject to copyright in the United States.\n"
          "EPA awards grants to states to improve air quality. Learn how to apply for a grant."),
    ("https://www.usgs.gov/programs/earthquake-hazards",
     "Earthquakes are caused by sudden slips on faults. The USGS monitors seismic activity across the country.\n"
     "Privacy Policy | Accessibility | FOIA"),
    ("https://www.nasa.gov/missions/artemis",
     "Artemis missions will land astronauts on the Moon. NASA is working with commercial and international partners.\n"
     "Copyright information: NASA content is generally not subject to copyright in the United States."),
    ("https://www.nih.gov/health-information",
     "Health information from NIH institutes helps you learn about diseases and conditions.\n"
     "Most NIH web content is in the public domain; copyright protection does not apply to works of the government."),
    ("https://www.fda.gov/consumers",
     "FDA protects public health by assuring the safety of food and drugs. Learn how to report a problem.\n"
     "The agency awarded grants and contracts to states last year. Questions about copyright? See our policies."),
])
def test_genuine_federal_pages_still_public_domain(mode, url, text):
    mode("on")
    d = g.decide(url, text)
    assert d["applies"] is True and d["reason_code"] == g.RC_APPLIES, (url, d["reason_code"])
