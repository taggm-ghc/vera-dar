"""PORTAL-WIRE-A-1110 (WP3 answer-only web route, WP4 store lanes, WP5 corpus load): host_role is wired in. Offline."""
import copy
import json

import pytest

from tests import acq_support as A
from tests import host_role_support as HR
from tests.test_acq_discovered import Router, redirect
from tests.test_fixr_legacy_rows_1028 import _base_rows
from vera import acq_discovered, auto_acquire, fulldoc_db, fulltext_loader, host_role, public_fetch, web_route

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"
PORTAL_HOPS = ["https://en.wikipedia.org/wiki/Hadron", "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1",
               "https://ar5iv.org/abs/2101.00001", "https://archive.ph/abcde", "https://www.researchgate.net/publication/1",
               "https://openalex.org/W111"]


@pytest.fixture(autouse=True)
def _roles():
    HR.install()
    yield
    HR.reset()


# ------------------------------------------------------------------------------------------------ WP3: answer-only web
@pytest.mark.parametrize("url", PORTAL_HOPS)
def test_web_portal_urls_are_not_candidates_and_not_hops(url):
    assert web_route.is_web_candidate({"url": url, "title": "t"}) is False
    assert web_route.make_hop_gate("https://example.org/p", cfg={})(url) in ("read_portal", "read_arxiv_portal")


def test_web_unknown_host_is_a_candidate_and_a_hop():
    assert web_route.is_web_candidate({"url": "https://unlisted.example.net/a", "title": "t"}) is True
    assert web_route.make_hop_gate("https://example.org/p", cfg={})("https://unlisted.example.net/a") is None


@pytest.mark.parametrize("target", PORTAL_HOPS)
def test_web_redirect_to_a_portal_is_refused_before_any_request(target):
    sent = []

    def fetch(url, hop_gate=None, robots_cache=None):
        for hop in (url, target):
            why = hop_gate(hop)
            if why:
                raise public_fetch.FetchRefused(why)
            sent.append(hop)
        raise AssertionError("the portal hop was requested")
    res = web_route.read_web("What are hadrons?", [{"url": "https://example.org/hadrons", "title": "Hadrons"}], fetch=fetch,
                             cfg={**web_route.load_config(), "topic_order": {"enabled": False}})
    assert sent == ["https://example.org/hadrons"] and res.pages == [] and len(res.listed) == 1
    assert res.dev[0]["gate"]["read"] in ("read_portal", "read_arxiv_portal")


def test_web_final_url_is_rechecked_even_when_the_fetch_gate_was_bypassed():
    def fetch(url, hop_gate=None, robots_cache=None):  # a transport that reports a portal as the final URL
        return {"url": "https://en.wikipedia.org/wiki/Hadron", "text": "Hadrons are particles. " * 40, "status": 200}
    res = web_route.read_web("What are hadrons?", [{"url": "https://unlisted.example.net/h", "title": "H"}], fetch=fetch,
                             cfg={**web_route.load_config(), "topic_order": {"enabled": False}})
    assert res.pages == [] and res.dev[0]["gate"]["read"] == "read_portal"


def test_web_unknown_host_is_read_in_the_answer_lane_and_denied_for_store():
    u = "https://unlisted.example.net/a"
    assert host_role.is_evidence_allowed(u, "answer") is True and host_role.is_evidence_allowed(u, "store") is False
    assert host_role.is_hop_allowed(u, "answer") is True and host_role.is_hop_allowed(u, "store") is False


def test_search_providers_wikipedia_intake_keeps_old_behaviour_and_adds_the_wiki_family():
    from vera import search_providers as sp
    assert sp._is_wikipedia("https://en.wikipedia.org/wiki/X") and sp._is_wikipedia("https://www.wikidata.org/wiki/Q1")
    assert sp._is_wikipedia("https://en.wikinews.org/wiki/Y") and not sp._is_wikipedia("https://example.org/wikipedia")


# ------------------------------------------------------------------------------------------------ WP4: store lanes
def _gg(web, url, chain=None, hosts=None):
    c = A.cfg()
    if hosts:
        c["fetch"]["allowed_hosts"] = list(c["fetch"]["allowed_hosts"]) + list(hosts)
    return auto_acquire._guarded_get(url, c, A.deps([], web), 100000, 5.0, False, chain=chain)


@pytest.mark.parametrize("target", ["https://arxiv.org/pdf/2101.00001", "https://ar5iv.org/abs/2101.00001",
                                    "https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1", "https://archive.ph/x",
                                    "https://en.wikipedia.org/wiki/X"])
def test_store_redirect_to_a_portal_or_mirror_is_refused_before_the_request(target):
    web = Router({}, {}, {EPMC: redirect(target)})
    # even with the portal host allowlisted (a config error) the role refuses it first
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, EPMC, hosts=[target.split("/")[2]])
    assert e.value.code == "R-ACQ-PORTAL" and A.doc_urls(web.urls) == [EPMC]


def test_store_unknown_host_on_the_allowlist_is_denied_and_the_allowlist_still_applies():
    web = Router({}, {}, {EPMC: redirect("https://unlisted.example.net/x")})
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, EPMC)  # not on the allowlist: the existing check, unchanged
    assert e.value.code == "R-ACQ-HOST"
    web = Router({}, {}, {EPMC: redirect("https://unlisted.example.net/x")})
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, EPMC, hosts=["unlisted.example.net"])  # on the allowlist, but no ORIGINAL / RoR role
    assert e.value.code == "R-ACQ-PORTAL" and A.doc_urls(web.urls) == [EPMC]


def test_store_europe_pmc_fulltext_is_repository_of_record_and_search_stays_metadata_only():
    assert host_role.is_evidence_allowed(EPMC, "store") is True
    web = Router({}, {}, {EPMC: A.Resp(body=b"<article/>", ctype="application/xml")})
    raw, final = _gg(web, EPMC)
    assert final == EPMC and raw
    # the JSON search call is an identifier lookup (want_json): the portal role of the search path does not stop it
    assert host_role.is_evidence_allowed("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=x", "store") is False


def test_store_europe_pmc_ppr_and_preprint_records_are_refused_before_any_request():
    for cand, doi in (({"source": "PPR", "pmcid": "PMC9999999"}, "10.9999/x.1"), ({"oa": {"source": "PPR"}}, "10.9999/x.2"),
                      ({"pub_types": ["Preprint"]}, "10.9999/x.3"), ({}, "10.1101/2020.01.01.123456"), ({}, "10.48550/arXiv.2101.00001")):
        web = Router({}, {}, {})
        d = A.deps([], web)
        with pytest.raises(auto_acquire._Stop) as e:
            auto_acquire.evaluate_candidate(A.job(), 1.0, doi, cand, None, A.cfg(), d, {}, {})
        assert e.value.code == "R-ACQ-PORTAL" and web.calls == [], (cand, doi)


def test_store_candidate_found_through_a_portal_url_is_not_refused_for_its_url_alone():
    meta = auto_acquire.portal_meta({"url": "https://openalex.org/W111", "doi": "10.9999/ok.1"}, "10.9999/ok.1")
    assert host_role.role_for_record(meta).role != "PORTAL"


def test_store_landing_route_hop_to_a_portal_and_to_an_unknown_host_is_refused():
    acq = A.cfg()
    acq["landing_route"] = {**acq.get("landing_route", {}), "publisher_hosts": ["www.biorxiv.org", "unlisted.example.net"]}
    for host in ("www.biorxiv.org", "unlisted.example.net"):
        web = Router({}, {}, {})
        d = A.deps([], web)
        with pytest.raises(auto_acquire._Stop) as e:
            acq_discovered._b_get(f"https://{host}/a", {host}, {"text/html"}, 1000, 2, d, acq, __import__("time").monotonic, None, [])
        assert e.value.code == "R-ACQ-PORTAL" and web.calls == [], host


def test_store_manifest_ingest_refuses_portal_items_and_keeps_other_refusals_specific():
    base = {"doi": "10.9999/x.1", "landing_url": "https://www.nature.com/articles/x", "fulltext_url": "https://www.nature.com/x.xml"}
    assert fulltext_loader.portal_verdict(base) is None
    for over in ({"doi": "10.1101/2020.01.01.123456"}, {"landing_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1"},
                 {"fulltext_url": "https://www.biorxiv.org/content/x.full.xml"}, {"source": "PPR"},
                 {"landing_url": "https://en.wikipedia.org/wiki/X"}):
        with pytest.raises(fulltext_loader.Refused) as e:
            fulltext_loader.check_portal({**base, **over})
        assert e.value.code == "R-ACQ-PORTAL", over


# ------------------------------------------------------------------------------------------------ WP5: corpus load
def test_stored_portal_rows_are_not_served_and_are_counted(monkeypatch, tmp_path):
    cfg, _, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=False)
    assert len(sources) >= 3
    s = copy.deepcopy(sources)
    s[0]["provenance"]["item"]["landing_url"] = "https://arxiv.org/abs/2101.00001"  # an arXiv row stored earlier
    s[1]["provenance"]["acquisition"] = {"resolver": "arxiv", "arxiv": {"id": "2101.00001"}}
    n0 = fulldoc_db.portal_refused_count()
    c = fulldoc_db.load_from_rows(s, copy.deepcopy(spans), cfg)
    codes = sorted(u["code"] for u in c.unusable)
    assert codes == ["R-ACQ-PORTAL", "R-ACQ-PORTAL"] and len(c.acquired) == len(sources) - 2
    served = {a["document_id"] for a in c.acquired}
    assert s[0]["provenance"]["document_id"] not in served and s[1]["provenance"]["document_id"] not in served
    assert fulldoc_db.portal_refused_count() == n0 + 2


def test_stored_europe_pmc_row_still_serves(monkeypatch, tmp_path):
    cfg, _, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=False)
    s = copy.deepcopy(sources)
    s[0]["provenance"]["item"]["landing_url"] = "https://europepmc.org/article/PMC/PMC1234567"
    c = fulldoc_db.load_from_rows(s, copy.deepcopy(spans), cfg)
    assert "R-ACQ-PORTAL" not in [u["code"] for u in c.unusable]


def test_portal_quarantine_does_not_write(monkeypatch, tmp_path):
    cfg, _, sources, spans = _base_rows(monkeypatch, tmp_path, legacy=False)
    s = copy.deepcopy(sources)
    s[0]["provenance"]["item"]["doi"] = "10.1101/2020.01.01.123456"
    before = json.dumps(s, sort_keys=True)
    fulldoc_db.load_from_rows(s, copy.deepcopy(spans), cfg)
    assert json.dumps(s, sort_keys=True) == before
