"""DAR-B-1023-C1: identifier flow (provider record -> grounded_ask.retrieve_all -> acq_discovered candidates ->
acquisition resolver) and deterministic recorded-shape fixtures (OpenAlex /works, Europe PMC /rest/search, DDG rows).
Fixtures: tests/fixtures/dar_b/*.json (SYNTHETIC, shaped like the real responses; DOIs 10.9999/..., PMC9000xxx).
Offline: injected transports / fake http router only. No network, no DB, no model."""
from __future__ import annotations

import copy
import json
import time
import urllib.parse
from pathlib import Path

import pytest

from tests import acq_support as A
from tests.test_acq_discovered import Router, cfg_on, epmc_web, run
from vera import acq_config, acq_discovered, auto_acquire, grounded_ask, inline_acquire
from vera import search_providers as sp
from vera.eval_config import load_eval_config

FX = Path(__file__).parent / "fixtures" / "dar_b"
SEARCH = load_eval_config().search
RCFG = {"num_results": 8, "snippet_max_chars": 1500, "providers": [], "deadline_s": 10, "resolve": {"enabled": False}}


def fx(name):
    return json.loads((FX / name).read_text())


class R:
    def __init__(self, body, status=200):
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.status_code, self.headers = status, {}


@pytest.fixture(autouse=True)
def _direct(monkeypatch):
    for m in (auto_acquire, inline_acquire, acq_discovered):
        m.reset_for_tests()
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    yield
    for m in (auto_acquire, inline_acquire, acq_discovered):
        m.reset_for_tests()


# ------------------------------------------------------------------------------------------------ parsers
def oa_provider(works=None):
    s = {**SEARCH["openalex"]}
    doc = fx("openalex_works.json")
    if works is not None:
        doc["results"] = works
    return sp.OpenAlexProvider(s, transport=lambda url, params, headers, timeout: R(doc)), doc


def oa_records(with_oa=True):
    p, _ = oa_provider()
    tok = sp.OPENALEX_OA_FIELDS.set(True)
    try:
        resp = p.call("outbreak transmission incidence", 10, 5)
    finally:
        sp.OPENALEX_OA_FIELDS.reset(tok)
    recs = p.parse(resp, 10)
    if not with_oa:
        for r in recs:
            r.pop("oa", None)
    return {r["openalex_id"].rsplit("/", 1)[1]: {**r, "provider": "openalex"} for r in recs}


def test_openalex_fixture_normalization():
    recs = oa_records()
    assert set(recs) == {"W9000001", "W9000002", "W9000003", "W9000004", "W9000005"}
    a = recs["W9000001"]
    assert a["doi"] == "10.9999/DARB.ALL3" and a["url"] == "https://doi.org/10.9999/DARB.ALL3"  # prefix stripped
    assert auto_acquire._doi(a) == "10.9999/darb.all3"  # the acquisition normaliser lowercases
    assert a["openalex_id"] == "https://openalex.org/W9000001"
    assert a["snippet"] == "Transmission incidence rose during the outbreak pathogen"  # abstract rebuilt in order
    assert a["oa"] == {"is_oa": True, "has_oa_location": True, "licence": "cc-by", "pmcid": "PMC9000001"}
    assert a["licence"]["metadata"]["id"] and a["stored_text_kind"] == "abstract_metadata"
    assert recs["W9000005"]["oa"]["pmcid"] == "PMC9000005"  # lower-case pmc in ids.pmcid -> upper-case PMC
    assert recs["W9000005"]["doi"] is None
    o = recs["W9000003"]
    assert o["doi"] is None and o["url"] == "https://example.org/w3"  # landing page, no DOI
    assert o["oa"]["has_oa_location"] is False and sp.acq_prefilter(o) == "not_oa"
    assert recs["W9000004"]["oa"]["licence"] == "cc-by-nd" and sp.acq_prefilter(recs["W9000004"]) == "nd"
    assert sp.acq_prefilter(a) is None


def test_openalex_select_adds_oa_fields_only_with_oa_context():
    seen = []
    p = sp.OpenAlexProvider(SEARCH["openalex"], transport=lambda u, params, h, t: seen.append(params) or R(fx("openalex_works.json")))
    p.call("outbreak transmission incidence", 5, 5)
    tok = sp.OPENALEX_OA_FIELDS.set(True)
    try:
        p.call("outbreak transmission incidence", 5, 5)
    finally:
        sp.OPENALEX_OA_FIELDS.reset(tok)
    assert seen[0]["select"] == sp.OPENALEX_SELECT and seen[1]["select"].endswith(sp.OPENALEX_OA_SELECT_EXTRA)


def test_reconstruct_abstract_order_and_truncation():
    inv = fx("openalex_works.json")["results"][0]["abstract_inverted_index"]
    assert sp.reconstruct_abstract(inv, 1000).startswith("Transmission incidence rose during")
    assert sp.reconstruct_abstract(inv, 12) == "Transmission..."
    assert sp.reconstruct_abstract(None, 10) == "" and sp.reconstruct_abstract({}, 10) == ""


@pytest.mark.parametrize("raw,want", [("pmc9000001", "PMC9000001"), ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9000002/", "PMC9000002"),
                                      ("PMC9000003", "PMC9000003"), ("12345", None), (None, None)])
def test_pmcid_of_normalizes(raw, want):
    assert sp._pmcid_of(raw) == want


def epmc_hits(n=10):
    ecfg = copy.deepcopy(A.cfg()["search"]["europepmc"])
    return sp.europepmc_search("outbreak", n, 5, ecfg, transport=lambda u, p, h, t: R(fx("europepmc_search.json"))), ecfg


def test_europepmc_fixture_keeps_only_open_access_pmcid_hits_and_normalizes():
    out, _ = epmc_hits()
    by = {c["pmcid"]: c for c in out}
    assert set(by) == {"PMC9000001", "PMC9000005"}  # no-PMCID, PPR (refused source) and not-OA hits dropped
    assert by["PMC9000001"]["doi"] == "10.9999/darb.all3" and by["PMC9000001"]["url"] == "https://doi.org/10.9999/darb.all3"
    assert by["PMC9000005"]["doi"] is None and by["PMC9000005"]["url"] == "https://europepmc.org/article/PMC/PMC9000005"
    assert all(c["provider"] == "europepmc" and c["oa"]["epmc_ft"] is True and c["oa"]["pmcid"] == c["pmcid"] for c in out)
    # the DOI-less OA hit is still a candidate by PMCID (PLAN-NONDOI-1022); the no-OA metadata path is not prefiltered
    assert all(sp.acq_prefilter(c) is None for c in out)


def ddg_rows():
    return fx("ddg_results.json")


def test_ddg_identifier_from_fixture_urls():
    rows = ddg_rows()["results"]
    ids = [sp.ddg_identifier(r["url"]) for r in rows]
    assert ids[0] == {"kind": "doi", "value": "10.9999/darb.doi1"}  # lower-cased, /full stripped
    assert ids[1] == {"kind": "doi", "value": "10.9999/darb.doi2"}
    assert ids[2] == {"kind": "pmcid", "value": "PMC9000009"}
    assert ids[3] == {"kind": "pmcid", "value": "PMC9000010"}
    assert ids[4] is None  # non-scholarly URL


def ddg_cands(calls_left=1):
    doc = ddg_rows()
    seen = []

    def tx(url, params, headers, timeout):
        seen.append(params["query"])
        return R(doc["europepmc_lookup"])
    out, counts = sp.ddg_to_candidates(doc["results"], {"search_url": "https://epmc.invalid/s"},
                                       {"max_landing_fetches": 0, "landing_groups": []}, deadline=time.monotonic() + 30,
                                       clock=time.monotonic, calls_left=calls_left, transport=tx)
    return out, counts, seen


def test_ddg_to_candidates_maps_ids_and_drops_non_scholarly():
    out, counts, seen = ddg_cands()
    assert [c["doi"] for c in out] == ["10.9999/darb.doi1", "10.9999/darb.doi2", "10.9999/darb.pmc9", "10.9999/darb.pmc10"]
    assert [c.get("pmcid") for c in out] == [None, None, "PMC9000009", "PMC9000010"]  # pmcid kept when looked up
    assert all(c["acq_source"] == "ddg" and c["provider"] == "ddg" for c in out)
    assert counts["ddg_results"] == 5 and counts["ddg_mapped"] == 4 and counts["ddg_held"] == 1
    assert not any("reuters" in (c["url"] + str(c["discovered_url"])) for c in out)
    assert len(seen) == 1 and "PMCID:PMC9000009" in seen[0] and "PMCID:PMC9000010" in seen[0]
    assert [c["pmcid"] for c in out if c.get("pmcid")] == [c["oa"]["pmcid"] for c in out if c.get("pmcid")]


def test_ddg_pmc_rows_held_without_lookup_budget():
    out, counts, seen = ddg_cands(calls_left=0)
    assert [c["doi"] for c in out] == ["10.9999/darb.doi1", "10.9999/darb.doi2"] and not seen and counts["ddg_held"] == 3


def test_europepmc_ids_lookup_maps_pmcid_and_pmid_to_lowercase_doi():
    m = sp.europepmc_ids_lookup([("pmcid", "PMC9000009"), ("pmid", "90000010")], 5, {"search_url": "x"},
                                transport=lambda u, p, h, t: R(ddg_rows()["europepmc_lookup"]))
    assert m["PMC9000009"] == {"doi": "10.9999/darb.pmc9", "pmcid": "PMC9000009"}
    assert m["pmid:90000010"]["doi"] == "10.9999/darb.pmc10"


def test_merge_candidates_dedupes_across_provider_fixtures_and_absorbs_oa_metadata():
    recs = oa_records()
    ep, _ = epmc_hits()
    dd, _, _ = ddg_cands()
    ddg_same = {**dd[0], "doi": "10.9999/darb.all3", "pmcid": None}
    merged = sp.merge_candidates([ddg_same], [recs["W9000001"], *ep])
    dois = [c.get("doi") for c in merged]
    assert dois.count("10.9999/darb.all3") == 1 and merged[0]["acq_source"] == "ddg"  # first occurrence keeps its place
    assert merged[0]["oa"]["epmc_ft"] is True and merged[0]["oa"]["licence"] == "cc-by"
    # PMCID-only OpenAlex record (oa.pmcid) and the Europe PMC PMCID-only hit are the same identity
    m2 = sp.merge_candidates([recs["W9000005"]], [ep[1]])
    assert len(m2) == 1 and m2[0]["oa"]["epmc_ft"] is True


# ------------------------------------------------------------------------------------------------ identifier flow
def acq_cfg(**kw):
    return cfg_on(**kw)


def feed(records):
    out, unused = grounded_ask.retrieve_all("q", RCFG, search=lambda q: [dict(r) for r in records])
    return out, unused


def cands_for(records):
    out, unused = feed(records)
    return acq_discovered.discovered_candidates(out + unused, acq_cfg())


def test_flow_all_three_ids_survive_retrieve_all_and_reach_the_candidate():
    rec = {**oa_records(with_oa=False)["W9000001"], "pmcid": "PMC9000001"}  # provider record: openalex_id + DOI + PMCID
    out, _ = feed([rec])
    assert len(out) == 1
    assert out[0]["doi"] == "10.9999/DARB.ALL3" and out[0]["pmcid"] == "PMC9000001"
    assert out[0]["openalex_id"] == "https://openalex.org/W9000001" and out[0]["provider"] == "openalex"
    cands, held = cands_for([rec])
    assert held == [] and len(cands) == 1
    c = cands[0]
    assert c["doi"] == "10.9999/darb.all3"  # normalised
    assert c["discovery"]["record_id"] == "https://openalex.org/W9000001" and c["discovery"]["service"] == "openalex"
    assert c["acq_source"] == "discovered" and "licence" not in c and "stored_text_kind" not in c
    # by design (acq_discovered.py:~178, auto_acquire.rank_candidates) the PMCID is dropped when a DOI exists: DOI wins
    assert "pmcid" not in c


def test_flow_doi_only():
    rec = oa_records(with_oa=False)["W9000002"]
    cands, held = cands_for([rec])
    assert held == [] and cands[0]["doi"] == "10.9999/darb.doionly" and "pmcid" not in cands[0]
    assert cands[0]["discovery"]["record_id"] == "https://openalex.org/W9000002"


def test_flow_pmcid_only_keeps_pmcid_and_openalex_id():
    rec = {"provider": "openalex", "url": "https://example.org/w5", "title": "SYNTHETIC pmcid only", "snippet": "abstract text",
           "doi": None, "pmcid": "pmc9000005", "openalex_id": "https://openalex.org/W9000005"}
    out, _ = feed([rec])
    assert out[0]["pmcid"] == "pmc9000005" and out[0]["openalex_id"] == "https://openalex.org/W9000005"
    cands, held = acq_discovered.discovered_candidates(out, acq_cfg())
    assert held == [] and cands[0]["pmcid"] == "PMC9000005" and cands[0]["doi"] is None
    assert cands[0]["discovery"]["record_id"] == "https://openalex.org/W9000005"


def test_flow_openalex_only_is_link_only_no_doi_with_record_id():
    rec = oa_records(with_oa=False)["W9000003"]
    out, _ = feed([rec])
    assert out[0]["openalex_id"] == "https://openalex.org/W9000003"
    cands, held = acq_discovered.discovered_candidates(out, acq_cfg())
    assert cands == [] and len(held) == 1
    h = held[0]
    assert h["outcome"] == "link_only" and h["reason"] == acq_config.reason(acq_cfg(), "no_doi")
    assert h["record_id"] == "https://openalex.org/W9000003" and h["gate_a"]["factors"]["identifier"] == "none"


def test_flow_abstractless_original_keeps_ids_in_unused():
    rec = {"provider": "openalex", "discovered_via": "openalex", "url": "https://doi.org/10.9999/darb.all3", "title": "T",
           "snippet": "", "doi": "10.9999/darb.all3", "pmcid": "PMC9000001", "openalex_id": "https://openalex.org/W9000001"}
    linked = [{"title": "L", "url": "https://example.org/l", "snippet": ""}]  # a link-only hit whose resolution traces the original
    cfg = {**RCFG, "resolve": {"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5}}
    out, unused = grounded_ask.retrieve_all("q", cfg, search=lambda q: linked, resolver=lambda r, c, d: [dict(rec)])
    assert out == [] and unused[0]["doi"] == "10.9999/darb.all3" and unused[0]["pmcid"] == "PMC9000001"
    assert unused[0]["openalex_id"] == "https://openalex.org/W9000001" and unused[0]["provider"] == "openalex"
    cands, _ = acq_discovered.discovered_candidates(unused, acq_cfg())
    assert cands[0]["discovery"]["record_id"] == "https://openalex.org/W9000001"


def test_flow_doaj_id_survives_retrieve_all_and_is_the_record_id():
    """DOAJ provider emits doaj_id (search_providers.py:1139); acq_discovered._record_id prefers it (L134)."""
    s = {**SEARCH["doaj"]}
    doc = {"results": [{"id": "doajrec9000001", "bibjson": {"year": "2024", "month": "3", "title": "SYNTHETIC doaj record",
                                                          "abstract": "Transmission incidence outbreak",
                                                          "identifier": [{"type": "doi", "id": "10.9999/darb.doaj1"}],
                                                          "journal": {"title": "Synth J", "language": ["EN"]}}}]}
    p = sp.DOAJProvider(s, transport=lambda *a, **k: R(doc))
    rec = {**p.parse(R(doc), 5)[0], "provider": "doaj"}
    assert rec["doaj_id"] == "doajrec9000001"
    out, _ = feed([rec])
    assert out[0].get("doaj_id") == "doajrec9000001", "grounded_ask.retrieve_all drops doaj_id"
    cands, _ = acq_discovered.discovered_candidates(out, acq_cfg())
    assert cands[0]["discovery"]["record_id"] == "doajrec9000001"


def test_flow_dedup_key_is_lowercased_doi():
    a = {"provider": "openalex", "url": "https://doi.org/10.9999/X1", "title": "A", "snippet": "s", "doi": "10.9999/X1"}
    b = {"provider": "doaj", "url": "https://other.example/x1", "title": "B", "snippet": "s", "doi": "10.9999/x1"}
    out, _ = feed([a, b])
    assert len(out) == 1 and out[0]["provider"] == "openalex"


# ------------------------------------------------------------------------------------------------ acquisition routing
def search_queries(web):
    return [urllib.parse.unquote(urllib.parse.urlparse(u).query) for u in web.urls if "/rest/search" in u]


def test_acquire_all_three_resolves_by_doi_and_fetches_the_pmcid_fulltext():
    rec = {**oa_records(with_oa=False)["W9000001"], "pmcid": "PMC9000001"}
    out, _ = feed([rec])
    web, _raw = epmc_web(doi="10.9999/darb.all3", pmcid="PMC9000001")
    o, store = run(acq_cfg(), out, web)
    q = search_queries(web)
    assert q and 'DOI:"10.9999/darb.all3"' in q[0] and "PMCID" not in q[0]
    assert any("/PMC9000001/fullTextXML" in u for u in web.urls)
    e = o.public()["discovered"][0]
    assert e["record_id"] == "https://openalex.org/W9000001" and e["service"] == "openalex" and e["gate_a"]["decision"] == "fetch"
    assert o.status == "ok" and o.stored == 1
    assert store.plans[0]["source"]["provenance"]["acquisition"]["discovery"]["record_id"] == "https://openalex.org/W9000001"


def test_acquire_pmcid_only_resolves_by_pmcid_and_promotes_the_doi():
    rec = {"provider": "openalex", "url": "https://example.org/w5", "title": "SYNTHETIC outbreak transmission incidence study",
           "snippet": "transmission incidence outbreak pathogen", "pmcid": "PMC9000005",
           "openalex_id": "https://openalex.org/W9000005"}
    out, _ = feed([rec])
    doi = "10.9999/darb.p5"
    raw = A.fulltext_synth.jats("acq_epi", "epi", doi) if hasattr(A, "fulltext_synth") else None
    from tests import fulltext_synth
    raw = fulltext_synth.jats("acq_epi", "epi", doi)
    web = Router({"PMC9000005": A.epmc_record(doi, pmcid="PMC9000005")}, {"PMC9000005": raw})
    o, store = run(acq_cfg(), out, web)
    q = search_queries(web)
    assert q and 'PMCID:"PMC9000005"' in q[0] and "DOI:" not in q[0]
    assert o.status == "ok" and o.stored == 1
    assert o.public()["discovered"][0]["record_id"] == "https://openalex.org/W9000005"
    assert store.plans[0]["source"]["provenance"]["item"]["doi"] == doi  # DOI promoted from the Europe PMC core record


def test_acquire_openalex_only_makes_no_europepmc_call_and_is_link_only():
    out, _ = feed([oa_records(with_oa=False)["W9000003"]])
    web, _ = epmc_web()
    o, store = run(acq_cfg(), out, web)
    e = o.public()["discovered"][0]
    assert e["outcome"] == "link_only" and e["reason"] == acq_config.reason(acq_cfg(), "no_doi")
    assert e["record_id"] == "https://openalex.org/W9000003"
    assert not search_queries(web) and not store.plans


def test_acquire_not_open_access_europepmc_record_is_rejected_after_resolve():
    rec = oa_records(with_oa=False)["W9000002"]
    out, _ = feed([rec])
    web, _ = epmc_web(doi="10.9999/darb.doionly", pmcid="PMC9000002", rec_kw={"oa": "N"})
    o, store = run(acq_cfg(), out, web)
    e = o.public()["discovered"][0]
    assert e["outcome"] != "admitted" and not store.plans
    assert any('DOI:"10.9999/darb.doionly"' in q for q in search_queries(web))


def test_nd_licence_openalex_record_is_prefiltered_before_any_fetch_via_merge_path():
    """The ND pre-filter acts on candidates that carry `oa` (acquisition search / Europe PMC); fixture record W9000004."""
    nd = oa_records()["W9000004"]
    ranked_codes = {}
    ranked = auto_acquire.rank_candidates("outbreak transmission incidence", [nd, oa_records()["W9000001"]], acq_cfg(), set(),
                                          auto_acquire._fd_cfg(), lambda c: ranked_codes.__setitem__(c, ranked_codes.get(c, 0) + 1))
    assert [d for _s, d, _c, _i in ranked] == ["10.9999/DARB.ALL3".lower()] or [d for _s, d, _c, _i in ranked] == ["10.9999/darb.all3"]
    assert ranked_codes.get(acq_config.reason(acq_cfg(), "no_licence")) == 1


def test_ddg_candidates_go_to_europepmc_by_doi_with_pmcid_from_lookup():
    dd, _, _ = ddg_cands()
    ranked = auto_acquire.rank_candidates("outbreak transmission incidence", dd, acq_cfg(), set(), auto_acquire._fd_cfg(),
                                          lambda c: None)
    assert {d for _s, d, _c, _i in ranked} == {"10.9999/darb.doi1", "10.9999/darb.doi2", "10.9999/darb.pmc9", "10.9999/darb.pmc10"}
    assert all(c["acq_source"] == "ddg" for _s, _d, c, _i in ranked)
