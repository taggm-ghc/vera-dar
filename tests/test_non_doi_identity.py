"""PLAN-NONDOI-1022 (owner decision 2026-10-22: documents without a DOI are acceptable; Wikipedia is not a source):
the PMCID identity for acquired full texts with no DOI. OFFLINE and SYNTHETIC only (tests/fulltext_synth.py,
tests/acq_support.py fakes): no network, no DB, no model call. AC1-AC8 of work/plan-nondoi-1022/PLAN.md s8."""
import copy
import hashlib
import json

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from tests import nondoi_golden
from vera import acq_config, auto_acquire, fulldoc_admission, fulldoc_corpus, fulldoc_db
from vera import fulltext_loader as fl
from vera.dar_corpus import doc_key_set

PMC = "PMC7654321"
LANDING = f"https://europepmc.org/article/PMC/{PMC}"
BASE = fl.load_config()


@pytest.fixture(autouse=True)
def _reset():
    auto_acquire.reset_for_tests()
    fulldoc_corpus.reset_for_tests()
    yield
    auto_acquire.reset_for_tests()
    fulldoc_corpus.reset_for_tests()


def icfg():
    return acq_config.ingest_cfg(BASE, A.cfg())


def nd_item(**over):
    it = fs.item("syn_pmc", None, landing_url=LANDING, verifier_status="AUTO-VERIFIED",
                 identifier={"kind": "pmcid", "id": PMC, "url": LANDING})
    it.update(over)
    return it


def parts(**jats_kw):
    return nondoi_golden._parts(fs.jats("syn_pmc", "epi", "", seed=1, **jats_kw), "syn_pmc", BASE)


def refused(item, cfg, text_meta=None):
    text, meta = text_meta or parts()
    with pytest.raises(fl.Refused) as e:
        fl.prepare_from_parts(item, text, meta, cfg, now=nondoi_golden.NOW)
    return e.value.code


# ------------------------------------------------------------------------------------------------ AC1 DOI unchanged
def test_ac1_doi_outputs_are_byte_identical_to_the_pre_change_tree():
    # sha256 of nondoi_golden.dumps(doi_outputs()) computed on the PRE-change tree (65af0ac + DAR-TRACE-1022 fix.diff),
    # work/nondoi-1022/golden/base.json; the post-change file (after.json) compared byte-equal with cmp.
    got = hashlib.sha256(nondoi_golden.dumps(nondoi_golden.doi_outputs()).encode("utf-8")).hexdigest()
    assert got == "0f41c9faf40ec413bd46f518b5b985898bfd3da445d98e616437b193764c6bef"


def test_ac1_doi_item_with_an_identifier_keeps_the_doi_identity():
    it = nd_item(doi="10.9999/syn.pmcdoi", landing_url="https://doi.org/10.9999/syn.pmcdoi")
    text, meta = parts()
    rec = fl.prepare_from_parts(it, text, meta, icfg(), now=nondoi_golden.NOW)
    assert rec["canonical_url"] == "https://doi.org/10.9999/syn.pmcdoi" and rec["provenance"]["doi"] == "10.9999/syn.pmcdoi"


# ------------------------------------------------------------------------------------------------ AC2 curated path
def test_ac2_base_config_still_refuses_a_doi_less_item():
    assert "non_doi_identifier_kinds" not in BASE
    assert refused(nd_item(verifier_status=BASE["verified_status"]), BASE) == "R-FIELD"


# ------------------------------------------------------------------------------------------------ AC3 identity checks
@pytest.mark.parametrize("over", [
    {"identifier": {"kind": "wikipedia_revision", "id": "123456", "url": LANDING}},   # Wikipedia is not a source
    {"identifier": {"kind": "pmid", "id": "12345678", "url": LANDING}},               # not listed
    {"identifier": {"kind": "pmcid", "id": "PMC12x", "url": LANDING}},
    {"identifier": {"kind": "pmcid", "id": "0", "url": LANDING}},
    {"identifier": {"kind": "pmcid", "id": "PMC12345678901", "url": LANDING}},        # 11 digits
    {"identifier": {"kind": "pmcid", "id": "pmc7654321", "url": LANDING}},            # case matters
    {"identifier": {"kind": "pmcid", "id": PMC + "\n", "url": LANDING}},
    {"identifier": {"kind": "pmcid", "id": PMC, "url": "http://europepmc.org/article/PMC/" + PMC}},
    {"identifier": {"kind": "pmcid", "id": PMC}},
    {"identifier": None},
    {"identifier": "pmcid:" + PMC},
    {"doi": ""},
    {"doi": "none"},
])
def test_ac3_doi_less_item_is_refused_without_a_valid_identity(over):
    assert refused(nd_item(**over), icfg()) in ("R-FIELD", "R-DOI", "R-PLACEHOLDER")  # "" is a placeholder


def test_ac3_unanchored_pattern_in_config_never_waives():
    c = icfg()
    c["non_doi_identifier_kinds"] = {"pmcid": "PMC.*"}
    assert refused(nd_item(), c) == "R-FIELD"


def test_ac3_valid_pmcid_identity_is_accepted():
    text, meta = parts()
    rec = fl.prepare_from_parts(nd_item(), text, meta, icfg(), now=nondoi_golden.NOW)
    assert rec["canonical_url"] == LANDING and rec["provenance"]["licence_decision"] == "allow"


# ------------------------------------------------------------------------------------------------ AC4 other filters
@pytest.mark.parametrize("over,jats_kw,cfg_over,want", [
    ({"title": "(title to confirm at fetch)"}, {}, {}, "R-PLACEHOLDER"),
    ({}, {}, {"min_fulltext_chars": 10 ** 7}, "R-SHORT"),          # length filter (shipped 8000) still runs
    ({}, {"abstract_sents": 250}, {}, "R-ABSTRACT"),                 # body-to-abstract ratio
    ({}, {"with_body": False}, {}, "R-ABSTRACT"),
    ({"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}, {}, {},
     "R-LIC-REJECT"),
    ({}, {"licence_url": "https://example.invalid/some-publisher-licence"}, {}, "R-LIC-HOLD"),  # unrecognised: hold
    ({"verifier_status": "UNVERIFIED"}, {}, {}, "R-VERIFIER"),
    ({"landing_url": "ftp://example.invalid/x"}, {}, {}, "R-URL"),
])
def test_ac4_every_other_loader_filter_still_refuses(over, jats_kw, cfg_over, want):
    c = {**icfg(), **cfg_over}
    if over.get("landing_url"):
        over = {**over, "identifier": {"kind": "pmcid", "id": PMC, "url": LANDING}}
    assert refused(nd_item(**over), c, parts(**jats_kw)) == want


# ------------------------------------------------------------------------------------------------ acquisition fakes
def pmc_web(pmcid=PMC, rec_doi=None, jats_doi="", topic="epi", **jats_kw):
    rec = A.epmc_record(rec_doi, pmcid=pmcid)
    raw = fs.jats("acq_pmc", topic, jats_doi, seed=3, **jats_kw)
    return A.Web({pmcid: rec}, {pmcid: raw})


def pmc_cand(pmcid=PMC, **kw):
    return {"url": f"https://europepmc.org/article/PMC/{pmcid}", "title": "SYNTHETIC outbreak transmission incidence "
            "study", "snippet": "transmission incidence outbreak pathogen", "doi": None, "provider": "europepmc",
            "pmcid": pmcid, "oa": {"is_oa": True, "has_oa_location": True, "licence": None, "pmcid": pmcid,
                                   "epmc_ft": True}, **kw}


def run(cands, web, c=None, lookup=None, store=None):
    store = store or A.Store()
    d = A.deps(cands, web, lookup=lookup, store=store)
    out = auto_acquire.run_job(A.job(), c or A.cfg(), d, clock=A.clock_at(), env=A.ENV_ON)
    return out, store, web


# ------------------------------------------------------------------------------------------------ AC5 store + reader
def test_ac5_pmcid_only_document_is_stored_with_null_doi_and_the_reader_accepts_it():
    out, store, web = run([pmc_cand()], pmc_web())
    assert out["stored"] == 1, out
    plan = store.plans[0]
    prov = plan["source"]["provenance"]
    assert "doi" in prov and prov["doi"] is None and plan["canonical_url"] == LANDING
    assert prov["item"]["identifier"] == {"kind": "pmcid", "id": PMC, "url": LANDING}
    assert prov["document_id"] == "acq-" + hashlib.sha256(("pmcid:" + PMC).encode()).hexdigest()[:16]
    assert prov["acquisition"]["marker"] == "vera-dar-auto-acquire-v1"  # marker unchanged (v1)
    assert any('PMCID%3A%22' + PMC in u for u in web.urls)
    src, spans = A.rows_from_plans(store.plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())
    assert c.status == "ok" and len(c.acquired) == 1 and c.acquired[0]["doi"] is None


def test_ac4_ac5_reader_still_re_scans_for_injection_and_needs_the_overlay():
    _, store, _ = run([pmc_cand()], pmc_web())
    src, spans = A.rows_from_plans(store.plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(),
                                      {**A.cfg(), "injection_scan": {"max_hits": 0, "patterns": ["transmission"]}})
    assert not c.acquired and any(u["code"] == fulldoc_db.R_ACQINJECT for u in c.unusable)
    no_allow = A.cfg()
    del no_allow["ingest_overlay"]["non_doi_identifier_kinds"]
    c2 = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), no_allow)
    assert not c2.acquired  # without the allowlist the reader re-check refuses the row (fail closed)


@pytest.mark.parametrize("web_kw,cfg_patch,codes", [
    ({"topic": "clim"}, None, {"R-ACQ-IRRELEVANT", "R-ACQ-FEWSPANS"}),                  # zero relevant spans
    ({}, ["synthetic sentence 3"], {"R-ACQ-INJECT"}),                                    # injection at write
    ({"licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}, None, {"R-LIC-REJECT"}),  # embedded ND
])
def test_ac4_acquisition_gates_still_refuse_pmcid_documents(web_kw, cfg_patch, codes):
    c = A.cfg()
    if cfg_patch:
        c["injection_scan"]["patterns"] = c["injection_scan"]["patterns"] + cfg_patch
    out, store, _ = run([pmc_cand()], pmc_web(**web_kw), c=c)
    assert out["stored"] == 0 and store.plans == [] and set(out["codes"]) <= codes and out["codes"], out


def test_ac4_unclear_record_licence_is_held_for_pmcid_documents():
    web = pmc_web()
    web.records[PMC]["license"] = "some publisher licence"
    out, store, _ = run([pmc_cand()], web)
    assert out["stored"] == 0 and out["codes"] == {"R-ACQ-NOLIC": 1}


# ------------------------------------------------------------------------------------------------ AC6 ranking
def _rank(cands, served=()):
    seen = []
    out = auto_acquire.rank_candidates(A.Q_EPI, cands, A.cfg(), set(served), fulldoc_corpus.load_config(), seen.append)
    return out, seen


def test_ac6_two_pmcid_only_kept_duplicate_dropped_neither_is_nodoi():
    cands = [pmc_cand("PMC1111111"), pmc_cand("PMC2222222"), pmc_cand("PMC1111111"),
             {"title": "no identifiers", "url": "https://example.org/x"}]
    out, codes = _rank(cands)
    assert [c["pmcid"] for _s, d, c, _i in out] == ["PMC1111111", "PMC2222222"] and all(d is None for _s, d, _c, _i in out)
    assert sorted(codes) == ["R-ACQ-DUPLICATE", "R-ACQ-NODOI"]


def test_ac6_served_pmcid_document_is_skipped_by_its_landing_url():
    out, codes = _rank([pmc_cand()], served=doc_key_set(None, LANDING))
    assert out == [] and codes == ["R-ACQ-KNOWN"]


def test_ac6_sort_never_compares_none_with_a_doi_on_a_tie():
    cands = [pmc_cand("PMC3333333"), A.cand("10.9999/acq.tie"), pmc_cand("PMC4444444")]
    out, _ = _rank(cands)  # equal titles and snippets: tier and score tie
    assert [d for _s, d, _c, _i in out] == [None, "10.9999/acq.tie", None]


def test_ac6_doi_candidate_wins_over_pmcid_when_both_present():
    out, _ = _rank([A.cand("10.9999/acq.both", pmcid=PMC)])
    assert out[0][1] == "10.9999/acq.both"


# ------------------------------------------------------------------------------------------------ AC7 DOI promotion
def test_ac7_pmcid_record_with_a_doi_is_stored_under_the_doi_identity():
    doi = "10.9999/acq.promoted"
    out, store, _ = run([pmc_cand()], pmc_web(rec_doi=doi.upper()))
    assert out["stored"] == 1, out
    prov = store.plans[0]["source"]["provenance"]
    assert store.plans[0]["canonical_url"] == f"https://doi.org/{doi}" and prov["doi"] == doi
    assert prov["document_id"] == "acq-" + hashlib.sha256(doi.encode()).hexdigest()[:16]
    assert "identifier" not in prov["item"]


def test_ac7_doi_in_the_jats_article_id_also_promotes():
    doi = "10.9999/acq.jatsdoi"
    out, store, _ = run([pmc_cand()], pmc_web(jats_doi=doi))
    assert out["stored"] == 1 and store.plans[0]["canonical_url"] == f"https://doi.org/{doi}"


@pytest.mark.parametrize("where", ["record", "jats"])
def test_ac7_promoted_doi_already_stored_is_known_not_a_second_copy(where):
    doi = "10.9999/acq.dup"
    web = pmc_web(rec_doi=doi) if where == "record" else pmc_web(jats_doi=doi)
    out, store, _ = run([pmc_cand()], web, lookup=A.Lookup(known={doi}))
    assert out["stored"] == 0 and store.plans == [] and out["codes"] == {"R-ACQ-KNOWN": 1}


def test_ac7_pmcid_without_doi_has_europe_pmc_canonical_and_attribution():
    _, store, _ = run([pmc_cand()], pmc_web())
    src, spans = A.rows_from_plans(store.plans)
    c = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg())
    d = c.acquired[0]
    att = fulldoc_admission.attribution(d, fulldoc_corpus.load_config())
    assert d["url"] == LANDING and d["doi"] is None and att and att["source"] == LANDING


def test_ac7_resolve_by_pmcid_refuses_a_mismatched_record():
    web = pmc_web()
    web.records[PMC]["pmcid"] = "PMC9999999"
    out, store, _ = run([pmc_cand()], web)
    assert out["stored"] == 0 and out["codes"] == {"R-ACQ-NOTOA": 1}


# ------------------------------------------------------------------------------------------------ AC8 overlay keys
def test_ac8_overlay_copies_only_labels_and_the_identity_allowlist():
    c = A.cfg()
    ov = c["ingest_overlay"]
    assert ov["non_doi_identifier_kinds"] == {"pmcid": "^PMC[0-9]{1,10}$"}  # wikipedia_revision dropped (owner)
    hostile = copy.deepcopy(c)
    hostile["ingest_overlay"].update({"min_fulltext_chars": 1, "min_body_chars": 1, "min_body_to_abstract_ratio": 0,
                                      "doi_pattern": ".*", "required_item_fields": [], "url_pattern": ".*",
                                      "cc_licence_patterns": [".*"], "licence_rules": {}, "reason_codes": {}})
    out = acq_config.ingest_cfg(BASE, hostile)
    changed = {k for k in set(out) | set(BASE) if out.get(k) != BASE.get(k)}
    assert changed <= {"verified_status", "admitted_by", "question_text", "licence_observations_enabled",
                       "embedded_licence_unknown_holds", "non_doi_identifier_kinds", "candidate"}, changed
    assert "non_doi_identifier_kinds" in changed and BASE == fl.load_config()  # base never mutated
    assert json.loads(json.dumps(out))["min_fulltext_chars"] == BASE["min_fulltext_chars"] == 8000


# ------------------------------------------------------------------ INT-1022-B disposition (b): pre-network guard
@pytest.mark.parametrize("pmcid", [None, "", "   ", "PMC", "12345", 'PMC1" OR DOI:"x', "PMC12345678901"])
def test_resolve_oa_refuses_before_any_network_call_without_doi_or_valid_pmcid(pmcid):
    web = A.Web({}, {})
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.resolve_oa(None, A.cfg(), A.deps([], web), pmcid=pmcid)
    assert e.value.code == acq_config.reason(A.cfg(), "not_oa")
    assert web.urls == []  # no request was made


def test_resolve_oa_valid_pmcid_still_queries_once():
    web = A.Web({}, {})
    with pytest.raises(auto_acquire._Stop):
        auto_acquire.resolve_oa(None, A.cfg(), A.deps([], web), pmcid="pmc1234567")
    assert len(web.urls) == 1 and "PMC1234567" in web.urls[0]
