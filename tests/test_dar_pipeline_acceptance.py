"""DAR-PIPE-1022 v2: OFFLINE acceptance tests for the discovery -> acquisition -> answer pipeline (owner spec A-F),
adapted to the 7da574a (CMF-1022) design. Stage into repo/tests/ at integration time (see README.md).

Design facts the tests rely on (7da574a, read):
- Discovered records reach acquisition ONLY when config inline.use_discovered is true (shipped TRUE since ENAB-1023; the
  flag-off rollback state is forced false on the rig's private config copy). Tests set it on
  the rig's private config copy (rig.cfg), never in a file.
- Full text is fetched ONLY through Europe PMC (search by DOI, then fullTextXML by PMCID). The fake is A.Web/Router.
- Per-candidate outcomes and reasons are in contract.inline_acquisition.discovered, not on candidates[].
- Inline acquisition needs acq_config.enabled (DB-env gate patched by the `h` fixture) and a fake deps factory (the `rig`).
Nothing opens a network socket or a DB connection (tests/conftest.py boundary + fakes).

Section FIX: tests for the DELTA.md 7c follow-up fixes. They assert the INTENDED behaviour and are expected to FAIL on the
unmodified 7da574a code until the fix lands. They are not xfail-marked on purpose."""
import json
import re

import pytest
import requests

from tests import acq_support as A
from tests import fulltext_synth
from tests._pipeline_support import (ABSTRACT_SENTINEL, DOI, PMC, PRIVATE_IP, Q_EPI, Q_NONE, W1, DiscSearch, _decide,
                                     arxiv_src, ask_offtopic, discovered, disc_entries, entry_for, epmc_web, h,  # noqa: F401
                                     passages_sent_to_model, reason_code, rig, stored_text, GROUND, PRICING)
from vera import acq_config, acq_discovered, ask_service, auto_acquire, grounded_ask, inline_acquire, selective_fetch

EPMC_SEARCH = "/rest/search"
EPMC_FULLTEXT = f"/{PMC}/fullTextXML"


def _fetched(rig, frag):
    return [u for u in rig.web.urls if frag in u]


def _not_an_answer_from_new_doc(rig, r):
    assert r.result_kind != "answer"
    assert rig.store.plans == [] and rig.h.answer_calls == []


# ------------------------------------------------------------------------------------------------- current-state baseline
def test_BASELINE_flag_off_discovered_candidates_do_not_enter_acquisition(rig):
    """With the ROLLBACK setting (inline.use_discovered forced false on the rig copy; shipped is true) a discovered, otherwise eligible record is not handed to
    acquisition: nothing is fetched for it and no per-candidate outcome exists. Expected: PASS before and after the fix."""
    # DAR-B-1023: the shipped file now has use_discovered true (ENAB-1023); this baseline proves the rollback lever:
    # with the flag forced false on the rig's private copy, discovered records do not enter acquisition.
    rig.cfg["inline"]["use_discovered"] = False
    r, ds = ask_offtopic(rig, [discovered()], use_discovered=False)
    assert r.result_kind != "answer"
    assert not _fetched(rig, DOI) and not _fetched(rig, EPMC_FULLTEXT), "a discovered record was fetched with the flag off"
    assert rig.store.plans == [] and rig.h.answer_calls == []
    assert "discovered" not in ((r.contract or {}).get("inline_acquisition") or {})


# ------------------------------------------------------------------------------------------------- A: discovery
def test_A_corpus_miss_runs_discovery_search(rig):
    """Corpus miss -> the request's discovery (reading-list) search runs, on the restricted off-topic seam."""
    r, ds = ask_offtopic(rig, [discovered()])
    assert ds.calls and all(c is True for c in ds.calls)
    assert rig.outcomes, "inline acquisition did not run on a corpus miss"


def test_A_neg_corpus_hit_does_not_run_discovery_or_acquisition(rig):
    rig.h.bundle()  # default synthetic bundle contains the epidemiology paper -> a base-corpus hit
    ds = DiscSearch([discovered()])
    _decide(rig.mp, True, "layer1", "ok")
    r = ask_service.answer_in_scope(Q_EPI, "m", PRICING, grounding=(ds, GROUND[1]))
    assert r.result_kind == "answer"
    assert ds.calls == [] and rig.outcomes == [] and rig.store.plans == []


# ------------------------------------------------------------------------------------------------- B: acquisition
def test_B_eligible_discovered_record_enters_acquisition_via_europe_pmc(rig):
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    assert _fetched(rig, EPMC_SEARCH), "Europe PMC was not asked about the discovered DOI"
    assert _fetched(rig, EPMC_FULLTEXT), "the Europe PMC full text was not fetched"
    e = entry_for(r)
    assert e["service"] == "openalex" and e["outcome"] == "admitted" and e["gate_a"]["decision"] == "fetch"
    assert e["gate_b"]["decision"] == "admit"


def test_B_neg_nd_licensed_record_is_not_admitted_and_says_why(rig):
    web, _ = epmc_web(rec_kw={"licence": "cc by-nd"})
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r)
    assert e and e["outcome"] != "admitted" and e.get("reason", "").startswith(("R-ACQ-", "R-LIC-"))  # 7da574a ND refusal code is R-LIC-REJECT


def test_B_neg_preprint_without_licence_route_is_link_only_not_fetched(rig):
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [arxiv_src()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r, "arXiv:2401.00001")
    assert e["outcome"] == "link_only" and e["reason"] == reason_code(rig, "preprint_licence_unverified")
    assert e["epistemic"] == acq_discovered.PREPRINT_EPISTEMIC
    assert not _fetched(rig, "arxiv"), "no arXiv full-text route may exist"


# ------------------------------------------------------------------------------------------------- C: admission controls
def test_C_acquired_content_with_nd_licence_is_not_admitted(rig):
    """Behavioural: the Europe PMC record looks fine, the fetched JATS itself declares CC BY-ND."""
    web, _ = epmc_web(jats_kw={"licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"})
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    assert _fetched(rig, EPMC_FULLTEXT), "the fetch must have happened for the licence check to be meaningful"
    _not_an_answer_from_new_doc(rig, r)


def test_C_acquired_abstract_only_content_is_not_admitted(rig):
    raw = fulltext_synth.jats("acq_epi", "epi", DOI, with_body=False)
    web, _ = epmc_web()
    web.jats[PMC] = raw
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)


def test_C_acquired_irrelevant_content_is_not_admitted_as_evidence(rig):
    web, _ = epmc_web(topic="clim")
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r)
    assert e["outcome"] == "rejected" and e["reason"] in (reason_code(rig, "irrelevant"), reason_code(rig, "few_spans"))


def test_C_security_default_transport_is_pinned_and_public_only():
    """Wiring assertion on the real default deps factory (source text; no connection is made). Behavioural DNS-rebinding
    and pin tests are in tests/test_cmf_i2_pinning.py and tests/test_cmf_i3_pin_hygiene.py (ported, unchanged)."""
    import inspect
    src = inspect.getsource(auto_acquire)
    assert "limited_by_url(selective_fetch.pinned_get)" in src
    assert "url_check=selective_fetch.is_public_url" in src


@pytest.mark.parametrize("url", [f"https://{PRIVATE_IP}/x", "https://127.0.0.1/x", "https://[::1]/x",
                                 "https://[::ffff:10.0.0.5]/x", "http://169.254.169.254/x"])
def test_C_security_non_public_destinations_are_refused_without_connecting(url, monkeypatch):
    import socket
    import ipaddress
    from urllib.parse import urlsplit

    def literal_gai(host, port, *a, **k):  # offline resolver: a literal IP resolves to itself (no real resolver call)
        ip = ipaddress.ip_address(host)
        fam = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (str(ip), port))]
    monkeypatch.setattr(socket, "getaddrinfo", literal_gai)
    assert urlsplit(url).hostname
    ok, _why = selective_fetch.is_public_url(url)  # literal IPs: no DNS and no socket
    assert ok is False


def test_C_security_non_public_destination_is_never_fetched_and_not_leaked(rig):
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [discovered()], web=web,
                          deps_mutate=lambda d: setattr(d, "url_check", lambda u: (False, f"non-public address {PRIVATE_IP}")))
    assert r.result_kind != "answer" and rig.store.plans == []
    assert not _fetched(rig, EPMC_FULLTEXT), "a destination refused by the public-address check was fetched"
    assert PRIVATE_IP not in json.dumps(r.contract or {}, default=str) and PRIVATE_IP not in (r.answer or "")


# Existing-control regression guards (units of modules shipped at 7da574a)
def test_C_licence_fail_closed_unclear_and_nd_refused():
    from vera import fulltext_loader as fl
    from vera import jats_text
    cfg = fl.load_config()
    raw = fulltext_synth.jats("syn_new", "epi", "10.9999/syn.new", seed=7)
    conv = jats_text.convert(raw, "jats-xml", cfg["converter"])
    meta = {k: v for k, v in conv.items() if k != "text"}
    for over in ({"licence": None, "licence_url": None},
                 {"licence": "CC BY-ND 4.0", "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}):
        with pytest.raises(fl.Refused):
            fl.prepare_from_parts(fulltext_synth.item("syn_new", "10.9999/syn.new", **over), conv["text"], meta, cfg)


def test_C_integrity_text_hash_mismatch_refused(tmp_path):
    from vera import fulltext_loader as fl
    cfg = fl.load_config()
    (tmp_path / "texts").mkdir()
    (tmp_path / "texts/syn_new.txt").write_bytes(b"tampered text")
    (tmp_path / "texts/syn_new.meta.json").write_text(json.dumps({"text_sha256": "0" * 64}))
    with pytest.raises(fl.Refused) as e:
        fl.load_text(fulltext_synth.item("syn_new", "10.9999/syn.new"), tmp_path, cfg)
    assert e.value.code == cfg["reason_codes"]["text_hash"]


def test_C_provenance_recorded_on_acquired_record(rig):
    """The stored plan of a discovered, admitted document carries the P0-5 provenance (discovery, URL chain, gates,
    retention) and NOT the index metadata licence as the article licence."""
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    assert len(rig.store.plans) == 1
    a = rig.store.plans[0]["source"]["provenance"]["acquisition"]
    assert a["discovery"]["record_id"] == W1 and a["resolver"] == "europepmc"
    assert a["url_chain"] == web.urls and a["gate_b"]["decision"] == "admit"
    assert a["retention"]["mode"] == "hash-only" and a["retention"]["raw_sha256"]
    assert a["licence_evidence"]["index_metadata_licence"] == "not used"
    assert "publicdomain" not in json.dumps(rig.store.plans[0]["source"]["provenance"]).lower().replace("index_metadata", "")


def test_C_relevance_offtopic_question_is_not_grounded(rig):
    rig.h.bundle()
    _decide(rig.mp, True, "layer1", "ok")
    r = ask_service.answer_in_scope(Q_NONE, "m", PRICING, grounding=(DiscSearch([]), GROUND[1]))
    assert r.result_kind != "answer"


# ------------------------------------------------------------------------------------------------- D: grounded answer
def test_D_same_request_answer_cites_passages_of_the_newly_admitted_document(rig):
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    assert r.result_kind == "answer", r.contract
    assert len(rig.store.plans) == 1 and DOI in json.dumps(rig.store.plans[0]).lower()
    assert len(rig.h.answer_calls) == 1, "the answer must be written in the SAME request"
    # the model saw, and the answer cites, passages of the new document only
    adm = r.evidence.admitted
    assert adm and all(a["document_id"].startswith("acq-") for a in adm)
    new_text = stored_text(rig.store.plans)
    passages = passages_sent_to_model(rig)
    assert passages and all(p in new_text for p in passages), "a passage did not come from the newly acquired full text"
    span_ids = {a["span_id"] for a in adm}
    assert r.citations, "the answer has no citations"
    for c in r.citations:
        sid = c["span_id"] if isinstance(c, dict) else c.span_id
        assert sid in span_ids


def test_D_abstracts_are_never_presented_as_full_documents(rig):
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    assert r.result_kind == "answer"
    blob = json.dumps([r.answer, r.contract, [a for a in r.evidence.admitted]], default=str)
    assert ABSTRACT_SENTINEL not in blob, "the discovery abstract leaked into the answer, contract or admitted evidence"
    assert ABSTRACT_SENTINEL not in json.dumps(rig.h.answer_calls, default=str), "the abstract reached the answer model"
    assert ABSTRACT_SENTINEL not in json.dumps(rig.store.plans, default=str), "the abstract was stored as document text"
    assert all(a["document_id"].startswith("acq-") for a in r.evidence.admitted)  # only acquired FULL documents


def test_D_neg_abstract_only_discovery_yields_no_answer(rig):
    """A discovered record whose full text cannot be obtained stays a link: the abstract never grounds an answer."""
    web, _ = epmc_web()
    web.records.clear()  # Europe PMC has no open-access record for the DOI
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r)
    assert e["outcome"] == "rejected" and e["reason"] == reason_code(rig, "not_oa")
    assert ABSTRACT_SENTINEL not in json.dumps(r.answer or "")


# ------------------------------------------------------------------------------------------------- E: failure explanations
def test_E_licence_unclear_is_explained_per_candidate(rig):
    web, _ = epmc_web(rec_kw={"licence": ""})
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r)
    assert e["outcome"] in ("rejected", "deferred")
    assert e["reason"] in (reason_code(rig, "no_licence"), reason_code(rig, "licence_absent"),
                           reason_code(rig, "licence_mismatch"), reason_code(rig, "not_oa"))


def test_E_no_open_access_record_is_explained_per_candidate(rig):
    web, _ = epmc_web()
    web.records.clear()
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    e = entry_for(r)
    assert e["outcome"] == "rejected" and e["reason"] == reason_code(rig, "not_oa")


def test_E_unreachable_full_text_is_explained_per_candidate(rig):
    web, _ = epmc_web(over={EPMC_FULLTEXT: requests.ConnectionError("fake: unreachable")})
    r, _ds = ask_offtopic(rig, [discovered()], web=web)
    _not_an_answer_from_new_doc(rig, r)
    e = entry_for(r)
    assert e["outcome"] == "rejected" and e["reason"] == reason_code(rig, "unreachable")


def test_E_reasons_are_not_on_the_public_candidates_list():
    """Where reasons are NOT: public candidates carry no per-candidate acquisition reason fields (design fact; explains
    why the v1 explanation_for() on candidates[] could not work)."""
    assert not {"reason", "gate_a", "outcome"} & set(grounded_ask.PUBLIC_CANDIDATE_FIELDS)


# ------------------------------------------------------------------------------------------------- F: stub
@pytest.mark.skip(reason="Phase 3 proposal, not authorized")
def test_F_no_verified_evidence_distinguishes_preliminary_synthesis_from_justified_abstention():
    """STUB. With no verified evidence the output must say which of two things it is: a labelled preliminary synthesis
    (clearly not evidence) or a justified abstention (with the reason). Acceptance criteria not yet specified."""
    raise NotImplementedError


# ================================================================================================= FIX: DELTA.md 7c
# Expected on unmodified 7da574a: FAIL (the intended behaviour is not implemented yet). Expected after the fix: PASS.
SRC_CFG = {"num_results": 5, "snippet_max_chars": 1500, "providers": [], "deadline_s": 10, "resolve": {"enabled": False}}


def _ground_cfg(**over):
    return {**SRC_CFG, **over}


# --- BR-2 / E5: discovery only when inline acquisition is enabled (ask_service.py:326)
def test_FIX_E5_no_discovery_when_inline_acquisition_is_disabled(rig, monkeypatch):
    """use_discovered true but inline.enabled false: acquire() returns 'off', so the discovery search must not be made.
    Expected now: FAIL (discovery_results is still called); after the fix: PASS."""
    calls = []
    real = acq_discovered.discovery_results
    monkeypatch.setattr(acq_discovered, "discovery_results", lambda *a, **k: calls.append(1) or real(*a, **k))
    ask_offtopic(rig, [discovered()], mutate=lambda c: c["inline"].update(enabled=False))
    assert calls == [], "discovery ran although inline acquisition is disabled"


def test_FIX_E5_control_discovery_runs_when_inline_acquisition_is_enabled(rig, monkeypatch):
    """Control for the E5 test. Expected: PASS before and after the fix."""
    calls = []
    real = acq_discovered.discovery_results
    monkeypatch.setattr(acq_discovered, "discovery_results", lambda *a, **k: calls.append(1) or real(*a, **k))
    ask_offtopic(rig, [discovered()])
    assert calls == [1]


# --- BR-3 / L-a: PMCID passthrough (grounded_ask.py:184)
def test_FIX_PMCID_is_carried_into_the_numbered_sources():
    """A discovery result that has an abstract and a PMCID but no DOI must keep its pmcid. Expected now: FAIL."""
    raw = [{"title": "T", "url": "https://example.org/a", "snippet": "abstract text", "pmcid": PMC, "provider": "openalex"}]
    out, _unused = grounded_ask.retrieve_all("q", _ground_cfg(), search=lambda q: raw)
    assert out and out[0].get("pmcid") == PMC


def test_FIX_PMCID_only_record_becomes_an_acquisition_candidate_end_to_end():
    """retrieve_all output -> acq_discovered adapter: a PMCID-only record is a candidate, not a no_doi hold. Expected
    now: FAIL (the pmcid is dropped before the adapter)."""
    raw = [{"title": "T", "url": "https://example.org/a", "snippet": "abstract text", "pmcid": PMC, "provider": "openalex"}]
    out, _ = grounded_ask.retrieve_all("q", _ground_cfg(), search=lambda q: raw)
    cands, held = acq_discovered.discovered_candidates(out, acq_config.load_config())
    assert [c.get("pmcid") for c in cands] == [PMC] and held == []


def test_FIX_PMCID_guard_public_candidates_never_expose_the_pmcid():
    """Guard for the fix (DELTA: confirm the public response strips it). Expected: PASS before and after."""
    pub = grounded_ask.public_candidates([{"n": 1, "title": "T", "url": "https://example.org/a", "pmcid": PMC,
                                           "snippet": "s", "provider": "openalex"}])
    assert "pmcid" not in json.dumps(pub, default=str).lower()


# --- BR-4 / L-b: abstract-less originals become acquisition candidates
def test_FIX_abstractless_original_keeps_its_identifiers_for_acquisition():
    """A traced original without an abstract is listed in `unused` today, WITHOUT doi/pmcid. The fix must carry them so
    the record can become an acquisition candidate (never evidence, never grounded on). Expected now: FAIL."""
    linked = [{"title": "Link only", "url": "https://example.org/landing", "snippet": ""}]

    def resolver(r, rcfg, deadline_at):
        return [{"title": "Original", "url": "https://example.org/original", "snippet": "", "doi": DOI,
                 "discovered_via": "openalex"}]
    cfg = _ground_cfg(resolve={"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5})
    out, unused = grounded_ask.retrieve_all("q", cfg, search=lambda q: linked, resolver=resolver)
    assert out == [] and len(unused) == 1  # never grounded on (holds today and after)
    assert unused[0].get("doi") == DOI


def test_FIX_abstractless_original_is_handed_to_acquisition(rig):
    """End to end via ask_service. The request's discovery seam is the REAL grounded_ask.retrieve_all result (the fake
    DiscSearch used elsewhere returns records directly and so bypasses the abstract-less drop: at 7da574a that variant
    passes). Linked result -> resolver finds an abstract-less original with a DOI. Today retrieve_all lists it only in
    `unused` (no doi) and returns nothing, so acquisition never sees it. Signal: Europe PMC is asked about the DOI.
    Expected now: FAIL; after the fix (original returned as an acquisition candidate): PASS."""
    linked = [{"title": "Link only", "url": "https://example.org/landing", "snippet": ""}]

    def resolver(r, rcfg, deadline_at):
        return [{"title": "Original", "url": f"https://doi.org/{DOI}", "snippet": "", "doi": DOI, "openalex_id": W1,
                 "discovered_via": "openalex"}]
    cfg = _ground_cfg(resolve={"enabled": True, "max_results_to_resolve": 4, "deadline_s": 5})
    out, unused = grounded_ask.retrieve_all("q", cfg, search=lambda q: linked, resolver=resolver)
    # the seam is built from the abstract-less originals (`unused`), independently of `out` (which stays empty: never evidence)
    assert out == [] and len(unused) == 1
    web, _ = epmc_web()
    r, _ds = ask_offtopic(rig, unused, web=web)
    assert [u for u in rig.web.urls if EPMC_SEARCH in u and "acq.epi4" in u], "an abstract-less record never reached acquisition"


def test_FIX_abstractless_original_reaches_acquisition_through_the_ask_boundary_as_candidate_only(rig, monkeypatch):
    """/ask boundary (main.py wiring of fixes.patch): the request's sources_fn exposes `.unused`; discovery_results
    appends it, so an abstract-less original with a DOI reaches acquisition as a CANDIDATE. It is never evidence: the
    reading list/sources stay empty, no passage is stored or sent to the model, no citation. Europe PMC has no OA record,
    so nothing can legitimately ground an answer. Expected without the patch: FAIL (acquisition never asks about the DOI)."""
    from fastapi.testclient import TestClient
    import main
    from vera import scope_router  # noqa: F401
    key = "k" * 20
    monkeypatch.setenv("VERA_API_KEY", key)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    from vera import public_mode as pm
    pm.reset_limiter(None)
    original = {"title": "Abstractless original", "url": f"https://doi.org/{DOI}", "doi": DOI, "openalex_id": W1,
                "discovered_via": "openalex"}
    searches = []
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg, **k: searches.append(1) or ([], [dict(original)]))
    monkeypatch.setattr(main, "GROUNDING", dict(SRC_CFG))
    web, _ = epmc_web()
    web.records.clear()  # no open-access record: acquisition cannot produce text
    rig.cfg["inline"]["use_discovered"] = True
    rig.setup(web=web, cands=[])
    _decide(rig.mp, False, "layer2_llm", "llm_out:x", "off_topic")
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (
        ask_service.answer_in_scope(q, "m", PRICING, grounding=g), "x"))  # g is main.py's own (sources_fn, GROUNDING)
    resp = TestClient(main.app).post("/ask", json={"question": Q_EPI}, headers={"X-API-Key": key})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert searches, "main.py's sources_fn never ran the discovery search"
    assert [u for u in rig.web.urls if EPMC_SEARCH in u and "acq.epi4" in u], "the abstract-less original never reached acquisition"
    assert body.get("result_kind") != "answer" and (body.get("sources") or []) == []
    assert not body.get("citations") and rig.store.plans == [] and rig.h.answer_calls == []
    assert "Abstractless original" not in json.dumps(body.get("sources") or [])


# --- BR-5 / L-c: discovery cap vs discovered_max
def test_discovery_cap_effective_count_is_min_of_num_results_and_discovered_max():
    """Invariant (PASS expected). Owner-side judgement: 5 vs 6 is not a pipeline defect (no config change authorized).
    Shipped values at 7da574a: grounding.num_results = 5, acq discovered_max = 6, so at most 5 discovered records are
    offered and discovered_max is not reachable. The effective discovered count must equal min(num_results, discovered_max)."""
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json").read_text())["grounding"]
    srcs = [{"title": f"T{i}", "url": f"https://example.org/{i}", "snippet": f"abstract {i}", "doi": f"10.9999/cap.{i}",
             "provider": "openalex"} for i in range(10)]
    out, _ = grounded_ask.retrieve_all("q", {**cfg, "resolve": {"enabled": False}}, search=lambda q: srcs)
    dmax = acq_discovered.discovered_max(acq_config.load_config())
    assert len(out) == min(cfg["num_results"], dmax)
