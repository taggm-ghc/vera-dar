"""DDFIX-1023/ACCEPT (copied by INTEG from work/ddfix-1023/accept, sha256 d3b5b413...b54e; P5 relaxed per the
integrator decision: reading-list sources allowed on a non-answer, no evidence/claims/citations): owner acceptance points for the live question (UI build 9b55c75)

    "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"

Tested at the public /ask service boundary, vera.ask_service.answer_in_scope(question, model, pricing,
grounding=(sources_fn, grounding_cfg)), on the unchanged offline rig (tests/_pipeline_support.py: no DB, no socket).

REAL (not faked): scope_router.route + layer 1 + verdict parsing, ask_service, inline_acquire, auto_acquire ranking and
evaluate_candidate, licence_gate, selective_fetch, fulltext_loader, corpus admission / relevance gate, research_outcome.
FAKED (boundaries only): the layer-2 classifier LLM (scope_router.default_chat), the discovery search (grounding[0]),
the acquisition search (Deps.search), HTTP (Deps.http_get: A.Web, Europe PMC JSON + JATS), the store (A.Store), the
read-only lookup, the answer writer (ask_service.complete) and the claim checker (claim_check.check_answer), the
fulldoc bundle (synthetic base corpus with no multi-agent paper), arXiv waits (arxiv_fulltext._sleep: no-op).

False positives (point 2): the five records the live run returned, all labelled "too many candidates" and none ingested
(owner evidence 2026-10-23). Real titles and identifiers only; no abstract is invented (snippet = title). Relevant
documents are SYNTHETIC (titles prefixed SYNTHETIC, invented 10.9999 DOIs, invented JATS text).
"""
import json
import re

import pytest

from tests import acq_support as A
from tests import fulltext_synth as fs
from tests._pipeline_support import PRICING, DiscSearch, h, rig  # noqa: F401  (h, rig: fixtures)
from tests.test_dar_inline_acquire import GROUND
from vera import acq_config, acq_discovered, arxiv_fulltext, ask_service, discovery_panel, scope_router

Q = "What are the de facto status quo best practices agent archetypes used in multi-agent orchestration?"
REAL_ROUTE = scope_router.route  # captured before the rig replaces it with a fixed decision
# The layer-2 classifier's realistic verdict under the shipped prompt (multi-agent orchestration is not a listed field)
VERDICT = {"in_scope": False, "category": "off_topic", "reason": "multi-agent orchestration is outside the listed fields"}

FP = [  # (doi, title, arxiv_id or None): the live run's five lexical false positives
    ("10.48550/arXiv.2512.02349", "ACM COMPUTE 2025 Best Practices Track Proceedings", "2512.02349v2"),
    ("10.1177/01492063241226918",
     "The Status of Status Research: A Review of the Types, Functions, Levels, and Audiences", None),
    ("10.3790/vaw.2025.1457703", "The U.S. Dollar Dominance: Origins, Status Quo, and Implications for Europe", None),
    ("10.48550/arXiv.2409.18288", "The track-length extension fitting algorithm for energy measurement of interacting "
     "particles in liquid argon TPCs and its performance with ProtoDUNE-SP data", "2409.18288v3"),
    ("10.48550/arXiv.2607.16922", "Pedestrian Archetypes Extension -- More Pedestrian Models for Autonomous Vehicle "
     "Safety Testing", "2607.16922v1"),
]
REL = [  # (doc_id, doi, pmcid, title, snippet): SYNTHETIC on-topic papers (multi-agent LLM orchestration, roles)
    ("mas_roles", "10.9999/ddfix.masroles.z", "PMC9100001",
     "SYNTHETIC role specialisation and coordination in LLM multi-agent systems",
     "planner executor critic roles coordinate language model agents"),
    ("mas_orch", "10.9999/ddfix.masorch.z", "PMC9100002",
     "SYNTHETIC orchestrator-worker and supervisor patterns for multi-agent LLM orchestration",
     "a supervisor delegates subtasks to worker agents through hierarchical coordination"),
]
REL_DOIS = {r[1].lower() for r in REL}
FP_DOIS = {f[0].lower() for f in FP}
# identity markers (lower case) found in a candidate record or a requested URL: the DOI suffix (URL-encoding safe) and
# the bare arXiv id (version-free)
FP_MARK = [(f[0].split("/", 1)[1].lower(), f[2].split("v")[0] if f[2] else None) for f in FP]
REL_MARK = [r[1].split("/", 1)[1].lower() for r in REL]
TOO_MANY = discovery_panel.REASON_TEXT["R-ACQ-DEFER_CAP"]  # the public label "Not checked: too many candidates ..."
MAS = ["agent", "archetypes", "orchestration", "multi-agent", "planner", "supervisor", "worker", "critic", "roles",
       "coordination", "practices", "router", "executor", "handoff", "delegation"]
ANSWER = "Planner, supervisor-worker and critic roles are the prevailing multi-agent orchestration archetypes [1]."
# status vocabulary of research_outcome.annotate_diag: a candidate given an inline slot and actually tried
ATTEMPTED = {"admitted", "fetched_rejected", "acquired_not_admitted", "selected_not_fetched"}


def _is_fp(c) -> bool:
    t = json.dumps(c, default=str).lower()
    return any(d in t or (a and a in t) for d, a in FP_MARK)


def _is_rel(c) -> bool:
    t = json.dumps(c, default=str).lower()
    return any(m in t for m in REL_MARK)


def _slot_used(c) -> bool:
    """Given an inline slot: tried, or stopped only because the per-request document budget was already met."""
    return c.get("status") in ATTEMPTED or (c.get("status") == "skipped" and c.get("status_reason") == "doc_budget")


def _words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def _fp_cand(doi, title, aid):
    c = {"doi": doi, "url": f"https://arxiv.org/abs/{aid}" if aid else f"https://doi.org/{doi}", "title": title,
         "snippet": title, "provider": "openalex"}
    return c


def _rel_cand(_did, doi, _pmc, title, snippet):
    return {"doi": doi, "url": f"https://doi.org/{doi}", "title": title, "snippet": snippet, "provider": "openalex"}


def _web(licence_url="https://creativecommons.org/licenses/by/4.0/", over=None, rel=REL):
    recs, jats = {}, {}
    for did, doi, pmc, title, _s in rel:
        recs[doi] = A.epmc_record(doi, pmcid=pmc, title=title)
        jats[pmc] = fs.jats(did, "mas", doi, licence_url=licence_url)
    return A.Web(recs, jats, over)  # unknown URLs (every false positive) -> 404


@pytest.fixture
def acc(rig, monkeypatch):  # noqa: F811
    monkeypatch.setitem(fs.TOPICS, "mas", MAS)
    acq_discovered.reset_for_tests()
    acq_discovered._ROBOTS.clear()
    getattr(arxiv_fulltext, "_HOST_LAST", {}).clear()
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    monkeypatch.setattr(scope_router, "route", REAL_ROUTE)  # the REAL scope gate decides
    calls = []

    def classifier(messages, model, l2, client=None, extra_body=None):  # the layer-2 LLM boundary only
        calls.append(messages)
        return json.dumps(VERDICT), 10, 5
    monkeypatch.setattr(scope_router, "default_chat", classifier)
    rig.classifier_calls = calls
    rig.h.answer_text = ANSWER
    yield rig
    getattr(arxiv_fulltext, "_HOST_LAST", {}).clear()
    acq_discovered._ROBOTS.clear()


def _ask(rig, cands, web, q=Q):
    # INTEG (livechk-real finding): the reading-list seam returns records NUMBERED as production's retrieve_all does
    # (grounded_ask.py:180 assigns int n to every source); unnumbered fakes made the HTTP route fail AskResponse.
    cands = [{**c, "n": i} for i, c in enumerate(cands, 1)]
    rig.setup(web=web, cands=cands)
    ds = DiscSearch(cands)  # discovery returns the same records (same providers)
    r = ask_service.answer_in_scope(q, "m", PRICING, grounding=(ds, GROUND[1]))
    ia = (r.contract or {}).get("inline_acquisition") or {}
    return r, ds, ia


def _cand_for(ia, marker) -> dict:
    """The candidate record whose identifiers contain `marker` (DOI suffix or arXiv id), else {}."""
    hits = [c for c in ia.get("candidates") or () if isinstance(c, dict) and marker in json.dumps(c, default=str).lower()]
    hits.sort(key=lambda c: bool(c.get("held")))  # the acquisition record before a held discovery duplicate
    return hits[0] if hits else {}


def _mixed(rig):
    cands = [_fp_cand(*f) for f in FP] + [_rel_cand(*x) for x in REL]  # providers' order: false positives first
    return _ask(rig, cands, _web())


def _cited_dois(rig, r):
    by_doc = {p["source"]["provenance"]["document_id"]: p for p in rig.store.plans}
    out = set()
    for s in r.sources or ():
        p = by_doc.get(str(s.get("document_id")))
        out.add(json.dumps(p["source"]["provenance"]).lower() if p else "?")
    return {d for d in REL_DOIS | FP_DOIS if any(d in o for o in out)}, out


# ------------------------------------------------------------------------------------------------ fixture sanity
def test_fixture_false_positives_are_lexically_competitive():
    """The five live titles share at least as many question words as the synthetic relevant papers (so ranking by
    lexical overlap alone cannot be relied on). Fixture-local count; no product function involved."""
    q = _words(Q)
    fp = [len(q & _words(t)) for _d, t, _a in FP]
    rel = [len(q & _words(f"{t} {s}")) for _i, _d, _p, t, s in REL]
    assert max(fp) >= min(rel), (fp, rel)


def test_control_relevant_fixture_text_is_admissible_without_the_idioms(acc):
    """Control (not an owner point): the same synthetic relevant full texts ARE admitted by the real gates and answer
    when the question drops the idioms "de facto" / "status quo" / "best". So a p3/p4 failure on the live question is
    the question's content stems, not unusable fixture text."""
    q = "Which agent archetypes and practices are used in multi-agent orchestration?"
    r, _ds, ia = _ask(acc, [_rel_cand(*x) for x in REL], _web(), q=q)
    assert all(_cand_for(ia, m).get("status") == "admitted" for m in REL_MARK), ia.get("candidates")
    assert r.result_kind == "answer", (r.result_kind, r.answer)


# ------------------------------------------------------------------------------------------------ point 1
def test_p1_unfamiliar_research_domain_triggers_research_not_a_subject_matter_refusal(acc):
    r, ds, ia = _ask(acc, [_rel_cand(*x) for x in REL], _web())  # only relevant records: isolates the scope gate
    assert acc.classifier_calls, "the real router never consulted the layer-2 classifier"
    assert ds.calls, "discovery was not called"
    assert acc.search_calls, "no acquisition search was made (no research)"
    assert ia.get("status") not in (None, "off"), ia
    se = acc.cfg["scope_expansion"]
    assert r.result_kind != "decline" and scope_router.decline_text() not in (r.answer or ""), r.answer
    assert scope_router.offtopic_notice() not in (r.answer or ""), "subject-matter (off-topic) notice shown"
    assert not (r.answer or "").startswith(se["override_answer_prefix"]), "answer framed as outside VERA-DAR's scope"
    assert (r.contract or {}).get("offtopic") is not True, r.contract


# ------------------------------------------------------------------------------------------------ point 2
# Orchestrator decision 2026-10-23 (owner brief: "relevant publications outrank lexical false positives"; "improve
# relevance assessment before allocating scarce inline slots"): every relevant record ranks above every false positive;
# no false positive is selected while a relevant candidate is left unselected (displaced by the cap); a record matching
# NO question concept is dropped (R-ACQ-IRRELEVANT); a WEAK lexical match ("Pedestrian Archetypes" shares "archetypes")
# may use a spare slot only after every relevant record was selected, and admission then judges it.
WEAK_FP = {"2607.16922"}  # arXiv id of the one live false positive sharing a question concept
IRRELEVANT = "R-ACQ-IRRELEVANT"


def _is_weak(c) -> bool:
    t = json.dumps(c, default=str).lower()
    return any(a in t for a in WEAK_FP)


def _recs_for(ia, mark):
    return [c for c in list(ia.get("discovered") or ()) + list(ia.get("candidates") or ())
            if isinstance(c, dict) and mark in json.dumps(c, default=str).lower()]


def _p2_checks(acc, ia, rel_marks):
    cands = [c for c in ia.get("candidates") or () if isinstance(c, dict)]
    fps = [c for c in cands if _is_fp(c)]
    for mark in rel_marks:  # every relevant paper was given an inline slot; none of its records says "too many"
        recs = _recs_for(ia, mark)
        assert any(_slot_used(c) for c in recs), (mark, "relevant paper got no inline slot", recs)
        assert not [c for c in recs if c.get("status_reason") in ("candidate_cap", "ranked_below_cap")
                    or discovery_panel._reason_line(c) == TOO_MANY], (mark, "relevant paper deferred as over the cap", recs)
    # every relevant record ranks above every false positive (the inline candidate record is in rank order)
    rel_i = [i for i, c in enumerate(cands) if any(m in json.dumps(c, default=str).lower() for m in rel_marks)]
    fp_i = [i for i, c in enumerate(cands) if _is_fp(c)]
    assert rel_i and (not fp_i or max(rel_i) < min(fp_i)), (rel_i, fp_i)
    # a record matching no concept is dropped as irrelevant: never given a slot, never requested, never stored
    strong = [c for c in fps if not _is_weak(c)]
    assert all(c.get("reason") == IRRELEVANT and not _slot_used(c) for c in strong), \
        [(c.get("public_id"), c.get("status"), c.get("reason")) for c in strong]
    hit = [u for u in acc.web.urls if any((d in u.lower() or (a and a in u)) and a not in WEAK_FP for d, a in FP_MARK)]
    assert not hit, ("no-concept false-positive records were requested", hit)
    stored = " ".join(json.dumps(p["source"]["provenance"]).lower() for p in acc.store.plans)
    assert not any(d in stored for d, _a in FP_MARK)
    # no false positive is selected while a relevant candidate is left unselected
    if any(_slot_used(c) for c in fps):
        assert all(any(_slot_used(c) for c in _recs_for(ia, m)) for m in rel_marks)
    # no listed candidate may show "too many candidates" while fewer than the inline slot count were attempted
    slots = int(acq_config.bound(acc.cfg, "max_candidates_resolved"))
    used = sum(1 for c in cands if _slot_used(c))
    listed = [e for e in list(ia.get("discovered") or ()) + cands if isinstance(e, dict)]
    capped = [e.get("public_id") or e.get("url") or e.get("record_id") for e in listed
              if discovery_panel._reason_line(e) == TOO_MANY]
    assert not capped or used >= slots, (f"{len(capped)} labelled {TOO_MANY!r} but only {used} of {slots} inline "
                                         f"slots were attempted", capped)
    return cands, fps, slots


def test_p2_relevant_papers_outrank_lexical_false_positives_before_slots(acc):
    _r, _ds, ia = _mixed(acc)
    _p2_checks(acc, ia, REL_MARK)


# SYNTHETIC: as many relevant papers as there are inline slots (max_candidates_resolved 6)
REL_MANY = [(f"mas_many{i}", f"10.9999/ddfix.masmany{i}.z", f"PMC91002{i:02d}",
             f"SYNTHETIC agent archetypes in multi-agent orchestration, study {i}",
             "planner supervisor worker critic roles coordinate language model agents") for i in range(6)]


def test_p2_relevant_candidates_fill_every_slot_and_no_false_positive_is_selected(acc):
    marks = [r[1].split("/", 1)[1].lower() for r in REL_MANY]
    cands = [_fp_cand(*f) for f in FP] + [_rel_cand(*x) for x in REL_MANY]  # false positives first, as live
    _r, _ds, ia = _ask(acc, cands, _web(rel=REL_MANY))
    got, fps, slots = _p2_checks(acc, ia, marks[:int(acq_config.bound(acc.cfg, "max_candidates_resolved"))])
    assert len(REL_MANY) >= slots
    assert not [c for c in fps if _slot_used(c)], [(c.get("public_id"), c.get("status")) for c in fps if _slot_used(c)]
    hit = [u for u in acc.web.urls if any(d in u.lower() or (a and a in u) for d, a in FP_MARK)]
    assert not hit, ("a false positive was requested although relevant candidates filled every slot", hit)


# ------------------------------------------------------------------------------------------------ point 3
def test_p3_eligible_licensed_full_text_reaches_acquisition_and_passes_admission(acc):
    r, _ds, ia = _mixed(acc)
    fetched = " ".join(acc.web.urls)
    for (_did, doi, pmc, _t, _s), mark in zip(REL, REL_MARK):
        c = _cand_for(ia, mark)
        assert c.get("status_reason") != "candidate_cap", c
        assert f"/{pmc}/fullTextXML" in fetched, (doi, "full text never fetched")
        assert c.get("status") == "admitted", c  # real licence gate + selective fetch + admission passed
    stored = " ".join(json.dumps(p["source"]["provenance"]).lower() for p in acc.store.plans)
    assert all(d in stored for d in REL_DOIS)
    assert all(int(p.get("admitted_spans") or 0) >= 1 for p in acc.store.plans), [p.get("admitted_spans") for p in acc.store.plans]


# ------------------------------------------------------------------------------------------------ point 4
def test_p4_admitted_evidence_supports_a_cited_answer(acc):
    r, _ds, _ia = _mixed(acc)
    assert r.result_kind == "answer", (r.result_kind, r.answer, (r.contract or {}).get("fallback"))
    assert r.evidence is not None and r.evidence.admitted
    adm = {str(a.get("document_id")) for a in r.evidence.admitted}
    cited = {str(s.get("document_id")) for s in r.sources or ()}
    assert cited and cited <= adm, (cited, adm)
    hit, raw = _cited_dois(acc, r)
    assert hit and hit <= REL_DOIS, ("citations are not the relevant papers", hit, raw)
    assert len(acc.h.answer_calls) == 1 and len(acc.h.check_calls) == 1  # the writer and the claim check ran once
    stored_text = "".join(s["text"] for p in acc.store.plans for s in p["spans"])
    sent = [p["text"] for p in json.loads(acc.h.answer_calls[0][-1]["content"])["passages"]]
    assert sent and all(t in stored_text for t in sent)  # the model saw only verbatim admitted full-text passages


# ------------------------------------------------------------------------------------------------ point 5
@pytest.mark.parametrize("failure", ["fetch_and_licence", "all_unreachable"])
def test_p5_no_verifiable_evidence_gives_a_justified_abstention(acc, failure):
    if failure == "fetch_and_licence":  # one fetch fails (503), the other is CC BY-ND (real licence gate rejects)
        web = _web(licence_url="https://creativecommons.org/licenses/by-nd/4.0/",
                   over={f"/{REL[0][2]}/fullTextXML": A.Resp(status=503)})
    else:
        web = _web(over={"fullTextXML": A.Resp(status=503)})
    r, _ds, ia = _ask(acc, [_rel_cand(*x) for x in REL], web)
    assert r.result_kind != "answer", (r.result_kind, r.answer)
    # INTEG decision (DDFIX-1023): no answer, no admitted evidence and no citations. Reading-list `sources` on a non-answer
    # are allowed (the in-scope L0 shape); they are discovery records, never evidence.
    assert not (r.evidence and r.evidence.admitted)
    assert not r.claims and not r.citations and not (r.claim_check or {}).get("claims")
    assert not re.search(r"\[\d+\]", r.answer or ""), r.answer  # no citation markers in the user text
    assert all(not s.get("span") and not s.get("text_sha256") for s in r.sources or ()), r.sources  # no evidence spans
    assert acc.store.plans == []
    assert ANSWER not in (r.answer or "")  # no unsupported generated text reaches the user
    cands = ia.get("candidates") or []
    assert cands and all(c.get("status") and c.get("status_reason") for c in cands), cands  # justified per candidate
    assert not any(c.get("status") == "admitted" for c in cands)
    se = acc.cfg["scope_expansion"]
    assert any(se[k] in (r.answer or "") for k in ("inline_notice_none", "inline_notice_unanswered")), r.answer
    if failure == "fetch_and_licence":
        nd = _cand_for(ia, REL_MARK[1])
        assert nd.get("status_reason") == "licence_rejected", nd


# ------------------------------------------------------------------------------------------------ point 5 over HTTP
@pytest.mark.parametrize("failure", ["fetch_and_licence", "all_unreachable"])
def test_p5_http_route_returns_a_valid_abstention(acc, monkeypatch, failure):
    """The real /ask route (TestClient on main.app, AskResponse validation) with the same fakes: HTTP 200, a non-answer,
    no admitted evidence and no citations (livechk-real saw HTTP 500 with unnumbered fake candidates)."""
    import main
    from fastapi.testclient import TestClient
    from tests import test_fulldoc_e2e as e2e
    from vera import grounded_ask
    if failure == "fetch_and_licence":
        web = _web(licence_url="https://creativecommons.org/licenses/by-nd/4.0/",
                   over={f"/{REL[0][2]}/fullTextXML": A.Resp(status=503)})
    else:
        web = _web(over={"fullTextXML": A.Resp(status=503)})
    cands = [{**_rel_cand(*x), "n": i} for i, x in enumerate(REL, 1)]
    acc.setup(web=web, cands=cands)
    monkeypatch.setattr(main, "retrieve_all", lambda question, cfg, **k: ([dict(c) for c in cands], []))
    monkeypatch.setattr(main, "GROUNDING", {**(grounded_ask.load_grounding() or {}), **e2e.GROUNDING})
    resp = TestClient(main.app).post("/ask", json={"question": Q}, headers={"X-API-Key": "k" * 20})
    assert resp.status_code == 200, resp.text[:500]
    body = resp.json()
    assert body["result_kind"] != "answer" and not body.get("admitted_evidence") and not body.get("citations"), body
    assert ANSWER not in (body.get("answer") or "")
    assert all(isinstance(c.get("n"), int) for c in body.get("candidates") or ())
    assert acc.store.plans == []
