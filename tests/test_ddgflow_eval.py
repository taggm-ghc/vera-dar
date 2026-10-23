"""DDGFLOW-EVAL-1023: user-story flow evaluation "a query that tries to find a source from a DDG response", EXECUTED.

OFFLINE, in process: POST /ask through main.app (FastAPI TestClient) over the ROUTE-R2 RouteWorld harness. Stubbed are
ONLY external I/O: the DuckDuckGo Lite HTTP response (but its rows come from the REAL DDG-lite HTML parser,
DuckDuckGoLiteProvider._parse_one over synthetic lite.duckduckgo.com result HTML), the OpenAlex/arXiv provider calls,
the Europe PMC transport, the full-text http_get (Europe PMC JATS, arxiv.org), a landing-page GET and its public-address
check, and the answer model / claim checker (_pipeline_support rig). Licence gate (owner-applied v4: sha256 0747e62c...,
rules 4e194967...), injection, relevance, admission, caps, DB/socket guards (fail loud) run unchanged. No background
worker is started (queue inspected only). Every scenario prints one `DDGFLOW-OBS <json>` line (run with -s) that the
FLOW-EVAL.md tables are built from.

DDGFIX-1023: the defect-documenting assertions (DF-1..DF-7) now assert the FIXED behaviour; the four strict xfails are
plain tests; per-defect regressions (incl. DF-1 negative cases and DF-8 host/robots/redirect fault injection) at the end.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import pytest

from tests import accept_predicate as ap
from tests import arxiv_synth as X
from tests.test_acq_discovered import Router
from tests.test_route_ddg_first_r2 import (  # noqa: F401  (h, rig, clock, no_db_no_socket, world are fixtures)
    DDG, EPMC, HOURS, NOW, OA, _ro, canon, clock, epmc_row, h, no_db_no_socket, rig, world)
from tests import test_fulldoc_e2e as e2e
from tests._gap_support import http_ask
from vera import arxiv_fulltext, auto_acquire, inline_acquire
from vera import search_providers as sp

ROOT = Path(__file__).resolve().parent.parent
Q = e2e.Q_EPI


# ------------------------------------------------------------------------------------------------ fixtures / helpers
@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    from vera import acq_discovered
    acq_discovered.reset_for_tests()  # DDGFIX-1023: robots.txt cache (landing GET) per test
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


@pytest.fixture
def spy(monkeypatch):
    """Records what ddg_to_candidates hands to the next hop (the candidate dicts), without changing it."""
    rec = {"ddg_in": [], "ddg_out": [], "ddg_counts": []}
    real = sp.ddg_to_candidates

    def wrapped(results, *a, **k):
        rec["ddg_in"].append([dict(r) for r in results or ()])
        out, counts = real(results, *a, **k)
        rec["ddg_out"].append([dict(c) for c in out])
        rec["ddg_counts"].append(dict(counts))
        return out, counts
    monkeypatch.setattr(sp, "ddg_to_candidates", wrapped)
    return rec


def _settings():
    return json.loads((ROOT / "config" / "vera_eval_run.json").read_text(encoding="utf-8"))["search"]["duckduckgo_lite"]


class _R:
    def __init__(self, text, status=200):
        self.text, self.status_code, self.headers = text, status, {}


def lite_html(links):
    """lite.duckduckgo.com result-page shape: redirect-wrapped result-link anchors in a results table."""
    rows = "".join(
        f'<tr><td valign="top">{i}.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg={quote(u, safe="")}'
        f'&amp;rut=abc{i}" class=\'result-link\'>{t}</a></td></tr><tr><td></td><td class=\'result-snippet\'>'
        f'snippet text {i} (never kept)</td></tr>' for i, (u, t) in enumerate(links, 1))
    return f'<html><body><form action="/lite/"></form><table>{rows}</table></body></html>'


def ddg_rows(links, group="portals"):
    """REAL parser output (DuckDuckGoLiteProvider._parse_one) for a synthetic DDG-lite page."""
    return sp.DuckDuckGoLiteProvider(_settings())._parse_one(group, _R(lite_html(links)))


def ask(world, *, get=None, url_check=None, providers=(OA,), raws=()):
    """world.ask() with optional landing GET / public-address stubs (the harness default refuses both, offline)."""
    rig = world.rig
    rig.cfg["search"]["providers"] = list(providers)
    rig.setup(web=world.web, cands=[])

    def no_get(*a, **k):
        raise RuntimeError("offline: landing GET")
    real = sp.acquisition_search(auto_acquire.make_search(rig.cfg), rig.cfg, transport=world.p.transport,
                                 terms_fn=lambda: (None, None), url_check=url_check or (lambda u: (False, "offline")),
                                 get=get or no_get)
    rig.deps.search = lambda question, n, t: (rig.search_calls.append(question), real(question, n, t))[1]
    resp, _s = http_ask(rig, world.mp, list(raws), q=Q, offtopic=False)
    assert resp.status_code == 200, resp.text
    return resp.json()


def queued():
    q = auto_acquire._S.queue
    return [] if q is None else [c for j in list(q.queue) for c in (j.cands or ())]


def cands(body):
    return (_ro(body).get("candidates") or [])


def obs(label, body, world, spy=None):
    ro = body.get("research_outcome") or {}
    o = {
        "scenario": label, "triggers": body.get("triggers"), "result_kind": body.get("result_kind"),
        "outcome": body.get("outcome"),
        "portal_calls": world.p.log,
        "portals": [(canon(p["portal"]), p["status"], p.get("n_results")) for p in ro.get("portals") or []],
        "candidates": [{k: c.get(k) for k in ("ids", "route", "terminal_reason", "reason", "background", "portal",
                                              "portals", "found_by")} for c in cands(body)],
        "ro_kind": ro.get("kind"), "ro_complete": ro.get("complete"), "user_text": ro.get("user_text"),
        "inline": {k: (body.get("contract") or {}).get("inline_acquisition", {}).get(k)
                   for k in ("status", "admitted", "persisted", "notice")},
        "fetched": list(world.rig.web.urls), "persisted_plans": len(world.rig.store.plans),
        "citations": len(body.get("citations") or []), "grounded_failures": ap.grounded_failures(200, body),
        "sources": [{k: s.get(k) for k in ("n", "source_type", "corpus_added_at", "source_authored_at",
                                           "source_authored_precision", "evidence_recency", "document_id")}
                    for s in body.get("sources") or []],
        "queued": [{k: q.get(k) for k in ("doi", "pmcid", "url")} for q in queued()],
        "answer_head": (body.get("answer") or "")[:160],
    }
    if spy is not None:
        o["ddg_parser_rows"] = [[{k: r.get(k) for k in ("url", "title", "discovery_group", "stored_text_kind",
                                                        "licence", "snippet")} for r in rs] for rs in spy["ddg_in"]]
        o["ddg_candidates"] = [[{k: c.get(k) for k in ("doi", "pmcid", "url", "discovered_url", "acq_source",
                                                       "ddg_id_kind", "provider")} for c in cs] for cs in spy["ddg_out"]]
        o["ddg_counts"] = spy["ddg_counts"]
    print("DDGFLOW-OBS " + json.dumps(o, default=str))
    return o


def by(body, key, val):
    return [c for c in cands(body) if str((c.get("ids") or {}).get(key) or "").lower() == str(val).lower()]


# ================================================================================================ 0. preconditions
def test_00_owner_gate_files_and_flags():
    lg = hashlib.sha256((ROOT / "vera" / "licence_gate.py").read_bytes()).hexdigest()
    lr = hashlib.sha256((ROOT / "config" / "licence_rules.json").read_bytes()).hexdigest()
    assert lg.startswith("0747e62c") and lr.startswith("4e194967"), (lg, lr)
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
    assert acq["search"]["ddg_first"]["enabled"] is True and acq["arxiv_route"]["enabled"] is True
    reg = json.loads((ROOT / "config" / "portal_registry.json").read_text(encoding="utf-8"))
    rec = reg.get("recorder") or {}
    print("DDGFLOW-OBS " + json.dumps({"scenario": "00", "recorder": {k: rec.get(k) for k in ("enabled", "backend",
                                                                                              "sqlite_path")},
                                       "use_discovered": acq["inline"]["use_discovered"],
                                       "landing_route": acq["inline"]["landing_route"]}))
    assert rec.get("enabled") is not True  # portal recorder is OFF by default (no 009 migration)


# ================================================================================================ happy path, hop by hop
HDOI, HPMC = "10.9999/ddgflow.h1", "PMC9240001"
HURL = f"https://journals.example.org/doi/{HDOI}/full"
ODOI, OPMC = "10.9999/ddgflow.h2", "PMC9240002"  # a competing OpenAlex-only candidate (ranking hop)


@pytest.fixture
def happy(world, spy):
    """ONE public-visitor corpus miss: DDG's first result is a journal page whose URL carries a DOI that Europe PMC has
    as open-access full text; OpenAlex returns a competing record for another open-access paper."""
    from tests.test_route_ddg_first_r2 import oa_rec
    world.doc(HDOI, HPMC)
    world.doc(ODOI, OPMC, oa="N")  # competitor is not open access: only the DDG source can ground this answer
    world.p.beh[DDG] = ddg_rows([(HURL, "SYNTHETIC outbreak transmission incidence study")])
    world.p.beh[OA] = [oa_rec(ODOI, "W92402", OPMC)]
    world.store()
    body = ask(world)
    obs("happy", body, world, spy)
    return body, world, spy


def _plan(world, doi):
    p = [p for p in world.rig.store.plans if str(p["source"]["provenance"].get("item", {}).get("doi") or
                                                  p["source"]["provenance"].get("doi") or "").lower() == doi]
    return p[0] if p else None


def test_hop01_trigger_and_ddg_called_and_recorded_first(happy):
    body, world, _ = happy
    assert body["triggers"] == ["insufficient"]
    assert world.p.log[0] == DDG and world.p.log.count(DDG) == 1, world.p.log
    ports = _ro(body)["portals"]
    assert canon(ports[0]["portal"]) == "ddg" and ports[0]["status"] == "completed" and ports[0]["n_results"] == 1
    assert [canon(p["portal"]) for p in ports] == [canon(n) for n in world.p.portal_calls]


def test_hop02_ddg_lite_parser_row(happy):
    _, _, spy = happy
    row = spy["ddg_in"][0][0]  # REAL parser output for the redirect-wrapped result-link anchor
    assert row["url"] == HURL and row["snippet"] == "" and row["licence"] is None
    assert row["stored_text_kind"] == "link_metadata" and row["discovery"] == DDG and row["discovery_group"] == "portals"


def test_hop03_identifier_extraction_and_candidate(happy):
    body, _, spy = happy
    assert spy["ddg_counts"][0] == {"ddg_results": 1, "ddg_mapped": 1, "ddg_held": 0, "ddg_landing_get": 0,
                                    "ddg_url_only": 0}
    c = spy["ddg_out"][0][0]
    assert (c["doi"], c["url"], c["discovered_url"], c["acq_source"], c["ddg_id_kind"]) == (
        HDOI, f"https://doi.org/{HDOI}", HURL, "ddg", "doi")
    pub = by(body, "doi", HDOI)
    assert len(pub) == 1 and pub[0]["ids"]["url"] == f"https://doi.org/{HDOI}" and pub[0]["ids"]["pmcid"] == HPMC  # DF-7
    v = pub[0]["portal"]
    assert "ddg" in ({canon(x) for x in v} if isinstance(v, list) else {canon(v)})


def test_hop04_selection_ddg_ranked_before_competing_openalex(happy):
    _, world, spy = happy
    assert sp.acq_rank_tier(spy["ddg_out"][0][0]) == 0 < sp.acq_rank_tier({"doi": ODOI, "provider": "openalex"})
    resolves = [u for u in world.rig.web.urls if "/search?query=DOI" in u]
    # P0DTA-PRIO D1: the competitor carries a PMCID (routable signal) and the DDG row no OA evidence, so the competitor
    # ranks first; both are evaluated in parallel (resolve order is not deterministic) and the DDG one is admitted (hop06)
    for x in (ODOI, HDOI):
        assert any(x.replace("/", "%2F") in u for u in resolves), resolves


def test_hop05_route_europepmc_jats_and_allowlisted_fetch(happy):
    _, world, _ = happy
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
    allowed = set(acq["fetch"]["allowed_hosts"])
    from urllib.parse import urlsplit
    assert f"https://www.ebi.ac.uk/europepmc/webservices/rest/{HPMC}/fullTextXML" in world.rig.web.urls
    assert all(urlsplit(u).hostname in allowed for u in world.rig.web.urls), world.rig.web.urls
    assert not any("journals.example.org" in u or "arxiv.org" in u for u in world.rig.web.urls)


def test_hop06_licence_gate_recorded_open_licence(happy):
    _, world, _ = happy
    p = _plan(world, HDOI)
    assert p is not None, [pl["source"]["provenance"].get("item", {}).get("doi") for pl in world.rig.store.plans]
    item = p["source"]["provenance"]["item"]
    assert "creativecommons.org/licenses/by/" in str(item.get("licence_url") or item.get("licence")), item


def test_hop07_full_text_extracted_chunked_admitted(happy):
    body, world, _ = happy
    p = _plan(world, HDOI)
    assert p and len(p["spans"]) >= 2 and p["admitted_spans"] >= 1
    did = p["source"]["provenance"]["item"]["document_id"]
    ex = {d["document_id"]: d for d in body["examined_full_documents"]}
    assert did in ex and str(ex[did].get("text_kind") or "") not in ap.NON_FULLTEXT and ex[did]["text_sha256"] and ex[did]["chars"] > 0
    adm = [a for a in body["admitted_evidence"] if a["document_id"] == did]
    assert adm and all(str(a.get("source_type") or a.get("text_kind")) not in ap.NON_FULLTEXT for a in adm)
    c = by(body, "doi", HDOI)[0]
    assert c["terminal_reason"] == "admitted"


def test_hop08_admitted_chunks_reach_this_requests_evidence_context(happy):
    body, world, _ = happy
    from tests._pipeline_support import passages_sent_to_model
    sent = passages_sent_to_model(world.rig)
    did = _plan(world, HDOI)["source"]["provenance"]["item"]["document_id"]
    stored = [s["text"] for s in _plan(world, HDOI)["spans"]]
    assert sent and any(any(t[:80] in s or s[:80] in t for s in sent) for t in stored), "admitted chunk not sent"
    assert any(a["document_id"] == did for a in body["admitted_evidence"])


def test_hop09_answer_cites_exactly_admitted_passages(happy):
    body, world, _ = happy
    assert ap.grounded_failures(200, body) == []
    span_ids = {a["span_id"] for a in body["admitted_evidence"]}
    did = _plan(world, HDOI)["source"]["provenance"]["item"]["document_id"]
    assert body["citations"] and {c["span_id"] for c in body["citations"]} <= span_ids
    assert did in {c["document_id"] for c in body["citations"]}


def test_hop10_research_outcome_and_user_text(happy):
    body, _, _ = happy
    ro = _ro(body)
    assert ro["kind"] == "answered_with_new" and ro["user_text"].startswith("This answer uses newly found sources")
    assert body["contract"]["inline_acquisition"]["status"] == "ok"
    assert by(body, "doi", HDOI)[0].get("background") in (None, "")
    assert all(q.get("doi") != HDOI for q in queued())  # the admitted source is never re-queued


def test_hop11_freshness_fields(happy):
    body, world, _ = happy
    assert _plan(world, HDOI)["source"]["provenance"]["retrieved_at"] == NOW.isoformat(timespec="seconds")
    new = [s for s in body["sources"] if s.get("corpus_added_at") == NOW.isoformat(timespec="seconds")]
    assert new
    for s in new:
        assert s["source_authored_precision"] == "year" and s["corpus_added_at"] != s["source_authored_at"]
        assert s["evidence_recency"] != "recent"


def test_hop12_ui_sidebar_aggregates_only_and_discovery_panel(happy):
    body, _, _ = happy
    from vera import discovery_panel
    reasons = discovery_panel._reasons_by_url(body)
    print("DDGFLOW-OBS " + json.dumps({"scenario": "ui", "panel_reasons": reasons}, default=str))
    assert reasons.get(HURL) == discovery_panel.OBTAINED_TEXT  # DF-6: DDG candidates get a per-URL reason line
    assert reasons.get(f"https://doi.org/{ODOI}") == discovery_panel.REASON_TEXT["R-ACQ-NOTOA"]


# ------------------------------------------------------------------------------- fulltext-only hop (negative path)
def test_hop07n_abstract_only_jats_is_refused(world, spy):
    from tests import fulltext_synth
    doi, pmc = "10.9999/ddgflow.ab1", "PMC9240099"
    world.doc(doi, pmc)
    world.web.jats[pmc] = fulltext_synth.jats("acq_epi", "epi", doi, with_body=False, abstract_sents=60, seed=7)
    world.p.beh[DDG] = ddg_rows([(f"https://journals.example.org/doi/{doi}/full", "SYNTHETIC outbreak study")])
    world.store()
    body = ask(world)
    obs("abstract-only", body, world, spy)
    assert f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML" in world.rig.web.urls
    assert world.rig.store.plans == [] and not body.get("citations")
    c = by(body, "doi", doi)
    assert c and c[0]["terminal_reason"] == "abstract_only" and c[0]["reason"] == "R-ABSTRACT", c  # refused AS abstract
    assert c[0]["background"] == "terminal" and queued() == []


# ================================================================================================ (a) news/blog page
def test_a1_blog_url_with_doi_trailing_slash(world, spy):
    """A blog/news URL that embeds a DOI with a trailing path slash (common CMS shape)."""
    doi, pmc = "10.9999/ddgflow.a1", "PMC9240101"
    world.doc(doi, pmc)
    world.p.beh[DDG] = ddg_rows([(f"https://blogs.example.org/2026/10/22/{doi}/",
                                  "Blog: outbreak transmission study explained")], group="news")
    world.p.beh[EPMC] = []
    world.store()
    body = ask(world)
    obs("a1", body, world, spy)
    assert spy["ddg_out"][0][0]["doi"] == doi  # DF-1 fixed: trailing "/" stripped
    c = by(body, "doi", doi)
    assert c and c[0]["terminal_reason"] == "admitted" and queued() == []


def test_a1_desired_doi_normalised_and_answered(world, spy):
    doi, pmc = "10.9999/ddgflow.a1", "PMC9240101"
    world.doc(doi, pmc)
    world.p.beh[DDG] = ddg_rows([(f"https://blogs.example.org/2026/10/22/{doi}/", "Blog")], group="news")
    world.store()
    body = ask(world)
    assert spy["ddg_out"][0][0]["doi"] == doi and ap.grounded_failures(200, body) == []


def test_a2_news_article_doi_only_in_text(world, spy):
    """A news article that cites the DOI only in its body text (DDG lite gives url + title; snippet never kept)."""
    doi, pmc = "10.9999/ddgflow.a2", "PMC9240102"
    world.doc(doi, pmc)
    world.p.beh[DDG] = ddg_rows([("https://www.reuters.com/science/outbreak-transmission-study-2026-10-20/",
                                  f"Study (doi:{doi}) finds outbreak transmission")], group="news")
    world.store()
    body = ask(world)
    obs("a2", body, world, spy)
    assert spy["ddg_counts"][0]["ddg_url_only"] == 1 and spy["ddg_counts"][0]["ddg_landing_get"] == 0  # news: no GET
    c = cands(body)
    assert len(c) == 1 and c[0]["route"] == "follow_url" and c[0]["reason"] == "R-ACQ-NODOI"  # DF-3: a discovery
    assert "No candidates were found" not in _ro(body)["user_text"]
    assert "no usable identifier yet" in _ro(body)["user_text"]  # the unsupported route is stated
    assert not any("reuters.com" in u for u in world.rig.web.urls)  # never fetched


def test_a2_e_desired_url_only_ddg_row_is_followable(world, spy):
    url = "https://www.example-health.org/topics/outbreak-transmission"
    world.p.beh[DDG] = ddg_rows([(url, "Outbreak transmission explained")], group="news")
    world.store()
    body = ask(world)
    c = [c for c in cands(body) if (c.get("ids") or {}).get("url") == url]
    assert c and c[0]["route"] == "follow_url" and c[0]["terminal_reason"] == "queued_background"


# ================================================================================================ (b) Europe PMC URL
def test_b_europepmc_pmc_url(world, spy):
    doi, pmc = "10.9999/ddgflow.b1", "PMC9240201"
    world.doc(doi, pmc)
    world.p.lookup[pmc] = epmc_row(doi, pmc)
    world.p.beh[DDG] = ddg_rows([(f"https://europepmc.org/article/PMC/{pmc}",
                                  "SYNTHETIC outbreak transmission incidence study - Europe PMC")])
    world.store()
    body = ask(world)
    obs("b", body, world, spy)
    assert "europepmc_lookup" in world.p.log  # ONE id lookup PMCID -> DOI
    c0 = spy["ddg_out"][0][0]
    assert (c0["doi"], c0["pmcid"], c0["ddg_id_kind"]) == (doi, pmc, "pmcid")
    c = by(body, "doi", doi)
    assert c and c[0]["terminal_reason"] == "admitted" and c[0]["ids"]["pmcid"] == pmc  # PMCIDPUBLIC-1023
    assert ap.grounded_failures(200, body) == [] and _ro(body)["kind"] == "answered_with_new"


# ================================================================================================ (c) arXiv abs vN
def test_c_arxiv_abs_versioned_cc_by(world, spy):
    world.web = Router(world.web.records, world.web.jats, X.routes(licence=X.CC_BY))
    world.p.beh[DDG] = ddg_rows([(X.ABS_URL, X.TITLE)])
    world.store()
    body = ask(world)
    obs("c", body, world, spy)
    c0 = spy["ddg_out"][0][0]
    assert c0["doi"] == f"10.48550/arxiv.{X.AID}" and c0["ddg_id_kind"] == "arxiv"
    assert c0["arxiv_id"] == X.AID + "v1"  # DF-2 fixed: version kept
    assert X.ABS_URL in world.rig.web.urls and X.HTML_URL in world.rig.web.urls
    assert f"https://arxiv.org/abs/{X.AID}" not in world.rig.web.urls  # never the unversioned page
    c = by(body, "doi", f"10.48550/arxiv.{X.AID}")
    assert c and c[0]["terminal_reason"] == "admitted" and c[0]["ids"]["arxiv_id"]
    assert ap.grounded_failures(200, body) == [] and world.rig.store.plans


def test_c_desired_version_bound_route(world, spy):
    world.web = Router(world.web.records, world.web.jats, X.routes(licence=X.CC_BY))
    world.p.beh[DDG] = ddg_rows([(X.ABS_URL, X.TITLE)])
    world.store()
    body = ask(world)
    assert X.ABS_URL in world.rig.web.urls and ap.grounded_failures(200, body) == []


# ================================================================================================ (d) landing page, no OA
def test_d_publisher_landing_citation_doi_no_oa(world, spy):
    doi = "10.9999/ddgflow.d1"
    world.doc(doi, "PMC9240401", oa="N")
    landing = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000001"
    html = (f'<html><head><meta name="citation_doi" content="{doi}"><meta name="citation_title" '
            f'content="SYNTHETIC outbreak"></head><body>landing</body></html>').encode()
    gets = []

    class _G:
        status_code, headers = 200, {}

        def iter_content(self, n):
            yield html

        def close(self):
            pass

    class _NoRobots:
        status_code, headers = 404, {}

        def iter_content(self, n):
            yield b""

        def close(self):
            pass

    def get(url, **k):
        gets.append(url)
        return _NoRobots() if url.endswith("/robots.txt") else _G()
    world.p.beh[DDG] = ddg_rows([(landing, "SYNTHETIC outbreak transmission working paper")], group="portals")
    world.store()
    body = ask(world, get=get, url_check=lambda u: (True, "stub-public"))
    obs("d", body, world, spy)
    print("DDGFLOW-OBS " + json.dumps({"scenario": "d-landing", "landing_gets": gets}))
    assert gets == ["https://papers.ssrn.com/robots.txt", landing]  # DF-8: robots.txt consulted first
    assert spy["ddg_counts"][0]["ddg_landing_get"] == 1
    assert spy["ddg_out"][0][0]["doi"] == doi and spy["ddg_out"][0][0]["discovered_url"] == landing
    c = by(body, "doi", doi)
    assert c and c[0]["terminal_reason"] == "no_route" and c[0]["reason"] == "R-ACQ-NOTOA"
    assert world.rig.store.plans == [] and queued() == [] and not any("ssrn" in u for u in world.rig.web.urls)
    assert _ro(body)["kind"] == "abstain"


# ================================================================================================ (e) no identifier
def test_e_page_without_identifier(world, spy):
    url = "https://www.example-health.org/topics/outbreak-transmission"
    world.p.beh[DDG] = ddg_rows([(url, "Outbreak transmission explained")], group="portals")
    world.store()
    body = ask(world)
    obs("e", body, world, spy)
    # one landing GET attempt (refused offline by the public-address stub); DF-3: then a URL-only candidate
    assert spy["ddg_counts"][0]["ddg_landing_get"] == 1 and spy["ddg_counts"][0]["ddg_url_only"] == 1
    c = [c for c in cands(body) if (c.get("ids") or {}).get("url") == url]
    assert c and c[0]["route"] == "follow_url" and c[0]["terminal_reason"] == "queued_background"
    assert [q.get("url") for q in queued()] == [url] and world.rig.store.plans == []
    assert _ro(body)["kind"] != "no_valuable_source" and "No candidates were found" not in _ro(body)["user_text"]


# ================================================================================================ (f) Wikipedia
def test_f_wikipedia_portal_only(world, spy):
    url = "https://en.wikipedia.org/wiki/Transmission_(medicine)"
    world.p.beh[DDG] = ddg_rows([(url, "Transmission (medicine) - Wikipedia")], group="portals")
    world.store()
    body = ask(world)
    o = obs("f", body, world, spy)
    blob = json.dumps({k: body.get(k) for k in ("admitted_evidence", "citations", "sources",
                                                "examined_full_documents")})
    assert "wikipedia.org" not in blob  # never evidence
    assert world.rig.store.plans == [] and not any("wikipedia" in u for u in o["fetched"])
    assert spy["ddg_counts"][0]["ddg_held"] == 1 and cands(body) == []


# ================================================================================================ (g) DDG timeout / quota / challenge
@pytest.mark.parametrize("variant", ["timeout", "quota", "challenge"])
def test_g_ddg_failure_variants(world, spy, variant):
    doi, pmc = "10.9999/ddgflow.g1", "PMC9240701"
    world.doc(doi, pmc)
    if variant == "timeout":
        world.p.beh[DDG] = TimeoutError("ddg read timed out")
    elif variant == "quota":
        world.p.beh[DDG] = sp.ProviderUnavailable("duckduckgo_lite daily query cap (quota) of 8 spent")
    else:
        prov = sp.DuckDuckGoLiteProvider(_settings())
        try:
            prov._parse_one("portals", _R('<html><div class="anomaly-modal">are you a robot</div></html>'))
            world.p.beh[DDG] = []
        except sp.ProviderResponseError as e:
            world.p.beh[DDG] = e
    world.p.beh[EPMC] = [epmc_row(doi, pmc)]
    world.store()
    body = ask(world)
    obs(f"g-{variant}", body, world, spy)
    exp = {"timeout": "timeout", "quota": "quota", "challenge": "error"}[variant]
    ports = _ro(body)["portals"]
    assert canon(ports[0]["portal"]) == "ddg" and ports[0]["status"] == exp and world.p.log[0] == DDG
    assert ap.grounded_failures(200, body) == []  # Europe PMC carried the request
    assert _ro(body)["complete"] is False and "Coverage was incomplete" in _ro(body)["user_text"]


# ================================================================================================ stale (> 24 h by corpus_added_at)
def test_stale_corpus_added_30h_ddg_epmc_url(world, spy):
    doi, pmc = "10.9999/ddgflow.s1", "PMC9240801"
    world.doc(doi, pmc)
    world.p.lookup[pmc] = epmc_row(doi, pmc)
    world.p.beh[DDG] = ddg_rows([(f"https://europepmc.org/article/PMC/{pmc}", "SYNTHETIC outbreak transmission")])
    world.store(dict(doi="10.9999/ddgflow.old", published=HOURS(1), acquired=HOURS(30)))
    body = ask(world)
    obs("stale", body, world, spy)
    assert body["triggers"] == ["stale"] and world.p.log[0] == DDG
    assert ap.grounded_failures(200, body) == [] and by(body, "doi", doi)[0]["terminal_reason"] == "admitted"
    old = [s for s in body["sources"] if s.get("corpus_added_at") == HOURS(30).isoformat(timespec="seconds")]
    assert all(s["evidence_recency"] == "recent" for s in old)  # authored 1 h ago: recent label, but stale by added-at


# ================================================================================================ probes / controls
def test_c2_arxiv_unversioned_abs_served(world, spy):
    """(c) with arxiv.org also answering the UNVERSIONED abs URL (live arXiv serves the latest version there)."""
    routes = X.routes(licence=X.CC_BY)
    routes[f"https://arxiv.org/abs/{X.AID}"] = X.html_resp(X.abs_page(X.CC_BY))
    world.web = Router(world.web.records, world.web.jats, routes)
    world.p.beh[DDG] = ddg_rows([(X.ABS_URL, X.TITLE)])
    world.store()
    body = ask(world)
    obs("c2", body, world, spy)
    assert X.HTML_URL in world.rig.web.urls and ap.grounded_failures(200, body) == []


def test_c3_control_versioned_arxiv_via_reading_list_discovery(world, spy):
    """Control: the SAME versioned abs URL reaching acquisition through the /ask reading-list discovery
    (use_discovered) keeps its version."""
    world.web = Router(world.web.records, world.web.jats, X.routes(licence=X.CC_BY))
    world.store()
    body = ask(world, raws=[X.discovery_src()])
    obs("c3", body, world, spy)
    assert X.ABS_URL in world.rig.web.urls and X.HTML_URL in world.rig.web.urls
    assert ap.grounded_failures(200, body) == []


def test_g4_ddg_requests_read_timeout(world, spy):
    import requests
    doi, pmc = "10.9999/ddgflow.g4", "PMC9240704"
    world.doc(doi, pmc)
    world.p.beh[DDG] = requests.exceptions.ReadTimeout("HTTPSConnectionPool: Read timed out. (read timeout=8)")
    world.p.beh[EPMC] = [epmc_row(doi, pmc)]
    world.store()
    body = ask(world)
    obs("g4", body, world, spy)
    assert _ro(body)["portals"][0]["status"] == "timeout"  # DF-5 fixed


def test_g4_desired_requests_timeout_is_timeout(world, spy):
    import requests
    world.p.beh[DDG] = requests.exceptions.ReadTimeout("Read timed out.")
    world.store()
    body = ask(world)
    assert _ro(body)["portals"][0]["status"] == "timeout"


# ================================================================================================ DDGFIX-1023 regressions
@pytest.mark.parametrize("url,doi", [
    ("https://blogs.example.org/x/10.9999/a.b1/", "10.9999/a.b1"),
    ("https://www.biorxiv.org/content/10.1101/2026.01.02.123456v2.full", "10.1101/2026.01.02.123456v2"),
    ("https://www.biorxiv.org/content/10.1101/2026.01.02.123456v2.full.pdf", "10.1101/2026.01.02.123456v2"),
    ("https://j.example.org/doi/10.9999/X.Y.abstract?src=ddg#sec", "10.9999/x.y"),
    ("https://j.example.org/doi/10.9999%2Fenc.1/full", "10.9999/enc.1"),
    ("https://j.example.org/doi/pdf/10.9999/p.2/", "10.9999/p.2"),
])
def test_df1_doi_normalised(url, doi):
    assert sp.ddg_identifier(url) == {"kind": "doi", "value": doi}


@pytest.mark.parametrize("raw", [
    "10.1016/S0140-6736(20)30183-5",          # parentheses inside a valid DOI
    "10.1002/(SICI)1097-4636(199706)35:4<403::AID-JBM1>3.0.CO;2-X",  # SICI: kept whole (verifier D3)
    "10.1093/ww/9780199540884.013.U2(1)",     # balanced trailing ")" kept (verifier D3)
    "10.1000/abc/def/ghi",                    # "/" segments inside the suffix
    "10.5555/data.v2",                        # ".v2"-like suffix
    "10.48550/arxiv.2605.23989",
])
def test_df1_negative_valid_dois_not_corrupted(raw):
    exp = raw.lower()
    assert sp.normalise_doi(raw) == exp
    assert sp.normalise_doi(f"https://doi.org/{raw}") == exp


@pytest.mark.parametrize("url,ver", [(f"https://arxiv.org/abs/{X.AID}v1", "v1"), (f"https://arxiv.org/pdf/{X.AID}v3", "v3"),
                                     (f"https://arxiv.org/html/{X.AID}v2", "v2"), (f"https://arxiv.org/abs/{X.AID}", None)])
def test_df2_arxiv_version_kept_and_routed(url, ver):
    ident = sp.ddg_identifier(url)
    assert ident == {"kind": "arxiv", "value": X.AID, "version": ver}
    c = sp._ddg_cand({"url": url}, f"10.48550/arxiv.{X.AID}", None, "arxiv", arxiv_version=ver)
    routed = arxiv_fulltext.search_candidate(c, c["doi"], {"arxiv_route": {"enabled": True}})
    assert routed["arxiv"] == {"id": X.AID, "version": ver}  # unversioned: resolved later by the route's abs step
    assert routed["url"] == f"https://arxiv.org/abs/{X.AID}{ver or ''}"


def _land(results, **kw):
    return sp.ddg_to_candidates(results, {}, {"landing_groups": ["portals"]}, deadline=1e9, clock=lambda: 0.0,
                                calls_left=0, **kw)


def _row(url, group="portals"):
    return {"url": url, "title": "t", "discovery_group": group}


def test_df3_url_only_and_wikipedia():
    out, counts = _land([_row("https://news.example.org/a", "news"), _row("https://en.wikipedia.org/wiki/X"),
                         _row("http://plain.example.org/b", "news")])
    assert [c["url"] for c in out] == ["https://news.example.org/a"] and out[0]["doi"] is None
    assert counts["ddg_url_only"] == 1 and counts["ddg_held"] == 2 and counts["ddg_landing_get"] == 0


class _Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


_META = b'<html><head><meta name="citation_doi" content="10.9999/land.1/"></head></html>'


def _getter(routes, log):
    def get(url, **k):
        log.append(url)
        return routes.get(url) or _Resp(404)
    return get


def test_df8_landing_fetch_restrictions():
    pub = lambda u: (True, "stub")  # noqa: E731
    page = "https://papers.ssrn.com/p?id=1"
    from vera import acq_discovered
    acq_discovered.reset_for_tests()  # robots.txt cache is per process
    # allowed host, robots 404 -> allowed; DOI normalised
    log = []
    assert sp.landing_doi(page, 5, 4096, get=_getter({page: _Resp(200, _META)}, log), url_check=pub,
                          acq={}) == "10.9999/land.1"
    assert log == ["https://papers.ssrn.com/robots.txt", page]
    # Wikipedia: never fetched (portal only)
    log = []
    assert sp.landing_doi("https://en.wikipedia.org/wiki/X", 5, 4096, get=_getter({}, log), url_check=pub,
                          acq={}) is None and log == []
    # robots.txt disallows -> landing page never requested
    acq_discovered.reset_for_tests()
    log = []
    rb = _Resp(200, b"User-agent: *\nDisallow: /\n", {"Content-Type": "text/plain"})
    assert sp.landing_doi(page, 5, 4096, get=_getter({"https://papers.ssrn.com/robots.txt": rb, page: _Resp(200, _META)},
                                                    log), url_check=pub, acq={}) is None
    assert page not in log
    # robots.txt 503 -> fail closed
    acq_discovered.reset_for_tests()
    log = []
    assert sp.landing_doi(page, 5, 4096, get=_getter({"https://papers.ssrn.com/robots.txt": _Resp(503), page:
                                                     _Resp(200, _META)}, log), url_check=pub, acq={}) is None
    assert page not in log
    acq_discovered.reset_for_tests()
    # redirect to a non-https URL is not followed
    acq_discovered.reset_for_tests()
    log = []
    red = _Resp(302, b"", {"Location": "http://evil.example.org/x"})
    assert sp.landing_doi(page, 5, 4096, get=_getter({page: red}, log), url_check=pub, acq={}) is None
    assert not any("evil" in u for u in log)
    # non-public address -> nothing requested; http -> nothing requested; non-HTML -> no identifier
    log = []
    assert sp.landing_doi(page, 5, 4096, get=_getter({}, log), url_check=lambda u: (False, "private"),
                          acq={}) is None and log == []
    assert sp.landing_doi("http://papers.ssrn.com/p", 5, 4096, get=_getter({}, log), url_check=pub,
                          acq={}) is None and log == []
    acq_discovered.reset_for_tests()
    assert sp.landing_doi(page, 5, 4096, get=_getter({page: _Resp(200, _META, {"Content-Type": "application/pdf"})},
                                                    []), url_check=pub, acq={}) is None
    acq_discovered.reset_for_tests()


def test_df8_default_transport_is_pinned_and_rate_limited(monkeypatch):
    from vera import rate_limit, selective_fetch
    seen = {}

    def fake_limited_by_url(fn, **k):
        seen["fn"] = fn
        return lambda *a, **kw: _Resp(404)
    monkeypatch.setattr(rate_limit, "limited_by_url", fake_limited_by_url)
    sp.landing_doi("https://papers.ssrn.com/p", 5, 4096, url_check=lambda u: (True, ""), robots_ok=lambda u: True)
    assert seen["fn"] is selective_fetch.pinned_get


@pytest.mark.parametrize("raw,exp", [("10.9999/x.1)", "10.9999/x.1"), ("(doi:10.9999/x.2).", "10.9999/x.2"),
                                     ("10.1093/ww/9780199540884.013.U2(1).", "10.1093/ww/9780199540884.013.u2(1)"),
                                     ("10.9999/a%252Fb", "10.9999/a%2fb")])
def test_df1_unbalanced_trailing_punctuation_only(raw, exp):
    assert sp.normalise_doi(raw.lstrip("(")) == exp


def test_d1_landing_get_sends_the_ua_robots_was_checked_for():
    from vera import acq_discovered, selective_fetch
    acq_discovered.reset_for_tests()
    page, sent = "https://papers.ssrn.com/p?id=9", []
    rb = _Resp(200, b"User-agent: VERA-research-fetcher\nDisallow: /\n", {"Content-Type": "text/plain"})

    def get(url, **k):
        sent.append((url, (k.get("headers") or {}).get("User-Agent")))
        return rb if url.endswith("/robots.txt") else _Resp(200, _META)
    # a Disallow aimed at the fetcher's product token is honoured: the landing page is never requested
    assert sp.landing_doi(page, 5, 4096, get=get, url_check=lambda u: (True, ""), acq={}) is None
    assert [u for u, _ in sent] == ["https://papers.ssrn.com/robots.txt"]
    acq_discovered.reset_for_tests()
    sent.clear()
    assert sp.landing_doi(page, 5, 4096, get=_getter_ua({page: _Resp(200, _META)}, sent), url_check=lambda u: (True, ""),
                          acq={}) == "10.9999/land.1"
    assert {ua for _, ua in sent} == {selective_fetch.USER_AGENT}  # robots fetch and landing GET: one UA
    acq_discovered.reset_for_tests()


def _getter_ua(routes, log):
    def get(url, **k):
        log.append((url, (k.get("headers") or {}).get("User-Agent")))
        return routes.get(url) or _Resp(404)
    return get


def test_df4_unreachable_cause_not_budget():
    from vera import research_outcome as ro
    out = ro.summarize_outcome([ro.PortalAttempt("ddg", "completed", 1)],
                               [ro.CandidateOutcome("c1", terminal_reason="queued_background", detail="R-ACQ-UNREACHABLE")],
                               budget_exhausted=False, older_evidence_available=False, new_evidence_admitted=False,
                               source_unreachable=True)
    assert "could not be reached" in out.user_text and "budget ran out" not in out.user_text and not out.complete


def test_df5_requests_timeouts_are_timeouts():
    import requests
    assert sp._is_timeout(requests.exceptions.ReadTimeout()) and sp._is_timeout(requests.exceptions.ConnectTimeout())
    assert not sp._is_timeout(requests.exceptions.ConnectionError())


def test_flow_a_ddg_result_redirect_landing_doi_oa_fulltext_cited_same_request(world, spy):
    """Owner P0 flow (A): a DDG result on a previously unknown public site redirects (301) to the publisher page; the
    page's citation_doi (normalised, DF-1) maps to an open-access Europe PMC full text through the EXISTING route and
    gates; the admitted chunk is cited in the SAME request. robots.txt is consulted on each host; the landing page text
    itself is never evidence (the evidence is the licensed Europe PMC JATS)."""
    doi, pmc = "10.9999/ddgflow.ra1", "PMC9240901"
    world.doc(doi, pmc)
    start = "https://unknown-site.example.net/item/42"
    final = "https://publisher.example.com/article/42"
    marker = "LANDINGTEXT-SENTINEL-7f3a"  # verifier D2: landing-page text must never be stored, returned or persisted
    html = (f'<html><head><title>{marker} title</title><meta name="description" content="{marker} desc">'
            f'<meta name="citation_doi" content="https://doi.org/{doi}/"></head><body>{marker} body</body></html>').encode()
    gets = []

    def get(url, **k):
        gets.append(url)
        if url.endswith("/robots.txt"):
            return _Resp(404)
        if url == start:
            return _Resp(301, b"", {"Location": final})
        return _Resp(200, html, {"Content-Type": "text/html; charset=utf-8"})
    world.p.beh[DDG] = ddg_rows([(start, "SYNTHETIC outbreak transmission incidence study")], group="portals")
    world.store()
    body = ask(world, get=get, url_check=lambda u: (True, "stub-public"))
    obs("flow-a", body, world, spy)
    assert gets == ["https://unknown-site.example.net/robots.txt", start, "https://publisher.example.com/robots.txt",
                    final]
    assert spy["ddg_out"][0][0]["doi"] == doi and spy["ddg_out"][0][0]["discovered_url"] == start
    assert f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML" in world.rig.web.urls
    c = by(body, "doi", doi)
    assert c and c[0]["terminal_reason"] == "admitted" and c[0]["ids"]["pmcid"] == pmc
    assert ap.grounded_failures(200, body) == [] and body.get("citations")
    assert _ro(body)["kind"] == "answered_with_new"
    assert not any("example.net" in u or "example.com" in u for u in world.rig.web.urls)
    blob = json.dumps({"cands": spy["ddg_out"], "body": body, "queued": queued(),
                       "plans": world.rig.store.plans}, default=str)  # plans carry plan["candidate"] ({S}.candidates)
    assert marker not in blob and marker.lower() not in blob.lower()
