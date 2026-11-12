"""ROBOTS-LABEL-1111 (WP-L) tests 28-33: R2 (OD-1111-ACCESS-DENIED-NO-BYPASS, owner 2026-11-11). A robots.txt 403 is a
robots failure; a 401/403 on the actual article or download endpoint is an ACCESS DENIAL: its own code, never queued for
retry, never bypassed for the same document (DOI, PMCID or URL) by another candidate, lane, redirect, landing page or
later request. Fake HTTP only; no network, no DB."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from tests.test_public_fetch_1103 import CFG, Net, Resp, html, robots
from vera import acq_discovered, auto_acquire, discovery_panel, inline_acquire, research_outcome, research_record
from vera import public_fetch as pf
from vera import selective_fetch as sf

DOI = "10.9999/acq.epi4"
from vera.acq_config import ingest_cfg as acq_config_ingest  # noqa: E402

PMC = "PMC1234567"
JATS = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMC}/fullTextXML"


def _web(over=None, doi=DOI):
    web, _raw = A.epi_setup(doi=doi, pmcid=PMC, over=over)
    return web


def _eval(web, doi=DOI, cand=None, cfg=None):
    return auto_acquire.evaluate_candidate(A.job(), 1.0, doi, cand or A.cand(doi), None, cfg or A.cfg(),
                                           A.deps([], web), {}, {}, clock=A.clock_at())


def _code(web, **kw):
    with pytest.raises(auto_acquire._Stop) as e:
        _eval(web, **kw)
    return e.value.code


def _n(web, frag):
    return sum(1 for u in web.urls if frag in u)


def test_t28_robots_403_is_a_robots_code_and_a_document_403_is_an_access_denial():
    web = _web({"/robots.txt": A.Resp(status=403, body=b"", ctype="text/plain")})
    c = _code(web)
    assert c.startswith("R-ACQ-NOROUTE_UNCHECKED-STATUS_403") and "ACCESS_DENIED" not in c
    acq_discovered.reset_for_tests()
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    c = _code(web)
    assert c == "R-ACQ-ACCESS_DENIED-ROUTE" and "ROBOTS" not in c  # a document 403 never yields a robots code
    line = discovery_panel.reason_text(c)
    assert "access to this document was refused" in line and "full-text service" in line
    assert "source's own site was not contacted" in line
    assert research_outcome.reason_from_code(c) == "no_route"
    assert not research_record._fetch_made({"status": "selected_not_fetched", "reason": c})  # went to the service


def test_t28b_search_endpoint_403_is_route_denied_and_not_remembered():
    web = _web({"/rest/search": A.Resp(status=401, body=b"", ctype="text/html")})
    c = _code(web)
    assert c == "R-ACQ-ROUTE_DENIED" and "refused the lookup" in discovery_panel.reason_text(c)
    assert acq_discovered._DENIED == {}  # A-1: the search endpoint is not remembered


def test_t28c_landing_page_403_is_access_denied_source_hop():
    acq_discovered.reset_for_tests()
    A.seed_access("doi.org")
    web = A.Web({}, {}, {"doi.org/10.9999/x": A.Resp(status=403, body=b"", ctype="text/html")})
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get("https://doi.org/10.9999/x", {"doi.org"}, {"text/html"}, 1000, 2, A.deps([], web),
                              A.cfg(), A.clock_at(), None, [])
    assert e.value.code == "R-ACQ-ACCESS_DENIED"
    assert "access to this document was refused" in discovery_panel.reason_text(e.value.code)
    assert research_record._fetch_made({"status": "fetched_rejected", "reason": e.value.code})  # a request was made
    assert acq_discovered.is_denied(urls=("https://doi.org/10.9999/x",))


def test_t29_denials_are_not_transient_and_not_queued():
    cfg = A.cfg()
    for k in ("access_denied", "route_denied"):
        code = cfg["reason_codes"][k]
        assert inline_acquire._transient(cfg, code) is False
        assert research_outcome.skip_class(code) is None
    assert inline_acquire._transient(cfg, "R-ACQ-ACCESS_DENIED-ROUTE") is False
    acq_discovered.deny(acq_discovered.denial_ids(doi=DOI))
    r = auto_acquire.enqueue_candidates("How was the transmission related to incidence during an outbreak?",
                                        [A.cand(DOI), A.cand("10.9999/other")], cfg=A.cfg(), env=A.ENV_ON,
                                        start_worker=False)
    assert 0 in r["already"] and 0 not in r["queued"] and 1 in r["queued"]  # the denied identity is dropped


def test_t30_same_doi_other_candidate_and_portal_via_make_no_request():
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    assert _code(web) == "R-ACQ-ACCESS_DENIED-ROUTE"
    before = list(web.urls)
    assert _code(web, cand=A.cand(DOI, url="https://publisher.example/other")) == "R-ACQ-ACCESS_DENIED"
    arxiv = A.cand(DOI, portal_via="arXiv 2401.00001")
    assert _code(web, cand=arxiv) == "R-ACQ-ACCESS_DENIED"
    assert web.urls == before  # zero requests to anything


def test_t30b_different_doi_same_pmcid_searches_but_never_re_requests_the_xml():
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    assert _code(web) == "R-ACQ-ACCESS_DENIED-ROUTE"
    other = "10.9999/acq.vers9"  # a version DOI whose resolve_oa result carries the same (denied) PMCID
    web.records[other] = dict(web.records[DOI], doi=other)
    n_xml = _n(web, "fullTextXML")
    assert _code(web, doi=other, cand=A.cand(other)) == "R-ACQ-ACCESS_DENIED"
    assert _n(web, "fullTextXML") == n_xml and _n(web, "/rest/search") == 2  # the search was sent, the XML was not


def test_t30c_pmcid_only_candidate_promoted_to_a_denied_doi():
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    assert _code(web) == "R-ACQ-ACCESS_DENIED-ROUTE"
    n_xml = _n(web, "fullTextXML")
    cand = {"pmcid": PMC, "title": "t", "url": "https://europepmc.org/article/PMC/1234567"}
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.evaluate_candidate(A.job(), 1.0, "", cand, None, A.cfg(), A.deps([], web), {}, {}, clock=A.clock_at())
    assert e.value.code.startswith("R-ACQ-ACCESS_DENIED") and _n(web, "fullTextXML") == n_xml


def test_t30d_redirect_to_a_denied_url_is_refused_in_b_get_and_guarded_get():
    acq_discovered.deny(acq_discovered.denial_ids(urls=("https://doi.org/10.9999/z", "https://pub.example/a/z")))
    A.seed_access("doi.org")
    A.seed_access("pub.example")
    web = A.Web({}, {}, {"doi.org/10.9999/y": A.Resp(status=302, body=b"", ctype="text/html",
                                                     headers={"Location": "https://pub.example/a/z"})})
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get("https://doi.org/10.9999/y", {"doi.org", "pub.example"}, {"text/html"}, 1000, 2,
                              A.deps([], web), A.cfg(), A.clock_at(), None, [])
    assert e.value.code == "R-ACQ-ACCESS_DENIED" and not any("pub.example/a/z" in u for u in web.urls)
    web2 = _web()
    acq_discovered.deny(acq_discovered.denial_ids(urls=(JATS,)))
    with pytest.raises(auto_acquire._Stop) as e2:
        auto_acquire._guarded_get(JATS, A.cfg(), A.deps([], web2), 1000, 5.0, False, clock=A.clock_at())
    assert e2.value.code == "R-ACQ-ACCESS_DENIED" and web2.urls == []


def test_t30e_ddg_landing_get_skips_a_denied_url_and_remembers_a_403():
    from vera import search_providers
    url = "https://journal.example/article/1"
    calls = []

    def get(u, **kw):
        calls.append(u)
        return Resp(403, {"Content-Type": "text/html"}, b"")
    got = search_providers.landing_doi(url, 5.0, 10000, get=get, url_check=lambda u: (True, ""),
                                       robots_ok=lambda u: True)
    assert got is None and len(calls) == 1
    assert search_providers.landing_doi(url, 5.0, 10000, get=get, url_check=lambda u: (True, ""),
                                        robots_ok=lambda u: True) is None
    assert len(calls) == 1  # the denied landing page was not requested again


def test_t31_landing_is_never_reached_after_a_fulltext_denial():
    cfg = A.cfg()
    cfg["inline"]["use_discovered"] = True
    cfg["inline"]["landing_route"] = True
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    cand = A.cand(DOI, discovery={"service": "ddg"})
    c = _code(web, cand=cand, cfg=cfg)
    assert c == "R-ACQ-ACCESS_DENIED-ROUTE" and not any("doi.org" in u for u in web.urls)


def test_t32_later_request_is_refused_and_user_agent_is_unchanged():
    web = _web({"fullTextXML": A.Resp(status=403, body=b"", ctype="text/html")})
    assert _code(web) == "R-ACQ-ACCESS_DENIED-ROUTE"
    n = len(web.urls)
    assert _code(web) == "R-ACQ-ACCESS_DENIED" and len(web.urls) == n
    assert sf.USER_AGENT == sf.USER_AGENT.strip() and "bot" not in sf.USER_AGENT.lower().replace("bots", "")  # constant
    seen = []

    def spy(url, headers=None, **kw):
        seen.append((headers or {}).get("User-Agent"))
        return _web()(url, headers=headers, **kw)
    acq_discovered.reset_for_tests()
    try:
        _eval(spy)
    except auto_acquire._Stop:
        pass
    except Exception:  # noqa: BLE001  (a later stage may stop; only the request headers matter here)
        pass
    assert seen and set(seen) == {sf.USER_AGENT}


def test_t33_web_route_denial_official_lane_and_positive_control():
    pages = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): Resp(403, {"Content-Type": "text/html"}, b""),
             ("b.example", "/robots.txt"): robots(), ("b.example", "/q"): html()}
    dns = {h: ["93.184.216.34"] for h in ("a.example", "b.example")}
    net = Net(dns, pages)
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=net.resolve, transport=net.transport)
    assert e.value.reason == "access_denied"
    n = len(net.calls)
    for _ in range(2):  # official_lane / lane_evidence / a later request for the same URL: refused, no request
        with pytest.raises(pf.FetchRefused) as e2:
            pf.fetch("https://a.example/p", CFG, resolve=net.resolve, transport=net.transport)
        assert e2.value.reason == "access_denied"
    assert len(net.calls) == n
    # positive control: an independent source in the same process is still fetched
    assert pf.fetch("https://b.example/q", CFG, resolve=net.resolve, transport=net.transport)["text"] == "hello"


def test_t33b_web_route_phrase_for_a_denied_page():
    from vera import web_route as wr
    pages = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): Resp(401, {"Content-Type": "text/html"}, b"")}
    net = Net({"a.example": ["93.184.216.34"]}, pages)

    def f(url, **kw):
        return pf.fetch(url, CFG, resolve=net.resolve, transport=net.transport, **kw)
    r = wr.read_web("q", [{"url": "https://a.example/p", "title": "T", "snippet": "s", "acq_source": "ddg"}], fetch=f,
                    registry=[], deny_list=[])
    assert r.pages == [] and r.dev[0]["gate"]["read"] == "read_access_denied"
    assert r.listed[0]["phrase"] == "not read: the site denied access to this page"


def test_t30f_pmcid_only_candidate_whose_jats_doi_is_denied_records_access_denied():
    # D-1 (verifier): the DOI found only in the JATS article-id is re-checked after the `known` check (RC-2)
    web = _web()
    web.records[PMC] = A.epmc_record("", pmcid=PMC)  # the search by PMCID returns a record that carries no DOI
    acq_discovered.deny(acq_discovered.denial_ids(doi=DOI))
    cand = {"pmcid": PMC, "title": "t", "url": "https://europepmc.org/article/PMC/1234567"}
    from vera import fulltext_loader
    cfg = A.cfg()
    icfg = acq_config_ingest(fulltext_loader.load_config(), cfg)
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.evaluate_candidate(A.job(), 1.0, "", cand, None, cfg, A.deps([], web), icfg,
                                        auto_acquire._fd_cfg(), clock=A.clock_at())
    assert e.value.code == "R-ACQ-ACCESS_DENIED"
    assert _n(web, "fullTextXML") <= 1  # the identity was only learned from the JATS itself; nothing further is fetched


def test_t30g_resolved_doi_that_differs_from_the_candidates_is_checked_too(monkeypatch):
    # D-2 (verifier): both DOIs are checked, not only when the candidate has none
    web = _web()
    real = auto_acquire.resolve_oa

    def other_doi(*a, **k):
        return {**real(*a, **k), "doi": "10.9999/acq.resolved"}
    monkeypatch.setattr(auto_acquire, "resolve_oa", other_doi)
    acq_discovered.deny(acq_discovered.denial_ids(doi="10.9999/acq.resolved"))
    assert _code(web) == "R-ACQ-ACCESS_DENIED" and _n(web, "fullTextXML") == 0
