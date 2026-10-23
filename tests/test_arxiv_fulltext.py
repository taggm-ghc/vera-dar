"""ARX-A2: arXiv 2605.23989v1 full-text representation route (vera/arxiv_fulltext.py). OFFLINE: synthetic abs/LaTeXML
fixtures (tests/arxiv_synth.py), fake transports, stubbed answer model and claim checker; no network, DB or LLM.

Under the CURRENT licence_gate.py (arXiv e-print/PDF/HTML never stored) a CC-licensed arXiv HTML is fetched,
extracted and then refused AT the licence gate with R-ACQ-ARXIV_HTML_POLICY. The admission + same-request answer
tests SIMULATE an owner-approved rule (test-only monkeypatch of licence_gate.licence_decision: arXiv /html/ treated
like its /abs/ page when the paper's own licence is CC); they prove the downstream path, not the policy."""
import copy

import pytest
from fastapi.testclient import TestClient

from tests import accept_predicate as ap
from tests import acq_support as A
from tests import arxiv_synth as X
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, redirect, run  # noqa: F401  (_direct: autouse)
from tests.test_acq_serve import _decide, h  # noqa: F401  (h is a fixture)
from tests.test_dar_inline_acquire import OFF_Q, rig  # noqa: F401  (rig is a fixture)
from vera import acq_config, arxiv_fulltext, fulltext_loader, licence_gate

_REAL_DECISION = licence_gate.licence_decision
ROBOTS = "https://arxiv.org/robots.txt"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: waits.append(s))
    return waits


def _proposed_rule(c):
    """TEST-ONLY simulation of a possible owner decision; NOT shipped code."""
    u = str(c.get("url") or "")
    lic = ((c.get("licence") or {}).get("content") or {})
    if u.startswith("https://arxiv.org/html/") and "creativecommons.org" in str(lic.get("url") or ""):
        return _REAL_DECISION({**c, "url": u.replace("/html/", "/abs/", 1)})
    return _REAL_DECISION(c)


@pytest.fixture
def allow_arxiv_html(monkeypatch):
    monkeypatch.setattr(licence_gate, "licence_decision", _proposed_rule)
    monkeypatch.setattr(fulltext_loader, "licence_decision", _proposed_rule)


@pytest.fixture
def spy_prepare(monkeypatch):
    calls = []
    real = fulltext_loader.prepare_from_parts
    monkeypatch.setattr(fulltext_loader, "prepare_from_parts", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    return calls


def _web(**kw):
    return Router({}, {}, X.routes(**kw))


def _go(web, cfg=None, src=None):
    return run(cfg or cfg_on(), [src or X.discovery_src()], web)


def _attempts():
    return arxiv_fulltext.last_attempts(X.AID + X.VER)


# ------------------------------------------------------------------------------------------------ route + gate
def test_html_attempted_through_safe_route_and_refused_at_current_licence_gate():
    web = _go_web = _web()
    o, store = _go(web)
    urls = [u for u, _h, _k in web.calls]
    assert urls == [ROBOTS, X.ABS_URL, X.HTML_URL]  # robots.txt, abs (identity + licence) then the HTML representation; nothing else
    for _u, hdrs, _k in web.calls[1:]:
        assert hdrs.get("Accept-Encoding") == "identity" and "Cookie" not in hdrs and "Authorization" not in hdrs
    e = disc(o)
    assert e["outcome"] == "rejected" and e["reason"] == "R-ACQ-ARXIV_HTML_POLICY", e
    assert e["epistemic"] == "preprint: not peer reviewed"
    assert store.plans == [] and o.status != "ok"
    last = _attempts()[-1]
    assert last["status"] == "refused" and last["reason"] == "R-LIC-REJECT" and "arXiv" in last["detail"]
    assert "europepmc" not in " ".join(urls) and "ebi.ac.uk" not in " ".join(urls)


def test_arxiv_route_off_keeps_the_d1_hold():
    c = cfg_on()
    c["arxiv_route"]["enabled"] = False
    web = _web()
    o, store = _go(web, c)
    assert disc(o)["reason"] == "R-ACQ-PREPRINT_LICENCE_UNVERIFIED" and web.calls == []


def test_non_arxiv_preprint_hold_unchanged():
    src = {"provider": "openalex", "doi": "10.1101/2024.01.01.123456", "url": "https://doi.org/10.1101/2024.01.01.123456",
           "title": "SYNTHETIC biorxiv", "openalex_id": "https://openalex.org/W9"}
    web = _web()
    o, _store = _go(web, src=src)
    assert disc(o)["reason"] == "R-ACQ-PREPRINT_LICENCE_UNVERIFIED" and web.calls == []


def test_openalex_record_with_arxiv_datacite_doi_is_routed():
    src = {"provider": "openalex", "doi": X.DOI, "url": X.ABS_URL, "title": X.TITLE, "openalex_id": "https://openalex.org/W8"}
    web = _web()
    o, _store = _go(web, src=src)
    assert [u for u, _h, _k in web.calls][:2] == [ROBOTS, X.ABS_URL] and disc(o)["reason"] == "R-ACQ-ARXIV_HTML_POLICY"


# ------------------------------------------------------------------------------------------------ licence
def test_arxiv_default_licence_is_held_with_precise_reason_and_no_content_fetch():
    web = _web(licence=X.ARXIV_DEFAULT)
    o, store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-ARXIV_LICENCE_HOLD" and store.plans == []
    assert [u for u, _h, _k in web.calls] == [ROBOTS, X.ABS_URL]  # the HTML is never fetched for a held licence


def test_missing_licence_link_is_held():
    o, store = _go(_web(licence=None))
    assert disc(o)["reason"] == "R-ACQ-LICENCE_ABSENT" and store.plans == []


def test_nd_licence_is_rejected():
    web = _web(licence=X.CC_BY_ND)
    o, store = _go(web)
    assert disc(o)["reason"] == "R-LIC-REJECT" and store.plans == []
    assert [u for u, _h, _k in web.calls] == [ROBOTS, X.ABS_URL]


# ------------------------------------------------------------------------------------------------ abstract-only
def test_html_missing_means_abstract_only_never_full_text(spy_prepare):
    o, store = _go(_web(html=A.Resp(status=404, ctype="text/html")))
    assert disc(o)["reason"] == "R-ABSTRACT" and store.plans == [] and spy_prepare == []
    reps = [(a["representation"], a["status"]) for a in _attempts()]
    assert reps == [("abs", "fetched"), ("html", "refused"), ("pdf", "not_attempted")]


def test_html_redirect_to_abstract_page_is_not_full_text(spy_prepare):
    web = Router({}, {}, {**X.routes(), X.HTML_URL: redirect(X.ABS_URL)})
    o, store = _go(web)
    assert disc(o)["reason"] == "R-ABSTRACT" and store.plans == [] and spy_prepare == []


def test_abstract_page_served_at_html_url_is_not_full_text(spy_prepare):
    o, store = _go(_web(html=X.html_resp(X.abs_page())))
    assert disc(o)["reason"] == "R-ABSTRACT" and store.plans == [] and spy_prepare == []


# ------------------------------------------------------------------------------------------------ fetch safety
def test_redirect_to_non_arxiv_host_is_refused():
    web = Router({}, {}, {**X.routes(), X.HTML_URL: redirect("https://evil.example.net/html/2605.23989v1")})
    o, store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-HOST" and store.plans == []
    assert "https://evil.example.net/html/2605.23989v1" not in [u for u, _h, _k in web.calls]


def test_http_redirect_is_refused():
    web = Router({}, {}, {**X.routes(), X.HTML_URL: redirect("http://arxiv.org/html/2605.23989v1")})
    o, _store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-HOST"


def test_wrong_content_type_is_refused():
    o, store = _go(_web(html=X.html_resp(X.latexml_html(), ctype="application/pdf")))
    assert disc(o)["reason"] == "R-ACQ-CONVERT" and store.plans == []


def test_other_version_in_final_url_is_refused():
    v2 = "https://arxiv.org/html/2605.23989v2"
    web = Router({}, {}, {**X.routes(), X.HTML_URL: redirect(v2), v2: X.html_resp(X.latexml_html())})
    o, _store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-IDENTITY_MISMATCH"


def test_abs_page_for_another_paper_is_refused():
    o, _store = _go(_web(abs_body=X.abs_page(arxiv_id="2605.00001")))
    assert disc(o)["reason"] == "R-ACQ-IDENTITY_MISMATCH"


def test_public_address_check_runs_on_every_hop():
    seen = []
    web = _web()
    store = A.Store()
    d = A.deps([], web, store=store)
    d.url_check = lambda u: seen.append(u) or (True, "")
    from vera import inline_acquire
    inline_acquire.acquire(A.Q_EPI, cfg_on(), env=A.ENV_ON, deps_factory=lambda a, e: d,
                           discovered=[X.discovery_src()])
    assert seen == [ROBOTS, X.ABS_URL, X.HTML_URL]


def test_production_transport_is_the_pinned_public_only_getter(monkeypatch):
    """default_deps wiring: arxiv.org requests go through selective_fetch.pinned_get (rate-limited arxiv_web portal)
    and the real is_public_url (resolved address must be public)."""
    import socket

    import sqlalchemy

    from tests.test_acq_wiring import _Engine
    from vera import auto_acquire, rate_limit, selective_fetch
    rc = rate_limit.load_config(rate_limit.CONFIG_PATH)
    assert "arxiv.org" in rc["portals"]["arxiv_web"]["hosts"]
    rc["portals"]["arxiv_web"]["rate_per_s"] = 1000.0  # the spacing itself is tested separately (no real waits here)
    rate_limit.set_registry_for_tests(rate_limit.Registry(rc))
    web = _web()
    pinned = []
    monkeypatch.setattr(selective_fetch, "pinned_get", lambda url, **kw: pinned.append(url) or web(url, **kw))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("151.101.3.42", 443))])
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url, **kw: _Engine(
        "wo" if "wo.invalid" in url else "ro", rows={"count(*)": [(0,)], "canonical_url = :u": [], "LIMIT 1": []}))
    d = auto_acquire.default_deps(cfg_on(), A.ENV_ON)
    d.search = lambda q, n, t: []
    from vera import inline_acquire
    o = inline_acquire.acquire(A.Q_EPI, cfg_on(), env=A.ENV_ON, deps_factory=lambda a, e: d,
                               discovered=[X.discovery_src()])
    rate_limit.set_registry_for_tests(None)
    assert pinned == [ROBOTS, X.ABS_URL, X.HTML_URL], (pinned, o.public())
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 443))])
    inline_acquire.reset_for_tests()
    rate_limit.set_registry_for_tests(rate_limit.Registry(rc))
    pinned.clear()
    o = inline_acquire.acquire(A.Q_EPI, cfg_on(), env=A.ENV_ON, deps_factory=lambda a, e: d,
                               discovered=[X.discovery_src()])
    rate_limit.set_registry_for_tests(None)
    assert ROBOTS not in pinned[1:] and X.HTML_URL not in pinned and X.ABS_URL not in pinned
    assert o.discovered[0]["reason"] in ("R-ACQ-HOST", "R-ACQ-ROBOTS")  # private address: nothing fetched


# ------------------------------------------------------------------------------------------------ admitted (simulated rule)
def test_cc_by_admitted_with_identity_attribution_and_clean_body(allow_arxiv_html):
    o, store = _go(_web())
    assert o.status == "ok" and disc(o)["outcome"] == "admitted", o.public()
    p = store.plans[0]
    prov = p["source"]["provenance"]
    a = prov["acquisition"]["arxiv"]
    assert (a["id"], a["version"], a["abs_url"], a["html_url"]) == (X.AID, "v1", X.ABS_URL, X.HTML_URL)
    assert a["licence"] == "cc by" and a["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert a["retrieved_at"] and a["published"] == "2026/05/21" and a["retrieved_at"] != a["published"]
    assert prov["title"] == X.TITLE and "Synthauthor, Alpha" in prov["authors"]
    assert "arXiv:2605.23989v1" in prov["attribution"] and prov["doi"] == X.DOI
    assert prov["url"] == X.ABS_URL and prov["fulltext_url"] == X.HTML_URL and prov["fulltext_format"] == "html"
    assert prov["acquisition"]["resolver"] == "arxiv-html" and prov["acquisition"]["url_chain"] == [X.ABS_URL, X.HTML_URL]
    body = p["source"]["content"] + " ".join(s["text"] for s in p["spans"])
    for tok in X.CHROME:
        assert tok not in body
    assert "SYNTHETIC sentence" in body and "Introduction" in body


def test_cc_by_chunks_answer_the_same_post_ask_request(rig, monkeypatch, allow_arxiv_html):
    """POST /ask (TestClient, main:app): the discovered arXiv record is acquired inline and its admitted passages are
    the evidence of the SAME response (stubbed answer model + claim checker)."""
    import main
    web = _web()
    rig.setup(web=web, cands=[])
    rig.h.answer_text = "The transmission of the pathogen was compared with incidence during the outbreak [1]."
    main_mod = rig.h.install()
    monkeypatch.setattr(main_mod, "retrieve_all", lambda q, cfg, **kw: ([X.discovery_src()], []))
    monkeypatch.setattr(main_mod, "_acquire", lambda q, t: None)
    from vera import inline_acquire
    monkeypatch.setattr(inline_acquire, "refresh_after_response", lambda: None)
    _decide(monkeypatch, False, "layer2_llm", "llm_out:x", "off_topic")
    r = TestClient(main.app).post("/ask", json={"question": OFF_Q}, headers={"X-API-Key": "k" * 20})
    body = r.json()
    assert r.status_code == 200 and body["result_kind"] == "answer", body.get("contract")
    ia = body["contract"]["inline_acquisition"]
    assert [e["outcome"] for e in ia["discovered"]] == ["admitted"]
    assert X.HTML_URL in [u for u, _h, _k in web.calls]
    assert body["admitted_evidence"] and all(a["document_id"].startswith("acq-") for a in body["admitted_evidence"])
    f = ap.grounded_failures(r.status_code, body)
    assert not [x for x in f if "build identity" not in x], f
    assert len(rig.h.answer_calls) == 1 and len(rig.h.check_calls) == 1


def test_reader_redecides_licence_from_the_stored_abs_licence(allow_arxiv_html):
    from vera import fulldoc_db
    o, store = _go(_web())
    srcs, spans = A.rows_from_plans(store.plans)
    from vera import auto_acquire
    c = fulldoc_db.load_acquired_rows([{**s, "provenance": __import__("json").loads(s["provenance"])} for s in srcs],
                                      spans, auto_acquire._fd_cfg(), cfg_on())
    assert c.status == "ok" and c.acquired
    bad = copy.deepcopy(store.plans[0])
    bad["source"]["provenance"]["meta"]["jats_licences"][0]["href"] = X.CC_BY_ND
    srcs, spans = A.rows_from_plans([bad])
    c = fulldoc_db.load_acquired_rows([{**s, "provenance": __import__("json").loads(s["provenance"])} for s in srcs],
                                      spans, auto_acquire._fd_cfg(), cfg_on())
    assert not c.acquired


def test_shipped_config_route_and_codes():
    c = acq_config.load_config()
    r = c["arxiv_route"]
    assert r["enabled"] is True and r["hosts"] == ["arxiv.org"] and r["content_types"] == ["text/html"]
    assert r["pdf_extractor"] is None
    assert c["fetch"]["allowed_hosts"] == ["www.ebi.ac.uk", "eutils.ncbi.nlm.nih.gov"]  # Europe PMC list not widened
    from vera import discovery_panel
    for k in ("arxiv_licence_hold", "arxiv_html_policy"):
        assert c["reason_codes"][k] in discovery_panel.REASON_TEXT


# ------------------------------------------------------------------------------------------------ robots + pacing
def test_requests_are_spaced_by_min_interval(_no_wait):
    _go(_web())
    assert len(_no_wait) == 2 and all(0 < w <= 3.0 for w in _no_wait)  # robots -> abs -> html, one at a time


def test_robots_disallow_refuses_before_any_content_fetch():
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(body=b"User-agent: *\nDisallow: /html/\n", ctype="text/plain")})
    o, store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-ROBOTS" and X.HTML_URL not in [u for u, _h, _k in web.calls]


def test_long_crawl_delay_is_refused_not_ignored():
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(body=b"User-agent: *\nCrawl-delay: 60\n", ctype="text/plain")})
    o, _store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-ROBOTS" and [u for u, _h, _k in web.calls] == [ROBOTS]


def test_crawl_delay_within_limit_is_honoured(_no_wait):
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(body=b"User-agent: *\nCrawl-delay: 10\n", ctype="text/plain")})
    _go(web)
    assert _no_wait and max(_no_wait) > 3.0


def test_robots_server_error_fails_closed():
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(status=503, ctype="text/plain")})
    o, _store = _go(web)
    assert disc(o)["reason"] == "R-ACQ-ROBOTS" and X.ABS_URL not in [u for u, _h, _k in web.calls]


# ------------------------------------------------------------------------------------------------ every CC variant
@pytest.mark.parametrize("lic,current,simulated", [
    ("http://creativecommons.org/licenses/by/4.0/", "R-ACQ-ARXIV_HTML_POLICY", "admitted"),
    ("http://creativecommons.org/licenses/by-sa/4.0/", "R-ACQ-ARXIV_HTML_POLICY", "admitted"),
    ("http://creativecommons.org/licenses/by-nc/4.0/", "R-ACQ-ARXIV_HTML_POLICY", "admitted"),
    ("http://creativecommons.org/licenses/by-nc-sa/4.0/", "R-ACQ-ARXIV_HTML_POLICY", "admitted"),
    ("http://creativecommons.org/licenses/by-nd/4.0/", "R-LIC-REJECT", "R-LIC-REJECT"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "R-LIC-REJECT", "R-LIC-REJECT"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "R-ACQ-ARXIV_LICENCE_HOLD", "R-ACQ-ARXIV_LICENCE_HOLD"),
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "R-ACQ-ARXIV_LICENCE_HOLD", "R-ACQ-ARXIV_LICENCE_HOLD"),
    ("https://example.org/some-licence", "R-ACQ-ARXIV_LICENCE_HOLD", "R-ACQ-ARXIV_LICENCE_HOLD"),
])
def test_each_licence_variant(monkeypatch, lic, current, simulated):
    o, _s = _go(_web(licence=lic))
    e = disc(o)
    assert (e.get("reason") or e["outcome"]) == current, e
    from vera import acq_discovered, auto_acquire, inline_acquire
    auto_acquire.reset_for_tests(); inline_acquire.reset_for_tests(); acq_discovered.reset_for_tests()  # noqa: E702
    monkeypatch.setattr(licence_gate, "licence_decision", _proposed_rule)
    monkeypatch.setattr(fulltext_loader, "licence_decision", _proposed_rule)
    o, _s = _go(_web(licence=lic))
    e = disc(o)
    assert (e.get("reason") or e["outcome"]) == simulated, e


def test_cc0_index_metadata_licence_never_counts():
    """The discovery record carries CC0 METADATA; with no licence link on the abs page the record is held."""
    src = X.discovery_src()
    assert src["licence"]["metadata"]["id"] == "CC0-1.0"
    o, store = _go(_web(licence=None), src=src)
    assert disc(o)["reason"] == "R-ACQ-LICENCE_ABSENT" and store.plans == []


def test_licence_gate_unchanged_for_other_urls():
    lic = {"content": {"id": "cc by", "url": "https://creativecommons.org/licenses/by/4.0/"}}
    assert licence_gate.licence_decision({"url": "https://www.ebi.ac.uk/x", "licence": lic})[0] == "allow"
    for u in ("https://arxiv.org/pdf/2605.23989v1", "https://arxiv.org/html/2605.23989v1",
              "https://arxiv.org/e-print/2605.23989v1"):
        assert licence_gate.licence_decision({"url": u, "licence": lic})[0] == "reject"
    assert licence_gate.licence_decision({"url": "https://arxiv.org/abs/2605.23989v1", "licence": lic})[0] == "allow"
