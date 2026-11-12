"""CONTROL-GAPS-1107: robots.txt and the TDM opt-out on EVERY acquisition hop, plus a coverage registry of outbound
fetch sites. Fake HTTP only (tests/acq_support.Web); no network, no DB.

Before: auto_acquire._guarded_get (Europe PMC REST search, Europe PMC JATS full text, arXiv) checked neither robots.txt
nor the TDM Reservation Protocol; acq_discovered._b_get checked robots.txt only. Both now call the one gate
acq_discovered.hop_access (robots.txt, then /.well-known/tdmrep.json), and refuse a TDM-Reservation: 1 header or an
HTML <meta name="tdm-reservation" content="1">. Every failure fails closed."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import acq_discovered, auto_acquire

ROOT = Path(__file__).resolve().parent.parent
JATS = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"
ROBOTS = "/robots.txt"
TDMREP = "/.well-known/tdmrep.json"


def _txt(body: str, status: int = 200):
    return A.Resp(status=status, body=body.encode(), ctype="text/plain")


def _json(obj, status: int = 200):
    return A.Resp(status=status, body=json.dumps(obj).encode(), ctype="application/json")


def _gg(web, url=JATS, want_json=False, cfg=None):
    return auto_acquire._guarded_get(url, cfg or A.cfg(), A.deps([], web), 100000, 5.0, want_json,
                                     clock=A.clock_at())


def _code(web, **kw):
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, **kw)
    return e.value.code


def _xml_web(over):
    return A.Web({}, {"PMC1": b"<article/>"}, over)


# ------------------------------------------------------------------------------------------------ robots.txt
def test_europepmc_fulltext_refused_when_robots_disallows_and_document_never_requested():
    web = _xml_web({ROBOTS: _txt("User-agent: *\nDisallow: /europepmc/\n")})
    assert _code(web) == "R-ACQ-ROBOTS"
    assert A.doc_urls(web.urls) == []  # the JATS URL itself was never requested


def test_europepmc_robots_module_binds_statement_and_own_family_not_hyperscaler_groups():
    # OD-1108-DISPOSITIONS-2 H13 through the same robots module: a hyperscaler-only group is advisory; the AI
    # statement and an own-family group still refuse.
    for robots in ("# AI crawlers are blocked\nUser-agent: *\nAllow: /\n",
                   "User-agent: GPTBot\nAllow: /\n\nUser-agent: VERA-DAR\nDisallow: /\n\nUser-agent: *\nAllow: /\n"):
        acq_discovered.reset_for_tests()
        assert _code(_xml_web({ROBOTS: _txt(robots)})) == "R-ACQ-ROBOTS"
    acq_discovered.reset_for_tests()
    web = _xml_web({ROBOTS: _txt("User-agent: GPTBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n")})
    _gg(web)  # no refusal: the document is fetched
    assert A.doc_urls(web.urls) != []


@pytest.mark.parametrize("resp,cause", [(_txt("", 500), "STATUS_5XX"), (_txt("", 502), "STATUS_5XX"),
                                        (ConnectionError("down"), "NETWORK")])
def test_robots_unreadable_fails_closed(resp, cause):
    web = _xml_web({ROBOTS: resp})
    # WPL-1111 (ROBOTS-LABEL-1111): an unreadable robots.txt is R-ACQ-ROBOTS_UNREADABLE-<CAUSE>, not a rule (R-ACQ-ROBOTS)
    assert _code(web) == f"R-ACQ-ROBOTS_UNREADABLE-{cause}" and A.doc_urls(web.urls) == []
    assert acq_discovered._ROBOTS["www.ebi.ac.uk"][1] is False  # a real failure is cached briefly (robots_fail_ttl_s)


# ------------------------------------------------------------------------------------------------ r2 (1): throttles
@pytest.mark.parametrize("path", [ROBOTS, TDMREP])
@pytest.mark.parametrize("status", [429, 503])  # 503 = a europepmc portal throttle_status (config/rate_limits.json)
def test_server_throttle_on_gate_reads_is_rate_limited_not_cached_not_robots_or_tdm(path, status):
    web = _xml_web({path: _txt("", status)})
    code = _code(web)
    assert code == "R-ACQ-RATELIMITED" and A.doc_urls(web.urls) == []  # refused for this request
    assert "www.ebi.ac.uk" not in acq_discovered._ROBOTS or acq_discovered._ROBOTS["www.ebi.ac.uk"][1] is not False
    assert "www.ebi.ac.uk" not in acq_discovered._TDMREP  # never cached as a verdict
    from vera import discovery_panel, research_outcome, research_record
    assert research_outcome._CODE_MAP[code] != "no_route" and code not in research_record._FETCH_YES  # not an attempt
    assert "reserves" not in discovery_panel.REASON_TEXT.get(code, "")
    assert _gg(_xml_web({}))[0] == b"<article/>"  # the next request asks again and succeeds


def test_throttle_status_not_in_portal_list_is_a_robots_refusal():
    web = _xml_web({ROBOTS: _txt("", 504)})  # 504 is not in the europepmc throttle_statuses
    assert _code(web) == "R-ACQ-ROBOTS_UNREADABLE-STATUS_5XX"  # WPL-1111: a refusal, now labelled as an unreadable file


def test_job_deadline_on_gate_read_is_not_cached_as_robots():
    t = [0.0]

    def slow(url, **kw):  # every request takes 10 s of the job's time
        t[0] += 10.0
        return _xml_web({})(url, **kw)
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire._guarded_get(JATS, A.cfg(), A.deps([], slow), 100000, 50.0, False, clock=lambda: t[0],
                                  deadline=5.0)
    assert e.value.code == "R-ACQ-DEADLINE"
    # robots.txt answered (404 = no rules, a real verdict); tdmrep.json ran out of JOB time: nothing cached for it
    assert acq_discovered._ROBOTS["www.ebi.ac.uk"][1] is True and "www.ebi.ac.uk" not in acq_discovered._TDMREP


def test_robots_404_and_no_tdmrep_allows_and_order_is_robots_tdmrep_document():
    web = _xml_web({})
    raw, final = _gg(web)
    assert raw == b"<article/>" and final == JATS
    assert web.urls == ["https://www.ebi.ac.uk/robots.txt", "https://www.ebi.ac.uk/.well-known/tdmrep.json", JATS]


def test_resolve_oa_search_api_is_gated_too():
    web = A.Web({}, {}, {ROBOTS: _txt("User-agent: *\nDisallow: /europepmc/webservices/rest/search\n")})
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire.resolve_oa("10.9999/x", A.cfg(), A.deps([], web), clock=A.clock_at())
    # WPL-1111: resolve_oa is the full-text service hop (Europe PMC), so the rule belongs to the service, not the source
    assert e.value.code == "R-ACQ-ROUTE_ROBOTS" and A.doc_urls(web.urls) == []


# ------------------------------------------------------------------------------------------------ r2 (2): host check
# Userinfo with an EMPTY password: urlsplit gives username "user", and acq_discovered._host_ok refuses any URL whose
# username is set, the same refusal as user:password. Chosen because it is neither credential-shaped (no password) nor
# email-shaped (":" before "@") for the release scan; test_empty_password_userinfo_is_refused_for_its_username pins it.
@pytest.mark.parametrize("target", ["https://www.ebi.ac.uk:8443/x.xml", "https://user:@www.ebi.ac.uk/x.xml"])
def test_redirect_to_other_port_or_userinfo_is_refused_on_the_existing_path(target):
    web = _xml_web({JATS: A.Resp(status=302, headers={"Location": target})})
    assert _code(web) == "R-ACQ-HOST"  # chain=None (the Europe PMC path): _host_ok on every hop
    assert not any(":8443" in u or "@" in u for u in web.urls)


def test_empty_password_userinfo_is_refused_for_its_username():
    from urllib.parse import urlsplit
    sp = urlsplit("https://user:@www.ebi.ac.uk/x.xml")
    assert sp.username == "user" and sp.password == ""  # username present: the reason _host_ok refuses it
    assert not acq_discovered._host_ok("https://user:@www.ebi.ac.uk/x.xml", {"www.ebi.ac.uk"})
    assert acq_discovered._host_ok("https://www.ebi.ac.uk/x.xml", {"www.ebi.ac.uk"})


# ------------------------------------------------------------------------------------------------ r2 (3): each gate
def test_uncached_gate_runs_before_the_document_address_check():
    """_guarded_get's first gate: with nothing cached, robots.txt and tdmrep.json are read BEFORE the document URL's
    address check, so that check is the last validation (its pin) before the document request."""
    events = []
    web = _xml_web({})

    def get(url, **kw):
        events.append(("get", url))
        return web(url, **kw)
    d = A.deps([], get)
    d.url_check = lambda u: events.append(("check", u)) or (True, "")
    auto_acquire._guarded_get(JATS, A.cfg(), d, 100000, 5.0, False, clock=A.clock_at())
    i_doc_check = events.index(("check", JATS))
    assert events[i_doc_check + 1] == ("get", JATS)
    assert ("get", "https://www.ebi.ac.uk/robots.txt") in events[:i_doc_check]
    assert ("get", "https://www.ebi.ac.uk/.well-known/tdmrep.json") in events[:i_doc_check]


def test_cached_gate_still_judges_each_path():
    """_guarded_get's second gate: with both verdicts cached (no request), a path the cached robots.txt disallows is
    refused; a cached tdmrep rule for the path refuses too."""
    from urllib import robotparser
    rp = robotparser.RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /europepmc/webservices/rest/PMC1/"])
    from vera import robots_ai
    robots_ai.annotate(rp, "User-agent: *\nDisallow: /europepmc/webservices/rest/PMC1/\n")
    acq_discovered._ROBOTS["www.ebi.ac.uk"] = (1.0, rp)
    acq_discovered._TDMREP["www.ebi.ac.uk"] = (1.0, [])
    web = _xml_web({})
    assert _code(web) == "R-ACQ-ROBOTS" and web.urls == []  # judged from the cache: nothing requested
    acq_discovered._ROBOTS["www.ebi.ac.uk"] = (1.0, True)
    acq_discovered._TDMREP["www.ebi.ac.uk"] = (1.0, [{"location": "/europepmc/", "tdm-reservation": 1}])
    assert _code(web) == "R-ACQ-TDM" and web.urls == []


def test_robots_cache_expires_after_its_ttl():
    cfg = A.cfg()
    ttl = float(cfg["inline"]["landing_route"]["robots_ttl_s"]) if isinstance(cfg["inline"].get("landing_route"), dict) \
        else float((cfg.get("landing_route") or {}).get("robots_ttl_s", 3600))
    t = [1.0]
    clock = lambda: t[0]  # noqa: E731
    deny = _xml_web({ROBOTS: _txt("User-agent: *\nDisallow: /\n")})
    with pytest.raises(auto_acquire._Stop):
        auto_acquire._guarded_get(JATS, cfg, A.deps([], deny), 100000, 5.0, False, clock=clock)
    t[0] += ttl - 1  # still fresh: the cached refusal stands, robots.txt not re-read
    allow = _xml_web({})
    with pytest.raises(auto_acquire._Stop):
        auto_acquire._guarded_get(JATS, cfg, A.deps([], allow), 100000, 5.0, False, clock=clock)
    assert allow.urls == []
    t[0] += 2  # expired: robots.txt is read again and now allows
    assert auto_acquire._guarded_get(JATS, cfg, A.deps([], allow), 100000, 5.0, False, clock=clock)[0] == b"<article/>"
    assert allow.urls[0] == "https://www.ebi.ac.uk/robots.txt"


# ------------------------------------------------------------------------------------------------ r2 (4): crawl-delay
def _crawl_cfg(on: bool):
    cfg = A.cfg()
    cfg["crawl_delay"] = {"enforce_on_acquisition": on}
    return cfg


def test_crawl_delay_flag_ships_on_and_off_means_no_wait(monkeypatch):
    assert A.cfg()["crawl_delay"]["enforce_on_acquisition"] is True  # H5-CRAWL-1107: H-5(a) approved (OD-1107-HITL)
    waits = []
    monkeypatch.setattr(acq_discovered, "_sleep", waits.append)
    web = _xml_web({ROBOTS: _txt("User-agent: *\nCrawl-delay: 10\n")})
    for _ in range(2):
        _gg(web, cfg=_crawl_cfg(False))
    assert waits == []


@pytest.mark.parametrize("fn", ["guarded", "b_get"])
def test_crawl_delay_enforced_when_flag_on(monkeypatch, fn):
    waits, t = [], [100.0]
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: waits.append(s) or t.__setitem__(0, t[0] + s))
    web = _xml_web({ROBOTS: _txt("User-agent: *\nCrawl-delay: 10\n")})
    cfg = _crawl_cfg(True)
    for _ in range(2):
        if fn == "guarded":
            auto_acquire._guarded_get(JATS, cfg, A.deps([], web), 100000, 5.0, False, clock=lambda: t[0])
        else:
            acq_discovered._b_get(JATS, {"www.ebi.ac.uk"}, {"application/xml"}, 100000, 2, A.deps([], web), cfg,
                                  lambda: t[0], None, None)
    # H5-CRAWL-1107: every request to the host is paced, the gate's own reads too: robots.txt (no delay known yet),
    # tdmrep.json +10, the document +10; the second document +10.
    assert [round(w) for w in waits] == [10, 10, 10]


def test_crawl_delay_past_the_deadline_is_a_crawl_delay_stop_not_an_attempt(monkeypatch):
    monkeypatch.setattr(acq_discovered, "_sleep", lambda s: None)
    web = _xml_web({ROBOTS: _txt("User-agent: *\nCrawl-delay: 10\n")})
    cfg = _crawl_cfg(True)
    from tests.test_h5_crawl_1107 import seed_robots  # H5-CRAWL-1107: gate reads cached, so only documents are paced
    seed_robots("www.ebi.ac.uk", "User-agent: *\nCrawl-delay: 10\n")
    auto_acquire._guarded_get(JATS, cfg, A.deps([], web), 100000, 5.0, False, clock=A.clock_at(1.0), deadline=5.0)
    n = len(web.urls)
    with pytest.raises(auto_acquire._Stop) as e:
        auto_acquire._guarded_get(JATS, cfg, A.deps([], web), 100000, 5.0, False, clock=A.clock_at(1.0), deadline=5.0)
    from vera import research_record
    assert e.value.code == "R-ACQ-CRAWL_DELAY" and len(web.urls) == n and e.value.code not in research_record._FETCH_YES


def test_every_redirect_hop_is_gated():
    other = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2/fullTextXML"  # OA-2: eutils left fetch.allowed_hosts; same host, robots-blocked path
    web = _xml_web({JATS: A.Resp(status=302, headers={"Location": other}),
                    ROBOTS: _txt("User-agent: *\nDisallow: /europepmc/webservices/rest/PMC2/\n")})
    assert _code(web) == "R-ACQ-ROBOTS"
    assert other not in web.urls and JATS in web.urls


def test_run_job_end_to_end_stores_nothing_when_robots_disallows():
    web, _raw = A.epi_setup(over={ROBOTS: _txt("User-agent: *\nDisallow: /\n")})
    store = A.Store()
    out = auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                               clock=A.clock_at(), env=A.ENV_ON)
    assert store.plans == [] and out["codes"] == {"R-ACQ-ROUTE_ROBOTS": 1}, out  # WPL-1111: the service hop


# ------------------------------------------------------------------------------------------------ TDM reservation
def test_tdmrep_reservation_refuses():
    web = _xml_web({TDMREP: _json([{"location": "/europepmc/*", "tdm-reservation": 1}])})
    assert _code(web) == "R-ACQ-TDM" and A.doc_urls(web.urls) == []


def test_tdmrep_reservation_on_another_path_allows():
    web = _xml_web({TDMREP: _json([{"location": "/other/", "tdm-reservation": 1},
                                   {"location": "/", "tdm-reservation": 0}])})
    assert _gg(web)[0] == b"<article/>"


@pytest.mark.parametrize("resp", [_txt("", 500), _txt("", 403), A.Resp(body=b"{not json", ctype="application/json"),
                                  _json({"not": "an array"}), TimeoutError("slow")])
def test_tdmrep_unreadable_fails_closed(resp):
    web = _xml_web({TDMREP: resp})
    assert _code(web) == "R-ACQ-TDM"


def test_tdm_reservation_header_refuses():
    web = _xml_web({JATS: A.Resp(body=b"<article/>", ctype="application/xml", headers={"TDM-Reservation": "1"})})
    assert _code(web) == "R-ACQ-TDM"


def test_tdm_meta_on_an_html_page_refuses():
    cfg = A.cfg()
    cfg["fetch"]["content_types"] = list(cfg["fetch"]["content_types"]) + ["text/html"]
    page = b'<html><head><meta name="tdm-reservation" content="1"></head><body>x</body></html>'
    web = _xml_web({JATS: A.Resp(body=page, ctype="text/html")})
    assert _code(web, cfg=cfg) == "R-ACQ-TDM"


def test_landing_route_b_get_is_gated_for_tdm_too():
    web = _xml_web({TDMREP: _json([{"location": "/", "tdm-reservation": 1}])})
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get(JATS, {"www.ebi.ac.uk"}, {"application/xml"}, 100000, 2, A.deps([], web), A.cfg(),
                              A.clock_at(), None, None)
    assert e.value.code == "R-ACQ-TDM"


def test_tdm_code_is_mapped_for_the_ui_and_outcome():
    from vera import discovery_panel, research_outcome
    assert auto_acquire.acq_config.reason(A.cfg(), "tdm") == "R-ACQ-TDM"
    assert "R-ACQ-TDM" in discovery_panel.REASON_TEXT and research_outcome._CODE_MAP["R-ACQ-TDM"] == "no_route"


# ------------------------------------------------------------------------------------------------ coverage registry
# Every function in vera/ that references an outbound HTTP primitive, and how it is covered. A new fetch path that is
# not listed here fails test_every_outbound_fetch_site_is_registered; a "gate" entry whose function no longer names
# its robots check fails test_gated_sites_still_name_their_robots_check.
GATE, TRANSPORT, ROBOTS_FETCH = "gate", "transport", "robots_fetch"
FIRST_PARTY, LLM_API = "exempt_first_party", "exempt_llm_api"
NO_REQUEST = "no_request"  # references a network module only to classify errors or parse URLs
# API-ALLOW-1107 (owner OD-1107-HITL): every API site goes through vera.api_permissions (config/api_permissions.json,
# deny by default). The former class "api_no_robots_check_owner_decision" is gone. Entry: (API_GATE, the function that
# holds the gate call, the call it must make).
API_GATE = "api_permission_gate"
TRACKED_GAP = "tracked_gap"  # none left: the Wikipedia portal became robots-gated at e87ef55 (WIKI-REDESIGN-1107)

REGISTRY: dict[str, tuple] = {
    "vera/acq_discovered.py:_b_get": (GATE, "hop_access("),
    "vera/auto_acquire.py:_guarded_get": (GATE, "hop_access("),
    "vera/selective_fetch.py:_download": (GATE, "robots_allowed"),
    "vera/search_providers.py:landing_doi": (GATE, "robots_ok("),
    "vera/selective_fetch.py:fetch_html": (GATE, "_download("),
    "vera/selective_fetch.py:fetch_candidate": (GATE, "_download("),
    "vera/auto_acquire.py:default_deps": (TRANSPORT, "vera/auto_acquire.py:_guarded_get", "hop_access("),
    "vera/selective_fetch.py:pinned_get": (TRANSPORT, "vera/selective_fetch.py:_download", "robots_allowed"),
    "vera/public_fetch.py:default_transport": (TRANSPORT, "vera/public_fetch.py:fetch", "robots_verdict("),
    "vera/public_fetch.py:default_transport._Conn": (TRANSPORT, "vera/public_fetch.py:fetch", "robots_verdict("),
    "vera/public_fetch.py:default_transport._Pool": (TRANSPORT, "vera/public_fetch.py:fetch", "robots_verdict("),
    "vera/selective_fetch.py:_connect_phase_error": (NO_REQUEST,),
    "vera/selective_fetch.py:_parsers_agree": (NO_REQUEST,),
    "vera/search_providers.py:ddg_default_transport": (TRANSPORT, "vera/search_providers.py:DuckDuckGoLiteProvider.call",
                                                       "self._robots_check("),
    "vera/selective_fetch.py:robots_allowed": (ROBOTS_FETCH,),
    "vera/search_providers.py:DuckDuckGoLiteProvider._default_robots": (ROBOTS_FETCH,),
    "vera/public_mode.py:get_corpus_summary": (FIRST_PARTY,),
    "vera/public_mode.py:get_progress": (FIRST_PARTY,),
    "vera/public_mode.py:post_ask": (FIRST_PARTY,),
    "vera/public_mode.py:post_ask_tracked": (FIRST_PARTY,),
    "vera/ui_wake.py:<module>": (FIRST_PARTY,),
    "vera/ui_wake.py:is_healthy": (FIRST_PARTY,),
    "vera/ui_wake.py:kick_restart": (FIRST_PARTY,),
    "vera/api_kickstart.py:default_health": (FIRST_PARTY,),
    "vera/api_triage.py:triage_api": (FIRST_PARTY,),
    "vera/trace_eval/ui.py:fetch_live": (FIRST_PARTY,),
    "vera/trace_eval/ui.py:fetch_private": (FIRST_PARTY,),
    "vera/memory_review_panel.py:decide": (FIRST_PARTY,),
    "vera/memory_review_panel.py:fetch_pending": (FIRST_PARTY,),
    "vera/judge_select.py:_collect": (LLM_API,),
    "vera/judge_select.py:candidates": (LLM_API,),
    "vera/judge_select.py:default_http_get": (LLM_API,),
    "vera/judge_select.py:default_judge_transport": (LLM_API,),
    "vera/judge_select.py:list_models": (LLM_API,),
    "vera/judge_select.py:select_judge": (LLM_API,),
    "vera/m6/judge.py:OpenAICompatJudge._call": (LLM_API,),
    "vera/m6/judge.py:preflight_judge": (LLM_API,),
    "vera/mvp_preflight.py:run_preflight": (LLM_API,),
    # CHUNK-TAXO-INT-1109 (OD-1109-DEMB-3): Cloudflare Workers AI embeddings API; https to api.cloudflare.com only, pinned in
    # code; a vendor API call like the LLM calls above, not a crawl (no robots/TDM question, no api_permissions entry).
    "vera/chunk_taxo_embed.py:CloudflareEmbedder._post": (LLM_API,),
    "vera/chunk_taxo_embed.py:_NoRedirect": (LLM_API,),
    "vera/chunk_taxo_embed.py:_default_open": (LLM_API,),
    "vera/run_mvp.py:run_mvp": (LLM_API,),
    "vera/rate_limit.py:httpx_transport._LimitedTransport.__init__": (LLM_API,),
    "vera/rate_limit.py:openai_http_client": (LLM_API,),
    "vera/search_providers.py:arxiv_default_transport": (API_GATE, "vera/search_providers.py:arxiv_default_transport", "api_permissions.call("),
    "vera/search_providers.py:openalex_default_transport": (API_GATE, "vera/search_providers.py:openalex_default_transport", "api_permissions.call("),
    "vera/search_providers.py:europepmc_default_transport": (API_GATE, "vera/search_providers.py:europepmc_default_transport", "api_permissions.call("),
    "vera/search_providers.py:doaj_default_transport": (API_GATE, "vera/search_providers.py:doaj_default_transport", "api_permissions.call("),
    "vera/wiki_ref_portal.py:_api_get": (API_GATE, "vera/wiki_ref_portal.py:_api_get", "api_permissions.call("),  # SEARCHCAPS-1110 Wikipedia search API
    "vera/portal_inspire.py:_default_get": (API_GATE, "vera/portal_inspire.py:_default_get", "api_permissions.call("),  # ROUTES-1108 inspire (ships owner_switch deny)
    "vera/list_ingest.py:_api_get": (API_GATE, "vera/list_ingest.py:_api_get", "api_permissions.call("),  # LIST-INGEST-1110 identifier-only metadata lookups
    "vera/source_resolver.py:_api_get": (API_GATE, "vera/source_resolver.py:_api_get", "api_permissions.call("),
    "vera/source_resolver.py:_s2": (API_GATE, "vera/source_resolver.py:_api_get", "api_permissions.call("),
    # OPENALEX-FIX-1110: the OpenAlex leg of link resolution goes through the metered client, whose transport is
    # openalex_default_transport (api_permissions.call); a test's injected http_get replaces only the transport
    "vera/source_resolver.py:_oa_get": (API_GATE, "vera/search_providers.py:openalex_default_transport", "api_permissions.call("),
    "vera/source_resolver.py:lookup_doi": (API_GATE, "vera/source_resolver.py:_api_get", "api_permissions.call("),
    "vera/source_resolver.py:lookup_title": (API_GATE, "vera/source_resolver.py:_api_get", "api_permissions.call("),
    "vera/reporting_lane.py:ReportingLane.__init__": (API_GATE, "vera/reporting_lane.py:_default_http_get", "api_permissions.require("),
    "vera/reporting_lane.py:_open": (API_GATE, "vera/reporting_lane.py:_default_http_get", "api_permissions.require("),
    "vera/reporting_lane.py:_no_redirect_handler._H": (API_GATE, "vera/reporting_lane.py:_default_http_get", "api_permissions.require("),
    # WIKI-REDESIGN-1107 (e87ef55): article pages only, each after selective_fetch.robots_allowed (portal ships off)
    "vera/wiki_ref_portal.py:discover": (GATE, "robots_ok("),
    # ROUTES-1108: the generic reference portal. fetch_page calls deps.robots_ok (selective_fetch.robots_allowed: own UA,
    # * and AI-agent groups, fail closed) before its one GET; the host's tdmrep.json is read only from fetch_page, after
    # the page's robots check passed (TDMRep well-known file; same rate portal, byte cap and deadline).
    "vera/reference_portal.py:fetch_page": (GATE, "robots_ok("),
    "vera/reference_portal.py:_default_deps": (TRANSPORT, "vera/reference_portal.py:fetch_page", "robots_ok("),
    "vera/reference_portal.py:_tdmrep_ok": (TRANSPORT, "vera/reference_portal.py:fetch_page", "robots_ok("),
    "vera/wiki_ref_portal.py:_default_robots.ok": (ROBOTS_FETCH,),
    "vera/wiki_ref_portal.py:get_article": (TRANSPORT, "vera/wiki_ref_portal.py:discover", "robots_ok("),
    "vera/wiki_ref_portal.py:make_portal_search.search": (TRANSPORT, "vera/wiki_ref_portal.py:discover", "robots_ok("),
    "vera/search_providers.py:acquisition_search": (TRANSPORT, "vera/wiki_ref_portal.py:discover", "robots_ok("),
}

_REQ = {"get", "post", "put", "head", "request", "Session"}
_NAMES = ("pinned_get", "http_get", "http_post", "urlopen")
# Network client modules: any import of one (any alias, any form) must be listed in IMPORTS (r2 (3)).
_NET_MODULES = ("requests", "httpx", "urllib.request", "http.client", "urllib3", "aiohttp", "socket")
IMPORTS = {  # file -> the network modules it may import (anything else fails test_network_imports_are_registered)
    "vera/acq_discovered.py": set(), "vera/chunk_taxo_embed.py": {"urllib.request"}, "vera/api_kickstart.py": {"requests"}, "vera/api_triage.py": {"requests"},
    "vera/judge_select.py": {"requests"}, "vera/m6/judge.py": {"requests"},
    "vera/memory_review_panel.py": {"requests"}, "vera/public_fetch.py": {"requests", "socket", "urllib3"},
    "vera/public_mode.py": {"requests"}, "vera/progress_ui.py": {"requests"}, "vera/rate_limit.py": {"httpx"}, "vera/reporting_lane.py": {"urllib.request"},
    "vera/search_and_fetch.py": {"requests"}, "vera/search_providers.py": {"requests"},
    "vera/selective_fetch.py": {"requests", "socket", "urllib3"}, "vera/source_resolver.py": {"requests"},
    "vera/trace_eval/ui.py": {"requests"}, "vera/ui_wake.py": {"requests"}, "vera/wiki_ref_portal.py": {"requests"},
}


def _net_mod(name: str) -> str | None:
    for m in _NET_MODULES:
        if name == m or name.startswith(m + "."):
            return m
    return None


def _imports(tree) -> tuple[set, dict]:
    """(network modules imported, local alias -> "module" or "module:attr") for one file, at any nesting level."""
    mods, alias = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                m = _net_mod(a.name)
                if m:
                    mods.add(m)
                    alias[a.asname or a.name.split(".")[0]] = m if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                full = f"{n.module}.{a.name}"
                m = _net_mod(n.module) or _net_mod(full)
                if m:
                    mods.add(m)
                    alias[a.asname or a.name] = full if _net_mod(full) == full else f"{n.module}:{a.name}"
    return mods, alias


def _scan(path: Path, rel: str) -> dict:
    """{"<rel>:<qualname>": function source} for every scope that references an outbound HTTP primitive."""
    src = path.read_text(encoding="utf-8")
    out, stack = {}, []
    _mods, alias = _imports(ast.parse(src))
    net_alias = {k for k, v in alias.items() if ":" in v or v in ("requests", "httpx", "urllib3", "aiohttp")
                 or v in ("urllib.request", "http.client")}

    class V(ast.NodeVisitor):
        def _scope(self, n):
            stack.append(n)
            self.generic_visit(n)
            stack.pop()
        visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

        def _hit(self):
            key = f"{rel}:{'.'.join(s.name for s in stack) or '<module>'}"
            out[key] = ast.get_source_segment(src, stack[-1]) if stack else ""

        def visit_Attribute(self, n):
            v = n.value
            if isinstance(v, ast.Name) and ((v.id == "requests" and n.attr in _REQ)
                                            or (v.id == "httpx" and n.attr in {"Client", "HTTPTransport", "get", "post"})):
                self._hit()
            if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and (
                    (v.value.id == "urllib" and v.attr == "request") or (v.value.id == "http" and v.attr == "client")):
                self._hit()
            if isinstance(v, ast.Name) and v.id in net_alias and v.id not in ("requests", "httpx", "urllib"):
                self._hit()
            if n.attr in _NAMES:
                self._hit()
            self.generic_visit(n)

        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and (n.id in _NAMES or (n.id in net_alias and n.id not in
                                                                    ("requests", "httpx", "urllib"))):
                self._hit()  # r2: an aliased module (import requests as rq) or name (from requests import get as x)
    V().visit(ast.parse(src))
    return out


def _all_sites() -> dict:
    out = {}
    for p in sorted((ROOT / "vera").rglob("*.py")):
        if "tests" in p.relative_to(ROOT).parts:
            continue
        out.update(_scan(p, p.relative_to(ROOT).as_posix()))
    return out


def _func_node(key: str):
    rel, qual = key.split(":", 1)
    node = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for part in qual.split("."):
        node = next(n for n in ast.iter_child_nodes(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part)
    return node


def _calls(key: str, name: str) -> bool:
    """True when the function's AST holds a CALL to `name` (a bare name or an attribute): comments, strings and
    docstrings naming it do not count (r2 (3))."""
    name = name.rstrip("(").split(".")[-1]
    for n in ast.walk(_func_node(key)):
        if isinstance(n, ast.Call):  # the callee expression, e.g. f(...), x.f(...) or (a or f)(...)
            for m in ast.walk(n.func):
                if (isinstance(m, ast.Name) and m.id == name) or (isinstance(m, ast.Attribute) and m.attr == name):
                    return True
    return False


def test_every_outbound_fetch_site_is_registered():
    found = set(_all_sites())
    new, stale = sorted(found - set(REGISTRY)), sorted(set(REGISTRY) - found)
    assert not new, ("new outbound fetch site(s) without a robots/TDM classification; route them through "
                     f"acq_discovered.hop_access, public_fetch.fetch or selective_fetch.robots_allowed: {new}")
    assert not stale, f"registry entries no longer found (update REGISTRY): {stale}"


def test_gated_sites_still_name_their_robots_check():
    for key, entry in REGISTRY.items():
        if entry[0] == GATE:
            assert _calls(key, entry[1]), f"{key} no longer calls {entry[1]}"
        elif entry[0] == TRANSPORT:
            assert _calls(entry[1], entry[2]), f"{key}'s gate {entry[1]} no longer calls {entry[2]}"
        elif entry[0] == API_GATE:
            assert entry[2].startswith("api_permissions."), f"{key}: an API site must name an api_permissions gate"
            assert _calls(entry[1], entry[2]), f"{key}'s API gate {entry[1]} no longer calls {entry[2]}"


def test_gate_check_ignores_comments_and_strings(tmp_path, monkeypatch):
    src = 'def f(u):\n    # hop_access(u) is NOT called here\n    s = "hop_access(u)"\n    return requests.get(u)\n'
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "m.py").write_text(src, encoding="utf-8")
    import sys
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert not _calls("vera/m.py:f", "hop_access(")


def test_registry_fails_on_an_ungated_api_site(tmp_path, monkeypatch):
    """API-ALLOW-1107: an API site registered as api_permission_gate whose gate function does not call the gate (only
    names it in a comment or string) fails the registry check; an unregistered one fails as a new site."""
    src = ('def gated(u):\n    return api_permissions.call(requests.get, u)\n'
           'def ungated(u):\n    # api_permissions.call( is NOT called here\n    s = "api_permissions.call("\n'
           '    return requests.get(u)\n')
    (tmp_path / "vera").mkdir()
    (tmp_path / "vera" / "m.py").write_text(src, encoding="utf-8")
    import sys
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert _calls("vera/m.py:gated", "api_permissions.call(")
    assert not _calls("vera/m.py:ungated", "api_permissions.call(")
    monkeypatch.setattr(sys.modules[__name__], "REGISTRY",
                        {"vera/m.py:ungated": (API_GATE, "vera/m.py:ungated", "api_permissions.call(")})
    with pytest.raises(AssertionError, match="no longer calls"):
        test_gated_sites_still_name_their_robots_check()
    monkeypatch.setattr(sys.modules[__name__], "REGISTRY", {})
    with pytest.raises(AssertionError, match="new outbound fetch site"):
        test_every_outbound_fetch_site_is_registered()


FORMER_UNGATED_API_SITES = (  # CONTROL-GAPS-1107's frozen list of 12 (renamed only where the code changed)
    "vera/search_providers.py:arxiv_default_transport", "vera/search_providers.py:openalex_default_transport",
    "vera/search_providers.py:europepmc_default_transport", "vera/search_providers.py:doaj_default_transport",
    "vera/source_resolver.py:_api_get",  # was "<module>" (the module-level limited getter, now this function)
    "vera/source_resolver.py:_s2", "vera/source_resolver.py:lookup_doi", "vera/source_resolver.py:lookup_title",
    "vera/reporting_lane.py:ReportingLane.__init__",  # PORTAL-WIRE-B-1110: source_resolver.with_abstract removed (no portal abstract fetch)
    "vera/reporting_lane.py:_open",  # was "_default_http_get" (its urllib body moved here, behind the gate)
    "vera/reporting_lane.py:_no_redirect_handler._H")


@pytest.mark.parametrize("site", FORMER_UNGATED_API_SITES)
def test_each_former_ungated_api_site_calls_the_gate(site):
    """API-ALLOW-1107 r2 (5): every one of the 12 former api_no_robots_check_owner_decision sites is still found by the
    scan, is classified api_permission_gate, and its gate function really calls the api_permissions gate."""
    assert len(FORMER_UNGATED_API_SITES) == 11  # 12 frozen; PORTAL-WIRE-B-1110 deleted source_resolver.with_abstract
    assert site in _all_sites(), f"{site} no longer found by the scan: update the list with the reason"
    kind, gate_fn, call = REGISTRY[site]
    assert kind == API_GATE and call.startswith("api_permissions.")
    assert _calls(gate_fn, call), f"{site}: {gate_fn} does not call {call}"


def test_aliased_imports_are_caught(tmp_path):
    src = ("import requests as rq\nfrom requests import get as fetch_it\nimport http.client\n"
           "def a(u):\n    return rq.get(u)\n"
           "def b(u):\n    return fetch_it(u)\n"
           "def c(h):\n    return http.client.HTTPSConnection(h)\n")
    p = tmp_path / "x.py"
    p.write_text(src, encoding="utf-8")
    assert set(_scan(p, "x.py")) == {"x.py:a", "x.py:b", "x.py:c"}
    assert _imports(ast.parse(src))[0] == {"requests", "http.client"}


def test_network_imports_are_registered():
    found = {}
    for p in sorted((ROOT / "vera").rglob("*.py")):
        if "tests" in p.relative_to(ROOT).parts:
            continue
        mods = _imports(ast.parse(p.read_text(encoding="utf-8")))[0]
        if mods:
            found[p.relative_to(ROOT).as_posix()] = mods
    extra = {f: sorted(m - IMPORTS.get(f, set())) for f, m in found.items() if m - IMPORTS.get(f, set())}
    assert not extra, f"unregistered network-module imports (classify the new fetch path first): {extra}"


def test_no_ungated_api_sites_remain():
    """API-ALLOW-1107: the frozen CONTROL-GAPS-1107 list of 12 ungated API sites is now fully gated (api_permission_gate);
    no entry may use any other class for an API site, and none may come back as ungated."""
    assert all(e[0] in (GATE, TRANSPORT, ROBOTS_FETCH, FIRST_PARTY, LLM_API, NO_REQUEST, API_GATE, TRACKED_GAP)
               for e in REGISTRY.values())
    assert sum(1 for e in REGISTRY.values() if e[0] == API_GATE) == 15  # OPENALEX-FIX-1110: + source_resolver._oa_get; SEARCHCAPS-1110: + wiki_ref_portal._api_get (Wikipedia search API); LIST-INGEST-1110: + list_ingest._api_get; PORTAL-WIRE-B-1110: - source_resolver.with_abstract
    assert sum(1 for e in REGISTRY.values() if e[0] == TRACKED_GAP) == 0  # Wikipedia portal gated since e87ef55
