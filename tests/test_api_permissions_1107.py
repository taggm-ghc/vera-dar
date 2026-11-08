"""API-ALLOW-1107 (owner OD-1107-HITL, 2026-11-07): deny-by-default API permission gate (vera/api_permissions.py,
config/api_permissions.json). Fakes only; no network, no DB.

Checks: a denied API sends no request (search transports, source_resolver, reporting lane, acquisition hop incl. its
robots.txt read) and logs the fixed code R-API-DENIED; a permitted entry enforces its conditions from config (rate
portal, ceilings, 429 handling, User-Agent, no email, content scope); a missing entry denies; an invalid file denies
everything; the owner switch flips an entry with no code change."""
from __future__ import annotations

import copy
import json
import logging

import pytest

from tests import acq_support as A
from vera import api_permissions as AP
from vera import auto_acquire, rate_limit, reporting_lane, search_providers, source_resolver

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_JATS = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"
CROSSREF_UA = "VERA-DAR/1.0 (+https://github.com/taggm-ghc/vera-dar)"


def _raw() -> dict:
    return json.loads(AP.CONFIG_PATH.read_text(encoding="utf-8"))


def _policy(edit=None) -> AP.Policy:
    raw = copy.deepcopy(_raw())
    if edit:
        edit(raw["entries"])
    return AP._validate(raw)


class _Counter:
    def __init__(self, resp=None):
        self.calls, self.resp = [], resp or A.Resp(body=b"{}")

    def __call__(self, url, *a, **kw):
        self.calls.append((url, kw))
        return self.resp


# ------------------------------------------------------------------------------------------------ shipped verdicts
@pytest.mark.parametrize("url", ["https://api.openalex.org/works?search=x", "https://api.crossref.org/works/10.1/x",
                                 "https://api.gdeltproject.org/api/v2/doc/doc?query=x"])
def test_shipped_permitted_entries(url):
    d = AP.decide(url, "metadata")
    assert d.allowed, d


@pytest.mark.parametrize("url,entry", [
    ("https://doaj.org/api/v4/search/articles/x", "doaj_api"), ("https://export.arxiv.org/api/query?x", "arxiv_export_api"),
    ("https://arxiv.org/api/query", "arxiv_web_api"), ("https://api.unpaywall.org/v2/10.1/x", "unpaywall"),
    ("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/x", "semantic_scholar"),
    ("https://hn.algolia.com/api/v1/search?query=x", "hn_algolia")])
def test_shipped_denied_entries(url, entry):
    d = AP.decide(url, "metadata")
    assert (d.allowed, d.code, d.entry) == (False, AP.DENIED, entry)


def test_missing_entry_denies():
    d = AP.decide("https://api.example.org/v1/search?q=x")
    assert (d.allowed, d.code, d.entry, d.why) == (False, AP.DENIED, None, "no_entry")
    # a path outside every prefix on a mixed host is no entry either (and is website content, not an API URL)
    assert not AP.decide("https://www.ebi.ac.uk/other/").allowed
    assert not AP.is_api_url("https://www.ebi.ac.uk/other/") and not AP.is_api_url("https://arxiv.org/abs/2401.00001")
    assert AP.is_api_url("https://api.openalex.org/anything") and AP.is_api_url(EPMC_JATS)


def test_unreadable_or_invalid_config_denies_everything(tmp_path, caplog):
    bad = tmp_path / "p.json"
    bad.write_text('{"default": "allow", "entries": {}}', encoding="utf-8")
    for pol in (AP.load_policy(bad), AP.load_policy(tmp_path / "missing.json")):
        assert pol.error and not AP.decide("https://api.openalex.org/works", policy=pol).allowed


def test_shipped_file_has_required_fields_and_no_secret_or_email():
    raw = _raw()
    assert raw["default"] == "deny"
    for eid, e in raw["entries"].items():
        assert e["evidence"] and all(ev.get("url") and ev.get("read_level") and ev.get("read") for ev in e["evidence"])
        assert e["review_by"] >= e["read"] and e["owner_switch"] in ("default", "permit", "deny"), eid
    assert "@" not in json.dumps(raw)


# ------------------------------------------------------------------------------------------------ denied = no request
def test_denied_search_transport_sends_nothing_and_logs_code(monkeypatch, caplog):
    _deny_epmc()
    import requests
    fake = _Counter()
    monkeypatch.setattr(requests, "get", fake)
    with caplog.at_level(logging.WARNING, logger="vera"):
        for tx in (search_providers.europepmc_default_transport, search_providers.doaj_default_transport,
                   search_providers.arxiv_default_transport):
            url = {"europepmc_default_transport": EPMC_SEARCH, "doaj_default_transport": "https://doaj.org/api/v4/x",
                   "arxiv_default_transport": "https://export.arxiv.org/api/query"}[tx.__name__]
            with pytest.raises(AP.ApiDenied) as e:
                tx(url, {"query": "x"}, {"User-Agent": "t"}, 1.0)
            assert e.value.code == AP.DENIED
    assert fake.calls == []
    assert sum("R-API-DENIED" in r.getMessage() for r in caplog.records) == 3
    assert not any("query" in r.getMessage() or "ebi.ac.uk" in r.getMessage() for r in caplog.records)


def test_europepmc_search_with_the_default_transport_is_skipped_not_sent(monkeypatch):
    _deny_epmc()
    import requests
    fake = _Counter()
    monkeypatch.setattr(requests, "get", fake)
    with pytest.raises(AP.ApiDenied):
        search_providers.europepmc_search("measles transmission", 5, 1.0, {"search_url": EPMC_SEARCH})
    assert fake.calls == []


def test_source_resolver_skips_denied_semantic_scholar_and_keeps_openalex(monkeypatch):
    work = {"id": "W1", "doi": "https://doi.org/10.1/x", "title": "T", "publication_date": "2020-01-01",
            "abstract_inverted_index": None, "type": "article"}
    resp = A.Resp(body=json.dumps(work).encode())
    resp.text = json.dumps(work)
    fake = _Counter(resp)
    monkeypatch.setattr(source_resolver.requests, "get", fake)
    rec = source_resolver.lookup_doi("10.1/x", 1.0)
    assert rec and rec["title"] == "T" and rec["snippet"] == ""  # no abstract: S2 fill skipped
    assert [u for u, _ in fake.calls] == [source_resolver.OPENALEX_WORK.format(doi="10.1/x")]


def test_reporting_lane_denied_portal_is_skipped_without_request_or_backoff(monkeypatch):
    opened = []
    monkeypatch.setattr(reporting_lane, "_open", lambda *a, **k: opened.append(a) or (200, {}, b"{}"))
    lane = reporting_lane.ReportingLane(http_get=reporting_lane._default_http_get)
    assert lane._call("hn", reporting_lane.HN_URL + "?query=x", (1, 1)) is None
    assert lane.last_why == "api_denied" and opened == [] and lane._fails.get("hn", 0) == 0 and lane.available("hn")


def test_acquisition_hop_denied_before_any_request_including_robots(caplog):
    _deny_epmc()
    web = A.Web({}, {"PMC1": b"<article/>"})
    deps = A.deps([], AP.gate_api_urls(web))
    for url, want_json in ((EPMC_SEARCH + "?query=x", True), (EPMC_JATS, False)):
        with pytest.raises(auto_acquire._Stop) as e:
            auto_acquire._guarded_get(url, A.cfg(), deps, 100000, 5.0, want_json, clock=A.clock_at())
        assert e.value.code == "R-API-DENIED"
    assert web.urls == []  # neither the API nor www.ebi.ac.uk/robots.txt was requested


def test_gated_transport_refuses_on_its_own():
    _deny_epmc()
    web = A.Web({}, {})
    with pytest.raises(AP.ApiDenied):
        AP.gate_api_urls(web)(EPMC_JATS, headers={}, timeout=1)
    assert web.urls == []
    AP.gate_api_urls(web)("https://www.ebi.ac.uk/robots.txt", headers={}, timeout=1)  # website content passes
    assert web.urls == ["https://www.ebi.ac.uk/robots.txt"]


def test_new_reason_codes_are_mapped_for_ui_and_outcome():
    from vera import discovery_panel, research_outcome
    for c in (AP.DENIED, AP.COND, AP.SCOPE):
        assert c in discovery_panel.REASON_TEXT and research_outcome._CODE_MAP[c] == "no_route"
        assert "europepmc" not in discovery_panel.REASON_TEXT[c].lower()  # no host in the public phrase


# ------------------------------------------------------------------------------------------------ permitted: conditions
def test_permitted_call_goes_through_its_rate_portal(monkeypatch):
    import requests
    fake = _Counter()
    monkeypatch.setattr(requests, "get", fake)
    before = rate_limit.get_registry().snapshot()["openalex"]["requests"]
    search_providers.openalex_default_transport("https://api.openalex.org/works", {"search": "x"}, {"User-Agent": "u"}, 1)
    assert len(fake.calls) == 1 and rate_limit.get_registry().snapshot()["openalex"]["requests"] == before + 1


def test_permitted_entry_sets_its_required_user_agent():
    fake = _Counter()
    AP.call(fake, "https://api.crossref.org/works/10.1/x", headers={"User-Agent": "other", "Accept": "a"}, timeout=1)
    (_url, kw), = fake.calls
    assert kw["headers"] == {"Accept": "a", "User-Agent": CROSSREF_UA}


def test_rate_portal_skip_sends_nothing():
    reg = rate_limit.get_registry()
    reg.portal("crossref").cooldown_until = reg.portal("crossref")._clock() + 999
    fake = _Counter()
    with pytest.raises(rate_limit.RateLimited):
        AP.call(fake, "https://api.crossref.org/works", headers={}, timeout=1)
    assert fake.calls == []


@pytest.mark.parametrize("edit,why", [
    (lambda e: e["openalex"]["conditions"].update(max_rate_per_s=0.5), "rate_above_ceiling"),
    (lambda e: e["crossref_rest"]["conditions"].update(max_concurrency=0), "concurrency_above_ceiling"),
    (lambda e: e["openalex"]["conditions"].update(rate_portal="no_such_portal"), "rate_portal_missing"),
    (lambda e: e["openalex"]["conditions"].update(rate_portal="crossref"), "rate_portal_host"),
    (lambda e: e["openalex"]["conditions"].pop("rate_portal"), "no_rate_portal"),
    (lambda e: e["openalex"]["conditions"].update(send_email=True), "email_required"),
])
def test_permitted_entry_with_unenforceable_conditions_is_denied(edit, why):
    pol = _policy(edit)
    url = "https://api.crossref.org/works" if "concurrency" in why else "https://api.openalex.org/works"
    d = AP.decide(url, "metadata", pol)
    assert (d.allowed, d.code, d.why) == (False, AP.COND, why)


def test_429_must_be_a_throttle_on_the_portal(monkeypatch):
    reg = rate_limit.get_registry()
    monkeypatch.setitem(reg.portal("openalex").c, "throttle_statuses", (503,))
    d = AP.decide("https://api.openalex.org/works", "metadata")
    assert (d.allowed, d.why) == (False, "429_not_throttle")


def test_content_scope_is_enforced():
    d = AP.decide("https://api.openalex.org/works", "fulltext")
    assert (d.allowed, d.code) == (False, AP.SCOPE)


# ------------------------------------------------------------------------------------------------ owner switch
def _deny_epmc():
    """The pre-OD-1108 state of Europe PMC (owner_switch deny), for tests of the denial machinery."""
    AP.set_policy_for_tests(_policy(lambda e: e["europepmc_rest"].update(owner_switch="deny")))


def _flip_epmc(entries, decision="OD-1107-TEST"):
    entries["europepmc_rest"].update(owner_switch="permit", owner_decision=decision)


def test_owner_switch_permits_europepmc_with_no_code_change(monkeypatch):
    pol = _policy(_flip_epmc)
    assert AP.decide(EPMC_SEARCH, "metadata", pol).allowed and AP.decide(EPMC_JATS, "fulltext", pol).allowed
    AP.set_policy_for_tests(pol)
    web = A.Web({}, {"PMC1": b"<article/>"})
    raw, _ = auto_acquire._guarded_get(EPMC_JATS, A.cfg(), A.deps([], AP.gate_api_urls(web)), 100000, 5.0, False,
                                       clock=A.clock_at())
    assert raw == b"<article/>" and A.doc_urls(web.urls) == [EPMC_JATS]  # robots/TDM gate still ran first


def test_owner_permit_without_decision_is_denied():
    pol = _policy(lambda e: _flip_epmc(e, decision=""))
    assert AP.decide(EPMC_SEARCH, "metadata", pol).why == "permit_without_decision"


def test_owner_deny_overrides_a_permitted_verdict():
    pol = _policy(lambda e: e["openalex"].update(owner_switch="deny"))
    d = AP.decide("https://api.openalex.org/works", "metadata", pol)
    assert (d.allowed, d.code, d.why) == (False, AP.DENIED, "owner_deny")


def test_review_due_is_a_note_not_a_denial():
    assert "europepmc_rest" in AP.review_due("2026-12-08")
    assert AP.decide("https://api.openalex.org/works", "metadata").allowed  # dates never deny


@pytest.mark.parametrize("url", ["https://www.ebi.ac.uk/EuropePMC/WebServices/REST/search?q=x",
                                 "https://www.ebi.ac.uk/europepmc//webservices/rest/PMC1/fullTextXML",
                                 "https://WWW.EBI.AC.UK/europepmc/webservices/rest"])
def test_case_and_slash_variants_still_match_the_denied_entry(url):
    _deny_epmc()
    assert AP.is_api_url(url) and AP.decide(url).entry == "europepmc_rest" and not AP.decide(url).allowed


# ------------------------------------------------------------------------------------------------ r2 (review REVISE)
class _Redirecting:
    """Fake requests.get: answers each URL from a map; records (url, allow_redirects, headers)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, *a, **kw):
        self.calls.append((url, kw.get("allow_redirects"), dict(kw.get("headers") or {})))
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return A.Resp(status=404)


def _redir(to):
    return A.Resp(status=302, body=b"", headers={"Location": to})


@pytest.mark.parametrize("target", ["https://export.arxiv.org/api/query?x", EPMC_SEARCH, "https://example.org/page",
                                    "https://arxiv.org/%61pi/query"])
def test_redirect_from_a_permitted_api_to_a_denied_target_is_refused(target):
    _deny_epmc()
    fake = _Redirecting({"api.openalex.org": _redir(target)})
    with pytest.raises(AP.ApiDenied):
        AP.call(fake, "https://api.openalex.org/works", params={"search": "x"}, headers={}, timeout=1)
    assert [c[0] for c in fake.calls] == ["https://api.openalex.org/works"]  # nothing sent to the target
    assert fake.calls[0][1] is False  # fn never follows redirects itself


def test_redirect_between_permitted_apis_is_followed_and_each_hop_regated():
    fake = _Redirecting({"api.openalex.org": _redir("https://api.crossref.org/works/10.1/x"),
                         "api.crossref.org": A.Resp(body=b"{}")})
    r = AP.call(fake, "https://api.openalex.org/works", params={"search": "x"}, headers={"User-Agent": "u"}, timeout=1)
    assert r.status_code == 200 and [c[0] for c in fake.calls] == ["https://api.openalex.org/works",
                                                                   "https://api.crossref.org/works/10.1/x"]
    assert all(c[1] is False for c in fake.calls)
    assert fake.calls[0][2]["User-Agent"] == "u" and fake.calls[1][2]["User-Agent"] == CROSSREF_UA  # per-hop entry


def test_redirect_loop_is_bounded():
    fake = _Redirecting({"api.openalex.org": _redir("https://api.openalex.org/works?again")})
    with pytest.raises(AP.ApiDenied) as e:
        AP.call(fake, "https://api.openalex.org/works", headers={}, timeout=1)
    assert e.value.why == "too_many_redirects" and len(fake.calls) == AP.MAX_API_REDIRECTS + 1


def test_caller_allow_redirects_false_gets_the_3xx_back_unfollowed():
    fake = _Redirecting({"api.openalex.org": _redir("https://export.arxiv.org/api/query")})
    r = AP.call(fake, "https://api.openalex.org/works", headers={}, timeout=1, allow_redirects=False)
    assert r.status_code == 302 and len(fake.calls) == 1


@pytest.mark.parametrize("tx,url", [
    ("openalex_default_transport", "https://api.openalex.org/works"),
    ("arxiv_default_transport", "https://export.arxiv.org/api/query"),
    ("doaj_default_transport", "https://doaj.org/api/v4/search/articles/x"),
    ("europepmc_default_transport", EPMC_SEARCH)])
def test_every_requests_get_transport_sends_allow_redirects_false(monkeypatch, tx, url):
    import requests
    fake = _Redirecting({"api.openalex.org": _redir("https://export.arxiv.org/api/query")})
    monkeypatch.setattr(requests, "get", fake)
    pol = _policy(lambda e: [e[k].update(owner_switch="permit", owner_decision="OD-1107-TEST")
                             for k in ("arxiv_export_api", "doaj_api", "europepmc_rest")])
    AP.set_policy_for_tests(pol)
    if tx == "openalex_default_transport":  # a permitted API redirecting to an arXiv API the owner holds
        AP.set_policy_for_tests(_policy())
        with pytest.raises(AP.ApiDenied):
            getattr(search_providers, tx)(url, {"q": "x"}, {}, 1)
    else:
        getattr(search_providers, tx)(url, {"q": "x"}, {}, 1)
    assert fake.calls and all(c[1] is False for c in fake.calls) and len(fake.calls) == 1


def test_source_resolver_transport_refuses_a_redirect_to_a_denied_api(monkeypatch):
    fake = _Redirecting({"api.openalex.org": _redir("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/x")})
    monkeypatch.setattr(source_resolver.requests, "get", fake)
    assert source_resolver.lookup_title("one two three four five", 1.0, 0.5) is None  # a miss, not an error
    assert [c[0] for c in fake.calls] == [source_resolver.OPENALEX_SEARCH] and fake.calls[0][1] is False


@pytest.mark.parametrize("decision,ok", [("OD-1107-HITL", True), ("OD-1107-TEST", True), ("owner said so", False),
                                         ("OD-11-07", False), ("od-1107-hitl", False), ("", False), (None, False)])
def test_owner_permit_needs_a_decision_id_pattern(decision, ok):
    pol = _policy(lambda e: e["europepmc_rest"].update(owner_switch="permit", owner_decision=decision))
    assert AP.decide(EPMC_SEARCH, "metadata", pol).allowed is ok


def test_gated_hop_transport_enforces_scope_for_the_callers_purpose():
    pol = _policy(lambda e: (_flip_epmc(e), e["europepmc_rest"]["conditions"].update(content_scope=["metadata"])))
    AP.set_policy_for_tests(pol)
    web = A.Web({}, {"PMC1": b"<article/>"})
    g = AP.gate_api_urls(web)
    g(EPMC_SEARCH + "?q=x", headers={}, timeout=1, api_purpose="metadata")
    with pytest.raises(AP.ApiDenied) as e:
        g(EPMC_JATS, headers={}, timeout=1, api_purpose="fulltext")
    assert e.value.code == AP.SCOPE
    with pytest.raises(AP.ApiDenied):  # no purpose named: the stricter "fulltext"
        g(EPMC_SEARCH + "?q=y", headers={}, timeout=1)
    assert web.urls == [EPMC_SEARCH + "?q=x"]
    with pytest.raises(auto_acquire._Stop) as s:  # the acquisition hop passes its purpose (JATS = fulltext)
        auto_acquire._guarded_get(EPMC_JATS, A.cfg(), A.deps([], g), 100000, 5.0, False, clock=A.clock_at())
    assert s.value.code == AP.SCOPE


@pytest.mark.parametrize("url", ["https://arxiv.org/%61pi/query", "https://arxiv.org/x/../api/query",
                                 "https://arxiv.org/./api/query", "https://arxiv.org/%2561pi/query",
                                 "https://arxiv.org/abs/../../api/query", "https://arxiv.org/API;v=1/query",
                                 "https://www.ebi.ac.uk/europepmc/x/../webservices/%72est/search"])
def test_percent_encoding_and_dot_segments_still_match_the_denied_entry(url):
    _deny_epmc()
    d = AP.decide(url)
    assert AP.is_api_url(url) and not d.allowed and d.entry in ("arxiv_web_api", "europepmc_rest")


def test_normalised_paths_do_not_overmatch():
    assert AP.normalise_path("/apix/q") == "/apix/q/" and not AP.is_api_url("https://arxiv.org/apix/q")
    assert not AP.is_api_url("https://arxiv.org/abs/2401.00001") and AP.normalise_path("/../..") == "/"


# ------------------------------------------------------------------------------------------------ r3
@pytest.mark.parametrize("url,entry,allowed", [
    ("https://api.openalex.org./works", "openalex", True),
    ("https://arxiv.org./api/query", "arxiv_web_api", False),
    ("https://www.ebi.ac.uk./europepmc/webservices/rest/search", "europepmc_rest", True),
    ("https://export.arxiv.org../api/query", "arxiv_export_api", False)])
def test_trailing_dot_hosts_match_the_dotless_entry(url, entry, allowed):
    d = AP.decide(url, "metadata")
    assert AP.is_api_url(url) and d.entry == entry and d.allowed is allowed


def test_trailing_dot_redirect_target_is_regated():
    fake = _Redirecting({"api.openalex.org": _redir("https://arxiv.org./api/query")})
    with pytest.raises(AP.ApiDenied):
        AP.call(fake, "https://api.openalex.org/works", headers={}, timeout=1)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("decision", ["OD-1107-x garbage", "OD-1107-HITL extra", "OD-1107-", "OD-1107-hitl",
                                      "OD-1107-HITL\n", " OD-1107-HITL"])
def test_owner_decision_must_fully_match(decision):
    pol = _policy(lambda e: e["europepmc_rest"].update(owner_switch="permit", owner_decision=decision))
    assert AP.decide(EPMC_SEARCH, "metadata", pol).why == "permit_without_decision"


# ------------------------------------------------------------------------------------------------ API-ALLOW-R3-1108
# OD-1108-HITL: per-host table. Permitted: OpenAlex, Crossref. Conditional: Europe PMC (recorded conditions), GDELT
# (metadata only, 0.2 req/s). Denied: Semantic Scholar, HN Algolia, DOAJ, Unpaywall, arXiv export API and arxiv.org/api.
S2 = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/x"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc?query=x"
HN = "https://hn.algolia.com/api/v1/search?query=x"


def test_r3_host_table_one_row_per_host():
    rows = {"https://api.openalex.org/works": ("openalex", True), "https://api.crossref.org/works/10.1/x": ("crossref_rest", True),
            EPMC_SEARCH + "?query=x": ("europepmc_rest", True), GDELT: ("gdelt_doc", True),
            S2: ("semantic_scholar", False), HN: ("hn_algolia", False),
            "https://doaj.org/api/v4/search/articles/x": ("doaj_api", False),
            "https://api.unpaywall.org/v2/10.1/x": ("unpaywall", False),
            "https://export.arxiv.org/api/query?x": ("arxiv_export_api", False), "https://arxiv.org/api/query": ("arxiv_web_api", False)}
    for url, (entry, ok) in rows.items():
        d = AP.decide(url, "metadata")
        assert (d.entry, d.allowed) == (entry, ok), (url, d)


@pytest.mark.parametrize("url", [S2, HN, "https://api.unpaywall.org/v2/10.1/x", "https://doaj.org/api/v4/x",
                                 "https://export.arxiv.org/api/query?x"])
def test_r3_denied_hosts_send_nothing(url):
    fake = _Counter()
    with pytest.raises(AP.ApiDenied) as e:
        AP.call(fake, url, headers={}, timeout=1)
    assert e.value.code == AP.DENIED and fake.calls == []


def test_r3_owner_cannot_be_bypassed_for_s2_and_hn_without_a_decision():
    for eid, url in (("semantic_scholar", S2), ("hn_algolia", HN)):
        raw = _raw()["entries"][eid]
        assert raw["owner_switch"] == "deny" and raw["verdict"] == "UNRESOLVED" and "OD-1108-HITL" in raw["owner_decision"]
        assert not AP.decide(url, "metadata").allowed


def test_r3_europepmc_conditions_are_the_recorded_ones():
    e = _raw()["entries"]["europepmc_rest"]
    c = e["conditions"]
    assert e["owner_switch"] == "permit" and e["owner_decision"] == "OD-1108-HITL"
    assert c["content_scope"] == ["metadata", "fulltext"] and c["crawl_delay_s"] == 10
    assert c["max_rate_per_s"] == 0.1 and c["max_concurrency"] == 1 and c["respect_429"] is True
    assert "contact" not in c.get("user_agent", "") and c["user_agent"].startswith("VERA-DAR/") and "https://" in c["user_agent"]
    assert c["licence_gate"] and c["attribution"]
    d = AP.decide(EPMC_JATS, "fulltext")
    assert d.allowed and d.entry == "europepmc_rest"
    assert AP.decide(EPMC_SEARCH, "metadata").allowed
    assert AP.decide(EPMC_JATS, "tdm").code == AP.SCOPE  # outside the recorded scope


def test_r3_europepmc_rate_is_capped_by_crawl_delay_in_rate_limits():
    from vera import rate_limit
    pc = rate_limit.get_registry().portal("europepmc").c
    assert float(pc["rate_per_s"]) <= 0.1 and int(pc["max_concurrency"]) == 1 and int(pc["burst"]) == 1


@pytest.mark.parametrize("field,value,why", [("rate_per_s", 5.0, "rate_above_ceiling"), ("max_concurrency", 4, "concurrency_above_ceiling")])
def test_r3_europepmc_is_conditional_on_the_rate_limits(monkeypatch, field, value, why):
    from vera import rate_limit
    monkeypatch.setitem(rate_limit.get_registry().portal("europepmc").c, field, value)
    d = AP.decide(EPMC_SEARCH, "metadata")
    assert (d.allowed, d.code, d.why) == (False, AP.COND, why)  # R-API-COND: nothing is sent


def test_r3_europepmc_429_must_stay_a_throttle(monkeypatch):
    from vera import rate_limit
    monkeypatch.setitem(rate_limit.get_registry().portal("europepmc").c, "throttle_statuses", (503,))
    assert AP.decide(EPMC_SEARCH, "metadata").why == "429_not_throttle"


def test_r3_europepmc_request_carries_the_contact_user_agent():
    fake = _Counter()
    AP.call(fake, EPMC_SEARCH + "?query=x", headers={"User-Agent": "other"}, timeout=1)
    assert fake.calls and fake.calls[0][1]["headers"]["User-Agent"].startswith("VERA-DAR/1.0 (+https://")


def test_r3_europepmc_scope_gate_in_the_hop_transport():
    web = A.Web({}, {"PMC1": b"<article/>"})
    g = AP.gate_api_urls(web)
    g(EPMC_JATS, headers={}, timeout=1, api_purpose="fulltext")
    g(EPMC_SEARCH + "?q=x", headers={}, timeout=1, api_purpose="metadata")
    assert len(web.urls) == 2


def test_r3_gdelt_is_metadata_only_at_its_recorded_rate():
    e = _raw()["entries"]["gdelt_doc"]
    c = e["conditions"]
    assert c["content_scope"] == ["metadata"] and c["max_rate_per_s"] == 0.2 and c["max_concurrency"] == 1 and c["respect_429"]
    assert "OD-1108-HITL" in e["owner_decision"]
    assert AP.decide(GDELT, "metadata").allowed
    assert AP.decide(GDELT, "fulltext").code == AP.SCOPE
    assert not AP.decide("https://api.gdeltproject.org/other", "metadata").allowed  # outside /api/v2/doc/ (no entry)


def test_r3_gdelt_condition_fails_if_the_rate_limit_is_raised(monkeypatch):
    from vera import rate_limit
    monkeypatch.setitem(rate_limit.get_registry().portal("gdelt_doc").c, "rate_per_s", 1.0)
    d = AP.decide(GDELT, "metadata")
    assert (d.allowed, d.code, d.why) == (False, AP.COND, "rate_above_ceiling")


def test_r3_permitted_hosts_are_metadata_only():
    for u in ("https://api.openalex.org/works", "https://api.crossref.org/works/10.1/x"):
        assert AP.decide(u, "metadata").allowed and AP.decide(u, "fulltext").code == AP.SCOPE


def test_r3_arxiv_api_search_stays_off_even_with_a_permit_missing_decision():
    for u in ("https://export.arxiv.org/api/query?x", "https://arxiv.org/api/query"):
        assert AP.decide(u, "metadata").why == "owner_deny"
