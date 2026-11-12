"""ALLDOMAINS-WPG-IMPL-1111 (WP-G): the generic ORIGINAL_HTML route and its USFED parity tests (work/USFED-ROUTE-1111
section 7.1, run against every enabled row of config/permitted_originals.json). Offline: fake HTTP only (the repo's kernel
guard forbids sockets), no DB. Page fixtures are synthetic reconstructions of the page STRUCTURE read in USFED-C0-1111
(banner, header/nav, <main> with series blurb and a related-links block, footer); the saved real C0 pages are used only in
the optional measurement test at the end (skipped when scratch/USFED-C0-1111 is absent), never copied into the repo."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from tests import acq_support as A
from vera import (acq_config, acq_discovered, auto_acquire, discovery_panel, fulltext_loader, fulltext_routes,
                  gov_public_domain, html_main, jats_text, original_html, research_outcome)

SHIPPED = json.loads((Path(original_html.CONFIG_PATH)).read_text(encoding="utf-8"))
DOE = "https://www.energy.gov/science/doe-explainsquarks-and-gluons"
DOE_HOST = "www.energy.gov"
NASA = "https://science.nasa.gov/dark-matter"
Q = "What are quarks and gluons?"
ROBOTS_OK = "User-agent: *\nAllow: /\n"
PARA = ("Quarks are elementary particles that combine to form protons and neutrons, and gluons are the force carrying "
        "particles that bind quarks together through the strong force inside every atomic nucleus. ")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    original_html.reset_for_tests()
    acq_discovered.reset_for_tests()
    yield
    original_html.reset_for_tests()


def set_mode(monkeypatch, m="shadow", **limits):
    c = copy.deepcopy(SHIPPED)
    c["mode"] = m
    c["limits"].update(limits)
    monkeypatch.setattr(original_html, "load_config", lambda path=None: c)
    return c


# ------------------------------------------------------------------------------------------------ fixtures
def page(body_paras=None, *, title="DOE Explains...Quarks and Gluons", footer="Office of Science", head_extra="",
         related=True, header_byline="", extra_main="", empty_main=False, blurb=True):
    paras = body_paras if body_paras is not None else [PARA * 2 for _ in range(4)]
    secs = "".join(f"<h2>Section {i}</h2><p>{p}</p>" for i, p in enumerate(paras, 1))
    blurb_h = ('<div class="series-blurb"><p>DOE Explains is a series of short articles from the Office of Science '
               "about the concepts behind its research.</p></div>") if blurb else ""
    rel = ('<div class="related-resources"><h2>Resources and Related Terms</h2><ul><li><a href="/a">Higgs boson</a></li>'
           '<li><a href="/b">Standard Model</a></li></ul></div>') if related else ""
    main = "<main></main>" if empty_main else f"<main><h1>{title}</h1>{blurb_h}{secs}{extra_main}{rel}</main>"
    return (f'<!doctype html><html><head><title>{title} | Department of Energy</title>{head_extra}</head><body>'
            f'<div class="usa-banner">An official website of the United States government</div>'
            f'<header><nav><a href="/">Home</a><a href="/science">Office of Science</a></nav>{header_byline}</header>'
            f"{main}<footer><p>{footer}</p><a href=\"/privacy\">Privacy</a></footer></body></html>").encode()


class Web:
    """Fake http_get recording every URL: robots, tdmrep (404) and pages by URL; redirects via Resp headers."""

    def __init__(self, pages=None, robots=ROBOTS_OK, robots_status=200, over=None):
        self.pages, self.robots, self.rs, self.over, self.urls = pages or {}, robots, robots_status, over or {}, []

    def __call__(self, url, headers=None, timeout=None, allow_redirects=None, stream=None, **kw):
        self.urls.append(url)
        if url.endswith("/robots.txt"):
            return A.Resp(status=self.rs, body=self.robots.encode(), ctype="text/plain")
        if url.endswith("/.well-known/tdmrep.json"):
            return A.Resp(status=404)
        if url in self.over:
            v = self.over[url]
            return v() if callable(v) else v
        if url in self.pages:
            return A.Resp(status=200, body=self.pages[url], ctype="text/html; charset=utf-8")
        return A.Resp(status=404)

    def docs(self):
        return A.doc_urls(self.urls)


def deps(web, lookup=None):
    return A.deps([], web, lookup=lookup)


def cand(url=DOE, **kw):
    return {"url": url, "title": "DOE Explains...Quarks and Gluons", "source": "permitted_catalogue", "original_html": True, **kw}


def run(web, c=None, cfg=None, job=None, min_spans=None):
    cfg = cfg or A.cfg()
    icfg = acq_config.ingest_cfg(fulltext_loader.load_config(), cfg)
    return auto_acquire.evaluate_candidate(job or A.job(question=Q), 1.0, "", c or cand(), None, cfg, deps(web), icfg,
                                           auto_acquire._fd_cfg(), clock=A.clock_at(), min_spans=min_spans)


def code(web, c=None, cfg=None, job=None):
    with pytest.raises(auto_acquire._Stop) as e:
        run(web, c, cfg, job)
    return e.value.code


ROWS = [r for r in SHIPPED["rows"]]


def row_base(r):
    """The row's first stem or prefix, as a path."""
    return (r.get("path_stems") or r["path_prefixes"])[0].rstrip("/")


def row_url(r):
    """A real in-row page URL: stem + slug in one segment, or prefix + one more segment."""
    return f"https://{r['host']}{row_base(r)}" + ("-x" if r.get("path_stems") else "/x")


# ================================================================================ step 2: route claim (per row)
@pytest.mark.parametrize("r", ROWS, ids=lambda r: r["id"])
def test_claim_positive_and_negative_per_row(monkeypatch, r):
    set_mode(monkeypatch, "shadow")
    h, p, ok = r["host"], row_base(r), row_url(r)
    assert original_html.claims_url(ok)
    assert original_html.claims_url(ok + "/")  # trailing slash: one canonical form
    tail = ok[len(f"https://{h}"):]
    for bad in (f"https://{h}/other/path",                       # same host, other path
                f"https://{h}.example.com{tail}", f"https://x{h}{tail}",  # lookalike hosts
                f"http://{h}{tail}",                             # http
                f"https://{h}:8443{tail}",                       # port
                f"https://user@{h}{tail}", f"https://user:pw@{h}{tail}",  # userinfo
                f"https://{h}{tail}?q=1", f"https://{h}{tail}#frag",      # query, fragment
                f"https://{h}{p}/../../user/login",              # dot-segments leave the prefix
                f"https://{h}{p}/%2e%2e/%2e%2e/user/login",      # %2e%2e segments
                f"https://{h}{p}/..;/..;/x", f"https://{h}{tail};jsessionid=1",  # D5: path parameters
                f"https://{h}{p}%2fx", f"https://{h}{p}\\x", f"https://{h}//science",
                f"https://{h}{p}-evil/x", f"https://{h}{p}X/y",  # D1: a longer first segment with more path
                "https://fnal.gov/science/doe-explains-x", "https://www.fnal.gov" + tail):
        assert not original_html.claims_url(bad), bad


def test_d1_whole_segment_boundaries_stems_and_prefixes(monkeypatch):
    set_mode(monkeypatch, "shadow")
    c = original_html.claims_url
    # DOE row (stem): the real page shape is stem + slug in ONE segment; the bare stem (the index) and deeper paths are not
    assert c("https://www.energy.gov/science/doe-explainsneutrinos")
    assert c("https://www.energy.gov/science/doe-explainsthe-higgs-boson")
    assert not c("https://www.energy.gov/science/doe-explains")                       # the index page is links only
    assert not c("https://www.energy.gov/science/doe-explains-evil/x")                # deeper than one segment
    assert not c("https://www.energy.gov/science/doe-explains/neutrinos")
    # NASA row (prefix): the prefix itself and anything under prefix + "/"; never a longer first segment
    assert c("https://science.nasa.gov/dark-matter") and c("https://science.nasa.gov/dark-matter/x")
    assert c("https://science.nasa.gov/dark-energy") and c("https://science.nasa.gov/universe/black-holes")
    for bad in ("https://science.nasa.gov/dark-matterX", "https://science.nasa.gov/dark-matter-evil/x",
                "https://science.nasa.gov/dark-energyX", "https://science.nasa.gov/universeX/y",
                "https://science.nasa.gov/universe-x"):
        assert not c(bad), bad
    row = {"path_prefixes": ["/a/"], "path_stems": []}
    assert original_html.path_ok(row, "/a") and original_html.path_ok(row, "/a/b") and not original_html.path_ok(row, "/ab")
    # the per-hop checker (redirect targets) uses the same boundary
    chk = original_html.path_checker(ROWS[1])
    assert chk("https://science.nasa.gov/dark-matter/x") and not chk("https://science.nasa.gov/dark-matter-evil/x")


def test_d5_rows_need_a_dns_name_host(monkeypatch):
    c = set_mode(monkeypatch, "shadow")
    for bad in ("127.0.0.1", "10.0.0.5", "8.8.8.8", "0x7f.1", "1.2.3.4.5", "localhost", "[::1]"):
        c["rows"] = [{**copy.deepcopy(ROWS[1]), "id": "ip", "host": bad}]
        assert original_html.rows(c) == [], bad
    c["rows"] = [{**copy.deepcopy(ROWS[1]), "id": "ok", "host": "www.example.gov"}]
    assert len(original_html.rows(c)) == 1
    c["rows"] = [{**copy.deepcopy(ROWS[1]), "id": "semi", "path_prefixes": ["/a;b"]}]
    assert original_html.rows(c) == []
    c["rows"] = [{**copy.deepcopy(ROWS[1]), "id": "empty", "path_prefixes": [], "path_stems": []}]
    assert original_html.rows(c) == []


def test_normalisation_is_one_canonical_form():
    n = original_html.normalise_url
    assert n("https://WWW.Energy.GOV/science/doe-explainsquarks-and-gluons/") == DOE
    assert n("https://www.energy.gov/science/%64oe-explainsquarks-and-gluons") == DOE  # unreserved decoded
    assert n("https://www.energy.gov/a/b/../c") == "https://www.energy.gov/a/c"
    assert n("https://www.energy.gov/../x") is None  # climbs above the root
    assert n("https://www.energy.gov/a%20b") == "https://www.energy.gov/a%20b"  # reserved stays encoded
    assert n("https://www.energy.gov/a%zz") is None


def test_mode_off_claims_nothing_and_makes_no_request(monkeypatch):
    set_mode(monkeypatch, "off")
    assert not original_html.claims_url(DOE)
    web = Web({DOE: page()})
    assert fulltext_routes.select(cand(), None, A.cfg())[0].name == "europepmc"  # falls to the default route
    with pytest.raises(auto_acquire._Stop):
        original_html.evaluate(A.job(), 1.0, None, cand(), None, A.cfg(), deps(web), {}, {})  # defensive refusal
    assert web.urls == []  # the route itself never requests anything while off


def test_shipped_config_is_off_with_wave1_rows_only():
    assert SHIPPED["mode"] == "off"
    assert all(r["licence_basis"] == "us_federal_pd" for r in ROWS)
    assert {r["host"] for r in ROWS if r["store"]} == {DOE_HOST}  # storage is DOE-only (NASA waits for the contractor fix)
    assert all(re.fullmatch(r"[a-z0-9.-]+", r["host"]) and not r["host"].startswith("fnal") for r in ROWS)


def test_unsupported_basis_or_malformed_row_is_ignored(monkeypatch):
    c = set_mode(monkeypatch, "shadow")
    c["rows"].append({**copy.deepcopy(ROWS[0]), "id": "x", "host": "www.loc.gov", "licence_basis": "page_declared"})
    c["rows"].append({**copy.deepcopy(ROWS[0]), "id": "y", "host": "www.nps.gov", "review": {}})
    c["rows"].append({**copy.deepcopy(ROWS[0]), "id": "z", "host": "WWW.BLS.GOV"})
    assert not original_html.claims_url("https://www.loc.gov/science/doe-explains-x")
    assert not original_html.claims_url("https://www.nps.gov/science/doe-explains-x")
    assert not original_html.claims_url("https://www.bls.gov/science/doe-explains-x")


def test_route_registry_order_and_default(monkeypatch):
    set_mode(monkeypatch, "shadow")
    assert [r.name for r in fulltext_routes.ROUTES] == ["arxiv", "original_html", "europepmc"]
    assert fulltext_routes.select(cand(), None, A.cfg())[0].name == "original_html"
    assert fulltext_routes.select(A.cand("10.1234/abc"), "10.1234/abc", A.cfg())[0].name == "europepmc"
    # a DOI candidate at a row URL stays on the DOI route (the route claims only identifier-less URLs)
    assert fulltext_routes.select({**cand(), "doi": "10.1234/abc"}, "10.1234/abc", A.cfg())[0].name == "europepmc"
    assert fulltext_routes.select({"url": "https://www.energy.gov/news/x"}, None, A.cfg())[0].name == "europepmc"


def test_europepmc_allowlist_is_not_widened():
    assert acq_config.load_config()["fetch"]["allowed_hosts"] == ["www.ebi.ac.uk"]


def test_rank_candidates_lets_a_row_url_through_only_when_the_mode_is_not_off(monkeypatch):
    cfg, drops = A.cfg(), {}
    kw = dict(served=set(), fdcfg=auto_acquire._fd_cfg(), count=lambda c: None)
    set_mode(monkeypatch, "off")
    assert auto_acquire.rank_candidates(Q, [cand()], cfg, drops=drops, **kw) == []
    set_mode(monkeypatch, "shadow")
    out = auto_acquire.rank_candidates(Q, [cand(), {"url": "https://www.energy.gov/news/x", "title": "x"}], cfg, **kw)
    assert [c["url"] for _s, _d, c, _i in out] == [DOE]  # the off-prefix URL is still dropped as NODOI


# ================================================================================ steps 3, 3a: gates and robots
def test_gates_refuse_with_existing_codes(monkeypatch):
    set_mode(monkeypatch, "shadow")
    cfg = A.cfg()
    rc = lambda k: acq_discovered.refusal_code(cfg, k)  # noqa: E731
    assert code(Web({DOE: page()}, robots="User-agent: *\nDisallow: /science/\n")) == rc("robots")
    acq_discovered.reset_for_tests()  # the robots verdict is cached per host
    assert code(Web({DOE: page()}, robots="", robots_status=500)).startswith(rc("robots_unreadable"))
    acq_discovered.reset_for_tests()
    web = Web({DOE: page()}, over={DOE: A.Resp(200, page(), "text/html", {"TDM-Reservation": "1"})})
    assert code(web) == rc("tdm")
    web = Web({DOE: page()}, over={DOE: A.Resp(200, page(), "text/html", {"X-Robots-Tag": "noai"})})
    assert code(web).startswith("R-ACQ-")
    # off-host redirect and a PORTAL host hop
    for target in ("https://evil.example.org/x", "https://www.science.gov/x"):
        acq_discovered.reset_for_tests()
        web = Web({}, over={DOE: A.Resp(302, b"", "text/html", {"Location": target})})
        assert code(web) == rc("host") and [u for u in web.docs() if "evil" in u or "science.gov" in u] == []


@pytest.mark.parametrize("status", [401, 403, 451, 418])
def test_robots_4xx_other_than_404_410_fails_closed(monkeypatch, status):
    set_mode(monkeypatch, "shadow")
    web = Web({DOE: page()}, robots="", robots_status=status)
    c = code(web)
    assert c.startswith(acq_discovered.refusal_code(A.cfg(), "robots_unreadable")) and web.docs() == []


@pytest.mark.parametrize("status", [404, 410])
def test_robots_404_410_is_unrestricted_but_page_signals_still_prevail(monkeypatch, status):
    set_mode(monkeypatch, "shadow")
    web = Web({DOE: page()}, robots="", robots_status=status)
    c = code(web)
    assert c == acq_config.reason(A.cfg(), "original_shadow") or c.startswith("R-ACQ-ORIGINAL")  # fetched and processed
    assert DOE in web.docs()
    web = Web({}, robots="", robots_status=status, over={DOE: A.Resp(200, page(), "text/html", {"TDM-Reservation": "1"})})
    assert code(web) == acq_discovered.refusal_code(A.cfg(), "tdm")


def test_robots_gate_is_the_shared_wpl_gate_on_both_entrypoints():
    # parity: the route does not carry its own robots code; both shared gates fail closed on 403 (WP-L)
    import inspect
    assert "robots" not in inspect.getsource(original_html.evaluate).lower().replace("robots_unreadable", "")


# ================================================================================ step 3b: access denial, no bypass
def test_document_403_is_access_denied_once_with_no_variant_and_no_handoff(monkeypatch):
    set_mode(monkeypatch, "shadow")
    web = Web({}, over={DOE: A.Resp(403, b"", "text/html")})
    c1 = code(web)
    plain = acq_discovered.refusal_code(A.cfg(), "access_denied")
    assert c1 == plain  # 3b-i: route_hop=False -> the plain code, not ACCESS_DENIED-ROUTE
    assert web.docs() == [DOE]  # exactly one request, none to http/apex/slash/query variants
    assert acq_discovered.is_denied(urls=(DOE,))
    n = len(web.urls)
    assert code(web) == plain and len(web.urls) == n  # 3b-ii: the second attempt makes no request at all
    # variants of the same canonical page are the same identity (3b-iv)
    assert code(web, cand(DOE + "/")) == plain and len(web.urls) == n
    assert research_outcome.reason_from_code(plain) == "no_route"


def test_denial_trips_the_per_host_breaker(monkeypatch):
    set_mode(monkeypatch, "shadow")
    other = "https://www.energy.gov/science/doe-explainsneutrinos"
    web = Web({other: page()}, over={DOE: A.Resp(401, b"", "text/html")})
    code(web)
    assert original_html.host_tripped(DOE_HOST)
    n = len(web.urls)
    assert code(web, cand(other)) == acq_discovered.refusal_code(A.cfg(), "access_denied") and len(web.urls) == n


def test_snapshot_denied_list_blocks_before_any_request(monkeypatch):
    set_mode(monkeypatch, "shadow")
    snap = {"version": 1, "harvested_at": "t", "script_version": "t", "robots_status": {}, "entries": [],
            "denied": [{"url": DOE, "code": "R-ACQ-ACCESS_DENIED", "seen_at": "t"}], "boilerplate": {}}
    snap["sha256"] = original_html.snapshot_digest(snap)
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: snap)
    web = Web({DOE: page()})
    assert code(web) == acq_discovered.refusal_code(A.cfg(), "access_denied") and web.urls == []


NEUTRINOS = "https://www.energy.gov/science/doe-explainsneutrinos"


def test_d2_denial_seeded_from_outside_makes_zero_requests_and_does_not_trip_the_breaker(monkeypatch):
    # the shared WP-L memory (_DENIED), written by another lane/worker path, must stop the route before ANY request
    # (robots included) through the URL-level check, with the host breaker NOT involved
    set_mode(monkeypatch, "shadow")
    acq_discovered.deny(acq_discovered.denial_ids(urls=(NEUTRINOS,)))
    web = Web({NEUTRINOS: page()})
    assert code(web, cand(NEUTRINOS)) == acq_discovered.refusal_code(A.cfg(), "access_denied")
    assert web.urls == [] and not original_html.host_tripped(DOE_HOST)
    # an un-denied sibling on the same host is still fetched (the breaker was not the reason)
    assert code(Web({DOE: page()}), cand(DOE)) == acq_config.reason(A.cfg(), "original_shadow")


def test_d2_redirect_to_a_denied_url_is_refused_before_it_is_requested(monkeypatch):
    set_mode(monkeypatch, "shadow")
    acq_discovered.deny(acq_discovered.denial_ids(urls=(NEUTRINOS,)))
    web = Web({NEUTRINOS: page()}, over={DOE: A.Resp(301, b"", "text/html", {"Location": NEUTRINOS})})
    assert code(web) == acq_discovered.refusal_code(A.cfg(), "access_denied")
    assert NEUTRINOS not in web.urls and web.docs() == [DOE]


def test_429_stays_a_pacing_code_not_a_denial(monkeypatch):
    set_mode(monkeypatch, "shadow")
    web = Web({}, over={DOE: A.Resp(429, b"", "text/html")})
    assert acq_config.is_rate_limited(A.cfg(), code(web)) and not original_html.host_tripped(DOE_HOST)


# ================================================================================ step 3c, 3d: redirects and URLs
def test_redirect_checks_host_and_prefix_https_and_hop_cap(monkeypatch):
    set_mode(monkeypatch, "shadow")
    host_code = acq_config.reason(A.cfg(), "host")
    for target in ("https://www.energy.gov/science/other",                 # same host, outside the prefix
                   "https://www.energy.gov/science/doe-explains/../../user/login",
                   "http://www.energy.gov/science/doe-explainsneutrinos",  # downgrade
                   "https://www.energy.gov:8443/science/doe-explainsneutrinos"):
        web = Web({}, over={DOE: A.Resp(301, b"", "text/html", {"Location": target})})
        assert code(web) == host_code, target
        assert all("user/login" not in u and u == DOE or u.endswith(("robots.txt", "tdmrep.json")) for u in web.urls)
    hops = {f"{DOE}{i}": A.Resp(302, b"", "text/html", {"Location": f"{DOE}{i + 1}"}) for i in range(0, 12)}
    hops[DOE] = A.Resp(302, b"", "text/html", {"Location": DOE + "0"})
    assert code(Web({}, over=hops)) == acq_config.reason(A.cfg(), "unreachable")


def test_same_row_redirect_inside_the_prefix_is_followed(monkeypatch):
    set_mode(monkeypatch, "shadow")
    tgt = "https://www.energy.gov/science/doe-explainsquarks"
    web = Web({tgt: page()}, over={DOE: A.Resp(301, b"", "text/html", {"Location": tgt})})
    assert code(web) in (acq_config.reason(A.cfg(), "original_shadow"), acq_config.reason(A.cfg(), "irrelevant"),
                         acq_config.reason(A.cfg(), "few_spans"))
    assert tgt in web.docs()


def test_index_and_sitemap_urls_are_untrusted(monkeypatch):
    set_mode(monkeypatch, "shadow")
    row = ROWS[0]
    html = (b'<a href="/science/doe-explainsquarks">q</a><a href="https://evil.example.org/science/doe-explains-x">e</a>'
            b'<a href="/science/doe-explains/../../user/login">t</a><a href="/science/doe-explainsx?q=1">q</a>'
            b'<a href="http://www.energy.gov/science/doe-explainsy">h</a><a href="/science/doe-explainsquarks/">dup</a>')
    assert original_html.parse_index_links(html, "https://www.energy.gov/science/doe-explains", row) == [
        "https://www.energy.gov/science/doe-explainsquarks"]
    sm = (b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
          b"<url><loc>https://www.energy.gov/science/doe-explainsneutrinos</loc></url>"
          b"<url><loc>https://evil.example.org/science/doe-explains-z</loc></url>"
          b"<url><loc>https://www.energy.gov/other</loc></url></urlset>")
    assert original_html.parse_sitemap(sm, row)["urls"] == ["https://www.energy.gov/science/doe-explainsneutrinos"]


# ================================================================================ parsers: defusedxml, DOCTYPE, gzip, caps
@pytest.mark.parametrize("raw,why", [
    (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><urlset><url><loc>&a;</loc></url></urlset>', "doctype"),
    (b'<?xml version="1.0"?><!doctype urlset><urlset/>', "doctype"),
    (b"\x1f\x8b\x08\x00" + b"x" * 40, "gzip"),
    (b"<urlset><url>", "malformed"),
    (b"<html><body/></html>", "root"),
])
def test_sitemap_refusals(raw, why):
    with pytest.raises(original_html.SitemapRefused) as e:
        original_html.parse_sitemap(raw, ROWS[0])
    assert str(e.value) == why


def test_sitemap_size_cap_counts_decoded_bytes(monkeypatch):
    set_mode(monkeypatch, "shadow", max_sitemap_bytes=100)
    with pytest.raises(original_html.SitemapRefused) as e:
        original_html.parse_sitemap(b"<urlset>" + b" " * 200 + b"</urlset>", ROWS[0])
    assert str(e.value) == "oversize"


def test_d3_sitemap_parse_calls_defusedxml_with_dtd_entities_external_forbidden(monkeypatch):
    import defusedxml.ElementTree as DET
    real, seen = DET.fromstring, []

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)
    monkeypatch.setattr(DET, "fromstring", spy)
    sm = (b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
          b"<url><loc>https://www.energy.gov/science/doe-explainsneutrinos</loc></url></urlset>")
    assert original_html.parse_sitemap(sm, ROWS[0])["urls"]
    assert seen and seen[0].get("forbid_dtd") is True and seen[0].get("forbid_entities") is True \
        and seen[0].get("forbid_external") is True  # swapping in stdlib ElementTree would make this list empty


def test_oversize_html_is_refused_whole_and_nothing_is_stored(monkeypatch):
    set_mode(monkeypatch, "on", max_html_bytes=2000)
    assert code(Web({DOE: page()})) == acq_config.reason(A.cfg(), "original_oversize")
    web = Web({DOE: page()}, over={DOE: A.Resp(200, page(), "text/html", {"Content-Encoding": "gzip"})})
    assert code(web) == acq_config.reason(A.cfg(), "encoding")  # non-identity encodings are refused outright


def test_non_html_content_type_is_refused(monkeypatch):
    set_mode(monkeypatch, "shadow")
    web = Web({}, over={DOE: A.Resp(200, b"%PDF-1.4", "application/pdf")})
    assert code(web) == acq_config.reason(A.cfg(), "convert")


# ================================================================================ step 5: licence (full text, before pruning)
def pd(text, url=DOE, href=""):
    return gov_public_domain.decide(url, text, href)


def test_licence_clean_page_and_image_credit_only_are_pd_marker_pages_are_held():
    full = lambda raw: html_main.visible_text(html_main.parse(raw))  # noqa: E731
    assert pd(full(page()))["applies"] is True
    assert pd(full(page(extra_main="<p>Image courtesy of Brookhaven National Laboratory.</p>")))["applies"] is True
    assert pd(full(page(footer="© 2025 Example Press")))["applies"] is False
    assert pd(full(page(extra_main="<p>Photo courtesy of a visiting scientist</p><p>Reprinted with permission.</p>")))[
        "applies"] is False
    assert pd(full(page(extra_main="<p>Operated by Example Research Alliance LLC for the U.S. Department of Energy.</p>")))[
        "applies"] is False  # contractor pattern
    assert pd(full(page()), url="https://www.fnal.gov/science/doe-explains-x")["applies"] is False  # excluded lab
    assert pd(full(page()), url="https://www.mass.gov/science/doe-explains-x")["applies"] is False  # state .gov


@pytest.mark.parametrize("marker", ["© 2024 Example Press", "Reprinted with permission from Example News",
                                    "This work was prepared by Example Lab under contract number DE-AC02-05CH11231"])
def test_markers_in_pruned_regions_still_hold_the_page(monkeypatch, marker):
    set_mode(monkeypatch, "on")
    for where in ({"footer": marker}, {"header_byline": f"<p>{marker}</p>"}, {"extra_main": f'<div class="share"><p>{marker}</p></div>'}):
        assert code(Web({DOE: page(**where)})) == acq_config.reason(A.cfg(), "original_not_pd"), where


def test_d4_page_licence_check_is_consulted_not_only_the_pd_rule(monkeypatch):
    # "contributed to this report" is a wire cue that page_licence.check holds (acc_third_party_page) while
    # gov_public_domain.decide alone would grant public domain: ignoring pl.ok in the route would store the page
    set_mode(monkeypatch, "on")
    raw = page(extra_main="<p>Jane Roe contributed to this report.</p>")
    full = html_main.visible_text(html_main.parse(raw))
    assert pd(full)["applies"] is True  # the PD rule alone passes it
    from vera import page_licence
    assert page_licence.check(DOE, {"text": full, "licence_href": "", "url": DOE})["ok"] is False
    assert code(Web({DOE: raw})) == acq_config.reason(A.cfg(), "original_not_pd")


def test_declared_licence_link_holds_the_page(monkeypatch):
    set_mode(monkeypatch, "on")
    raw = page(head_extra='<link rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">')
    assert code(Web({DOE: raw})) == acq_config.reason(A.cfg(), "original_not_pd")


def test_contractor_patterns_honoured_through_decide(monkeypatch):
    # the route hands the FULL text to gov_public_domain.decide: whatever contractor_patterns say, it follows. A JPL byline
    # pattern (branch govpd-1111, not merged here) is simulated by extending the config the rule loads.
    set_mode(monkeypatch, "on")
    cfgd = copy.deepcopy(gov_public_domain.load_config())
    cfgd["page_override"]["contractor_patterns"].append(r"(?m)^by .*\n.*jet propulsion laboratory")
    monkeypatch.setattr(gov_public_domain, "load_config", lambda path=None: cfgd)
    raw = page(extra_main="<p>By Robin Example</p><p>NASA's Jet Propulsion Laboratory</p>")
    assert code(Web({DOE: raw})) == acq_config.reason(A.cfg(), "original_not_pd")


def test_pd_rule_off_means_not_pd(monkeypatch):
    set_mode(monkeypatch, "on")
    cfgd = {**gov_public_domain.load_config(), "mode": "off"}
    monkeypatch.setattr(gov_public_domain, "load_config", lambda path=None: cfgd)
    assert code(Web({DOE: page()})) == acq_config.reason(A.cfg(), "original_not_pd")


# ================================================================================ step 6: generic extraction
X = SHIPPED["extract"]
CC = fulltext_loader.load_config()["converter"]


def convert(raw, bp=None):
    r = html_main.extract_main(raw, X, DOE_HOST, bp)
    return jats_text.html_to_text(r["html"], CC), r


def lines(t):
    return [re.sub(r"\s+", " ", ln).strip().lower() for ln in t.split("\n") if ln.strip()]


def family(n, **kw):
    out = []
    for i in range(n):
        body = [f"{PARA}Page {i} adds its own explanation number {i} about detectors and collisions. " * 2 for _ in range(3)]
        out.append(page(body, title=f"DOE Explains...Topic {i}", **kw))
    return out


def test_extraction_prunes_template_text_keeps_headings_and_first_paragraph():
    pages = family(4)
    cnt = {}
    for p in pages[1:]:
        for h in html_main.block_hashes(p, X):
            cnt[h] = cnt.get(h, 0) + 1
    bp = {h for h, c in cnt.items() if c >= 2}
    out, r = convert(pages[0], bp)
    text, ls = out["text"], lines(out["text"])
    assert any("page 0 adds its own explanation" in ln for ln in ls)
    for tmpl in ("an official website of the united states government", "home", "privacy", "resources and related terms",
                 "higgs boson", "doe explains is a series of short articles"):
        assert not any(tmpl in ln for ln in ls), tmpl
    assert "Section 1" in text and "Section 3" in text  # headings kept
    # metric L (leakage) and F300 over the family's shared lines, title excluded
    shared = set()
    allc = {}
    for p in pages:
        for ln in set(lines(jats_text.html_to_text(p, CC)["text"])):
            allc[ln] = allc.get(ln, 0) + 1
    shared = {ln for ln, c in allc.items() if c >= 2 and len(ln) > 2}
    tot = sum(len(x) for x in ls) or 1
    leak = sum(len(x) for x in ls if x in shared and not x.startswith("section ")) / tot
    assert leak <= 0.05
    head300 = " ".join(ls)[:300]
    assert not [x for x in ls if x in shared and not x.startswith(("section ", "doe explains")) and x in head300]


def test_repeated_meaningful_headings_survive_the_boilerplate_set():
    pages = family(4)
    bp = set()
    for p in pages:
        bp |= html_main.block_hashes(p, X)  # every leaf block of every page is "boilerplate"
    out, _r = convert(pages[0], bp)
    assert "Section 1" in out["text"] and "Resources and Related Terms" not in out["text"]  # pruned by 6b, not by 6c
    assert [s for s in out["sections"] if "Section" in (s.get("title") or "")]


def test_link_dense_prose_paragraph_is_kept_under_the_density_limit():
    prose = ("Physicists study <a href='/q'>quarks</a>, <a href='/g'>gluons</a> and the strong force at accelerators; "
             "the long explanatory sentence continues with ordinary words so that links are a small share of the text "
             "of this paragraph and it is plainly body prose and not a list of resource links at all.")
    raw = page([prose + " " + PARA * 2, PARA * 3, PARA * 3])
    out, _ = convert(raw)
    assert "plainly body prose" in out["text"]


def test_empty_main_falls_back_to_body_then_no_body(monkeypatch):
    r = html_main.extract_main(page(empty_main=True), X, DOE_HOST)
    assert r["root"] == "body"  # fallback; the converter then needs headed sections
    with pytest.raises(jats_text.ConversionError):
        jats_text.html_to_text(r["html"], CC)
    set_mode(monkeypatch, "on")
    only = b"<html><head><title>x</title></head><body><main></main><p>just a loose paragraph " + b"word " * 400 + b"</p></body></html>"
    assert code(Web({DOE: only})) == acq_config.reason(A.cfg(), "original_no_body")  # never partial text


def test_thin_page_is_refused(monkeypatch):
    set_mode(monkeypatch, "on")
    thin = page(["Quarks are particles."], related=False, blurb=False)
    assert code(Web({DOE: thin})) == acq_config.reason(A.cfg(), "original_thin")


def test_per_paragraph_retention_every_prose_paragraph_keeps_90_percent():
    paras = [f"{PARA}Paragraph {i} also names <a href='/x'>a linked term</a> inline, as real prose does. " * 2 for i in range(5)]
    out, _ = convert(page(paras))
    flat = re.sub(r"\s+", " ", out["text"])
    for p in paras:
        plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p)).strip()
        if len(plain) >= 80:
            assert plain in flat or sum(1 for w in plain.split() if w in flat) / len(plain.split()) >= 0.9


def test_provenance_is_captured_before_pruning_from_pruned_regions():
    raw = page(header_byline='<p class="byline">By Jane Example, Office of Science</p>',
               head_extra='<meta name="author" content="Jane Example"><link rel="license" href="https://example.gov/terms">',
               footer="Image credit: Example Lab. Content is in the public domain unless noted.")
    doc = html_main.parse(raw)
    p = html_main.capture_provenance(doc, SHIPPED["provenance"])
    assert any("Jane Example" in b for b in p["bylines"]) and "Jane Example" in p["authors"]
    assert p["licence_href"] == "https://example.gov/terms"
    assert any("public domain" in s for s in p["statements"])
    out, _ = convert(raw)
    assert "By Jane Example" not in out["text"]  # pruned from the body, kept in provenance


def test_parser_bounds():
    with pytest.raises(html_main.HtmlRefused):
        html_main.parse(b"<div>" * 500)
    with pytest.raises(html_main.HtmlRefused):
        html_main.parse(b"<p>x</p>" * 50, max_nodes=10)


# ================================================================================ mode ladder and a full success path
def test_shadow_runs_every_step_stores_nothing_and_logs_counts_only(monkeypatch, caplog):
    set_mode(monkeypatch, "shadow")
    caplog.set_level("INFO", logger="vera.autoacq")
    web = Web({DOE: page()})
    assert code(web) == acq_config.reason(A.cfg(), "original_shadow")
    logged = " ".join(r.getMessage() for r in caplog.records if "original_html_shadow" in r.getMessage())
    assert logged and "energy.gov" not in logged and "quark" not in logged.lower()
    assert research_outcome.reason_from_code("R-ACQ-ORIGINAL_SHADOW") == "no_route"


def test_on_mode_returns_a_plan_for_the_store_row_and_shadow_for_a_non_store_row(monkeypatch):
    set_mode(monkeypatch, "on")
    web = Web({DOE: page()})
    plan, _isbn = run(web)
    prov = plan["source"]["provenance"]
    assert prov["licence"]["content"]["id"] == "LicenseRef-us-government-work"
    assert prov["discovered_via"] == "original_html" and prov["acquisition"]["resolver"] == "original_html"
    oh = prov["acquisition"]["original_html"]
    assert oh["row"] == "energy_gov_doe_explains" and oh["gov_pd"]["applies"] is True and "page" in oh
    assert plan["canonical_url"].startswith("https://www.energy.gov/")
    assert prov["identifier"]["kind"] == "original_url" if "identifier" in prov else True
    # NASA row: storage off until the contractor fix merges, even in mode on
    nasa_page = page(title="Dark Matter")
    c = code(Web({NASA: nasa_page}), c=cand(NASA))
    assert c == acq_config.reason(A.cfg(), "original_shadow")


def test_stored_row_reads_back_through_the_acquired_reader(monkeypatch):
    set_mode(monkeypatch, "on")
    from vera import fulldoc_db
    plan, _ = run(Web({DOE: page()}))
    srcs, spans = A.rows_from_plans([plan])
    cfg = A.cfg()
    c = fulldoc_db.load_acquired_rows(srcs, spans, auto_acquire._fd_cfg(), cfg,
                                      ingest_cfg=acq_config.ingest_cfg(fulltext_loader.load_config(), cfg))
    assert c.status == "ok" and len(c.acquired) == 1


def test_residual_floor_and_licence_branch_key_on_the_acquisition_resolver_not_the_items_claim(monkeypatch):
    set_mode(monkeypatch, "on")
    from vera import fulldoc_db
    plan, _ = run(Web({DOE: page()}))
    assert plan["source"]["provenance"]["item"]["discovered_via"] == "original_html"
    assert len(plan["source"]["content"]) < 4000  # only the route's own floor lets this page in

    def load(p):
        srcs, spans = A.rows_from_plans([p])
        cfg = A.cfg()
        return fulldoc_db.load_acquired_rows(srcs, spans, auto_acquire._fd_cfg(), cfg,
                                             ingest_cfg=acq_config.ingest_cfg(fulltext_loader.load_config(), cfg))
    assert len(load(plan).acquired) == 1
    forged = copy.deepcopy(plan)  # same item claim, but the acquisition record names another resolver
    forged["source"]["provenance"]["acquisition"]["resolver"] = "europepmc"
    c = load(forged)
    assert len(c.acquired) == 0 and c.unusable  # neither the 1500 floor nor the federal licence branch applies
    # a manifest item (prepare_from_parts without route=) can never claim the shorter floor
    item = copy.deepcopy(plan["source"]["provenance"]["item"])
    meta = {**plan["source"]["provenance"]["meta"], "has_body": True, "body_chars": 2000, "abstract_chars": 0}
    icfg = acq_config.ingest_cfg(fulltext_loader.load_config(), A.cfg())
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_text_record(item, "x" * 2500, meta, icfg)
    fulltext_loader.check_content("x" * 2500, meta, original_html.relaxed_ingest_cfg(icfg, SHIPPED))


def test_length_floor_is_relaxed_only_for_this_route():
    base = fulltext_loader.load_config()
    r = original_html.relaxed_ingest_cfg(base, SHIPPED)
    assert r["min_body_chars"] == 1500 and base["min_body_chars"] == 4000  # copy; input untouched
    assert r["min_fulltext_chars"] == 1500 and r["min_body_to_abstract_ratio"] == base["min_body_to_abstract_ratio"]
    meta = {"has_body": True, "body_chars": 2000, "abstract_chars": 0}
    with pytest.raises(fulltext_loader.Refused):
        fulltext_loader.check_content("x" * 2000, meta, base)  # an ordinary item still needs the full floor
    fulltext_loader.check_content("x" * 2000, meta, r)


def test_new_reason_codes_have_outcome_and_user_text():
    for k in ("original_not_pd", "original_no_body", "original_oversize", "original_thin", "original_shadow"):
        c = acq_config.reason(A.cfg(), k)
        assert c in research_outcome.KNOWN_CODES and discovery_panel.reason_text(c) != discovery_panel.REASON_GENERIC


# ================================================================================ catalogue snapshot (read only)
def snap(entries=(), denied=(), boiler=None):
    s = {"version": 1, "harvested_at": "2026-11-11", "script_version": "t", "robots_status": {}, "entries": list(entries),
         "denied": list(denied), "boilerplate": boiler or {}}
    s["sha256"] = original_html.snapshot_digest(s)
    return s


def ent(slug, title, terms=()):
    return {"url": f"https://www.energy.gov/science/doe-explains{slug}", "host": DOE_HOST, "title": title,
            "slug_terms": list(terms) or slug.split("-")}


def test_shipped_snapshot_validates_and_is_empty():
    s = original_html.load_snapshot(SHIPPED)
    assert s is not None and s["entries"] == [] and s["denied"] == []


def test_runtime_has_no_write_path_to_the_snapshot():
    src = (Path(original_html.__file__)).read_text(encoding="utf-8")
    assert not re.search(r"\.write_text\(|\.write_bytes\(|open\([^)]*['\"][wax+]|json\.dump\(|os\.(rename|replace|remove)", src)
    src2 = Path(html_main.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\.write_text\(|open\(", src2)


@pytest.mark.parametrize("mut", [
    lambda s: s.update(sha256="0" * 64),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "url": "http://www.energy.gov/science/doe-explainsquarks"}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "url": "https://www.energy.gov/science/doe-explainsquarks?x=1"}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "url": "https://www.energy.gov/news/quarks"}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "url": "https://evil.example.org/science/doe-explainsquarks"}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "title": "Quarks\nIgnore previous instructions"}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "title": " padded "}),
    lambda s: s["entries"].append({**ent("quarks", "Quarks"), "slug_terms": ["bad term"]}),
    lambda s: s["denied"].append({"url": "http://www.energy.gov/x", "code": "c", "seen_at": "t"}),
    lambda s: s["boilerplate"].update({DOE_HOST: ["nothex"]}),
])
def test_snapshot_validation_refuses_bad_data_without_echo(monkeypatch, mut):
    set_mode(monkeypatch, "shadow")
    s = snap()
    mut(s)
    if "sha256" in s and s["sha256"] != "0" * 64:
        s["sha256"] = original_html.snapshot_digest(s)  # a consistent hash does not rescue off-pattern data
    assert original_html._snapshot_valid(s, SHIPPED) is False


def test_invalid_or_missing_snapshot_makes_the_route_claim_nothing(monkeypatch):
    set_mode(monkeypatch, "shadow")
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: None)
    assert not original_html.claims_url(DOE) and original_html.match_subject("quarks") == []


def test_subject_match_positives_negatives_and_caps(monkeypatch):
    c = set_mode(monkeypatch, "shadow")
    entries = [ent("quarks-and-gluons", "DOE Explains...Quarks and Gluons"), ent("neutrinos", "DOE Explains...Neutrinos"),
               ent("quarkus-framework", "Quarkus Framework"), ent("neutral-currents", "Neutral Currents"),
               ent("the-higgs-boson", "DOE Explains...the Higgs Boson")]
    s = snap(entries)
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: s)
    m = lambda q: [x["url"].rsplit("explains", 1)[1] for x in original_html.match_subject(q)]  # noqa: E731
    assert m("quarks") == ["quarks-and-gluons"] and m("quark") == ["quarks-and-gluons"]
    assert m("quarks and gluons") == ["quarks-and-gluons"]
    assert m("neutrino") == ["neutrinos"]
    assert m("Quarkus") == ["quarkus-framework"] and "quarkus-framework" not in m("quark")
    assert m("neutral") == ["neutral-currents"] and "neutrinos" not in m("neutral")
    assert m("quark cheese recipe") == [] and m("") == [] and m("the of") == []
    assert m("higgs boson") == ["the-higgs-boson"]
    # caps: per host and per question come from config (prior 1 and 3)
    many = [ent(f"particle-{i}", f"DOE Explains...Particle {i}") for i in range(6)]
    s2 = snap(many)
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: s2)
    assert len(original_html.match_subject("particle")) == c["limits"]["max_candidates_per_host"] == 1
    c["limits"]["max_candidates_per_host"] = 5
    assert len(original_html.match_subject("particle")) == 3  # per-question ceiling
    c["mode"] = "off"
    assert original_html.match_subject("particle") == []


def test_denied_snapshot_entry_is_never_offered(monkeypatch):
    set_mode(monkeypatch, "shadow")
    e = ent("neutrinos", "DOE Explains...Neutrinos")
    s = snap([e], denied=[{"url": e["url"], "code": "R-ACQ-ACCESS_DENIED", "seen_at": "t"}])
    monkeypatch.setattr(original_html, "load_snapshot", lambda cfg=None: s)
    assert original_html.match_subject("neutrino") == []


# ================================================================================ budget / queue wiring is shared code
def test_route_has_no_private_counter_or_network_import():
    src = Path(original_html.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(requests|urllib3|socket|http\.client)\b", src, re.M)
    assert "docs_day" not in src and "note_docs" not in src


# ================================================================================ optional: the saved real C0 pages
C0 = Path(__file__).resolve().parents[3] / "USFED-C0-1111" / "pages"  # <root>/scratch/USFED-C0-1111/pages from a worktree
C0_OK = C0.is_dir() and any(C0.glob("*.html"))


@pytest.mark.skipif(not C0_OK, reason="saved USFED-C0-1111 pages not present (never copied into the repo)")
def test_real_c0_pages_meet_g_b_leave_one_out():
    files = sorted(C0.glob("*.html"))
    idx = ("science_doe_explains.html", "science_nasa_gov_universe.html")
    samples = [f for f in files if not f.name.endswith(idx)]
    host = lambda f: "doe" if "energy_gov" in f.name else "nasa"  # noqa: E731
    raw_lines = {f: set(lines(jats_text.html_to_text(f.read_bytes(), CC)["text"])) for f in files if f.name}
    for f in samples:
        h = host(f)
        others = [o for o in samples if host(o) == h and o != f]
        cnt = {}
        for o in others:
            for x in html_main.block_hashes(o.read_bytes(), X):
                cnt[x] = cnt.get(x, 0) + 1
        bp = {k for k, c in cnt.items() if c >= max(2, len(others) // 2)}
        out, _r = convert(f.read_bytes(), bp)
        ls = lines(out["text"])
        shared = {}
        for g in files:
            if host(g) == h:
                for ln in raw_lines[g]:
                    shared[ln] = shared.get(ln, 0) + 1
        shared = {ln for ln, c in shared.items() if c >= 2 and len(ln) > 2}
        tot = sum(map(len, ls)) or 1
        # headings are kept by design (plan pass 12: 6c never drops a heading), so a shared heading is not leakage; the
        # page's own title line is excluded as in the plan's metric. Recorded finding: counting headings fails F300 on
        # one NASA page ("Black Holes", shared across the black-holes subtree).
        heads = {re.sub(r"\s+", " ", s["title"]).strip().lower() for s in out["sections"] if s.get("title")}
        title = ls[0] if ls else ""
        leak = sum(len(x) for x in ls if x in shared and x != title and x not in heads) / tot
        j = " ".join(ls)
        f300 = [x for x in ls if x in shared and x != title and x not in heads and j.find(x) < 300]
        assert out["sections"] and leak <= 0.05 and not f300, (f.name, leak, f300)
