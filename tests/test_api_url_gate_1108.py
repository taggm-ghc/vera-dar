"""API-URL-GATE-1108: generic web fetchers check config/api_permissions.json on every hop.

An API URL that arrives as a search result, a portal link or a redirect target is refused unless its entry permits
the use ("fulltext" for a generic page fetch); non-API pages are unaffected; an unusable config refuses everything.
Offline: DNS, transports and robots are fakes; no socket is ever opened."""
import socket

import pytest

from vera import api_permissions as ap
from vera import public_fetch as pf
from vera import search_providers as sp
from vera import selective_fetch as sf

DENIED_URLS = [
    "https://export.arxiv.org/api/query?search_query=all:x",       # arXiv export API (owner deny)
    "https://arxiv.org/api/query?id_list=1234.5678",               # arXiv web API path (owner deny)
    "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/x",   # Semantic Scholar (owner deny)
    "https://api.unpaywall.org/v2/10.1/x?email=a@b.c",             # Unpaywall (owner deny)
    "https://API.SemanticScholar.org./graph/v1/paper/x",           # case + trailing dot: same host
    "https://arxiv.org/x/../api/query",                            # dot-segment into the API path
    "https://api.semanticscholar\u3002org/graph/v1/paper/x",      # ideographic full stop (IDNA maps it to ".")
    "https://\uff41pi.unpaywall.org/v2/10.1/x",                    # full-width "a" (NFKC maps it to "a")
]
METADATA_ONLY = "https://api.openalex.org/works/W1"                # permitted, but metadata only -> R-API-SCOPE
PERMITTED = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML"  # permit, fulltext in scope
PAGE = "https://a.example/p"


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body
        self.encoding = "utf-8"

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


def html(body=b"<p>hello</p>"):
    return Resp(200, {"Content-Type": "text/html; charset=utf-8"}, body)


def redirect(to):
    return Resp(302, {"Location": to})


def test_shipped_policy_classifies_the_fixtures():
    for u in DENIED_URLS:
        assert ap.is_api_url(u) and not ap.decide(u, "fulltext").allowed, u
    assert ap.decide(METADATA_ONLY, "fulltext").code == ap.SCOPE
    assert ap.decide(METADATA_ONLY, "metadata").allowed
    assert ap.decide(PERMITTED, "fulltext").allowed
    assert not ap.is_api_url(PAGE)


# ------------------------------------------------------------------ selective_fetch.pinned_get (the shared transport)
@pytest.fixture
def no_net(monkeypatch):
    """is_public_url refuses (so a URL that passes the API gate stops at the address check, still offline)."""
    seen = []

    def check(url):
        seen.append(url)
        return False, "offline-test"
    monkeypatch.setattr(sf, "is_public_url", check)
    monkeypatch.setattr(sf.requests, "Session", lambda: pytest.fail("no session may be opened"))
    return seen


@pytest.mark.parametrize("url", DENIED_URLS + [METADATA_ONLY])
def test_pinned_get_refuses_api_urls_before_resolution(no_net, url):
    with pytest.raises(sf.ApiRefused) as ei:
        sf.pinned_get(url)
    assert not no_net  # refused before the address check: nothing resolved, nothing sent
    assert isinstance(ei.value, ap.ApiDenied) and isinstance(ei.value, sf.requests.RequestException)
    assert ei.value.code in (ap.DENIED, ap.SCOPE)


@pytest.mark.parametrize("url", [PERMITTED, PAGE])
def test_pinned_get_lets_permitted_and_non_api_urls_through_the_gate(no_net, url):
    with pytest.raises(sf.PinnedRefused) as ei:
        sf.pinned_get(url)
    assert not isinstance(ei.value, sf.ApiRefused) and no_net == [url]  # reached the (fake) address check


def test_gate_api_urls_metadata_purpose_reaches_the_transport(no_net):
    get = ap.gate_api_urls(sf.pinned_get)
    with pytest.raises(sf.PinnedRefused) as ei:
        get(METADATA_ONLY, api_purpose="metadata")
    assert not isinstance(ei.value, sf.ApiRefused) and no_net == [METADATA_ONLY]
    with pytest.raises(ap.ApiDenied):
        get(METADATA_ONLY)  # default purpose fulltext: out of the entry's scope
    assert ap.current_purpose() == "fulltext"  # the scope does not leak past the call


# --------------------------------------------------------------- selective_fetch.fetch_html / fetch_candidate hops
class Get:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url, **kw):
        self.calls.append(url)
        return self.pages[url]


def _fetch_html(url, get):
    robots = []
    out = sf.fetch_html(url, http_get=get, url_check=lambda u: (True, None), pacer=sf.HostPacer({}),
                        robots_ok=lambda u, **k: robots.append(u) or True)
    return out, robots


@pytest.mark.parametrize("url", DENIED_URLS + [METADATA_ONLY])
def test_fetch_html_refuses_search_result_api_urls(url):
    get = Get({})
    (body, final, err), robots = _fetch_html(url, get)
    assert body is None and err.startswith("blocked: R-API-") and not get.calls and not robots


@pytest.mark.parametrize("target", DENIED_URLS)
def test_fetch_html_refuses_redirect_into_api_host(target):
    get = Get({PAGE: redirect(target)})
    (body, _f, err), robots = _fetch_html(PAGE, get)
    assert body is None and err.startswith("blocked: R-API-")
    assert get.calls == [PAGE] and robots == [PAGE]  # the API hop got neither robots.txt nor a request


def test_fetch_html_non_api_and_permitted_pass():
    get = Get({PAGE: html(), PERMITTED: html(b"<p>xml</p>")})
    (body, final, err), _ = _fetch_html(PAGE, get)
    assert err is None and "hello" in body and final == PAGE
    (body, final, err), _ = _fetch_html(PERMITTED, get)
    assert err is None and "xml" in body and final == PERMITTED


def test_fetch_candidate_refuses_api_url():
    get = Get({})
    r = sf.fetch_candidate(DENIED_URLS[2], http_get=get, url_check=lambda u: (True, None), pacer=sf.HostPacer({}),
                           robots_ok=lambda u, **k: True)
    assert not r.ok and r.error.startswith("blocked: R-API-") and not get.calls


# ------------------------------------------------------------------------------------------- public_fetch.fetch hops
CFG = {"user_agent": "UA-test", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 5000,
       "max_decompressed_bytes": 20000, "max_robots_bytes": 2000, "timeout_s": 5,
       "content_types": ["text/html", "application/xhtml+xml", "text/plain"]}
ROBOTS = Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")


class Net:
    def __init__(self, pages):
        self.pages, self.calls, self.lookups = pages, [], []

    def resolve(self, host, port):
        self.lookups.append(host)
        return [(socket.AF_INET, 0, 0, "", ("93.184.216.34", port))]

    def transport(self, host, ip, port, target, headers, timeout):
        self.calls.append((host, target, dict(headers)))
        return self.pages[(host, target)]


@pytest.mark.parametrize("url", DENIED_URLS + [METADATA_ONLY])
def test_public_fetch_refuses_api_urls_before_dns(url):
    n = Net({})
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(url, CFG, resolve=n.resolve, transport=n.transport)
    assert ei.value.reason == "api_denied" and not n.lookups and not n.calls


@pytest.mark.parametrize("target", DENIED_URLS)
def test_public_fetch_refuses_redirect_into_api_host(target):
    n = Net({("a.example", "/robots.txt"): ROBOTS, ("a.example", "/p"): redirect(target)})
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(PAGE, CFG, resolve=n.resolve, transport=n.transport)
    assert ei.value.reason == "api_denied"
    assert [c[:2] for c in n.calls] == [("a.example", "/robots.txt"), ("a.example", "/p")]  # nothing to the API host


def test_public_fetch_non_api_unaffected():
    n = Net({("a.example", "/robots.txt"): ROBOTS, ("a.example", "/p"): html()})
    r = pf.fetch(PAGE, CFG, resolve=n.resolve, transport=n.transport)
    assert r["text"] == "hello" and n.calls[-1][2]["User-Agent"] == "UA-test"


def test_public_fetch_permitted_api_uses_entry_ua_and_rate_portal():
    path = "/europepmc/webservices/rest/PMC123/fullTextXML"
    n = Net({("www.ebi.ac.uk", "/robots.txt"): ROBOTS, ("www.ebi.ac.uk", path): html(b"<p>ok</p>")})
    r = pf.fetch(PERMITTED, CFG, resolve=n.resolve, transport=n.transport)
    ua = ap.decide(PERMITTED, "fulltext").conditions.get("user_agent")
    assert r["text"] == "ok" and n.calls[-1][1] == path
    if ua:
        assert n.calls[-1][2]["User-Agent"] == ua
    # the europepmc portal admitted that request: an immediate second one is skipped (burst 1), nothing sent
    before = len(n.calls)
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(PERMITTED, CFG, resolve=n.resolve, transport=n.transport)
    assert ei.value.reason == "rate_limited" and all(c[1] != path for c in n.calls[before:])


# ----------------------------------------------------------------------------------- search_providers.landing_doi
@pytest.mark.parametrize("url", DENIED_URLS[:4])
def test_landing_doi_refuses_api_url_and_redirect(url):
    get = Get({PAGE: redirect(url)})
    kw = dict(get=get, url_check=lambda u: (True, None), robots_ok=lambda u: True)
    assert sp.landing_doi(url, 5.0, 10000, **kw) is None and not get.calls
    assert sp.landing_doi(PAGE, 5.0, 10000, **kw) is None and get.calls == [PAGE]


# ---------------------------------------------------------------------------------------- fail closed on bad config
@pytest.fixture
def broken_policy():
    from pathlib import Path
    pol = ap.load_policy(Path(__file__).parent / "no_such_api_permissions_1108.json")  # unreadable: nothing written
    assert pol.error
    ap.set_policy_for_tests(pol)
    yield
    ap.set_policy_for_tests(None)


def test_unusable_config_refuses_every_generic_fetch(broken_policy, no_net):
    with pytest.raises(sf.ApiRefused):
        sf.pinned_get(PAGE)
    get = Get({PAGE: html()})
    (body, _f, err), _ = _fetch_html(PAGE, get)
    assert body is None and err == "blocked: R-API-DENIED" and not get.calls
    n = Net({("a.example", "/robots.txt"): ROBOTS, ("a.example", "/p"): html()})
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(PAGE, CFG, resolve=n.resolve, transport=n.transport)
    assert ei.value.reason == "api_denied" and not n.calls
    with pytest.raises(ap.ApiDenied):
        ap.gate_api_urls(lambda u, **k: pytest.fail("sent"))(PAGE)


def test_unparseable_url_refused_not_raised():
    get = Get({})
    (body, _f, err), _ = _fetch_html("https://[::1]\\x.com/", get)
    assert body is None and err == "blocked: R-API-DENIED" and not get.calls
