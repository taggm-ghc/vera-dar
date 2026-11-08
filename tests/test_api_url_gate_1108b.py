"""API-URL-GATE-1108b: unlisted API-like URLs and IP literals are refused by the generic fetchers (config as data,
calibrated so no page the app fetches today is refused), a permitted API hop uses its own entry's rate portal, and
the purpose is passed explicitly (the contextvar is a fallback bound to one URL). Offline: no socket is opened."""
import json
import logging
import re
import socket
from pathlib import Path

import pytest

from vera import api_permissions as ap
from vera import public_fetch as pf
from vera import rate_limit
from vera import selective_fetch as sf

ROOT = Path(__file__).resolve().parent.parent
UNLISTED_API = [
    "https://openrouter.ai/api/v1/models",                                    # api/v<N>
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc",      # eutils host
    "https://www.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id=1",            # /eutils/ path
    "https://en.wikipedia.org/w/api.php?action=query",                        # MediaWiki API
    "https://api.example.org/v1/items",                                       # api.* host
    "https://example.org/api",                                                # bare /api
    "https://example.org/graphql",                                            # GraphQL endpoint
    "https://example.org/search?q=x&format=json",                             # query marker
    "https://example.org/select?q=x&wt=json",                                 # Solr JSON
]
IP_LITERALS = ["https://93.184.216.34/p", "https://[2606:4700::1111]/p", "http://127.1/", "http://2130706433/",
               "https://0x5db8d822/p"]
# Page shapes the app fetches today (generic fetchers, auto-acquisition landing pages, official and news lanes).
CURRENT_PAGES = [
    "https://arxiv.org/abs/2401.01234v2", "https://arxiv.org/pdf/2401.01234v2", "https://arxiv.org/html/2401.01234v2",
    "https://doi.org/10.1038/s41586-020-2649-2", "https://europepmc.org/article/MED/12345678",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/", "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    "https://en.wikipedia.org/wiki/Epidemic", "https://www.nature.com/articles/s41586-020-2649-2",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000001",
    "https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)", "https://www.cdc.gov/flu/index.html",
    "https://docs.slack.dev/", "https://www.bbc.co.uk/news/science-environment-1", "https://example.org/docs/api",
    # documentation pages about APIs (calibration: a bare /api/ segment would have refused these)
    "https://developers.openai.com/api/docs/pricing", "https://info.arxiv.org/help/api/tou.html",
    "https://unpaywall.org/products/api", "https://www.semanticscholar.org/product/api",
]


def _refused(url, purpose=None):
    try:
        ap.gate_url(url, purpose)
    except ap.ApiDenied as e:
        return e.why
    return None


@pytest.mark.parametrize("url", UNLISTED_API)
def test_unlisted_api_like_refused_and_logged(url, caplog):
    with caplog.at_level(logging.WARNING, logger="vera"):
        why = _refused(url)
    assert why and why.startswith("unlisted_api_like:")
    assert any("R-API-DENIED" in r.getMessage() and why in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("url", CURRENT_PAGES)
def test_current_pages_not_refused(url):
    assert _refused(url) is None


# ------------------------------------------------------- calibration corpus: every URL / host in config and code
_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+")
_HOST = re.compile(r"^(?=.{4,253}$)([a-z0-9-]+\.)+[a-z]{2,}$")
# Hosts whose refusal is intended: API endpoints (listed entries, LLM providers reached through their SDKs, local API
# base URLs) and api.slack.com (official-lane host with no allowed path). Anything else refused is a calibration bug.
EXPECTED_REFUSED_HOSTS = {
    "api.gdeltproject.org", "api.openalex.org", "api.crossref.org", "api.semanticscholar.org", "api.unpaywall.org",
    "export.arxiv.org", "doaj.org", "hn.algolia.com", "openrouter.ai", "api.groq.com", "api.openai.com",
    "api.slack.com", "eutils.ncbi.nlm.nih.gov", "127", "127.0.0.1", "inspirehep.net"}


def _corpus():
    urls, hosts = set(), set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            urls.update(u.rstrip(".,)") for u in _URL.findall(o))
            if _HOST.match(o.lower()):
                hosts.add(o.lower())
    for p in sorted((ROOT / "config").rglob("*.json")):
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except ValueError:
            continue
    for p in (ROOT / "vera").rglob("*.py"):
        urls.update(u.rstrip(".,)\"'") for u in _URL.findall(p.read_text(encoding="utf-8")))
    return sorted(urls | {f"https://{h}/" for h in hosts})


def test_no_config_or_code_url_outside_the_expected_api_set_is_refused():
    from urllib.parse import urlsplit
    corpus = _corpus()
    assert len(corpus) > 150  # the calibration corpus is really read
    bad = []
    for u in corpus:
        try:
            host = (urlsplit(u).hostname or "").lower()
        except ValueError:
            continue
        if _refused(u) and host not in EXPECTED_REFUSED_HOSTS:
            bad.append(u)
    assert not bad, f"pages refused by the API gate (recalibrate config/api_like_patterns.json): {bad}"


def test_listed_entry_decides_not_the_patterns():
    assert _refused("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML", "fulltext") is None
    assert _refused("https://api.openalex.org/works/W1", "metadata") is None   # api.* host, but listed + permitted
    assert _refused("https://api.openalex.org/works/W1", "fulltext") == "scope_fulltext"


# ------------------------------------------------------------------------------------------------- IP literals (2)
@pytest.mark.parametrize("url", IP_LITERALS)
def test_ip_literal_refused_unless_listed(url):
    assert _refused(url) == "ip_literal"
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname
    like = ap.load_like()
    ap.set_like_for_tests(ap.LikePatterns(like.host_prefixes, like.path_patterns, like.query_markers,
                                          frozenset({host})))
    try:
        assert _refused(url) is None
    finally:
        ap.set_like_for_tests(None)


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body, self.encoding = status, headers or {}, body, "utf-8"

    def iter_content(self, n):
        yield self.body

    def close(self):
        pass


class Get:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url, **kw):
        self.calls.append(url)
        return self.pages[url]


@pytest.mark.parametrize("url", UNLISTED_API[:3] + IP_LITERALS[:2])
def test_every_generic_path_refuses(url, monkeypatch):
    monkeypatch.setattr(sf, "is_public_url", lambda u: pytest.fail("resolved"))
    with pytest.raises(sf.ApiRefused):
        sf.pinned_get(url)
    g = Get({"https://a.example/p": Resp(302, {"Location": url})})
    body, _f, err = sf.fetch_html("https://a.example/p", http_get=g, url_check=lambda u: (True, None),
                                  pacer=sf.HostPacer({}), robots_ok=lambda u, **k: True)
    assert body is None and err == "blocked: R-API-DENIED" and g.calls == ["https://a.example/p"]
    calls = []
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(url, {"user_agent": "UA", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 5000,
                       "max_decompressed_bytes": 20000, "max_robots_bytes": 2000, "timeout_s": 5,
                       "content_types": ["text/html"]},
                 resolve=lambda h, p: calls.append(h) or [(socket.AF_INET, 0, 0, "", ("93.184.216.34", p))],
                 transport=lambda *a: calls.append(a))
    assert ei.value.reason == "api_denied" and not calls


@pytest.mark.parametrize("url", ["https://example.org/s?forma%74=json", "https://example.org/s?format=%6Ason",
                                 "https://example.org/s?a=1;format=json", "https://example.org/s?FORMAT=JSON",
                                 "https://example.org/s?format=%256Ason"])
def test_query_marker_evasions_refused(url):
    assert _refused(url) == "unlisted_api_like:format_json"


def test_pinned_get_gates_params_merged_into_the_url(monkeypatch):
    monkeypatch.setattr(sf, "is_public_url", lambda u: pytest.fail("resolved"))
    with pytest.raises(sf.ApiRefused):
        sf.pinned_get("https://example.org/search", params={"q": "x", "format": "json"})


def test_unencodable_host_refused():
    assert _refused("https://xn--aé\u0000.example/p") == "unparseable_url" or _refused(
        "https://é" + "a" * 70 + ".example/p") == "unparseable_url"


def test_reporting_lane_names_its_purpose():
    import inspect

    from vera import reporting_lane
    assert 'require(url, "metadata")' in inspect.getsource(reporting_lane._default_http_get)


def test_unusable_like_config_refuses_everything():
    ap.set_like_for_tests(ap.load_like(ROOT / "config" / "no_such_api_like_1108b.json"))
    try:
        assert _refused("https://a.example/p") == "config_unusable"
    finally:
        ap.set_like_for_tests(None)


# ------------------------------------------------------------------------------------- rate portals (4), purpose (6)
PERMITTED = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"


@pytest.fixture
def portals(monkeypatch):
    used = []
    real = rate_limit.call_with_limit

    def spy(ad, fn, *a, **kw):
        used.append(ad.name)
        return real(ad, fn, *a, **kw)
    monkeypatch.setattr(rate_limit, "call_with_limit", spy)
    monkeypatch.setattr(sf, "is_public_url", lambda u: (False, "offline-test"))
    return used


def test_fixed_portal_transport_uses_the_api_entrys_own_portal_once(portals):
    entry_portal = ap.decide(PERMITTED, "fulltext").conditions["rate_portal"]
    get = ap.limited_transport(sf.pinned_get, fixed_portal="wikipedia")
    with pytest.raises(sf.PinnedRefused):
        get(PERMITTED, api_purpose="fulltext")
    assert portals == [entry_portal]  # never "wikipedia"; pinned_get does not take it a second time


def test_fixed_portal_still_applies_to_pages(portals):
    get = ap.limited_transport(sf.pinned_get, fixed_portal="wikipedia")
    with pytest.raises(sf.PinnedRefused):
        get("https://en.wikipedia.org/wiki/Epidemic")
    assert portals == ["wikipedia"]


def test_bare_pinned_get_takes_the_entry_portal(portals):
    with pytest.raises(sf.PinnedRefused):
        sf.pinned_get(PERMITTED, api_purpose="fulltext")
    assert portals == [ap.decide(PERMITTED, "fulltext").conditions["rate_portal"]]


def test_purpose_explicit_and_fallback_bound_to_one_url(monkeypatch):
    monkeypatch.setattr(sf, "is_public_url", lambda u: (False, "offline-test"))
    meta = "https://api.openalex.org/works/W1"
    assert ap.accepts_purpose(sf.pinned_get) and ap.accepts_purpose(ap.gate_api_urls(sf.pinned_get))
    assert not ap.accepts_purpose(lambda url, **kw: None)
    with pytest.raises(sf.PinnedRefused) as ei:
        sf.pinned_get(meta, api_purpose="metadata")
    assert not isinstance(ei.value, sf.ApiRefused)
    with ap.purpose_scope("metadata", "https://api.openalex.org/works/OTHER"):
        with pytest.raises(sf.ApiRefused):
            sf.pinned_get(meta)  # a binding for another URL never widens this one
    with ap.purpose_scope("metadata", meta):
        with pytest.raises(sf.PinnedRefused) as ei:
            sf.pinned_get(meta)
        assert not isinstance(ei.value, sf.ApiRefused)


def test_limited_transport_strict_keeps_oa1_fail_closed():
    get = ap.limited_transport(lambda url, **kw: pytest.fail("sent"), strict=True)
    with pytest.raises(rate_limit.UnlistedHost):
        get("https://unlisted-host.example/p")


# ---- 1108C (review finding 2): wider API-like shapes, owner disposition H2
NEW_API_SHAPES = [
    ("https://example.org/api/v1beta1/models", "api_versioned"),
    ("https://example.org/api/v2beta/items", "api_versioned"),
    ("https://en.wikipedia.org/api/rest_v1/page/summary/Epidemic", "api_rest_versioned"),
    ("https://en.wikipedia.org/w/api.php/extra/path?x=1", "mediawiki_api"),
    ("https://en.wikipedia.org/w/rest.php/v1/page/Epidemic", "mediawiki_rest"),
    ("https://example.org/s?q=x&format=jsonp", "format_json"),
    ("https://example.org/s?q=x&output=jsonp", "output_json"),
    ("https://example.org/select?q=x&wt=json.nl", "solr_json"),
]
ORDINARY_WEB = [
    "https://example.org/apis-and-you", "https://example.org/rapid/v1beta",
    "https://example.org/s?q=x&format=html", "https://example.org/page?output=pdf",
    "https://en.wikipedia.org/wiki/Epidemic", "https://en.wikipedia.org/wiki/API",
    "https://en.wikipedia.org/wiki/Api.php", "https://en.wikipedia.org/wiki/Special:Search?search=json",
]


@pytest.mark.parametrize("url,pid", NEW_API_SHAPES)
def test_1108c_new_shapes_are_api_like(url, pid):
    assert ap.api_like(url) == pid


@pytest.mark.parametrize("url", ORDINARY_WEB)
def test_1108c_ordinary_web_urls_are_not_api_like(url):
    assert ap.api_like(url) is None
    assert _refused(url) is None
