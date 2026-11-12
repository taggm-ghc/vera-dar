"""DENIAL-COVERAGE-1111 (OD-1111-ACCESS-DENIED-NO-BYPASS): one guard in the shared transports. Offline: the real
selective_fetch.pinned_get / public_fetch.default_transport run with requests.Session.get patched; requests per
(host, path) are counted. No socket is opened."""
import socket

import pytest
import requests

from vera import acq_discovered as ad
from vera import public_fetch as pf
from vera import selective_fetch as sf

PUB_IP = "93.184.216.34"


class Resp:
    def __init__(self, status=200, headers=None, body=b"<html><p>hi</p></html>"):
        self.status_code = status
        self.headers = headers if headers is not None else {"Content-Type": "text/html"}
        self.body = body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class Net:
    """requests.Session.get stand-in: `script` maps (host, path) -> list of Resp (last one repeats)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, session, url, headers=None, **kw):
        host = (headers or {}).get("Host", "?")
        path = url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else ""
        key = (host, "/" + path.split("?", 1)[0])
        self.calls.append(key)
        seq = self.script.get(key) or [Resp(200)]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def count(self, host, path):
        return self.calls.count((host, path))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ad.reset_for_tests()
    with sf._ROBOTS_LOCK:
        sf._ROBOTS.clear()
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, 0, 0, "", (PUB_IP, p))])
    yield
    ad.reset_for_tests()


def net(monkeypatch, script):
    n = Net(script)
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: n.get(self, url, **kw))
    return n


H = "pub.example"


def test_t1_second_get_of_a_403_document_is_refused_unsent(monkeypatch):
    n = net(monkeypatch, {(H, "/a"): [Resp(403)]})
    assert sf.pinned_get(f"https://{H}/a").status_code == 403
    with pytest.raises(sf.DocumentDenied):
        sf.pinned_get(f"https://{H}/a")
    assert n.count(H, "/a") == 1
    assert issubclass(sf.DocumentDenied, sf.PinnedRefused)


@pytest.mark.parametrize("status,remembered", [(401, True), (403, True), (404, False), (429, False), (500, False)])
def test_t2_only_401_403_are_remembered(monkeypatch, status, remembered):
    n = net(monkeypatch, {(H, "/a"): [Resp(status)]})
    sf.pinned_get(f"https://{H}/a")
    if remembered:
        with pytest.raises(sf.DocumentDenied):
            sf.pinned_get(f"https://{H}/a")
        assert n.count(H, "/a") == 1
    else:
        sf.pinned_get(f"https://{H}/a")
        assert n.count(H, "/a") == 2


@pytest.mark.parametrize("path", ["/robots.txt", "/.well-known/tdmrep.json"])
def test_t3_robots_endpoint_403_is_not_remembered(monkeypatch, path):
    n = net(monkeypatch, {(H, path): [Resp(403)]})
    sf.pinned_get(f"https://{H}{path}")
    sf.pinned_get(f"https://{H}{path}")
    assert n.count(H, path) == 2
    assert not ad.is_denied(urls=(f"https://{H}{path}",))


@pytest.mark.parametrize("path", ["/robots.txt.html", "/a/robots.txt", "/.well-known/tdmrep.json.bak",
                                  "/x/.well-known/tdmrep.json"])
def test_t4_lookalike_paths_are_document_requests(monkeypatch, path):
    n = net(monkeypatch, {(H, path): [Resp(403)]})
    sf.pinned_get(f"https://{H}{path}?x=1")
    with pytest.raises(sf.DocumentDenied):
        sf.pinned_get(f"https://{H}{path}")
    assert n.count(H, path) == 1


def test_t5_metadata_api_403_not_remembered_fulltext_api_403_remembered(monkeypatch):
    u = "https://api.openalex.org/works/W1"
    n = net(monkeypatch, {("api.openalex.org", "/works/W1"): [Resp(403)]})
    sf.pinned_get(u, api_purpose="metadata")
    sf.pinned_get(u, api_purpose="metadata")
    assert n.count("api.openalex.org", "/works/W1") == 2 and not ad.is_denied(urls=(u,))
    # Europe PMC fullTextXML is a download endpoint: fulltext purpose, and the guarded default (none), are documents
    e = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"
    n2 = net(monkeypatch, {("www.ebi.ac.uk", "/europepmc/webservices/rest/PMC1/fullTextXML"): [Resp(403)]})
    sf.pinned_get(e, api_purpose="fulltext")
    assert ad.is_denied(urls=(e,))
    for purpose in ("fulltext", None):
        with pytest.raises(sf.DocumentDenied):
            sf.pinned_get(e, api_purpose=purpose)
    assert n2.count("www.ebi.ac.uk", "/europepmc/webservices/rest/PMC1/fullTextXML") == 1


def test_t6_denial_key_is_host_and_path(monkeypatch):
    n = net(monkeypatch, {(H, "/a"): [Resp(403)]})
    sf.pinned_get(f"https://{H}/a?x=1")
    with pytest.raises(sf.DocumentDenied):
        sf.pinned_get(f"https://{H}/a?x=2")
    sf.pinned_get(f"https://{H}/b")
    assert n.count(H, "/a") == 1 and n.count(H, "/b") == 1


def test_t7_redirect_to_a_denied_target_is_never_requested(monkeypatch):
    ad.deny(ad.denial_ids(urls=(f"https://{H}/secret",)))
    redirect = Resp(302, {"Location": f"https://{H}/secret"})
    n = net(monkeypatch, {("src.example", "/go"): [redirect], (H, "/robots.txt"): [Resp(404)],
                          ("src.example", "/robots.txt"): [Resp(404)]})
    body, _final, err = sf.fetch_html("https://src.example/go", pacer=sf.HostPacer({}))
    assert body is None and err == "blocked: access_denied"
    assert n.count(H, "/secret") == 0
    # public_fetch web route: the hop check refuses before the transport (guard never reached)
    sent = []
    cfg = {"user_agent": "UA", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 5000,
           "max_decompressed_bytes": 20000, "max_robots_bytes": 2000, "timeout_s": 5, "content_types": ["text/html"]}

    def transport(host, ip, port, target, headers, timeout):
        sent.append((host, target))
        if target == "/robots.txt":
            return Resp(404)
        return Resp(302, {"Location": f"https://{H}/secret"})
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch("https://src.example/go", cfg, resolve=lambda h, p: [(socket.AF_INET, 0, 0, "", (PUB_IP, p))],
                 transport=transport)
    assert ei.value.reason == "access_denied" and (H, "/secret") not in sent


def test_t8_callers_first_line_unchanged_reason_code(monkeypatch):
    """public_fetch._get refuses on its own memory check before the transport: transport count 0, reason unchanged."""
    ad.deny(ad.denial_ids(urls=(f"https://{H}/p",)))
    sent = []
    cfg = {"user_agent": "UA", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 5000,
           "max_decompressed_bytes": 20000, "max_robots_bytes": 2000, "timeout_s": 5, "content_types": ["text/html"]}
    with pytest.raises(pf.FetchRefused) as ei:
        pf.fetch(f"https://{H}/p", cfg, resolve=lambda h, p: [(socket.AF_INET, 0, 0, "", (PUB_IP, p))],
                 transport=lambda *a: sent.append(a))
    assert ei.value.reason == "access_denied" and not sent


def test_t9_d5_fetch_html_and_d6_fetch_candidate_do_not_rerequest(monkeypatch):
    n = net(monkeypatch, {(H, "/a"): [Resp(403)], (H, "/robots.txt"): [Resp(404)]})
    body, _f, err = sf.fetch_html(f"https://{H}/a", pacer=sf.HostPacer({}))
    assert body is None and err == "http 403"
    body, _f, err = sf.fetch_html(f"https://{H}/a", pacer=sf.HostPacer({}))
    assert body is None and err == "blocked: access_denied"
    res = sf.fetch_candidate(f"https://{H}/a", pacer=sf.HostPacer({}))
    assert "access_denied" in str(getattr(res, "error", "") or res)
    assert n.count(H, "/a") == 1


def test_t10_d7_d8_shaped_callers_and_public_fetch_rebuilt_url(monkeypatch):
    # D7/D8 call pinned_get directly (http_get=pinned_get, allow_redirects=False, stream=True)
    n = net(monkeypatch, {(H, "/wiki/X"): [Resp(403)]})
    kw = dict(headers={"User-Agent": "UA"}, timeout=5, allow_redirects=False, stream=True)
    sf.pinned_get(f"https://{H}/wiki/X", **kw)
    with pytest.raises(sf.PinnedRefused):
        sf.pinned_get(f"https://{H}/wiki/X", **kw)
    assert n.count(H, "/wiki/X") == 1
    # D1 transport: URL rebuilt from host, port and target (query dropped from the key; non-443 port kept in the URL)
    n2 = net(monkeypatch, {(f"{H}:8443", "/q"): [Resp(403)]})
    monkeypatch.setattr(pf, "register_abort", lambda *_a, **_k: None)
    r = pf.default_transport(H, PUB_IP, 8443, "/q?z=1", {"Host": f"{H}:8443"}, 5)
    assert r.status_code == 403 and ad.is_denied(urls=("https://pub.example:8443/q?z=2",))
    with pytest.raises(sf.DocumentDenied):
        pf.default_transport(H, PUB_IP, 8443, "/q?z=3", {"Host": f"{H}:8443"}, 5)
    assert n2.count(f"{H}:8443", "/q") == 1


def test_wiki_ref_portal_get_article_maps_the_refusal_to_unreachable(monkeypatch):
    from vera import wiki_ref_portal as w
    n = net(monkeypatch, {("en.wikipedia.org", "/wiki/X"): [Resp(403)]})
    cfg = {"article_host": "en.wikipedia.org", "allowed_hosts": ["en.wikipedia.org"], "timeout_s": 5,
           "max_response_bytes": 1000, "user_agent": "UA"}
    args = dict(http_get=sf.pinned_get, url_check=lambda u: True, clock=lambda: 0.0, deadline=None)
    for _ in range(2):
        with pytest.raises(w.PortalError) as ei:
            w.get_article("https://en.wikipedia.org/wiki/X", cfg, **args)
        assert ei.value.args[0] == "unreachable" or str(ei.value) == "unreachable"
    assert n.count("en.wikipedia.org", "/wiki/X") == 1


JATS = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9/fullTextXML"
JATS_KEY = ("www.ebi.ac.uk", "/europepmc/webservices/rest/PMC9/fullTextXML")


def test_defect1_fulltext_endpoint_is_never_exempt_whatever_purpose_is_declared(monkeypatch):
    from vera import api_permissions
    n = net(monkeypatch, {JATS_KEY: [Resp(200)]})
    ad.deny(ad.denial_ids(urls=(JATS,)))
    with api_permissions.holding("europepmc"):
        for purpose in ("fulltext", "metadata", None):
            with pytest.raises(sf.DocumentDenied):
                sf.pinned_get(JATS, api_purpose=purpose)
        with api_permissions.purpose_scope("metadata", JATS):
            with pytest.raises(sf.DocumentDenied):
                sf.pinned_get(JATS)
    assert n.count(*JATS_KEY) == 0


def test_defect1_a_metadata_endpoint_of_the_same_api_is_still_exempt(monkeypatch):
    from vera import api_permissions
    s = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    n = net(monkeypatch, {("www.ebi.ac.uk", "/europepmc/webservices/rest/search"): [Resp(403)]})
    with api_permissions.holding("europepmc"):
        sf.pinned_get(s, api_purpose="metadata")
        sf.pinned_get(s, api_purpose="metadata")
    assert n.count("www.ebi.ac.uk", "/europepmc/webservices/rest/search") == 2 and not ad.is_denied(urls=(s,))


@pytest.mark.parametrize("variant", [
    f"https://{H}./doc/a", f"https://{H.upper()}/doc/a", f"https://{H}/doc/%61", f"https://{H}/doc/%61#f",
    f"https://{H}//doc/a", f"https://{H}///doc/a", f"https://{H}/doc/./a", f"https://{H}/doc/x/../a",
    f"https://{H}:443/doc/a", f"https://{H}/doc/a?q=1",
])
def test_defect2_denial_key_is_normalised_at_lookup(monkeypatch, variant):
    n = net(monkeypatch, {(H, "/doc/a"): [Resp(403)]})
    sf.pinned_get(f"https://{H}/doc/a")
    with pytest.raises(sf.DocumentDenied):
        sf.pinned_get(variant)
    assert n.count(H, "/doc/a") == 1 and sum(1 for h, _p in n.calls if h == H) == 1


@pytest.mark.parametrize("recorded", [f"https://{H}./doc/a", f"https://{H}/doc/%61", f"https://{H}//doc/a",
                                      f"https://{H}/doc/./a"])
def test_defect2_denial_key_is_normalised_at_record_time(monkeypatch, recorded):
    ad.deny(ad.denial_ids(urls=(recorded,)))
    assert ad.is_denied(urls=(f"https://{H}/doc/a",))
    assert not ad.is_denied(urls=(f"https://{H}/doc/b",)) and not ad.is_denied(urls=(f"https://{H}/doc/%2Fa",))


def test_defect2_reserved_escapes_stay_distinct():
    assert ad._id_url(f"https://{H}/a%2Fb") != ad._id_url(f"https://{H}/a/b")
    assert ad._id_url(f"https://{H}/a%2fb") == ad._id_url(f"https://{H}/a%2Fb")
    assert ad._id_url(f"https://{H}") == ad._id_url(f"https://{H}/")


def test_d8_reference_portal_fetch_page_does_not_rerequest_a_403(monkeypatch):
    import json
    from pathlib import Path

    from vera import reference_portal as rp
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "reference_portals.json").read_text())
    url = "https://en.wikipedia.org/wiki/Denied_Page"
    n = net(monkeypatch, {("en.wikipedia.org", "/wiki/Denied_Page"): [Resp(403)]})
    deps = rp.Deps(http_get=sf.pinned_get, url_check=lambda u: (True, ""), robots_ok=lambda u, ua: True,
                   no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True, crawl_ok=lambda u, ua: True,
                   limited=lambda portal, fn: fn)
    for _ in range(2):
        with pytest.raises(rp.Skip) as ei:
            rp.fetch_page(url, {"rate_portal": "wikipedia"}, cfg, deps, lambda: 0.0, None)
        assert str(ei.value) in ("unreachable", ei.value.args[0] if ei.value.args else "unreachable")
    assert n.count("en.wikipedia.org", "/wiki/Denied_Page") == 1


def test_t13_real_pinned_get_doi_org_once_denied_endpoint_never(monkeypatch):
    import time

    from tests import acq_support as A
    from vera import auto_acquire
    n = net(monkeypatch, {("doi.org", "/10.9999/y"): [Resp(302, {"Location": "https://pub.example/a/z",
                                                                  "Content-Type": "text/html"})],
                          ("doi.org", "/robots.txt"): [Resp(404)], ("doi.org", "/.well-known/tdmrep.json"): [Resp(404)],
                          ("pub.example", "/robots.txt"): [Resp(404)]})
    ad.deny(ad.denial_ids(urls=("https://pub.example/a/z",)))
    A.seed_access("doi.org")
    A.seed_access("pub.example")
    deps = A.deps([], sf.pinned_get)
    deps.url_check = sf.is_public_url
    with pytest.raises(auto_acquire._Stop) as e:
        ad._b_get("https://doi.org/10.9999/y", {"doi.org", "pub.example"}, {"text/html"}, 10000, 2, deps, A.cfg(),
                  time.monotonic, None, [])
    assert e.value.code == "R-ACQ-ACCESS_DENIED"
    assert n.count("doi.org", "/10.9999/y") == 1 and n.count("pub.example", "/a/z") == 0
