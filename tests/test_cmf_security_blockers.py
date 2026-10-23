"""CMF-1022-T4 RED tests for the three ACTIVE security defects (owner decision OD-CMF-1022c): L1, 2b, 3b.

Defect tests were xfail(strict=True); markers removed when the I1 fixes landed.
Preservation guards are plain asserts and must pass before and after the fix. Offline: socket.getaddrinfo is
monkeypatched to a fixed public/private answer; HTTP goes through stubs only. No DNS, no network, no DB.
"""
import socket

import pytest

from tests import acq_support as A
from vera import acq_config, auto_acquire, inline_acquire, search_providers, selective_fetch, source_resolver
from vera import fulltext_loader as fl


@pytest.fixture(autouse=True)
def _ddgfix_layers_off(monkeypatch):
    """DDGFIX-1023 DF-8: landing_doi now also checks robots.txt before any GET.
    These tests target the SSRF / redirect / pinning layers beneath them, which are unchanged; the new layer is
    switched open here and are tested in tests/test_ddgflow_eval.py (test_df8_*)."""
    monkeypatch.setattr(search_providers, "_landing_robots", lambda *a, **k: (lambda u: True))

PUBLIC_IP = "93.184.216.34"  # documentation-grade public address; never contacted


# ------------------------------------------------------------------------------------------------ helpers
@pytest.fixture
def dns(monkeypatch):
    """getaddrinfo -> mapping host->ip (default public). Records (host, port) lookups. Raises on anything real."""
    table, calls = {}, []

    def fake(host, port, *a, **k):
        calls.append((host, port))
        ip = table.get(host, PUBLIC_IP)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    fake.table, fake.calls = table, calls
    return fake


class Hops:
    """http_get stub: url -> Resp (or a Location redirect); records every URL actually requested."""

    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def __call__(self, url, headers=None, timeout=None, allow_redirects=None, stream=None):
        self.urls.append(url)
        r = self.routes.get(url)
        if r is None:
            return A.Resp(status=404)
        if isinstance(r, str):
            return A.Resp(status=302, headers={"Location": r})
        return r


def _page(body=b"<html>ok</html>"):
    return A.Resp(body=body, ctype="text/html")


START = "https://journal.example.org/a"


def _fetch_with_redirect(target, dns_):
    g = Hops({START: target})
    body, final, err = source_resolver.fetch_html(START, http_get=g, pacer=selective_fetch.HostPacer({}))
    return g, body, err


def _landing(url_to_routes, start):
    g = Hops(url_to_routes)
    out = search_providers.landing_doi(start, 5.0, 100_000, get=g)
    return g, out


DOI_PAGE = A.Resp(body=b'<meta name="citation_doi" content="10.1234/x">', ctype="text/html")

BAD_PORT_URLS = ["https://host.example.org:22/x", "https://host.example.org:8080/x", "https://host.example.org:0/x",
                 "http://host.example.org:22/x", "http://host.example.org:8080/x"]
CRED_URLS = ["https://user@host.example.org/x", "https://user:pass@host.example.org/x", "https://@host.example.org/x",
             "http://user@host.example.org/x", "https://user:@host.example.org/x"]


# ------------------------------------------------------------------------------------------------ L1
def _no_licence_web():
    return A.epi_setup(jats_kw={"licence_url": None})  # JATS with no <license> element; index says "cc by"


# level: L1 unit seam
def test_L1_empty_licence_elements_is_held_not_admitted_on_index_licence():
    icfg = acq_config.ingest_cfg(fl.load_config(), A.cfg())
    with pytest.raises(fl.Refused) as ei:
        auto_acquire.licence_elements_decision([], "cc by", icfg)
    assert ei.value.code == icfg["reason_codes"]["licence_hold"]


# level: L1 inline entry point (inline_acquire.acquire -> evaluate_candidate)
def test_L1_inline_path_holds_jats_without_licence_element(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    acq = A.cfg()
    acq["inline"]["enabled"] = True
    web, _ = _no_licence_web()
    store = A.Store()
    d = A.deps([A.cand("10.9999/acq.epi4")], web, store=store)
    out = inline_acquire.acquire(A.Q_EPI, acq, env=A.ENV_ON, deps_factory=lambda a, e: d)
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    assert out.admitted == 0 and store.plans == [], out
    assert out.codes.get("R-LIC-HOLD") == 1, out.codes


# level: L1 background entry point (run_job -> evaluate_candidate)
def test_L1_background_run_job_holds_jats_without_licence_element():
    auto_acquire.reset_for_tests()
    web, _ = _no_licence_web()
    store = A.Store()
    out = auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                               clock=A.clock_at(), env=A.ENV_ON)
    auto_acquire.reset_for_tests()
    assert out["stored"] == 0 and store.plans == [], out
    assert out["codes"] == {"R-LIC-HOLD": 1}, out


# guard: a recognised embedded licence matching the index still admits (must pass before and after)
def test_L1_guard_matching_embedded_licence_still_admitted():
    auto_acquire.reset_for_tests()
    web, _ = A.epi_setup()
    store = A.Store()
    out = auto_acquire.run_job(A.job(), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                               clock=A.clock_at(), env=A.ENV_ON)
    auto_acquire.reset_for_tests()
    assert out["stored"] == 1


# ------------------------------------------------------------------------------------------------ 3b: ports
@pytest.mark.parametrize("url", BAD_PORT_URLS)
def test_3b_direct_non_standard_port_refused_by_is_public_url(dns, url):
    ok, why = selective_fetch.is_public_url(url)
    assert ok is False, url


@pytest.mark.parametrize("url", ["https://host.example.org:443/x", "http://host.example.org:80/x",
                                 "https://host.example.org/x", "http://host.example.org/x"])
def test_3b_guard_default_and_explicit_standard_ports_allowed(dns, url):
    assert selective_fetch.is_public_url(url) == (True, "")


@pytest.mark.parametrize("url", BAD_PORT_URLS[:3])
def test_3b_direct_bad_port_never_requested_by_fetch_html(dns, url):
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert body is None and err and g.urls == []


@pytest.mark.parametrize("url", BAD_PORT_URLS[:3])
def test_3b_direct_bad_port_never_requested_by_landing_doi(dns, url):
    g, out = _landing({url: DOI_PAGE}, url)
    assert out is None and g.urls == []


@pytest.mark.parametrize("target", ["https://redir.example.org:22/x", "https://redir.example.org:8080/x"])
def test_3b_fetch_html_redirect_to_bad_port_refused_before_followup_get(dns, target):
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err and err.startswith("blocked")


@pytest.mark.parametrize("target", ["https://redir.example.org:22/x", "https://redir.example.org:8080/x"])
def test_3b_landing_doi_redirect_to_bad_port_refused_before_followup_get(dns, target):
    g, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g.urls == [START] and out is None


def test_3b_fetch_html_redirect_chain_later_hop_changes_port(dns):
    h2, h3 = "https://b.example.org/y", "https://c.example.org:8080/z"
    g = Hops({START: h2, h2: h3, h3: _page()})
    body, final, err = source_resolver.fetch_html(START, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert g.urls == [START, h2] and body is None and err


def test_3b_landing_doi_redirect_chain_later_hop_changes_port(dns):
    h2, h3 = "https://b.example.org/y", "https://c.example.org:8080/z"
    g, out = _landing({START: h2, h2: h3, h3: DOI_PAGE}, START)
    assert g.urls == [START, h2] and out is None


def test_3b_guard_redirect_to_explicit_443_and_80_followed(dns):
    g = Hops({START: "https://b.example.org:443/y", "https://b.example.org:443/y": "http://c.example.org:80/z",
              "http://c.example.org:80/z": _page()})
    body, final, err = source_resolver.fetch_html(START, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert err is None and body and len(g.urls) == 3


def test_3b_guard_landing_doi_follows_standard_port_redirect(dns):
    g, out = _landing({START: "https://b.example.org:443/y", "https://b.example.org:443/y": DOI_PAGE}, START)
    assert out == "10.1234/x" and len(g.urls) == 2


# ------------------------------------------------------------------------------------------------ 2b: userinfo
@pytest.mark.parametrize("url", CRED_URLS)
def test_2b_direct_credential_url_refused_by_is_public_url(dns, url):
    ok, _ = selective_fetch.is_public_url(url)
    assert ok is False, url


@pytest.mark.parametrize("url", CRED_URLS[:3])
def test_2b_direct_credential_url_never_requested_by_fetch_html(dns, url):
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert body is None and err and g.urls == []


@pytest.mark.parametrize("url", CRED_URLS[:3])
def test_2b_direct_credential_url_never_requested_by_landing_doi(dns, url):
    g, out = _landing({url: DOI_PAGE}, url)
    assert out is None and g.urls == []


@pytest.mark.parametrize("target", ["https://user@redir.example.org/x", "https://user:pass@redir.example.org/x",
                                    "https://@redir.example.org/x"])
def test_2b_fetch_html_redirect_to_credential_url_refused_before_followup_get(dns, target):
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err and err.startswith("blocked")


@pytest.mark.parametrize("target", ["https://user@redir.example.org/x", "https://user:pass@redir.example.org/x",
                                    "https://@redir.example.org/x"])
def test_2b_landing_doi_redirect_to_credential_url_refused_before_followup_get(dns, target):
    g, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g.urls == [START] and out is None


# ------------------------------------------------------------------------------------------------ preservation guards
@pytest.mark.parametrize("ip", ["10.0.0.5", "192.168.1.1", "172.16.0.9", "127.0.0.1", "169.254.169.254", "169.254.1.1",
                                "0.0.0.0", "::1", "fe80::1", "fd00::1"])
def test_guard_non_public_addresses_blocked(dns, ip):
    dns.table["host.example.org"] = ip
    ok, why = selective_fetch.is_public_url("https://host.example.org/x")
    assert ok is False and "non-public" in why


def test_guard_metadata_ip_literal_blocked_and_never_requested(dns):
    url = "http://169.254.169.254/latest/meta-data/"
    dns.table["169.254.169.254"] = "169.254.169.254"
    assert selective_fetch.is_public_url(url)[0] is False
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert body is None and err.startswith("blocked") and g.urls == []


@pytest.mark.parametrize("target", ["http://169.254.169.254/latest/", "https://internal.example.org/x",
                                    "http://127.0.0.1/x"])
def test_guard_redirect_to_non_public_address_refused_before_followup_get(dns, target):
    dns.table["internal.example.org"] = "10.1.2.3"
    dns.table["169.254.169.254"] = "169.254.169.254"
    dns.table["127.0.0.1"] = "127.0.0.1"
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err.startswith("blocked")
    g2, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g2.urls == [START] and out is None


@pytest.mark.parametrize("url", ["ftp://host.example.org/x", "file:///etc/passwd", "gopher://host.example.org/x",
                                 "javascript:alert(1)", "data:text/html,x", "//host.example.org/x"])
def test_guard_non_http_schemes_refused(dns, url):
    assert selective_fetch.is_public_url(url)[0] is False
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert body is None and err and g.urls == []


def test_guard_redirect_to_non_http_scheme_refused(dns):
    g, body, err = _fetch_with_redirect("ftp://host.example.org/x", dns)
    assert g.urls == [START] and body is None and err.startswith("blocked")


def test_guard_fetch_html_redirect_cap_holds(dns):
    n = selective_fetch.MAX_REDIRECTS
    routes = {f"https://h{i}.example.org/": f"https://h{i + 1}.example.org/" for i in range(n + 3)}
    routes[START] = "https://h0.example.org/"
    g = Hops(routes)
    body, final, err = source_resolver.fetch_html(START, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert body is None and "too many redirects" in err and len(g.urls) == n + 1


def test_guard_landing_doi_redirect_cap_holds(dns):
    routes = {f"https://h{i}.example.org/": f"https://h{i + 1}.example.org/" for i in range(6)}
    routes[START] = "https://h0.example.org/"
    routes["https://h2.example.org/"] = DOI_PAGE  # beyond the 3-hop budget: must not be reached
    g, out = _landing(routes, START)
    assert out is None and len(g.urls) <= 3


# ------------------------------------------------------------------------------------------------ NEW-1: parser disagreement
PARSER_PROBES = ["https://a.com\\b.com/", "https://a.com\\@b.com/", "https://a.com%5Cb.com/", "https://a.com\\\\/",
                 "https://a.com\\/", "https://a.com\tb.com/", "https://a.com\nb.com/", "https://a.com b.com/",
                 "https://[::1]\\x.com/", "https://a.com\x00.b.com/"]


@pytest.mark.parametrize("url", PARSER_PROBES)
def test_new1_parser_disagreement_refused_by_is_public_url(dns, url):
    ok, why = selective_fetch.is_public_url(url)
    assert ok is False and why
    assert dns.calls == []  # refused before any name resolution


@pytest.mark.parametrize("url", PARSER_PROBES[:5] + PARSER_PROBES[8:9])
def test_new1_direct_probe_never_requested_by_fetch_html_or_landing_doi(dns, url):
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert g.urls == [] and body is None
    g2, out = _landing({url: DOI_PAGE}, url)
    assert out is None and g2.urls == []


@pytest.mark.parametrize("target", PARSER_PROBES[:5] + PARSER_PROBES[8:9])
def test_new1_fetch_html_redirect_to_probe_refused_before_followup_get(dns, target):
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err and err.startswith("blocked")


@pytest.mark.parametrize("target", PARSER_PROBES[:5] + PARSER_PROBES[8:9])
def test_new1_landing_doi_redirect_to_probe_refused_before_followup_get(dns, target):
    g, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g.urls == [START] and out is None


# guards: allowed. A backslash in the PATH (after the authority) is allowed: both parsers agree on the host and
# requests percent-encodes it, so the connection target cannot change.
@pytest.mark.parametrize("url", ["https://host.example.org/x", "https://HOST.example.org:443/x", "http://host.example.org:80/",
                                 "https://[2606:4700::1111]:443/x", "https://host.example.org/a\\b",
                                 "https://host.example.org/p?q=a\\b#f\\g"])
def test_new1_guard_normal_urls_and_path_backslash_allowed(dns, url):
    assert selective_fetch.is_public_url(url) == (True, "")


# ------------------------------------------------------------------------------------------------ scheme/port pairs (OD-CMF-1022g)
MISMATCH_URLS = ["http://host.example.org:443/x", "https://host.example.org:80/x"]


@pytest.mark.parametrize("url", MISMATCH_URLS)
def test_pair_mismatch_refused_by_is_public_url(dns, url):
    ok, why = selective_fetch.is_public_url(url)
    assert ok is False and why and dns.calls == []


@pytest.mark.parametrize("url", MISMATCH_URLS)
def test_pair_mismatch_never_requested_by_fetch_html_or_landing_doi(dns, url):
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert g.urls == [] and body is None
    g2, out = _landing({url: DOI_PAGE}, url)
    assert out is None and g2.urls == []


@pytest.mark.parametrize("target", MISMATCH_URLS)
def test_pair_mismatch_redirect_refused_before_followup_get(dns, target):
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err and err.startswith("blocked")
    g2, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g2.urls == [START] and out is None


@pytest.mark.parametrize("url", ["http://host.example.org/x", "http://host.example.org:80/x", "https://host.example.org/x",
                                 "https://host.example.org:443/x"])
def test_pair_matching_guard_allowed(dns, url):
    assert selective_fetch.is_public_url(url) == (True, "")


# ------------------------------------------------------------------------------------------------ encoded delimiters in the host
# Intentional policy (OD-CMF-1022g): no percent-encoded host syntax.
ENC = ["%40", "%3A", "%2F", "%5C", "%5c", "%2E", "%2e", "%00", "%0A", "%0a", "%09"]
ENC_URLS = [f"https://a.com{e}b.com/x" for e in ENC]


@pytest.mark.parametrize("url", ENC_URLS)
def test_encoded_host_delimiter_refused_by_is_public_url(dns, url):
    ok, why = selective_fetch.is_public_url(url)
    assert ok is False and why and dns.calls == []


@pytest.mark.parametrize("url", ENC_URLS)
def test_encoded_host_delimiter_never_requested_direct(dns, url):
    g = Hops({url: _page()})
    body, final, err = source_resolver.fetch_html(url, http_get=g, pacer=selective_fetch.HostPacer({}))
    assert g.urls == [] and body is None and dns.calls == []
    g2, out = _landing({url: DOI_PAGE}, url)
    assert out is None and g2.urls == [] and dns.calls == []


@pytest.mark.parametrize("target", ENC_URLS)
def test_encoded_host_delimiter_redirect_refused_before_followup_get_or_dns(dns, target):
    g, body, err = _fetch_with_redirect(target, dns)
    assert g.urls == [START] and body is None and err and err.startswith("blocked")
    assert not any("%" in h for h, _ in dns.calls)
    g2, out = _landing({START: target, target: DOI_PAGE}, START)
    assert g2.urls == [START] and out is None
    assert not any("%" in h for h, _ in dns.calls)
