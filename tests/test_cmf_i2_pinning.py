"""CMF-1022-I2: preventive IP pinning (E1/A4 DNS rebinding) and NEW-2 (IPv4-compatible / NAT64 embedded IPv4).

OFFLINE. socket.getaddrinfo is replaced by a scripted resolver (per host, a queue of answers; it raises for any host it
does not know). urllib3's socket factory (urllib3.util.connection.create_connection) is replaced by a recorder that
returns an in-memory fake socket, and urllib3's TLS wrap (connection._ssl_wrap_socket_and_match_hostname) by a
recorder of its TLS arguments. The REAL requests Session, HTTPAdapter, PoolManager, connection pool and
HTTPConnection code run in between, so the recorded (host, port) is the actual address the socket connect is called
with, and the recorded server_hostname / assert_hostname / cert_reqs are what urllib3 would hand to the TLS layer.
No DNS, no network, no DB. The TLS-recorder test needs urllib3 2.x (the private function name); it is skipped otherwise.
"""
import io
import socket
import ssl
import time
from types import SimpleNamespace

import pytest
import requests
import urllib3

from tests import acq_support as A
from vera import auto_acquire, rate_limit, search_providers, selective_fetch, source_resolver

PUB4 = "93.184.216.34"
PUB4B = "93.184.216.35"
PUB6 = "2606:2800:220:1:248:1893:25c8:1946"
URLLIB3_2 = int(urllib3.__version__.split(".")[0]) >= 2


def _ok_body(body=b"<html><meta name='citation_doi' content='10.1234/abc'>ok</html>", ctype="text/html",
             status="200 OK", extra=b""):
    return (f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n".encode() + extra
            + b"Connection: close\r\n\r\n" + body)


class _FakeSock:
    def __init__(self, net, host, port):
        self.net, self.dest, self.sent = net, (host, port), b""

    def sendall(self, data):
        self.sent += data

    def makefile(self, mode="rb", *a, **k):
        head = self.sent.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
        path = head[0].split(" ")[1]
        hdr = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in head[1:] if ":" in ln}
        self.net.requests.append({"dest": self.dest, "host": hdr.get("host"), "path": path})
        return io.BytesIO(self.net.routes.get((hdr.get("host"), path), _ok_body(b"", status="404 Not Found")))

    def settimeout(self, *_):
        pass

    def setsockopt(self, *_):
        pass

    def gettimeout(self):
        return 5.0

    def close(self):
        pass


@pytest.fixture
def net(monkeypatch):
    """Scripted DNS + recorded socket connects + recorded TLS parameters."""
    st = SimpleNamespace(answers={}, dns_calls=[], connects=[], tls=[], requests=[], routes={})

    def fake_gai(host, port, *a, **k):
        st.dns_calls.append(host)
        q = st.answers.get(host)
        if not q:
            raise socket.gaierror(f"unknown test host {host!r}")
        ips = q.pop(0) if len(q) > 1 else q[0]
        return [((socket.AF_INET6 if ":" in ip else socket.AF_INET), socket.SOCK_STREAM, 6, "", (ip, port or 0))
                for ip in ips]
    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)

    def fake_create_connection(address, *a, **k):
        st.connects.append((address[0].strip("[]"), address[1]))
        return _FakeSock(st, address[0].strip("[]"), address[1])
    monkeypatch.setattr(urllib3.util.connection, "create_connection", fake_create_connection)

    if URLLIB3_2:
        import urllib3.connection as uc

        def fake_wrap(sock, **kw):
            st.tls.append(kw)
            return uc._WrappedAndVerifiedSocket(socket=sock, is_verified=True)
        monkeypatch.setattr(uc, "_ssl_wrap_socket_and_match_hostname", fake_wrap)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    return st


def _fetch(url):
    return source_resolver.fetch_html(url, pacer=selective_fetch.HostPacer({}))  # DEFAULT transport (pinned_get)


needs_u3 = pytest.mark.skipif(not URLLIB3_2, reason="TLS recorder targets urllib3 2.x internals")


# ------------------------------------------------------------------------------------------------ rebinding
@needs_u3
@pytest.mark.parametrize("second", ["127.0.0.1", "169.254.169.254"])
def test_i2_rebind_fetch_html_connects_to_first_validated_ip_and_never_resolves_again(net, second):
    net.answers["rebind.example.org"] = [[PUB4], [second], [second]]
    net.routes[("rebind.example.org", "/a")] = _ok_body()
    body, final, err = _fetch("https://rebind.example.org/a")
    assert err is None and "ok" in body and final == "https://rebind.example.org/a"
    assert net.connects == [(PUB4, 443)]                       # the socket went to the validated IP
    assert net.dns_calls == ["rebind.example.org"]             # resolved ONCE; urllib3 never resolved
    assert net.requests[0]["host"] == "rebind.example.org"


@needs_u3
@pytest.mark.parametrize("second", ["127.0.0.1", "169.254.169.254"])
def test_i2_rebind_landing_doi_connects_to_first_validated_ip_and_never_resolves_again(net, second):
    net.answers["rebind.example.org"] = [[PUB4], [second]]
    net.routes[("rebind.example.org", "/a")] = _ok_body()
    assert search_providers.landing_doi("https://rebind.example.org/a", 5.0, 100_000) == "10.1234/abc"
    assert net.connects == [(PUB4, 443)] and net.dns_calls == ["rebind.example.org"]


@needs_u3
def test_i2_rebind_guarded_get_connects_to_first_validated_ip(net):
    net.answers["www.ebi.ac.uk"] = [["193.62.193.80"], ["127.0.0.1"]]
    net.routes[("www.ebi.ac.uk", "/x.xml")] = _ok_body(b"<article/>", ctype="application/xml")
    deps = SimpleNamespace(http_get=selective_fetch.pinned_get, url_check=selective_fetch.is_public_url)
    raw, final = auto_acquire._guarded_get("https://www.ebi.ac.uk/x.xml", A.cfg(), deps, 100_000, 5.0, False)
    assert raw == b"<article/>" and net.connects == [("193.62.193.80", 443)]
    assert net.dns_calls == ["www.ebi.ac.uk"]


def test_i2_default_deps_transport_is_the_pinned_getter():
    import inspect
    src = inspect.getsource(auto_acquire.default_deps)
    assert "limited_by_url(selective_fetch.pinned_get)" in src and "requests.get" not in src


def test_i2_default_transports_are_pinned():
    import inspect
    assert "pinned_get" in inspect.getsource(search_providers.landing_doi)
    assert "http_get or pinned_get" in inspect.getsource(selective_fetch.fetch_html)
    assert "http_get or pinned_get" in inspect.getsource(selective_fetch.fetch_candidate)


# ------------------------------------------------------------------------------------------------ multi-address
def test_i2_multi_address_answer_with_one_private_is_refused_before_any_connect(net):
    net.answers["mixed.example.org"] = [[PUB4, "10.0.0.5"]]
    body, final, err = _fetch("https://mixed.example.org/a")
    assert body is None and err.startswith("blocked: blocked non-public address 10.0.0.5")
    assert net.connects == []
    with pytest.raises(selective_fetch.PinnedRefused):
        selective_fetch.pinned_get("https://mixed.example.org/a")
    assert net.connects == []


@needs_u3
def test_i2_multi_address_all_public_connects_to_first_and_falls_back_only_within_validated_set(net, monkeypatch):
    net.answers["two.example.org"] = [[PUB4, PUB4B], ["127.0.0.1"]]
    net.routes[("two.example.org", "/a")] = _ok_body()
    real = urllib3.util.connection.create_connection

    def first_fails(address, *a, **k):
        if address[0] == PUB4:
            net.connects.append((PUB4, address[1]))
            raise ConnectionRefusedError("refused")
        return real(address, *a, **k)
    monkeypatch.setattr(urllib3.util.connection, "create_connection", first_fails)
    body, _final, err = _fetch("https://two.example.org/a")
    assert err is None and net.connects == [(PUB4, 443), (PUB4B, 443)]
    assert net.dns_calls == ["two.example.org"]


# ------------------------------------------------------------------------------------------------ TLS / Host / verify
@needs_u3
def test_i2_sni_host_header_and_certificate_verification_use_the_hostname(net):
    net.answers["tls.example.org"] = [[PUB4]]
    net.routes[("tls.example.org", "/p?q=1")] = _ok_body()
    body, _f, err = _fetch("https://tls.example.org/p?q=1")
    assert err is None
    (kw,) = net.tls
    assert kw["server_hostname"] == "tls.example.org"          # SNI
    assert kw["assert_hostname"] == "tls.example.org"          # certificate hostname check
    from urllib3.util.ssl_ import resolve_cert_reqs
    assert resolve_cert_reqs(kw["cert_reqs"]) == ssl.CERT_REQUIRED   # verification stays on
    assert kw["assert_fingerprint"] is None
    assert net.requests == [{"dest": (PUB4, 443), "host": "tls.example.org", "path": "/p?q=1"}]


@needs_u3
def test_i2_pinned_session_passes_verify_true_and_ignores_env(net, monkeypatch):
    seen = []
    real_send = requests.Session.send

    def spy(self, request, **kw):
        seen.append((self.trust_env, kw.get("verify"), kw.get("proxies")))
        return real_send(self, request, **kw)
    monkeypatch.setattr(requests.Session, "send", spy)
    net.answers["v.example.org"] = [[PUB4]]
    net.routes[("v.example.org", "/")] = _ok_body()
    assert _fetch("https://v.example.org/")[2] is None
    assert seen and all(t is False and v is True and not p for t, v, p in seen)


@needs_u3
def test_i2_proxy_env_cannot_bypass_pinning(net, monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(var, "http://10.9.9.9:3128")
    net.answers["px.example.org"] = [[PUB4]]
    net.routes[("px.example.org", "/")] = _ok_body()
    net.routes[("px.example.org", "/h")] = _ok_body()
    assert _fetch("https://px.example.org/")[2] is None
    assert _fetch("http://px.example.org/h")[2] is None
    assert net.connects == [(PUB4, 443), (PUB4, 80)]           # never the proxy address
    assert "10.9.9.9" not in net.dns_calls


def test_i2_pinned_get_refuses_to_follow_redirects():
    with pytest.raises(ValueError):
        selective_fetch.pinned_get("https://x.example.org/", allow_redirects=True)


# ------------------------------------------------------------------------------------------------ redirects
@needs_u3
def test_i2_redirect_hop_is_revalidated_and_repinned(net):
    net.answers["a.example.org"] = [[PUB4], ["127.0.0.1"]]
    net.answers["b.example.org"] = [[PUB4B], ["169.254.169.254"]]
    net.routes[("a.example.org", "/x")] = _ok_body(b"", status="302 Found", extra=b"Location: https://b.example.org/y\r\n")
    net.routes[("b.example.org", "/y")] = _ok_body()
    body, final, err = _fetch("https://a.example.org/x")
    assert err is None and final == "https://b.example.org/y"
    assert net.connects == [(PUB4, 443), (PUB4B, 443)]
    assert net.dns_calls == ["a.example.org", "b.example.org"]
    assert [k["server_hostname"] for k in net.tls] == ["a.example.org", "b.example.org"]
    assert [r["host"] for r in net.requests] == ["a.example.org", "b.example.org"]


@needs_u3
@pytest.mark.parametrize("private", ["127.0.0.1", "169.254.169.254", "10.1.2.3", "::1", "64:ff9b::7f00:1"])
def test_i2_redirect_to_host_resolving_private_is_refused(net, private):
    net.answers["a.example.org"] = [[PUB4]]
    net.answers["evil.example.net"] = [[private]]
    net.routes[("a.example.org", "/x")] = _ok_body(b"", status="302 Found",
                                                   extra=b"Location: https://evil.example.net/meta\r\n")
    body, final, err = _fetch("https://a.example.org/x")
    assert body is None and err.startswith("blocked:")
    assert net.connects == [(PUB4, 443)]
    net.connects.clear()
    assert search_providers.landing_doi("https://a.example.org/x", 5.0, 100_000) is None
    assert net.connects == [(PUB4, 443)]


# ------------------------------------------------------------------------------------------------ IPv6
@needs_u3
def test_i2_ipv6_answer_is_pinned_and_redirect_hop_too(net):
    net.answers["v6.example.org"] = [[PUB6], ["::1"]]
    net.answers["v4.example.org"] = [[PUB4]]
    net.routes[("v6.example.org", "/a")] = _ok_body(b"", status="301 Moved",
                                                    extra=b"Location: https://v4.example.org/b\r\n")
    net.routes[("v4.example.org", "/b")] = _ok_body()
    body, final, err = _fetch("https://v6.example.org/a")
    assert err is None
    assert net.connects == [(PUB6, 443), (PUB4, 443)]
    assert net.tls[0]["server_hostname"] == "v6.example.org" and net.requests[0]["host"] == "v6.example.org"
    assert net.dns_calls == ["v6.example.org", "v4.example.org"]


def test_i2_ordinary_global_ipv6_still_passes(net):
    net.answers["g6.example.org"] = [[PUB6, "2001:4860:4860::8888"]]
    assert selective_fetch.is_public_url("https://g6.example.org/") == (True, "")


# ------------------------------------------------------------------------------------------------ NEW-2
NEW2_REFUSED = ["::127.0.0.1", "::7f00:1", "::a9fe:a9fe", "::a00:1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe",
                "64:ff9b::a00:5", "64:ff9b:1::7f00:1", "64:ff9b:1::a9fe:a9fe", "64:ff9b:1::5db8:d822",
                "64:ff9b:1:ffff::1"]


@pytest.mark.parametrize("addr", NEW2_REFUSED)
def test_i2_new2_embedded_non_public_ipv4_refused_from_dns(net, addr):
    net.answers["n.example.org"] = [[addr]]
    ok, why = selective_fetch.is_public_url("https://n.example.org/")
    assert ok is False and "non-public" in why
    body, final, err = _fetch("https://n.example.org/")
    assert body is None and err.startswith("blocked:") and net.connects == []


@pytest.mark.parametrize("addr", ["::127.0.0.1", "::a9fe:a9fe", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe",
                                  "64:ff9b:1::7f00:1"])
def test_i2_new2_embedded_non_public_ipv4_refused_as_literal(monkeypatch, addr):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda h, p, *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (h, p, 0, 0))])
    assert selective_fetch.is_public_url(f"https://[{addr}]/")[0] is False


@pytest.mark.parametrize("addr", ["64:ff9b::5db8:d822", "::5db8:d822"])
def test_i2_new2_public_embedded_ipv4_is_allowed_by_design(net, addr):
    """Design choice: NAT64 well-known-prefix and IPv4-compatible addresses are judged by the EMBEDDED IPv4; a public
    embedded address (93.184.216.34) passes, because DNS64 legitimately synthesizes 64:ff9b::<public v4>. The local-use
    prefix 64:ff9b:1::/48 is always refused (see NEW2_REFUSED)."""
    net.answers["p.example.org"] = [[addr]]
    assert selective_fetch.is_public_url("https://p.example.org/") == (True, "")


def test_i2_new2_ip_helper_matches_policy():
    import ipaddress as ia
    f = selective_fetch._ip_is_public
    assert f(ia.ip_address(PUB4)) and f(ia.ip_address(PUB6)) and f(ia.ip_address("64:ff9b::808:808"))
    for a in NEW2_REFUSED + ["::ffff:7f00:1", "::ffff:a9fe:a9fe", "127.0.0.1", "169.254.169.254", "fe80::1"]:
        assert not f(ia.ip_address(a)), a


# ------------------------------------------------------------------------------------------------ pin hygiene
@needs_u3
def test_i2_pin_is_single_use_and_bound_to_the_exact_url(net):
    net.answers["s.example.org"] = [[PUB4], [PUB4B]]
    net.routes[("s.example.org", "/a")] = _ok_body()
    net.routes[("s.example.org", "/b")] = _ok_body()
    assert selective_fetch.is_public_url("https://s.example.org/a") == (True, "")
    selective_fetch.pinned_get("https://s.example.org/b").close()   # different URL: validates itself (2nd answer)
    assert net.connects == [(PUB4B, 443)] and net.dns_calls == ["s.example.org", "s.example.org"]


@needs_u3
def test_i2_stale_pin_is_not_used(net, monkeypatch):
    net.answers["t.example.org"] = [[PUB4], ["127.0.0.1"]]
    assert selective_fetch.is_public_url("https://t.example.org/") == (True, "")
    real = time.monotonic
    monkeypatch.setattr(selective_fetch.time, "monotonic", lambda: real() + 3600)
    with pytest.raises(selective_fetch.PinnedRefused):
        selective_fetch.pinned_get("https://t.example.org/")   # re-validated; the rebound answer is refused
    assert net.connects == []


def test_i2_rate_limited_wrapper_keeps_the_original_url():
    seen = []
    w = rate_limit.limited_by_url(lambda url, **kw: seen.append(url))
    w("https://unlisted.example.org/x", headers={})
    assert seen == ["https://unlisted.example.org/x"]
