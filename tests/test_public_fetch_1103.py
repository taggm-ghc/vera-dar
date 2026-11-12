"""NONASK-1103 S6: public-any fetch, offline. DNS and transport are fakes; no socket is ever opened."""
import gzip
import logging
import socket

import pytest

from tests import leak_sentinel as L
from vera import public_fetch as pf

CFG = {"user_agent": "UA-test", "deny_hosts": ["denied.example"], "max_redirects": 2, "max_compressed_bytes": 5000,
       "max_decompressed_bytes": 20000, "max_robots_bytes": 2000, "timeout_s": 5,
       "content_types": ["text/html", "application/xhtml+xml", "text/plain"]}


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class Net:
    def __init__(self, dns, pages):
        self.dns, self.pages, self.calls, self.lookups = dns, pages, [], []

    def resolve(self, host, port):
        self.lookups.append(host)
        ans = self.dns[host]
        ans = ans.pop(0) if isinstance(ans, list) and ans and isinstance(ans[0], list) else ans
        return [(socket.AF_INET, 0, 0, "", (a, port)) for a in ans]

    def transport(self, host, ip, port, target, headers, timeout):
        self.calls.append((host, ip, port, target, headers))
        r = self.pages[(host, target)]
        if isinstance(r, Exception):
            raise r
        return r


def html(body=b"<p>hello</p>"):
    return Resp(200, {"Content-Type": "text/html; charset=utf-8"}, body)


def robots(txt=b"User-agent: *\nAllow: /\n"):
    return Resp(200, {"Content-Type": "text/plain"}, txt)


def run(url, dns, pages, cfg=CFG):
    n = Net(dns, pages)
    return n, pf.fetch(url, cfg, resolve=n.resolve, transport=n.transport)


GOOD = {"a.example": ["93.184.216.34"]}
OKPAGES = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): html()}


def test_happy_path_pins_ip_and_uses_ua_identity_encoding():
    n, r = run("https://a.example/p", GOOD, OKPAGES)
    assert r["text"] == "hello"
    host, ip, port, target, hdrs = n.calls[-1]
    assert ip == "93.184.216.34" and port == 443 and hdrs["Accept-Encoding"] == "identity" and hdrs["User-Agent"] == "UA-test"
    assert [c[3] for c in n.calls] == ["/robots.txt", "/p"]  # robots over the same pinned transport


@pytest.mark.parametrize("url", [
    "ftp://a.example/p", "https://u:p@a.example/p", "https://a.example:8443/p", "http://a.example:443/p",
    "https://denied.example/p", "https://sub.denied.example/p", "https://a.example\\@evil/p", "https://a.example:99999/p"])
def test_url_rules_refused_before_dns(url):
    n = Net({}, {})
    with pytest.raises(pf.FetchRefused):
        pf.fetch(url, CFG, resolve=n.resolve, transport=n.transport)
    assert not n.lookups and not n.calls


@pytest.mark.parametrize("ip", ["10.0.0.1", "127.0.0.1", "169.254.169.254", "169.254.1.1", "100.64.0.1", "100.127.255.254",
                                "192.168.1.1", "172.16.0.5", "::1", "fd00:ec2::254", "fc00::1", "fe80::1",
                                "::ffff:10.0.0.1", "::ffff:8.8.8.8", "0.0.0.0"])
def test_blocked_addresses(ip):
    n = Net({"a.example": [ip]}, OKPAGES)
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport)
    assert e.value.reason == "blocked_address" and not n.calls


def test_any_blocked_answer_refuses_mixed_set():
    n = Net({"a.example": ["93.184.216.34", "10.0.0.1"]}, OKPAGES)
    with pytest.raises(pf.FetchRefused):
        pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport)


def test_http_is_upgraded_to_https_443():
    n, r = run("http://a.example/p", GOOD, OKPAGES)
    assert n.calls[-1][2] == 443 and r["url"].startswith("https://")


def test_redirect_rechecked_per_hop_to_private_refused():
    pages = {**OKPAGES, ("a.example", "/p"): Resp(302, {"Location": "https://b.example/x"})}
    n = Net({**GOOD, "b.example": ["10.1.1.1"]}, pages)
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport)
    assert e.value.reason == "blocked_address" and "b.example" in n.lookups


def test_redirect_to_deny_listed_and_robots_disallowed_and_cap():
    pages = {**OKPAGES, ("a.example", "/p"): Resp(302, {"Location": "https://denied.example/x"})}
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, pages)
    assert e.value.reason == "deny_listed"
    pages = {("a.example", "/robots.txt"): robots(b"User-agent: *\nDisallow: /q\n"),
             ("a.example", "/p"): Resp(301, {"Location": "/q"})}
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, pages)
    assert e.value.reason == "robots"
    pages = {("a.example", "/robots.txt"): robots(), **{("a.example", f"/{i}"): Resp(302, {"Location": f"/{i + 1}"}) for i in range(6)}}
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/0", GOOD, pages)
    assert e.value.reason == "redirects"


def test_dns_rebind_second_resolution_is_checked_and_each_hop_resolves_once():
    # first answer public, later answer private: the redirect hop re-resolves and is refused
    pages = {**OKPAGES, ("a.example", "/p"): Resp(302, {"Location": "https://a.example/q"})}
    n = Net({"a.example": [["93.184.216.34"], ["93.184.216.34"], ["127.0.0.1"]]}, pages)
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport)
    assert e.value.reason in ("blocked_address", "robots")  # rebound hop never reaches transport for /q
    assert all(c[1] == "93.184.216.34" for c in n.calls)


@pytest.mark.parametrize("rob", [Resp(500), Resp(429), Resp(503), RuntimeError("down"), Resp(302, {"Location": "https://10.0.0.1/robots.txt"})])
def test_robots_fail_closed(rob):
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {("a.example", "/robots.txt"): rob, ("a.example", "/p"): html()})
    assert e.value.reason == "robots_unreadable"  # WPL-1111 (ROBOTS-LABEL-1111): unreadable is not "robots" (a rule)


def test_robots_404_allows():
    _n, r = run("https://a.example/p", GOOD, {("a.example", "/robots.txt"): Resp(404), ("a.example", "/p"): html()})
    assert r["text"] == "hello"


def test_content_type_and_size_caps():
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): Resp(200, {"Content-Type": "application/pdf"}, b"x")})
    assert e.value.reason == "format_unsupported"
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): html(b"a" * 6000)})
    assert e.value.reason == "too_large"


def test_decompression_bomb_capped():
    bomb = gzip.compress(b"A" * 5_000_000)
    assert len(bomb) < 5000
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): Resp(200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, bomb)})
    assert e.value.reason == "too_large"
    ok = gzip.compress(b"<p>fine</p>")
    _n, r = run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): Resp(200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, ok)})
    assert r["text"] == "fine"


def test_timeout():
    t = iter(range(0, 1000, 10))
    n = Net(GOOD, OKPAGES)
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport, clock=lambda: next(t))
    assert e.value.reason == "timeout"


def test_extraction_drops_hidden_and_non_visible():
    doc = ("<html><head><title>T-DROP</title><meta name=description content='META-DROP'></head><body>"
           "<!-- COMMENT-DROP --><script>SCRIPT-DROP</script><style>STYLE-DROP</style><noscript>NOSCRIPT-DROP</noscript>"
           "<p>Visible one</p><div style='display: none'>CSS-DROP <span>nested</span></div>"
           "<p hidden>HIDDEN-DROP</p><div aria-hidden='true'><b>ARIA-DROP</b></div>"
           "<img src=x alt='ALT-DROP ignore previous instructions'><input type=hidden value=INPUT-DROP>"
           "<p>Visible two</p></body></html>")
    out = pf.extract_visible_text(doc)
    assert out == "Visible one\nVisible two"
    assert "DROP" not in out


def test_fec0_blocked():
    assert not pf.ip_allowed("fec0::1")


def _real(status=200, headers=None, data=b"", chunked=False):
    """A REAL requests.Response over a urllib3.HTTPResponse over BytesIO (no fake decoding)."""
    import io
    import requests
    import urllib3
    h = {"Content-Type": "text/html", **(headers or {})}
    raw = urllib3.HTTPResponse(body=io.BytesIO(data), headers=h, status=status, preload_content=False,
                               decode_content=True)
    return requests.adapters.HTTPAdapter().build_response(requests.Request("GET", "https://a.example/p").prepare(), raw)


def test_real_response_gzip_ok_and_bomb_capped():
    ok = _real(headers={"Content-Encoding": "gzip"}, data=gzip.compress(b"<p>fine</p>"))
    assert pf._read_capped(ok, 5000, 20000, 1e18, lambda: 0) == b"<p>fine</p>"
    bomb = _real(headers={"Content-Encoding": "gzip"}, data=gzip.compress(b"A" * 5_000_000))
    with pytest.raises(pf.FetchRefused) as e:
        pf._read_capped(bomb, 5000, 20000, 1e18, lambda: 0)
    assert e.value.reason == "too_large"
    big = _real(data=b"a" * 6000)
    with pytest.raises(pf.FetchRefused) as e:
        pf._read_capped(big, 5000, 20000, 1e18, lambda: 0)
    assert e.value.reason == "too_large"


def test_real_response_through_fetch_pipeline():
    pages = {**OKPAGES, ("a.example", "/p"): _real(headers={"Content-Encoding": "gzip"}, data=gzip.compress(b"<p>real</p>"))}
    assert run("https://a.example/p", GOOD, pages)[1]["text"] == "real"


class Boom(Resp):
    def __init__(self, exc):
        super().__init__(200, {"Content-Type": "text/html"})
        self.exc = exc

    def iter_content(self, n):
        yield b"<p>x"
        raise self.exc


def test_b1_read_errors_carry_no_page_bytes():
    import urllib3
    s = L.new()
    for exc in (ValueError(s), urllib3.exceptions.InvalidChunkLength(type("R", (), {"tell": lambda self: 0, "length_remaining": 0})(), s.encode()), zlib_error(s)):
        with pytest.raises(pf.FetchRefused) as e:
            run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): Boom(exc)})
        assert e.value.reason == "read_error"
        L.assert_no_leak(s, exc=[e.value])
    # robots path: any exception fails closed
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/robots.txt"): Boom(ValueError(s))})
    assert e.value.reason == "robots_unreadable"  # WPL-1111: any exception on the robots path is unreadable, still closed
    L.assert_no_leak(s, exc=[e.value])
    bad = gzip.compress(b"<p>x</p>")[:12] + s.encode()
    with pytest.raises(pf.FetchRefused) as e:
        run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): Resp(200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, bad)})
    L.assert_no_leak(s, exc=[e.value])


def zlib_error(s):
    import zlib
    return zlib.error(s)


def test_s2_trickling_server_killed_by_watchdog():
    import threading
    import time as _t
    ev = threading.Event()

    class Trickle(Resp):
        def __init__(self):
            super().__init__(200, {"Content-Type": "text/html"})

        def iter_content(self, n):
            yield b"<p>"
            ev.wait(30)  # blocked in recv until the watchdog closes the response
            raise OSError("closed")

        def close(self):
            ev.set()
    cfg = {**CFG, "timeout_s": 0.3}
    n = Net(GOOD, {**OKPAGES, ("a.example", "/p"): Trickle()})
    t0 = _t.monotonic()
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", cfg, resolve=n.resolve, transport=n.transport)
    assert e.value.reason == "timeout" and _t.monotonic() - t0 < 5


def test_s2_dns_lookup_is_bounded():
    import threading
    ev = threading.Event()
    cfg = {**CFG, "timeout_s": 0.2}
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", cfg, resolve=lambda h, p: ev.wait(30), transport=lambda *a: None)
    ev.set()
    assert e.value.reason == "dns"


def test_m2_robots_cached_per_fetch_and_redirect_chain():
    pages = {**OKPAGES, ("a.example", "/p"): Resp(302, {"Location": "/q"}), ("a.example", "/q"): html()}
    n, r = run("https://a.example/p", GOOD, pages)
    assert [c[3] for c in n.calls].count("/robots.txt") == 1


def test_m_missing_head_close_keeps_body():
    assert pf.extract_visible_text("<head><title>t</title><p>VISIBLE</p>") == "VISIBLE"


HIDDEN = {
    "self_closing_script": "<script/>X-DROP</script>",
    "self_closing_style": "<style/>X-DROP</style>",
    "self_closing_template": "<template/>X-DROP</template>",
    "self_closing_noscript": "<noscript/>X-DROP</noscript>",
    "self_closing_hidden": "<div hidden/>X-DROP</div>",
    "visibility": "<p style='visibility:hidden'>X-DROP</p>",
    "opacity": "<p style='opacity:0'>X-DROP</p>",
    "offscreen": "<p style='position:absolute;left:-9999px'>X-DROP</p>",
    "font0": "<p style='font-size:0'>X-DROP</p>",
    "fonttiny": "<p style='font-size:1px'>X-DROP</p>",
    "textindent": "<p style='text-indent:-9999px'>X-DROP</p>",
    "clip": "<p style='clip:rect(0,0,0,0)'>X-DROP</p>",
    "scale0": "<p style='transform:scale(0)'>X-DROP</p>",
    "samecolour": "<p style='color:#fff;background-color:#fff'>X-DROP</p>",
    "heightzero": "<p style='height:0;overflow:hidden'>X-DROP</p>",
    "css_escape": "<p style='display:\\6e one'>X-DROP</p>",
    "class_rule": "<style>.h{display:none}</style><p class='a h'>X-DROP</p>",
    "id_rule": "<style>#z { opacity:0 }</style><p id=z>X-DROP</p>",
    "dialog": "<dialog>X-DROP</dialog>",
    "xmp": "<xmp>X-DROP</xmp>",
    "iframe_srcdoc": "<iframe srcdoc='X-DROP'></iframe>",
    "svg_text": "<svg><text>X-DROP</text></svg>",
    "aria": "<span aria-hidden='true'>X-DROP</span>",
    "noscript": "<noscript>X-DROP</noscript>",
    "template": "<template>X-DROP</template>",
}


@pytest.mark.parametrize("name", sorted(HIDDEN))
def test_s3_hidden_text_dropped(name):
    assert pf.extract_visible_text("<p>keep</p>" + HIDDEN[name] + "<p>end</p>") == "keep\nend"


def test_s3_open_dialog_kept():
    assert "shown" in pf.extract_visible_text("<dialog open>shown</dialog>")


def test_sentinel_capture_all_loggers_stdout_warnings():
    s = L.new()
    with L.capture_all() as cap:
        lg = logging.getLogger("uvicorn.access.test")
        lg.propagate = False
        lg.setLevel(logging.CRITICAL)
        lg.debug("x %s", s)
        assert L.find_leaks(s, run=cap)[:1] == ["run"]
    s2 = L.new()
    with L.capture_all() as cap:
        print(s2)
        assert "run" in L.find_leaks(s2, run=cap)
    s3 = L.new()
    with L.capture_all() as cap:
        warnings.warn(s3)
        assert "run" in L.find_leaks(s3, run=cap)
    s4 = L.new()
    with L.capture_all() as cap:
        pass
    assert "run" not in L.find_leaks(s4, run=cap)
    assert L.variants(s)[-3:] and any("=" not in v for v in L.variants(s))


def test_sentinel_detects_exception_and_depth():
    s = L.new()
    assert "exc" in L.find_leaks(s, exc=[ValueError(s)])
    deep = cur = []
    for _ in range(300):
        nxt = []
        cur.append(nxt)
        cur = nxt
    with pytest.raises(AssertionError):
        L.find_leaks(s, deep=deep)


def test_fetch_itself_persists_and_logs_nothing():
    s = L.new()
    with L.capture_all() as cap:
        n, r = run("https://a.example/p", GOOD, {**OKPAGES, ("a.example", "/p"): html(f"<p>{s}</p>".encode())})
    assert s in r["text"]
    assert "run" not in L.find_leaks(s, run=cap, calls=[(c[0], c[1], c[3], c[4]) for c in n.calls])


import warnings  # noqa: E402


@pytest.mark.parametrize("junk", ["a;" * 2_000_000, "/*" * 2_000_000, "a{" * 2_000_000, "}" * 4_000_000,
                                  "@media{" * 500_000, "x:y;" * 1_000_000])
def test_n1_adversarial_style_is_linear(junk):
    import time as _t
    t0 = _t.monotonic()
    out = pf.extract_visible_text("<style>" + junk + "</style><p>ok</p>")
    assert out == "ok" and _t.monotonic() - t0 < 1.0


def test_n1_extraction_runs_under_deadline():
    t = iter(range(0, 10_000))
    with pytest.raises(pf.FetchRefused) as e:
        pf.extract_visible_text("<p>" * 100, "text/html", lambda: next(t), 5)
    assert e.value.reason == "timeout"


def test_s2_header_trickle_bounded():
    import threading
    import time as _t
    ev = threading.Event()
    closed = []

    def slow(host, ip, port, target, headers, timeout):
        ev.wait(20)  # a server trickling response headers: the transport never returns in time
        return type("R", (), {"close": lambda self: closed.append(1)})()
    cfg = {**CFG, "timeout_s": 0.3}
    n = Net(GOOD, OKPAGES)
    t0 = _t.monotonic()
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", cfg, resolve=n.resolve, transport=slow)
    assert e.value.reason in ("timeout", "robots", "robots_unreadable") and _t.monotonic() - t0 < 5
    ev.set()
    _t.sleep(0.2)
    assert closed  # a late response is closed, not leaked


class Slot:
    __slots__ = ("a", "b")

    def __init__(self, a, b=None):
        self.a, self.b = a, b


def test_s4_slots_temporaries_cannot_hide_a_sentinel():
    s = L.new()
    objs = [Slot("x" * i) for i in range(500)] + [Slot(s)] + [Slot(i) for i in range(500)]
    assert L.find_leaks(s, objs=objs)[:1] == ["objs"]
    nested = [[Slot({"k": "v" * i}, Slot(i)) for i in range(300)], Slot(Slot(Slot(s)))]
    assert "nested" in L.find_leaks(s, nested=nested)


MORE = {
    "colour_name": "<p style='color:white;background:#FFF'>X-DROP</p>",
    "colour_rgb": "<p style='color:rgb(255,255,255);background-color:white'>X-DROP</p>",
    "colour_hex3": "<p style='color:#abc;background-color:#aabbcc'>X-DROP</p>",
    "descendant": "<style>div .h{display:none}</style><div><span class=h>X-DROP</span></div>",
    "attr_sel": "<style>[data-x=y]{opacity:0}</style><p data-x='y'>X-DROP</p>",
    "attr_contains": "<style>p[class*=hid]{display:none}</style><p class='ahidb'>X-DROP</p>",
    "neg_margin": "<p style='margin-left:-99999px'>X-DROP</p>",
    "neg_pct": "<p style='position:absolute;left:-100%'>X-DROP</p>",
    "translatex": "<p style='transform:translateX(-9999px)'>X-DROP</p>",
    "translate": "<p style='transform:translate(-200%,0)'>X-DROP</p>",
    "media_nested": "<style>@media screen{.h{display:none}}</style><p class=h>X-DROP</p>",
}


@pytest.mark.parametrize("name", sorted(MORE))
def test_minor_more_hiding_dropped(name):
    assert pf.extract_visible_text("<p>keep</p>" + MORE[name] + "<p>end</p>") == "keep\nend"


def test_minor_different_colours_kept():
    assert "shown" in pf.extract_visible_text("<p style='color:white;background:black'>shown</p>")


def test_r3_moderate_offscreen_absolute_dropped_but_relative_kept():
    assert pf.extract_visible_text("<p style='position:absolute;left:-500px'>X</p><p>k</p>") == "k"
    assert pf.extract_visible_text("<p style='position:absolute;left:-100px'>X</p><p>k</p>") == "k"
    assert "X" in pf.extract_visible_text("<p style='position:absolute;left:-50px'>X</p>")
    assert "X" in pf.extract_visible_text("<p style='position:relative;left:-500px'>X</p>")


def test_r3_abandoned_workers_are_unblocked_and_bounded():
    import threading
    import time as _t
    base = threading.active_count()
    started = []

    def hang(host, ip, port, target, headers, timeout):
        ev = threading.Event()
        pf.register_abort(ev.set)  # what default_transport does with the live socket
        started.append(1)
        ev.wait(30)
        raise OSError("aborted")
    for _ in range(30):
        with pytest.raises(pf.FetchRefused) as e:
            pf._call_transport(hang, ("h", "1.2.3.4", 443, "/", {}, 0.05), 0.05)
        assert e.value.reason == "timeout"
    _t.sleep(0.5)
    assert threading.active_count() <= base + 1  # aborted workers exited; no growth
    assert len(started) == 30


def test_r3_busy_past_cap():
    import threading
    ev = threading.Event()
    old = list(pf._ABANDONED)
    stuck = [threading.Thread(target=ev.wait, args=(30,), daemon=True) for _ in range(pf.MAX_ABANDONED)]
    for t in stuck:
        t.start()
    pf._ABANDONED[:] = stuck
    try:
        with pytest.raises(pf.FetchRefused) as e:
            pf._call_transport(lambda *a: None, (), 0.1)
        assert e.value.reason == "busy"
    finally:
        ev.set()
        pf._ABANDONED[:] = old


def test_fetch_returns_noai_header_and_meta_robots_signals_only():
    page = (b'<html><head><meta name="robots" content="NoAI, noindex"><meta name="description" content="x">'
            b'<meta name="googlebot" content="noimageai"></head><body><p>hello</p></body></html>')
    r = Resp(200, {"Content-Type": "text/html", "X-Robots-Tag": "noai", "Content-Signal": "ai-input=no",
                   "Set-Cookie": "a=b", "Server": "s"}, page)
    net = Net({"s.example": ["93.184.216.34"]}, {("s.example", "/robots.txt"): robots(), ("s.example", "/p"): r})
    out = pf.fetch("https://s.example/p", CFG, resolve=net.resolve, transport=net.transport)
    # WP6-P2: normalised signal strings only; no raw header dict (Set-Cookie, Server never kept)
    assert out["x_robots_tag"] == "noai" and out["content_signal"] == "ai-input=no"
    assert out["meta_robots"] == "noai, noindex, noimageai"
    assert set(out) == {"text", "url", "content_type", "x_robots_tag", "content_signal", "meta_robots", "licence_href",
                        "meta_author", "tdm_reservation", "canonical_url", "canonical_signals", "outbound_links",
                        "third_party_text"}  # H19-T3a; H19-FIX-1109 A8/A11; H19-FIX2-1109
    assert "hello" in out["text"] and "a=b" not in repr(out) and "headers" not in out


def test_fetch_signal_keys_empty_when_absent():
    net = Net({"s.example": ["93.184.216.34"]},
              {("s.example", "/robots.txt"): robots(), ("s.example", "/p"): html()})
    out = pf.fetch("https://s.example/p", CFG, resolve=net.resolve, transport=net.transport)
    assert out["x_robots_tag"] == out["content_signal"] == out["meta_robots"] == ""


# ---- robots Crawl-delay (NONASK-1103 I-A6 C5)
class Clk:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _cd_run(url, robots_txt, clk):
    pages = {("a.example", "/robots.txt"): robots(robots_txt), ("a.example", "/p"): html(), ("a.example", "/q"): html()}
    n = Net(GOOD, pages)
    return n, pf.fetch(url, CFG, resolve=n.resolve, transport=n.transport, clock=clk)


def _pages(n):
    return [c[3] for c in n.calls if c[3] != "/robots.txt"]


@pytest.mark.parametrize("txt", [b"User-agent: UA-test\nCrawl-delay: 10\n", b"User-agent: *\nCrawl-delay: 10\n"])
def test_crawl_delay_ua_and_star_second_fetch_refused_then_allowed(txt):
    from vera import selective_fetch as sf
    sf.reset_for_tests()
    clk = Clk()
    n, _ = _cd_run("https://a.example/p", txt, clk)
    clk.t += 5
    n2 = Net(GOOD, {("a.example", "/robots.txt"): robots(txt), ("a.example", "/q"): html()})
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/q", CFG, resolve=n2.resolve, transport=n2.transport, clock=clk)
    assert e.value.reason == "crawl_delay" and _pages(n2) == []  # not sent early
    clk.t += 6
    assert pf.fetch("https://a.example/q", CFG, resolve=n2.resolve, transport=n2.transport, clock=clk)["text"] == "hello"
    assert _pages(n2) == ["/q"]
    sf.reset_for_tests()


def test_no_crawl_delay_no_wait():
    from vera import selective_fetch as sf
    sf.reset_for_tests()
    clk = Clk()
    n, _ = _cd_run("https://a.example/p", b"User-agent: *\nAllow: /\n", clk)
    n2 = Net(GOOD, {("a.example", "/robots.txt"): robots(), ("a.example", "/q"): html()})
    assert pf.fetch("https://a.example/q", CFG, resolve=n2.resolve, transport=n2.transport, clock=clk)["text"] == "hello"
    sf.reset_for_tests()


def test_signal_value_normalises_and_caps():
    assert pf.signal_value("  NoAI,\r\n\tNOINDEX ") == "noai, noindex"
    assert pf.signal_value(None) == "" and len(pf.signal_value("x" * 1000)) == 300
