"""NONASK-1103 S6 (plan 6.9): guarded "public-any" fetch mode and visible-text extraction.

https only (http on port 80 is upgraded to https on 443), ports 443/80, no userinfo. Each hop resolves DNS once, blocks
every non-public address, pins the IP, and connects to it with SNI and certificate verification against the hostname.
robots.txt goes through the same pinned path and fails closed. Page text is returned to the caller and never stored,
logged or cached here. Reuses selective_fetch guards by import; does not edit them."""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import threading
import time
import zlib
from html.parser import HTMLParser
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

from vera import selective_fetch as sf

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "public_fetch.json"
_REDIRECTS = (301, 302, 303, 307, 308)
_EXTRA_BLOCKED = tuple(ipaddress.ip_network(n) for n in (
    "169.254.169.254/32", "fd00:ec2::254/128", "100.64.0.0/10", "fc00::/7", "fe80::/10", "fec0::/10", "169.254.0.0/16"))


class FetchRefused(Exception):
    """Fetch stopped. `reason` is a short fixed code; the message never carries page text."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(path) == CONFIG_PATH and isinstance(cfg, dict):  # LISTS-TO-DB-1104: fetch.deny_hosts (union with the file)
        try:
            from vera import list_tables
            cfg = list_tables.overlay("public_fetch.json", cfg)
        except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
            pass
    return cfg


def ip_allowed(ip) -> bool:
    if isinstance(ip, str):
        ip = ipaddress.ip_address(ip)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return False
    if any(ip in n for n in _EXTRA_BLOCKED if n.version == ip.version):
        return False
    return sf._ip_is_public(ip)


def host_denied(host: str, cfg: dict) -> bool:
    h = (host or "").lower().rstrip(".")
    return any(h == d.lower().rstrip(".") or h.endswith("." + d.lower().rstrip(".")) for d in cfg.get("deny_hosts", []))


def check_url(url: str, cfg: dict, resolve=None, timeout=None):
    """Validate one hop. Returns (url_https, host, port, [ips]) or raises FetchRefused. Resolves exactly once."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise FetchRefused("unparseable") from None
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise FetchRefused("scheme")
    if parts.username is not None or parts.password is not None:
        raise FetchRefused("userinfo")
    if not sf._parsers_agree(url, parts):
        raise FetchRefused("parser_disagree")
    if port not in (None, 80, 443) or (parts.scheme, port) in (("http", 443), ("https", 80)):
        raise FetchRefused("port")
    host = parts.hostname.lower().rstrip(".")
    if host_denied(host, cfg):
        raise FetchRefused("deny_listed")
    if parts.scheme == "http":  # upgrade, never plain text
        url = parts._replace(scheme="https", netloc=host if ":" not in host else f"[{host}]").geturl()
        port = None
    try:
        ipaddress.ip_address(host)
        literal = [host]
    except ValueError:
        literal = None
    try:
        infos = literal or [i[4][0] for i in _bounded_resolve(resolve, host, port or 443, timeout)]
    except (socket.gaierror, OSError):
        raise FetchRefused("dns") from None
    ips: list[str] = []
    for a in infos:
        a = a if isinstance(a, str) else a[4][0]
        if not ip_allowed(a):
            raise FetchRefused("blocked_address")  # ANY non-public answer refuses the hop
        if a not in ips:
            ips.append(a)
    if not ips:
        raise FetchRefused("dns")
    return url, host, port or 443, ips


MAX_ABANDONED = 8  # safety bound: workers still alive after a transport timeout; past it new fetches are "busy"
_ABANDONED: list = []
_ABANDONED_LOCK = threading.Lock()  # WP6-P2b M2: _ABANDONED is shared by transport and DNS workers


def _reserve_worker_slot() -> None:
    """Prune finished workers; refuse as "busy" once MAX_ABANDONED abandoned (transport or DNS) workers are alive."""
    with _ABANDONED_LOCK:
        _ABANDONED[:] = [t for t in _ABANDONED if t.is_alive()]
        if len(_ABANDONED) >= MAX_ABANDONED:
            raise FetchRefused("busy")


def _abandon(t) -> None:
    with _ABANDONED_LOCK:
        _ABANDONED.append(t)
_TLS = threading.local()


class _Abort:
    """Closers registered by a transport worker; fired when the caller gives up so the worker unblocks."""

    def __init__(self):
        self.fns, self.fired, self.lock = [], False, threading.Lock()

    def add(self, fn):
        with self.lock:
            if not self.fired:
                self.fns.append(fn)
                return
        fn()

    def fire(self):
        with self.lock:
            self.fired, fns = True, list(self.fns)
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


def register_abort(fn) -> None:
    """Called by a transport (in its worker thread) to register a closer (session, socket shutdown)."""
    a = getattr(_TLS, "abort", None)
    if a is not None:
        a.add(fn)


def _sock_closer(sock):
    def close():
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
    return close


def default_transport(host, ip, port, target, headers, timeout):
    """Real transport: connect to the pinned IP, SNI + certificate check against host. Never used in tests."""
    import requests
    s = requests.Session()
    s.trust_env = False
    ad = sf._PinnedTLSAdapter(host)
    try:  # best effort: let the abort hook shut down the live socket (a trickling server never frees the worker)
        from urllib3.connection import HTTPSConnection
        from urllib3.connectionpool import HTTPSConnectionPool

        class _Conn(HTTPSConnection):
            def connect(self):
                super().connect()
                register_abort(_sock_closer(self.sock))

        class _Pool(HTTPSConnectionPool):
            ConnectionCls = _Conn
        ad.init_poolmanager(1, 1)
        ad.poolmanager.pool_classes_by_scheme = {**ad.poolmanager.pool_classes_by_scheme, "https": _Pool}
    except Exception:  # noqa: BLE001
        pass
    s.mount("https://", ad)
    register_abort(s.close)
    lit = f"[{ip}]" if ":" in ip else ip
    try:
        resp = s.get(f"https://{lit}:{port}{target}", headers=headers, timeout=timeout, allow_redirects=False,
                     stream=True, verify=True)
    except Exception:
        s.close()
        raise
    orig = resp.close

    def _close():
        try:
            orig()
        finally:
            s.close()
    resp.close = _close
    return resp


def _bounded_resolve(resolve, host, port, timeout):
    """getaddrinfo (or the injected resolver) with a timeout; a late answer is discarded."""
    fn = resolve or socket.getaddrinfo
    if timeout is None:
        return fn(host, port)
    box: dict = {}
    _reserve_worker_slot()  # WP6-P2b M2: abandoned DNS threads count against MAX_ABANDONED too

    def run():
        try:
            box["r"] = fn(host, port)
        except BaseException as exc:  # noqa: BLE001
            box["e"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(max(0.05, timeout))
    if "r" not in box:
        if "e" not in box and t.is_alive():
            _abandon(t)
        if "e" in box and isinstance(box["e"], (socket.gaierror, OSError)):
            raise socket.gaierror("dns")
        raise socket.gaierror("dns timeout")
    return box["r"]


def _chunks(resp, n=8192):
    """Wire bytes: raw.stream(decode_content=False) on a real response (we decompress under our own cap, S1); a
    response double without .raw uses iter_content."""
    raw = getattr(resp, "raw", None)
    if raw is not None and hasattr(raw, "stream"):
        return raw.stream(n, decode_content=False)
    return resp.iter_content(n)


def _read_capped(resp, cap_c: int, cap_d: int, end: float, clock) -> bytes:
    enc = str(resp.headers.get("Content-Encoding") or "identity").strip().lower()
    if enc not in ("identity", "gzip", "deflate"):
        raise FetchRefused("encoding")
    dec = zlib.decompressobj(47 if enc == "gzip" else zlib.MAX_WBITS) if enc != "identity" else None
    out, comp, total = [], 0, 0
    failed = False
    try:
        for chunk in _chunks(resp):
            if clock() > end:
                raise FetchRefused("timeout")
            comp += len(chunk)
            if comp > cap_c:
                raise FetchRefused("too_large")
            if dec is None:
                total += len(chunk)
                out.append(chunk)
            else:
                buf = chunk
                while buf and not dec.eof:
                    data = dec.decompress(buf, cap_d - total + 1)
                    total += len(data)
                    out.append(data)
                    if total > cap_d:
                        raise FetchRefused("too_large")
                    buf = dec.unconsumed_tail
            if total > cap_d:
                raise FetchRefused("too_large")
    except FetchRefused:
        raise
    except Exception:  # noqa: BLE001 - B1: library errors can embed page bytes; never let them out
        failed = True
    if failed:  # raised outside the handler so __context__ does not retain the library exception either
        raise FetchRefused("read_error")
    return b"".join(out)


def _kill(resp):
    """Watchdog action: close the response and, best effort, shut the socket so a blocked recv returns."""
    try:
        raw = getattr(resp, "raw", None)
        sock = getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None)
        sock = getattr(sock, "_sock", None)
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
    except Exception:  # noqa: BLE001
        pass
    try:
        resp.close()
    except Exception:  # noqa: BLE001
        pass


def _call_transport(transport, args, timeout):
    """S2: connect, TLS and header reads are bounded by a wall-clock join, not only by per-recv socket timeouts. A
    response that arrives after the deadline is closed and discarded."""
    box: dict = {}
    lock = threading.Lock()
    _reserve_worker_slot()
    abort = _Abort()

    def run():
        _TLS.abort = abort
        try:
            r = transport(*args)
        except BaseException as exc:  # noqa: BLE001
            r, box["e"] = None, exc
        with lock:
            if box.get("abandoned") and r is not None:
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass
            else:
                box["r"] = r
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(max(0.05, timeout))
    with lock:
        if "r" not in box:
            box["abandoned"] = True
            _abandon(t)
            abort.fire()
            raise FetchRefused("timeout") if "e" not in box else FetchRefused("unreachable")
    if box.get("r") is None:
        raise FetchRefused("unreachable")
    return box["r"]


def _get(url, cfg, resolve, transport, clock, end, allowed_ctypes, robots_check, cap_c, cap_d, ok_robots=False, meta=None,
         crawl_check=None, hop_gate=None):
    """Follow redirects manually; returns (body, final_url, status, content_type). `meta` (optional dict) receives the
    no-AI response headers of the final 200 response (X-Robots-Tag, Content-Signal only). `hop_gate(url)` (WP6-P2b S1,
    R-050) runs on EVERY hop (the first URL and each redirect target) after check_url and BEFORE robots.txt or the
    request for that hop: a non-empty return is a refusal code, raised as FetchRefused(code); nothing is sent to it."""
    cur, hops, seen = url, 0, set()
    while True:
        if clock() > end:
            raise FetchRefused("timeout")
        try:  # API-URL-GATE-1108: also before DNS, on the URL as given (re-checked below on the canonical URL)
            sf.api_gate(cur, "fulltext")
        except sf.ApiRefused:
            raise FetchRefused("api_denied") from None
        cur, host, port, ips = check_url(cur, cfg, resolve, timeout=end - clock())
        if cur in seen:
            raise FetchRefused("cycle")
        seen.add(cur)
        # API-URL-GATE-1108: an API URL (first URL, redirect target, robots.txt or tdmrep.json on an API host) needs
        # its config/api_permissions.json entry for "fulltext" BEFORE the hop gate, robots.txt or the request;
        # fails closed when that config is unusable. Refused -> FetchRefused("api_denied"), nothing sent.
        try:
            api_d = sf.api_gate(cur, "fulltext")
        except sf.ApiRefused:
            raise FetchRefused("api_denied") from None
        if hop_gate is not None:
            refused = hop_gate(cur)
            if refused:
                raise FetchRefused(str(refused))  # read-restricted hop: neither robots.txt nor the page is requested
        if robots_check is not None and not robots_check(cur, host, port):
            raise FetchRefused("robots")
        if crawl_check is not None and not sf.crawl_reserve(host, crawl_check(host, port), clock):
            raise FetchRefused("crawl_delay")  # never sent early; robots.txt itself is not delayed (crawl_check None)
        parts = urlsplit(cur)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        hdrs = {"User-Agent": cfg["user_agent"], "Accept-Encoding": "identity",
                "Host": host + ("" if port == 443 else f":{port}")}
        permit = None
        if api_d is not None:  # API-URL-GATE-1108: a permitted API hop: the entry's User-Agent and rate portal
            from vera import api_permissions, rate_limit
            hdrs = api_permissions.headers_for(api_d, hdrs)
            permit = rate_limit.get_registry().portal(str(api_d.conditions["rate_portal"])).acquire(
                deadline=time.monotonic() + max(0.0, end - clock()))  # review (5): never wait past this fetch's end
            if not permit.ok:
                raise FetchRefused("rate_limited")  # skipped, nothing sent
        resp = None
        try:
            for ip in ips[:sf.PIN_MAX_ADDRS]:
                try:
                    resp = _call_transport(transport, (host, ip, port, target, hdrs, max(0.1, end - clock())),
                                           end - clock())
                    break
                except FetchRefused as exc:
                    if exc.reason == "timeout":
                        raise
                    continue
        except BaseException as e:
            if permit is not None:
                permit.record(error=e)
            raise
        if permit is not None:
            if resp is None:
                permit.record(error=ConnectionError("unreachable"))
            else:
                permit.record(status=getattr(resp, "status_code", None), headers=getattr(resp, "headers", None))
        if resp is None:
            raise FetchRefused("unreachable")
        fired = threading.Event()

        def _fire(r=resp):
            fired.set()
            _kill(r)
        wd = threading.Timer(max(0.05, end - clock()), _fire)
        wd.daemon = True
        wd.start()
        try:
            if resp.status_code in _REDIRECTS and resp.headers.get("Location"):
                hops += 1
                if hops > int(cfg.get("max_redirects", 2)):
                    raise FetchRefused("redirects")
                cur = urljoin(cur, resp.headers["Location"])
                continue
            if resp.status_code != 200:
                if ok_robots and 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return b"", cur, resp.status_code, ""
                raise FetchRefused("status")
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in allowed_ctypes:
                raise FetchRefused("format_unsupported")
            if meta is not None:
                meta["headers"] = {k: str(resp.headers.get(h)) for k, h in _SIGNAL_HEADERS if resp.headers.get(h)}
            try:
                return _read_capped(resp, cap_c, cap_d, end, clock), cur, 200, ctype
            except FetchRefused:
                if fired.is_set():
                    raise FetchRefused("timeout") from None
                raise
        finally:
            wd.cancel()
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass


def crawl_delay_for(cache, host, port, cfg) -> float:
    """Crawl-delay (seconds) from the cached parsed robots.txt for this UA, else the * group; 0 if none."""
    rp = (cache or {}).get((host, port))
    if rp in (None, True, False):
        return 0.0
    try:
        from vera import robots_ai as _ra  # OD-1106-ROBOTS-AI: largest Crawl-delay among the applying groups
        v = _ra.crawl_delay(rp, cfg["user_agent"])
        return float(v) if v else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def robots_allowed(url, host, port, cfg, resolve, transport, clock, end, cache=None, hop_gate=None) -> bool:
    """robots.txt over the same pinned path. 200 parsed; other 4xx (not 429) allow; 429, 5xx, error, redirect loop,
    oversize or ANY exception fails closed. Parsed rules are cached per host (rules only) for one fetch() call, or for
    one request when the caller passes its own cache (fetch(robots_cache=...)). A robots.txt redirect to a hop refused by
    `hop_gate` is never requested and fails closed."""
    key = (host, port)
    try:
        if cache is not None and key in cache:
            rp = cache[key]
            return True if rp is True else _ai_ok(rp, cfg["user_agent"], url)
        ru = f"https://{host}{'' if port == 443 else f':{port}'}/robots.txt"
        body, _f, status, _c = _get(ru, cfg, resolve, transport, clock, end, ("text/plain", "text/html", ""),
                                    None, int(cfg.get("max_robots_bytes", 200000)),
                                    int(cfg.get("max_robots_bytes", 200000)), ok_robots=True, hop_gate=hop_gate)
        if status != 200:
            if cache is not None:
                cache[key] = True
            return True
        rp = robotparser.RobotFileParser()
        txt = body.decode("utf-8", "replace")
        rp.parse(txt.splitlines())
        from vera import robots_ai as _ra  # OD-1106-ROBOTS-AI
        _ra.annotate(rp, txt)
        if cache is not None:
            cache[key] = rp
            from vera import source_permissions as _sp  # WP6-P2b D2: robots.txt Content-Signal, signal text only
            cache[("content_signal",) + key] = signal_value(_sp.robots_content_signal(txt))
        return _ai_ok(rp, cfg["user_agent"], url)
    except Exception:  # noqa: BLE001
        return False


def _ai_ok(rp, ua, url) -> bool:
    """Own group / * allow AND no AI agent group or AI statement blocks (OD-1106-ROBOTS-AI)."""
    from vera import robots_ai as _ra
    return _ra.allowed(rp, ua, url) and _ra.block_reason(rp, url) is None


_SIGNAL_HEADERS = (("x-robots-tag", "X-Robots-Tag"), ("content-signal", "Content-Signal"),
                    ("content-usage", "Content-Usage"),
                    ("tdm-reservation", "TDM-Reservation"))
_META_TAG = re.compile(r"<meta\b[^>]*>", re.I)
_META_ATTR = re.compile(r"([a-zA-Z\-:]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")
_META_NAMES = ("robots", "googlebot", "vera-dar", "ai")


def _meta_robots(html: str) -> list:
    """content values (lower-cased) of <meta name=robots|googlebot|vera-dar|ai> in the first 200 kB. Signals only."""
    out = []
    for m in _META_TAG.findall(html[:200000]):
        a = {k.lower(): (v1 or v2 or v3) for k, v1, v2, v3 in _META_ATTR.findall(m)}
        if a.get("name", "").lower() in _META_NAMES and a.get("content"):
            out.append(a["content"].lower())
    return out


_LINK_TAG = re.compile(r"<(?:link|a)\b[^>]*>", re.I)


def _tags_attrs(html: str, tag_re) -> list:
    out = []
    for m in tag_re.findall(html[:200000]):
        out.append({k.lower(): (v1 or v2 or v3) for k, v1, v2, v3 in _META_ATTR.findall(m)})
    return out


def _licence_href(html: str, base: str) -> str:
    """H19 A7: the first <link|a rel~=license href> as an absolute http(s) URL (signal string, max 300 chars), else ''."""
    for a in _tags_attrs(html, _LINK_TAG):
        if {"license", "licence"} & set(a.get("rel", "").lower().split()) and a.get("href"):
            u = urljoin(base, a["href"].strip())
            if urlsplit(u).scheme in ("http", "https"):
                return signal_value(u)
    return ""


def _meta_author(html: str) -> str:
    """H19: byline from <meta name=author> or <meta property=article:author> only (signal string, max 200), else ''."""
    for a in _tags_attrs(html, _META_TAG):
        if (a.get("name", "").lower() == "author" or a.get("property", "").lower() == "article:author") \
                and a.get("content"):
            return signal_value(a["content"])[:200]
    return ""


def _tdm_meta(html: str) -> bool:
    """H19 A9: <meta name="tdm-reservation" content="1">."""
    return any(a.get("name", "").lower() == "tdm-reservation" and a.get("content", "").strip() == "1"
               for a in _tags_attrs(html, _META_TAG))


def _tdm_loc_match(loc: str, path: str) -> bool:
    """TDMRep location pattern: a path prefix; '*' matches any run; '$' ends the pattern (robots.txt style)."""
    if not isinstance(loc, str) or not loc.startswith("/"):
        return False
    rx = "".join(".*" if ch == "*" else ("$" if ch == "$" else re.escape(ch)) for ch in loc)
    return re.match(rx, path) is not None


def tdmrep_reserved(rules, path: str) -> bool:
    """True when ANY rule whose location matches `path` has tdm-reservation 1 (fail closed: the stricter of several
    matching rules applies). `rules` is the parsed /.well-known/tdmrep.json array; anything malformed raises ValueError."""
    if not isinstance(rules, list):
        raise ValueError("tdmrep.json is not an array")
    hit = False
    for r in rules:
        if not isinstance(r, dict) or not isinstance(r.get("location"), str):
            raise ValueError("tdmrep.json rule malformed")
        if _tdm_loc_match(r["location"], path) and str(r.get("tdm-reservation")).strip() == "1":
            hit = True
    return hit


def _tdmrep_signal(final, cfg, resolve, tr, clock, end, cache, hop_gate) -> str:
    """H19 A9: '' (no reservation), '1' (a matching rule reserves) or 'unreadable' (fail closed). /.well-known/tdmrep.json
    over the same pinned path, hop gate and size cap as robots.txt; fetched at most once per host per request
    (cache). 404 and 410 = no reservation; every other failure (timeout, 5xx, other 4xx, refused hop, bad JSON) =
    unreadable."""
    fp = urlsplit(final)
    host, port = (fp.hostname or "").lower().rstrip("."), fp.port or 443
    key = ("tdmrep", host, port)
    try:
        if key not in cache:
            u = f"https://{host}{'' if port == 443 else f':{port}'}/.well-known/tdmrep.json"
            cap = int(cfg.get("max_robots_bytes", 200000))
            try:
                body, _f, status, _c = _get(u, cfg, resolve, tr, clock, end, ("application/json", "text/plain", ""),
                                            None, cap, cap, ok_robots=True, hop_gate=hop_gate)
                if status in (404, 410):
                    cache[key] = []
                elif status != 200:
                    cache[key] = "unreadable"
                else:
                    rules = json.loads(body.decode("utf-8", "replace"))
                    tdmrep_reserved(rules, "/")  # validates the shape now; raises ValueError when malformed
                    cache[key] = rules
            except Exception:  # noqa: BLE001  refused hop, timeout, 5xx, malformed JSON: reserved
                cache[key] = "unreadable"
        rules = cache[key]
        if rules == "unreadable":
            return "unreadable"
        return "1" if tdmrep_reserved(rules, fp.path or "/") else ""
    except Exception:  # noqa: BLE001
        return "unreadable"


def fetch(url: str, cfg: dict | None = None, *, resolve=None, transport=None, clock=time.monotonic, hop_gate=None,
          robots_cache: dict | None = None, tdm_check: bool = False) -> dict:
    """Returns {"text", "url", "content_type", "x_robots_tag", "content_signal", "meta_robots", "licence_href", "meta_author", "tdm_reservation"} (H19: the last three are signal strings only; tdm_reservation is "", "1" or "unreadable" and the /.well-known/tdmrep.json fetch runs only when tdm_check=True; the three before are normalised no-AI signal strings only; content_signal joins the response header and the final host's robots.txt Content-Signal values, WP6-P2b D2) or raises FetchRefused. Nothing is written anywhere.
    `hop_gate(url) -> refusal code or None` is applied to every hop before any request to it (see _get).
    `robots_cache` (WP6-P2b M2): a caller-owned dict that keeps parsed robots.txt rules per host across the fetches of
    one request (rules only, never page text); default a fresh dict per call."""
    cfg = cfg or load_config()
    tr = transport or default_transport
    end = clock() + float(cfg.get("timeout_s", 8))
    cache: dict = robots_cache if robots_cache is not None else {}
    failed = False
    meta: dict = {}
    try:
        body, final, _s, ctype = _get(
            url, cfg, resolve, tr, clock, end, tuple(cfg["content_types"]),
            lambda u, h, p: robots_allowed(u, h, p, cfg, resolve, tr, clock, end, cache, hop_gate),
            int(cfg["max_compressed_bytes"]), int(cfg["max_decompressed_bytes"]), meta=meta,
            crawl_check=lambda h, p: crawl_delay_for(cache, h, p, cfg), hop_gate=hop_gate)
        decoded = body.decode("utf-8", "replace")
        text = extract_visible_text(decoded, ctype, clock, end)
    except FetchRefused:
        raise
    except Exception:  # noqa: BLE001
        failed = True
    if failed:
        raise FetchRefused("error")
    sig = meta.get("headers", {})  # WP6-P2: only the two signal values, normalised; never the raw header dict
    fp = urlsplit(final)
    robots_sig = cache.get(("content_signal", (fp.hostname or "").lower().rstrip("."), fp.port or 443), "")
    from vera import robots_ai as _ra, source_permissions as _sp2  # A1 / A2
    rp_final = cache.get(((fp.hostname or "").lower().rstrip("."), fp.port or 443))
    try:
        usage = _ra.usage_signals(rp_final, final) if rp_final is not None else ""  # A1: informational, never blocks
    except Exception:  # noqa: BLE001
        usage = ""
    cs = ", ".join(v for v in (str(sig.get("content-signal") or "").strip(), _sp2.content_usage_signal(sig.get("content-usage", "")),
                               robots_sig, usage) if v)  # header + robots.txt
    html = decoded if ctype != "text/plain" else ""
    tdm = "1" if (str(sig.get("tdm-reservation") or "").strip() == "1" or _tdm_meta(html)) else ""
    if tdm_check and not tdm:  # H19 A9: opt-in per caller (reporting lane, Wikipedia-citation reads); fail closed
        tdm = _tdmrep_signal(final, cfg, resolve, tr, clock, end, cache, hop_gate)
    return {"text": text, "url": final, "content_type": ctype,
            "x_robots_tag": signal_value(sig.get("x-robots-tag")), "content_signal": signal_value(cs),
            "meta_robots": signal_value(", ".join(_meta_robots(decoded)) if ctype != "text/plain" else ""),
            "licence_href": _licence_href(html, final), "meta_author": _meta_author(html), "tdm_reservation": tdm}


def signal_value(v) -> str:
    """A no-AI / derivative signal as a normalised string: lower-case, single spaces, control characters dropped,
    at most 300 characters. Nothing else from the response is kept (need-to-know, never stored)."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(v or ""))
    return re.sub(r"\s+", " ", s).strip().lower()[:300]


# ------------------------------------------------------------------------------------------ visible-text extraction
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_DROP = {"script", "style", "noscript", "template", "svg", "iframe", "object", "canvas", "select", "option",
         "title", "math", "xmp", "noembed", "noframes"}
_BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "ul", "ol", "table"}
_NUM = re.compile(r"^(-?\d*\.?\d+)([a-z%]*)$")
STYLE_SCAN_MAX = 262144  # safety bound on <style> text scanned (same kind as the byte caps); excess is ignored
MAX_RULES = 500
_COLOURS = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
            "yellow": "#ffff00", "gray": "#808080", "grey": "#808080", "silver": "#c0c0c0", "navy": "#000080",
            "orange": "#ffa500", "purple": "#800080", "lime": "#00ff00", "aqua": "#00ffff", "cyan": "#00ffff",
            "fuchsia": "#ff00ff", "magenta": "#ff00ff", "maroon": "#800000", "olive": "#808000", "teal": "#008080"}


def _num(v: str):
    m = _NUM.match(v.strip())
    return (float(m.group(1)), m.group(2)) if m else (None, "")


def _strip_comments(t: str) -> str:
    out, i = [], 0
    while True:
        j = t.find("/*", i)
        if j < 0:
            out.append(t[i:])
            return "".join(out)
        out.append(t[i:j])
        k = t.find("*/", j + 2)
        if k < 0:
            return "".join(out)
        i = k + 2


def _decls(style: str) -> dict:
    style = style[:4000]
    style = re.sub(r"\\([0-9a-fA-F]{1,6})\s?", lambda m: chr(min(int(m.group(1), 16), 0x10FFFF)), style)
    style = _strip_comments(style).lower()
    d = {}
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = "".join(v.replace("!important", "").split())
    return d


def _colour(v: str):
    """Normalise a colour (name, #rgb, #rrggbb, rgb()/rgba() with alpha 1) to #rrggbb, else None."""
    v = v.strip().lower()
    if v in _COLOURS:
        return _COLOURS[v]
    m = re.fullmatch(r"#([0-9a-f]{3,8})", v)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        return "#" + h[:6] if len(h) in (6, 8) else None
    m = re.fullmatch(r"rgba?\((\d+)[, ](\d+)[, ](\d+)(?:[,/]\s*([\d.]+))?\)", v)
    if m and (m.group(4) in (None, "1", "1.0")):
        return "#%02x%02x%02x" % tuple(min(255, int(m.group(i))) for i in (1, 2, 3))
    return None


def _far(n, u, lim=-999):
    return n is not None and (n <= lim or (u in ("%", "vw", "vh", "vmin", "vmax") and n <= -100))


def _style_hidden(d: dict) -> bool:
    g = d.get
    if g("display") == "none" or g("visibility") in ("hidden", "collapse"):
        return True
    o, _u = _num(g("opacity", ""))
    if o is not None and o <= 0.01:
        return True
    fs, u = _num(g("font-size", ""))
    if fs is not None and ((u in ("", "px", "pt") and fs < 2) or (u in ("em", "rem", "%") and fs < (0.1 if u != "%" else 10))):
        return True
    ti, tu = _num(g("text-indent", ""))
    if _far(ti, tu):
        return True
    for side in ("left", "top", "right", "bottom", "margin-left", "margin-top", "margin-right", "margin-bottom"):
        n, nu = _num(g(side, ""))
        pos = g("position")
        if _far(n, nu, -100 if pos in ("absolute", "fixed") and not side.startswith("margin") else -999) and (
                side.startswith("margin") or pos in ("absolute", "fixed", "relative")):
            return True
    clip = g("clip", "") + g("clip-path", "")
    if re.search(r"rect\(0(px)?,0(px)?,0(px)?,0(px)?\)|inset\(50%\)|inset\(100%\)|circle\(0|polygon\(0", clip):
        return True
    tf = g("transform", "")
    if "scale(0)" in tf or "scale(0," in tf or "scalex(0)" in tf or "scaley(0)" in tf:
        return True
    for m in re.finditer(r"translate[xy]?\((-?[\d.]+)([a-z%]*)", tf[:300]):
        try:
            if _far(float(m.group(1)), m.group(2)):
                return True
        except ValueError:
            pass
    for dim in ("height", "width", "max-height", "max-width"):
        n, _u = _num(g(dim, ""))
        if n == 0 and g("overflow") == "hidden":
            return True
    fg = _colour(g("color", ""))
    bgv = g("background-color") or g("background") or ""
    bg = _colour(bgv) or next((c for c in map(_colour, bgv.replace("),", ") ").split()) if c), None)
    return bool(fg and bg and fg == bg)


def _compound(sel: str):
    """Last compound of a selector (descendant/child/sibling combinators: ancestry ignored, so it over-drops).
    Returns (tag, id, classes, attrs) or None. Linear, input length bounded by the caller."""
    sel = re.split(r"[\s>+~]+", sel.strip())[-1] if sel.strip() else ""
    if not sel or len(sel) > 200:
        return None
    attrs = [(m.group(1), m.group(2) or "", (m.group(3) or "").strip("\"'")) for m in
             re.finditer(r"\[\s*([\w-]+)\s*(?:([~|^$*]?=)\s*([^\]]*))?\]", sel)]
    base = re.sub(r"\[[^\]]*\]", "", sel)
    m = re.fullmatch(r"([a-z0-9*]*)((?:[.#][\w-]+)*)", base)
    if not m:
        return None
    ids = re.findall(r"#([\w-]+)", m.group(2))
    cls = re.findall(r"\.([\w-]+)", m.group(2))
    tag = m.group(1) if m.group(1) != "*" else ""
    if not (tag or ids or cls or attrs):
        return None
    return tag, (ids[0] if ids else ""), frozenset(cls), tuple(attrs)


def _hidden_selectors(body: str, clock=None, end=None):
    """Linear pass over <style> blocks (total text capped at STYLE_SCAN_MAX): rules whose declarations hide."""
    rules, budget, i = [], STYLE_SCAN_MAX, 0
    low = body.lower()
    while budget > 0 and len(rules) < MAX_RULES:
        st = low.find("<style", i)
        if st < 0:
            break
        gt = low.find(">", st)
        if gt < 0:
            break
        en = low.find("</style", gt)
        en = len(body) if en < 0 else en
        blk = _strip_comments(body[gt + 1:min(en, gt + 1 + budget)])
        budget -= max(1, en - gt - 1)
        i = max(en, gt + 1)
        pos = 0
        while len(rules) < MAX_RULES:
            if clock is not None and clock() > end:
                raise FetchRefused("timeout")
            ob = blk.find("{", pos)
            if ob < 0:
                break
            cb = blk.find("}", ob)
            cb = len(blk) if cb < 0 else cb
            if blk[ob + 1:cb].find("{") >= 0:  # nested block (@media): skip the at-rule header, rescan inside
                pos = ob + 1
                continue
            sel_start = max(blk.rfind("}", 0, ob), blk.rfind(";", 0, ob), blk.rfind("{", 0, ob)) + 1
            sels = blk[sel_start:ob][:2000]
            if sels.lstrip().startswith("@"):
                pos = cb + 1
                continue
            if _style_hidden(_decls(blk[ob + 1:cb])):
                for one in sels.split(",")[:50]:
                    c = _compound(one.lower())
                    if c:
                        rules.append(c)
            pos = cb + 1
    return rules


def _attr_match(attrs: dict, tests) -> bool:
    for name, op, val in tests:
        if name not in attrs:
            return False
        have = str(attrs[name] or "").lower()
        val = val.lower()
        ok = {"": True, "=": have == val, "~=": val in have.split(), "^=": have.startswith(val),
              "$=": have.endswith(val), "*=": val in have, "|=": have == val or have.startswith(val + "-")}.get(op, False)
        if not ok:
            return False
    return True


def _hidden(tag: str, attrs: dict, rules=()) -> bool:
    if "hidden" in attrs or str(attrs.get("aria-hidden", "")).strip().lower() == "true":
        return True
    if tag == "dialog" and "open" not in attrs:
        return True
    if rules:
        cl = set(str(attrs.get("class", "")).lower().split())
        ident = str(attrs.get("id", "")).strip().lower()
        for rt, ri, rc, ra in rules:
            if (not rt or rt == tag) and (not ri or ri == ident) and rc <= cl and _attr_match(attrs, ra):
                return True
    return _style_hidden(_decls(str(attrs.get("style", ""))))


class _Visible(HTMLParser):
    def __init__(self, sel=(), clock=None, end=None):
        super().__init__(convert_charrefs=True)
        self.clock, self.end = clock, end
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []  # (tag, drops_content)
        self.skip = 0
        self.sel = sel

    def handle_starttag(self, tag, attrs):
        if self.clock is not None and self.clock() > self.end:
            raise FetchRefused("timeout")
        if tag in _VOID:
            return
        drop = tag in _DROP or _hidden(tag, dict(attrs), self.sel)
        self.stack.append((tag, drop))
        self.skip += drop
        if not self.skip and tag in _BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # HTML ignores "/>" on non-void elements: <script/>X</script> is an open script (S3)
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                for _t, d in self.stack[i:]:
                    self.skip -= d
                del self.stack[i:]
                break
        if not self.skip and tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
    # handle_comment, handle_decl, handle_pi are intentionally not overridden: comments are dropped


def extract_visible_text(body: str, content_type: str = "text/html", clock=None, end=None) -> str:
    """Pure CPU work. With clock/end it raises FetchRefused("timeout") past the overall deadline (fetch passes them)."""
    if "html" in content_type:
        p = _Visible(_hidden_selectors(body, clock, end), clock, end)
        for i in range(0, len(body), 65536):
            if clock is not None and clock() > end:
                raise FetchRefused("timeout")
            p.feed(body[i:i + 65536])
        p.close()
        body = "".join(p.parts)
    lines = (" ".join(line.split()) for line in body.splitlines())
    return "\n".join(l for l in lines if l)
