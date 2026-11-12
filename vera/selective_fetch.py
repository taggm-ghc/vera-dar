"""Selective fetch into staging: download, hash, extract text, record provenance.

Never raises for an ordinary fetch problem: returns a FetchResult with
ok=False and a specific `error` so the runner can persist the failure and
carry on. Bounded: per-request timeout, max redirects, max bytes.

Safety: URLs come from a search engine and are untrusted. Only http(s) to
public IPs is fetched (redirects re-checked), and fetched text is stored as
data only - it carries no instruction authority.

robots.txt (NONASK-1103 COMP, owner rule R-050): every hop's host is checked before any fetch (RFC 9309: 200 ->
parsed for our User-Agent; 4xx other than 429 -> allowed; 429, 5xx, error or unparseable -> NOT allowed, fail
closed), cached per host in this process for a short TTL. PDFs are recorded as unsupported until a parser is added.
"""
import hashlib
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests
import requests.adapters

from vera import api_permissions
from vera.cost_ledger import CostLedger

USER_AGENT = "VERA-research-fetcher/0.1 (+https://github.com/taggm-ghc/vera-dar; bounded research)"
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/json", "application/xml", "text/xml")


@dataclass
class FetchResult:
    url: str
    ok: bool
    error: str = ""
    content_text: str = ""
    content_hash: str = ""  # sha256 hex of content_text (UTF-8), so vera_vjay.sources.content is re-verifiable
    provenance: dict = field(default_factory=dict)


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def extract_text(body: str, content_type: str) -> str:
    if "html" in content_type:
        p = _TextExtractor()
        p.feed(body)
        body = "".join(p.parts)
    lines = (" ".join(line.split()) for line in body.splitlines())
    return "\n".join(l for l in lines if l)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parsers_agree(url: str, parts) -> bool:
    """CMF-1022 NEW-1: urlsplit, urllib3 and requests disagree on backslashes (and some escapes) in the authority, so
    the name validated here could differ from the host connected to. Refuse control/whitespace characters anywhere, a
    backslash in the authority (scheme://... up to the first / ? #), and any case where urllib3's host differs from
    urlsplit's. A backslash in the path after the authority is allowed: both parsers agree on the host there and
    requests percent-encodes it, so it cannot change the connection target."""
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in url):
        return False
    rest = url.split("://", 1)[1] if "://" in url else url
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    # INTENTIONAL URL-VALIDATION POLICY (OD-CMF-1022g): no percent-encoded host syntax. A valid DNS name or IP literal
    # never contains "%", so any "%" in the host (%40 @, %3A :, %2F /, %5C \\, %2E ., %00, %0A, %09, any case) is refused.
    if "\\" in authority or "%" in (parts.hostname or ""):
        return False
    try:
        from urllib3.util import parse_url
        h3 = parse_url(url).host
    except Exception:  # noqa: BLE001 - any parse error refuses
        return False
    if not h3:
        return False
    return h3.strip("[]").lower() == (parts.hostname or "").lower()


def is_public_url(url: str) -> tuple[bool, str]:
    _PIN.slot = None  # CMF-1022-I3 R1: clear first, so a refused check never leaves an earlier pin usable
    try:
        parts = urlsplit(url)
    except ValueError:
        return False, "unparseable URL"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False, "only http(s) URLs are fetched"
    if not _parsers_agree(url, parts):
        return False, "URL parsers disagree on the host (backslash, control character or escape)"
    if parts.username is not None or parts.password is not None:  # CMF-1022 2b: no credential-bearing URLs (before DNS)
        return False, "credentials in the URL are not allowed"
    try:
        port = parts.port  # raises ValueError on an invalid port
    except ValueError:
        return False, "invalid port"
    if port not in (None, 80, 443):  # CMF-1022 3b: web ports only
        return False, "only ports 80 and 443 are fetched"
    if (parts.scheme, port) in (("http", 443), ("https", 80)):  # OD-CMF-1022g: scheme/port pairs must match
        return False, "scheme and port do not match (http:80 / https:443 only)"
    eff_port = port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parts.hostname, eff_port)
    except socket.gaierror as exc:
        return False, f"dns failure: {exc}"
    addrs: list[str] = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not _ip_is_public(ip):  # CMF-1022-I2: ANY non-public address refuses the whole answer
            return False, f"blocked non-public address {ip}"
        if str(ip) not in addrs:
            addrs.append(str(ip))
    if not addrs:
        return False, "dns returned no address"
    _PIN.slot = (url, parts.scheme, parts.hostname, eff_port, tuple(addrs), time.monotonic())  # E1: pin for this hop
    return True, ""


# ------------------------------------------------------------------------------------------------ CMF-1022-I2 (E1/A4)
_NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")      # RFC 6052 well-known prefix: IPv4 in the low 32 bits
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")  # RFC 8215 local-use: embedding position is network-specific
_V4_COMPAT = ipaddress.ip_network("::/96")             # RFC 4291 s2.5.5.1 deprecated IPv4-compatible (::a.b.c.d)


def _ip_is_public(ip) -> bool:
    """is_global plus NEW-2. IPv4-compatible (::a.b.c.d) and NAT64 well-known-prefix (64:ff9b::/96) addresses are
    judged by their EMBEDDED IPv4 address (a public embedded IPv4 is allowed: DNS64 legitimately synthesizes
    64:ff9b::<public v4>). The local-use NAT64 prefix 64:ff9b:1::/48 is refused outright: the IPv4 position depends on
    the local prefix length, and IANA marks the block not globally reachable. IPv4-mapped (::ffff:0:0/96) is already
    refused by is_global and is also checked by its embedded address."""
    if not ip.is_global:
        return False
    if ip.version == 6:
        if ip in _NAT64_LOCAL:
            return False
        if ip.ipv4_mapped is not None and not ip.ipv4_mapped.is_global:
            return False
        if ip in _NAT64_WKP or ip in _V4_COMPAT:
            if not ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF).is_global:
                return False
    return True


class _PinSlot(threading.local):
    slot = None


# One pin per thread: the addresses is_public_url just validated for ONE exact URL string. pinned_get consumes it, so a
# hop is resolved once (in the check) and connected to a validated IP without a second resolution.
_PIN = _PinSlot()
# Calibration (CMF-1022-I3 R3): both bounds are planner priors, UNCALIBRATED on real data (no live run, 2026-10-22).
# - PIN_MAX_AGE_S: a pin is consumed by the get that directly follows its check in the same thread; the only gap
#   between them is the per-host pacer sleep (<= 3 s, arXiv) and in-process work, so 60 s is far above any expected
#   gap while bounding how stale a still-validated-public address can be. Exceeding it costs one extra resolution
#   (fail safe: re-validate), never an unchecked connect.
# - PIN_MAX_ADDRS: one fallback address covers a dead first address (e.g. dual-stack v6 answer with no v6 route)
#   while capping the connect wait in a hop at 2 x timeout; only addresses from the validated set are ever tried.
PIN_MAX_AGE_S = 60.0  # an older pin is not used; pinned_get then re-validates (one fresh resolution) - never unchecked
PIN_MAX_ADDRS = 2     # at most this many validated addresses are tried, and only after a connect-phase failure


class PinnedRefused(requests.exceptions.InvalidURL):
    """pinned_get refused the URL (failed validation); nothing was sent."""


class ApiRefused(PinnedRefused, api_permissions.ApiDenied):
    """API-URL-GATE-1108: pinned_get refused an API URL that config/api_permissions.json does not permit for this
    purpose (or the config is unusable: every URL refused). Nothing was sent. A RequestException (so generic callers
    treat it as a failed fetch) AND an ApiDenied (code R-API-DENIED / R-API-COND / R-API-SCOPE, entry, why)."""

    def __init__(self, err: "api_permissions.ApiDenied"):
        PinnedRefused.__init__(self, f"blocked: {err.code} ({err.entry or 'no_entry'}: {err.why})")
        self.code, self.entry, self.why = err.code, err.entry, err.why


def api_gate(url: str, purpose: str | None = None):
    """API-URL-GATE-1108: api_permissions.gate_url for a generic fetch hop; a refusal is raised as ApiRefused.
    Returns None (not an API URL) or the permitting Decision."""
    try:
        return api_permissions.gate_url(url, purpose)
    except api_permissions.ApiDenied as e:
        raise ApiRefused(e) from None


class DocumentDenied(PinnedRefused):
    """DENIAL-COVERAGE-1111: the document URL is in the access-denial memory (an earlier 401/403 on this very
    endpoint, OD-1111-ACCESS-DENIED-NO-BYPASS). Nothing was sent. A PinnedRefused, so callers treat it as a failed hop."""


_ROBOTS_ENDPOINT_PATHS = ("/robots.txt", "/.well-known/tdmrep.json")  # exact paths only (the "robots endpoint")


def _is_document_request(url: str, purpose: str | None, api_decision) -> bool:
    """False for the robots endpoints (exact path) and for a gated API hop whose effective purpose is metadata;
    True otherwise. An API hop with purpose fulltext (or none: the guarded default) is a download endpoint."""
    try:
        path = urlsplit(url).path
    except ValueError:
        return True
    if path in _ROBOTS_ENDPOINT_PATHS:
        return False
    if api_decision is not None and (purpose or api_permissions.current_purpose(url)) == "metadata":
        return False
    return True


def document_request_guard(url: str, purpose: str | None = None, api_decision=None) -> None:
    """DENIAL-COVERAGE-1111: raise DocumentDenied, before anything is sent, when a document request targets a URL
    that earlier answered 401/403 (memory: acq_discovered.is_denied; keyed host + path). The single guard shared by
    pinned_get and public_fetch.default_transport; existing caller checks stay as the first line."""
    if not _is_document_request(url, purpose, api_decision):
        return
    from vera import acq_discovered as _ad
    if _ad.is_denied(urls=(url,)):
        raise DocumentDenied("blocked: access_denied (earlier 401/403 on this document; not requested again)")


def document_response_note(url: str, status, purpose: str | None = None, api_decision=None) -> None:
    """DENIAL-COVERAGE-1111: remember a 401/403 on a document request (never raises). Metadata-API and robots
    endpoint refusals are not remembered. One sanitised cause line (host only)."""
    try:
        if status not in (401, 403) or not _is_document_request(url, purpose, api_decision):
            return
        from vera import acq_discovered as _ad
        if _ad.is_denied(urls=(url,)):
            return
        _ad.deny(_ad.denial_ids(urls=(url,)))
        _ad.cause_log(urlsplit(url).hostname or "", "web", "access_denied", f"STATUS_{int(status)}", False, status)
    except Exception:  # noqa: BLE001  diagnostics and memory never break a fetch
        pass


def _take_pin(url: str):
    pin, _PIN.slot = _PIN.slot, None
    if pin is None or pin[0] != url or time.monotonic() - pin[5] > PIN_MAX_AGE_S:
        return None
    return pin


class _PinnedTLSAdapter(requests.adapters.HTTPAdapter):
    """HTTPS adapter for ONE hop: the request URL carries the validated IP literal, so urllib3 never resolves a name;
    SNI (server_hostname) and the certificate hostname check (assert_hostname) use the ORIGINAL hostname. Certificate
    verification is untouched (requests' verify=True -> cert_reqs=CERT_REQUIRED, CA bundle as before)."""

    def __init__(self, tls_hostname: str):
        self._tls_hostname = tls_hostname
        super().__init__(max_retries=0)

    def init_poolmanager(self, *a, **kw):
        kw["server_hostname"] = self._tls_hostname
        kw["assert_hostname"] = self._tls_hostname
        super().init_poolmanager(*a, **kw)


def _connect_phase_error(exc) -> bool:
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    import urllib3
    reason = getattr(exc.args[0] if exc.args else None, "reason", None)
    return isinstance(reason, urllib3.exceptions.NewConnectionError)


def pinned_get(url: str, *, headers=None, timeout=None, allow_redirects=False, stream=False, params=None,
               api_purpose: str | None = None):
    """requests.get-compatible GET that connects ONLY to an address validated for this exact URL (E1/A4 DNS
    rebinding). Uses the pin left by is_public_url(url) in this thread, else runs is_public_url(url) itself (one
    resolution). Never follows redirects (callers re-validate and re-pin every hop). Env proxies, netrc and env CA
    bundles are ignored (trust_env=False): a proxy would resolve the name itself and defeat pinning."""
    if allow_redirects:
        raise ValueError("pinned_get never follows redirects; re-validate each hop")
    # API-URL-GATE-1108: every hop on this transport (search-result URLs, portal links, redirect targets, robots.txt)
    # is checked against config/api_permissions.json (and 1108b: unlisted API-like URLs, IP literals) before anything
    # is resolved or sent. api_purpose: the caller's explicit purpose (1108b (6)); None -> the guarded fallback
    # (current_purpose(url), default "fulltext"). Refused -> ApiRefused. A permitted API URL gets the entry's
    # User-Agent and goes through the entry's OWN rate portal (1108b (4)) unless limited_transport already holds it.
    d = api_gate(url, api_purpose)
    if params:  # 1108b review (1): the URL actually sent carries params; gate that one too (same purpose)
        api_gate(requests.Request("GET", url, params=params).prepare().url,
                 api_purpose or api_permissions.current_purpose(url))
    document_request_guard(url, api_purpose, d)  # DENIAL-COVERAGE-1111: a denied document is never requested again
    if d is not None:
        headers = api_permissions.headers_for(d, headers)
        portal = str(d.conditions["rate_portal"])
        if api_permissions.held_portal() != portal:
            from vera import rate_limit
            with api_permissions.holding(portal):  # the inner call re-gates and then sends under this permit
                return rate_limit.call_with_limit(rate_limit.get_registry().portal(portal), pinned_get, url,
                                                  headers=headers, timeout=timeout, stream=stream, params=params,
                                                  api_purpose=api_purpose)
    pin = _take_pin(url)
    if pin is None:
        ok, why = is_public_url(url)
        pin = _take_pin(url) if ok else None
        if pin is None:
            raise PinnedRefused(f"blocked: {why or 'no validated address'}")
    _url, scheme, hostname, port, addrs, _t = pin
    parts = urlsplit(url)
    name = hostname.rstrip(".")
    default = 443 if scheme == "https" else 80
    host_hdr = (f"[{name}]" if ":" in name else name) + ("" if port == default else f":{port}")
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    hdrs = dict(headers or {})
    for k in [k for k in hdrs if k.lower() == "host"]:
        del hdrs[k]
    hdrs["Host"] = host_hdr
    last = None
    for ip in addrs[:PIN_MAX_ADDRS]:
        s = requests.Session()
        s.trust_env = False
        if scheme == "https":
            s.mount("https://", _PinnedTLSAdapter(name))
        lit = f"[{ip}]" if ":" in ip else ip
        try:
            resp = s.get(f"{scheme}://{lit}:{port}{target}", headers=hdrs, timeout=timeout, allow_redirects=False,
                         stream=stream, params=params, verify=True)
        except requests.exceptions.ConnectionError as exc:
            s.close()
            if not _connect_phase_error(exc):
                raise
            last = exc
            continue
        if not stream:
            s.close()
        document_response_note(url, resp.status_code, api_purpose, d)
        return resp
    raise last


# Why: arXiv's published limit is one request per 3 s; we apply it to abstract-page fetches too, not only to the
# API call, so a burst of page fetches cannot exceed what R1 approved. Mirrors config search.arxiv.min_interval_s
# (a test pins the two values together); other hosts are not paced.
PACED_HOSTS: dict[str, float] = {"arxiv.org": 3.0}


class HostPacer:
    """Minimum spacing between fetches to the same paced host. Clock and sleep are injectable for tests."""

    def __init__(self, intervals: dict[str, float] | None = None, *, clock=time.monotonic, sleep=time.sleep):
        self.intervals = dict(PACED_HOSTS if intervals is None else intervals)
        self.clock, self.sleep = clock, sleep
        self.last: dict[str, float] = {}

    def wait(self, url: str) -> None:
        host = (urlsplit(url).hostname or "").lower()
        interval = self.intervals.get(host)
        if not interval:
            return
        gap = self.last.get(host, float("-inf")) + interval - self.clock()
        if gap > 0:
            from vera import progress as _prog  # NOISY-UI: politeness (crawl-delay) wait is visible
            with _prog.waiting(_prog.ACT_POLITE, gap):
                self.sleep(gap)
        self.last[host] = self.clock()


_DEFAULT_PACER = HostPacer()

# Robots Crawl-delay gate: a request that would arrive before last_request + delay is NEVER sent early (refused instead).
_CRAWL_LOCK = threading.Lock()
_CRAWL_LAST: dict = {}  # host -> monotonic time of the last request actually sent (insertion order = oldest first)
_CRAWL_MAX_HOSTS = 512


ROBOTS_TTL_S = 300.0       # short per-host in-process cache of a parsed robots.txt (also bounds staleness)
ROBOTS_FAIL_TTL_S = 60.0   # a failed robots fetch (fail closed) is remembered briefly, not retried per URL
ROBOTS_MAX_BYTES = 512 * 1024
_ROBOTS: dict = {}         # host -> (monotonic time, parser | True (allow all) | False (fail closed))
_ROBOTS_LOCK = threading.Lock()
_ROBOTS_SIGNAL: dict = {}  # WP6-P2b D2: (scheme, host, ua) -> robots.txt Content-Signal values (signal text only)


_ROBOTS_TEXT: dict = {}    # ASK-PARALLEL-FIX-1105: host -> raw robots.txt text (Crawl-delay only; insertion order)
_ROBOTS_TEXT_MAX = 512


def robots_cache_clear() -> None:
    with _ROBOTS_LOCK:
        _ROBOTS.clear()
        _ROBOTS_SIGNAL.clear()
        _ROBOTS_TEXT.clear()
        _CACHED_LOGGED.clear()


def remember_robots_text(host: str, text: str | None) -> None:
    """Keep the raw robots.txt of a host (None forgets it), so a non-integer Crawl-delay, which urllib.robotparser
    drops, is still honoured. Bounded; never raises."""
    try:
        host = (host or "").lower()
        with _ROBOTS_LOCK:
            _ROBOTS_TEXT.pop(host, None)
            if text is not None:
                _ROBOTS_TEXT[host] = str(text)[:ROBOTS_MAX_BYTES]
                while len(_ROBOTS_TEXT) > _ROBOTS_TEXT_MAX:
                    _ROBOTS_TEXT.pop(next(iter(_ROBOTS_TEXT)))
    except Exception:  # noqa: BLE001
        pass


def parse_crawl_delay(text: str, user_agent=None) -> float | None:
    """Crawl-delay of the robots.txt group that applies to user_agent (same matching as urllib.robotparser: a group
    naming a substring of our product token wins, else the * group). None = no Crawl-delay line in that group; a
    float >= 0 = the delay (fractions kept; several lines: the largest); math.inf = a malformed value (empty, not a
    number, negative or not finite), which callers treat as "do not fetch this host"."""
    import math
    token = (user_agent or USER_AGENT).split("/")[0].lower()
    groups, cur, in_agents = [], None, False
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, val = (x.strip() for x in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if not in_agents:
                cur = {"agents": [], "delays": []}
                groups.append(cur)
            cur["agents"].append(val.lower())
            in_agents = True
        elif cur is not None:
            in_agents = False
            if key == "crawl-delay":
                try:
                    v = float(val)
                except ValueError:
                    v = math.inf
                cur["delays"].append(v if math.isfinite(v) and v >= 0 else math.inf)
    pick = next((g for g in groups if any(a != "*" and a and a in token for a in g["agents"])), None) \
        or next((g for g in groups if "*" in g["agents"]), None)
    delays = list(pick["delays"]) if pick else []
    try:  # the largest Crawl-delay among the groups that apply; advisory (hyperscaler) groups kept as pacing (H13)
        from vera import robots_ai as _ra
        ai = {str(t).lower() for t in _ra.load_config()["advisory_agent_tokens"]}
        for g in groups:
            if any(a in ai for a in g["agents"]):
                delays += g["delays"]
        own = {_ra._base(str(t)) for t in _ra.load_config()["own_tokens"]}  # ROBOTS-FOLLOWUP-1108: any own-token group
        for g in groups:
            if any(_ra._base(a) in own for a in g["agents"]):
                delays += g["delays"]
    except Exception:  # noqa: BLE001
        delays.append(math.inf)  # config unreadable: fail closed
    return max(delays) if delays else None


def robots_text_crawl_delay(host: str, user_agent=None) -> float | None:
    """parse_crawl_delay of the remembered robots.txt of host; None when no text is remembered. Never raises."""
    try:
        with _ROBOTS_LOCK:
            txt = _ROBOTS_TEXT.get((host or "").lower())
        return None if txt is None else parse_crawl_delay(txt, user_agent)
    except Exception:  # noqa: BLE001
        return None


_CACHED_LOGGED: dict = {}  # host -> monotonic time of the last cached=1 cause line


def _cause_log_sf(host, cause, cached, status=None) -> None:
    """WPL-1111 R3: the stdout cause line (vera.robots_cause), hop=portal. Never raises."""
    try:
        from vera import acq_discovered as _ad
        _ad.cause_log(host, "portal", "robots_unreadable", ("CACHED_" if cached else "") + str(cause), cached, status)
    except Exception:  # noqa: BLE001
        pass


def _cause_cached(parts, host, hit, hold, now) -> None:
    """One cached=1 line per host per TTL (the fail TTL, or ROBOTS_TTL_S for a held 4xx stop)."""
    ttl = ROBOTS_TTL_S if hold else ROBOTS_FAIL_TTL_S
    last = _CACHED_LOGGED.get(host)
    if last is None or now - last >= ttl:
        _CACHED_LOGGED[host] = now
        _cause_log_sf(host, hit[2] if len(hit) > 2 and hit[2] else "UNKNOWN", True)


def robots_allowed(url, *, http_get, url_check, pacer=None, timeout_s=8.0, clock=time.monotonic,
                   user_agent=None) -> bool:
    """True only if robots.txt of the URL's host permits USER_AGENT to fetch url. Fails closed. Never raises."""
    from urllib import robotparser
    ua = user_agent or USER_AGENT
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return False
    now = clock()
    with _ROBOTS_LOCK:
        hit = _ROBOTS.get((parts.scheme, host, ua))
    # WPL-1111 RC-1: a stop on a robots.txt 4xx (R1) is held for ROBOTS_TTL_S as the old "allow" was (cause recorded in
    # the same write as a third tuple element; a 2-tuple False, as older fixtures write, keeps the fail TTL)
    hold = hit is not None and hit[1] is False and len(hit) > 2 and hit[2] in ("STATUS_401", "STATUS_403", "STATUS_4XX")
    if hit is not None and now - hit[0] < (ROBOTS_FAIL_TTL_S if hit[1] is False and not hold else ROBOTS_TTL_S):
        rp = hit[1]
        if rp is False:
            _cause_cached(parts, host, hit, hold, now)
    else:
        rp, signal, cause, _status = False, "", "NETWORK", None
        try:
            cur = f"{parts.scheme}://{parts.netloc}/robots.txt"
            for _hop in range(MAX_REDIRECTS + 1):
                ok, _why = url_check(cur)
                if not ok:
                    cause = "HOST"
                    break
                (pacer or _DEFAULT_PACER).wait(cur)
                resp = http_get(cur, headers={"User-Agent": ua}, timeout=timeout_s,
                                allow_redirects=False, stream=True)
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                    cur = urljoin(cur, resp.headers["Location"])
                    continue
                if resp.status_code == 200:
                    body = b"".join(resp.iter_content(65536))[:ROBOTS_MAX_BYTES]
                    p = robotparser.RobotFileParser()
                    txt = body.decode("utf-8", "ignore")
                    p.parse(txt.splitlines())
                    from vera import robots_ai as _ra  # OD-1106-ROBOTS-AI: AI-agent groups and statements
                    _ra.annotate(p, txt)
                    rp = p
                    remember_robots_text(host, txt)  # ASK-PARALLEL-FIX-1105: non-integer Crawl-delay
                    from vera import source_permissions as _sp
                    signal = " ".join(_sp.robots_content_signal(txt).split()).lower()[:300]
                elif resp.status_code in (404, 410):
                    rp = True
                elif resp.status_code == 429:
                    cause = "RATELIMITED"  # stays a stop (fail TTL); only the cause line is truthful
                elif 400 <= resp.status_code < 500:
                    # R1 (OD-1111-ROBOTS-4XX): only 404/410 mean "no restrictions"; any other 4xx stops (a recorded
                    # deviation from RFC 9309 section 2.3.1.3), held for ROBOTS_TTL_S so no request is added
                    cause = "STATUS_401" if resp.status_code == 401 else "STATUS_403" if resp.status_code == 403 \
                        else "STATUS_4XX"
                else:
                    cause = "STATUS_5XX" if resp.status_code >= 500 else "STATUS_OTHER"
                _status = resp.status_code
                break
            else:
                cause = "REDIRECTS"  # the redirect cap was used up without an answer
        except Exception:  # noqa: BLE001  any error: fail closed
            rp = False
        with _ROBOTS_LOCK:
            _ROBOTS[(parts.scheme, host, ua)] = (now, rp, cause if rp is False else None)
            _ROBOTS_SIGNAL[(parts.scheme, host, ua)] = signal
            while len(_ROBOTS) > 64:
                k = next(iter(_ROBOTS))
                _ROBOTS.pop(k)
                _ROBOTS_SIGNAL.pop(k, None)
        if rp is False:
            _cause_log_sf(host, cause, False, _status)
    if rp is True:
        return True
    if rp is False:
        return False
    try:
        from vera import robots_ai as _ra  # RFC 9309 matcher for the own family, else *, plus the AI statement (H13)
        return _ra.allowed(rp, ua, url) and _ra.block_reason(rp, url) is None
    except Exception:  # noqa: BLE001  fail closed
        return False


def robots_ai_reason(url, user_agent=None) -> str | None:
    """robots_ai_statement when the cached robots.txt of url's host states that AI crawlers/assistants are blocked,
    else None (hyperscaler-only groups are advisory, H13). Never fetches, never raises."""
    try:
        from vera import robots_ai as _ra
        parts = urlsplit(url)
        with _ROBOTS_LOCK:
            hit = _ROBOTS.get((parts.scheme, (parts.hostname or "").lower(), user_agent or USER_AGENT))
        return None if hit is None else _ra.block_reason(hit[1], url)
    except Exception:  # noqa: BLE001
        return None


def robots_content_signal(url, user_agent=None) -> str:
    """WP6-P2b D2: the Content-Signal values of the robots.txt robots_allowed already parsed for this URL's host
    ("" when none or not cached). Never fetches, never raises."""
    try:
        parts = urlsplit(url)
        k = (parts.scheme, (parts.hostname or "").lower(), user_agent or USER_AGENT)
        with _ROBOTS_LOCK:
            sig, hit = _ROBOTS_SIGNAL.get(k, ""), _ROBOTS.get(k)
        try:  # A1: per-path ai-train=no from the __usage_ai-training group; informational, never blocks
            from vera import robots_ai as _ra
            use = _ra.usage_signals(hit[1], url) if hit is not None else ""
        except Exception:  # noqa: BLE001
            use = ""
        return ", ".join(v for v in (sig, use) if v)
    except Exception:  # noqa: BLE001
        return ""


def robots_no_ai(url, user_agent=None) -> bool:
    """True when the host's robots.txt carries a no-AI signal (Content-Signal ai-input=no), as source_permissions
    evaluates it: such a page may not be used for an answer, so it is not requested at all on this path."""
    from vera import source_permissions as _sp
    return _sp.robots_no_ai(_sp.robots_entries(robots_content_signal(url, user_agent)))


def robots_crawl_delay(url, user_agent=None) -> float:
    """NONASK-1103 WP6-P2: the Crawl-delay (seconds) of the robots.txt already parsed for this URL's host by
    robots_allowed (in-process cache; this UA, else the * group). 0 when none or not cached. Never fetches, never raises."""
    try:
        ua = user_agent or USER_AGENT
        parts = urlsplit(url)
        with _ROBOTS_LOCK:
            hit = _ROBOTS.get((parts.scheme, (parts.hostname or "").lower(), ua))
        rp = hit[1] if hit is not None else None
        if rp in (None, True, False):
            return 0.0
        t = robots_text_crawl_delay(parts.hostname or "", ua)  # ASK-PARALLEL-FIX-1105: fractions; inf = malformed
        if t is not None:
            return float(t)
        from vera import robots_ai as _ra
        v = _ra.crawl_delay(rp, ua)
        return float(v) if v else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def crawl_reserve(host: str, delay: float, clock=time.monotonic) -> bool:
    """Atomically: True and record "now" as the host's last request time if >= delay seconds have passed (or delay is
    falsy); False (nothing recorded) if it would be early. Call immediately before sending a request. Thread-safe."""
    if not delay or delay <= 0:
        return True
    host = (host or "").lower()
    with _CRAWL_LOCK:
        now = clock()
        last = _CRAWL_LAST.get(host)
        if last is not None and now < last + delay:
            return False
        _CRAWL_LAST.pop(host, None)
        _CRAWL_LAST[host] = now
        while len(_CRAWL_LAST) > _CRAWL_MAX_HOSTS:
            _CRAWL_LAST.pop(next(iter(_CRAWL_LAST)))
        return True


def crawl_wait_s(host: str, delay: float, clock=time.monotonic) -> float:
    """ASK-PARALLEL-FIX-1105: seconds until crawl_reserve(host, delay) would succeed (0.0 now). Records nothing; a
    caller that may wait sleeps this long, then calls crawl_reserve (which stays the only gate). Never raises."""
    try:
        if not delay or delay <= 0:
            return 0.0
        with _CRAWL_LOCK:
            last = _CRAWL_LAST.get((host or "").lower())
            return 0.0 if last is None else max(0.0, last + float(delay) - clock())
    except Exception:  # noqa: BLE001
        return 0.0


def reset_for_tests() -> None:
    with _CRAWL_LOCK:
        _CRAWL_LAST.clear()


def _download(url, *, timeout_s, http_get, url_check, pacer, robots_ok=None, crawl_delays=None,
              crawl_clock=time.monotonic):
    """Guarded GET shared by fetch_candidate and fetch_html: public-URL check on every hop, per-host pacing,
    manual redirects (at most MAX_REDIRECTS), text content types only, MAX_BYTES cap.
    Returns (raw, ctype, final_url, hops, encoding, error); error is None on success."""
    current, hops = url, 0
    try:
        while True:
            # API-URL-GATE-1108: an API URL (first URL or redirect target) needs its api_permissions entry for
            # "fulltext" BEFORE robots.txt or the page is requested, whatever transport was injected.
            try:
                dec = api_gate(current, "fulltext")
            except ApiRefused as e:
                return None, None, None, hops, None, f"blocked: {e.code}"
            try:  # DENIAL-COVERAGE-1111: first line, before robots.txt or the page is requested
                document_request_guard(current, "fulltext", dec)
            except DocumentDenied:
                return None, None, None, hops, None, "blocked: access_denied"
            ok, why = url_check(current)
            if not ok:
                return None, None, None, hops, None, f"blocked: {why}"
            if not (robots_ok or robots_allowed)(current, http_get=http_get, url_check=url_check, pacer=pacer,
                                                 timeout_s=timeout_s):
                ai = robots_ai_reason(current) if robots_ok is None else None  # OD-1106-ROBOTS-AI
                return None, None, None, hops, None, (f"blocked: {ai}" if ai else
                                                      "blocked: robots.txt disallows or could not be read")
            if robots_ok is None and robots_no_ai(current):  # WP6-P2b D2: robots.txt Content-Signal ai-input=no
                return None, None, None, hops, None, "blocked: robots.txt no-AI signal"
            (pacer or _DEFAULT_PACER).wait(current)
            cd = (crawl_delays or {}).get((urlsplit(current).hostname or "").lower())
            if not cd and robots_ok is None:  # WP6-P2: the Crawl-delay robots_allowed just parsed, same per-host gate
                cd = robots_crawl_delay(current)
            if cd == float("inf"):  # ASK-PARALLEL-FIX-1105: malformed Crawl-delay: do not fetch this host at all
                return None, None, None, hops, None, "crawl_delay"
            if cd and not crawl_reserve(urlsplit(current).hostname or "", cd, crawl_clock):
                return None, None, None, hops, None, "crawl_delay"
            hdrs = {"User-Agent": USER_AGENT}
            # 1108b (4)(6): purpose passed explicitly to a gated transport (pinned_get, which also takes the permitted
            # API hop's own rate portal); a skip is a rate limit, nothing sent.
            extra = {"api_purpose": "fulltext"} if api_permissions.accepts_purpose(http_get) else {}
            from vera import rate_limit
            try:
                resp = http_get(current, headers=hdrs, timeout=timeout_s, allow_redirects=False, stream=True, **extra)
            except rate_limit.RateLimited:
                return None, None, None, hops, None, "rate_limited"
            except DocumentDenied:  # raised by the transport guard (a redirect target seen as denied meanwhile)
                return None, None, None, hops, None, "blocked: access_denied"
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                hops += 1
                if hops > MAX_REDIRECTS:
                    return None, None, None, hops, None, f"too many redirects (>{MAX_REDIRECTS})"
                try:
                    current = urljoin(current, resp.headers["Location"])
                except ValueError:  # CMF-1022 NEW-1: an unparseable Location is a blocked hop, not a crash
                    return None, None, None, hops, None, "blocked: unparseable redirect"
                continue
            break
        if resp.status_code != 200:
            return None, None, None, hops, None, f"http {resp.status_code}"
        try:  # ROBOTS-FOLLOWUP-1108: Content-Usage ai-use=n on the page response bars evidence use (fail closed)
            from vera import source_permissions as _sp
            if _sp.content_usage_signal(resp.headers.get("Content-Usage", "")) == "ai-input=no":
                return None, None, None, hops, None, "blocked: Content-Usage ai-use=n"
        except Exception:  # noqa: BLE001
            return None, None, None, hops, None, "blocked: Content-Usage unreadable"
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype not in TEXT_TYPES:
            return None, None, None, hops, None, f"unsupported content-type {ctype or 'missing'!r}"
        chunks, size = [], 0
        for chunk in resp.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                return None, None, None, hops, None, f"body exceeds {MAX_BYTES} bytes"
            chunks.append(chunk)
        raw = b"".join(chunks)
    except requests.RequestException as exc:
        return None, None, None, hops, None, f"{type(exc).__name__}: {exc}"
    return raw, ctype, current, hops, getattr(resp, "encoding", None) or "utf-8", None


def fetch_html(url: str, *, timeout_s: float = 8.0, http_get=None, url_check=is_public_url,
               pacer: HostPacer | None = None, robots_ok=None, crawl_delays=None,
               crawl_clock=time.monotonic) -> tuple[str | None, str | None, str | None]:
    """Raw decoded body (markup kept, for link and metadata extraction) under the same guards.
    Returns (body, final_url, error). Default transport: pinned_get (CMF-1022-I2, E1/A4)."""
    http_get = http_get or pinned_get
    raw, _ctype, final, _hops, enc, err = _download(url, timeout_s=timeout_s, http_get=http_get,
                                                     url_check=url_check, pacer=pacer, robots_ok=robots_ok,
                                                     crawl_delays=crawl_delays, crawl_clock=crawl_clock)
    if err:
        return None, None, err
    return raw.decode(enc, errors="replace"), final, None


def fetch_candidate(
    url: str,
    *,
    timeout_s: float = 15.0,
    http_get=None,
    url_check=is_public_url,
    ledger: CostLedger | None = None,
    pacer: HostPacer | None = None,
    robots_ok=None,
    crawl_delays=None,
    crawl_clock=time.monotonic,
) -> FetchResult:
    http_get = http_get or pinned_get  # CMF-1022-I2: pinned transport by default
    raw, ctype, current, hops, enc, err = _download(url, timeout_s=timeout_s, http_get=http_get,
                                                    url_check=url_check, pacer=pacer, robots_ok=robots_ok,
                                                    crawl_delays=crawl_delays, crawl_clock=crawl_clock)
    if err:
        return _fail(url, err, ledger)
    text = extract_text(raw.decode(enc, errors="replace"), ctype)
    if not text.strip():
        return _fail(url, "no extractable text", ledger)
    if ledger:
        ledger.record(kind="fetch", provider="http", cost_usd=0.0, units={"bytes": len(raw)})
    return FetchResult(
        url=url, ok=True, content_text=text, content_hash=sha256_hex(text.encode("utf-8")),
        provenance={
            "requested_url": url, "final_url": current, "http_status": 200, "content_type": ctype,
            "bytes": len(raw), "raw_sha256": sha256_hex(raw), "redirects": hops, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "fetcher": USER_AGENT,
        },
    )


def _fail(url: str, error: str, ledger: CostLedger | None) -> FetchResult:
    if ledger:
        ledger.record(kind="fetch", provider="http", cost_usd=0.0, ok=False, detail=f"{url}: {error}")
    return FetchResult(url=url, ok=False, error=error)
