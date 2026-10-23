"""Selective fetch into staging: download, hash, extract text, record provenance.

Never raises for an ordinary fetch problem: returns a FetchResult with
ok=False and a specific `error` so the runner can persist the failure and
carry on. Bounded: per-request timeout, max redirects, max bytes.

Safety: URLs come from a search engine and are untrusted. Only http(s) to
public IPs is fetched (redirects re-checked), and fetched text is stored as
data only - it carries no instruction authority.

Accepted divergence: robots.txt is not consulted (single-URL, low-volume,
research fetches); PDFs are recorded as unsupported until a parser is added.
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

from vera.cost_ledger import CostLedger

USER_AGENT = "VERA-research-fetcher/0.1 (+bounded research; contact via repo owner)"
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


def pinned_get(url: str, *, headers=None, timeout=None, allow_redirects=False, stream=False, params=None):
    """requests.get-compatible GET that connects ONLY to an address validated for this exact URL (E1/A4 DNS
    rebinding). Uses the pin left by is_public_url(url) in this thread, else runs is_public_url(url) itself (one
    resolution). Never follows redirects (callers re-validate and re-pin every hop). Env proxies, netrc and env CA
    bundles are ignored (trust_env=False): a proxy would resolve the name itself and defeat pinning."""
    if allow_redirects:
        raise ValueError("pinned_get never follows redirects; re-validate each hop")
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
            self.sleep(gap)
        self.last[host] = self.clock()


_DEFAULT_PACER = HostPacer()


def _download(url, *, timeout_s, http_get, url_check, pacer):
    """Guarded GET shared by fetch_candidate and fetch_html: public-URL check on every hop, per-host pacing,
    manual redirects (at most MAX_REDIRECTS), text content types only, MAX_BYTES cap.
    Returns (raw, ctype, final_url, hops, encoding, error); error is None on success."""
    current, hops = url, 0
    try:
        while True:
            ok, why = url_check(current)
            if not ok:
                return None, None, None, hops, None, f"blocked: {why}"
            (pacer or _DEFAULT_PACER).wait(current)
            resp = http_get(current, headers={"User-Agent": USER_AGENT}, timeout=timeout_s,
                            allow_redirects=False, stream=True)
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
               pacer: HostPacer | None = None) -> tuple[str | None, str | None, str | None]:
    """Raw decoded body (markup kept, for link and metadata extraction) under the same guards.
    Returns (body, final_url, error). Default transport: pinned_get (CMF-1022-I2, E1/A4)."""
    http_get = http_get or pinned_get
    raw, _ctype, final, _hops, enc, err = _download(url, timeout_s=timeout_s, http_get=http_get,
                                                     url_check=url_check, pacer=pacer)
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
) -> FetchResult:
    http_get = http_get or pinned_get  # CMF-1022-I2: pinned transport by default
    raw, ctype, current, hops, enc, err = _download(url, timeout_s=timeout_s, http_get=http_get,
                                                    url_check=url_check, pacer=pacer)
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
