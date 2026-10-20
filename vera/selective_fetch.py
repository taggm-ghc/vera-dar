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
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests

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


def is_public_url(url: str) -> tuple[bool, str]:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False, "only http(s) URLs are fetched"
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        return False, f"dns failure: {exc}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return False, f"blocked non-public address {ip}"
    return True, ""


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
                current = urljoin(current, resp.headers["Location"])
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


def fetch_html(url: str, *, timeout_s: float = 8.0, http_get=requests.get, url_check=is_public_url,
               pacer: HostPacer | None = None) -> tuple[str | None, str | None, str | None]:
    """Raw decoded body (markup kept, for link and metadata extraction) under the same guards.
    Returns (body, final_url, error)."""
    raw, _ctype, final, _hops, enc, err = _download(url, timeout_s=timeout_s, http_get=http_get,
                                                     url_check=url_check, pacer=pacer)
    if err:
        return None, None, err
    return raw.decode(enc, errors="replace"), final, None


def fetch_candidate(
    url: str,
    *,
    timeout_s: float = 15.0,
    http_get=requests.get,
    url_check=is_public_url,
    ledger: CostLedger | None = None,
    pacer: HostPacer | None = None,
) -> FetchResult:
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
