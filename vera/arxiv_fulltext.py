"""ARX-A2: arXiv full-text representation route for DISCOVERED arXiv records (config auto_acquire.json arxiv_route).

Before this route an arXiv record without a journal DOI was always held (R-ACQ-PREPRINT_LICENCE_UNVERIFIED): the only
full-text route was Europe PMC JATS, which refuses preprints, so the arXiv HTML representation was never resolved.

Flow for one versioned arXiv id (e.g. 2605.23989v1), every request through auto_acquire._guarded_get (https only,
exact host allowlist arxiv_route.hosts on every redirect hop, the public-address check, the pinned public-only
transport of deps.http_get, content type, size cap, wall-clock deadline):
  1. abs page https://arxiv.org/abs/<id>v<n>  -> identity (citation_arxiv_id), title, authors, date, and the paper's
     OWN licence link. CC BY / BY-SA / BY-NC / BY-NC-SA / CC0 continue; any ND or no-educational-use / fee-only marker
     is REJECTED (R-LIC-REJECT); arXiv's non-exclusive distribution licence is HELD (R-ACQ-ARXIV_LICENCE_HOLD); no
     licence link is HELD (R-ACQ-LICENCE_ABSENT); two different licences are HELD (R-ACQ-LICENCE_MISMATCH). Index
     metadata (CC0) is never used.
  2. representations in arxiv_route.representations order: "html" (https://arxiv.org/html/<id>v<n>, text/html, final
     URL must still be /html/<same id and version>, and the page must be a LaTeXML article, never the abstract page);
     "pdf" only when arxiv_route.pdf_extractor names an existing extractor (none exists: recorded, not fetched).
     When no representation yields an article, the record is abstract-only: R-ABSTRACT (never admitted as full text).
  3. article body extraction by the EXISTING converter (vera.jats_text html path) restricted to the <article>
     element (site header, nav, footer, scripts, styles, buttons dropped by the converter's skip list).
  4. the UNCHANGED gates: fulltext_loader.prepare_from_parts (body/abstract thresholds, then licence_gate), the
     licence-element decision the acquired-path reader re-applies, the injection scan, build_plan, relevance gate.
     licence_gate.py currently rejects every arxiv.org non-/abs/ URL (arXiv terms); that refusal is recorded as
     R-ACQ-ARXIV_HTML_POLICY (owner decision pending; this module never works around it).
Every attempt is recorded in provenance.acquisition.arxiv.attempts (admitted) and in last_attempts() (telemetry).
"""
from __future__ import annotations

import copy
import hashlib
import re
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit

from vera import acq_config

ROUTE = "arxiv-html"
# new-style ids only carry a version suffix we can pin; old-style (cs/0101001) ids also accepted
# P0DTA-ARXFIX: the ONE shared arXiv URL parser (http or https; query/fragment ignored); acq_discovered reuses it.
_ID_IN_URL = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/"
                        r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?(?:\.pdf)?/?(?:[?#].*)?$", re.I | re.A)
_ID_TEXT = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.I | re.A)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I | re.A)
_CC = re.compile(r"^https?://creativecommons\.org/(?:licenses/([a-z\-]+)/(\d\.\d)|publicdomain/zero/(1\.0))/?$", re.I)
_ARXIV_LIC = re.compile(r"^https?://arxiv\.org/licenses/([a-z\-]+)/(\d\.\d)/?(?:license\.html)?$", re.I)
FAMILY_LICENCE = {"by": "cc by", "by-sa": "cc by-sa", "by-nc": "cc by-nc", "by-nc-sa": "cc by-nc-sa",
                  "by-nd": "cc by-nd", "by-nc-nd": "cc by-nc-nd", "zero": "cc0"}

_LAST: dict = {}  # telemetry: arXiv id+version -> attempts of the most recent evaluation (bounded)


def rcfg(acq) -> dict:
    v = (acq or {}).get("arxiv_route") if isinstance(acq, dict) else None
    return v if isinstance(v, dict) else {}


def route_on(acq) -> bool:
    return rcfg(acq).get("enabled") is True


def last_attempts(key: str) -> list:
    return list(_LAST.get(key) or [])


# ------------------------------------------------------------------------------------------------ identity
def parse_arxiv_url(url) -> tuple[str, str | None] | None:
    """(lower-case id, version "vN" or None) of any arXiv abs/pdf/html URL shape, else None. Callers canonicalise to
    https://arxiv.org/abs/<id><vN> (the F-1 host check and licence gate are unchanged)."""
    m = _ID_IN_URL.match(str(url or "").strip())
    return (m.group(1).lower(), (m.group(2) or "").lower() or None) if m else None


def arxiv_ident(s: dict) -> tuple[str, str | None] | None:
    """(id, version or None) of an arXiv discovery record, from its URL, identifier or arXiv DataCite DOI."""
    m = _ID_IN_URL.match(str(s.get("url") or "").strip())
    if m:
        return m.group(1).lower(), (m.group(2) or "").lower() or None
    m = _ID_TEXT.match(str(s.get("identifier") or "").strip())
    if m:
        return m.group(1).lower(), (m.group(2) or "").lower() or None
    d = _ARXIV_DOI.match(str(s.get("doi") or "").strip())
    if d:
        m = _ID_TEXT.match(d.group(1))
        if m:
            return m.group(1).lower(), (m.group(2) or "").lower() or None
    return None


def discovered_candidate(s: dict, doi: str | None, kind: str | None, entry: dict, acq) -> dict | None:
    """The acquisition candidate for an arXiv discovery record that has no journal DOI (only arXiv's own 10.48550 DOI
    or none), when arxiv_route is enabled; else None (the caller keeps its unchanged hold). Non-arXiv preprints and
    arXiv records with a journal DOI (Europe PMC path, peer-reviewed version) are never routed here."""
    if not route_on(acq) or kind is None:
        return None
    ident = arxiv_ident(s)
    if ident is None or (doi and not _ARXIV_DOI.match(doi)):
        return None
    aid, ver = ident
    adoi = f"10.48550/arxiv.{aid}"
    return {"doi": adoi, "url": f"https://arxiv.org/abs/{aid}{ver or ''}",
            "title": " ".join(str(s.get("title") or "").split()), "snippet": " ".join(str(s.get("snippet") or "").split()),
            "acq_source": "discovered", "discovery": entry, "arxiv": {"id": aid, "version": ver}}


def search_candidate(cand: dict, doi: str | None, acq) -> dict | None:
    """INTEG-1023: the same rule as discovered_candidate for a candidate returned by the ACQUISITION search
    (search_providers.acquisition_search -> auto_acquire.evaluate_candidate): an arXiv-identified record with no journal
    DOI goes to this route when arxiv_route is enabled; else None (the unchanged Europe PMC path). Same allowlist,
    pacing and gates as the discovery path (evaluate)."""
    if not route_on(acq) or not isinstance(cand, dict) or isinstance(cand.get("arxiv"), dict):
        return None
    ident = arxiv_ident(cand)
    if ident is None or (doi and not _ARXIV_DOI.match(str(doi))):
        return None
    aid, ver = ident
    return {**cand, "doi": f"10.48550/arxiv.{aid}", "url": f"https://arxiv.org/abs/{aid}{ver or ''}",
            "arxiv": {"id": aid, "version": ver}}


# ------------------------------------------------------------------------------------------------ abs page
class _AbsPage(HTMLParser):
    """citation_* meta tags, licence links (link rel=license, a.has_license / title 'Rights to this article' / any
    a inside .abs-license), submission-history versions, and the arXiv abstract blockquote."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict = {}
        self.licences: list = []
        self.versions: list = []
        self._lic_depth = 0
        self._stack: list = []
        self._in_strong = False
        self.has_abstract_block = False

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        cls = a.get("class", "").lower().split()
        if tag == "meta" and a.get("name", "").lower().startswith("citation_"):
            self.meta.setdefault(a["name"].lower(), []).append(a.get("content", "").strip())
        if tag == "link" and a.get("rel", "").lower() == "license" and a.get("href"):
            self.licences.append(a["href"].strip())
        if tag in ("meta", "link", "br", "img", "input", "hr"):
            return
        self._stack.append(tag)
        if "abs-license" in cls:
            self._lic_depth = len(self._stack)
        if tag == "blockquote" and "abstract" in cls:
            self.has_abstract_block = True
        if tag == "a" and a.get("href") and (self._lic_depth or "has_license" in cls
                                             or a.get("title", "").strip().lower() == "rights to this article"):
            self.licences.append(a["href"].strip())
        if tag == "strong":
            self._in_strong = True

    def handle_endtag(self, tag):
        if tag in self._stack:
            while self._stack and self._stack.pop() != tag:
                pass
            if self._lic_depth and len(self._stack) < self._lic_depth:
                self._lic_depth = 0
        if tag == "strong":
            self._in_strong = False

    def handle_data(self, data):
        if self._in_strong:
            m = re.fullmatch(r"\s*\[(v\d+)\]\s*", data)
            if m:
                self.versions.append(m.group(1).lower())


def parse_abs(raw: bytes) -> _AbsPage:
    p = _AbsPage()
    p.feed(raw.decode("utf-8", "replace"))
    p.close()
    return p


def licence_of(urls: list, acq: dict, icfg: dict) -> tuple[str, str, str]:
    """(declared licence string, exact licence URL, family) from the abs page's licence links. Raises Refused/_Stop."""
    from vera import auto_acquire as aa
    from vera import fulltext_loader as fl
    from vera.licence_gate import reject_reason
    codes = icfg["reason_codes"]
    norm = sorted({u.strip().rstrip("/") + "/" for u in urls if u and u.strip()})
    if not norm:
        raise aa._Stop(acq_config.reason(acq, "licence_absent"))
    for u in norm:  # reject markers first (ND, no educational use, fee only), on every stated licence
        why = reject_reason(None, u)
        if why:
            raise fl.Refused(codes["licence_reject"], f"arXiv licence {u} is {why}")
    fams = set()
    for u in norm:
        m = _CC.match(u)
        if m:
            fams.add(m.group(1).lower() if m.group(1) else "zero")
        elif _ARXIV_LIC.match(u):
            fams.add("arxiv:" + _ARXIV_LIC.match(u).group(1).lower())
        else:
            fams.add("unknown")
    if len(fams) != 1:
        raise aa._Stop(acq_config.reason(acq, "licence_mismatch"))
    fam = next(iter(fams))
    allowed = allowed_families(acq)
    if fam.startswith("arxiv:") or fam == "unknown" or fam not in FAMILY_LICENCE or fam not in allowed:
        # arXiv's non-exclusive distribution licence grants arXiv the right to distribute, not reuse; a CC licence
        # not listed in arxiv_route.licence_families (e.g. CC0 as an article licence) is not admitted: hold
        raise aa._Stop(acq_config.reason(acq, "arxiv_licence_hold"))
    lic = FAMILY_LICENCE[fam]
    url = "https://" + norm[0].split("://", 1)[1]
    return lic, url, fam


def _gate_families():
    """The licence gate's own family list (config/licence_rules.json arxiv_html_cc_families) or None while that key
    does not exist (it arrives only with the owner-authorized gate change). A malformed value is (), fail closed."""
    import json
    from vera import licence_gate
    try:
        v = json.loads(licence_gate.RULES_PATH.read_text(encoding="utf-8")).get("arxiv_html_cc_families")
    except (OSError, ValueError, AttributeError):
        return ()
    if v is None:
        return None
    return tuple(str(x).lower() for x in v) if isinstance(v, list) else ()


def allowed_families(acq) -> set:
    """ARX-SECREV-1023 R4: ONE binding list. Once licence_rules.json carries arxiv_html_cc_families (the gate's list),
    the route admits only families in BOTH lists, so arxiv_route.licence_families can never widen the gate; until
    then arxiv_route.licence_families applies (and the gate refuses arXiv HTML anyway)."""
    route = {str(f).lower() for f in (rcfg(acq).get("licence_families") or ())}
    gate = _gate_families()
    return route if gate is None else route & set(gate)


def evidence_block(aid: str, ver: str, lic_url: str) -> dict:
    """ARX-SECREV-1023 R1: the route-verified licence evidence for one exact arXiv version. Set only by evaluate();
    the gate (after the owner-authorized change) re-checks it against the HTML URL and the content licence URL."""
    return {"id": aid, "version": ver, "abs_url": f"https://arxiv.org/abs/{aid}{ver}", "licence_url": lic_url,
            "source": "arxiv abs page"}


def stored_evidence(prov: dict) -> dict | None:
    """The read path's evidence: the block the route stored at write time, returned only when this row was written by
    the arXiv route and the block equals one rebuilt from the route's own acquisition.arxiv record; else None (the
    gate then keeps its refusal). Manifest/bundle rows have no route acquisition record and never get one."""
    acq = prov.get("acquisition") if isinstance(prov, dict) else None
    if not isinstance(acq, dict) or acq.get("resolver") != ROUTE or not isinstance(acq.get("arxiv"), dict):
        return None
    ax = acq["arxiv"]
    stored = (prov.get("licence") or {}).get("arxiv_evidence") if isinstance(prov.get("licence"), dict) else None
    if not (isinstance(ax.get("id"), str) and isinstance(ax.get("version"), str) and isinstance(ax.get("licence_url"), str)):
        return None
    ev = evidence_block(ax["id"], ax["version"], ax["licence_url"])
    if stored != ev or ax.get("abs_url") != ev["abs_url"]:
        return None
    return ev


# ------------------------------------------------------------------------------------------------ fetch
def _route_cfg(acq: dict, ctypes) -> dict:
    """A copy of the acquisition config whose fetch allowlist and content types are the arXiv route's own (the Europe
    PMC allowlist in fetch.allowed_hosts is never widened). Reason codes and bounds are unchanged."""
    c = copy.deepcopy(acq)
    c["fetch"] = {**(acq.get("fetch") or {}), "allowed_hosts": list(rcfg(acq).get("hosts") or ()),
                  "content_types": list(ctypes)}
    return c


_sleep = time.sleep  # tests replace it; production waits really
_HOST_LAST: dict = {}  # P0DTA-ARXFIX C: host -> clock() of the last request to it, process level (spans requests)
_HOST_LOCKS: dict = {}
_HOST_GUARD = threading.Lock()


def _host_lock(host: str):
    with _HOST_GUARD:
        return _HOST_LOCKS.setdefault(host, threading.Lock())


def doc_budget(acq, clock=time.monotonic) -> float:
    """P0DTA-ARXFIX C: the per-document time an arXiv-routed record needs: 2 x crawl-delay (abs, then html) plus one
    fetch timeout. The crawl-delay is the robots value when cached, else the configured maximum (never lowered)."""
    from vera import acq_discovered
    from vera import selective_fetch as sf
    r = rcfg(acq)
    cd = float(r.get("max_crawl_delay_s", 15))  # robots not cached yet: assume the configured maximum
    hit = acq_discovered._ROBOTS.get("arxiv.org")
    if hit is not None and hit[1] not in (None, True, False):
        v = hit[1].crawl_delay(sf.USER_AGENT)
        cd = float(v) if v is not None else float(r.get("min_interval_s", 3))
    return 2 * cd + float(acq_config.bound(acq, "fetch_timeout_s"))


def _get(url, acq, deps, max_bytes, clock, deadline, chain, pace: dict):
    """One arXiv request: robots.txt (cached; fail closed) and its crawl-delay, then this route's own spacing
    (arxiv_route.min_interval_s, one request at a time), then the guarded fetch."""
    from vera import acq_discovered
    from vera import auto_acquire as aa
    from vera import selective_fetch as sf
    r = rcfg(acq)
    hosts = {str(h).lower() for h in (r.get("hosts") or ())}
    host = (urlsplit(url).hostname or "").lower()
    gap = float(r.get("min_interval_s", 3))
    with _host_lock(host):  # one request at a time per host; pacing is process level, not per record
        return _get_locked(url, acq, deps, max_bytes, clock, deadline, chain, pace, r, hosts, host, gap)


def _get_locked(url, acq, deps, max_bytes, clock, deadline, chain, pace, r, hosts, host, gap):
    from vera import acq_discovered
    from vera import auto_acquire as aa
    from vera import selective_fetch as sf
    last = _HOST_LAST.get(host)
    if last is not None and last <= clock():  # the most recent request by ANY record to this host (read under the lock)
        pace["last"] = max(pace.get("last") or last, last)
    if r.get("robots", True):
        cached = host in acq_discovered._ROBOTS
        allowed = host in hosts and acq_discovered._robots_allowed(url, hosts, deps, acq, clock, deadline)
        if not cached:
            pace["last"] = _HOST_LAST[host] = clock()  # robots.txt was requested just now
        if not allowed:
            raise aa._Stop(acq_config.reason(acq, "robots"))
        rp = (acq_discovered._ROBOTS.get(host) or (0, None))[1]
        cd = rp.crawl_delay(sf.USER_AGENT) if rp not in (None, True, False) else None
        if cd is not None:
            if float(cd) > float(r.get("max_crawl_delay_s", 15)):
                raise aa._Stop(acq_config.reason(acq, "robots"))
            gap = max(gap, float(cd))
    if pace.get("last") is not None:
        wait = pace["last"] + gap - clock()
        if wait > 0:
            if deadline is not None and clock() + wait >= deadline:
                raise aa._Stop(acq_config.reason(acq, "crawl_delay"))  # P0DTA-ARXFIX C: distinct, background-eligible
            _sleep(wait)
    c = _route_cfg(acq, r.get("content_types") or ("text/html",))
    try:
        return aa._guarded_get(url, c, deps, int(max_bytes), float(acq_config.bound(acq, "fetch_timeout_s")), False,
                               clock=clock, deadline=deadline, chain=chain)
    finally:
        pace["last"] = _HOST_LAST[host] = clock()


def _is_latexml_article(raw: bytes) -> bool:
    head = raw[:400000].decode("utf-8", "ignore").lower()
    return "ltx_document" in head and "<article" in head


def _article_ccfg(ccfg: dict) -> dict:
    """The existing HTML converter, emitting body text only inside <article> (arXiv LaTeXML wraps the paper in
    article.ltx_document); the configured skip list (script, style, nav, header, footer, button, ...) still applies."""
    c = copy.deepcopy(ccfg)
    c["html_body_containers"] = ["article"]
    return c


# ------------------------------------------------------------------------------------------------ evaluate
def evaluate(job, score, doi, cand, isbn, acq, deps, icfg, fdcfg, clock=time.monotonic, deadline=None, dur=None,
             min_spans=None):
    """auto_acquire.evaluate_candidate for an arXiv-routed candidate. Same contract: returns (plan, isbn); raises
    auto_acquire._Stop / fulltext_loader.Refused with the terminal reason. Writes nothing."""
    from vera import acq_discovered
    from vera import auto_acquire as aa
    from vera import fulltext_loader as fl
    from vera import jats_text
    dur = {} if dur is None else dur
    r = rcfg(acq)
    aid = str((cand.get("arxiv") or {}).get("id") or "")
    ver = (cand.get("arxiv") or {}).get("version")
    chain = []
    pace: dict = {}
    attempts: list = []
    key = f"{aid}{ver or ''}"
    _LAST[key] = attempts
    while len(_LAST) > 32:
        _LAST.pop(next(iter(_LAST)))
    t = clock()
    abs_url = f"https://arxiv.org/abs/{aid}{ver or ''}"
    try:
        abs_raw, abs_final = _get(abs_url, acq, deps, r.get("abs_max_bytes", 1048576), clock, deadline, chain, pace)
    except aa._Stop as e:
        attempts.append({"representation": "abs", "url": abs_url, "status": "refused", "reason": e.code})
        raise
    asp = urlsplit(abs_final or "")
    if (asp.scheme != "https" or (asp.hostname or "") not in {str(h).lower() for h in (r.get("hosts") or ())}
            or asp.query or asp.fragment or asp.path.rstrip("/") != f"/abs/{aid}{ver or ''}"):
        # ARX-SECREV-1023 R2: the licence must come from the REQUESTED abs page (same id and version), never from
        # the page a redirect landed on
        attempts.append({"representation": "abs", "url": abs_url, "status": "refused",
                         "reason": "abs final URL is not the requested id/version abstract page", "final_url": abs_final})
        raise aa._Stop(acq_config.reason(acq, "identity_mismatch"))
    page = parse_abs(abs_raw)
    attempts.append({"representation": "abs", "url": abs_url, "status": "fetched", "final_url": abs_final})
    ids = {x.lower() for x in page.meta.get("citation_arxiv_id", [])}
    if ids and not ({aid, f"{aid}{ver or ''}"} & ids):
        raise aa._Stop(acq_config.reason(acq, "identity_mismatch"))
    if ver is None:  # an unversioned discovery record: pin the newest version the abs page lists
        if not page.versions:
            raise aa._Stop(acq_config.reason(acq, "identity_mismatch"))
        ver = max(page.versions, key=lambda v: int(v[1:]))
    lic, lic_url, fam = licence_of(page.licences, acq, icfg)  # before any content fetch: no fetch of what cannot be used
    abs_canon = f"https://arxiv.org/abs/{aid}{ver}"
    raw = final = None
    for rep in r.get("representations") or ("html",):
        if rep == "html":
            url = f"https://arxiv.org/html/{aid}{ver}"
            try:
                got, fin = _get(url, acq, deps, acq_config.bound(acq, "max_fetch_bytes"), clock, deadline, chain, pace)
            except aa._Stop as e:
                attempts.append({"representation": "html", "url": url, "status": "refused", "reason": e.code})
                if e.code == acq_config.reason(acq, "unreachable"):
                    continue  # not offered (404) or unreachable: try the next representation
                raise
            sp = urlsplit(fin)
            if sp.hostname not in set(r.get("hosts") or ()) or sp.path.rstrip("/") != f"/html/{aid}{ver}":
                attempts.append({"representation": "html", "url": url, "status": "refused",
                                 "reason": "final URL is not the requested version's HTML", "final_url": fin})
                if sp.path.startswith("/abs/"):
                    continue  # arXiv sends the abstract page when no HTML exists: that is not full text
                raise aa._Stop(acq_config.reason(acq, "identity_mismatch"))
            if not _is_latexml_article(got):
                attempts.append({"representation": "html", "url": url, "status": "refused",
                                 "reason": "not a LaTeXML article page (abstract or notice page)"})
                continue
            attempts.append({"representation": "html", "url": url, "status": "fetched", "final_url": fin})
            raw, final = got, fin
            break
        if rep == "pdf":
            ext = r.get("pdf_extractor")
            attempts.append({"representation": "pdf", "url": f"https://arxiv.org/pdf/{aid}{ver}",
                             "status": "not_attempted", "reason": "no PDF extractor in the codebase" if not ext
                             else f"pdf extractor {ext!r} not wired"})
    dur["fetch"] = dur.get("fetch", 0) + int((clock() - t) * 1000)
    if raw is None:
        raise fl.Refused(icfg["reason_codes"]["abstract_only"],
                         "arXiv record offers no extractable full-text representation; abstract page only")
    t = clock()
    try:
        conv = jats_text.convert(raw, "html", _article_ccfg(icfg["converter"]))
    except Exception:  # noqa: BLE001
        raise aa._Stop(acq_config.reason(acq, "convert")) from None
    text = conv["text"]
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {k: v for k, v in conv.items() if k != "text"}
    title = (page.meta.get("citation_title") or [""])[0] or cand.get("title") or (meta.get("meta") or {}).get("title") or ""
    authors = "; ".join(a for a in page.meta.get("citation_author", []) if a)
    published = (page.meta.get("citation_date") or page.meta.get("citation_online_date") or [""])[0]
    meta.update({"raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_file": None,
                 "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    did = "acq-" + hashlib.sha256(doi.encode("utf-8")).hexdigest()[:16]
    meta["document_id"] = did
    # the paper's own licence statement (arXiv abs page), re-decided by the acquired-path reader like JATS elements
    meta["jats_licences"] = [{"href": lic_url, "type": None, "ref": None, "text": None}]
    attribution = (f"{authors or 'Unknown authors'}. {title}. arXiv:{aid}{ver} ({abs_canon}). "
                   f"Licence: {lic.upper()} ({lic_url}).")
    item = {"document_id": did, "title": title, "doi": doi, "landing_url": abs_canon, "fulltext_url": final,
            "fulltext_format": "html", "text_file": f"acquired/{did}.html", "licence": lic, "licence_url": lic_url,
            "authors": authors, "published_year": (published[:4] if re.match(r"\d{4}", published) else ""),
            "fetched_at": now_iso, "attribution": attribution, "verifier_status": icfg["verified_status"],
            "read_status": "VERBATIM"}
    aa.licence_elements_decision(meta["jats_licences"], lic, icfg)
    try:
        # ARX-SECREV-1023 R1: the version-bound licence evidence is built HERE only (from the fetched abs page of this
        # exact id+version and the licence URL it states) and handed to the loader as an argument, never via item/meta
        rec = fl.prepare_from_parts(item, text, meta, icfg, now=now_iso,  # every loader filter + licence gate
                                    arxiv_evidence=evidence_block(aid, ver, lic_url))
    except fl.Refused as e:
        attempts.append({"representation": "html", "url": final, "status": "refused", "reason": e.code,
                         "detail": str(e)[:300]})
        if e.code == icfg["reason_codes"]["licence_reject"] and _arxiv_url_rule(final):
            raise aa._Stop(acq_config.reason(acq, "arxiv_html_policy")) from None
        raise
    if acq_config.injection_refused(f"{title}\n{text}", acq):
        raise aa._Stop(acq_config.reason(acq, "inject"))
    value = {"open_access": True, "licence_family": fam, "pub_types_n": 0}
    plan = aa.build_plan(rec, item, meta, acq, job, score, value, isbn, now_iso)
    res = {"pub_types": []}
    acq_discovered.annotate(plan, cand, res, "landing", chain, acq)
    a = plan["source"]["provenance"]["acquisition"]
    a["resolver"], a["repository"] = ROUTE, "arxiv.org"
    a["content_version"] = {"pub_types": [], "source": f"arXiv {aid}{ver} HTML (LaTeXML)"}
    if (cand.get("discovery") or {}).get("epistemic"):
        a["content_version"]["discovered_as"] = cand["discovery"]["epistemic"]
    a["licence_evidence"] = {"governing": "article licence: the paper's licence link on its arXiv abs page",
                             "index_metadata_licence": "not used"}
    a["inspection"]["converter"] = "html (article element only)"
    a["arxiv"] = {"id": aid, "version": ver, "abs_url": abs_canon, "html_url": final, "licence": lic,
                  "licence_url": lic_url, "published": published or None, "retrieved_at": now_iso,
                  "attempts": [dict(x) for x in attempts]}
    n_adm = aa.relevance_gate(job.question, plan, acq, icfg, fdcfg)
    dur["verify"] = dur.get("verify", 0) + int((clock() - t) * 1000)
    need = acq_config.persist_min_spans(acq) if min_spans is None else max(1, int(min_spans))
    if n_adm < need:
        raise aa._Stop(acq_config.reason(acq, "few_spans") if n_adm else acq_config.reason(acq, "irrelevant"), n_adm)
    plan["admitted_spans"] = n_adm
    acq_discovered.gate_b_admit(plan)
    return plan, isbn


def _arxiv_url_rule(url: str) -> bool:
    """True when licence_gate refuses this URL on its arXiv e-print rule alone (asked of the gate itself, with an
    allowing licence, so a refusal can only come from the URL rule)."""
    from vera.licence_gate import licence_decision
    d, _why = licence_decision({"url": url, "licence": {"content": {"id": "cc by", "url":
                                                                     "https://creativecommons.org/licenses/by/4.0/"}},
                                "stored_text_kind": "full_text"})
    return d == "reject"
