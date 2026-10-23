"""P0-5-ACQ (owner decisions D1-D5, 2026-10-22): bounded discovery-to-acquisition correction. Both steps ship OFF.

Step A (config inline.use_discovered, default false): the /ask request's OWN cached discovery results (the reading-list
search: arXiv, OpenAlex, DOAJ, DuckDuckGo) become acquisition candidates. The adapter keeps the candidate identity
(DOI / PMCID, title, URL) and the discovering service + record id, and DROPS the index-metadata licence (CC0 on the
index record never authorizes article content: D-constraint 3). The candidates then pass the UNCHANGED acquisition
stages (rank, resolve, fetch, licence, integrity, injection, relevance, admission). A deterministic Gate A (D2) labels
each discovered candidate fetch / defer / reject from existing metadata only (identifier, duplicate, served, the
existing OA pre-filter, the existing pre-rank cut and caps), with its factors and uncertainty; no score is stored and
no model is called. Gate B (D3) is the existing single-document admission, recorded as admit/reject plus its basis.

D1 preprints: an arXiv record without a journal DOI (or with only arXiv's own 10.48550 DOI) is NOT acquired: no
existing interface can read the paper's own licence (the arXiv Atom feed in use carries none; index CC0 does not
count), so it keeps its bibliographic link and the explicit reason R-ACQ-PREPRINT_LICENCE_UNVERIFIED. The positive
path (a per-paper CC licence read from arXiv) is an INCOMPLETE capability, not implemented here (no new provider).
Preprint status is kept as an epistemic attribute ("preprint: not peer reviewed"), never as peer review.

Step B (config inline.landing_route, default false; only with Step A): when the Europe PMC resolver finds no
open-access record for a discovered DOI, ONE bounded DOI -> landing page -> JATS XML attempt: every hop is https,
on the explicit landing_route host allowlists, public-address checked and robots.txt checked (revalidated at every
redirect, cycle detected); only a JATS XML location stated by the landing page is fetched (by the guarded fetcher,
its own allowlist); the licence comes from the JATS licence elements alone (absent or unclear = hold). A landing page
that offers only HTML or PDF ends with R-ACQ-FORMAT_UNSUPPORTED. This is not general full-text coverage.

D3b retention: the converted text is retained as immutable spans (existing store); the original bytes are NOT
retained (no existing raw store; a new one needs schema/storage approval): provenance records "retention: hash-only"
with raw_sha256, the source URL chain and version references.
"""
from __future__ import annotations

import copy
import hashlib
import re
import threading
import time
from urllib.parse import urljoin, urlsplit

from vera import acq_config

PREPRINT_EPISTEMIC = "preprint: not peer reviewed"
ARXIV_DOI_PREFIX = "10.48550/"
_STORED_KEYS = ("in_corpus", "span", "dar_span", "span_id")
RETENTION_REASON = ("no existing raw-content store: acquired sources keep content '' and the converted text as "
                    "immutable spans; storing original bytes needs a separate schema/storage approval (D3b)")


# ------------------------------------------------------------------------------------------------ flags
def _icfg(acq) -> dict:
    v = (acq or {}).get("inline") if isinstance(acq, dict) else None
    return v if isinstance(v, dict) else {}


def use_discovered(acq) -> bool:
    return _icfg(acq).get("use_discovered") is True


def landing_on(acq) -> bool:
    return use_discovered(acq) and _icfg(acq).get("landing_route") is True


def discovered_max(acq) -> int:
    try:
        return max(0, int(_icfg(acq).get("discovered_max", 6)))
    except (TypeError, ValueError):
        return 6


def discovery_results(grounding, restricted: bool):
    """The request's cached discovery search: EXACTLY the call sequence the reading list makes, so main.py's
    per-request cache (keyed by restricted + OpenALEX_PROFILE) returns the same results to both and no extra provider
    search is made when the reading list runs. In scope: sources_fn() (vera.ask_service._reading_list). Off-topic:
    sources_fn(restricted=True) under the off-topic OpenAlex profile, then the OR retry only when that is empty
    (vera.ask_service._offtopic_reading_list, RL-REL-1021/-B). Never raises; [] when unavailable."""
    try:
        fn = grounding[0] if grounding else None
        if not callable(fn):
            return []
        extra = getattr(fn, "unused", None)  # abstract-less originals: acquisition candidates only, never evidence
        if not restricted:
            got = list(fn() or [])
            return got + _unused_of(extra, False)
        from vera import scope_router as sr
        from vera import search_providers
        profile = sr.offtopic_openalex_profile()
        tok = search_providers.OPENALEX_PROFILE.set(profile)
        try:
            got = list(fn(restricted=True) or [])
            if not _listable(got) and profile is not None:
                search_providers.OPENALEX_PROFILE.reset(tok)
                tok = search_providers.OPENALEX_PROFILE.set(None)
                got = list(fn(restricted=True) or [])
            got += _unused_of(extra, True)
        finally:
            search_providers.OPENALEX_PROFILE.reset(tok)
        return got
    except Exception:  # noqa: BLE001
        return []


def _unused_of(extra, restricted: bool) -> list:
    """The abstract-less originals of the cached search; a failure here costs only these extras, never the main results."""
    try:
        return list(extra(restricted) or []) if callable(extra) else []
    except Exception:  # noqa: BLE001
        return []


def _listable(sources) -> bool:
    """The reading list's own emptiness test (public_candidates), so the OR retry happens exactly when it would."""
    try:
        from vera.grounded_ask import public_candidates
        return bool(public_candidates(sources))
    except Exception:  # noqa: BLE001
        return bool(sources)


# ------------------------------------------------------------------------------------------------ adapter (Step A)
def _clean(v, n: int = 500) -> str:
    """P2: untrusted discovery strings are bounded and stripped of control characters before they enter provenance."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", str(v or ""))[:n]


def _service(s: dict) -> str:
    return _clean(s.get("provider") or "unknown", 40).strip().lower() or "unknown"


def _arxiv_id(s: dict) -> str | None:
    from vera import arxiv_fulltext  # P0DTA-ARXFIX: shared parser (http/https, abs|pdf|html, query/fragment)
    p = arxiv_fulltext.parse_arxiv_url(s.get("url"))
    return p[0] if p else None


def _record_id(s: dict) -> str | None:
    for k in ("openalex_id", "doaj_id"):
        if s.get(k):
            return _clean(s[k], 200)
    a = _arxiv_id(s)
    if a:
        return "arXiv:" + a
    return _clean(s.get("identifier") or s.get("url") or "", 500) or None


# D1: preprint servers' own DOIs (arXiv DataCite; bioRxiv/medRxiv = 10.1101/<date.>digits, NOT the Cold Spring Harbor
# journals' 10.1101/gad.* / gr.* ...; SSRN, Research Square, Preprints.org, OSF, PsyArXiv, ChemRxiv)
_PREPRINT_DOI = re.compile(r"^(?:10\.48550/|10\.1101/(?:\d{4}\.\d{2}\.\d{2}\.)?\d{6,}(?:v\d+)?$|10\.2139/ssrn|"
                           r"10\.21203/rs\.|10\.20944/preprints|10\.31219/osf|10\.31234/osf|10\.26434/chemrxiv)", re.I)
_PREPRINT_NAMES = re.compile(r"\b(?:arxiv|biorxiv|medrxiv|chemrxiv|psyarxiv|ssrn|research square|preprints\.org)\b", re.I)


def _preprint_kind(s: dict, doi: str | None) -> str | None:
    """None, "arxiv" (an arXiv record: its DOI, when not 10.48550, is the journal version's) or "record" (the record
    itself is a preprint: a preprint-server DOI, a preprint source/type/name, or a Europe PMC preprint flag)."""
    if doi and _PREPRINT_DOI.match(doi):
        return "record"
    oa = s.get("oa") if isinstance(s.get("oa"), dict) else {}
    ptypes = s.get("pub_types") or s.get("pubTypeList") or []
    ptypes = ptypes.get("pubType", []) if isinstance(ptypes, dict) else ptypes
    ptypes = [ptypes] if isinstance(ptypes, str) else list(ptypes or [])
    if (str(s.get("source") or oa.get("source") or "").upper() == "PPR"
            or str(s.get("source_type") or s.get("type") or "").strip().lower() in ("preprint", "posted-content")
            or any(str(p).strip().lower() == "preprint" for p in ptypes)
            or _PREPRINT_NAMES.search(" ".join(str(s.get(k) or "") for k in ("publisher", "venue", "journal")))
            or (_service(s) in ("biorxiv", "medrxiv"))):
        return "record"
    if _service(s) == "arxiv" or _arxiv_id(s) is not None:
        return "arxiv"
    return None


def _entry(s: dict, preprint: bool) -> dict:
    e = {"service": _service(s), "record_id": _record_id(s), "url": _clean(s.get("url"), 2000)}
    if preprint:
        e["epistemic"] = PREPRINT_EPISTEMIC
    return e


def discovered_candidates(sources, acq) -> tuple[list, list]:
    """(acquisition candidates, held entries). Stored-corpus items are skipped (already evidence). Candidates carry
    doi/pmcid/url/title/snippet plus `discovery` (service, record id, url, epistemic) and acq_source "discovered";
    the index licence and stored_text_kind are DROPPED (never copied). Held entries are discovered records that cannot
    be acquired, each with an explicit Gate A reject reason (no identifier / preprint licence unverified)."""
    from vera import auto_acquire as aa
    from vera.search_providers import _pmcid_of
    cands, held = [], []
    n = discovered_max(acq)
    for s in sources or ():
        if not isinstance(s, dict) or any(s.get(k) for k in _STORED_KEYS):
            continue
        doi = aa._doi(s)
        pmcid = None if doi else _pmcid_of(s.get("pmcid"))
        kind = _preprint_kind(s, doi)
        e = _entry(s, kind is not None)
        from vera import arxiv_fulltext  # ARX-A2: arXiv records (no journal DOI) -> arXiv route when enabled
        ac = arxiv_fulltext.discovered_candidate(s, doi, kind, e, acq)
        if ac is None and (kind == "record" or (kind == "arxiv" and not doi)):  # D1: no route reads the paper's own licence
            held.append({**e, "gate_a": {"decision": "reject", "factors": {"identifier": "doi" if doi else "none",
                                                                           "preprint": True},
                                         "uncertainty": ["the preprint's own licence cannot be read through any "
                                                         "existing interface; index-metadata CC0 does not apply to it"]},
                         "outcome": "link_only", "reason": acq_config.reason(acq, "preprint_licence_unverified")})
            continue
        if ac is None and not doi and not pmcid:
            held.append({**e, "gate_a": {"decision": "reject", "factors": {"identifier": "none"}, "uncertainty": []},
                         "outcome": "link_only", "reason": acq_config.reason(acq, "no_doi")})
            continue
        c = ac if ac is not None else {
            "doi": doi, "url": str(s.get("url") or (f"https://doi.org/{doi}" if doi else "")),
            "title": " ".join(str(s.get("title") or "").split()), "snippet": " ".join(str(s.get("snippet") or "").split()),
            "acq_source": "discovered", "discovery": e}
        if pmcid and ac is None:
            c["pmcid"] = pmcid
        if s.get("source_type"):
            c["source_type"] = s["source_type"]
        if len(cands) >= n:  # past discovered_max: an explicit deferral (held records never use up the cap)
            held.append({**e, "gate_a": {"decision": "defer", "factors": {"identifier": "doi" if doi else "pmcid",
                                                                          "caps": {"discovered_max": n}},
                                         "uncertainty": ["not evaluated: over the discovered-candidate cap"]},
                         "outcome": "deferred", "reason": acq_config.reason(acq, "defer_cap")})
            continue
        cands.append(c)
    return cands, held


def digest(cands) -> str:
    ids = sorted(f"{c.get('doi') or ''}|{c.get('pmcid') or ''}|{(c.get('discovery') or {}).get('record_id') or ''}"
                 for c in cands or ())
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ Gate A (D2)
def gate_a(disc: list, merged: list, ranked: list, served: set, acq: dict) -> dict:
    """{id(candidate): entry} for every discovered candidate: a deterministic fetch / defer / reject decision from
    existing metadata and the existing ranking stage's result, with factors and uncertainty. No score is recorded."""
    from vera import auto_acquire as aa
    from vera import search_providers
    from vera.dar_corpus import doc_key_set
    in_merged = {id(c) for c in merged}
    in_ranked = {id(r[2]) for r in ranked}
    cap = int(acq_config.bound(acq, "max_candidates_resolved"))
    out = {}
    for c in disc:
        doi = aa._doi(c)
        ident = "doi" if doi else ("pmcid" if aa._pmcid(c) else "none")
        oa = c.get("oa") if isinstance(c.get("oa"), dict) else None
        f = {"identifier": ident, "duplicate": id(c) not in in_merged,
             "availability": "open-access metadata present" if oa else "unknown before resolution",
             "caps": {"max_candidates_resolved": cap, "discovered_max": discovered_max(acq)}}
        unc = ["the article's own licence and full-text availability are unknown until resolution"]
        if not oa:
            unc.append("no open-access metadata on the discovery record")
        if id(c) in in_ranked:
            f["relevance"] = "within the existing pre-rank cut (title and abstract overlap; lexical)"
            dec, reason = "fetch", None
        elif id(c) not in in_merged:
            dec, reason = "reject", acq_config.reason(acq, "duplicate")
        else:
            why = search_providers.acq_prefilter(c)
            if why:
                dec, reason = "reject", acq_config.reason(acq, "not_oa" if why == "not_oa" else "no_licence")
            elif doc_key_set(doi, c.get("url")) & set(served or ()):
                dec, reason = "reject", acq_config.reason(acq, "in_corpus")
            else:
                f["relevance"] = "outside the existing pre-rank cut"
                dec, reason = "defer", acq_config.reason(acq, "defer_cap")
        e = {**c["discovery"], "gate_a": {"decision": dec, "factors": f, "uncertainty": unc}}
        c["_gate_a"] = {"decision": dec, "factors": copy.deepcopy(f), "uncertainty": list(unc)}  # for provenance
        if reason:
            e["outcome"], e["reason"] = ("deferred" if dec == "defer" else "rejected"), reason
        out[id(c)] = e
    return out


GATE_A_BASIS = ("deterministic (P0-5 D2): identifier present, not a duplicate, not served, not refused by the "
                "open-access pre-filter, within the existing pre-rank cut")


def gate_a_record(cand: dict, acq: dict) -> dict:
    """The provenance form of a fetched candidate's Gate A decision: the factors and uncertainty actually computed
    for it by gate_a() (no score)."""
    g = cand.get("_gate_a") if isinstance(cand.get("_gate_a"), dict) else {}
    return {"decision": g.get("decision", "fetch"), "basis": GATE_A_BASIS,
            "factors": copy.deepcopy(g.get("factors") or {}), "uncertainty": list(g.get("uncertainty") or [])}


GATE_B_BASIS = ("answer-path admission over this single document met the minimum qualifying-passage rule "
                "(bounds.answer_min_admitted_spans; persisted only at bounds.persist_min_admitted_spans)")
GATE_B_REJECT_BASIS = ("answer-path admission over this single document did not meet the minimum qualifying-passage "
                       "rule (bounds.answer_min_admitted_spans)")


# ------------------------------------------------------------------------------------------------ provenance
def annotate(plan: dict, cand: dict, res: dict, route: str, chain: list | None, acq: dict) -> None:
    """Adds the P0-5 discovery/traversal record under provenance.acquisition (JSONB; marker unchanged, so the reader
    needs no change). Only called for discovered candidates (flag on)."""
    prov = plan["source"]["provenance"]
    a = prov["acquisition"]
    meta = prov.get("meta") or {}
    disc = dict(cand.get("discovery") or {})
    a["discovery"] = disc
    a["url_chain"] = list(chain or ())
    a["resolver"] = "europepmc" if route == "europepmc" else "doi-landing"
    a["repository"] = "europepmc" if route == "europepmc" else (urlsplit(
        (prov.get("item") or {}).get("fulltext_url") or "").hostname or "")
    a["content_version"] = ({"pub_types": list(res.get("pub_types") or []), "source": "Europe PMC core record"}
                            if route == "europepmc" else
                            {"pub_types": [], "source": "publisher JATS; version not stated by a repository record"})
    if disc.get("epistemic"):
        a["content_version"]["discovered_as"] = disc["epistemic"]
    a["licence_evidence"] = {"governing": "article licence: " + (
        "Europe PMC core record + every JATS licence element" if route == "europepmc" else
        "JATS licence elements of the fetched document only"), "index_metadata_licence": "not used"}
    a["inspection"] = {"injection": "pass", "converter": meta.get("converter") or meta.get("format") or "jats-xml"}
    a["gate_a"] = gate_a_record(cand, acq)
    a["retention"] = {"mode": "hash-only", "original_bytes": "not retained", "reason": RETENTION_REASON,
                      "raw_sha256": meta.get("raw_sha256"), "text_sha256": meta.get("text_sha256"),
                      "text": "converted text retained as immutable spans"}


def gate_b_admit(plan: dict) -> None:
    plan["source"]["provenance"]["acquisition"]["gate_b"] = {"decision": "admit", "basis": GATE_B_BASIS}


# ------------------------------------------------------------------------------------------------ Step B (D4)
# Security checklist work/demo-1022/p0-5-sec/CHECKLIST.md: every Step B request (robots, landing, target) goes through
# _b_get: https only, exact host allowlist (no suffix matching), no userinfo, port 443/none, the SSRF public-address
# check, robots.txt (fail closed on 5xx/timeout/unreachable), visited set (cycle), <= max_redirects, content type,
# Content-Encoding identity only (Accept-Encoding: identity sent), size cap, wall-clock deadline. The allowlist is
# checked BEFORE url_check and http_get. landing_doi / _guarded_get stay unchanged for their existing callers.
_ROBOTS: dict = {}
_ROBOTS_LOCK = threading.Lock()


def reset_for_tests() -> None:
    with _ROBOTS_LOCK:
        _ROBOTS.clear()


def _lcfg(acq) -> dict:
    v = (acq or {}).get("landing_route")
    return v if isinstance(v, dict) else {}


def landing_hosts(acq) -> set:
    """Landing hops: the DOI resolver hosts plus the owner-approved publisher hosts (exact names)."""
    lc = _lcfg(acq)
    return {str(h).lower() for h in list(lc.get("resolver_hosts") or ()) + list(lc.get("publisher_hosts") or ())}


def fulltext_hosts(acq) -> set:
    """JATS target hops: the owner-approved publisher hosts only (ships EMPTY: pending owner sign-off)."""
    return {str(h).lower() for h in _lcfg(acq).get("publisher_hosts") or ()}


def _norm(u: str) -> tuple:
    sp = urlsplit(u)
    return (sp.scheme.lower(), (sp.hostname or "").lower(), sp.path or "/", sp.query)


def _host_ok(u: str, hosts: set) -> bool:
    try:
        sp = urlsplit(u)
        port = sp.port
    except ValueError:
        return False
    return (sp.scheme == "https" and (sp.hostname or "").lower() in hosts and sp.username is None
            and sp.password is None and port in (None, 443))


def _b_get(url: str, hosts: set, ctypes: set, max_bytes: int, max_redirects: int, deps, acq, clock, deadline,
           chain: list | None, robots: bool = True, ok_status=(200,), timeout_s: float | None = None):
    """(body, final_url, status) or raises auto_acquire._Stop(reason). Returns status for robots (404/410 = allow)."""
    from vera import auto_acquire as aa
    from vera import rate_limit
    from vera import selective_fetch as sf
    lc = _lcfg(acq)
    check = deps.url_check or sf.is_public_url
    tmo = float(lc.get("timeout_s", 5) if timeout_s is None else timeout_s)
    end = clock() + tmo if deadline is None else min(clock() + tmo, deadline)
    visited, cur, hops = set(), url, 0
    while True:
        if clock() > end:
            raise aa._Stop(acq_config.reason(acq, "deadline" if deadline is not None and clock() > deadline
                                                 else "fetch_timeout"))  # P0DTA-DIAG: per-fetch clock != request deadline
        if not _host_ok(cur, hosts):
            raise aa._Stop(acq_config.reason(acq, "host"))
        ok = check(cur)
        if not (ok[0] if isinstance(ok, tuple) else ok):
            raise aa._Stop(acq_config.reason(acq, "host"))
        if _norm(cur) in visited:
            raise aa._Stop(acq_config.reason(acq, "cycle"))
        visited.add(_norm(cur))
        if robots and not _robots_allowed(cur, hosts, deps, acq, clock, deadline):
            raise aa._Stop(acq_config.reason(acq, "robots"))
        if chain is not None:
            chain.append(_clean(cur, 2000))
        try:
            resp = deps.http_get(cur, headers={"User-Agent": sf.USER_AGENT, "Accept-Encoding": "identity"},
                                 timeout=max(0.1, end - clock()), allow_redirects=False, stream=True)
        except rate_limit.RateLimited:
            raise aa._Stop(acq_config.reason(acq, "rate_limited")) from None
        except Exception:  # noqa: BLE001
            raise aa._Stop(acq_config.reason(acq, "unreachable")) from None
        try:
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                nxt = urljoin(cur, resp.headers["Location"])
                if _norm(nxt) in visited:  # a revisit is a cycle, reported before the redirect cap
                    raise aa._Stop(acq_config.reason(acq, "cycle"))
                hops += 1
                if hops > max_redirects:
                    raise aa._Stop(acq_config.reason(acq, "unreachable"))
                cur = nxt
                continue
            if resp.status_code not in ok_status:
                if robots is False and 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return b"", cur, resp.status_code  # robots.txt: 4xx = no rules (RFC 9309); 429/5xx fail closed
                raise aa._Stop(acq_config.reason(acq, "unreachable"))
            enc = str(resp.headers.get("Content-Encoding") or "identity").strip().lower()
            if enc != "identity":
                raise aa._Stop(acq_config.reason(acq, "encoding"))
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ctypes:
                raise aa._Stop(acq_config.reason(acq, "format_unsupported"))
            buf, size = [], 0
            for chunk in resp.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise aa._Stop(acq_config.reason(acq, "too_large"))
                if clock() > end:
                    raise aa._Stop(acq_config.reason(acq, "deadline" if deadline is not None and clock() > deadline
                                                 else "fetch_timeout"))  # P0DTA-DIAG: per-fetch clock != request deadline
                buf.append(chunk)
            return b"".join(buf), cur, resp.status_code
        finally:
            aa._close(resp)


def _robots_allowed(url: str, hosts: set, deps, acq, clock, deadline) -> bool:
    """robots.txt for the URL's host through _b_get (same allowlist, caps, deadline), cached per host (allow/parsed:
    robots_ttl_s; a failure: robots_fail_ttl_s). 200 -> parsed for our user agent; 4xx other than 429 -> allowed
    (RFC 9309); 429, 5xx, error or timeout -> NOT allowed (fail closed)."""
    from urllib import robotparser

    from vera import auto_acquire as aa
    from vera import selective_fetch as sf
    lc = _lcfg(acq)
    host = (urlsplit(url).hostname or "").lower()
    now = clock()
    with _ROBOTS_LOCK:
        hit = _ROBOTS.get(host)
    if hit is not None and now - hit[0] < float(lc.get("robots_ttl_s" if hit[1] is not False else
                                                         "robots_fail_ttl_s", 3600 if hit[1] is not False else 60)):
        rp = hit[1]
    else:
        try:
            body, _f, status = _b_get(f"https://{host}/robots.txt", hosts, {"text/plain"},
                                      int(lc.get("robots_max_bytes", 524288)), 2, deps, acq, clock, deadline, None,
                                      robots=False)
            if status != 200:
                rp = True
            else:
                p = robotparser.RobotFileParser()
                p.parse(body.decode("utf-8", "ignore").splitlines())
                rp = p
        except aa._Stop:
            rp = False
        with _ROBOTS_LOCK:
            _ROBOTS[host] = (now, rp)
            while len(_ROBOTS) > 64:
                _ROBOTS.pop(next(iter(_ROBOTS)))
    return bool(rp is True or (rp is not False and rp.can_fetch(sf.USER_AGENT, url)))


_META = re.compile(r"<meta\s[^>]*?name=[\"'](citation_xml_url|citation_pdf_url)[\"'][^>]*?>", re.I)
_CONTENT = re.compile(r"content=[\"']\s*([^\"'\s]+)", re.I)


def _xml_location(html: str) -> tuple[str | None, str]:
    """(the JATS XML URL the landing page states in citation_xml_url, else None; format seen: xml | pdf | html)."""
    pdf = False
    for m in _META.finditer(html):
        c = _CONTENT.search(m.group(0))
        if not c:
            continue
        if m.group(1).lower() == "citation_xml_url":
            return c.group(1), "xml"
        pdf = True
    return None, "pdf" if pdf else "html"


_FAMILY_LICENCE = {"by": "cc by", "by-sa": "cc by-sa", "by-nc": "cc by-nc", "by-nc-sa": "cc by-nc-sa",
                   "by-nd": "cc by-nd", "by-nc-nd": "cc by-nc-nd", "zero": "cc0"}


def landing_fetch(doi: str, cand: dict, acq: dict, deps, icfg: dict, clock=time.monotonic,
                  deadline: float | None = None, chain: list | None = None) -> tuple[bytes, str, str, dict]:
    """ONE bounded DOI -> landing -> JATS attempt (one location per candidate; depth = landing + 1 target). Returns
    (raw, final_url, landing_url, res) for build_document; raises auto_acquire._Stop / fulltext_loader.Refused."""
    from vera import auto_acquire as aa
    from vera import jats_text
    lc = _lcfg(acq)
    land, ft = landing_hosts(acq), fulltext_hosts(acq)
    chain = [] if chain is None else chain
    html_b, final, _st = _b_get(f"https://doi.org/{doi}", land, {"text/html", "application/xhtml+xml"},
                                int(lc.get("landing_max_bytes", 524288)), int(lc.get("landing_max_redirects", 2)),
                                deps, acq, clock, deadline, chain)
    loc, _fmt = _xml_location(html_b.decode("utf-8", "ignore"))
    if loc is None:  # HTML or PDF only: no PDF extractor, no HTML licence-evidence parser on this path
        raise aa._Stop(acq_config.reason(acq, "format_unsupported"))
    xml_url = urljoin(final, loc)
    # S7: untrusted URL from HTML: absolute https, on the full-text allowlist AND the same host as the final landing
    if not loc.lower().startswith("https://") or not _host_ok(xml_url, ft) or \
            (urlsplit(xml_url).hostname or "").lower() != (urlsplit(final).hostname or "").lower():
        raise aa._Stop(acq_config.reason(acq, "host"))
    raw, got, _st = _b_get(xml_url, ft, {str(c).lower() for c in (acq.get("fetch") or {}).get("content_types") or ()},
                           int(acq_config.bound(acq, "max_fetch_bytes")), int(lc.get("landing_max_redirects", 2)),
                           deps, acq, clock, deadline, chain, timeout_s=float(acq_config.bound(acq, "fetch_timeout_s")))
    # X1: the converter's guards accept these exact bytes BEFORE any other parse (jats_licences, article-type)
    try:
        conv = jats_text.convert(raw, (acq.get("fetch") or {}).get("fulltext_format", "jats-xml"), icfg["converter"])
    except Exception:  # noqa: BLE001
        raise aa._Stop(acq_config.reason(acq, "convert")) from None
    jdoi = aa._doi({"doi": (conv.get("meta") or {}).get("doi")})
    if jdoi != doi:  # L2: the document must state the same DOI the landing route was resolved for
        raise aa._Stop(acq_config.reason(acq, "identity_mismatch"))
    _refuse_article_type(raw, acq)
    lic = _licence_from_elements(aa.jats_licences(raw), acq)
    res = {"pmcid": None, "licence": lic, "licence_url": acq_config.licence_url_for(lic, acq),
           "title": _clean(cand.get("title"), 500), "authors": "", "year": "", "pub_types": [], "isbn_values": [],
           "is_book": False, "doi": doi}
    if not res["licence_url"]:
        raise aa._Stop(acq_config.reason(acq, "no_licence"))
    return raw, got, final, res


def _refuse_article_type(raw: bytes, acq: dict) -> None:
    from vera import auto_acquire as aa
    try:
        import defusedxml.ElementTree as DET
        at = str(DET.fromstring(raw).get("article-type") or "").strip().lower().replace("-", " ")
    except Exception:  # noqa: BLE001  the converter already accepted the bytes
        return
    refuse = {str(p).lower() for p in ((acq.get("resolve") or {}).get("refuse_pub_types") or ())} | {
        "retraction", "preprint", "expression of concern"}
    if at and at in refuse:
        raise aa._Stop(acq_config.reason(acq, "pub_type"))


_NC = re.compile(r"non[\s\-]*commercial|\bnc\b", re.I)
_ND = re.compile(r"no[\s\-]*deriv|\bnd\b", re.I)


def _licence_from_elements(elements: list, acq: dict) -> str:
    """L2 + publisher-tagging evidence (OAMI: a CC-BY URI beside text adding a non-commercial term). Step B has no
    repository-declared licence, so: zero <license> elements -> HOLD R-ACQ-LICENCE_ABSENT (never the "declared
    governs" fallback; ali:free_to_read alone is not a <license> and so also holds); every element needs a recognised
    licence URI (license/@xlink:href or ali:license_ref) -> else HOLD R-ACQ-NOLIC; element text that names another
    family or adds a non-commercial / no-derivatives term the URI lacks -> HOLD R-ACQ-LICENCE_MISMATCH; the URIs must
    agree on exactly one family. licence_elements_decision then re-checks the same elements (reject markers first)."""
    from vera import auto_acquire as aa
    if not elements:
        raise aa._Stop(acq_config.reason(acq, "licence_absent"))
    fams = set()
    for e in elements:
        uri = {f for f in (aa._family(e.get("href") or ""), aa._family(e.get("ref") or "")) if f}
        if len(uri) != 1:
            raise aa._Stop(acq_config.reason(acq, "no_licence" if not uri else "licence_mismatch"))
        fam = next(iter(uri))
        text = e.get("text") or ""
        tf = aa._text_family(text)
        if (tf and tf != fam) or (_NC.search(text) and "nc" not in fam.split("-")) or \
                (_ND.search(text) and "nd" not in fam.split("-")):
            raise aa._Stop(acq_config.reason(acq, "licence_mismatch"))
        fams.add(fam)
    if len(fams) != 1:
        raise aa._Stop(acq_config.reason(acq, "licence_mismatch"))
    fam = next(iter(fams))
    if fam not in _FAMILY_LICENCE:
        raise aa._Stop(acq_config.reason(acq, "no_licence"))
    return _FAMILY_LICENCE[fam]
