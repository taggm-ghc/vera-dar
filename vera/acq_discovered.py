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
import json
import logging
import math
import re
import sys
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


def discovered_candidates(sources, acq, question: str | None = None) -> tuple[list, list]:
    """(acquisition candidates, held entries). Stored-corpus items are skipped (already evidence). Candidates carry
    doi/pmcid/url/title/snippet plus `discovery` (service, record id, url, epistemic) and acq_source "discovered";
    the index licence and stored_text_kind are DROPPED (never copied). Held entries are discovered records that cannot
    be acquired, each with an explicit Gate A reject reason (no identifier / preprint licence unverified)."""
    from vera import auto_acquire as aa
    from vera.search_providers import _pmcid_of
    cands, held = [], []
    n = discovered_max(acq)
    off = lambda s: False  # noqa: E731
    if question:  # DDFIX-1023/REL D4: relevance order (and off-topic drop) BEFORE the discovered_max cut
        from vera import search_query
        kc = search_query.key_concepts(question, extra_generic=(((acq or {}).get("search") or {}).get(
            "openalex_profile") or {}).get("generic_terms") or ())
        sources = [s for s in sources or () if isinstance(s, dict)]
        tiers = dict(zip(map(id, sources), search_query.relevance_tiers(
            kc, [f"{s.get('title') or ''} {s.get('snippet') or ''}" for s in sources])))
        sources = sorted(sources, key=lambda s: tiers[id(s)])  # stable: provider order within a tier
        off = lambda s: tiers.get(id(s)) == 2  # noqa: E731
    for s in sources or ():
        if not isinstance(s, dict) or any(s.get(k) for k in _STORED_KEYS):
            continue
        doi = aa._doi(s)
        pmcid = None if doi else _pmcid_of(s.get("pmcid"))
        kind = _preprint_kind(s, doi)
        e = _entry(s, kind is not None)
        if off(s):  # DDFIX-1023/REL: dropped before any cap or slot (lexical false positive of the discovery search)
            held.append({**e, "gate_a": {"decision": "reject", "factors": {
                "relevance": "no key concept of the question in the title or abstract (deterministic, lexical)"},
                "uncertainty": ["judged on the listed title and abstract only"]},
                "outcome": "dropped", "reason": acq_config.reason(acq, "irrelevant")})
            continue
        from vera import arxiv_fulltext  # ARX-A2: arXiv records (no journal DOI) -> arXiv route when enabled
        aid = arxiv_fulltext.arxiv_ident(s) if arxiv_fulltext.PORTAL_ONLY else None
        if aid is not None and (not doi or arxiv_fulltext._ARXIV_DOI.match(doi)):
            # ARXIV-PORTAL-1110: an arXiv record without a journal DOI is a PORTAL seed (vera/arxiv_portal.py): followed
            # to its published version and references, never read; ranking drops it with R-ACQ-ARXIV_PORTAL
            # (a held portal entry, not a candidate: nothing is read; `_portal_seed` is taken off by inline_acquire and
            # handed to arxiv_portal.extend, never shown)
            seed = {"doi": f"10.48550/arxiv.{aid[0]}", "url": f"https://arxiv.org/abs/{aid[0]}{aid[1] or ''}",
                    "arxiv_id": aid[0], "title": " ".join(str(s.get("title") or "").split()),
                    "snippet": " ".join(str(s.get("snippet") or "").split()), "acq_source": "discovered"}
            held.append({**e, "gate_a": {"decision": "portal", "factors": {
                "identifier": "arxiv", "portal": "arXiv record: followed to its published version and its references; "
                                                 "never read"}, "uncertainty": []},
                "outcome": "portal", "reason": acq_config.reason(acq, "arxiv_portal"), "_portal_seed": seed})
            continue
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
def gate_a(disc: list, merged: list, ranked: list, served: set, acq: dict, drops: dict | None = None) -> dict:
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
        irrelevant = acq_config.reason(acq, "irrelevant")
        if id(c) in in_ranked:
            f["relevance"] = "within the existing pre-rank cut (title and abstract overlap; lexical)"
            dec, reason = "fetch", None
        elif id(c) not in in_merged:
            dec, reason = "reject", acq_config.reason(acq, "duplicate")
        elif (drops or {}).get(id(c)) == irrelevant:  # DDFIX-1023/REL: off-topic before the cut (title + abstract)
            f["relevance"] = "no key concept of the question in the title or abstract (deterministic, lexical)"
            dec, reason = "reject", irrelevant
        elif (drops or {}).get(id(c)) == acq_config.reason(acq, "arxiv_portal"):
            # ARXIV-PORTAL-1110: an arXiv record is a portal entry (followed to its published version and references by
            # vera/arxiv_portal.py), never a source to acquire
            f["portal"] = "arXiv record: followed to its published version and its references; never read"
            dec, reason = "portal", drops[id(c)]
        elif (drops or {}).get(id(c)) in (acq_config.reason(acq, "duplicate"), acq_config.reason(acq, "no_doi")):
            # WP7 (MIDTERMS-WP7-1106 c): the ranking stage's own reason (an in-request duplicate, no identifier) is
            # reported as it is, not folded into the candidate-cut label below
            dec, reason = "reject", drops[id(c)]
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
            e["outcome"], e["reason"] = ({"defer": "deferred", "portal": "portal"}.get(dec, "rejected")), reason
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
_TDMREP: dict = {}  # CONTROL-GAPS-1107: host -> (monotonic time, parsed tdmrep.json rules | "unreadable")
_CRAWL_LAST: dict = {}  # H-5 / H5-CRAWL-1107: host -> monotonic time of its latest BOOKED request slot (may be ahead)
_sleep = time.sleep  # test seam (tests never sleep for real)


_CAUSE_RE_HOST = re.compile(r"[a-z0-9.-]{1,253}")
_CAUSE_RE_CTYPE = re.compile(r"[a-z0-9.+/-]{1,64}")
_CACHED_LOGGED: dict = {}  # host -> monotonic time of the last cached=1 cause line (once per host per fail-TTL)
_DENIED: dict = {}  # WPL-1111 R2: identity -> time; process memory only (Q-4: a restart clears it)
_DENIED_LOCK = threading.Lock()
_DENIED_MAX = 4096  # A-3: FIFO bound; after 4096 later denials the oldest may be evicted (one re-request possible)
# R1 (OD-1111-ROBOTS-4XX): a robots.txt 4xx other than 404/410 is a stop, cached as long as the old "allow" was.
_STATUS_CAUSES = ("STATUS_401", "STATUS_403", "STATUS_4XX")


class _StdoutHandler(logging.StreamHandler):
    """R3: writes to the CURRENT sys.stdout at emit time (so a capture that swaps sys.stdout sees the line)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):  # noqa: D401
        return sys.stdout

    @stream.setter
    def stream(self, _v):
        pass


cause_logger = logging.getLogger("vera.robots_cause")
if not any(isinstance(h, _StdoutHandler) for h in cause_logger.handlers):
    cause_logger.addHandler(_StdoutHandler())
cause_logger.setLevel(logging.INFO)
cause_logger.propagate = False  # main.py basicConfig writes to stderr: no second copy of the line


def _san(v, rx, fallback: str) -> str:
    s = str(v or "").strip().lower()
    return s if rx.fullmatch(s) else fallback


def cause_log(host, hop: str, code: str, cause: str, cached: bool = False, status=None, ctype=None, nbytes=None,
              ms=0) -> None:
    """R3 / GOV-WPL-1111 RC-4: one stdout line per robots failure or access denial. Every field is fixed-set or
    sanitised; no URL path or query, no question text, no body, no header value beyond a signal name. Never raises."""
    try:
        def _i(v):
            try:
                return str(int(v))
            except (TypeError, ValueError):
                return "-"
        fixed = re.compile(r"[A-Za-z0-9_:.-]{1,64}")
        c1 = code if fixed.fullmatch(str(code)) else "-"
        c2 = cause if fixed.fullmatch(str(cause)) else "-"
        h = hop if hop in ("route", "source", "portal", "web") else "-"
        ct = "-" if ctype is None else _san(str(ctype).split(";")[0], _CAUSE_RE_CTYPE, "other")
        cause_logger.info("robots_cause host=%s hop=%s code=%s cause=%s cached=%d status=%s ctype=%s bytes=%s ms=%s",
                          _san(host, _CAUSE_RE_HOST, "-"), h, c1, c2, 1 if cached else 0, _i(status) if status
                          is not None else "-", ct, _i(nbytes) if nbytes is not None else "-", _i(ms))
    except Exception:  # noqa: BLE001  diagnostics never block
        pass


def _hop_of(url: str, acq=None) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return "route" if host == "www.ebi.ac.uk" else "source"


_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def _norm_path(path: str) -> str:
    """DENIAL-COVERAGE-1111 defect 2: one key per document. Percent-escapes of unreserved characters are decoded
    (other escapes keep upper-case hex), dot-segments are resolved and leading slashes collapse to one."""
    def _unesc(m):
        ch = chr(int(m.group(1), 16))
        return ch if ch in _UNRESERVED else "%" + m.group(1).upper()
    p = re.sub(r"%([0-9A-Fa-f]{2})", _unesc, path or "/")
    out = []
    for s in p.lstrip("/").split("/"):
        if s == ".":
            continue
        if s == "..":
            if out:
                out.pop()
            continue
        out.append(s)
    return "/" + "/".join(out)


def _id_url(u) -> str:
    try:
        sp = urlsplit(str(u or ""))
        host = (sp.hostname or "").lower().rstrip(".")
        return f"url:{host}{_norm_path(sp.path)}" if host else ""
    except ValueError:
        return ""


def denial_ids(doi=None, pmcid=None, urls=()) -> set:
    """R2 identities: normalised DOI, PMCID, and each URL as host + path (no query)."""
    out = set()
    d = str(doi or "").strip().lower()
    if d:
        out.add(f"doi:{d}")
    m = str(pmcid or "").strip().upper()
    if m:
        out.add(f"pmc:{m}")
    for u in urls or ():
        k = _id_url(u)
        if k:
            out.add(k)
    return out


def deny(ids) -> None:
    """Remember an access denial (R2): the same document is never requested again in this process."""
    now = time.monotonic()
    with _DENIED_LOCK:
        for k in ids or ():
            _DENIED.pop(k, None)
            _DENIED[k] = now
        while len(_DENIED) > _DENIED_MAX:
            _DENIED.pop(next(iter(_DENIED)))


def is_denied(doi=None, pmcid=None, urls=()) -> bool:
    ids = denial_ids(doi, pmcid, urls)
    with _DENIED_LOCK:
        return any(k in _DENIED for k in ids)


def refusal_code(acq, key: str) -> str:
    """reason code for a refusal key, with an optional -SUFFIX ("robots_unreadable-TIMEOUT" -> R-ACQ-ROBOTS_UNREADABLE-
    TIMEOUT; the same scheme as R-ACQ-RATELIMITED-<REASON>)."""
    base, _, suf = str(key).partition("-")
    return acq_config.reason(acq, base) + (f"-{suf}" if suf else "")


def _caused(aa, acq, key: str, cause: str, status=None):
    """A _Stop for reason `key` that records the robots-read cause (diagnostics only; the code is unchanged)."""
    stop = aa._Stop(acq_config.reason(acq, key))
    stop.cause = cause
    stop.status = status
    return stop


def access_denied_stop(acq, status: int, urls, route_hop: bool = False, want_json: bool = False):
    """R2: a 401/403 on a document endpoint. Writes the URL identities (the handler adds DOI / PMCID) and returns the
    _Stop to raise. want_json (the Europe PMC search endpoint) is ROUTE_DENIED and is not remembered (A-1)."""
    from vera import auto_acquire as aa
    host = ""
    for u in urls or ():
        host = (urlsplit(str(u)).hostname or "").lower()
    if want_json:
        cause_log(host, "route", acq_config.reason(acq, "route_denied"), f"STATUS_{int(status)}", False, status)
        return aa._Stop(acq_config.reason(acq, "route_denied"))
    ids = denial_ids(urls=urls)
    deny(ids)
    code = refusal_code(acq, "access_denied-ROUTE" if route_hop else "access_denied")
    cause_log(host, "route" if route_hop else "source", code, f"STATUS_{int(status)}", False, status)
    stop = aa._Stop(code)
    stop.denied_ids = ids
    return stop


def reset_for_tests() -> None:
    with _ROBOTS_LOCK:
        _ROBOTS.clear()
        _TDMREP.clear()
        _CRAWL_LAST.clear()
        _CACHED_LOGGED.clear()
    with _DENIED_LOCK:
        _DENIED.clear()


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
        if robots and is_denied(urls=(cur,)):  # WPL-1111 R2: a denied document URL is never requested again (every hop)
            raise aa._Stop(refusal_code(acq, "access_denied"))
        from vera import host_role  # PORTAL-RULE-1110 CP3: every landing / full-text hop is ORIGINAL / RoR / a resolver
        if robots and not host_role.is_hop_allowed(cur, "store"):  # robots.txt / tdmrep.json reads are not documents
            raise aa._Stop(acq_config.reason(acq, "portal"))
        ok = check(cur)
        if not (ok[0] if isinstance(ok, tuple) else ok):
            raise aa._Stop(acq_config.reason(acq, "host"))
        if _norm(cur) in visited:
            raise aa._Stop(acq_config.reason(acq, "cycle"))
        visited.add(_norm(cur))
        if robots:  # CONTROL-GAPS-1107: robots.txt AND the TDM opt-out, the one per-hop access gate (hop_access)
            t_gate = clock()
            refused = hop_access(cur, hosts, deps, acq, clock, deadline)
            if refused:
                raise aa._Stop(refusal_code(acq, refused))
            g = max(0.0, clock() - t_gate)  # H5-CRAWL-1107: the gate's own (paced) reads are not this fetch's time
            end = end + g if deadline is None else min(end + g, deadline)
        if chain is not None:
            chain.append(_clean(cur, 2000))
        w = crawl_wait(cur, acq, clock, deadline)  # H-5 / H5-CRAWL-1107: every request to the host, gate reads too
        end = end + w if deadline is None else min(end + w, deadline)  # the wait is not fetch time
        try:
            resp = deps.http_get(cur, headers={"User-Agent": sf.USER_AGENT, "Accept-Encoding": "identity"},
                                 timeout=max(0.1, end - clock()), allow_redirects=False, stream=True)
        except rate_limit.RateLimited as e:
            raise aa._Stop(acq_config.rate_limited_code(acq, e.reason)) from None  # F1: per-reason code
        except rate_limit.UnlistedHost:  # OA-1: no portal for the host, nothing sent
            raise aa._Stop(acq_config.reason(acq, "host")) from None
        except Exception:  # noqa: BLE001
            raise _caused(aa, acq, "unreachable", "NETWORK") from None
        try:
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                nxt = urljoin(cur, resp.headers["Location"])
                if _norm(nxt) in visited:  # a revisit is a cycle, reported before the redirect cap
                    raise aa._Stop(acq_config.reason(acq, "cycle"))
                hops += 1
                if hops > max_redirects:
                    raise _caused(aa, acq, "unreachable", "REDIRECTS")
                cur = nxt
                continue
            if resp.status_code not in ok_status:
                if robots and resp.status_code == 429:
                    # H5-CRAWL-1107: a server 429 on a document request is a rate limit, not a failed attempt (owner
                    # rule: 429/cooldown tries never count): background-eligible, never R-ACQ-UNREACHABLE.
                    raise aa._Stop(acq_config.rate_limited_code(acq, "throttled"))
                if robots is False and resp.status_code in _throttle_statuses(cur):
                    # CONTROL-GAPS-1107 r2: a server throttle (429 / the portal's throttle_statuses) on robots.txt or
                    # tdmrep.json is a rate limit, not a verdict: refused now, never cached, never an attempt.
                    raise aa._Stop(acq_config.reason(acq, "rate_limited"))
                if robots and resp.status_code in (401, 403):
                    # WPL-1111 R2 (OD-1111-ACCESS-DENIED-NO-BYPASS): an access denial on the document endpoint is not
                    # unreachable and is never retried or bypassed (the requested and the refusing URL are remembered)
                    raise access_denied_stop(acq, resp.status_code, (url, cur))
                if robots is False and 400 <= resp.status_code < 500 and resp.status_code != 429:
                    return b"", cur, resp.status_code  # robots.txt 4xx: the status goes to _robots_verdict (R1)
                raise _caused(aa, acq, "unreachable", "STATUS_5XX" if resp.status_code >= 500 else "STATUS_OTHER",
                              resp.status_code)
            enc = str(resp.headers.get("Content-Encoding") or "identity").strip().lower()
            if enc != "identity":
                raise aa._Stop(acq_config.reason(acq, "encoding"))
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ctypes:
                raise aa._Stop(acq_config.reason(acq, "format_unsupported"))
            if robots and tdm_header_reserved(resp.headers):  # CONTROL-GAPS-1107: TDM-Reservation response header
                raise aa._Stop(acq_config.reason(acq, "tdm"))
            sig = page_no_ai_reason(resp.headers) if robots else None  # ROBOTS-FOLLOWUP-1108: header no-AI signals
            if sig:
                raise page_signal_stop(acq, sig, cur)
            buf, size = [], 0
            for chunk in resp.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise aa._Stop(acq_config.reason(acq, "too_large"))
                if clock() > end:
                    raise aa._Stop(acq_config.reason(acq, "deadline" if deadline is not None and clock() > deadline
                                                 else "fetch_timeout"))  # P0DTA-DIAG: per-fetch clock != request deadline
                buf.append(chunk)
            body = b"".join(buf)
            if robots and tdm_body_reserved(body, ctype):  # CONTROL-GAPS-1107: <meta name="tdm-reservation" content="1">
                raise aa._Stop(acq_config.reason(acq, "tdm"))
            sig = page_no_ai_reason({}, body, ctype) if robots else None  # <meta> noai / ai-grounding in the page
            if sig:
                raise page_signal_stop(acq, sig, cur)
            return body, cur, resp.status_code
        finally:
            aa._close(resp)


def _stop_cause(stop, acq) -> str:
    """The robots-read cause key of a stop of the gate's own robots.txt request (WPL-1111)."""
    c = getattr(stop, "cause", None)
    if c:
        return c
    code = str(getattr(stop, "code", ""))
    for key, cause in (("fetch_timeout", "TIMEOUT"), ("encoding", "ENCODING"), ("format_unsupported", "CTYPE"),
                       ("too_large", "SIZE"), ("cycle", "CYCLE"), ("unreachable", "NETWORK")):
        if code == acq_config.reason(acq, key):
            return cause
    return "PARSE"


def _robots_verdict(url: str, hosts: set, deps, acq, clock, deadline) -> tuple:
    """robots.txt for the URL's host through _b_get (same allowlist, caps, deadline), cached per host. Returns
    ("allow" | "disallow" | "unreadable", cause): cause is None for allow, a verdict name for disallow, and the
    read-failure key for unreadable (CACHED_<key> when served from the cache). 200 -> parsed for our user agent;
    404/410 -> allowed; any other 4xx except 429 -> unreadable (R1, OD-1111-ROBOTS-4XX: a recorded deviation from
    RFC 9309 section 2.3.1.3; held for robots_ttl_s, so no request is added by the stop); 429, 5xx, error or
    timeout -> unreadable (fail closed). The cache value is (time, rp, cause); readers use [0] and [1] only, and a
    2-tuple (older fixtures) has no cause (fail TTL, cause CACHED)."""
    from urllib import robotparser

    from vera import auto_acquire as aa
    from vera import selective_fetch as sf
    lc = _lcfg(acq)
    host = (urlsplit(url).hostname or "").lower()
    hop = _hop_of(url)
    now = clock()
    with _ROBOTS_LOCK:
        hit = _ROBOTS.get(host)
    fresh, cause_w = False, None
    if hit is not None:
        cause_w = hit[2] if len(hit) > 2 else None
        long_ttl = hit[1] is not False or cause_w in _STATUS_CAUSES  # RC-1: a 4xx stop is held as the old allow was
    if hit is not None and now - hit[0] < float(lc.get("robots_ttl_s" if long_ttl else "robots_fail_ttl_s",
                                                         3600 if long_ttl else 60)):
        rp = hit[1]
    else:
        fresh = True
        t0 = clock()
        status_seen, ctype_seen, bytes_seen = None, None, None
        try:
            body, _f, status = _b_get(f"https://{host}/robots.txt", hosts, {"text/plain"},
                                      int(lc.get("robots_max_bytes", 524288)), 2, deps, acq, clock, deadline, None,
                                      robots=False)
            status_seen, bytes_seen = status, len(body)
            if status in (404, 410):
                rp = True
            elif status != 200:
                rp, cause_w = False, ("STATUS_401" if status == 401 else "STATUS_403" if status == 403
                                      else "STATUS_4XX")
            else:
                p = robotparser.RobotFileParser()
                txt = body.decode("utf-8", "ignore")
                p.parse(txt.splitlines())
                from vera import robots_ai as _ra  # OD-1106-ROBOTS-AI
                _ra.annotate(p, txt)
                from vera import source_permissions as _sp  # ROBOTS-FOLLOWUP-1108: Content-Signal / Content-Usage lines
                p.vera_use_signal = _sp.robots_content_signal(txt)
                rp = p
        except aa._Stop as e:
            if _passthrough(e, acq):
                raise  # CONTROL-GAPS-1107: refused as host / rate-limited (nothing sent), never cached as robots
            rp, cause_w = False, _stop_cause(e, acq)
            status_seen = getattr(e, "status", None)
        except Exception:  # noqa: BLE001  any other stop, parse or config error: fail closed
            rp, cause_w = False, "PARSE"
        with _ROBOTS_LOCK:
            _ROBOTS[host] = (now, rp, cause_w if rp is False else None)
            while len(_ROBOTS) > 64:
                _ROBOTS.pop(next(iter(_ROBOTS)))
        if rp is False:
            cause_log(host, hop, refusal_code(acq, "robots_unreadable-" + cause_w), cause_w, False, status_seen,
                      ctype_seen, bytes_seen, int((clock() - t0) * 1000))
    if rp is True:
        return "allow", None
    if rp is False:
        if not fresh:
            ttl = float(lc.get("robots_ttl_s", 3600)) if cause_w in _STATUS_CAUSES else float(
                lc.get("robots_fail_ttl_s", 60))
            with _ROBOTS_LOCK:
                last = _CACHED_LOGGED.get(host)
                first = last is None or now - last >= ttl
                if first:
                    _CACHED_LOGGED[host] = now
            if first:
                cause_log(host, hop, refusal_code(acq, "robots_unreadable-CACHED_" + (cause_w or "UNKNOWN")),
                          "CACHED_" + (cause_w or "UNKNOWN"), True)
        if fresh:
            return "unreadable", cause_w
        return "unreadable", ("CACHED_" + cause_w) if cause_w else "CACHED"
    try:
        from vera import robots_ai as _ra
        from vera import source_permissions as _sp
        why = None
        if _sp.robots_no_ai(_sp.robots_entries(getattr(rp, "vera_use_signal", ""))):  # ai-input=no / ai-use=n anywhere
            why = "verdict:content_signal"
        elif not _ra.allowed(rp, sf.USER_AGENT, url):
            why = "verdict:disallow"
        elif _ra.block_reason(rp, url) is not None:
            why = "verdict:ai_statement"
        if why is None:
            return "allow", None
        if fresh:
            cause_log(host, hop, acq_config.reason(acq, "robots"), why, False)
        return "disallow", why
    except Exception:  # noqa: BLE001
        if fresh:
            cause_log(host, hop, refusal_code(acq, "robots_unreadable-EVAL"), "EVAL", False)
        return "unreadable", "EVAL"  # H3: an exception while evaluating is "could not be checked", not a rule


def _robots_allowed(url: str, hosts: set, deps, acq, clock, deadline) -> bool:
    """True only when the robots verdict is allow (fail closed). Wrapper kept for its external callers."""
    return _robots_verdict(url, hosts, deps, acq, clock, deadline)[0] == "allow"


def _passthrough(stop, acq) -> bool:
    """CONTROL-GAPS-1107: a stop of the gate's own robots.txt / tdmrep.json request that is NOT a verdict about the
    file: the host or its address was refused, the limiter skipped or the server throttled (r2: R-ACQ-RATELIMITED*),
    or the JOB deadline ran out (r2). It is raised as itself (still a refusal for this request: fail closed) and is
    not cached, so neither ROBOTS nor TDM is recorded for a host that never answered."""
    code = str(getattr(stop, "code", ""))
    return (code in (acq_config.reason(acq, "host"), acq_config.reason(acq, "deadline"),
                     acq_config.reason(acq, "crawl_delay"))  # H5-CRAWL-1107: a pacing stop sent nothing
            or code.startswith(acq_config.reason(acq, "rate_limited")))


def _throttle_statuses(url: str) -> set:
    """429 plus the rate_limits.json portal's throttle_statuses for the URL's host (config/rate_limits.json)."""
    out = {429}
    try:
        from vera import rate_limit
        ad = rate_limit.get_registry().portal_for_url(url)
        if ad is not None:
            out |= {int(s) for s in ad.c.get("throttle_statuses") or ()}
    except Exception:  # noqa: BLE001  config error: 429 alone (the rest stay fail-closed refusals)
        pass
    return out


def crawl_enforced(acq) -> bool:
    """H-5 (approved OD-1107-HITL; H5-CRAWL-1107): config auto_acquire.json crawl_delay.enforce_on_acquisition,
    exactly true (shipped true; missing or non-true = off)."""
    c = (acq or {}).get("crawl_delay")
    return isinstance(c, dict) and c.get("enforce_on_acquisition") is True


def crawl_interval(url: str, rp=None) -> float:
    """H5-CRAWL-1107 (owner OD-1107-HITL H-5(a)): the per-host spacing of acquisition requests, from config only: the
    larger (stricter) of the host's robots.txt Crawl-delay (robots_ai.crawl_delay: own UA, * and AI-agent groups) and
    the spacing of the host's documented rate in config/rate_limits.json (1 / rate_per_s of its portal; rate_per_s is
    set at or below the documented ceiling there). 0.0 when the host declares no Crawl-delay: its portal's token
    bucket (vera.rate_limit, on every acquisition request through deps.http_get) then paces it alone. inf = a
    Crawl-delay that is not a finite number (the request is never sent)."""
    from vera import robots_ai
    from vera import selective_fetch as sf
    cd = robots_ai.crawl_delay(rp, sf.USER_AGENT) if rp not in (None, True, False) else None
    if not cd:
        return 0.0
    cd = float(cd)
    if not math.isfinite(cd) or cd < 0:
        return math.inf
    return max(cd, _documented_spacing(url))


def _documented_spacing(url: str) -> float:
    """1 / rate_per_s of the config/rate_limits.json portal serving the URL's host; 0.0 when none (or disabled)."""
    try:
        from vera import rate_limit
        ad = rate_limit.get_registry().portal_for_url(url)
        if ad is None or ad.c.get("enabled") is False:
            return 0.0
        r = float(ad.c.get("rate_per_s") or 0.0)
        return 1.0 / r if r > 0 else 0.0
    except Exception:  # noqa: BLE001  config error: Crawl-delay alone (never less strict than robots)
        return 0.0


def _min_call_s(url: str) -> float:
    """config/rate_limits.json min_call_s (the host's portal, else the defaults): the least time left for a request to
    be worth sending. A slot later than deadline - min_call_s is not booked."""
    try:
        from vera import rate_limit
        ad = rate_limit.get_registry().portal_for_url(url)
        if ad is not None:
            return float(ad.c.get("min_call_s") or 0.0)
        return float(rate_limit.load_config()["defaults"]["min_call_s"])
    except Exception:  # noqa: BLE001
        return 1.0


def crawl_wait(url: str, acq, clock, deadline) -> float:
    """H-5 (only when crawl_enforced; H5-CRAWL-1107: shipped true): every acquisition request to a host (robots.txt,
    tdmrep.json, landing, API and full text) books the host's next free slot UNDER the lock, then sleeps until it
    outside the lock, so concurrent workers are serialized per host: slot = max(now, last slot + crawl_interval).
    A slot that would leave less than min_call_s before the deadline is not booked and stops with R-ACQ-CRAWL_DELAY
    (no request sent: not an attempt; inline it is background-eligible, vera.inline_acquire._transient). A booked
    slot is never moved earlier. Off: no effect at all. Returns the seconds waited (the caller's per-fetch clock does
    not count them)."""
    if not crawl_enforced(acq):
        return 0.0
    from vera import auto_acquire as aa
    host = (urlsplit(url).hostname or "").lower()
    with _ROBOTS_LOCK:
        rp = (_ROBOTS.get(host) or (0, None))[1]
    gap = crawl_interval(url, rp)
    if gap == math.inf:
        raise aa._Stop(acq_config.reason(acq, "crawl_delay"))
    margin = _min_call_s(url) if gap else 0.0
    with _ROBOTS_LOCK:
        now = clock()
        last = _CRAWL_LAST.get(host)
        slot = now if last is None or not gap else max(now, last + gap)
        if slot > now and deadline is not None and slot + margin > deadline:
            raise aa._Stop(acq_config.reason(acq, "crawl_delay"))  # nothing booked, nothing sent
        _CRAWL_LAST.pop(host, None)
        _CRAWL_LAST[host] = slot if last is None else max(last, slot)
        while len(_CRAWL_LAST) > 512:
            _CRAWL_LAST.pop(next(iter(_CRAWL_LAST)))
    wait = slot - now
    if wait > 0:
        from vera import progress as _prog  # NOISY-UI
        with _prog.waiting(_prog.ACT_POLITE, wait):
            _sleep(wait)
        return wait
    return 0.0


def _tdmrep_allowed(url: str, hosts: set, deps, acq, clock, deadline) -> bool:
    """CONTROL-GAPS-1107 (TDM Reservation Protocol, the same rule as public_fetch._tdmrep_signal): the host's
    /.well-known/tdmrep.json through _b_get (same allowlist, caps, deadline; robots=False so 4xx returns its status),
    cached per host with the robots TTLs. 404/410 -> no reservation; 200 -> public_fetch.tdmrep_reserved for the URL's
    path; any other status, error, timeout or malformed JSON -> reserved (fail closed)."""
    from vera import auto_acquire as aa
    from vera import public_fetch
    lc = _lcfg(acq)
    sp = urlsplit(url)
    host = (sp.hostname or "").lower()
    now = clock()
    with _ROBOTS_LOCK:
        hit = _TDMREP.get(host)
    ok_hit = hit is not None and hit[1] != "unreadable"
    if hit is not None and now - hit[0] < float(lc.get("robots_ttl_s" if ok_hit else "robots_fail_ttl_s",
                                                       3600 if ok_hit else 60)):
        rules = hit[1]
    else:
        try:
            body, _f, status = _b_get(f"https://{host}/.well-known/tdmrep.json", hosts,
                                      {"application/json", "text/plain", ""}, int(lc.get("robots_max_bytes", 524288)),
                                      2, deps, acq, clock, deadline, None, robots=False)
            if status in (404, 410):
                rules = []
            elif status != 200:
                rules = "unreadable"
            else:
                rules = json.loads(body.decode("utf-8", "replace"))
                public_fetch.tdmrep_reserved(rules, "/")  # shape check now; ValueError -> unreadable
        except aa._Stop as e:
            if _passthrough(e, acq):
                raise  # refused as host / rate-limited (nothing sent), never cached
            rules = "unreadable"
        except Exception:  # noqa: BLE001  bad JSON, malformed rules: fail closed
            rules = "unreadable"
        with _ROBOTS_LOCK:
            _TDMREP[host] = (now, rules)
            while len(_TDMREP) > 64:
                _TDMREP.pop(next(iter(_TDMREP)))
    if rules == "unreadable":
        return False
    try:
        if public_fetch.tdmrep_reserved(rules, sp.path or "/"):
            public_fetch._tdm_purpose_shadow(rules, url)  # PURPOSE-GRANTS-1108: shadow record only; still refused
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def tdm_header_reserved(headers) -> bool:
    """CONTROL-GAPS-1107: a TDM-Reservation response header of 1 (source_permissions._tdm_reserved, the same rule)."""
    from vera import source_permissions
    try:
        return source_permissions._tdm_reserved({"tdm-reservation": (headers or {}).get("TDM-Reservation") or ""})
    except Exception:  # noqa: BLE001
        return True


def tdm_body_reserved(body: bytes, ctype: str) -> bool:
    """CONTROL-GAPS-1107: an HTML page's <meta name="tdm-reservation" content="1"> (public_fetch._tdm_meta)."""
    if (ctype or "").split(";")[0].strip().lower() not in ("text/html", "application/xhtml+xml"):
        return False
    from vera import public_fetch
    try:
        return public_fetch._tdm_meta(body[:200000].decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return True


def page_no_ai(headers, body: bytes = b"", ctype: str = "") -> bool:
    """True when page_no_ai_reason finds a signal or an error (fail closed)."""
    return bool(page_no_ai_reason(headers, body, ctype))


def page_no_ai_reason(headers, body: bytes = b"", ctype: str = "") -> str | None:
    """WPL-1111: the signal name (X_ROBOTS_TAG, META, CONTENT_SIGNAL, CONTENT_USAGE, AI_GROUNDING), "error" when the
    check itself failed (fail closed; not a signal), else None. The rule is page_no_ai's (below).

    ROBOTS-FOLLOWUP-1108: the page-level use-based no-AI signals the web route applies (source_permissions.evaluate),
    read on an acquisition response: X-Robots-Tag or <meta name=robots|googlebot|vera-dar|ai> noai / noimageai /
    noai-crawl / nocrawl, Content-Signal ai-input=no, Content-Usage ai-use=n, and Bing ai-grounding Disallow or
    DisallowExtended (header or <meta name=Bingbot>). Evidence use is barred, so the document is not acquired. Any
    error is True (fail closed)."""
    try:
        from vera import source_permissions as sp
        h = sp._lower_headers(headers)
        head = ""
        if (ctype or "").split(";")[0].strip().lower() in ("text/html", "application/xhtml+xml") and body:
            head = body[:200000].decode("utf-8", "replace")
        xrt = sp._tokens(h.get("x-robots-tag", ""))
        meta = sp._tokens(sp._meta_robots(head))
        bad = {"noai", "noimageai", "noai-crawl", "nocrawl"}
        if bad & xrt:
            return "X_ROBOTS_TAG"
        if bad & meta:
            return "META"
        if sp._content_signal_no_ai(h.get("content-signal", "")):
            return "CONTENT_SIGNAL"
        if sp._content_signal_no_ai(sp.content_usage_signal(h.get("content-usage", ""))):
            return "CONTENT_USAGE"
        if sp.ai_grounding_disallow(h.get("x-robots-tag", "")) or sp.ai_grounding_disallow(sp._meta_robots(head)):
            return "AI_GROUNDING"
        return None
    except Exception:  # noqa: BLE001
        return "error"


def page_signal_stop(acq, sig: str, url: str, route_hop: bool = False):
    """The _Stop for a page_no_ai_reason result at a source hop (or, route_hop, the full-text service hop), with the
    stdout cause line. A signal is ROBOTS-PAGE_NOAI / ROUTE_NOAI-<SIGNAL>; a check error is ...UNREADABLE-EVAL."""
    from vera import auto_acquire as aa
    if sig == "error":
        key = "route_unreadable-EVAL" if route_hop else "robots_unreadable-EVAL"
        if route_hop and not landing_on(acq):
            key = "noroute_unchecked-EVAL"
    else:
        key = f"route_noai-{sig}" if route_hop else "robots-PAGE_NOAI"
    code = refusal_code(acq, key)
    cause_log((urlsplit(url).hostname or ""), "route" if route_hop else "source", code, sig if sig != "error" else
              "EVAL", False)
    return aa._Stop(code)


def page_signal_key(reason: str) -> str:
    """Refusal key for a page_no_ai_reason result: "robots-PAGE_NOAI" for a signal (the page says no),
    "robots_unreadable-EVAL" for an error (the check failed: not a rule)."""
    return "robots_unreadable-EVAL" if reason == "error" else "robots-PAGE_NOAI"


def hop_access(url: str, hosts: set, deps, acq, clock, deadline) -> str | None:
    """CONTROL-GAPS-1107: THE access gate for every acquisition hop (acq_discovered._b_get and auto_acquire._guarded_get,
    which serves Europe PMC and arXiv). Returns the refusal reason key ("robots" or "tdm") or None. robots.txt first
    (owner R-050 / OD-1106-ROBOTS-AI), then the TDM Reservation Protocol (/.well-known/tdmrep.json). Fail closed."""
    v, cause = _robots_verdict(url, hosts, deps, acq, clock, deadline)  # WPL-1111: unreadable is not "disallow"
    if v == "unreadable":
        return "robots_unreadable-" + (cause or "UNKNOWN")
    if v != "allow":
        return "robots"
    if not _tdmrep_allowed(url, hosts, deps, acq, clock, deadline):
        return "tdm"
    return None


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
    if is_denied(doi=doi):  # WPL-1111 R2: an access denial for this DOI is never retried through the landing page
        raise aa._Stop(refusal_code(acq, "access_denied"))
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
    from vera import host_role  # PORTAL-RULE-1110: the text's own final host must be ORIGINAL / RoR (landing hops were checked per hop)
    if not host_role.is_evidence_allowed(got, "store"):
        raise aa._Stop(acq_config.reason(acq, "portal"))
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
