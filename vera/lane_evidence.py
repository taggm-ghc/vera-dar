"""NONASK-1103 S8 + lane span contract N2 (plan 6.7, 6.8, 6.8a; owner R-037, R-038, 2026-11-03 terms rule).

Collects the Official and Current-reporting lanes per sub-query (vera.subquery) and, for the reporting lane, per
temporality (vera.temporality: only parts that ask about the current state). Lane pages are answer-only:
  - they pass an admission step built from the existing admission scoring (fulldoc_admission._relevance/_matched,
    min_relevance, min_matched_stems, max_spans_per_doc) on chunks of the configured corpus chunk size; a page with no
    admitted chunk gives a link only;
  - admitted lane spans enter the evidence for THIS answer only, in their own slots (admission.max_spans_total lane
    spans at most, separate from the research lane), marked answer_only=True, lane, origin ("official" | "reported",
    the E-S1/E-S2 contract), lane_origin (official_lane | reporting_lane) and an attribution (outlet or vendor, title, URL). They are never labelled stored:
    vera.memory_store.answer_only() skips them (C2 contract) and evidence_trace drops their excerpt;
  - lane claims are labelled "official" or "reported", never "verified fact";
  - quotes reuse answer.quote_max_chars (200) through the unchanged citation builder;
  - claim records for lane sources hold ids and spans only (claim_records_view).
Permissions (I-A6 vera.source_permissions, R-049/R-050): a page is read only when its read gate passes (registry
access_verdict not forbid, owner deny-list, robots.txt via public_fetch) and used only when use_for_answer passes;
otherwise it is a reading-list link. Reporting outlet pages: at most admission.max_spans_total reads per request.
Lane pages are never ingested or stored, whatever the ingest gate says (R-037).

Nothing here writes a DB row, a cache entry, a queue item, memory or a log line holding page text. Both lane flags
ship OFF; with both off collect() does no work and returns an empty result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from vera import official_lane, reporting_lane
from vera.official_lane import LanePage

INSPECTION_LABEL = "read in full for this answer; not stored"
ORIGIN = {"official": "official_lane", "reporting": "reporting_lane", "web": "web_lane"}  # web: A6-W (I-A6)
EPISTEMIC = {"official": "official", "reporting": "reported", "web": "web"}
LICENCE_DISPOSITION = "permitted_transient_answer_only"  # S9 / vera-design.md (R-037): licence observation unclear
LANE_PROMPT_SENTENCE = ("Some passages are marked as public pages read for this answer (\"lane\": \"official\" or "
                        "\"reported\", or \"web\" for a web source); they are not verified and not stored. Describe them in your own words, attribute "
                        "them (\"according to <vendor> documentation\", \"as reported by <outlet>\"), and never present "
                        "them as verified fact.")
_ROUTING = Path(__file__).resolve().parent.parent / "config" / "domain_routing.json"


@dataclass(repr=False)
class LaneResult:
    """links: public-safe link dicts (lane, label, title, url, entity/outlet, published, read); pages: LanePage
    (internal, text never in repr); attribution: the GDELT credit when GDELT answered (shown with a link)."""
    links: list = field(default_factory=list)
    pages: list = field(default_factory=list)
    attributions: list = field(default_factory=list)
    # PERF-MISS-1105-FIX2 F2: per-portal call records {portal, status, reason, items} (closed tokens and counts; for
    # the request trace only, never in public()).
    diag: list = field(default_factory=list)

    def __repr__(self) -> str:
        return f"LaneResult(links={len(self.links)}, pages={len(self.pages)})"

    __str__ = __repr__

    def public(self) -> dict:
        """Counts and links only: no page text, no portal name, no registry state, no internal cause."""
        keep = ("lane", "label", "title", "url", "published", "outlet", "entity", "read", "credit")
        return {"links": [{k: lk.get(k) for k in keep if lk.get(k) is not None} for lk in self.links],
                "n_read": len(self.pages), "credits": [dict(a) for a in self.attributions],
                "scope": INSPECTION_LABEL}


def any_enabled() -> bool:
    return official_lane.lane_enabled() or reporting_lane.lane_enabled()


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _outlet_allowed(url: str, cfg: dict) -> bool:
    """Owner-curated outlet list in config/reporting_lane.json `outlets`: [{host, terms_verdict, terms_evidence}].
    Exact host; only "allow" with an evidence pointer permits a fetch (robots is checked by public_fetch)."""
    h = _host(url)
    for o in cfg.get("outlets") or []:
        if isinstance(o, dict) and str(o.get("host") or "").lower() == h:
            return o.get("terms_verdict") == "allow" and bool(o.get("terms_evidence"))
    return False


def _outlet_read_rules(url: str) -> dict:
    """OD-1106-HITL H5: optional per-outlet read limits from config/reporting_lane.json `outlets` (config, never
    hardcoded). path_prefixes / path_regex: the URL path must match (absent = no limit); deny_path_prefixes: never;
    require_any_markers: the fetched text must hold one (positive evidence); wire_markers: any hit means link only.
    Markers are case-insensitive. A declared but empty or malformed rule matches nothing (fail closed); an error
    denies everything."""
    deny_all = {"prefixes": ["\0"], "regex": None, "deny": [], "need": [], "wire": []}
    try:
        row = reporting_lane.outlet_row(url) or {}
        lst = lambda k: [str(x) for x in row.get(k) or [] if isinstance(x, str) and x.strip()]  # noqa: E731
        pp = [x for x in lst("path_prefixes") if x.startswith("/")]
        if "path_prefixes" in row and not pp:
            pp = ["\0"]
        rx = None
        if "path_regex" in row:
            try:
                rx = re.compile(str(row["path_regex"]))
            except (re.error, TypeError):
                return deny_all
        need = [x.lower() for x in lst("require_any_markers")]
        if "require_any_markers" in row and not need:
            return deny_all
        return {"prefixes": pp, "regex": rx, "deny": lst("deny_path_prefixes"), "need": need,
                "wire": [x.lower() for x in lst("wire_markers")]}
    except Exception:  # noqa: BLE001
        return deny_all


def _path_ok(url: str, rules) -> bool:
    if isinstance(rules, list):  # a bare list of prefixes
        rules = {"prefixes": rules, "regex": None, "deny": []}
    if any(ord(c) <= 32 or ord(c) == 127 for c in str(url)):  # urlsplit silently strips \n, \r, \t: deny them
        return False
    try:
        from urllib.parse import urlsplit
        sp = urlsplit(url)
        path = sp.path or "/"
    except ValueError:
        return False
    if any(path.startswith(d) for d in rules.get("deny") or []):
        return False
    if rules.get("prefixes") and not any(path.startswith(p) for p in rules["prefixes"]):
        return False
    return rules.get("regex") is None or (not sp.query and bool(rules["regex"].fullmatch(path)))


def _text_ok(text: str, rules: dict) -> bool:
    norm = lambda x: " ".join(str(x).lower().split())  # noqa: E731  collapse any whitespace run, lowercase
    low = norm(text)
    if any(norm(m) in low for m in rules.get("wire") or []):
        return False
    need = rules.get("need") or []
    return not need or any(norm(m) in low for m in need)


def collect(question: str, *, official=None, reporting=None, fetch=None, plan=None) -> LaneResult:
    """Never raises. Per sub-query: the official lane for its registry entities; the reporting lane only when the part
    is about the current state (temporality). Flags off: no decomposition, no call, an empty result."""
    res = LaneResult()
    try:
        off_on, rep_on = official_lane.lane_enabled(), reporting_lane.lane_enabled()
        if not (off_on or rep_on):
            return res
        from vera import subquery, temporality
        plan = plan or subquery.decompose(question)
        parts = list(plan.parts) or [subquery.SubPart("P1", question)]
        ol = official if official is not None else (official_lane.OfficialLane(fetch=fetch) if off_on else None)
        rl = reporting if reporting is not None else (reporting_lane.ReportingLane() if rep_on else None)
        seen: set = set()
        n_reads = [0]
        robots_cache: dict = {}  # robots.txt rules per host for this request (rules only)
        for p in parts:
            if ol is not None and p.entities:
                try:  # the part's extracted terms only (never the user's sentence), for the site: query
                    from vera.query_terms import extract_terms
                    terms = extract_terms(p.text, p.entities)
                except Exception:  # noqa: BLE001
                    terms = []
                got = ol.run(p.entities, terms)
                for lk in got["links"]:
                    if lk["url"] not in seen:
                        seen.add(lk["url"])
                        res.links.append(dict(lk, part=p.label))
                for pg in got["pages"]:
                    pg.part = p.label
                    res.pages.append(pg)
            # NEWS-LANE-1104 W3: a news / current-events question also reaches the reporting lane (vera.news_intent)
            # OFFICIAL-LANE-S4A G11: a time cue (volatile_entity=False) or news intent only; a volatile registry
            # entity alone (SubPart.temporal) no longer sends a vendor question to the news lane
            from vera import lane_router
            current = lane_router.reporting_cue(p.text, question)
            if rl is None or not current:
                continue
            found = rl.search(p.text, p.entities)
            res.diag.extend(d for d in (found.get("diag") or []) if isinstance(d, dict))
            if found.get("attribution") and found["attribution"] not in res.attributions:
                res.attributions.append(found["attribution"])
            for it in found.get("items") or []:
                url = it.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                try:  # defence in depth (H5): re-check the outlet verdict here, whatever the source of the item
                    verdict = reporting_lane.outlet_verdict(str(url))
                except Exception:  # noqa: BLE001
                    verdict = "links_only"
                if verdict == "exclude":
                    continue  # an excluded outlet is never shown, read or linked
                lk = {"lane": "reporting", "label": "Reported", "title": it.get("title"), "url": url,
                      "published": it.get("published") or "date unknown", "outlet": it.get("outlet"), "read": False,
                      "part": p.label}
                # NEWS-LANE-1104 W2 (SEC S2): an item the lane marks read_ok=False (links_only / unlisted outlet, or an
                # allow outlet inside its crawl delay) is a link only, never fetched.
                # NEWS-LANE-1104 R1: read only an item the lane explicitly tagged read_ok=True (an untagged item, e.g.
                # from a future or injected lane, is a link only: fail closed)
                can_read = it.get("read_ok") is True and verdict == "allow"  # links_only / unknown stay link-only
                rules = _outlet_read_rules(url) if can_read else {}
                perm = (_outlet_gate(url) if can_read and _path_ok(url, rules) else None)
                # vera.source_permissions: read unless restricted; use unless barred
                if perm is not None and perm.read and perm.use_for_answer and n_reads[0] < _read_cap():
                    n_reads[0] += 1
                    try:
                        from vera import web_route
                        page = web_route.gated_fetch(url, fetch=fetch or _public_fetch, robots_cache=robots_cache,
                                                     hop_allowed=_reporting_hop(url, it))  # W6 (SEC S1)
                        final = str(page.get("url") or url)
                        post = web_route.evaluate_fetched(final, page)  # WP6-P2b D3: noai, Content-Signal, derivative
                        if (_host(final) == _host(url) and post.read and post.use_for_answer
                                and (page.get("text") or "").strip()
                                and _text_ok(str(page.get("text")), rules) and _path_ok(final, rules)):
                            pg = LanePage("reporting", "", str(it.get("title") or ""), final, outlet=it.get("outlet") or
                                          _host(final), published=it.get("published"), text=str(page["text"]))
                            pg.part = p.label
                            res.pages.append(pg)
                            lk["read"] = True
                    except Exception:  # noqa: BLE001  refused (robots, status, type, size, host): link only
                        pass
                res.links.append(lk)
    except Exception:  # noqa: BLE001  a lane never breaks the answer path
        return LaneResult(links=list(res.links), diag=list(res.diag) + [{"portal": "-", "status": "called",
                                                                       "reason": "error", "items": 0}])
    return res


WIKI_CREDIT = "via Wikipedia citations"


def _wiki_host(url: str) -> bool:
    """True for a Wikipedia or Wikimedia-family host (config `wiki_link_drop_hosts`, exact or subdomain). A missing or
    malformed list drops every URL (fail closed); an unparsable host is dropped."""
    try:
        h = reporting_lane._domain(url)
        hosts = reporting_lane.load_config().get("wiki_link_drop_hosts")
        if not h or not isinstance(hosts, list) or not hosts:
            return True
        return any(h == str(x).lower() or h.endswith("." + str(x).lower()) for x in hosts)
    except Exception:  # noqa: BLE001
        return True


_ARCHIVE = re.compile(r"^https?://(?:www\.|web\.|wayback\.)?archive\.org(?::443)?/web/[0-9*]+(?:[a-z_]{2,3})?/(.+)$", re.I)
# snapshot mirrors whose inner URL cannot be recovered from the link: fail closed (deny all, allow exceptions)
_ARCHIVE_MIRRORS = ("archive.ph", "archive.today", "archive.is", "archive.li", "archive.vn", "archive.md", "archive.fo",
                    "archive.org", "webcache.googleusercontent.com")
_BARE_HOST = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?::\d+)?(?:[/?#]|$)", re.I)


def _is_archive_host(url: str) -> bool:
    try:
        h = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return True
    return any(h == m or h.endswith("." + m) for m in _ARCHIVE_MIRRORS)


def _unwrap_archive(url: str):
    """LINKS-RC-1107 r2: the page a Wayback snapshot URL archives (schemeless, percent-encoded, wildcard-timestamp, :443,
    archive.org/web/ and wayback.archive.org forms), repeated up to 3 times. Not an archive URL: returned unchanged.
    An archive/mirror URL whose inner http(s) URL cannot be recovered: None (the caller drops the link)."""
    from urllib.parse import unquote
    cur = url
    for _ in range(3):
        if not _is_archive_host(cur):
            return cur
        m = _ARCHIVE.match(cur)
        if not m:
            return None
        inner = m.group(1)
        if "://" not in inner and "%" in inner:
            inner = unquote(inner)
        if not inner.lower().startswith(("http://", "https://")):
            if not _BARE_HOST.match(inner):
                return None
            inner = "https://" + inner
        try:
            if not urlsplit(inner).hostname:
                return None
        except ValueError:
            return None
        cur = inner
    return None if _is_archive_host(cur) else cur


def _link_key(url: str) -> str:
    u = re.sub(r"^https?://(?:www\.)?", "", (_unwrap_archive(url) or url).strip().lower()).split("#", 1)[0]
    return u.rstrip("/")


def wiki_link_items(urls, seen=()) -> list:
    """NT-3: Wikipedia-citation URLs as LINK-ONLY news items (read False, never fetched, never evidence, never counted
    toward the full-text target). Outlet from the host; an outlet the owner list marks "exclude" is dropped; the
    credit is the only mention of the source. Pure: no network, no DB. Never raises."""
    out, have = [], {_link_key(str(s)) for s in seen if s}
    try:
        for u in urls or []:
            u = str(u or "")
            if not u.startswith(("http://", "https://")) or _link_key(u) in have:
                continue
            real = _unwrap_archive(u)  # outlet checks use the archived page's host, not web.archive.org
            from vera.search_providers import safe_target_url
            # unrecoverable archive/mirror URL, or an unwrapped inner URL that is not a safe public https target
            # (userinfo/backslash tricks, IP literals, internal hosts): dropped (fail closed)
            if real is None or (real != u and not safe_target_url(real)):
                continue
            if reporting_lane.outlet_verdict(real) == "exclude" or _wiki_host(real):
                continue
            have.add(_link_key(u))
            host = reporting_lane._domain(real)
            out.append({"lane": "reporting", "label": "Reported", "title": host, "url": u, "published": "date unknown",
                        "outlet": host, "read": False, "credit": WIKI_CREDIT})
    except Exception:  # noqa: BLE001
        return out
    return out


def merge_wiki_links(lanes: "LaneResult | None", question) -> "LaneResult | None":
    """NT-3: add the Wikipedia portal's links (wiki_ref_portal.take_links) to the lane result as link-only items.
    Returns lanes unchanged when there are none. Never raises."""
    try:
        from vera import wiki_ref_portal
        urls = wiki_ref_portal.take_links(str(question or ""))
        base = lanes if lanes is not None else LaneResult()
        items = wiki_link_items(urls, seen={lk.get("url") for lk in base.links})
        if not items:
            return lanes
        return LaneResult(links=list(base.links) + items, pages=list(base.pages),
                          attributions=list(base.attributions), diag=list(base.diag))
    except Exception:  # noqa: BLE001
        return lanes


def _news(text: str) -> bool:
    try:
        from vera import news_intent
        return news_intent.news_intent(text)
    except Exception:  # noqa: BLE001
        return False


def _reporting_hop(url: str, item: dict):
    """NEWS-LANE-1104 W6 (SEC S1): hop predicate for a reporting-lane page. Every hop (redirect target, robots.txt)
    must stay on the start host AND be an "allow" outlet (NEWS-LANE-1104 R1: always, tagged or not; a predicate error
    refuses). Checked by web_route.gated_fetch BEFORE any request to the hop."""
    start = _host(url)

    def ok(hop: str) -> bool:
        try:
            return _host(hop) == start and reporting_lane.outlet_verdict(hop) == "allow"
        except Exception:  # noqa: BLE001
            return False
    return ok


def _outlet_gate(url: str):
    """The shared permission record for an outlet page (None on error = link only). Lane pages are never ingested."""
    try:
        from vera import source_permissions
        return source_permissions.read_gate(url)
    except Exception:  # noqa: BLE001
        return None


def _read_cap() -> int:
    """Outlet pages read per request = admission.max_spans_total (the lane's own span slots; no new number)."""
    try:
        from vera import fulldoc_corpus
        return int(((fulldoc_corpus.load_config().get("admission") or {}).get("max_spans_total")) or 6)
    except Exception:  # noqa: BLE001
        return 6


def _public_fetch(url: str, tdm_check: bool = False, **kw) -> dict:
    """vera.public_fetch.fetch with the per-hop gate kwargs (dropped only for a stub that cannot take them).
    H19 A9 (review R1): `tdm_check` (set by web_route.gated_fetch, default True for outlets with an ND access block)
    is forwarded so /.well-known/tdmrep.json is read. Fail closed: when it is requested and the fetcher cannot take it,
    the read is refused (never silently skipped)."""
    from vera import public_fetch, web_route
    f = public_fetch.fetch
    if tdm_check:
        if not web_route._takes_named(f, "tdm_check"):
            raise public_fetch.FetchRefused("tdm_unchecked")
        kw["tdm_check"] = True
    return f(url, **kw) if web_route._takes_kwargs(f) else f(url)


def _chunks(text: str, size: int, min_size: int):
    """(start, end, text) windows of about `size` chars cut at whitespace (the stored-corpus chunk size)."""
    n, i = len(text), 0
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.rfind(" ", i + min_size, j)
            j = k if k > i else j
        seg = text[i:j].strip()
        if len(seg) >= min_size:
            s = i + (len(text[i:j]) - len(text[i:j].lstrip()))
            yield s, s + len(seg), seg
        i = j if j > i else n


def _chunk_cfg() -> tuple[int, int]:
    try:
        c = json.loads(_ROUTING.read_text(encoding="utf-8"))
        return int(c.get("chunk_chars", 400)), int(c.get("min_chunk_chars", 40))
    except (OSError, ValueError, TypeError):
        return 400, 40


def _page_licence_id(page) -> str | None:
    """GOV-PD-1108: the licence id a web page carries (us-federal-public-domain, mode on), else None. Provenance only:
    the page stays answer-only and is never stored (licence_decision unchanged)."""
    lic = getattr(page, "licence", None)
    return str(lic.get("id")) if isinstance(lic, dict) and lic.get("id") else None


def attribution_for(page: LanePage) -> dict:
    who = (f"{page.entity} documentation" if page.lane == "official"
           else f"Web source: {_host(page.url)}" if page.lane == "web"  # A6-W: a web page, never verified fact
           else f"As reported by {page.outlet or _host(page.url)}")
    att = {"creator": who, "title": page.title or _host(page.url), "source": page.url,
           "lane": EPISTEMIC[page.lane], "scope": INSPECTION_LABEL}
    try:  # H5: an outlet row may carry its licence statement (Wikinews: CC BY 4.0 after 2024-12-16, CC BY 2.5 before)
        note = (reporting_lane.outlet_row(page.url) or {}).get("licence_note") if page.lane == "reporting" else None
        if isinstance(note, str) and note.strip():
            att["licence"] = note
    except Exception:  # noqa: BLE001
        pass
    return att


def _nd_page(pg) -> bool:
    """CONTROL-GAPS S45: a page of an outlet whose row carries an ND `access` block is an ND source for vera.nd_guard
    (the existing output guard, run by fulldoc_answer). Fail closed: a lookup error counts as ND."""
    try:
        return isinstance((reporting_lane.outlet_row(pg.url) or {}).get("access"), dict)
    except Exception:  # noqa: BLE001
        return True


def admit_pages(question: str, pages: list, cfg: dict) -> tuple[list, list]:
    """(admitted span dicts, examined document dicts) for the lane pages, using the existing admission scoring and
    thresholds. Question = the page's sub-query text is not available here, so the original question is scored."""
    from vera import fulldoc_admission as fa
    adm = (cfg.get("admission") or {}) if isinstance(cfg, dict) else {}
    min_rel = float(adm.get("min_relevance", 0.4))
    min_hits = max(int(adm.get("min_matched_stems", 2)), 2)  # absolute floor, as admit()
    max_doc = int(adm.get("max_spans_per_doc", 2))
    max_total = int(adm.get("max_spans_total", 6))  # the lane's own slots (separate from the research lane)
    routing = fa._routing(cfg)
    icfg = routing.get("identification") or {}
    try:
        stem_n = int((routing.get("corpus") or {})["relevance_stem_len"])
    except (KeyError, TypeError, ValueError):
        return [], []
    size, min_size = _chunk_cfg()
    admitted, examined = [], []
    for i, pg in enumerate(pages, 1):
        did = f"lane-{pg.lane}-{i}"
        att = attribution_for(pg)
        nd = _nd_page(pg)
        cands = []
        for s, e, seg in _chunks(pg.text, size, min_size):
            rel = fa._relevance(question, seg, icfg, stem_n)
            hits, _q = fa._matched(question, seg, icfg, stem_n)
            if rel >= min_rel and hits >= min_hits:
                cands.append((rel, s, e, seg))
        cands.sort(key=lambda c: -c[0])
        took = 0
        for rel, s, e, seg in cands[:max_doc]:
            if len(admitted) >= max_total:
                break
            took += 1
            admitted.append({"span_id": f"{did}:{s}", "document_id": did, "canonical_source_id": pg.url,
                             "title": pg.title, "url": pg.url, "published": pg.published, "authors": None,
                             "licence": _page_licence_id(pg), "licence_url": None, "start": s, "end": e, "text": seg,
                             "text_sha256": fa._sha(seg), "relevance": round(rel, 4), "attribution": att,
                             "answer_only": True, "lane": pg.lane, "origin": EPISTEMIC[pg.lane],
                             "lane_origin": ORIGIN[pg.lane], "epistemic": EPISTEMIC[pg.lane], "stored": False,
                             **({"licence_class": "nd", "canonical_text": pg.text} if nd else {})})
        examined.append({"document_id": did, "canonical_source_id": pg.url, "title": pg.title, "url": pg.url,
                         "doi": None, "licence": _page_licence_id(pg), "licence_decision": LICENCE_DISPOSITION,
                         "text_sha256": None, "chars": len(pg.text), "admitted_spans": took, "answer_only": True,
                         "lane": pg.lane, "origin": EPISTEMIC[pg.lane], "lane_origin": ORIGIN[pg.lane],
                         "stored": False})
    return admitted, examined


def merge(ev, lanes: LaneResult | None, question: str, cfg: dict):
    """The evidence bundle plus the admitted lane spans (answer only). Pages without an admitted span are links only
    (not examined documents). Returns ev unchanged when there is nothing to add; never raises."""
    if lanes is None or not lanes.pages:
        return ev
    try:
        from vera.fulldoc_evidence import EvidenceBundle
        adm, exam = admit_pages(question, lanes.pages, cfg)
        exam = [d for d in exam if d["admitted_spans"]]
        if not adm:
            return ev
        if ev is None or getattr(ev, "status", None) == "corpus_unusable" and not getattr(ev, "admitted", None):
            return EvidenceBundle(question=question, examined_full_documents=tuple(exam), admitted=tuple(adm),
                                  status="admitted", trace={"lane_admitted": len(adm)})
        tr = dict(getattr(ev, "trace", {}) or {}, lane_admitted=len(adm))
        return replace(ev, examined_full_documents=tuple(ev.examined_full_documents) + tuple(exam),
                       admitted=tuple(ev.admitted) + tuple(adm), status="admitted", trace=tr)
    except Exception:  # noqa: BLE001
        return ev


def is_lane(src) -> bool:
    return isinstance(src, dict) and bool(src.get("lane")) and src.get("answer_only") is True


def epistemic_label(src) -> str | None:
    """"official" | "reported" for lane sources; None otherwise. Never "verified fact", whatever the page says."""
    return EPISTEMIC.get(src.get("lane")) if is_lane(src) else None


def lane_prompt(system: str, cfg: dict, per_part: bool = False) -> str:
    """S8: the writer prompt variant used when any passage is a lane page. The ':311' sentence ("acquired and
    verified") is replaced by the configured lane wording (answer.lane_prompt or answer.per_part.lane_prompt);
    if the sentence is not found the lane sentence is appended (fail safe: the model is always told)."""
    a = (cfg.get("answer") or {}) if isinstance(cfg, dict) else {}
    lp = ((a.get("per_part") or {}).get("lane_prompt") if per_part else None) or a.get("lane_prompt") or {}
    old = lp.get("replace") or ""
    new = lp.get("with") or LANE_PROMPT_SENTENCE
    if old and old in system:
        return system.replace(old, new)
    return system + "\n" + new


def claim_records_view(records, sources) -> list:
    """Claim records for persistence or trace: lane-cited records keep ids and spans only (never quote or sentence)."""
    lane_n = {s["n"] for s in sources or () if is_lane(s)}
    by_n = {s["n"]: s for s in sources or ()}
    out = []
    for r in records or ():
        cited = [n for n in r.get("citations") or () if n in lane_n]
        if not cited:
            out.append(r)
            continue
        out.append({"id": r.get("id"), "verdict": r.get("verdict"), "citations": list(r.get("citations") or ()),
                    "spans": [{"n": n, "span_id": by_n[n].get("span_id"), "span": by_n[n].get("span")} for n in cited],
                    "epistemic": sorted({epistemic_label(by_n[n]) for n in cited})})
    return out


def subquery_adequacy(question: str, kept, sources, resp_partial: bool = False, plan=None) -> list | None:
    """E-S2 wiring: [{label, declared, supported, resp_partial}] per vera.subquery part, for adequacy_gate.evaluate.
    A part is supported when a kept supported/partial claim cites a passage whose title, URL or text names one of the
    part's entities (or, for a part without entities, all of its content words). No new threshold. None for < 2 parts
    or on any error (the gate then behaves exactly as before)."""
    try:
        if plan is None:
            from vera import subquery
            plan = subquery.decompose(question)
        parts = list(plan.parts)
        if len(parts) < 2:
            return None
        by_n = {s.get("n"): s for s in sources or ()}
        cited = [by_n[n] for r in kept or () if r.get("verdict") in ("supported", "partial")
                 for n in r.get("citations") or () if n in by_n]
        hay = [" ".join(str(s.get(k) or "") for k in ("title", "url", "snippet")).casefold() for s in cited]
        out = []
        for p in parts:
            names = [str(e).casefold() for e in p.entities] or None
            if names:
                ok = any(any(nm in h for nm in names) for h in hay)
            else:
                words = [w for w in p.text.casefold().split() if len(w) > 3]
                ok = bool(words) and any(all(w in h for w in words) for h in hay)
            out.append({"label": p.label, "declared": [p.label], "supported": [p.label] if ok else [],
                        "resp_partial": bool(resp_partial)})
        return out
    except Exception:  # noqa: BLE001
        return None
