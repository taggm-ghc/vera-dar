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
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from vera import official_lane, reporting_lane
from vera.official_lane import LanePage

INSPECTION_LABEL = "read in full for this answer; not stored"
ORIGIN = {"official": "official_lane", "reporting": "reporting_lane"}
EPISTEMIC = {"official": "official", "reporting": "reported"}
LICENCE_DISPOSITION = "permitted_transient_answer_only"  # S9 / vera-design.md (R-037): licence observation unclear
LANE_PROMPT_SENTENCE = ("Some passages are marked as public pages read for this answer (\"lane\": \"official\" or "
                        "\"reported\"); they are not verified and not stored. Describe them in your own words, attribute "
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

    def __repr__(self) -> str:
        return f"LaneResult(links={len(self.links)}, pages={len(self.pages)})"

    __str__ = __repr__

    def public(self) -> dict:
        """Counts and links only: no page text, no portal name, no registry state, no internal cause."""
        keep = ("lane", "label", "title", "url", "published", "outlet", "entity", "read")
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
            current = p.temporal or temporality.detect(p.text).current
            if rl is None or not current:
                continue
            found = rl.search(p.text, p.entities)
            if found.get("attribution") and found["attribution"] not in res.attributions:
                res.attributions.append(found["attribution"])
            for it in found.get("items") or []:
                url = it.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                lk = {"lane": "reporting", "label": "Reported", "title": it.get("title"), "url": url,
                      "published": it.get("published") or "date unknown", "outlet": it.get("outlet"), "read": False,
                      "part": p.label}
                perm = _outlet_gate(url)  # vera.source_permissions: read unless restricted; use unless barred
                if perm is not None and perm.read and perm.use_for_answer and n_reads[0] < _read_cap():
                    n_reads[0] += 1
                    try:
                        page = (fetch or _public_fetch)(url)
                        final = str(page.get("url") or url)
                        if _host(final) == _host(url) and (page.get("text") or "").strip():
                            pg = LanePage("reporting", "", str(it.get("title") or ""), final, outlet=it.get("outlet") or
                                          _host(final), published=it.get("published"), text=str(page["text"]))
                            pg.part = p.label
                            res.pages.append(pg)
                            lk["read"] = True
                    except Exception:  # noqa: BLE001  refused (robots, status, type, size, host): link only
                        pass
                res.links.append(lk)
    except Exception:  # noqa: BLE001  a lane never breaks the answer path
        return LaneResult(links=list(res.links))
    return res


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


def _public_fetch(url: str) -> dict:
    from vera import public_fetch
    return public_fetch.fetch(url)


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


def attribution_for(page: LanePage) -> dict:
    who = f"{page.entity} documentation" if page.lane == "official" else f"As reported by {page.outlet or _host(page.url)}"
    return {"creator": who, "title": page.title or _host(page.url), "source": page.url,
            "lane": EPISTEMIC[page.lane], "scope": INSPECTION_LABEL}


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
                             "licence": None, "licence_url": None, "start": s, "end": e, "text": seg,
                             "text_sha256": fa._sha(seg), "relevance": round(rel, 4), "attribution": att,
                             "answer_only": True, "lane": pg.lane, "origin": EPISTEMIC[pg.lane],
                             "lane_origin": ORIGIN[pg.lane], "epistemic": EPISTEMIC[pg.lane], "stored": False})
        examined.append({"document_id": did, "canonical_source_id": pg.url, "title": pg.title, "url": pg.url,
                         "doi": None, "licence": None, "licence_decision": LICENCE_DISPOSITION,
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
