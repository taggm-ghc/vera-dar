"""MIDTERMS-WP10-1106 (owner OD-1106-MIDTERMS-D1D4, D3): the examined-source ledger of one request's acquisition.

One entry per distinct source this request's acquisition attempt met, with its kind (scholarly, news, official, web),
the depth at which its text was examined (full_text, partial_text, metadata_only) and whether it qualified (passed the
existing admission, thresholds unchanged). Only qualifying full- or partial-text entries count toward the six-source
target (vera.acq_rounds.target_of); duplicates, metadata-only items (discovered records, reading-list links, candidates
rejected before any body was examined) and examined-but-unqualified sources never count.

Depth: every current producer examines whole texts (fulltext_loader refuses abstract-only bodies; public_fetch
refuses oversize pages, never truncates), so examined entries are full_text. partial_text is recorded only when a
producer reports it explicitly (an examined document dict carrying text_depth="partial_text"); none does today.

Pure bookkeeping: no network, no DB, no logging. Keys, URLs, DOIs and titles never leave the ledger: summary() is
counts only (public contract and request trace).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

KINDS = ("scholarly", "news", "official", "web")
DEPTHS = ("full_text", "partial_text", "metadata_only")
_RANK = {"metadata_only": 0, "partial_text": 1, "full_text": 2}
SUMMARY_KEYS = ("counted", "full_text", "partial_text", "metadata_only", "examined_unqualified", "duplicates",
                "stored") + KINDS
_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#\"'<>]+)", re.I)
_DROP_PARAMS = ("fbclid", "gclid")


@dataclass(frozen=True)
class Entry:
    key: str          # primary dedup key; never shown or traced
    kind: str         # KINDS
    depth: str        # DEPTHS
    qualified: bool   # passed the existing admission
    stored: bool      # persisted to the collection in this request
    origin: str       # inline | reused | lane | web | candidate | discovered | link


def norm_url(url) -> str:
    """Lower-case scheme and host, http == https, no fragment, no utm_*/fbclid/gclid parameters, no trailing "/"."""
    u = str(url or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
    except ValueError:
        return u.lower().rstrip("/")
    scheme = (p.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith("utm_") and k.lower() not in _DROP_PARAMS]
    path = p.path.rstrip("/")
    return urlunsplit((scheme, (p.netloc or "").lower(), path, urlencode(q), ""))


def keys_of(ids: dict) -> set:
    """Every identity key of one source (DOI, PMCID, OpenAlex, arXiv, normalised URL); a doi.org URL yields its DOI."""
    ids = ids if isinstance(ids, dict) else {}
    out: set = set()
    doi = str(ids.get("doi") or "").strip().lower()
    urls = [str(ids.get(k) or "").strip() for k in ("url", "canonical_url", "canonical_source_id")]
    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]
    for u in urls:
        if "doi.org/" in u.lower():
            m = _DOI.search(u)
            if m and not doi:
                doi = m.group(1).lower()
    if doi:
        out.add("doi:" + doi.removeprefix("https://doi.org/").removeprefix("doi:").rstrip("."))
    if ids.get("pmcid"):
        out.add("pmcid:" + str(ids["pmcid"]).upper().removeprefix("PMC"))
    if ids.get("openalex_id"):
        out.add("openalex:" + str(ids["openalex_id"]).rsplit("/", 1)[-1].upper())
    if ids.get("arxiv_id"):
        out.add("arxiv:" + re.sub(r"v\d+$", "", str(ids["arxiv_id"]).lower()))
    for u in urls:
        n = norm_url(u)
        if n and "doi.org/" not in n:
            out.add("url:" + n)
    return out


def _host(url) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""


def kind_of(url, lane, *, outlet_hosts=(), official_hosts=()) -> str:
    """news / official / web for a lane page; scholarly for a pipeline document (lane None). A general-web page on an
    owner-listed outlet host is news, on an official-registry host official, else web. No new host lists."""
    if not lane:
        return "scholarly"
    if lane == "reporting":
        return "news"
    if lane == "official":
        return "official"
    h = _host(url)
    if h and h in set(outlet_hosts):
        return "news"
    if h and h in set(official_hosts):
        return "official"
    return "web"


def count_kinds(acq: dict | None) -> tuple:
    """bounds.acq_round_count_kinds (default all four kinds; web kept because the target already counted web pages)."""
    v = ((acq or {}).get("bounds") or {}).get("acq_round_count_kinds") if isinstance(acq, dict) else None
    if isinstance(v, (list, tuple)):
        ks = tuple(k for k in v if k in KINDS)
        if ks:
            return ks
    return KINDS


class Ledger:
    def __init__(self, kinds=KINDS):
        self._rows: list = []       # [{"keys": set, kind, depth, qualified, stored, origin}]
        self._index: dict = {}      # key -> row index
        self.duplicates = 0
        self.kinds = tuple(kinds) or KINDS

    def _find(self, keys) -> int | None:
        for k in keys:
            if k in self._index:
                return self._index[k]
        return None

    def has(self, ids: dict) -> bool:
        return self._find(keys_of(ids)) is not None

    def add(self, *, ids: dict, kind: str, depth: str, qualified: bool, stored: bool = False, origin: str) -> bool:
        """False = a duplicate of an earlier entry (any shared key). The first entry stays; a later one with a deeper
        text or that qualified upgrades it, never counted twice. All keys of one record are linked (a DOI and a URL on
        the same record make a later DOI-only or URL-only sighting a duplicate). Residual: a source first seen by DOI
        alone and later by a URL alone (no record carrying both) is not recognised as the same source. A source with no
        identity key is always a new row and is never counted (it cannot be deduplicated)."""
        kind = kind if kind in KINDS else "web"
        depth = depth if depth in DEPTHS else "metadata_only"
        keys = keys_of(ids)
        i = self._find(keys)
        if i is None:
            row = {"keys": set(keys), "kind": kind, "depth": depth, "qualified": bool(qualified), "stored": bool(stored),
                   "origin": str(origin)}
            self._rows.append(row)
            for k in keys:
                self._index[k] = len(self._rows) - 1
            return True
        self.duplicates += 1
        row = self._rows[i]
        if (_RANK[depth] > _RANK[row["depth"]]) or (qualified and not row["qualified"]):
            row["kind"], row["origin"] = kind, str(origin)
        row["depth"] = depth if _RANK[depth] > _RANK[row["depth"]] else row["depth"]
        row["qualified"] = row["qualified"] or bool(qualified)
        row["stored"] = row["stored"] or bool(stored)
        row["keys"] |= keys
        for k in keys:
            self._index.setdefault(k, i)
        return False

    def merge(self, other: "Ledger | None") -> "Ledger":
        """A new ledger: this one's entries, then the other's, with the same dedup (across acquisition rounds)."""
        out = Ledger(self.kinds)
        for src in (self, other):
            if src is None:
                continue
            for r in src._rows:
                ids = _ids_from_keys(r["keys"])
                out.add(ids=ids, kind=r["kind"], depth=r["depth"], qualified=r["qualified"], stored=r["stored"],
                        origin=r["origin"]) if ids else out._add_keyless(r)
            out.duplicates += src.duplicates
        return out

    def _add_keyless(self, r: dict) -> None:
        self._rows.append({**r, "keys": set()})

    def entries(self) -> list:
        return [Entry(key=sorted(r["keys"])[0] if r["keys"] else "", kind=r["kind"], depth=r["depth"],
                      qualified=r["qualified"], stored=r["stored"], origin=r["origin"]) for r in self._rows]

    def _counts(self, r) -> bool:
        # review F3: a row with no identity key can never be deduplicated, so it never counts toward the target
        return bool(r["keys"]) and r["depth"] in ("full_text", "partial_text") and r["qualified"] and r["kind"] in self.kinds

    def counted(self) -> int:
        return sum(1 for r in self._rows if self._counts(r))

    def summary(self) -> dict:
        """Counts only. full_text / partial_text and the per-kind counts are of the COUNTED entries; metadata_only and
        examined_unqualified are of all entries; stored = entries persisted to the collection in this request."""
        c = [r for r in self._rows if self._counts(r)]
        out = {"counted": len(c),
               "full_text": sum(1 for r in c if r["depth"] == "full_text"),
               "partial_text": sum(1 for r in c if r["depth"] == "partial_text"),
               "metadata_only": sum(1 for r in self._rows if r["depth"] == "metadata_only"),
               "examined_unqualified": sum(1 for r in self._rows
                                           if r["depth"] != "metadata_only" and not r["qualified"]),
               "duplicates": int(self.duplicates),
               "stored": sum(1 for r in self._rows if r["stored"])}
        for k in KINDS:
            out[k] = sum(1 for r in c if r["kind"] == k)
        return out


def _ids_from_keys(keys) -> dict:
    ids: dict = {}
    for k in keys:
        p, _, v = k.partition(":")
        if p == "doi":
            ids["doi"] = v
        elif p == "pmcid":
            ids["pmcid"] = v
        elif p == "openalex":
            ids["openalex_id"] = v
        elif p == "arxiv":
            ids["arxiv_id"] = v
        elif p == "url" and "url" not in ids:
            ids["url"] = v
    return ids


def valid_summary(s) -> bool:
    return isinstance(s, dict) and all(isinstance(s.get(k), int) and not isinstance(s.get(k), bool) and s[k] >= 0
                                       for k in SUMMARY_KEYS)


def _cand_ids(r: dict) -> dict:
    return {k: r.get(k) for k in ("doi", "pmcid", "openalex_id", "arxiv_id", "url") if r.get(k)}


def _host_lists():
    """(outlet hosts, official-registry hosts) from the existing lists; empty on any failure (pages are then web)."""
    outlets, official = set(), set()
    try:
        from vera import reporting_lane
        for o in (reporting_lane.load_config().get("outlets") or []):
            if isinstance(o, dict) and o.get("terms_verdict") == "allow" and o.get("terms_evidence") and o.get("host"):
                outlets.add(str(o["host"]).lower())
    except Exception:  # noqa: BLE001
        pass
    try:
        from vera import official_lane
        for row in official_lane._rows(None).values():
            official |= {h.lower() for h in official_lane.row_hosts(row)}
    except Exception:  # noqa: BLE001
        pass
    return outlets, official


def from_inline(o, *, ev=None, lanes=None, acq=None, host_lists=None) -> Ledger:
    """The ledger of one inline acquisition attempt (vera.inline_acquire.Outcome `o`), from what was examined:
    `ev` = the answer-time evidence bundle (ephemeral corpus admission merged with the lane pages), `lanes` = the
    request's lane/web result, then the per-candidate records and discovered records (metadata or unqualified)."""
    from vera import acq_config
    led = Ledger(count_kinds(acq))
    recs = [r for r in (getattr(o, "candidates", None) or []) if isinstance(r, dict)]
    stored_keys: set = set()
    reused_keys: set = set()
    for r in recs:
        if r.get("persisted") is True:
            stored_keys |= keys_of(_cand_ids(r))
        if r.get("outcome") == "reused":
            reused_keys |= keys_of(_cand_ids(r))
    hl = host_lists
    pages = list(getattr(lanes, "pages", None) or []) if lanes is not None else []
    if hl is None and (pages or any(isinstance(d, dict) and d.get("lane") == "web"
                                    for d in (getattr(ev, "examined_full_documents", None) or ()))):
        hl = _host_lists()
    outlets, official = hl if hl is not None else ((), ())
    for d in (getattr(ev, "examined_full_documents", None) or ()) if ev is not None else ():
        if not isinstance(d, dict):
            continue
        lane = d.get("lane")
        ids = {"doi": d.get("doi"), "url": d.get("url"), "canonical_source_id": d.get("canonical_source_id")}
        ks = keys_of(ids)
        depth = d.get("text_depth") if d.get("text_depth") in ("full_text", "partial_text") else "full_text"
        led.add(ids=ids, kind=kind_of(d.get("url"), lane, outlet_hosts=outlets, official_hosts=official),
                depth=depth, qualified=int(d.get("admitted_spans") or 0) > 0,
                stored=not lane and bool(ks & stored_keys),
                origin=("web" if lane == "web" else "lane") if lane else ("reused" if ks & reused_keys else "inline"))
    for p in pages:  # read in full but no admitted span (merge keeps only admitted pages as examined documents)
        ids = {"url": getattr(p, "url", None)}
        if led.has(ids):
            continue
        lane = getattr(p, "lane", None) or "web"
        led.add(ids=ids, kind=kind_of(ids["url"], lane, outlet_hosts=outlets, official_hosts=official),
                depth="full_text", qualified=False, origin="web" if lane == "web" else "lane")
    for lk in (getattr(lanes, "links", None) or []) if lanes is not None else []:
        if isinstance(lk, dict) and lk.get("read") is not True and lk.get("url"):
            led.add(ids={"url": lk.get("url")},
                    kind=kind_of(lk.get("url"), lk.get("lane") or "web", outlet_hosts=outlets, official_hosts=official),
                    depth="metadata_only", qualified=False, origin="link")
    post_fetch = {acq_config.reason(acq or {}, k) for k in ("irrelevant", "few_spans")}
    examined_ev = ev is not None and bool(getattr(ev, "examined_full_documents", None))
    for r in recs:
        oc = r.get("outcome")
        ids = _cand_ids(r)
        if oc in ("admitted", "reused"):
            if examined_ev:
                continue  # examined above, with its answer-time admission
            led.add(ids=ids, kind="scholarly", depth="full_text", qualified=True, stored=r.get("persisted") is True,
                    origin="reused" if oc == "reused" else "inline")
            continue
        if oc == "web_answer_only" and pages:
            continue  # the page itself is in the ledger
        body = (oc == "rejected" and r.get("reason") in post_fetch) or oc in ("reuse_rejected", "not_used")
        led.add(ids=ids, kind="scholarly", depth="full_text" if body else "metadata_only", qualified=False,
                origin="candidate")
    for dsc in (getattr(o, "discovered", None) or []):
        if isinstance(dsc, dict):
            ids = {"url": dsc.get("url")}
            if keys_of(ids):
                if not led.has(ids):
                    led.add(ids=ids, kind="scholarly", depth="metadata_only", qualified=False, origin="discovered")
            else:
                led._add_keyless({"kind": "scholarly", "depth": "metadata_only", "qualified": False, "stored": False,
                                  "origin": "discovered"})
    return led
