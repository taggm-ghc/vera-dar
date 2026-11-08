"""OA-3a (OA37-1108, owner OD-1108-HITL D3): one canonical key per scholarly work.

Today several dedup keys disagree (search_providers.merge_sources, search_and_fetch._Deduper, auto_acquire.cand_keys,
acq_discovered._record_id), so one work can take several evidence slots and the losing copy's provenance is lost.
This module is the shared key; OA-3b switches those callers to it, OA-3c records each discovery event.

Priority: normalised DOI > PMCID > PMID > arXiv id (version dropped) > normalised URL. An arXiv DataCite DOI
(10.48550/arXiv.<id>) is the arXiv id, not a journal DOI, so a preprint and its published version keep distinct keys
and are linked through `version_link` (`related_versions`), never merged into one key here.

Pure functions: no request, no DB, no config. A record is any dict with some of: doi, pmcid, pmid, arxiv ({id,
version} or str), identifier, url, discovered_url, oa.pmcid. Identifiers in the URL are read without a request
(search_providers.ddg_identifier); a URL never invents a DOI it does not contain.

`KeyIndex` groups records that share ANY identifier (a record carrying both a DOI and a PMCID links a DOI-only and a
PMCID-only copy of the same work); each group's canonical key is the highest-priority key any member carries.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "work_key.json"
MODES = ("off", "shadow", "on")
PRIORITY = ("doi", "pmcid", "pmid", "arxiv", "url")
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)
_ARXIV_ID = re.compile(r"^(?:arxiv:)?\s*((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[a-z]{2})?/\d{7}))(v\d+)?$", re.I)
_PMID = re.compile(r"^\d{1,12}$")


def _arxiv_bare(s) -> str | None:
    m = _ARXIV_ID.match(str(s or "").strip())
    return m.group(1).lower() if m else None


def identifiers(rec) -> dict:
    """{kind: value} for every identifier the record carries (kinds in PRIORITY), metadata only."""
    from vera import search_providers as sp
    from vera import url_key
    out: dict = {}
    if not isinstance(rec, dict):
        return out
    oa = rec.get("oa") if isinstance(rec.get("oa"), dict) else {}

    def put(kind, value):
        if value and kind not in out:
            out[kind] = value

    doi = sp.normalise_doi(rec.get("doi"))
    if doi:
        m = _ARXIV_DOI.match(doi)
        if m:
            put("arxiv", _arxiv_bare(m.group(1)))
        else:
            put("doi", doi)
    put("pmcid", sp._pmcid_of(rec.get("pmcid") or oa.get("pmcid")))
    pmid = str(rec.get("pmid") or "").strip()
    put("pmid", str(int(pmid)) if _PMID.match(pmid) and int(pmid) else None)  # "0123" == "123"
    ax = rec.get("arxiv")
    put("arxiv", _arxiv_bare(ax.get("id") if isinstance(ax, dict) else ax))
    put("arxiv", _arxiv_bare(rec.get("identifier")))
    for u in (rec.get("url"), rec.get("discovered_url")):
        if not isinstance(u, str):
            continue
        ident = sp.ddg_identifier(u)  # a DOI only when the URL path literally carries one (same rule as follow_url)
        if not ident or ident["kind"] == "handle":
            continue
        kind, value = ident["kind"], ident["value"]
        if kind == "doi":
            m = _ARXIV_DOI.match(value)
            kind, value = ("arxiv", _arxiv_bare(m.group(1))) if m else ("doi", value)
        elif kind == "arxiv":
            value = _arxiv_bare(value)
        put(kind, value)
    url = rec.get("url").strip() if isinstance(rec.get("url"), str) else ""
    if url:
        put("url", url_key.normalize_url(url) or None)
    return out


def keys(rec) -> list:
    """Every identifier as "kind:value", highest priority first."""
    ids = identifiers(rec)
    return [f"{k}:{ids[k]}" for k in PRIORITY if k in ids]


def work_key(rec) -> str | None:
    """The record's own canonical key (highest-priority identifier), or None when it carries none."""
    ks = keys(rec)
    return ks[0] if ks else None


def _rank(key: str) -> int:
    return PRIORITY.index(key.split(":", 1)[0])


def version_link(rec) -> dict | None:
    """{"preprint": "arxiv:<id>", "published": "doi:<doi>"} when one record names both an arXiv id and a journal DOI
    (e.g. an arXiv entry with its journal DOI); else None. The published version wins the evidence slot (OA-3b)."""
    ids = identifiers(rec)
    if "arxiv" in ids and "doi" in ids:
        return {"preprint": f"arxiv:{ids['arxiv']}", "published": f"doi:{ids['doi']}"}
    return None


class KeyIndex:
    """Union of records sharing any key. add(rec) -> group id (int, insertion order of the group's first record) or
    None for a record with no key; canonical(gid) -> the group's highest-priority key."""

    def __init__(self):
        self._parent: list = []
        self._by_key: dict = {}
        self._keys: list = []

    def _find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def add(self, rec) -> int | None:
        ks = keys(rec)
        if not ks:
            return None
        roots = sorted({self._find(self._by_key[k]) for k in ks if k in self._by_key})
        if roots:
            g = roots[0]
            for r in roots[1:]:
                self._parent[r] = g
                self._keys[g] |= self._keys[r]
        else:
            g = len(self._parent)
            self._parent.append(g)
            self._keys.append(set())
        for k in ks:
            self._by_key.setdefault(k, g)
        self._keys[g] |= set(ks)
        return g

    def group(self, rec) -> int | None:
        """The group of an already-seen record (any shared key), without adding it."""
        for k in keys(rec):
            if k in self._by_key:
                return self._find(self._by_key[k])
        return None

    def canonical(self, gid: int) -> str:
        g = self._find(gid)
        return min(self._keys[g], key=lambda k: (_rank(k), k))


def mode(path: Path | None = None) -> str:
    """OA-3b rollout mode from config/work_key.json: off | shadow | on. A missing, unreadable or unknown value is
    "off" (the old dedup keys only: unchanged behaviour)."""
    try:
        m = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8")).get("mode")
    except Exception:  # noqa: BLE001
        return "off"
    return m if m in MODES else "off"


def extra_keys(rec, have) -> set:
    """Canonical work keys of rec not already in `have` (the caller's own key strings, any prefix scheme)."""
    try:
        return {k for k in keys(rec) if k not in have}
    except Exception:  # noqa: BLE001  (a malformed record keeps the old keys only)
        return set()


def log_shadow(site: str, would_merge: int) -> None:
    """Shadow evidence: a count only (no identifiers, titles or URLs)."""
    if would_merge:
        logger.info("work_key shadow site=%s would_merge=%d", site, would_merge)


# ------------------------------------------------------------------------------------------------ OA-3c provenance
def discovery_event(rec, provider: str | None = None) -> dict:
    """One discovery event of a candidate: {provider, record_id (its work key), url}. Metadata only."""
    from vera import search_providers as sp
    try:
        prov = provider or sp.acq_source(rec)
    except Exception:  # noqa: BLE001
        prov = None
    url = rec.get("url") if isinstance(rec, dict) and isinstance(rec.get("url"), str) else None
    return {"provider": str(prov or rec.get("provider") or "unknown")[:64] if isinstance(rec, dict) else "unknown",
            "record_id": work_key(rec), "url": url[:2000] if url else None}


def note_discovery(hit: dict, rec: dict) -> None:
    """mode "on" only (callers check): append rec's discovery event to hit["discovered_by"] (hit's own event first);
    an identical event is kept once."""
    events = hit.get("discovered_by")
    if not isinstance(events, list):
        events = [discovery_event(hit)]
    ev = discovery_event(rec)
    if ev not in events:
        events.append(ev)
    hit["discovered_by"] = events


def annotate_provenance(plan: dict, cand: dict) -> None:
    """mode "on" only (callers check): additive keys under provenance["acquisition"] (already a reader-ignored key on
    the acquired path, fulldoc_db.ACQ_EXTRA_KEYS, like acq_discovered's discovery record; no DDL, no reader change):
    work_key, discovered_by [{provider, record_id, url, seen_at}] and related_versions (preprint <-> published).
    seen_at is the time this acquisition recorded the event (acquisition.acquired_at), not the provider's own
    discovery time, which candidates do not carry. Does nothing when the plan has no acquisition record."""
    prov = ((plan or {}).get("source") or {}).get("provenance") if isinstance(plan, dict) else None
    acq = prov.get("acquisition") if isinstance(prov, dict) else None
    if not isinstance(acq, dict) or not isinstance(cand, dict):
        return
    seen = acq.get("acquired_at")
    events = cand.get("discovered_by") if isinstance(cand.get("discovered_by"), list) else [discovery_event(cand)]
    acq["work_key"] = work_key(cand)
    acq["discovered_by"] = [{**e, "seen_at": seen} for e in events if isinstance(e, dict)]
    link = version_link(cand)
    if link:
        acq["related_versions"] = link
