"""Fair-use short-quote licence tier and takedown path (FAIRUSE-TIER-1108, owner decision OD-1108-FAIRUSE, option O2).

VERA-DAR quotes, cites and lists source references in discussion; it does not republish or redistribute works. This
module keeps that description true for documents whose licence grants no reuse (the arXiv default non-exclusive
distribution licence, an explicit all-rights-reserved notice):

* licence tier (vera.licence_gate.licence_status calls admits()): only when config/fair_use.json enabled is literally
  true, the governing licence matched an ACTIVE allowlist entry listed in eligible_licence_ids, and the acquisition
  route recorded an access check (robots allowed, no TDM reservation, no login). Unknown, missing or conflicting
  licences never reach this tier. The reject list runs before it; every access rule runs in the routes before the gate.
* output caps (limit_citations, limit_trace): per answer and per source, from config caps. They apply to every source
  whose licence is eligible, whatever the flag (fail closed).
* takedown (is_withdrawn): a source listed in config/source_withdrawals.json is excluded from retrieval, answers and
  memory recall. The table vera_vjay.source_withdrawals (migration 023, owner-run kit, NOT applied) and its reader ship
  together later; this module reads no table today.

Every reader fails closed for the tier (unreadable config = off) and fails safe for caps (unreadable caps = the
strictest built-in defaults below).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fair_use.json"
WITHDRAWALS_PATH = Path(__file__).resolve().parent.parent / "config" / "source_withdrawals.json"
BASIS = "fair-use-quote (O2, OD-1108-FAIRUSE)"
RC_FAIR_USE = "licence_fair_use_quote"
# Fixed in code, not config: these strings are part of the stored provenance, which the DB read path re-derives and
# compares for equality (vera.fulldoc_db); a config edit must never make stored rows unusable.
USE = "quoted, cited and listed as a source reference in discussion; not republished or redistributed"
# strictest fallback when the config cannot be read (never wider than the shipped values)
_DEFAULT_CAPS = {"quote_max_chars": 200, "max_quotes_per_source": 2, "max_quote_chars_per_source": 400,
                 "max_share_of_document": 0.05}
_REQUIRED_ACCESS = {"robots": "allowed", "tdm_reservation": "none", "login_required": False}


def load(path: Path | None = None) -> dict:
    try:
        d = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def enabled(cfg: dict | None = None) -> bool:
    c = load() if cfg is None else cfg
    return c.get("enabled") is True


def eligible_ids(cfg: dict | None = None) -> set:
    c = load() if cfg is None else cfg
    ids = c.get("eligible_licence_ids")
    return {str(x) for x in ids} if isinstance(ids, list) else set()


def caps(cfg: dict | None = None) -> dict:
    """The output caps; any missing or malformed value falls back to the built-in (strict) default."""
    c = (load() if cfg is None else cfg).get("caps")
    out = dict(_DEFAULT_CAPS)
    if isinstance(c, dict):
        for k, dv in _DEFAULT_CAPS.items():
            v = c.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and (
                    k != "max_share_of_document" or v <= 1):
                out[k] = type(dv)(v)
    return out


def access_ok(ac) -> bool:
    """True only for a route-recorded access check with every required value exactly as expected."""
    return isinstance(ac, dict) and all(ac.get(k) == v and type(ac.get(k)) is type(v) for k, v in _REQUIRED_ACCESS.items())


def access_record(checked_by: str, checked_at: str) -> dict:
    """The access check an acquisition route records on its item. Call it only AFTER the route's own robots.txt,
    TDM (tdmrep.json, TDM-Reservation header, tdm-reservation meta) checks passed on an anonymous HTTPS fetch."""
    return {**_REQUIRED_ACCESS, "checked_by": str(checked_by), "checked_at": str(checked_at)}


def admits(entry: dict | None, c: dict) -> bool:
    """The licence-gate test for this tier: flag on, the matched allowlist entry is eligible, the stored text is full
    content (not abstract metadata) and the candidate carries a valid access check. Any error = False."""
    try:
        cfg = load()
        return (enabled(cfg) and isinstance(entry, dict) and entry.get("status") == "active"
                and entry.get("reject") is not True and str(entry.get("id")) in eligible_ids(cfg)
                and c.get("stored_text_kind") != "abstract_metadata" and access_ok(c.get("access_check")))
    except Exception:  # noqa: BLE001  fail closed
        return False


def provenance_record(access_check: dict) -> dict:
    """licence.fair_use on the source's provenance: the basis, the use, and the access/reservation check result."""
    return {"basis": BASIS, "use": USE,
            "access_check": {k: access_check.get(k) for k in ("robots", "tdm_reservation", "login_required",
                                                                "checked_by", "checked_at")}}


def arxiv_family(fam: str) -> dict | None:
    """arXiv route: {'licence_id', 'licence_url'} for an arXiv licence family ('nonexclusive-distrib') when the tier is
    enabled and the family is configured, else None (the route keeps its hold)."""
    cfg = load()
    if not enabled(cfg):
        return None
    m = (cfg.get("arxiv_families") or {}).get(str(fam))
    if isinstance(m, dict) and m.get("licence_id") in eligible_ids(cfg) and isinstance(m.get("licence_url"), str):
        return {"licence_id": m["licence_id"], "licence_url": m["licence_url"]}
    return None


def arxiv_default_url_ok(url) -> bool:
    """licence_gate's arXiv /html exception: the content licence URL is a configured arXiv default licence URL and the
    tier is enabled."""
    cfg = load()
    return enabled(cfg) and isinstance(url, str) and any(
        isinstance(m, dict) and m.get("licence_url") == url for m in (cfg.get("arxiv_families") or {}).values())


# ------------------------------------------------------------------------------------------------ output caps
def is_fair_use_source(licence_id, licence_url=None) -> bool:
    """True when the licence matches an allowlist entry in eligible_licence_ids (flag ignored: caps always apply)."""
    try:
        from vera.licence_gate import match_entry
        e, _how = match_entry(licence_id, licence_url)
        return isinstance(e, dict) and str(e.get("id")) in eligible_ids()
    except Exception:  # noqa: BLE001
        return False


def _attribution_complete(a) -> bool:
    return isinstance(a, dict) and all(str(a.get(k) or "").strip() for k in ("creator", "title", "source", "licence")) \
        and str(a.get("source")).startswith(("https://", "http://"))


def limit_citations(citations: list, sources: list, doc_chars: dict | None = None) -> int:
    """Per-answer, per-source caps for fair-use sources: each quote <= quote_max_chars, at most max_quotes_per_source
    quotes and max_quote_chars_per_source characters per document, and at most max_share_of_document of the document's
    characters (when its length is known). A quote without a full attribution with a link to the original is dropped.
    Over a cap the quote is dropped and the span reference kept (as nd_guard G6). Mutates; returns quotes dropped."""
    by_n = {s.get("n"): s for s in sources if isinstance(s, dict)}
    cp = caps()
    fair = {n for n, s in by_n.items() if is_fair_use_source(s.get("licence"), s.get("licence_url"))}
    if not fair:
        return 0
    n_used: dict = {}
    c_used: dict = {}
    dropped = 0
    for c in citations:
        if not isinstance(c, dict) or c.get("n") not in fair or not c.get("quote"):
            continue
        s = by_n[c["n"]]
        key = str(s.get("document_id") or s.get("canonical_source_id") or s.get("url"))
        q = c["quote"]
        total = (doc_chars or {}).get(str(s.get("document_id")))
        share_cap = int(total * cp["max_share_of_document"]) if isinstance(total, int) and total > 0 else None
        ok = (len(q) <= cp["quote_max_chars"] and n_used.get(key, 0) + 1 <= cp["max_quotes_per_source"]
              and c_used.get(key, 0) + len(q) <= cp["max_quote_chars_per_source"]
              and (share_cap is None or c_used.get(key, 0) + len(q) <= share_cap)
              and _attribution_complete(c.get("attribution")))
        if ok:
            n_used[key] = n_used.get(key, 0) + 1
            c_used[key] = c_used.get(key, 0) + len(q)
        else:
            c["quote"], c["quote_span"] = None, None
            dropped += 1
    return dropped


def limit_trace(trace, citations: list, sources: list) -> None:
    """Public evidence trace: a fair-use passage's excerpt is replaced by its surviving (capped) citation quote, or
    removed, so the trace never shows text beyond the capped quotes. Mutates trace['passages']."""
    if not isinstance(trace, dict):
        return
    by_n = {s.get("n"): s for s in sources if isinstance(s, dict)}
    fair = {n for n, s in by_n.items() if is_fair_use_source(s.get("licence"), s.get("licence_url"))}
    if not fair:
        return
    quote = {}
    for c in citations or ():
        if isinstance(c, dict) and c.get("n") in fair and c.get("quote") and c["n"] not in quote:
            quote[c["n"]] = c["quote"]
    for row in trace.get("passages") or ():
        if isinstance(row, dict) and row.get("n") in fair and "excerpt" in row:
            row["excerpt"] = quote.get(row["n"])


# ------------------------------------------------------------------------------------------------ takedown
_DB_LOCK = threading.Lock()
_DB_CACHE: dict = {"at": None, "entries": []}
_ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:|10\.48550/arxiv\.)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})",
                       re.I)


def norm_url(u) -> str:
    s = str(u or "").strip().lower()
    if not s:
        return ""
    sp = urlsplit(s if "://" in s else "https://" + s)
    host = (sp.hostname or "").removeprefix("www.")
    return f"{host}{sp.path.rstrip('/')}" if host else ""


def _norm_doi(d) -> str:
    s = str(d or "").strip().lower()
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", s)
    return s


def _arxiv_id(*vals) -> str:
    for v in vals:
        m = _ARXIV_ID.search(str(v or ""))
        if m:
            return m.group(1).lower()
    return ""


def _keys(entry: dict) -> set:
    out = set()
    for k, f in (("document_id", str), ("doi", _norm_doi), ("url", norm_url), ("arxiv_id", lambda x: str(x).lower())):
        v = entry.get(k)
        if v:
            out.add((k, f(v)))
    return {x for x in out if x[1]}


def _config_entries(path: Path | None = None) -> list:
    try:
        d = json.loads((path or WITHDRAWALS_PATH).read_text(encoding="utf-8"))
        e = d.get("entries") if isinstance(d, dict) else None
        return [x for x in e if isinstance(x, dict)] if isinstance(e, list) else []
    except (OSError, ValueError):
        logger.warning("source withdrawals: config unreadable")
        return []


def withdrawn_keys() -> set:
    with _DB_LOCK:
        db = list(_DB_CACHE["entries"])  # empty until the migration-023 reader ships (work/FAIRUSE-TIER-1108)
    out = set()
    for e in _config_entries() + db:
        out |= _keys(e)
    return out


def is_withdrawn(doc: dict, keys: set | None = None) -> bool:
    """True when any identifier of the document (document_id, DOI, URL, canonical source URL, arXiv id) is withdrawn."""
    if not isinstance(doc, dict):
        return False
    ks = withdrawn_keys() if keys is None else keys
    if not ks:
        return False
    dois = {_norm_doi(doc.get("doi"))} | {_norm_doi(u) for u in (doc.get("url"), doc.get("source_url"),
                                                                  doc.get("canonical_source_id"))
                                          if re.match(r"https?://(?:dx\.)?doi\.org/", str(u or ""), re.I)}
    mine = {("document_id", str(doc.get("document_id") or "")), *{("doi", d) for d in dois},
            ("url", norm_url(doc.get("url"))), ("url", norm_url(doc.get("canonical_source_id"))),
            ("url", norm_url(doc.get("source_url"))),
            ("arxiv_id", _arxiv_id(doc.get("url"), doc.get("doi"), doc.get("canonical_source_id"), doc.get("source_url")))}
    return bool({m for m in mine if m[1]} & ks)


def reset_for_tests() -> None:
    with _DB_LOCK:
        _DB_CACHE.update(at=None, entries=[])
