"""Chunk freshness assessment (FRESH-C1, amended FRESH-AMEND-1023). Pure: no I/O, no global clock (caller injects ``now``).

Owner clarification 2026-10-23 (verbatim): "there should be a date stamp for when a chunk is added to the corpus. that is the
date that should be used for freshness testing. there should separately be a date stamp for each chunk's source
creation/authoring". Owner refinement: a missing or invalid corpus_added_at "counts as unknown and triggers research AND
triggers setting that value to the then now()" (the stamping itself is vera.corpus_stamp; this module only reports which
chunk stamps need it).

Two distinct per-chunk timestamps:
  corpus_added_at  -> the 24-hour freshness TRIGGER (status fresh | stale | unknown). Read from, in order:
      corpus_chunks: [{"corpus_added_at", "basis", "span_id"}, ...]  stored chunk stamps overlapping the passage (set by the
          corpus reader: evidence_spans.created_at, else sources.created_at, else provenance.retrieved_at); every stamp
          must be valid, the OLDEST decides; basis "chunk" when every stamp is the chunk's own, else "document"
      else corpus_added_at on the chunk or nested source/document/doc (basis corpus_added_basis, default "document")
      missing / unparseable / partial or date-only (not a full timestamp) / future -> "unknown" (a trigger) and the
      holding dict is listed in ChunkFreshness.backfill (revised to the request's now() by vera.corpus_stamp).
  source_authored_at -> ONLY the older / recent label (authored_recency recent | older | unknown), never the trigger.
      chunk-level: chunk_published, chunk_updated, chunk_date, chunk_modified, section_published (basis "chunk"), then the
      source-version date (basis "source"): updated, last_modified, source_updated, published, published_date, published_at,
      publication_date, issued, date. Precision 'day' (full date / timestamp), 'month' (YYYY-MM) or 'year' (YYYY).
      A partial date is 'older' only when the WHOLE period ended more than freshness_max_age_hours ago, otherwise
      'unknown': a partial date is never 'recent' (FRESH-VERIFY-1023 F1). Row/ingest times are never authored dates.
cfg keys: freshness_max_age_hours (owner-specified 24 h), freshness_future_skew_seconds (default 300, UNCALIBRATED),
  source_authored_partial_dates_trusted (default False: partial authored dates -> recency 'unknown'; True: 'older' or
  'unknown' as above). The trigger never reads it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_MAX_AGE_HOURS = 24  # owner-specified
DEFAULT_FUTURE_SKEW_S = 300  # uncalibrated
CHUNK_KEYS = ("chunk_published", "chunk_updated", "chunk_date", "chunk_modified", "section_published")
SOURCE_KEYS = ("updated", "last_modified", "source_updated",
               "published", "published_date", "published_at", "publication_date", "issued", "date")
CHK_KEYS = ("last_checked_at", "last_checked")
NEST_KEYS = ("source", "document", "doc")
PARTIAL_TRUST_KEY = "source_authored_partial_dates_trusted"


@dataclass
class ChunkFreshness:
    corpus_added_at: datetime | None
    basis: str  # corpus_added_at basis: 'chunk' | 'document' | 'unknown'
    status: str  # 'fresh' | 'stale' | 'unknown' (corpus_added_at vs the threshold)
    reason: str = ""
    source_authored_at: str | None = None  # normalised: YYYY | YYYY-MM | YYYY-MM-DD or ISO timestamp
    authored_precision: str | None = None  # 'year' | 'month' | 'day' | None
    authored_basis: str = "unknown"  # 'chunk' | 'source' | 'unknown'
    authored_recency: str = "unknown"  # 'recent' | 'older' | 'unknown'
    last_checked_at: object = None
    chunk_id: object = None
    backfill: list = field(default_factory=list)  # stamp dicts (by reference) whose corpus_added_at is missing/invalid


@dataclass
class FreshnessReport:
    chunks: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"fresh": 0, "stale": 0, "unknown": 0})
    authored_counts: dict = field(default_factory=lambda: {"recent": 0, "older": 0, "unknown": 0})
    any_stale_or_unknown: bool = False
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS


def utcnow() -> datetime:
    """FRESH-1023 clock hook (tests monkeypatch vera.freshness.utcnow for deterministic 24-hour boundaries). Also the
    write-time clock of corpus_added_at stamps (vera.auto_acquire, vera.corpus_stamp)."""
    return datetime.now(timezone.utc)


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first(obj, keys):
    for k in keys:
        v = _get(obj, k)
        if v not in (None, ""):
            return k, v
    return None, None


def _find(chunk, keys):
    k, v = _first(chunk, keys)
    if k is not None:
        return k, v
    for n in NEST_KEYS:
        k, v = _first(_get(chunk, n), keys)
        if k is not None:
            return k, v
    return None, None


def _utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)  # naive = UTC (documented assumption)
    return dt.astimezone(timezone.utc)


_PARTIAL = re.compile(r"(\d{4})(?:-(\d{2}))?")


def parse_timestamp(v):
    """Full timestamp or date only (partial year/month dates are NOT a valid corpus stamp). -> (datetime | None, reason)."""
    if isinstance(v, bool):
        return None, "unparseable"
    if isinstance(v, datetime):
        return _utc(v), ""
    if not isinstance(v, str):
        return None, "unparseable"
    s = v.strip()
    if _PARTIAL.fullmatch(s):
        return None, "partial timestamp"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return None, "date without time"  # owner 2026-10-23: imprecise corpus stamp -> invalid, revised to now()
    try:
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        return _utc(datetime.fromisoformat(s)), ""
    except ValueError:
        return None, "unparseable"


def stamp_valid(v, now: datetime, skew: timedelta) -> bool:
    d, _ = parse_timestamp(v)
    return d is not None and d - _utc(now) <= skew


def _authored(v):
    """-> (period start, period end, precision, normalised text) or (None, None, None, reason)."""
    if isinstance(v, bool):
        return None, None, None, "unparseable"
    if isinstance(v, datetime):
        d = _utc(v)
        return d, d, "day", d.isoformat()
    if isinstance(v, int):
        v = str(v)
    if not isinstance(v, str):
        return None, None, None, "unparseable"
    s = v.strip()
    m = _PARTIAL.fullmatch(s)
    try:
        if m:
            y, mo = int(m.group(1)), m.group(2)
            if mo is None:
                return (datetime(y, 1, 1, tzinfo=timezone.utc), datetime(y + 1, 1, 1, tzinfo=timezone.utc)
                        - timedelta(seconds=1), "year", s)
            mo = int(mo)
            start = datetime(y, mo, 1, tzinfo=timezone.utc)
            nxt = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
            return start, nxt - timedelta(seconds=1), "month", s
        t = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
        d = _utc(datetime.fromisoformat(t))
        return d, d, "day", s
    except ValueError:
        return None, None, None, "unparseable"


def _authored_of(ch, now_u, max_age, skew, partial_trusted):
    for b, finder in (("chunk", lambda c: _first(c, CHUNK_KEYS)), ("source", lambda c: _find(c, SOURCE_KEYS))):
        _, raw = finder(ch)
        if raw is None:
            continue
        start, end, prec, txt = _authored(raw)
        if start is None:
            continue
        if start - now_u > skew:
            return txt, prec, b, "unknown"  # future authoring date: untrustworthy
        if prec == "day":
            return txt, prec, b, ("recent" if now_u - start <= max_age else "older")
        # partial: never 'recent'; 'older' only when the whole period ended more than max_age ago
        if partial_trusted and now_u - end > max_age:
            return txt, prec, b, "older"
        return txt, prec, b, "unknown"
    return None, None, "unknown", "unknown"


def _added_of(ch, now_u, skew):
    """-> (datetime | None, basis, reason, backfill stamps)."""
    stamps = _get(ch, "corpus_chunks")
    if isinstance(stamps, (list, tuple)) and stamps:
        bad, dates, own = [], [], True
        for s in stamps:
            if not isinstance(s, dict):
                continue
            d, why = parse_timestamp(s.get("corpus_added_at"))
            if d is None or d - now_u > skew:
                bad.append(s)
            else:
                dates.append(d)
                own = own and s.get("basis") == "chunk"
        if bad:
            return None, "unknown", "missing or invalid corpus_added_at", bad
        if dates:
            return min(dates), ("chunk" if own else "document"), "", []
    holder = ch if isinstance(ch, dict) and _get(ch, "corpus_added_at") not in (None, "") else None
    if holder is None:
        for n in NEST_KEYS:
            h = _get(ch, n)
            if isinstance(h, dict) and h.get("corpus_added_at") not in (None, ""):
                holder = h
                break
    raw = _get(holder, "corpus_added_at") if holder is not None else _find(ch, ("corpus_added_at",))[1]
    target = [holder] if isinstance(holder, dict) else ([ch] if isinstance(ch, dict) else [])
    if raw in (None, ""):
        return None, "unknown", "no corpus_added_at", target
    d, why = parse_timestamp(raw)
    if d is None:
        return None, "unknown", f"corpus_added_at {why}", target
    if d - now_u > skew:
        return None, "unknown", "future corpus_added_at", target
    basis = _get(ch, "corpus_added_basis")
    return d, (basis if basis in ("chunk", "document") else "document"), "", []


def assess_freshness(chunks, *, now: datetime, cfg: dict) -> FreshnessReport:
    cfg = cfg or {}
    max_h = float(cfg.get("freshness_max_age_hours", DEFAULT_MAX_AGE_HOURS))
    skew = timedelta(seconds=float(cfg.get("freshness_future_skew_seconds", DEFAULT_FUTURE_SKEW_S)))
    partial_trusted = bool(cfg.get(PARTIAL_TRUST_KEY, False))
    now_u = _utc(now)
    max_age = timedelta(hours=max_h)
    rep = FreshnessReport(max_age_hours=max_h)
    for ch in chunks or []:
        _, chk = _find(ch, CHK_KEYS)
        added, basis, reason, backfill = _added_of(ch, now_u, skew)
        status = "unknown" if added is None else "fresh" if now_u - added <= max_age else "stale"
        a_txt, a_prec, a_basis, a_rec = _authored_of(ch, now_u, max_age, skew, partial_trusted)
        rep.chunks.append(ChunkFreshness(added, basis, status, reason, a_txt, a_prec, a_basis, a_rec, chk,
                                         _get(ch, "chunk_id") or _get(ch, "span_id"), backfill))
        rep.counts[status] += 1
        rep.authored_counts[a_rec] += 1
    rep.any_stale_or_unknown = (rep.counts["stale"] + rep.counts["unknown"]) > 0
    return rep
