"""Chunk freshness assessment (FRESH-C1). Pure: no I/O, no global clock (the caller injects ``now``).

Owner requirement: freshness of relevant chunks is judged against a 24-hour threshold of SOURCE-CONTENT age.
Ingestion / embedding / retrieval / acquisition / last-checked timestamps are copied through for the record but are
NEVER used for status. A recently ingested old publication is therefore stale. Missing or untrustworthy dates give
status "unknown" (the integrator triggers external verification for any_stale_or_unknown).

Metadata shapes read (dict or object; chunk may nest the source under "source"/"document"/"doc"):
  chunk-level date (basis "chunk"): chunk_published, chunk_updated, chunk_date, chunk_modified, section_published
  source-version date (basis "source"; first present wins, update before publication):
      updated, updated_at, last_modified, modified, source_updated, then
      published, published_date, published_at, publication_date, issued, date
  (base code carries "published" as a str: YYYY-MM-DD, YYYY-MM or a bare year, e.g. grounded_ask / dar_corpus /
  fulltext_loader; no source-update field exists there as of base 9d0d764.)
  acquired_at <- acquired_at | retrieved_at | fetched_at | ingested_at ;  last_checked_at <- last_checked_at | last_checked
cfg keys: freshness_max_age_hours (owner-specified 24 h; default 24 only if absent),
  freshness_future_skew_seconds (default 300, UNCALIBRATED), freshness_year_only_trusted (default False, UNCALIBRATED:
  year-only / year-month dates are treated as untrustworthy; if True they resolve to the LATEST instant of the period).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_MAX_AGE_HOURS = 24  # owner-specified
DEFAULT_FUTURE_SKEW_S = 300  # uncalibrated
CHUNK_KEYS = ("chunk_published", "chunk_updated", "chunk_date", "chunk_modified", "section_published")
SOURCE_KEYS = ("updated", "updated_at", "last_modified", "modified", "source_updated",
               "published", "published_date", "published_at", "publication_date", "issued", "date")
ACQ_KEYS = ("acquired_at", "retrieved_at", "fetched_at", "ingested_at")
CHK_KEYS = ("last_checked_at", "last_checked")
NEST_KEYS = ("source", "document", "doc")


@dataclass
class ChunkFreshness:
    source_date: datetime | None
    basis: str  # 'chunk' | 'source' | 'unknown'
    status: str  # 'fresh' | 'stale' | 'unknown'
    acquired_at: object = None
    last_checked_at: object = None
    reason: str = ""
    chunk_id: object = None


@dataclass
class FreshnessReport:
    chunks: list = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"fresh": 0, "stale": 0, "unknown": 0})
    any_stale_or_unknown: bool = False
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS


def utcnow() -> datetime:
    """FRESH-1023 clock hook (tests monkeypatch vera.freshness.utcnow for deterministic 24-hour boundaries)."""
    return datetime.now(timezone.utc)


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first(obj, keys):
    """(key, value) of the first non-empty key, else (None, None)."""
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


def _parse(v, year_only_trusted):
    """-> (datetime aware UTC | None, reason). Partial dates need year_only_trusted."""
    if isinstance(v, bool):
        return None, "unparseable"
    if isinstance(v, datetime):
        return _utc(v), ""
    if isinstance(v, int):
        v = str(v)
    if not isinstance(v, str):
        return None, "unparseable"
    s = v.strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", s)
    if m:
        if not year_only_trusted:
            return None, "year-only/partial date untrusted"
        y, mo = int(m.group(1)), m.group(2)
        try:
            if mo is None:
                return datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc), ""
            mo = int(mo)
            nxt = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
            return nxt - timedelta(seconds=1), ""
        except ValueError:
            return None, "unparseable"
    try:
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        return _utc(datetime.fromisoformat(s)), ""
    except ValueError:
        return None, "unparseable"


def assess_freshness(chunks, *, now: datetime, cfg: dict) -> FreshnessReport:
    cfg = cfg or {}
    max_h = float(cfg.get("freshness_max_age_hours", DEFAULT_MAX_AGE_HOURS))
    skew = timedelta(seconds=float(cfg.get("freshness_future_skew_seconds", DEFAULT_FUTURE_SKEW_S)))
    yo = bool(cfg.get("freshness_year_only_trusted", False))
    now_u = _utc(now)
    max_age = timedelta(hours=max_h)
    rep = FreshnessReport(max_age_hours=max_h)
    for ch in chunks or []:
        _, acq = _find(ch, ACQ_KEYS)
        _, chk = _find(ch, CHK_KEYS)
        date, basis, reason = None, "unknown", "no date"
        # chunk-level date if present and trustworthy; otherwise fall back to the source-version date
        for b, keys, finder in (("chunk", CHUNK_KEYS, lambda c, k: _first(c, k)), ("source", SOURCE_KEYS, _find)):
            _, raw = finder(ch, keys)
            if raw is None:
                continue
            d, why = _parse(raw, yo)
            if d is None:
                reason = why
            elif d - now_u > skew:
                reason = "future date"
            else:
                date, basis, reason = d, b, ""
                break
        if date is None:
            status = "unknown"
        else:
            status = "fresh" if now_u - date <= max_age else "stale"
        rep.chunks.append(ChunkFreshness(date, basis, status, acq, chk, reason,
                                         _get(ch, "chunk_id") or _get(ch, "span_id")))
        rep.counts[status] += 1
    rep.any_stale_or_unknown = (rep.counts["stale"] + rep.counts["unknown"]) > 0
    return rep
