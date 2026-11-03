"""WP-L (R-030): anonymous live aggregate statistics, OFF by default (config/live_stats.json `enabled`).

Stored fields are enums or ints only: no question text, no client id, no IP, no free text. The writer is fail-open and
runs as a background task (never inline in the request). The public reader returns day-bucketed counts with small-count
and complementary suppression and never tokens or cost; the owner reader adds token and cost totals.
NOT YET APPLIED: the table comes from the draft migration db/migrations/pending/013-DRAFT-*.sql.draft (owner-run DDL).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "live_stats.json"

RESULT_KINDS = ("answer", "partial_answer", "decline", "reading_list", "self_description", "other")
RESEARCH_OUTCOMES = ("answered_with_new", "qualified_older_only", "no_valuable_source", "incomplete_search",
                     "abstain", "none", "other")
SUPPRESSED = "<{n}"

_ENGINES: dict = {}
_ENG_LOCK = threading.Lock()
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def load_cfg(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {"enabled": False}
    except Exception:  # noqa: BLE001  fail closed: unreadable config means off
        return {"enabled": False}


def enabled(cfg: dict | None = None) -> bool:
    return (cfg if cfg is not None else load_cfg()).get("enabled") is True


def _int(v, hi: int = 10_000_000) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return max(0, min(int(v), hi))


def _enum(v, allowed: tuple, other: str = "other") -> str:
    return v if isinstance(v, str) and v in allowed else other


def project(body, latency_ms) -> dict:
    """Allowlist projection: enums and ints only; nothing else of the body is read."""
    b = body if isinstance(body, dict) else {}
    ro = b.get("research_outcome")
    rk = ro.get("kind") if isinstance(ro, dict) else None
    cc = b.get("claim_check")
    claims = (_int(cc.get("supported")) + _int(cc.get("partial"))) if isinstance(cc, dict) else 0
    cost = b.get("cost_usd")
    return {"result_kind": _enum(b.get("result_kind"), RESULT_KINDS),
            "research_outcome": "none" if rk is None else _enum(rk, RESEARCH_OUTCOMES),
            "claims": _int(claims),
            "citations": len(b["citations"]) if isinstance(b.get("citations"), list) else 0,
            "evidence_passages": len(b["admitted_evidence"]) if isinstance(b.get("admitted_evidence"), list) else 0,
            "latency_ms": _int(latency_ms),
            "tokens": _int(b.get("tokens_used")),
            "cost_usd": round(max(0.0, min(float(cost), 1e6)), 6)
                        if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0}


def _engine(cfg: dict, env, factory=None):
    url = (env.get(str(cfg.get("db_url_env_wo") or "VERA_DB_URL_WO")) or "").strip()
    if not url:
        return None
    with _ENG_LOCK:
        eng = _ENGINES.get(url)
        if eng is None:
            if factory is None:
                from sqlalchemy import create_engine as factory
            eng = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                          connect_args={"connect_timeout": int(cfg.get("connect_timeout_s") or 2),
                                        "options": f"-c statement_timeout={int(cfg.get('statement_timeout_ms') or 2000)}"})
            _ENGINES[url] = eng
    return eng


def _insert(row: dict, cfg: dict, env, factory) -> None:
    eng = _engine(cfg, env, factory)
    if eng is None:
        logger.warning("live_stats skipped: LS_NO_WRITER")
        return
    from sqlalchemy import text
    table = str(cfg.get("table") or "vera_vjay.live_ask_stats")
    sql = text(f"INSERT INTO {table} (result_kind, research_outcome, claims, citations, evidence_passages, latency_ms, "
               "tokens, cost_usd) VALUES (:result_kind, :research_outcome, :claims, :citations, :evidence_passages, "
               ":latency_ms, :tokens, :cost_usd)")
    with eng.begin() as conn:
        conn.execute(text("SET LOCAL statement_timeout = " + str(int(cfg.get("statement_timeout_ms") or 2000))))
        conn.execute(sql, row)


def record(body, latency_ms, *, cfg: dict | None = None, env=os.environ, engine_factory=None) -> bool:
    """Write one anonymous row. No-op unless enabled. Fail-open: never raises, logs a fixed code only (never the body).
    Stops waiting at `hard_timeout_s` (the write is abandoned to its own statement timeout). Returns True if written."""
    try:
        cfg = cfg if cfg is not None else load_cfg()
        if not enabled(cfg):
            return False
        row = project(body, latency_ms)
        box: dict = {}

        def _run():
            try:
                _insert(row, cfg, env, engine_factory)
                box["ok"] = True
            except Exception as exc:  # noqa: BLE001
                box["err"] = type(exc).__name__

        t = threading.Thread(target=_run, name="live-stats-write", daemon=True)
        t.start()
        t.join(float(cfg.get("hard_timeout_s") or 3))
        if t.is_alive():
            logger.warning("live_stats write abandoned: LS_TIMEOUT")
            return False
        if "err" in box:
            logger.warning("live_stats write failed: LS_WRITE_ERROR (%s)", box["err"])
            return False
        return bool(box.get("ok"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_stats record failed: LS_ERROR (%s)", type(exc).__name__)
        return False


# ------------------------------------------------------------------------------------------------ reader
def suppress_map(counts: dict, n_min: int) -> dict:
    """label -> count. Counts below n_min become '<n_min'. If exactly one is suppressed, the smallest remaining count is
    suppressed too (complementary), so total minus the others cannot reveal it. Zero counts are omitted."""
    label = SUPPRESSED.format(n=n_min)
    items = {k: v for k, v in counts.items() if v > 0}
    low = [k for k, v in items.items() if v < n_min]
    if len(low) == 1 and len(items) > 1:
        rest = sorted((k for k in items if k not in low), key=lambda k: items[k])
        low.append(rest[0])
    return {k: (label if k in low else v) for k, v in sorted(items.items())}


def _query(window_days: int, cfg: dict, env, factory, owner: bool) -> list:
    url = (env.get(str(cfg.get("db_url_env_ro") or "VERA_DB_URL_RO")) or "").strip()
    if not url:
        raise RuntimeError("no reader")
    if factory is None:
        from sqlalchemy import create_engine as factory
    from sqlalchemy import text
    with _ENG_LOCK:
        eng = _ENGINES.get(("ro", url))
        if eng is None:
            eng = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                          connect_args={"connect_timeout": int(cfg.get("connect_timeout_s") or 2),
                                        "options": f"-c statement_timeout={int(cfg.get('statement_timeout_ms') or 2000)}"})
            _ENGINES[("ro", url)] = eng
    table = str(cfg.get("table") or "vera_vjay.live_ask_stats")
    extra = ", COALESCE(sum(tokens),0) AS tokens, COALESCE(sum(cost_usd),0) AS cost_usd" if owner else ""
    sql = text(f"SELECT date_trunc('day', ts)::date::text AS day, result_kind, count(*) AS n, "
               f"percentile_disc(0.5) WITHIN GROUP (ORDER BY latency_ms) AS median_latency_ms{extra} "
               f"FROM {table} WHERE ts >= now() - make_interval(days => :d) GROUP BY 1, 2 ORDER BY 1, 2")
    with eng.connect() as conn:
        conn.execute(text("SET LOCAL statement_timeout = " + str(int(cfg.get("statement_timeout_ms") or 2000))))
        return [dict(r._mapping) for r in conn.execute(sql, {"d": window_days})]


def read_aggregates(window_days: int = 7, *, owner: bool = False, cfg: dict | None = None, env=os.environ,
                    engine_factory=None, now=time.monotonic) -> dict:
    """Cached (TTL) day-bucketed aggregates. Public output has no tokens or cost keys; owner output adds totals."""
    cfg = cfg if cfg is not None else load_cfg()
    if not enabled(cfg):
        return {"enabled": False, "total": 0}
    n_min = max(1, int(cfg.get("min_count") or 5))
    w = max(1, min(int(window_days), int(cfg.get("window_days_max") or 30)))
    key = (w, owner)
    ttl = float(cfg.get("cache_ttl_s") or 60)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now() - hit[0] < ttl:
            return hit[1]
    try:
        rows = _query(w, cfg, env, engine_factory, owner)
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_stats read failed: LS_READ_ERROR (%s)", type(exc).__name__)
        return {"enabled": True, "total": 0, "by_result_kind": {}, "by_day": {}, "typical_latency_ms": {}}
    rows = [{"day": str(r["day"]), "kind": _enum(r["result_kind"], RESULT_KINDS), "n": _int(r["n"]),
             "med": _int(r.get("median_latency_ms")), "tokens": _int(r.get("tokens")),
             "cost": float(r.get("cost_usd") or 0)} for r in rows]
    by_kind: dict = {}
    by_day: dict = {}
    meds: dict = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["n"]
        by_day[r["day"]] = by_day.get(r["day"], 0) + r["n"]
        meds.setdefault(r["kind"], []).append(r["med"])
    total = sum(by_kind.values())
    sup_kind = suppress_map(by_kind, n_min)
    label = SUPPRESSED.format(n=n_min)
    typical = {k: (label if sup_kind.get(k) == label else sorted(v)[len(v) // 2]) for k, v in meds.items() if k in sup_kind}
    out = {"enabled": True, "total": total if total >= n_min else label,
           "by_result_kind": sup_kind, "by_day": suppress_map(by_day, n_min), "typical_latency_ms": typical}
    if owner:
        out["tokens_total"] = sum(r["tokens"] for r in rows)
        out["cost_usd_total"] = round(sum(r["cost"] for r in rows), 6)
    with _CACHE_LOCK:
        _CACHE[key] = (now(), out)
    return out


def reset_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
    with _ENG_LOCK:
        _ENGINES.clear()
