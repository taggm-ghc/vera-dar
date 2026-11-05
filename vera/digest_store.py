"""PERF-MISS-1104-TRACE / owner rule "all runtime state lives in the DB": persist the latest corpus digest.

Read/write helper; vera/corpus_description.CorpusDescriber uses it (INTEGRATE-1104) only while
config/request_trace.json `digest_store.enabled` is true. The owner applied migration 015 on 2026-11-04 (creates
vera_vjay.corpus_digest_state); the flag stays false for this increment (deferred).

  save(digest, cache_key) -> bool     app-path INSERT through the write-only account (one new row per digest)
  load_latest() -> dict | None        SELECT of the newest row through the read-only account

The digest is the public corpus description (counts, summary, topics); it holds no visitor data. Fail-open: both calls
never raise and log a fixed code plus the exception type only (never a URL, host or the digest).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "request_trace.json"
TABLE_RE = re.compile(r"^vera_vjay\.[a-z_][a-z0-9_]{0,62}\Z")
KEY_RE = re.compile(r"^[A-Za-z0-9_.:|+-]{1,200}\Z")

_ENGINES: dict = {}
_LOCK = threading.Lock()


def load_cfg(path: Path | None = None) -> dict:
    try:
        cfg = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8")).get("digest_store")
        return cfg if isinstance(cfg, dict) else {"enabled": False}
    except Exception:  # noqa: BLE001  unreadable config means off
        return {"enabled": False}


def enabled(cfg: dict | None = None) -> bool:
    return (cfg if cfg is not None else load_cfg()).get("enabled") is True


def _table(cfg: dict) -> str:
    t = str(cfg.get("table") or "vera_vjay.corpus_digest_state")
    if not TABLE_RE.match(t):
        raise ValueError("bad table name")
    return t


def _engine(cfg: dict, env, role: str, factory=None):
    url = (env.get(str(cfg.get(f"db_url_env_{role}") or f"VERA_DB_URL_{role.upper()}")) or "").strip()
    if not url:
        return None
    with _LOCK:
        eng = _ENGINES.get((role, url))
        if eng is None:
            if factory is None:
                from sqlalchemy import create_engine as factory
            ms = int(cfg.get("statement_timeout_ms") or 2000)
            eng = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                          connect_args={"connect_timeout": int(cfg.get("connect_timeout_s") or 2),
                                        "options": f"-c statement_timeout={ms}"})
            _ENGINES[(role, url)] = eng
    return eng


def save(digest, cache_key: str, *, cfg: dict | None = None, env=os.environ, engine_factory=None) -> bool:
    """INSERT one row (cache_key, digest jsonb, sha256). No-op unless enabled; True only if written. Never raises."""
    try:
        cfg = cfg if cfg is not None else load_cfg()
        if not enabled(cfg) or not isinstance(digest, dict) or not (isinstance(cache_key, str) and KEY_RE.match(cache_key)):
            return False
        body = json.dumps(digest, separators=(",", ":"), sort_keys=True, ensure_ascii=False, default=str)
        if len(body.encode("utf-8")) > int(cfg.get("max_digest_bytes") or 65536):
            logger.warning("digest_store save skipped: DS_TOO_LARGE")
            return False
        eng = _engine(cfg, env, "wo", engine_factory)
        if eng is None:
            logger.warning("digest_store save skipped: DS_NO_WRITER")
            return False
        from sqlalchemy import text
        with eng.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout = " + str(int(cfg.get("statement_timeout_ms") or 2000))))
            conn.execute(text(f"INSERT INTO {_table(cfg)} (cache_key, digest, digest_sha256) "
                              "VALUES (:k, CAST(:d AS jsonb), :h)"),
                         {"k": cache_key, "d": body, "h": hashlib.sha256(body.encode("utf-8")).hexdigest()})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("digest_store save failed: DS_WRITE_ERROR (%s)", type(exc).__name__)
        return False


def load_latest(*, cfg: dict | None = None, env=os.environ, engine_factory=None) -> dict | None:
    """{"cache_key", "digest", "created_at"} of the newest row, or None (off, empty, or any error). Never raises."""
    try:
        cfg = cfg if cfg is not None else load_cfg()
        if not enabled(cfg):
            return None
        eng = _engine(cfg, env, "ro", engine_factory)
        if eng is None:
            return None
        from sqlalchemy import text
        with eng.connect() as conn:
            row = conn.execute(text(f"SELECT cache_key, digest, created_at FROM {_table(cfg)} "
                                    "ORDER BY digest_id DESC LIMIT 1")).first()
        if row is None:
            return None
        m = dict(row._mapping)
        d = m.get("digest")
        if isinstance(d, str):
            d = json.loads(d)
        if not isinstance(d, dict):
            return None
        ca = m.get("created_at")
        return {"cache_key": m.get("cache_key"), "digest": d, "created_at": ca.isoformat() if hasattr(ca, "isoformat") else ca}
    except Exception as exc:  # noqa: BLE001
        logger.warning("digest_store read failed: DS_READ_ERROR (%s)", type(exc).__name__)
        return None


def reset_for_tests() -> None:
    with _LOCK:
        _ENGINES.clear()
