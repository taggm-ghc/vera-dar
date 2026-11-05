"""LISTS-TO-DB-1104: read config lists from the DB views, fail closed to the config seed (OFF by default).

Owner directive 2026-11-04: data in lists, and lists in a table in the db. Reads only the two views
vera_vjay.lstreg_active_lists_v / lstreg_active_items_v (RO role). Any failure, or any malformed row of a list,
returns that list's SEED unchanged (never empty-by-error, never allow-all). Logs exception class names only.
NOT YET APPLIED: the tables and views come from the owner-run migration 016.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "list_tables.json"
SQL_SNAPSHOT = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
SQL_LISTS = "SELECT list_name, shape, combine, version FROM vera_vjay.lstreg_active_lists_v"
SQL_ITEMS = ("SELECT list_name, item_key, item_rank, version, payload FROM vera_vjay.lstreg_active_items_v "
             "ORDER BY list_name, item_rank NULLS LAST, item_key")

_ENGINES: dict = {}
_ENG_LOCK = threading.Lock()
_LOCK = threading.RLock()
_CACHE: dict = {"until": 0.0, "snap": None, "error": None, "tried": False, "busy": False}
_LOGGED: dict = {}
_LOG_EVERY_S = 60.0


def load_cfg(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {"read_from_db": False}
    except Exception:  # noqa: BLE001  unreadable config means off (seed returned)
        return {"read_from_db": False}


_CFG = load_cfg()


def reset_cache() -> None:
    with _LOCK:
        _CACHE.update(until=0.0, snap=None, error=None, tried=False, busy=False)
        _LOGGED.clear()
    with _ENG_LOCK:
        _ENGINES.clear()


def _on(cfg: dict) -> bool:
    return cfg.get("read_from_db") is True and isinstance(cfg.get("lists"), dict)


def _engine(cfg: dict, env, factory):
    url = (env.get(str(cfg.get("db_url_env") or "VERA_DB_URL_RO")) or "").strip()
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


def _fetch(cfg: dict, env, factory) -> dict | None:
    eng = _engine(cfg, env, factory)
    if eng is None:
        raise LookupError("LT_NO_URL")
    if factory is None:
        from sqlalchemy import text
    else:
        def text(s):  # injected (test) engines take the SQL string as is
            return s
    with eng.connect() as conn:
        conn.execute(text(SQL_SNAPSHOT))  # first statement: one snapshot for both views
        lists = [tuple(r) for r in conn.execute(text(SQL_LISTS)).fetchall()]
        items = [tuple(r) for r in conn.execute(text(SQL_ITEMS)).fetchall()]
    meta = {r[0]: {"shape": r[1], "combine": r[2], "version": r[3]} for r in lists}
    by: dict = {}
    for r in items:
        by.setdefault(r[0], []).append(r)
    for rows in by.values():
        rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 0, str(r[1])))
    return {"meta": meta, "items": by}


def _snapshot(cfg: dict, env, factory, clock):
    """Single-flight refresh (review F7): the network call runs outside _LOCK; threads that arrive while one
    refresh is in flight get the current (possibly stale or absent => seed) snapshot instead of blocking."""
    with _LOCK:
        now = clock()
        if now < _CACHE["until"] or _CACHE.get("busy"):
            return _CACHE["snap"]
        _CACHE["busy"] = True
    snap, err, ttl = None, None, float(cfg.get("failure_retry_s") or 60)
    try:
        snap = _fetch(cfg, env, factory)
        ttl = float(cfg.get("cache_ttl_s") or 300)
    except Exception as exc:  # noqa: BLE001  fail closed to seed, cache the failure
        err = type(exc).__name__
        logger.warning("list_tables refresh failed: LT_DB_ERROR (%s)", err)
    with _LOCK:
        _CACHE.update(snap=snap, error=err, tried=True, until=now + ttl, busy=False)
        return _CACHE["snap"]


def snapshot_status(*, cfg: dict | None = None) -> dict:
    cfg = _CFG if cfg is None else cfg
    if not _on(cfg):
        return {"source": "off", "lists": 0, "error": None}
    with _LOCK:
        snap = _CACHE["snap"]
        return {"source": "db" if snap is not None else "seed",
                "lists": len(snap["meta"]) if snap is not None else 0, "error": _CACHE["error"]}


def _payload(v):
    if isinstance(v, str):
        v = json.loads(v)
    if not isinstance(v, dict):
        raise ValueError("payload not an object")
    return v


def _build(spec: dict, rows: list, seed=None):
    shape = spec.get("shape")
    if shape == "strings":
        out = []
        for r in rows:
            if not isinstance(r[1], str) or not r[1]:
                raise ValueError("empty item_key")
            out.append(r[1])
        return out
    if shape == "records":
        kf, sf, out = spec.get("key_field"), spec.get("state_field"), []
        for r in rows:
            p = copy.deepcopy(_payload(r[4]))
            k = p.get(kf)
            if not isinstance(k, str) or not k or (k.lower() if spec.get("key_lower") else k) != r[1]:
                raise ValueError("key mismatch")
            if sf:
                p[sf] = "active"
            out.append(p)
        return out
    if shape == "stage_keys":
        out: dict = {}
        for r in rows:
            ik, p = r[1], _payload(r[4])
            st = p.get("stage")
            if not isinstance(st, str) or not st or ":" in st:
                raise ValueError("bad stage")
            if "key" not in p:  # stage row
                if ik != st or not isinstance(p.get("reason"), str) or st in out:
                    raise ValueError("bad stage row")
                out[st] = {"keys": [], "reason": p["reason"]}
                continue
            key, par = p["key"], p.get("parent")
            if st not in out or not isinstance(key, str) or not key:
                raise ValueError("key before stage")
            if "parent" in p:
                if not isinstance(par, str) or not par or ik != f"{st}:{par}.{key}":
                    raise ValueError("bad nested row")
                out[st].setdefault("nested", {}).setdefault(par, []).append(key)
            else:
                if ik != f"{st}:{key}":
                    raise ValueError("bad key row")
                out[st]["keys"].append(key)
        if isinstance(seed, dict):  # F3: a nested parent in the seed but absent from the DB is [] (none), never "free tokens"
            for st, sv in seed.items():
                nested = sv.get("nested") if isinstance(sv, dict) else None
                if st in out and isinstance(nested, dict):
                    for par in nested:
                        out[st].setdefault("nested", {}).setdefault(par, [])
        return out
    raise ValueError("unknown shape")


def _warn_once(name: str, now: float, exc_name: str) -> None:
    with _LOCK:
        last = _LOGGED.get(name)
        if last is not None and now - last < _LOG_EVERY_S:
            return
        _LOGGED[name] = now
    logger.warning("list_tables list failed: LT_LIST_ERROR list=%s (%s)", name, exc_name)


def get(name: str, seed, *, cfg: dict | None = None, env=os.environ, engine_factory=None, clock=time.monotonic):
    """DB value of list `name` in the config shape, else `seed` itself (identity) on any problem."""
    try:
        cfg = _CFG if cfg is None else cfg
        spec = cfg["lists"].get(name) if _on(cfg) else None
        if not isinstance(spec, dict):
            return seed
        snap = _snapshot(cfg, env, engine_factory, clock)
        meta = snap["meta"].get(name) if snap else None
        if meta is None or meta["shape"] != spec.get("shape"):
            return seed
        rows = snap["items"].get(name, [])
        union = spec.get("combine") == "union"
        if not rows:  # zero active items: seed unless the owner opted in per list (guards a commit race)
            return seed if union or spec.get("allow_empty") is not True or not isinstance(seed, (list, dict)) \
                else type(seed)()
        val = _build(spec, rows, seed)
        if union:
            if not isinstance(seed, list) or spec.get("shape") != "strings":
                return seed
            val = val + [s for s in seed if s not in val]
        return val
    except Exception as exc:  # noqa: BLE001  malformed row or any fault: seed for this list
        _warn_once(name, clock(), type(exc).__name__)
        return seed


def _walk(node, path):
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return False, None
        node = node[k]
    return True, node


def overlay(file_name: str, cfg_in: dict, **kw) -> dict:
    """Copy of `cfg_in` with each registered list of `file_name` replaced by get(); input never mutated."""
    out = dict(cfg_in)
    lists = (kw.get("cfg") if kw.get("cfg") is not None else _CFG).get("lists")
    for name, spec in (lists.items() if isinstance(lists, dict) else ()):
        if not isinstance(spec, dict) or spec.get("file") != file_name or not spec.get("path"):
            continue
        path = list(spec["path"])
        ok, cur = _walk(cfg_in, path)
        if not ok:
            continue
        node = out
        for k in path[:-1]:
            node[k] = dict(node[k])
            node = node[k]
        node[path[-1]] = get(name, cur, **kw)
    return out
