"""SETTINGS-W3-1110: DB-backed tunables and deny-lists over the release-pinned JSON config (ships OFF).

Precedence: code default -> JSON file (release-pinned) -> active DB row. A DB row is used only if
  1. its module's `read_from_db` flag (config/settings.json) is true (default false for every module),
  2. its key is in the hash-pinned registry (config/settings_registry.json, generated from the TUNABLE_DB manifest),
  3. it passes type, registry min/max/allowed and the one-way clamp against the file value (lower_only: db <= file,
     raise_only: db >= file), and
  4. the module's own validator (passed to `overlay`) accepts the merged config.
Anything else falls back to the file value. Access allow-lists never come from the DB: the DB holds DENY entries only
(`effective = file_allow AND NOT active unexpired db_deny`). Reads use the VERA_DB_URL_RO account (SELECT only), a 60 s
cache with single-flight background refresh and a <=500 ms statement timeout. On DB error the last good snapshot is kept
(known denies are never dropped), else file, else code defaults; warns once per window. Never raises into an ask.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = _CONFIG_DIR / "settings.json"
REGISTRY_PATH = _CONFIG_DIR / "settings_registry.json"
# Pin of the registry `keys` object (canonical JSON). Regenerate the registry and change this in the same release.
REGISTRY_SHA256 = "87e9374cfee9d6941c1140a600bb10a08d433beec8c546f67bb21e52b2dacc8e"
# Second layer: never DB-eligible, whatever the registry says.
REFUSED_STEMS = frozenset({"api_permissions", "licence_rules", "licence_allowlist", "fair_use", "purpose_grants",
                           "robots_ai", "source_permissions", "source_withdrawals", "us_federal_pd",
                           "us_gov_domains_federal", "env_precedence"})
REFUSED_PREFIXES = ("auto_acquire.landing_route.", "crawl_delay.")
SQL_SETTINGS = ("SELECT key, value, version FROM vera_vjay.set_settings WHERE state = 'active'")
SQL_DENY = ("SELECT list_name, pattern, created_at, expires_at, wildcard_confirmed FROM vera_vjay.set_deny "
            "WHERE state = 'active' AND expires_at > now()")
_DEFAULT_CFG = {"read_from_db": {}, "cache_ttl_s": 60.0, "query_timeout_ms": 500, "db_url_env": "VERA_DB_URL_RO",
                "deny_max_ttl_days": 30, "deny_max_items": 500}
_MISSING = object()

_LOCK = threading.RLock()
_ENGINES: dict = {}
_STATE: dict = {"until": 0.0, "snap": None, "busy": False}
_WARNED: dict = {}
_WARN_EVERY_S = 60.0


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cfg(path: Path | None = None) -> dict:
    try:
        raw = _load_json(path or CONFIG_PATH)
        cfg = dict(_DEFAULT_CFG)
        if isinstance(raw, dict):
            cfg.update({k: v for k, v in raw.items() if k in _DEFAULT_CFG})
        if not isinstance(cfg["read_from_db"], dict):
            cfg["read_from_db"] = {}
        return cfg
    except Exception:  # noqa: BLE001  unreadable config means everything off
        return dict(_DEFAULT_CFG)


def registry_digest(keys: dict) -> str:
    return hashlib.sha256(json.dumps(keys, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_registry(path: Path | None = None, pin: str | None = None) -> dict:
    """The key registry, or {} (nothing DB-eligible) if the file is unreadable or its hash differs from the pin."""
    try:
        raw = _load_json(path or REGISTRY_PATH)
        keys = raw["keys"]
        if not isinstance(keys, dict) or registry_digest(keys) != (pin or REGISTRY_SHA256):
            _warn_once("registry", "SET_REGISTRY_PIN_MISMATCH")
            return {}
        return {k: v for k, v in keys.items()
                if isinstance(v, dict) and v.get("module") not in REFUSED_STEMS
                and not k.startswith(REFUSED_PREFIXES)}
    except Exception as exc:  # noqa: BLE001
        _warn_once("registry", "SET_REGISTRY_ERROR " + type(exc).__name__)
        return {}


_CFG = load_cfg()
_REG = load_registry()


def reset(*, cfg: dict | None = None, registry: dict | None = None) -> None:
    """Test/boot helper: clear snapshot, warnings and engines; optionally replace config/registry."""
    global _CFG, _REG
    with _LOCK:
        _STATE.update(until=0.0, snap=None, busy=False)
        _WARNED.clear()
        _ENGINES.clear()
        if cfg is not None:
            _CFG = {**_DEFAULT_CFG, **cfg}
        if registry is not None:
            _REG = registry


def _warn_once(name: str, msg: str, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    with _LOCK:
        last = _WARNED.get(name)
        if last is not None and now - last < _WARN_EVERY_S:
            return
        _WARNED[name] = now
    logger.warning("settings: %s (%s)", msg, name)  # key/list name only, never a value


def module_enabled(stem: str) -> bool:
    return _CFG.get("read_from_db", {}).get(stem) is True


def any_enabled() -> bool:
    return any(v is True for v in _CFG.get("read_from_db", {}).values())


# ---------------------------------------------------------------------------------------------- validation

def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _coerce(reg: dict, v):
    """Value coerced to the registry type, or _MISSING when the type does not match."""
    t = reg.get("type")
    if t == "bool":
        return v if isinstance(v, bool) else _MISSING
    if t in ("int", "int_or_null"):
        if _num(v) and float(v).is_integer():
            return int(v)
        return _MISSING
    if t in ("float", "float_or_null"):
        return float(v) if _num(v) else _MISSING
    return _MISSING  # unknown type: refuse


def validate_value(reg: dict, v, file_value):
    """(ok, value). Type, registry bounds/allowed, and the one-way clamp against the file value."""
    c = _coerce(reg, v)
    if c is _MISSING:
        return False, None
    if reg.get("type") == "bool":
        allowed = reg.get("allowed")
        return (c in allowed, c) if isinstance(allowed, list) else (False, None)
    lo, hi = reg.get("min"), reg.get("max")
    if not (_num(lo) and _num(hi)) or not (lo <= c <= hi):  # numeric rows must carry both bounds
        return False, None
    clamp = reg.get("clamp")
    if clamp:
        if not _num(file_value):
            return False, None  # nothing to clamp against
        if (clamp == "lower_only" and c > file_value) or (clamp == "raise_only" and c < file_value):
            return False, None
    return True, c


# ---------------------------------------------------------------------------------------------- snapshot

def _to_dt(x):
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, str):
        d = datetime.fromisoformat(x.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    raise ValueError("not a datetime")


def _build_snapshot(rows_s, rows_d, cfg: dict, reg: dict, wall) -> dict:
    """Pure: raw DB rows -> {values:{key:(value,version)}, denies:{list:[(pattern,wild)]}, version}. Bad rows are skipped."""
    values: dict = {}
    for r in rows_s:
        try:
            key, val, ver = r[0], r[1], r[2]
            # jsonb arrives already parsed from the driver; a str here is a JSON string value and fails the type check
            if key in reg and isinstance(ver, int) and not isinstance(ver, bool):
                values[key] = (val, ver)  # per-value checks happen at read time against the live file value
            elif key not in reg:
                _warn_once("key:" + str(key)[:80], "SET_KEY_NOT_IN_REGISTRY ignored")
        except Exception:  # noqa: BLE001
            continue
    denies: dict = {}
    now = wall()
    max_ttl = float(cfg.get("deny_max_ttl_days") or 30) * 86400
    cap = int(cfg.get("deny_max_items") or 500)
    for r in rows_d:
        try:
            lst, pat, created, expires, wild = r[0], r[1], _to_dt(r[2]), _to_dt(r[3]), r[4]
            if not isinstance(lst, str) or not lst or not isinstance(pat, str) or not pat.strip() or len(pat) > 256:
                continue
            if expires.timestamp() <= now or (expires - created).total_seconds() > max_ttl:
                continue  # expired, or lives longer than the enforced TTL
            has_wild = any(c in pat for c in "*?[")
            if has_wild and wild is not True:
                continue
            bucket = denies.setdefault(lst, [])
            if len(bucket) < cap:
                bucket.append((pat.strip().lower(), has_wild))
        except Exception:  # noqa: BLE001  also drops allow-shaped rows (they lack these fields)
            continue
    ident = sorted([(k, v[1]) for k, v in values.items()]) + sorted((l, p) for l, b in denies.items() for p, _ in b)
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    return {"values": values, "denies": {l: tuple(b) for l, b in denies.items()}, "version": digest}


def _engine(cfg: dict, env, factory):
    url = (env.get(str(cfg.get("db_url_env") or "VERA_DB_URL_RO")) or "").strip()
    if not url:
        return None
    with _LOCK:
        eng = _ENGINES.get(url)
        if eng is None:
            if factory is None:
                from sqlalchemy import create_engine as factory
            eng = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                          connect_args={"connect_timeout": 1,
                                        "options": "-c statement_timeout=%d" % min(500, int(cfg.get("query_timeout_ms") or 500))})
            _ENGINES[url] = eng
    return eng


def _fetch_rows(cfg: dict, env, factory):
    eng = _engine(cfg, env, factory)
    if eng is None:
        raise LookupError("SET_NO_URL")
    if factory is None:
        from sqlalchemy import text
    else:
        def text(s):
            return s
    with eng.connect() as conn:
        s = [tuple(r) for r in conn.execute(text(SQL_SETTINGS)).fetchall()]
        d = [tuple(r) for r in conn.execute(text(SQL_DENY)).fetchall()]
    return s, d


_FETCH = None   # test hook: callable() -> (settings_rows, deny_rows)
_SPAWN = None   # test hook: callable(fn) run instead of a daemon thread
_CLOCK = time.monotonic
_WALL = time.time
_ENV = os.environ


def _refresh() -> None:
    """One refresh (runs in the background). Keeps the last good snapshot on any error."""
    err, new = None, None
    try:
        rows_s, rows_d = _FETCH() if _FETCH else _fetch_rows(_CFG, _ENV, None)
        new = _build_snapshot(rows_s, rows_d, _CFG, _REG, _WALL)
    except Exception as exc:  # noqa: BLE001
        err = type(exc).__name__
    now = _CLOCK()
    with _LOCK:
        old = _STATE["snap"]
        if err is not None:
            _STATE.update(until=now + 15.0, busy=False)  # retry sooner, keep last good (denies never dropped)
        else:
            _STATE.update(snap=new, until=now + float(_CFG.get("cache_ttl_s") or 60.0), busy=False)
    if err is not None:
        _warn_once("refresh", "SET_DB_ERROR using " + ("last-good snapshot" if old else "file/defaults") + " (" + err + ")", now)
    else:
        _log_changes(old, new)


def _log_changes(old: dict | None, new: dict) -> None:
    ov = (old or {}).get("values", {})
    for k in sorted(set(ov) | set(new["values"])):
        a, b = ov.get(k), new["values"].get(k)
        if (a and a[1]) != (b and b[1]) or (a and a[0]) != (b and b[0]):
            logger.info("settings.changed key=%s %s->%s value=%r", k, a[1] if a else "file", b[1] if b else "file",
                        b[0] if b else None)
    od, nd = (old or {}).get("denies", {}), new["denies"]
    for lst in sorted(set(od) | set(nd)):
        if od.get(lst) != nd.get(lst):
            logger.info("settings.changed deny_list=%s entries %d->%d", lst, len(od.get(lst, ())), len(nd.get(lst, ())))


def _snapshot() -> dict | None:
    """Current snapshot (maybe stale or None). Starts a single-flight background refresh when due; never blocks."""
    with _LOCK:
        if _CLOCK() < _STATE["until"] or _STATE["busy"]:
            return _STATE["snap"]
        _STATE["busy"] = True
        snap = _STATE["snap"]
    try:
        if _SPAWN:
            _SPAWN(_refresh)
        else:
            threading.Thread(target=_refresh, name="settings-refresh", daemon=True).start()
    except Exception:  # noqa: BLE001
        with _LOCK:
            _STATE["busy"] = False
    with _LOCK:  # a synchronous spawn (tests) may have just finished
        return _STATE["snap"] if _SPAWN else snap


def version() -> str | None:
    """Digest of the DB rows in force, for the request trace; None while every module flag is off or nothing loaded."""
    if not any_enabled():
        return None
    snap = _snapshot()
    return snap["version"] if snap else "file"


# ---------------------------------------------------------------------------------------------- public API

def get(stem: str, path: str, file_value):
    """Effective value of config/<stem>.json `path` (dotted): the DB value if allowed, else `file_value` itself."""
    try:
        if not module_enabled(stem):
            return file_value
        key = stem + "." + path
        reg = _REG.get(key)
        if reg is None:
            return file_value
        snap = _snapshot()
        row = snap["values"].get(key) if snap else None
        if row is None:
            return file_value
        ok, val = validate_value(reg, row[0], file_value)
        if not ok:
            _warn_once("invalid:" + key, "SET_ROW_INVALID ignored")
            return file_value
        return val
    except Exception as exc:  # noqa: BLE001
        _warn_once("get", "SET_GET_ERROR " + type(exc).__name__)
        return file_value


def _walk(cfg, parts):
    for p in parts:
        if not isinstance(cfg, dict) or p not in cfg:
            return False, None
        cfg = cfg[p]
    return True, cfg


def overlay(stem: str, cfg: dict, validate=None) -> dict:
    """`cfg` (the loaded config/<stem>.json) with accepted DB values applied; `cfg` itself when the module flag is
    off or nothing changes (identity, so flag-off behaviour is byte-identical). `validate(merged)` is the module's own
    validator; if it raises or returns False, the DB layer is dropped for the whole module."""
    try:
        if not module_enabled(stem) or not isinstance(cfg, dict):
            return cfg
        snap = _snapshot()
        if not snap:
            return cfg
        out, changed = copy.deepcopy(cfg), False
        prefix = stem + "."
        for key, (val, _ver) in snap["values"].items():
            reg = _REG.get(key)
            if reg is None or not key.startswith(prefix):
                continue
            parts = reg["path"].split(".")
            ok, fv = _walk(cfg, parts)
            if not ok:
                continue
            good, v = validate_value(reg, val, fv)
            if not good:
                _warn_once("invalid:" + key, "SET_ROW_INVALID ignored")
                continue
            if v == fv and type(v) is type(fv):
                continue
            node = out
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = v
            changed = True
        if not changed:
            return cfg
        if validate is not None:
            r = validate(out)
            if r is False:
                raise ValueError("module validator refused")
        return out
    except Exception as exc:  # noqa: BLE001  includes module validator errors: file config wins
        _warn_once("overlay:" + stem, "SET_OVERLAY_REFUSED " + type(exc).__name__)
        return cfg


def _match(item: str, denies) -> bool:
    s = str(item).strip().lower()
    for pat, wild in denies:
        if (wild and fnmatch.fnmatchcase(s, pat)) or s == pat:
            return True
    return False


def deny_entries(list_name: str) -> tuple:
    snap = _snapshot()
    return snap["denies"].get(list_name, ()) if snap else ()


def filter_allow(list_name: str, file_list, stem: str | None = None):
    """Allow-list `file_list` minus active DB denies. Can only remove; `file_list` itself when nothing applies."""
    try:
        if not isinstance(file_list, list) or not module_enabled(stem or list_name.split(".")[0]):
            return file_list
        d = deny_entries(list_name)
        if not d:
            return file_list
        return [x for x in file_list if not (isinstance(x, str) and _match(x, d))]
    except Exception as exc:  # noqa: BLE001
        _warn_once("deny:" + list_name, "SET_DENY_ERROR " + type(exc).__name__)
        return file_list


def extend_deny(list_name: str, file_denies, stem: str | None = None):
    """Deny-list `file_denies` plus active exact (non-wildcard) DB denies. Can only add denials."""
    try:
        if not isinstance(file_denies, list) or not module_enabled(stem or list_name.split(".")[0]):
            return file_denies
        extra = [p for p, wild in deny_entries(list_name) if not wild and p not in file_denies]
        return file_denies + extra if extra else file_denies
    except Exception as exc:  # noqa: BLE001
        _warn_once("deny:" + list_name, "SET_DENY_ERROR " + type(exc).__name__)
        return file_denies
