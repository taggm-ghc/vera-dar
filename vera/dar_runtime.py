"""DAR runtime corpus provider (DARINT L4 fix round 2). The only DAR entry point main.py may use (lazily).

Flag-gated and fail-closed: returns None unless config enabled AND corpus.stored_branch_enabled are JSON true,
decided BEFORE any env read or any import of dar_store / dar_corpus / sqlalchemy. Built at most once under a lock;
read-only snapshot restore; engine always disposed; failures logged by exception class name only (never the URL)
and cached with a retry TTL so a down DB is not hit on every /ask. Config: cfg["runtime"] (uncalibrated)."""
from __future__ import annotations

import logging
import os
import threading
import time

from vera import domain_registry

logger = logging.getLogger(__name__)

def _flag_on(cfg) -> bool:
    return (isinstance(cfg, dict) and cfg.get("enabled") is True
            and (cfg.get("corpus") or {}).get("stored_branch_enabled") is True)


def _reload_deadline(rt: dict, now_t: float):
    """now + reload_ttl_s, or None (never reload) when absent / non-numeric / <= 0."""
    try:
        ttl = float(rt["reload_ttl_s"])
    except (KeyError, TypeError, ValueError):
        return None
    return now_t + ttl if ttl > 0 else None


def _span_text(path):
    """The memory placeholder-span text (config memory.store.span_text) that the 007 exclusion predicate expects;
    None when unreadable. Bound as a parameter by DarReader, never hard-coded."""
    import json
    from pathlib import Path
    try:
        v = ((json.loads(Path(path).read_text()).get("memory") or {}).get("store") or {}).get("span_text")
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, str) and v else None


_MISSING = object()


def _version_interval(rt: dict):
    """Seconds between version checks (>= 0), or None when checks are off (absent / non-numeric / negative / bool)."""
    v = rt.get("version_check_s")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        return None
    return float(v)


def _engine_for(rt: dict, env, engine_factory):
    url = (env.get(str(rt.get("db_url_env") or "VERA_DB_URL_RO")) or "").strip()
    if not url:
        return None
    if engine_factory is None:
        from sqlalchemy import create_engine as engine_factory  # lazy
    return engine_factory(url, connect_args={"connect_timeout": int(rt["connect_timeout_s"]),
                                             "options": f"-c statement_timeout={int(rt['statement_timeout_ms'])}"},
                          pool_pre_ping=True, hide_parameters=True)


def _build(cfg: dict, rt: dict, env, engine_factory):
    """(corpus, version, cursor) from a full snapshot; (None, None, None) without a URL."""
    engine = _engine_for(rt, env, engine_factory)
    if engine is None:
        return None, None, None
    from vera import answer_contract
    from vera.dar_corpus import DarCorpus
    from vera.dar_store import DarReader
    try:
        kinds = (answer_contract.load_config().get("acquisition") or {}).get("non_document_text_kinds") or []
        corpus = DarCorpus(cfg)
        reader = DarReader(engine)
        try:  # taken BEFORE the snapshot so a concurrent ingest triggers a later refresh; failure/unsupported = None
            version = reader.corpus_version()
        except Exception:  # noqa: BLE001
            version = None
        bound = rt.get("snapshot_max_documents")  # optional bound on full texts loaded inside a request thread
        extra = {"max_documents": int(bound)} if isinstance(bound, (int, float)) and bound > 0 else {}
        snap = reader.load_snapshot(kinds, span_text=_span_text(answer_contract.CHAIN_CONFIG_PATH), **extra)
        cursor = snap.get("cursor") if isinstance(snap, dict) else None
        partial = snap.get("reason") if isinstance(snap, dict) and snap.get("partial") else None
        status = (corpus.restore(snap) or {}).get("status")
        if status != "restored":  # e.g. chunking_mismatch: refuse loudly (the failure is cached with a TTL), never serve a partial index
            exc = RuntimeError(f"restore status {status}")
            exc.reason = f"restore_{status}"
            raise exc
        try:
            corpus.partial_reason = partial  # surfaced by DarRuntime.status() -> dar_health (degraded), never "ok"
        except AttributeError:
            pass
        return corpus, version, cursor
    finally:
        engine.dispose()


def delta_limits(rt: dict) -> dict:
    """load_delta bound/lookback kwargs from cfg["runtime"] (D-053). Absent / invalid / non-positive keys are omitted
    (legacy behaviour: no lookback, no bound). A lookback at or above its document bound is dropped to a safe value
    (bound - 1) so re-read rows can never make every delta partial."""
    def _num(key, cast):
        v = rt.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            return None
        return cast(v)
    out = {}
    docs, assocs = _num("delta_max_documents", int), _num("delta_max_assocs", int)
    ids, secs = _num("delta_lookback_ids", int), _num("delta_lookback_s", float)
    if docs is not None:
        out["max_documents"] = docs
        if ids is not None and ids >= docs:
            ids = max(docs - 1, 0) or None
    if assocs is not None:
        out["max_assocs"] = assocs
    if ids is not None:
        out["lookback_ids"] = ids
    if secs is not None:
        out["lookback_s"] = secs
    return out


def _fetch_version_and_delta(rt: dict, env, engine_factory, cursor, want_delta: bool):
    """One short-lived engine: (version, delta|None). The delta is read only when the caller will apply it
    (cursor present, corpus supports apply_delta, version changed is decided by the caller via a callback-free
    two-step: version first, then the delta when `want_delta(version)` is true)."""
    engine = _engine_for(rt, env, engine_factory)
    if engine is None:
        raise RuntimeError("db url missing")
    from vera import answer_contract
    from vera.dar_store import DarReader
    try:
        reader = DarReader(engine)
        version = reader.corpus_version()
        if not want_delta(version):
            return version, None
        kinds = (answer_contract.load_config().get("acquisition") or {}).get("non_document_text_kinds") or []
        try:
            delta = reader.load_delta(cursor, kinds, span_text=_span_text(answer_contract.CHAIN_CONFIG_PATH),
                                      **delta_limits(rt))
        except Exception as exc:  # noqa: BLE001  (incl. a reader without load_delta): caller falls back to a full rebuild
            logger.warning("DAR delta load failed (%s)", type(exc).__name__)
            delta = None
        return version, delta
    finally:
        engine.dispose()


class DarRuntime:
    """All state per instance (lock + state dict), so independent runtimes can share one database."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {}  # corpus, version, cursor, next_at, next_check, reason, no_version_support, status_logged

    def reset_for_tests(self) -> None:
        with self._lock:
            self._state.clear()

    def _cached(self, now_t: float):
        """(hit, value). READ-ONLY (safe outside the lock). A hit is the cached corpus (or None) until its next_at
        deadline: reload_ttl_s after a success, failure_retry_ttl_s after a failure."""
        st = self._state.get("corpus", _MISSING)
        if st is _MISSING:
            return False, None
        next_at = self._state.get("next_at")
        if next_at is not None and now_t >= next_at:
            return False, st
        return True, st

    def status(self) -> str:
        """Status code only (no hosts/values/URLs); reads module state, never env/DB."""
        try:
            if not _flag_on(domain_registry.load_config()):
                return "disabled"
        except Exception:  # noqa: BLE001
            return "unavailable:" + "config_error"
        if self._state.get("corpus") is not None:
            if self._state.get("partial"):
                return "partial:" + str(self._state["partial"])
            if self._state.get("stale"):  # a refresh was refused: the OLD corpus is served and must not look current
                return "stale:" + str(self._state["stale"])
            return "ok"
        return "unavailable:" + (self._state.get("reason") or "not_built")

    def _check_due(self, rt: dict, now_t: float) -> bool:
        iv = _version_interval(rt)
        if iv is None or self._state.get("corpus") is None or self._state.get("no_version_support"):
            return False
        return now_t >= (self._state.get("next_check") or 0.0)

    def _check_version_locked(self, cfg, rt, env, engine_factory, clock):
        """Caller holds the lock. Cheap version check; a changed version is applied INCREMENTALLY (load_delta +
        corpus.apply_delta) and falls back to a full rebuild when that is impossible or fails. A failing check keeps
        the cached corpus."""
        st = self._state
        st["next_check"] = clock() + _version_interval(rt)
        corpus, cursor = st.get("corpus"), st.get("cursor")

        def want(ver):
            return ver != st.get("version") and cursor is not None and callable(getattr(corpus, "apply_delta", None))
        try:
            ver, delta = _fetch_version_and_delta(rt, env, engine_factory, cursor, want)
        except AttributeError as exc:  # reader without corpus_version: no version support, serve as before
            st["no_version_support"] = True
            logger.warning("DAR version check unsupported (%s)", type(exc).__name__)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("DAR version check failed (%s)", type(exc).__name__)
            try:
                st["next_check"] = clock() + float(rt["failure_retry_ttl_s"])
            except (KeyError, TypeError, ValueError):
                pass
            return
        if ver == st.get("version"):
            return
        if isinstance(delta, dict) and delta.get("partial"):  # a bound was hit: never apply, never advance the version
            logger.warning("DAR delta partial (%s); full rebuild", delta.get("reason"))
            delta = None
        if delta is not None:
            try:
                res = corpus.apply_delta(delta)
                if isinstance(res, dict) and res.get("status") == "applied":
                    st["version"] = ver
                    st["cursor"] = delta.get("cursor") or cursor
                    return
                logger.warning("DAR delta not applied (%s); full rebuild",
                               res.get("status") if isinstance(res, dict) else "bad_result")
            except Exception as exc:  # noqa: BLE001
                logger.warning("DAR delta apply failed (%s); full rebuild", type(exc).__name__)
        self._rebuild_locked(cfg, rt, env, engine_factory, clock)

    def get_corpus(self, env=os.environ, engine_factory=None, now=None):
        """DarCorpus or None. NEVER raises: any error is logged by exception class name only and yields None (or, when
        a reload fails, the previous corpus)."""
        try:
            return self._get_corpus(env, engine_factory, now)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DAR runtime error (%s)", type(exc).__name__)
            return self._state.get("corpus")

    def refresh(self, env=os.environ, engine_factory=None, now=None):
        """Forced FULL rebuild from the stored tables (for the full-text loader). Same failure semantics as a TTL
        reload: the old corpus is kept on failure. Returns the current corpus; never raises. None when the flag is off
        (env untouched)."""
        try:
            cfg = domain_registry.load_config()
            if not _flag_on(cfg):
                return None
            clock = now or time.monotonic
            rt = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
            with self._lock:
                self._rebuild_locked(cfg, rt, env, engine_factory, clock)
                return self._state.get("corpus")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DAR runtime refresh error (%s)", type(exc).__name__)
            return self._state.get("corpus")

    def _get_corpus(self, env, engine_factory, now):
        cfg = domain_registry.load_config()
        if not _flag_on(cfg):
            return None
        clock = now or time.monotonic
        hit, val = self._cached(clock())
        if hit:
            rt = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
            if val is not None and self._check_due(rt, clock()):
                if not self._lock.acquire(blocking=False):  # someone is already checking/rebuilding: serve cached
                    return val
                try:
                    if self._check_due(rt, clock()):
                        self._check_version_locked(cfg, rt, env, engine_factory, clock)
                    return self._state.get("corpus")
                finally:
                    self._lock.release()
            return val
        # A stale-but-working corpus is served while one thread rebuilds; a missing one waits for the build (built once).
        if val is not None:
            if not self._lock.acquire(blocking=False):
                return val
        else:
            self._lock.acquire()
        try:
            hit, val = self._cached(clock())
            if hit:
                return val
            rt = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
            self._rebuild_locked(cfg, rt, env, engine_factory, clock)
            return self._state.get("corpus")
        finally:
            self._lock.release()

    def _rebuild_locked(self, cfg, rt, env, engine_factory, clock):
        """Build a NEW corpus (full snapshot) and swap it in; caller holds the lock. On failure the previous corpus
        (if any) stays."""
        st = self._state
        old = st.get("corpus")
        try:
            corpus, version, cursor = _build(cfg, rt, env, engine_factory)
            if corpus is None and old is not None:
                exc = RuntimeError("db url missing on reload")
                exc.reason = "no_url"
                raise exc
            st["stale"] = None
            st["partial"] = getattr(corpus, "partial_reason", None) if corpus is not None else st.get("partial")
            st.update(corpus=corpus, version=version if corpus is not None else st.get("version"),
                      cursor=cursor if corpus is not None else st.get("cursor"),
                      next_at=_reload_deadline(rt, clock()) if corpus is not None else None,
                      reason=None if corpus is not None else "no_url")
            iv = _version_interval(rt)
            if iv is not None:
                st["next_check"] = clock() + iv
            self._log_status_once()
        except Exception as exc:  # noqa: BLE001  (never fails /ask; class name only, URL never logged)
            logger.warning("DAR stored corpus unavailable (%s)", type(exc).__name__)
            st["reason"] = getattr(exc, "reason", None) or type(exc).__name__
            if old is not None:
                st["stale"] = "rebuild_" + st["reason"]
            iv = _version_interval(rt)
            if iv is not None:
                try:
                    st["next_check"] = clock() + float(rt["failure_retry_ttl_s"])
                except (KeyError, TypeError, ValueError):
                    st["next_check"] = clock() + iv
            try:
                retry_at = clock() + float(rt["failure_retry_ttl_s"])
            except (KeyError, TypeError, ValueError):
                retry_at = float("inf")  # no configured TTL: do not hammer the DB; fix the config
            st.update(corpus=old, next_at=retry_at)
            self._log_status_once()

    def _log_status_once(self) -> None:
        if self._state.get("status_logged"):
            return
        self._state["status_logged"] = True
        logger.info("DAR runtime status %s", "ok" if self._state.get("corpus") is not None
                    else "unavailable:" + (self._state.get("reason") or "not_built"))


_DEFAULT = DarRuntime()
_STATE = _DEFAULT._state  # back-compat alias (same dict object; reset clears in place)
_LOCK = _DEFAULT._lock


def reset_for_tests() -> None:
    _DEFAULT.reset_for_tests()


def status() -> str:
    return _DEFAULT.status()


def get_corpus(env=os.environ, engine_factory=None, now=None):
    return _DEFAULT.get_corpus(env, engine_factory, now)


def refresh(env=os.environ, engine_factory=None, now=None):
    return _DEFAULT.refresh(env, engine_factory, now)
