"""CHUNK-TAXO-JOB-1109: in-app background job that runs the mechanical chunk taxonomy (plan v3.7, OD-1109-TAXO-AUTO,
OD-1109-DEMB-2). Pattern of vera/retention_purge.py; started by ONE guarded call in main.py.

  * flag off (config chunk_taxo.enabled is not True): start() returns at once; no thread, no engine, no connection;
  * never on a request thread (one daemon thread); never raises; logs a code and the exception class only (no chunk text, host
    or URL); exponential backoff after a failed or stopped pass (initial_delay_s doubling, capped at interval_s);
  * inert (one log line, thread ends) when the embedder is "none", the engine module is missing, the DB account is not a member
    of vera_vjay_rw (or is a member of vera_vjay_wo), or the 020/024 tables are absent;
  * schema: 020 only -> compute-only (build, quality, log; no taxonomy write) unless job.allow_curator_origin_on_020 is literally
    true; 022 present -> write mode (nodes proposed -> active in one transaction under the 022 guard);
  * progress lives in the DB: a vector row per span (tax_chunk_vectors, keyed by text sha256) committed after every batch, so a
    restart resumes; centers are rebuilt only when every vector is present; each finished pass inserts one tax_job_runs row;
  * bounded: embed batch (RSS guard: halve above rss_soft_mb, stop above rss_hard_mb), page_size, tick_row_cap, max_rows and
    max_seconds per pass, statement_timeout and lock_timeout; the model is loaded when a pass starts and released after it;
  * idempotent: every taxonomy insert is ON CONFLICT DO NOTHING.
Run state is also readable in process through status(); the DB-backed reader for the API pages is vera/chunk_taxo_status.py.
No LLM, no network. Document UNION is not stored (020 *_eff_v views); only VOTE rows are.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from types import SimpleNamespace

from vera import chunk_taxo_store as store
from vera import fulltext_loader  # module level on purpose: test_chunk_taxo_mem_1109 traces allocations from j.run() onward
from vera.chunk_taxo_embed import Deferred, EmbedError, load_embedder

log = logging.getLogger("vera.chunk_taxo")

JOB_DEFAULTS = {
    "interval_s": 3600, "initial_delay_s": 600, "step_delay_s": 2, "page_size": 500, "tick_row_cap": 2000,
    "statement_timeout_ms": 30000, "lock_timeout_ms": 5000, "connect_timeout_s": 5, "text_chars": 8000,
    "finish_now": False,  # WP-F (CHUNK-ASSOC-1113): read as a literal bool by Job.__init__, never through _num
    "embed_batch": 32, "max_rows": 100000, "max_seconds": 3000, "rss_soft_mb": 380, "rss_hard_mb": 420,
    # TAXO-ROBUST-1111: safety bounds, not decision thresholds (calibrate from the first live run's fail_total_chars / bisect metrics)
    "embed_batch_chars": 48000, "quarantine_max": 50, "quarantine_max_consecutive": 4,
}
_BOUNDS = {"interval_s": (1, 10**7), "initial_delay_s": (0, 10**7), "step_delay_s": (0, 3600), "page_size": (1, 5000),
           "tick_row_cap": (1, 100000), "statement_timeout_ms": (100, 600000), "lock_timeout_ms": (100, 600000),
           "connect_timeout_s": (1, 120), "text_chars": (50, 20000),
           "embed_batch": (1, 512), "max_rows": (1, 10**7), "max_seconds": (1, 86400), "rss_soft_mb": (1, 10**6),
           "rss_hard_mb": (1, 10**6), "embed_batch_chars": (1000, 10**6), "quarantine_max": (1, 10000),
           "quarantine_max_consecutive": (1, 1000)}
# TAXO-ROBUST-1111 E5: per-input failures that bisection may quarantine, with the int stored for each (0 = bad_response).
_QUARANTINABLE = {"http_400": 400, "http_413": 413, "http_422": 422, "bad_response": 0}
LOCK_KEY = {"k1": 0x56455241, "k2": 0x43545801}  # WP-W W0: namespaced two-int advisory key (VERA, CT 1); a safety key, not a threshold


class _WriterLost(Exception):
    """The held single-writer lock is no longer held on this session (dropped connection): no further write in this pass."""


_BUDGET_CODES = ("budget_reached", "quota_reached")  # WP-F: the Deferred codes that may start a finish-now first build
_ABBR = {"domain": "d", "topic": "t", "subtopic": "s"}
_WORKING = ("scan", "build", "w_nodes", "w_links", "w_votes", "w_retire", "finish")
TERMINAL = ("idle", "stopped")
_LABEL_NORM = re.compile(r"\s+")


_TRIM_WARNED = [False]


def malloc_trim_release(enabled: bool = True) -> bool:
    """Post-pass allocator release: gc.collect(), then glibc malloc_trim(0) (Linux/glibc only; a no-op elsewhere).
    Returns True when malloc_trim ran. Never raises; a failure is logged once per process."""
    try:
        gc.collect()
    except Exception:  # noqa: BLE001
        pass
    if not enabled or not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes  # noqa: PLC0415
        import platform  # noqa: PLC0415
        if platform.libc_ver()[0] != "glibc":
            return False
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except Exception as exc:  # noqa: BLE001
        if not _TRIM_WARNED[0]:
            _TRIM_WARNED[0] = True
            log.info("chunk_taxo malloc_trim unavailable (%s); skipped", type(exc).__name__)
        return False


def _num(jcfg: dict, key: str):
    lo, hi = _BOUNDS[key]
    v = jcfg.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi) or (isinstance(v, float) and math.isnan(v)):
        return JOB_DEFAULTS[key]
    return v


def _opt_num(v):
    """A positive finite number or None (a rebuild trigger left unset). Booleans and text are None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return None
    return float(v)


def _ceil_int(x) -> int:
    """tax_job_runs.neurons_used is INTEGER >= 0."""
    try:
        return max(0, int(math.ceil(float(x))))
    except (TypeError, ValueError, OverflowError):
        return 0


def _g(o, name, default=None):
    if isinstance(o, dict):
        return o.get(name, default)
    return vars(o).get(name, default) if hasattr(o, "__dict__") else default


_NO_SOURCE = "\0no_source"  # sentinel: the span's source id was absent from the lookup


def _source_verdicts(rows) -> dict:
    """ARXIV-DEFENSE-1113: source_id -> None (not a portal) | rule id (portal) | "error" (undecidable: fail closed).
    rows are (source_id, canonical_url, provenance) from store.fetch_source_portal. A string provenance is parsed as JSON; a
    non-dict result, a parse failure, a raising portal_verdict or "internal_error" is "error"; None is read as {} so the URL
    checks still run. Nothing here logs or keeps text."""
    out = {}
    for sid, canon, prov in rows:
        try:
            if isinstance(prov, (str, bytes)):
                prov = json.loads(prov)
            if prov is None:
                prov = {}
            if not isinstance(prov, dict):
                out[sid] = "error"
                continue
            v = fulltext_loader.portal_verdict(prov.get("item"), prov, canon)
            out[sid] = "error" if v == "internal_error" else v
        except Exception:  # noqa: BLE001
            out[sid] = "error"
    return out


def _numeric_only(x, depth=0):
    """Status metrics keep numbers and booleans only (no text of any kind)."""
    if isinstance(x, bool) or isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict) and depth < 4:
        return {str(k)[:48]: _numeric_only(v, depth + 1) for k, v in list(x.items())[:64]
                if isinstance(_numeric_only(v, depth + 1), (bool, int, float, dict))}
    return None


_SAFE_CODE = re.compile(r"^[a-z0-9_]{1,40}$")
_RT_CODES = frozenset({"embed_count_mismatch", "dim_mismatch", "bad_phase", "node_not_active",
                       "quarantine_cap_consecutive", "quarantine_cap_total"})


def _cut_token(x) -> str:
    """TAXO-REASON-1111 L5: a text-free token: charset [A-Za-z0-9_.], at most 40 characters."""
    return re.sub(r"[^A-Za-z0-9_.]", "", x if isinstance(x, str) else "")[:40]


def _reason_for(exc) -> str:
    """TAXO-REASON-1111: closed-vocabulary, text-free failure reason. Never raises (L3).
    EmbedError: 'embed:<code>' when the message is a vetted code, else 'embed:unknown'; a plain RuntimeError with a fixed
    allow-listed code: that code; anything else: the class name only (L2: the message, args, repr and any DB-error attribute
    such as orig, statement, params or pgerror are never read)."""
    try:
        if isinstance(exc, EmbedError):  # before the RuntimeError branch: EmbedError subclasses RuntimeError (L1)
            msg = str(exc)
            return "embed:" + msg if _SAFE_CODE.fullmatch(msg) else "embed:unknown"
        if type(exc) is RuntimeError:
            msg = str(exc)
            if msg in _RT_CODES:
                return msg
        return _cut_token(type(exc).__name__) or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _int_or_none(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _unit_mean(vecs):
    if not vecs:
        return None
    n = len(vecs[0])
    acc = [0.0] * n
    for v in vecs:
        for i in range(n):
            acc[i] += float(v[i])
    norm = math.sqrt(sum(x * x for x in acc))
    return [x / norm for x in acc] if norm > 0 else acc


def _clean_label(label: str, taken: set, slug: str) -> str:
    s = "".join(ch for ch in str(label or "") if ch.isprintable() and ch not in "\r\n\t").strip()[:100] or "Semantic center"
    cand, n = s, 0
    while _LABEL_NORM.sub(" ", cand.strip()).lower() in taken:
        n += 1
        cand = f"{s} ({slug[-4:]}{'' if n == 1 else n})"[:120]
    taken.add(_LABEL_NORM.sub(" ", cand.strip()).lower())
    return cand


def _anchor_payload(terms):
    clean = []
    for t in terms or []:
        t = re.sub(r"[^a-z0-9 -]", "", str(t).lower()).strip()[:48]
        if t and re.match(r"^[a-z0-9]", t) and t not in clean:
            clean.append(t)
    clean = clean[:30]
    if len(clean) < 2:
        return None
    text = ("Mechanical semantic center. Top terms: " + ", ".join(clean))[:1200]
    return text, clean


def _similarity(distance):
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d):
        return None
    return round(max(-1.0, min(1.0, 1.0 - d)), 6)


class Job:
    def __init__(self, cfg=None, core=None, embedder=None, engine_factory=None, env=None, clock=time.monotonic, wall=time.time,
                 rss_reader=None):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        jc = self.cfg.get("job")  # C1 load_cfg does not pass "job" through yet; env overrides below work regardless
        self.jcfg = jc if isinstance(jc, dict) else {}
        env = env if env is not None else os.environ
        for key, var in (("max_rows", "CHUNK_TAXO_MAX_ROWS"), ("max_seconds", "CHUNK_TAXO_MAX_SECONDS"),
                         ("rss_soft_mb", "CHUNK_TAXO_RSS_SOFT_MB"), ("rss_hard_mb", "CHUNK_TAXO_RSS_HARD_MB"),
                         ("embed_batch", "CHUNK_TAXO_EMBED_BATCH")):
            if var in env:
                try:
                    self.jcfg = dict(self.jcfg, **{key: float(env[var]) if "." in str(env[var]) else int(env[var])})
                except ValueError:
                    pass
        self.trim_on = self.jcfg.get("malloc_trim") is not False and str(env.get("CHUNK_TAXO_MALLOC_TRIM", "1")).lower() not in ("0", "false", "off", "no")
        self.core = core
        self._emb = embedder
        self._emb_injected = embedder is not None
        self._emb_reason = "ok" if embedder is not None else None
        self._factory = engine_factory
        self._env = env
        self._clock, self._wall = clock, wall
        self._rss = rss_reader or read_rss_mb
        self.on = self.cfg.get("enabled") is True
        self.step_delay = 0.0
        self.page = int(_num(self.jcfg, "page_size"))
        self.cap = int(_num(self.jcfg, "tick_row_cap"))
        self.chars = int(_num(self.jcfg, "text_chars"))
        self.batch = int(_num(self.jcfg, "embed_batch"))
        self.batch_chars = int(_num(self.jcfg, "embed_batch_chars"))
        self.q_max = int(_num(self.jcfg, "quarantine_max"))
        self.q_max_consec = int(_num(self.jcfg, "quarantine_max_consecutive"))
        self._quarantine: dict = {}  # span_id -> (sha256 of text, code); in process only, kept across passes (E6)
        self.max_rows = int(_num(self.jcfg, "max_rows"))
        self.max_seconds = float(_num(self.jcfg, "max_seconds"))
        self.rss_soft = float(_num(self.jcfg, "rss_soft_mb"))
        self.rss_hard = float(_num(self.jcfg, "rss_hard_mb"))
        self.allow_curator = self.jcfg.get("allow_curator_origin_on_020") is True
        # WP-F (CHUNK-ASSOC-1113): when the neuron budget defers embedding, finish the scan reuse-only and build/write from the
        # vectors already stored. Literal true only; a missing or non-bool key is off.
        self.finish_now = self.jcfg.get("finish_now") is True
        # WP-I (D3, D5): incremental assignment mode (off | shadow | on; anything else is off) and the optional rebuild triggers (null = off)
        self.incremental = self.jcfg.get("incremental") if self.jcfg.get("incremental") in ("off", "shadow", "on") else "off"
        self.rebuild_unassigned_share = _opt_num(self.jcfg.get("rebuild_unassigned_share"))
        self.rebuild_interval_s = _opt_num(self.jcfg.get("rebuild_interval_s"))
        self.engines_created = 0
        self.rows_embedded = 0
        self.peak_rss = 0.0
        self.build_estimate_mb = 0.0
        self._t0 = 0.0
        self._stop = None
        self._lock = threading.Lock()
        self._eng = None
        self._wconn = None
        self._held, self._hcm = None, None  # WP-W W0: the pass's single held connection (pool of one) and its context manager
        self._w_started = self._w_done = False  # WP-W: W1 begun, W4 committed (write_incomplete when started and not done)
        self._run_id = None
        self._pass_neurons = 0.0
        self._fin = None
        self._phase = "start"
        self._fail_span, self._fail_batch = None, None  # TAXO-REASON-1111: the batch being embedded (ints only)
        self._mode = None
        self._inert = None
        self._reset_run()
        self._status = {"state": "off" if not self.on else "pending", "version": None, "metrics": {}, "display_ok": False,
                        "reasons": [], "last_run": None}
        self._ids: dict = {}

    def status(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["metrics"] = _numeric_only(s["metrics"]) or {}
            s["reasons"] = [str(r)[:64] for r in s["reasons"]][:16]
            return s

    def _set(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)

    def run(self) -> dict:
        """One bounded, resumable pass. Never raises. Returns {state, complete, rows_embedded, ...} (counts only)."""
        if not self.on:
            return {"state": "off", "complete": False, "rows_embedded": 0}
        try:
            return self._run_pass()
        finally:
            self._post_pass_release()  # every exit: normal, early stop, error

    def _post_pass_release(self) -> None:
        """Drop leftover per-pass references, then return freed allocator memory to the OS (RSS stays flat between passes)."""
        try:
            self._release_buffers()
            self._vote_rows, self._retire_keys, self._fin = [], [], None
        except Exception:  # noqa: BLE001
            pass
        malloc_trim_release(self.trim_on)

    def _run_pass(self) -> dict:
        try:
            self._t0 = self._wall()
            self.rows_embedded, self._stop, self._phase, self._inert = 0, None, "start", None
            self._fail_span, self._fail_batch = None, None
            self._reset_run()
            t_end = self._clock() + self.max_seconds
            while self._phase not in TERMINAL and not self._inert and self._stop is None:
                if self._clock() >= t_end:
                    self._stop = "max_seconds"
                    break
                try:
                    self._step()
                except Deferred as d:  # quota or neuron budget: resume after 00:00 UTC, not a failure
                    self._stop, self._retry_in = "deferred", d.retry_in_s
                    break
                except _WriterLost:  # WP-W W0: the session lock is gone: abandon the pass before any further write
                    self._stop = "writer_lost"
                    self._abandon_held()
                    break
                except Exception as exc:  # noqa: BLE001
                    reason = _reason_for(exc)
                    info = self._fail_info()  # before the buffers and the engine are dropped
                    info.update(self._robust_info())
                    info.update(self._scan_info())
                    self._close_run("failed", reason=reason, extra=info)
                    self._release_buffers()
                    self._drop_engine()
                    self._set(state="error", reasons=[reason[:64]])
                    self._log_failure(reason, info)
                    return {"state": "error", "complete": False, "rows_embedded": self.rows_embedded}
            if self._inert:
                self._record_inert(self._inert)  # after the probe connection is closed (pool of one)
            if self._run_id is not None:
                self._close_run("deferred", reason=self._defer_reason, extra={**self._robust_info(), **self._scan_info()})
            self._release_buffers()  # early stop or end of pass: no chunk or vector buffer survives it
            self._drop_engine()
            self._release_embedder()
            if self._stop == "deferred":
                self._set(state="deferred", reasons=["deferred_until_utc_midnight"] + ([self._defer_reason] if self._defer_reason else []))
                log.info("chunk_taxo run deferred until 00:00 UTC rows_embedded=%d", self.rows_embedded)
            elif self._stop:
                self._set(state="partial", reasons=[self._stop])
                log.info("chunk_taxo run stopped: %s rows_embedded=%d", self._stop, self.rows_embedded)
            st = self.status()
            out = {"state": st["state"], "complete": self._phase == "idle" and not self._stop and not self._inert,
                   "rows_embedded": self.rows_embedded, "stop": self._stop, "retry_in_s": self._retry_in}
            if self._reuse_only and self.finish_now:  # WP-F: the budget deferred embedding; the loop waits for the budget reset, not the interval
                out["finish_now"] = True
            return out
        except Exception as exc:  # noqa: BLE001
            reason = _reason_for(exc)
            try:
                self._set(state="error", reasons=[reason[:64]])
            except Exception:  # noqa: BLE001
                pass
            self._log_failure(reason, None)
            return {"state": "error", "complete": False, "rows_embedded": self.rows_embedded}

    def _fail_info(self) -> dict:
        """TAXO-REASON-1111: integers only (no span text, no hash of it): where and with what batch shape the pass failed.
        Never raises."""
        out: dict = {}
        try:
            out["fail_rows_embedded"] = int(self.rows_embedded)
            sid = _int_or_none(self._fail_span)
            if sid is not None:
                out["fail_span_id"] = sid
            fb = self._fail_batch
            if isinstance(fb, dict):
                for k, v in fb.items():
                    if _int_or_none(v) is not None:
                        out[str(k)] = v
            st = getattr(self._emb, "stats_run", None)
            if isinstance(st, dict):
                for k in ("last_http_status", "attempts"):
                    if _int_or_none(st.get(k)) is not None:
                        out[k] = st[k]
        except Exception:  # noqa: BLE001
            pass
        return out

    def _robust_info(self, final: bool = False) -> dict:
        """TAXO-ROBUST-1111 E6: ints only (no text, sha or message). Empty when nothing was quarantined, skipped or bisected, so a
        clean pass stores what it stored before. Never raises."""
        try:
            fn = ({"no_vector_yet": int(self.no_vector_yet), "no_vector_yet_min_span_id": int(self._nvy_min or 0), "stale_vector": int(self.stale_vector)}
                  if self._reuse_only else {})
            if self._w_started and not self._w_done and not final:  # W1 committed, W4 not: the next build rebuilds from the mixed state
                fn["write_incomplete"] = 1
            if not (self._quarantine or self.skipped_empty or self.vectors_missing or self.bisect_requests or self._consecutive
                    or self.skipped_portal or self.skipped_portal_error or self.skipped_no_source):
                return fn
            q = {"q%d" % sid: int(_QUARANTINABLE.get(v[1], 0)) for sid, v in list(self._quarantine.items())[:64]}
            return {**fn, "quarantined": len(self._quarantine), "skipped_empty": int(self.skipped_empty),
                    "skipped_portal": int(self.skipped_portal), "skipped_portal_error": int(self.skipped_portal_error),
                    "skipped_no_source": int(self.skipped_no_source),
                    "vectors_missing": int(self.vectors_missing), "q_consecutive": int(self._consecutive),
                    "q_last_status": int(self._q_last), "bisect_requests": int(self.bisect_requests),
                    "bisect_failed_chars": int(self.bisect_failed_chars), "q": q}
        except Exception:  # noqa: BLE001
            return {}

    def _scan_info(self) -> dict:
        """WP-I D6 run-row contract (confirmed from WP-C's proposal, work/WP-FWI-1113/NOTES.md): numbers only, written on EVERY close
        (finished, deferred, partial, failed). A pass that stopped mid-scan counted only the pages it read, so `scan_complete` 0 marks
        lower bounds. Never raises."""
        try:
            out = {"scan_complete": 1 if self._scan_done else 0, "skipped_portal": int(self.skipped_portal),
                   "skipped_portal_error": int(self.skipped_portal_error), "skipped_no_source": int(self.skipped_no_source),
                   "skipped_empty": int(self.skipped_empty)}
            for cls, n in self._excl_vec.items():
                out["excluded_with_vector_" + cls] = int(n)
            out["excluded_with_vector_empty"] = 0  # a vectored empty span is never classed empty (the vector check comes first)
            if self._no_rebuild:
                out["no_rebuild_snapshot_incomplete"] = 1
            return out
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _batch_shape(part) -> dict:
        return {"fail_batch_n": len(part), "fail_min_chars": min(len(r[2].strip()) for r in part),
                "fail_max_chars": max(len(r[2]) for r in part), "fail_total_chars": sum(len(r[2]) for r in part),
                "fail_nonascii_chars": sum(1 for r in part for ch in r[2] if ord(ch) > 127)}

    def _refresh_neurons(self) -> None:
        st = getattr(self._emb, "stats_run", None)
        if isinstance(st, dict):
            self._pass_neurons = float(st.get("neurons", 0.0))

    def _embed_isolating(self, part, shas, out) -> None:
        """E5: embed `part`; on a quarantinable per-input failure bisect, quarantining single spans. Fills `out` (span_id -> vector)
        with the spans that embedded, also when a cap or another error is raised partway (the caller stores them).
        Caps raise a plain RuntimeError with a fixed code. _pass_neurons is refreshed even on a raise."""
        try:
            self._bisect(part, shas, out, top=True)
        finally:
            self._refresh_neurons()

    def _bisect(self, part, shas, out, top=False) -> None:
        self._fail_span, self._fail_batch = part[0][0], self._batch_shape(part)
        if not top:
            self.bisect_requests += 1
        try:
            vecs = self._emb.embed([r[2] for r in part])
        except EmbedError as exc:
            code = str(exc)
            if code not in _QUARANTINABLE:
                raise
            self.bisect_failed_chars += sum(len(r[2]) for r in part)
            if len(part) > 1:
                mid = len(part) // 2
                self._bisect(part[:mid], shas, out)
                self._bisect(part[mid:], shas, out)
                return
            sid = part[0][0]
            self._consecutive += 1
            self._q_last = _QUARANTINABLE[code]
            if self._consecutive >= self.q_max_consec:
                for s in self._streak:  # not shown to be per-input faults: release them
                    self._quarantine.pop(s, None)
                self._streak = []
                raise RuntimeError("quarantine_cap_consecutive") from None
            if len(self._quarantine) >= self.q_max:
                raise RuntimeError("quarantine_cap_total") from None
            self._quarantine[sid] = (shas[sid], code)
            self._streak.append(sid)
            return
        if len(vecs) != len(part):
            raise RuntimeError("embed_count_mismatch")
        self._consecutive, self._streak = 0, []
        for r, v in zip(part, vecs):
            out[r[0]] = v

    def _log_failure(self, reason: str, info) -> None:
        """One log line, ints and closed tokens only (L4: no exc_info, no stack_info, no log.exception). Never raises."""
        try:
            info = info if isinstance(info, dict) else self._fail_info()
            sid = info.get("fail_span_id")
            log.warning("chunk_taxo job failed: CT_JOB_ERROR reason=%s phase=%s rows_embedded=%d span_id=%s batch_n=%d "
                        "last_http_status=%d attempts=%d", re.sub(r"[^A-Za-z0-9_.:]", "", str(reason))[:64], _cut_token(self._phase),
                        int(self.rows_embedded),
                        ("%d" % sid) if _int_or_none(sid) is not None else "n/a", int(info.get("fail_batch_n", 0)),
                        int(info.get("last_http_status", 0)), int(info.get("attempts", 0)))
        except Exception:  # noqa: BLE001
            pass

    def _close_run(self, state: str, reason=None, extra=None) -> None:
        """Best effort: mark an unfinished pass's running row (keeps its neurons_used). Never raises."""
        rid, self._run_id = self._run_id, None
        if rid is None:
            return
        try:
            with self._cx(True) as conn:
                store.run_sql(conn, "job_run_close", {"run_id": rid, "state": state, "neurons": _ceil_int(self._pass_neurons),
                                                      "metrics": json.dumps(_numeric_only(extra) or {}),
                                                      "reasons": [str(reason or self._stop or state)[:64]]})
        except Exception:  # noqa: BLE001
            pass

    def _release_embedder(self) -> None:
        """The model is loaded when a run starts and released when it ends (the next run loads it again)."""
        emb, self._emb = self._emb, None
        if self._emb_injected:
            self._emb = emb
            return
        try:
            if emb is not None and hasattr(emb, "close"):
                emb.close()
        except Exception:  # noqa: BLE001
            pass
        del emb
        import gc  # noqa: PLC0415
        gc.collect()

    def _go_inert(self, code: str) -> None:
        self._inert = code
        self._drop_engine()
        self._set(state="inert", reasons=[code])
        log.warning("chunk_taxo job inert: %s", code)

    def _record_inert(self, code: str) -> None:
        """Self-report: at most one closed 'failed' tax_job_runs row per process start (rw app path). Never raises."""
        global _INERT_ROW_DONE
        with _GUARD_ROW:
            if _INERT_ROW_DONE:
                return
            _INERT_ROW_DONE = True
        try:
            with self._cx(True) as conn:
                store.run_sql(conn, "job_run_inert", {"reasons": [str(code)[:64]]})
        except Exception as exc:  # noqa: BLE001
            log.warning("chunk_taxo inert run row not written: CT_INERT_ROW_ERROR (%s)", type(exc).__name__)
        finally:
            self._drop_engine()

    def _drop_engine(self) -> None:
        self._release_held()  # W0: unlock and return the connection before the engine goes
        eng, self._eng = self._eng, None
        if eng is not None:
            try:
                eng.dispose()
            except Exception:  # noqa: BLE001
                pass

    def _engine(self):
        if self._eng is None:
            from vera.db import get_role_engine  # noqa: PLC0415
            factory = self._factory
            if factory is None:
                from sqlalchemy import create_engine  # noqa: PLC0415
                factory = create_engine
            ct, st, lt = int(_num(self.jcfg, "connect_timeout_s")), int(_num(self.jcfg, "statement_timeout_ms")), int(_num(self.jcfg, "lock_timeout_ms"))

            def make(url, **kw):
                return factory(url, pool_size=1, max_overflow=0, **kw,
                               connect_args={"connect_timeout": ct, "options": f"-c statement_timeout={st} -c lock_timeout={lt}"})
            self._eng = get_role_engine("rw", env=self._env, engine_factory=make)
            self.engines_created += 1
        return self._eng

    def _release_buffers(self) -> None:
        """Drop chunk texts, vector blocks and the pending model: nothing large stays on self between passes."""
        self._chunks, self._vecs, self._pending = [], [], None
        self._dims = set()

    def _guard(self) -> bool:
        """RSS hard guard. On a trip: stop the pass cleanly (run() then releases every buffer) and report True."""
        if self._read_rss() > self.rss_hard:
            self._stop = "rss_hard"
            self._release_buffers()
            return True
        return False

    def _headroom_short(self) -> bool:
        """Pre-build headroom: stop (partial, reason rss_build, buffers released) when current RSS + the core's estimate of the
        build's extra memory would exceed rss_hard_mb. The estimate comes from n, d, k_max and the text size."""
        n = len(self._chunks)
        d = int(self._vecs[0].shape[1]) if self._vecs else 0
        params = self.cfg.get("params") if isinstance(self.cfg.get("params"), dict) else {}
        est_fn = getattr(self.core, "estimate_build_bytes", None)
        if est_fn is None:
            from vera import chunk_taxo as _ct  # noqa: PLC0415
            est_fn = _ct.estimate_build_bytes
        k_max = min(int(params.get("k_max", 40)), max(1, n // max(1, int(params.get("min_cluster_size", 3)))))
        est_mb = est_fn(n, d, k_max, sum(len(c["text"]) for c in self._chunks)) / 1048576
        self.build_estimate_mb = round(est_mb, 1)
        if self._read_rss() + est_mb > self.rss_hard:
            self._stop = "rss_build"
            self._release_buffers()
            log.info("chunk_taxo build deferred: rss + estimate %.0f MB over limit %.0f MB (n=%d d=%d)", est_mb, self.rss_hard, n, d)
            return True
        return False

    def _count_unvectored(self, rows, have=None) -> None:
        """WP-F / WP-I D7: rows (span_id, source_id, text) that stay without a CURRENT vector this pass (not empty, not quarantined). A row
        that still holds a stored vector of another model or text is `stale_vector` (the plan's class), the rest are `no_vector_yet`."""
        rows = list(rows)
        stale = [r for r in rows if have and r[0] in have]
        rows = [r for r in rows if not (have and r[0] in have)]
        self.stale_vector += len(stale)
        self.no_vector_yet += len(rows)
        if rows:
            lo = min(int(r[0]) for r in rows)
            self._nvy_min = lo if self._nvy_min is None else min(self._nvy_min, lo)

    def _active_build(self) -> bool:
        """WP-F: True when an active induction node of this job exists (the _load_prior filter). Read-only; an error counts as a build
        (fail closed: no first-build exception) and is not raised."""
        try:
            with self._cx() as conn:
                for lvl in store.LEVELS:
                    for n in store.fetch_nodes(conn, lvl):
                        if n["origin"] == "induction" and (n["origin_ref"] or "").startswith(store.ORIGIN_REF_PREFIX) and n["state"] == "active":
                            return True
            return False
        except Exception:  # noqa: BLE001
            return True

    def _reset_run(self) -> None:
        self._release_buffers()
        self._cursor = 0
        self._scan_done = False
        self._w_pos = 0
        self._v_pos = 0
        self._vote_rows = []
        self._retire_keys = []
        self._run_new_nodes = 0
        self._run_id = None
        self._pass_neurons = 0.0
        self.metrics_run = {}
        self._retry_in = None
        self._reuse_only, self.no_vector_yet, self._nvy_min, self._defer_reason = False, 0, None, None  # WP-F: per pass
        self.stale_vector = 0  # WP-I D7: spans whose stored vector is not current (other model or changed text) and were not re-embedded
        self.skipped_empty, self.vectors_missing = 0, 0  # TAXO-ROBUST-1111: per pass
        self.skipped_portal, self.skipped_portal_error, self.skipped_no_source = 0, 0, 0  # ARXIV-DEFENSE-1113: per pass
        self._verdicts = {}
        self._excl = {"portal": [], "portal_error": [], "no_source": [], "empty": []}  # WP-W: span ids the scan positively excluded
        self._scan_end, self._snap, self._prior_linked = None, {}, set()
        self._excl_vec = {"portal": 0, "portal_error": 0, "no_source": 0}  # WP-I D6: excluded spans that still hold a stored vector
        self._no_rebuild, self._inc, self._shadow, self._ctx_version, self._prior_ids = False, {}, None, None, {}
        self._w_started = self._w_done = False
        self.bisect_requests, self.bisect_failed_chars = 0, 0
        self._consecutive, self._streak, self._q_last = 0, [], 0

    def _step(self):
        ph = self._phase
        fn = {"start": self._do_start, "scan": self._do_scan, "build": self._do_build, "write": self._do_write,
              "assign": self._do_assign, "w_nodes": self._do_nodes, "w_links": self._do_links, "w_votes": self._do_votes, "w_retire": self._do_retire,
              "finish": self._do_finish}.get(ph)
        if fn is None:
            raise RuntimeError("bad_phase")
        return fn()

    def _do_start(self):
        if self._emb is None:
            ecfg = self.cfg if self.cfg.get("embedder") else dict(self.cfg, embedder=self._env.get("CHUNK_TAXO_EMBEDDER") or "none")
            self._emb, self._emb_reason = load_embedder(ecfg, env=self._env)
        if self._emb is None:
            self._go_inert(self._emb_reason or "embedder_none")
            return
        if self.core is None:
            try:
                from vera import chunk_taxo as core  # noqa: PLC0415
                self.core = core
            except Exception:  # noqa: BLE001
                self._go_inert("engine_missing")
                return
        with self._cx() as conn:
            ok, why = store.probe_account(conn)
            if not ok:
                self._go_inert(why)
                return
            has020, has022, has024 = store.probe_schema(conn)
        if not has020:
            self._go_inert("schema_020_missing")
            return
        if not has024:
            self._go_inert("schema_024_missing")
            return
        self._mode = "write" if has022 else ("curator" if self.allow_curator else "compute")
        if not self._acquire():  # W0: another job process holds the writer lock: end the pass before any embedding or run row
            self._stop = "writer_busy"
            return
        with self._cx(True) as conn:
            self._run_id = store.run_start(conn)
        self._phase = "scan"
        self._set(state="scanning")

    def _read_rss(self) -> float:
        try:
            v = float(self._rss())
        except Exception:  # noqa: BLE001
            v = 0.0
        self.peak_rss = max(self.peak_rss, v)
        return v

    def _do_scan(self):
        """Read a page of spans, reuse stored vectors whose text sha256 and model match, embed the rest in small batches and
        commit each batch (progress = rows in tax_chunk_vectors). Stops cleanly on the row, time or RSS bounds."""
        import numpy as np  # noqa: PLC0415
        model_name = str(getattr(self._emb, "name", None) or self.cfg.get("embedder_model") or "embedder")
        read = 0
        while read < self.cap:
            if self._guard():  # per page, so the reuse path (all vectors stored) is guarded too
                return
            lim = min(self.page, self.cap - read)
            with self._cx() as conn:
                page = store.fetch_spans(conn, self._cursor, lim, self.chars)
                # ARXIV-DEFENSE-1113: portal spans are never embedded or labelled; the page (not rows) drives the cursor and end test.
                # A failed lookup raises into the job's error path before anything is sent.
                new_ids = sorted({r[1] for r in page if r[1] not in self._verdicts})
                if new_ids:
                    self._verdicts.update(_source_verdicts(store.fetch_source_portal(conn, new_ids)))
                n0 = {c: len(ids) for c, ids in self._excl.items()}
                rows = []
                for r in page:
                    v = self._verdicts.get(r[1], _NO_SOURCE)
                    if v is None:
                        rows.append(r)
                    elif v == _NO_SOURCE:
                        self.skipped_no_source += 1
                        self._excl["no_source"].append(r[0])
                    elif v == "error":
                        self.skipped_portal_error += 1
                        self._excl["portal_error"].append(r[0])
                    else:
                        self.skipped_portal += 1
                        self._excl["portal"].append(r[0])
                have = store.fetch_vectors(conn, [r[0] for r in rows])
                new_excl = {c: ids[n0[c]:] for c, ids in self._excl.items() if c != "empty"}
                if any(new_excl.values()):  # D6: an excluded span may still hold a stored vector (B8); counted once per class
                    held = store.fetch_vectors(conn, [i for ids in new_excl.values() for i in ids])
                    for c, ids in new_excl.items():
                        self._excl_vec[c] += sum(1 for i in ids if i in held)
            shas ={r[0]: hashlib.sha256(r[2].encode("utf-8")).hexdigest() for r in rows}
            cur = {sid for sid, h in have.items() if h[0] == model_name and h[1] == shas.get(sid)}
            todo = []
            for r in rows:
                if r[0] in cur:
                    continue
                if r[2].strip() == "":  # E3: an empty or whitespace span is never sent (free, deterministic on every pass)
                    self.skipped_empty += 1
                    self._excl["empty"].append(r[0])
                    continue
                qd = self._quarantine.get(r[0])
                if qd is not None:
                    if qd[0] == shas[r[0]]:
                        continue  # quarantined and unchanged
                    del self._quarantine[r[0]]  # text changed: retry
                todo.append(r)
            if self._reuse_only:  # WP-F: budget spent earlier in this pass; embed nothing, count what still lacks a vector
                self._count_unvectored(todo, have)
                todo = []
            fresh = {}
            i = 0
            while i < len(todo):
                if self.rows_embedded >= self.max_rows:
                    self._stop = "max_rows"
                    return
                if self._guard():
                    return
                rss = self._read_rss()
                if rss > self.rss_soft and self.batch > 1:
                    self.batch = max(1, self.batch // 2)
                # E4: count ceiling (embed_batch, the embedder's batch_limit, rows left), then a character budget (one oversize row goes alone)
                cnt = min(self.batch, self.max_rows - self.rows_embedded)
                bl = getattr(self._emb, "batch_limit", None)
                if isinstance(bl, int) and not isinstance(bl, bool) and bl > 0:
                    cnt = min(cnt, bl)
                part, tot = [], 0
                for r in todo[i:i + cnt]:
                    if part and tot + len(r[2]) > self.batch_chars:
                        break
                    part.append(r)
                    tot += len(r[2])
                if hasattr(self._emb, "set_day_used"):  # the day's neurons from the DB (survives a restart), not process memory
                    with self._cx() as conn:
                        self._emb.set_day_used(store.neurons_today(conn))
                got, err = {}, None
                try:
                    self._embed_isolating(part, shas, got)  # sets _fail_span/_fail_batch to the sub-batch that failed
                except Exception as exc:  # noqa: BLE001  (Deferred, caps, other errors): keep what already embedded, then re-raise
                    err = exc
                recs = [{"span_id": r[0], "model": model_name, "dim": len(got[r[0]]), "sha": shas[r[0]],
                         "vec": [float(x) for x in got[r[0]]]} for r in part if r[0] in got]
                try:
                    with self._cx(True) as conn:
                        if recs:
                            store.run_sql(conn, "vectors_upsert", recs)
                        self._refresh_neurons()
                        if self._run_id is not None:
                            store.run_sql(conn, "job_run_neurons", {"run_id": self._run_id, "neurons": _ceil_int(self._pass_neurons)})
                except Exception:  # noqa: BLE001
                    if err is None:
                        raise
                    recs = []  # the original error is the reason; the partial result could not be stored
                if err is not None:
                    self.rows_embedded += len(recs)
                    if isinstance(err, Deferred) and err.code in _BUDGET_CODES:
                        # D2 (WP-F's mechanism, generalised): keep what embedded, embed nothing more this pass, finish the scan on stored
                        # vectors; what the pass then does (first build under finish_now, incremental assignment, or nothing) is decided
                        # at the end of the scan by _after_budget_scan
                        for rec in recs:
                            fresh[rec["span_id"]] = np.asarray(rec["vec"], dtype=np.float32)
                        self._reuse_only, self._retry_in = True, err.retry_in_s
                        self._count_unvectored([r for r in todo if r[0] not in fresh], have)
                        break
                    raise err
                for rec in recs:
                    fresh[rec["span_id"]] = np.asarray(rec["vec"], dtype=np.float32)
                self.rows_embedded += len(recs)
                del recs, got
                self._fail_span, self._fail_batch = None, None
                i += len(part)
                self._read_rss()
            kept = [r for r in rows if r[0] in fresh or r[0] in cur]  # E7: rows with a vector this pass
            self.vectors_missing += len(rows) - len(kept)
            if kept:  # one float32 block per page; the whole set is joined once at build (never a list of float lists)
                first = fresh[kept[0][0]] if kept[0][0] in fresh else have[kept[0][0]][2]
                blk = np.zeros((len(kept), len(first)), dtype=np.float32)
                for k, (sid, src, txt) in enumerate(kept):
                    self._chunks.append({"span_id": sid, "source_id": src, "text": txt})
                    v = fresh[sid] if sid in fresh else have[sid][2]
                    self._dims.add(len(v))
                    if len(v) == blk.shape[1]:
                        blk[k] = v
                self._vecs.append(blk)
            del have, fresh
            if page:
                self._cursor = page[-1][0]
                read += len(page)
            if len(page) < lim:
                self._scan_done = True
                break
        if self._scan_done:
            with self._cx() as conn:  # the database clock, so it compares with tax_chunk_vectors.updated_at (WP-W, D3 inputs)
                row = store.run_sql(conn, "db_now").first()
            self._scan_end = float(row[0]) if row is not None and row[0] is not None else None
            if self.skipped_portal or self.skipped_portal_error or self.skipped_no_source:
                log.info("chunk_taxo scan: skipped portal=%d portal_error=%d no_source=%d", self.skipped_portal,
                         self.skipped_portal_error, self.skipped_no_source)
            if len(self._dims) > 1:
                raise RuntimeError("dim_mismatch")  # a model/dim change leaves mixed vectors: refuse, re-embed on the next pass
            if self._reuse_only:
                return self._after_budget_scan()
            if not self._chunks:
                self._set(state="idle", reasons=["no_chunks"], last_run=self._wall())
                self._phase = "idle"
                return
            self._phase = "build"

    def _after_budget_scan(self):
        """D2: the budget stopped embedding and the scan finished reuse-only; the snapshot is incomplete (some eligible span has no
        vector). First build: only under finish_now (WP-F) and with a vector to build on; otherwise no build until the snapshot is complete.
        With a build present: mode `on` assigns the new spans incrementally and does not rebuild; `off` and `shadow` rebuild only on a
        complete snapshot, so they write nothing and record `no_rebuild_snapshot_incomplete`."""
        if not self._active_build():
            if self.finish_now and self._chunks:
                self._phase = "build"
                return
            self._stop = "deferred"
            if not self._chunks and self.finish_now:
                self._defer_reason = "no_vectors"
            return
        if self.incremental == "on" and self._chunks and self._mode == "write":
            self._phase = "assign"
            return
        self._no_rebuild = True
        self._stop = "deferred"

    def _matrix(self):
        import numpy as np  # noqa: PLC0415
        blocks, self._vecs = self._vecs, []  # blocks are freed as soon as they are joined
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def _do_build(self):
        core = self.core
        if self._guard():  # before the clustering step
            return 0.0
        if self.incremental == "on" and self._mode == "write" and not self._reuse_only and self._chunks and self._try_incremental():
            return 0.0  # D3: assigned the new spans on the last build's centers; no rebuild this pass
        if self._headroom_short():  # the build adds about 8*n*d bytes more: do not start it without room (rss_hard_mb)
            return 0.0
        params = self.cfg.get("params") if isinstance(self.cfg.get("params"), dict) else self.cfg
        chunks = [(c["span_id"], c["source_id"], c["text"]) for c in self._chunks]  # C1 contract: (span_id, source_id, text)
        self._snap = {"max_span_id": max(int(c[0]) for c in chunks), "n_spans": len(chunks), "n_vectors": len(chunks)}
        vf = getattr(core, "version_for", None)  # the version hashes params, seeds and span ids: known before the build
        if self._mode == "write" and vf is not None and self._unchanged(str(vf([c[0] for c in chunks], params))):
            return self._finish_unchanged(str(vf([c[0] for c in chunks], params)))
        if self.incremental == "shadow" and self._mode == "write" and not self._reuse_only:
            self._shadow_compute()  # in memory only: no link is written by the shadow step
        vecs = self._matrix()
        prior = self._load_prior() if self._mode in ("write", "curator") else None
        model = core.build(chunks, vecs, params, prior=prior)
        if self._mode == "write" and vf is None and self._unchanged(str(_g(model, "version"))):
            return self._finish_unchanged(str(_g(model, "version")))
        metrics = core.quality(model, vecs)
        ok, reasons = core.display_gate(metrics, self.cfg.get("display_gate"))
        self._pending = SimpleNamespace(model=model, metrics=metrics, ok=bool(ok), reasons=list(reasons or []))
        self._set(version=str(_g(model, "version")), metrics=_numeric_only(metrics) or {}, display_ok=bool(ok),
                  reasons=[str(r)[:64] for r in (reasons or [])][:16])
        log.info("chunk_taxo built version=%s chunks=%d display_ok=%s", str(_g(model, "version"))[:32], len(self._chunks), bool(ok))
        if self._mode == "compute":
            self._pending_done = "compute_only"
            self._phase = "finish"
            return 0.0
        self._phase = "write"
        self._set(state="writing")
        return 0.0

    # ---- no-op rule (WP-W, D1): an unchanged corpus writes nothing and records `unchanged`
    def _unchanged(self, version: str) -> bool:
        """True when the latest finished build row (the last build that reached W4: its run row is inserted in W4's own transaction)
        has this version AND no vector was written after that build's scan end (a re-embedded span under the same id forces a rebuild).
        Any missing input is False: build and write as before."""
        try:
            with self._cx() as conn:
                row = store.run_sql(conn, "job_run_build_latest").first()
                if row is None or str(row[0]) != version:
                    return False
                m = row[1]
                if isinstance(m, (str, bytes)):
                    m = json.loads(m)
                end = m.get("scan_end_epoch") if isinstance(m, dict) else None
                if isinstance(end, bool) or not isinstance(end, (int, float)):
                    return False
                ve = store.run_sql(conn, "vectors_epoch").first()
                if ve is None or ve[0] is None or float(ve[0]) > float(end):
                    return False
                if self._active_node_counts(conn) != {lv: m.get("n_nodes_" + lv) for lv in store.LEVELS}:
                    return False  # active nodes beyond the build's own (an interrupted write: two versions active, or a killed pass): rebuild
                return not self._write_incomplete(conn)  # and a closed row of an aborted pass after that build: rebuild
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _as_dict(m):
        if isinstance(m, (str, bytes)):
            try:
                m = json.loads(m)
            except ValueError:
                return {}
        return m if isinstance(m, dict) else {}

    def _active_node_counts(self, conn) -> dict:
        """Active induction nodes of this job per level (the _load_prior filter). After a completed W4 this equals the build's own
        count: reused nodes keep their original origin_ref, so counting versions by origin_ref would be wrong; counting nodes is not."""
        out = {}
        for lvl in store.LEVELS:
            out[lvl] = sum(1 for n in store.fetch_nodes(conn, lvl) if n["origin"] == "induction"
                           and (n["origin_ref"] or "").startswith(store.ORIGIN_REF_PREFIX) and n["state"] == "active")
        return out

    def _write_incomplete(self, conn) -> bool:
        """True when a pass aborted between W1 and W4 after the latest build row (its run row carries `write_incomplete`). The newest run
        rows are read newest first up to the build row; the running row of this pass is skipped."""
        for state, mm in store.run_sql(conn, "job_run_recent").fetchall():
            mm = self._as_dict(mm)
            if str(state) == "running":
                continue
            if "scan_end_epoch" in mm:
                return False
            if mm.get("write_incomplete") or self._as_dict(mm.get("run")).get("write_incomplete"):
                return True
        return False

    def _finish_unchanged(self, version: str):
        """Close the pass's running row as finished with `unchanged` and issue no W1-W4 statement. Never inserts a build row."""
        self._pending_done = "unchanged"
        info = {"unchanged": 1, "version_known": 1}
        info.update(self._robust_info())
        info.update(self._scan_info())
        rid, self._run_id = self._run_id, None
        if rid is not None:
            with self._cx(True) as conn:
                store.run_sql(conn, "job_run_close", {"run_id": rid, "state": "finished", "neurons": _ceil_int(self._pass_neurons),
                                                      "metrics": json.dumps({"run": _numeric_only(info) or {}}), "reasons": ["unchanged"]})
        self.fails = 0
        self._release_buffers()
        self._phase = "idle"
        self._set(state="unchanged", version=version, last_run=self._wall())
        log.info("chunk_taxo run unchanged version=%s", version[:32])
        return 0.0

    # ---- D3 incremental assignment (WP-I). Ships `off`. `shadow` computes, in memory only, what `on` would assign and reports the
    # agreement with the rebuild; `on` assigns spans that have a vector and no strict link, against the centers of the last build.

    def _inc_inputs(self):
        """-> (reason, data). reason is None when assignment can run; otherwise `no_build_params` (no finished build row carrying the
        stored cutoffs, or no active nodes) or `write_incomplete` (a pass after that build was aborted between W1 and W4)."""
        with self._cx() as conn:
            row = store.run_sql(conn, "job_run_build_latest").first()
            if row is None:
                return "no_build_params", None
            m = self._as_dict(row[1])
            end = m.get("scan_end_epoch")
            if isinstance(end, bool) or not isinstance(end, (int, float)) or not isinstance(m.get("max_span_id"), (int, float)):
                return "no_build_params", None
            after = {int(r[0]) for r in store.run_sql(conn, "vectors_after", {"epoch": float(end)}).fetchall()}
            if self._write_incomplete(conn):  # a pass aborted after that build (its closed row says so), or ...
                return "write_incomplete", None
            if all(isinstance(m.get("n_nodes_" + lv), (int, float)) for lv in store.LEVELS) and \
                    self._active_node_counts(conn) != {lv: m["n_nodes_" + lv] for lv in store.LEVELS}:
                return "write_incomplete", None  # ... active nodes beyond the build's own (a pass killed after W1)
        prior = self._load_prior()
        if prior is None:
            return "no_build_params", None
        null = {}
        for lv in store.LEVELS:
            if any(n["level"] == lv for n in prior["nodes"]):
                dq = m.get("dist_q_" + lv)
                if isinstance(dq, bool) or not isinstance(dq, (int, float)) or "has_margin_" + lv not in m:
                    return "no_build_params", None
                null[lv] = {"dist_q": float(dq), "margin_q": float(m["margin_q_" + lv]) if m.get("has_margin_" + lv) == 1 and
                            isinstance(m.get("margin_q_" + lv), (int, float)) else None}
        if not isinstance(m.get("min_support"), (int, float)):
            return "no_build_params", None
        return None, {"version": str(row[0]), "m": m, "prior": prior, "null": null, "after": after}

    def _inc_model(self, data):
        """A core-compatible model of the last build rebuilt from the stored strict members: center = unit mean of the member vectors
        present in this scan (a topic's members include its sub-topics', a domain's its topics', as _load_prior propagates them)."""
        import numpy as np  # noqa: PLC0415
        X = self._matrix()
        self._vecs = [X]  # keep the matrix: a rebuild may still follow
        pos = {int(c["span_id"]): i for i, c in enumerate(self._chunks)}
        nodes = []
        for n in data["prior"]["nodes"]:
            idx = [pos[s] for s in n["members"] if s in pos]
            if not idx:
                continue
            c = np.asarray(X[idx], dtype=np.float64).mean(axis=0)
            nrm = float(np.linalg.norm(c))
            if nrm > 0:
                nodes.append({"key": n["key"], "level": n["level"], "parent_key": n["parent_key"], "center": (c / nrm).tolist(),
                              "members": list(n["members"])})
        params = dict(self.cfg.get("params") if isinstance(self.cfg.get("params"), dict) else {}, min_support=data["m"]["min_support"])
        return SimpleNamespace(nodes=nodes, null=data["null"], params=params, version=data["version"]), X, pos

    def _inc_targets(self, data):
        mx = int(data["m"]["max_span_id"])
        return [i for i, c in enumerate(self._chunks)
                if c["span_id"] not in self._prior_linked and (c["span_id"] > mx or c["span_id"] in data["after"])]

    def _inc_associate(self, data):
        model, X, _pos = self._inc_model(data)
        out, errs = {}, 0
        for i in self._inc_targets(data):
            try:
                out[int(self._chunks[i]["span_id"])] = self.core.associate(model, X[i])
            except Exception:  # noqa: BLE001  (a bad vector: the span stays unassigned, counted)
                errs += 1
        return out, errs

    def _write_incremental_page(self, conn, spans, assoc, ids, version):
        strict = {k: [] for k in store.CHUNK_LINKS}
        for sid in spans:
            lr = self._link_rows(assoc.get(sid), ids)
            if lr is not None:
                strict[lr[0]].append({"a": sid, "b": lr[1]})
        for kind, rows in strict.items():
            if rows:
                store.run_sql(conn, f"link_retire_pair_routing_{kind}", rows)  # T28: a same-version routing row on the pair goes first
        self._write_chunk_page(conn, spans, assoc, ids, version)

    def _assign(self, data) -> None:
        assoc, errs = self._inc_associate(data)
        ids = {lv: {} for lv in store.LEVELS}
        for slug, (lv, eid) in self._prior_ids.items():
            ids[lv][slug] = eid
        version = data["version"]
        self._ctx_version = version
        spans = sorted(assoc)
        for i in range(0, len(spans), self.page):
            with self._tx() as conn:
                self._write_incremental_page(conn, spans[i:i + self.page], assoc, ids, version)
        assigned = sum(1 for s in spans if self._link_rows(assoc[s], ids) is not None)
        self._inc.update({"inc_targets": len(spans) + errs, "inc_assigned": assigned, "inc_abstained": len(spans) - assigned,
                          "inc_errors": errs, "inc_neurons": 0})

    def _finish_incremental(self, reason: str) -> None:
        """Close the pass's running row as finished with `incremental`; never inserts a build row."""
        self._pending_done = "incremental"
        info = {**self._scan_info(), **self._inc, "rows_embedded": self.rows_embedded}
        info.update(self._robust_info(final=True))
        rid, self._run_id = self._run_id, None
        if rid is not None:
            with self._cx(True) as conn:
                store.run_sql(conn, "job_run_close", {"run_id": rid, "state": "finished", "neurons": _ceil_int(self._pass_neurons),
                                                      "metrics": json.dumps({"run": _numeric_only(info) or {}}), "reasons": [reason]})
        self.fails = 0
        self._release_buffers()
        self._phase = "idle"
        self._set(state="incremental", last_run=self._wall())

    def _try_incremental(self) -> bool:
        """Mode `on`, complete snapshot: assign instead of rebuilding unless no params / write_incomplete / a rebuild trigger fires.
        Returns True when the pass was finished here."""
        reason, data = self._inc_inputs()
        if reason is not None:
            self._inc["inc_" + reason] = 1
            return False
        total = len(self._chunks)
        unassigned = sum(1 for c in self._chunks if c["span_id"] not in self._prior_linked)
        if self.rebuild_unassigned_share is not None and total and unassigned / total >= self.rebuild_unassigned_share:
            self._inc["inc_rebuild_trigger_unassigned"] = 1
            return False
        if self.rebuild_interval_s is not None and self._scan_end is not None and \
                self._scan_end - float(data["m"]["scan_end_epoch"]) >= self.rebuild_interval_s:
            self._inc["inc_rebuild_trigger_interval"] = 1
            return False
        self._assign(data)
        self._finish_incremental("incremental")
        return True

    def _do_assign(self):
        """Budget-stopped scan (D2), build present, mode `on`: assign the spans that have a vector; never rebuild; the pass ends deferred."""
        reason, data = self._inc_inputs()
        if reason is not None:
            self._inc["inc_" + reason] = 1
            self._no_rebuild = True
            self._stop = "deferred"
            return 0.0
        self._assign(data)
        self._finish_incremental("incremental")
        self._stop = "deferred"
        return 0.0

    def _shadow_compute(self) -> None:
        """Mode `shadow`: before the rebuild, compute in memory what `on` would assign (no write); compared after the build."""
        try:
            reason, data = self._inc_inputs()
            if reason is not None:
                self._shadow = {"reason": reason, "assign": {}}
                return
            assoc, errs = self._inc_associate(data)
            self._shadow = {"reason": None, "assign": assoc, "errors": errs}
        except Exception:  # noqa: BLE001  (shadow must never stop a pass)
            self._shadow = None

    def _inc_metrics(self) -> dict:
        """Run-row numbers of the shadow comparison (agreement after identity matching, T20) or of nothing."""
        sh = self._shadow
        if not sh:
            return {}
        out = {"shadow_targets": len(sh["assign"]), "shadow_no_build_params": 1 if sh.get("reason") else 0}
        try:
            model = self._pending.model
            reused = {str(_g(n, "key")): _g(n, "reused_from") for n in (_g(model, "nodes") or [])}
            assoc = _g(model, "assoc") or {}
            agree = compared = abstain = 0
            for sid, old in sh["assign"].items():
                o_lvl = old.get("level") if isinstance(old, dict) else None
                if o_lvl is None:
                    abstain += 1
                    continue
                new = assoc.get(sid)
                n_lvl = new.get("level") if isinstance(new, dict) else None
                if n_lvl is None:
                    continue
                compared += 1
                if n_lvl == o_lvl and reused.get(str(new.get(n_lvl))) == old.get(o_lvl):
                    agree += 1
            out.update({"shadow_compared": compared, "shadow_agree": agree, "shadow_abstain": abstain})
        except Exception:  # noqa: BLE001
            pass
        return out

    # ---- prior snapshot (identity across runs) from the stored links; no vectors are stored
    def _load_prior(self):
        idx = {c["span_id"]: i for i, c in enumerate(self._chunks)}
        nodes, id2slug, level_of = {}, {}, {}
        with self._cx() as conn:
            for lvl in store.LEVELS:
                for n in store.fetch_nodes(conn, lvl):
                    if n["origin"] == "induction" and (n["origin_ref"] or "").startswith(store.ORIGIN_REF_PREFIX) and n["state"] == "active":
                        nodes[n["slug"]] = {"key": n["slug"], "level": lvl, "parent_key": None, "members": set(), "label": n["label"]}
                        id2slug[(lvl, n["id"])] = n["slug"]
            if not nodes:
                return None
            self._prior_ids = {s: k for k, s in id2slug.items()}  # slug -> (level, entity id), for D3 writes
            for kind, plv, clv in (("domain_topic", "domain", "topic"), ("topic_subtopic", "topic", "subtopic")):
                for a, b in store.fetch_hierarchy(conn, kind):
                    c, p = id2slug.get((clv, b)), id2slug.get((plv, a))
                    if c and p:
                        nodes[c]["parent_key"] = p
            # WP-W (pass 7 N6, pass 9 M3): each span contributes to exactly one node, the one of its newest active STRICT link: the deepest
            # level first (the two tables' link_id sequences are not comparable), then the highest link_id within that table.
            best = {}
            for kind, lvl, rank in (("chunk_topic", "topic", 0), ("chunk_subtopic", "subtopic", 1)):
                after, after_lid = 0, 0
                while True:
                    rows = store.fetch_strict_links(conn, kind, after, after_lid, self.page)
                    for span, eid, lid in rows:
                        s = id2slug.get((lvl, eid))
                        if s and span in idx and ((rank, lid) > best.get(span, ((-1, -1), None))[0]):
                            best[span] = ((rank, lid), s)
                    if len(rows) < self.page:
                        break
                    after, after_lid = rows[-1][0], rows[-1][2]
            for span, (_k, s) in best.items():
                nodes[s]["members"].add(span)
            self._prior_linked = set(best)
        for lvl in ("subtopic", "topic"):  # members propagate upward
            for n in nodes.values():
                if n["level"] == lvl and n["parent_key"] in nodes:
                    nodes[n["parent_key"]]["members"] |= n["members"]
        return {"version": "prior", "nodes": [{"key": n["key"], "level": n["level"], "parent_key": n["parent_key"],
                                               "members": sorted(n["members"]), "label": n["label"]} for n in nodes.values()]}

    # ---- writes. WP-W (CHUNK-ASSOC-1113 D1) reversed the one-transaction-per-pass design of orchestrator decision v3.8b (c) (INFO-1): the pass no longer
    # commits all-or-nothing; it commits per phase as described below. Embedding batches (a cache) still commit on their own earlier.
    # D1: activate first, then one transaction per page of spans or sources, then a closing transaction (W4). Every phase below commits on its own, so a restart or budget stop mid-write loses no
    # visible label: a span not yet swapped keeps its old links, a swapped span shows its new ones.
    def _tx(self):
        if self._wconn is not None:  # W4: the retire and the run-row insert share one transaction
            return contextlib.nullcontext(self._wconn)
        return self._wtx()

    @contextlib.contextmanager
    def _wtx(self):
        """A write transaction: the held-lock check (W0), then the 022 guard context for every guarded statement of the transaction."""
        with self._cx(True) as conn:
            if self._held is not None and not self._lock_held(conn):
                raise _WriterLost()
            ver = self._ctx_version or (str(_g(self._pending.model, "version")) if self._pending is not None else None)
            if self._mode == "write" and ver is not None:
                store.set_ctx(conn, hashlib.sha256((store.ORIGIN_REF_PREFIX + ver).encode("utf-8")).hexdigest())
            yield conn

    # ---- W0 single writer: the pass holds the pool's one connection and a session advisory lock on it
    @staticmethod
    def _end(conn, ok: bool) -> None:
        fn = getattr(conn, "commit", None) if ok else getattr(conn, "rollback", None)  # test doubles may offer no transaction API
        if fn is None:
            return
        if ok:
            fn()  # a failed commit must reach the pass's error path: never report `complete` for a write that did not commit
            return
        try:
            fn()
        except Exception:  # noqa: BLE001  (a rollback on a dead connection: the original error is the one that matters)
            pass

    @contextlib.contextmanager
    def _cx(self, write: bool = False):
        """Every database use of a pass. While the pass holds its connection this yields it and ends the transaction explicitly
        (commit, or rollback on error: a held connection autobegins, an open read would hold locks for the pass); otherwise a
        fresh engine connection, as before W0."""
        if self._held is None:
            with (self._engine().begin() if write else self._engine().connect()) as conn:
                yield conn
            return
        conn = self._held
        try:
            yield conn
        except BaseException:
            self._end(conn, False)
            raise
        self._end(conn, True)

    def _lock_held(self, conn) -> bool:
        row = store.run_sql(conn, "advisory_check", LOCK_KEY).first()
        return bool(row[0]) if row is not None else False

    def _acquire(self) -> bool:
        """Take the single connection and the session lock; False (and nothing held) when another session holds the lock."""
        cm = self._engine().connect()
        conn = cm.__enter__()
        try:
            row = store.run_sql(conn, "advisory_lock", LOCK_KEY).first()
            self._end(conn, True)
            ok = bool(row[0]) if row is not None else False
        except BaseException:
            self._end(conn, False)
            cm.__exit__(None, None, None)
            raise
        if not ok:
            cm.__exit__(None, None, None)
            return False
        self._hcm, self._held = cm, conn
        return True

    def _release_held(self) -> None:
        cm, conn, self._hcm, self._held = self._hcm, self._held, None, None
        if cm is None:
            return
        try:
            store.run_sql(conn, "advisory_unlock", LOCK_KEY)
            self._end(conn, True)
        except Exception:  # noqa: BLE001
            pass
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def _abandon_held(self) -> None:
        """The lock is gone: close the connection without trusting it (no unlock call)."""
        cm, self._hcm, self._held = self._hcm, None, None
        try:
            if cm is not None:
                cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def _do_write(self):
        self._phase = "w_nodes"
        return 0.0

    def _model_nodes(self, model):
        return [{"key": str(_g(n, "key")), "level": _g(n, "level"), "parent_key": _g(n, "parent_key"), "label": _g(n, "label"),
                 "terms": list(_g(n, "terms") or []), "reused_from": _g(n, "reused_from")} for n in (_g(model, "nodes") or [])]

    def _do_nodes(self):
        model, version = self._pending.model, str(_g(self._pending.model, "version"))
        origin, state0, oref = ("induction", "proposed", store.ORIGIN_REF_PREFIX + version) if self._mode == "write" else \
            ("curator", "active", store.ORIGIN_REF_PREFIX + version)
        nodes = self._model_nodes(model)
        self._w_started = True  # W1: from here an aborted pass is `write_incomplete` until W4 commits
        with self._tx() as conn:  # _wtx sets the 022 guard context (vera.tax_actor / vera.tax_batch) for this transaction
            existing, taken = {}, set()
            for lvl in store.LEVELS:
                for n in store.fetch_nodes(conn, lvl):
                    existing[(lvl, n["slug"])] = n["id"]
                    taken.add(_LABEL_NORM.sub(" ", n["label"].strip()).lower())
            slug_of, new_by_lvl = {}, {l: [] for l in store.LEVELS}
            for n in nodes:
                lvl = n["level"]
                if lvl not in store.LEVELS:
                    continue
                rf = n["reused_from"]
                if rf and (lvl, str(rf)) in existing:
                    slug_of[n["key"]] = str(rf)
                    continue
                slug = f"ct-{_ABBR[lvl]}-" + hashlib.sha256(f"{version}|{n['key']}".encode("utf-8")).hexdigest()[:12]
                slug_of[n["key"]] = slug
                if (lvl, slug) not in existing:
                    new_by_lvl[lvl].append((slug, n))
            for lvl in store.LEVELS:
                rows = [{"slug": s, "label": _clean_label(n["label"], taken, s), "origin": origin, "origin_ref": oref, "state": state0}
                        for s, n in new_by_lvl[lvl]]
                store.run_sql(conn, f"node_insert_{lvl}", rows)
                self._run_new_nodes += len(rows)
            ids = {}
            for lvl in store.LEVELS:
                ids[lvl] = store.node_ids(conn, lvl, sorted({slug_of[n["key"]] for n in nodes if n["level"] == lvl}))
            if self._mode == "write":
                for lvl in ("topic", "subtopic"):
                    arows = []
                    for s, n in new_by_lvl[lvl]:
                        pay = _anchor_payload(n["terms"])
                        if pay and s in ids[lvl]:
                            arows.append({"eid": ids[lvl][s][0], "text": pay[0], "terms": pay[1], "origin_ref": oref})
                    store.run_sql(conn, f"anchor_insert_{lvl}", arows)
                for lvl in store.LEVELS:
                    slugs = [s for s, _n in new_by_lvl[lvl]]
                    if slugs:
                        store.run_sql(conn, f"node_activate_{lvl}", {"slugs": slugs})
                for lvl in ("topic", "subtopic"):
                    eids = [ids[lvl][s][0] for s, _n in new_by_lvl[lvl] if s in ids[lvl]]
                    if eids:
                        store.run_sql(conn, f"anchor_activate_{lvl}", {"eids": eids})
                ids = {lvl: store.node_ids(conn, lvl, sorted({slug_of[n["key"]] for n in nodes if n["level"] == lvl})) for lvl in store.LEVELS}
            for lvl in store.LEVELS:
                for s, (_i, st) in ids[lvl].items():
                    if st != "active":
                        raise RuntimeError("node_not_active")
            # hierarchy (W1): old pairs the new version does not assert stay active until W4 (old sub-topic links still derive their
            # topic and domain); a pair both versions assert is retired and re-inserted here, so W4 never removes it
            by_key = {n["key"]: n for n in nodes}
            for kind, plv, clv in (("domain_topic", "domain", "topic"), ("topic_subtopic", "topic", "subtopic")):
                rows = []
                for n in nodes:
                    p = by_key.get(n["parent_key"]) if n["level"] == clv and n["parent_key"] is not None else None
                    if p is not None and p["level"] == plv:
                        rows.append({"a": ids[plv][slug_of[p["key"]]][0], "b": ids[clv][slug_of[n["key"]]][0], "version": version,
                                     "score": None, "score_kind": None})
                store.run_sql(conn, f"link_retire_pair_other_{kind}", rows)
                store.run_sql(conn, f"link_insert_{kind}", rows)
            # a node the new version itself uses is never retired (a re-run after a crash between W1 and W4 meets its own empty nodes)
            retire = [str(k) for k in (_g(model, "retired_keys") or []) if str(k) not in set(slug_of.values())]
        self._ids = {lvl: {n["key"]: ids[lvl][slug_of[n["key"]]][0] for n in nodes if n["level"] == lvl} for lvl in store.LEVELS}
        self._retire_keys = retire
        self._plan = [int(c["span_id"]) for c in self._chunks]
        self._w_pos = 0
        self._phase = "w_links"
        return 0.0

    def _link_rows(self, assoc_item, ids):
        """(table kind, id) for the deepest level a chunk reached, or None (abstained at every storable level)."""
        if not isinstance(assoc_item, dict):
            return None
        lvl = assoc_item.get("level")
        if lvl == "subtopic":
            kind, key = "chunk_subtopic", assoc_item.get("subtopic")
        elif lvl == "topic":
            kind, key = "chunk_topic", assoc_item.get("topic")
        else:
            return None
        eid = ids[lvl].get(key) if key is not None else None
        return (kind, eid) if eid is not None else None

    def _extra_link_rows(self, assoc_item, ids):
        """Routing-only generous links: [(table kind, id, cosine, score_kind)] (score = raw chunk-to-node cosine)."""
        out = []
        if not isinstance(assoc_item, dict):
            return out
        for ent in assoc_item.get("multi") or ():
            lvl, key, sim, sk = ent
            kind = {"subtopic": "chunk_subtopic", "topic": "chunk_topic"}.get(lvl)
            eid = ids.get(lvl, {}).get(key) if kind else None
            if eid is not None and isinstance(sim, (int, float)) and math.isfinite(sim) and re.match(r"^cosine_[a-z0-9]+$", str(sk)):
                out.append((kind, eid, round(float(sim), 6), str(sk)))
        return out

    def _write_chunk_page(self, conn, spans, assoc, ids, version):
        for kind in store.CHUNK_LINKS:
            store.run_sql(conn, f"link_retire_{kind}", {"version": version, "ids": list(spans)})
        by_kind = {k: [] for k in store.CHUNK_LINKS}
        for sid in spans:
            item = assoc.get(sid)
            lr = self._link_rows(item, ids)
            seen = set()
            if lr is not None:
                sim = _similarity(item.get("distance"))
                by_kind[lr[0]].append({"a": sid, "b": lr[1], "version": version, "score": sim, "score_kind": "cosine" if sim is not None else None})
                seen.add(lr)
            for kind, eid, sim, sk in self._extra_link_rows(item, ids):  # unique per (span, node): no duplicate rows
                if (kind, eid) not in seen:
                    seen.add((kind, eid))
                    by_kind[kind].append({"a": sid, "b": eid, "version": version, "score": sim, "score_kind": sk})
        for kind, rows in by_kind.items():
            store.run_sql(conn, f"link_insert_{kind}", rows)

    def _do_links(self):
        model = self._pending.model
        version, assoc = str(_g(model, "version")), _g(model, "assoc") or {}
        done = 0
        while done < self.cap and self._w_pos < len(self._plan):
            spans = self._plan[self._w_pos:self._w_pos + min(self.page, self.cap - done)]
            with self._tx() as conn:
                self._write_chunk_page(conn, spans, assoc, self._ids, version)
            self._w_pos += len(spans)
            done += len(spans)
        if self._w_pos >= len(self._plan):
            span_source = {int(c["span_id"]): int(c["source_id"]) for c in self._chunks}
            subj = self.core.doc_subjects(model, span_source)
            self._vote_rows = self._votes_from(subj)
            self._v_pos = 0
            self._phase = "w_votes"
            return 0.0
        return self.step_delay

    def _votes_from(self, subj):
        """[(source_id, kind, level, key, share|None)] from doc_subjects: {src: {"vote": {topic, domain, share, domain_share}}}."""
        rows = []
        for sid, entry in (subj.items() if isinstance(subj, dict) else []):
            vote = _g(entry, "vote") or {}
            for lvl, kind, sk in (("topic", "source_topic", "share"), ("domain", "source_domain", "domain_share")):
                key = _g(vote, lvl)
                if key is not None:
                    rows.append((int(sid), kind, lvl, str(key), _g(vote, sk)))
        rows.sort(key=lambda r: (r[0], r[1]))
        return rows

    def _do_votes(self):
        version = str(_g(self._pending.model, "version"))
        sources = sorted({int(c["source_id"]) for c in self._chunks})
        done = 0
        while done < self.cap and self._v_pos < len(sources):
            page = sources[self._v_pos:self._v_pos + min(self.page, self.cap - done)]
            pset = set(page)
            with self._tx() as conn:
                for kind in store.SOURCE_LINKS:
                    store.run_sql(conn, f"link_retire_{kind}", {"version": version, "ids": page})
                by_kind = {k: [] for k in store.SOURCE_LINKS}
                for sid, kind, lvl, key, share in self._vote_rows:
                    eid = self._ids[lvl].get(key)
                    if sid in pset and eid is not None:
                        sh = None if share is None or not isinstance(share, (int, float)) or not math.isfinite(share) else round(float(share), 6)
                        by_kind[kind].append({"a": sid, "b": eid, "version": version, "score": sh, "score_kind": "share" if sh is not None else None})
                for kind, rows in by_kind.items():
                    store.run_sql(conn, f"link_insert_{kind}", rows)
            self._v_pos += len(page)
            done += len(page)
        if self._v_pos >= len(sources):
            self._phase = "w_retire"
            return 0.0
        return self.step_delay

    def _do_retire(self):
        """W4 (one transaction): retire superseded nodes, their anchors and the old hierarchy pairs the new version did not assert, retire
        this classifier's links of positively excluded spans, and insert the build row (the run row closes in the same transaction)."""
        keys = set(getattr(self, "_retire_keys", []) or [])
        version = str(_g(self._pending.model, "version"))
        with self._tx() as conn:
            self._wconn = conn
            try:
                for lvl in store.LEVELS:
                    slugs = sorted(keys)
                    if not slugs:
                        break
                    if self._mode == "write" and lvl != "domain":
                        eids = [i for i, _st in store.node_ids(conn, lvl, slugs).values()]
                        if eids:
                            store.run_sql(conn, f"anchor_retire_{lvl}", {"eids": eids})
                    store.run_sql(conn, f"node_retire_{lvl}", {"slugs": slugs})
                for kind in store.HIERARCHY:
                    store.run_sql(conn, f"link_retire_{kind}", {"version": version})
                self._retire_excluded(conn)
                self._fin = self._finish_rows()
            finally:
                self._wconn = None
        self._w_done = True
        metrics, version = self._fin
        self._finish_status(metrics, version)
        return 0.0

    def _retire_excluded(self, conn) -> None:
        """W4: a span the scan positively excluded (portal, portal_error, no_source, empty) loses this classifier's links, whether or not
        its node was reused. Spans merely without a current vector keep theirs. Counted per class (`outside_build_unlabelled_<class>`)
        for the spans that held a strict link; the sum is `outside_build_unlabelled`."""
        allx = {sid for ids in self._excl.values() for sid in ids}
        self._outside = {}
        if not allx:
            return
        linked = set()
        for kind in store.CHUNK_LINKS:
            after, after_lid = 0, 0
            while True:
                rows = store.fetch_strict_links(conn, kind, after, after_lid, self.page)
                linked.update(r[0] for r in rows if r[0] in allx)
                if len(rows) < self.page:
                    break
                after, after_lid = rows[-1][0], rows[-1][2]
        ids = sorted(allx)
        for kind in store.CHUNK_LINKS:
            for i in range(0, len(ids), self.page):
                store.run_sql(conn, f"link_retire_spans_{kind}", {"ids": ids[i:i + self.page]})
        self._outside = {c: len(linked & set(v)) for c, v in self._excl.items()}

    def _finish_rows(self):
        model, pend = self._pending.model, self._pending
        version = str(_g(model, "version"))
        elapsed = max(1e-6, self._wall() - self._t0)
        self.metrics_run = {"rows_embedded": self.rows_embedded, "seconds": round(elapsed, 3),
                            "rows_per_s": round(self.rows_embedded / elapsed, 4), "peak_rss_mb": round(self.peak_rss, 1)}
        self.metrics_run.update(_numeric_only(self._robust_info(final=True)) or {})
        self.metrics_run.update(self._scan_info())
        self.metrics_run.update(self._inc_metrics())
        if self.finish_now:  # WP-F: present only with the flag on (flag off keeps main's run-row keys)
            self.metrics_run["partial_snapshot"] = 1 if self._reuse_only else 0
        metrics = dict(_numeric_only(pend.metrics) or {})
        if self._mode != "compute":  # WP-W: the W4 build row's snapshot and the counted label decreases (D1, D3 inputs)
            assoc = _g(model, "assoc") or {}
            new_linked = {int(s) for s, it in assoc.items() if self._link_rows(it, self._ids) is not None} if isinstance(assoc, dict) else set()
            built = {int(c["span_id"]) for c in self._chunks}
            out = getattr(self, "_outside", {}) or {}
            self.metrics_run["abstained_now_unlabelled"] = len((self._prior_linked & built) - new_linked)
            self.metrics_run["outside_build_unlabelled"] = int(sum(out.values()))
            for cls, n in out.items():
                self.metrics_run["outside_build_unlabelled_" + cls] = int(n)
            metrics.update({k: v for k, v in self._snap.items()})
            for lv in store.LEVELS:  # the build's own node count per level: more active nodes later means an interrupted write (write_incomplete)
                metrics["n_nodes_" + lv] = sum(1 for n in self._model_nodes(model) if n["level"] == lv)
            if self._scan_end is not None:
                metrics["scan_end_epoch"] = round(float(self._scan_end), 3)
            # D3 stored inputs: per-level cutoffs and a presence flag (margin_q is absent for a level with too few nodes, and absence
            # must not read as "params missing"); numbers only, far below the 64-key cut of _numeric_only
            null, prm = _g(model, "null") or {}, _g(model, "params") or {}
            for lv in store.LEVELS:
                nl = null.get(lv) if isinstance(null, dict) else None
                if isinstance(nl, dict) and isinstance(nl.get("dist_q"), (int, float)):
                    metrics["dist_q_" + lv] = float(nl["dist_q"])
                    mq = nl.get("margin_q")
                    metrics["has_margin_" + lv] = 1 if isinstance(mq, (int, float)) else 0
                    if isinstance(mq, (int, float)):
                        metrics["margin_q_" + lv] = float(mq)
            if isinstance(prm, dict) and isinstance(prm.get("min_support"), (int, float)):
                metrics["min_support"] = prm["min_support"]
        if hasattr(self._emb, "stats"):
            self.metrics_run["embedder"] = _numeric_only(self._emb.stats()) or {}
        metrics["run"] = self.metrics_run
        mjson = json.dumps(metrics, sort_keys=True)
        with self._tx() as conn:
            if self._mode != "compute":
                store.run_sql(conn, "job_run_insert", {
                    "version": version[:128], "started": self._t0, "finished": self._wall(),
                    "metrics": mjson, "display_ok": pend.ok, "reasons": [str(r)[:64] for r in pend.reasons][:16]})
            if self._run_id is not None:  # the pass's running row (holds the neurons) is closed in the same transaction
                store.run_sql(conn, "job_run_close", {"run_id": self._run_id, "state": "finished", "neurons": _ceil_int(self._pass_neurons),
                                                      "metrics": mjson, "reasons": []})
        return metrics, version

    def _do_finish(self):
        if self._wconn is not None:
            self._fin = self._finish_rows()
            return 0.0
        metrics, version = self._finish_rows()
        self._finish_status(metrics, version)
        return 0.0

    def _finish_status(self, metrics, version):
        self._run_id = None
        self.fails = 0
        self._drop_engine()
        self._release_buffers()  # texts and vectors are not kept after a run
        self._phase = "idle"
        self._set(state=getattr(self, "_pending_done", None) or "complete", last_run=self._wall(), metrics=metrics)
        log.info("chunk_taxo run complete version=%s rows_embedded=%d peak_rss_mb=%.0f", version[:32], self.rows_embedded, self.peak_rss)


def read_rss_mb() -> float:
    """Resident set size in MB from /proc/self/status (0.0 when unavailable)."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# ---------------------------------------------------------------------- module singleton (started from main.py)
_JOB: Job | None = None
_THREAD: threading.Thread | None = None
_STOP = threading.Event()
_GUARD = threading.Lock()
_GUARD_ROW = threading.Lock()
_INERT_ROW_DONE = False  # one inert self-report row per process start


def status() -> dict:
    j = _JOB
    if j is None:
        return {"state": "off", "version": None, "metrics": {}, "display_ok": False, "reasons": [], "last_run": None}
    return j.status()


def reset_for_tests() -> None:
    global _JOB, _THREAD, _INERT_ROW_DONE
    _INERT_ROW_DONE = False
    _STOP.set()
    t = _THREAD
    if t is not None:
        t.join(timeout=2)
    with _GUARD:
        _JOB, _THREAD = None, None
    _STOP.clear()


def loop(job: Job, sleep=None, max_passes=None) -> None:
    """Initial delay, then one pass per interval; after an error or a stopped (partial) pass wait with exponential backoff."""
    jc = job.jcfg
    interval, delay = float(_num(jc, "interval_s")), float(_num(jc, "initial_delay_s"))
    wait = sleep or _STOP.wait
    fails, n = 0, 0
    if wait(delay) and sleep is None:
        return
    while not _STOP.is_set():
        started = time.monotonic()
        res = job.run()
        n += 1
        if max_passes is not None and n >= max_passes:
            return
        if res.get("state") in ("off", "inert"):
            return
        if res.get("state") == "deferred" or (res.get("finish_now") and res.get("state") not in ("error", "partial")):
            fails, w = 0, max(60.0, float(res.get("retry_in_s") or interval))
        elif res.get("state") in ("error", "partial"):
            fails += 1
            w = min(interval, max(1.0, delay) * (2 ** min(fails - 1, 20)))
        else:
            fails, w = 0, interval
        if sleep is not None:
            wait(w)
        elif _wait_or_wake(w, started, _wake_min(jc, interval)):
            return


_WAKE = threading.Event()


def wake() -> None:
    """D5: an ingest committed. Process-local, O(1), never raises, no DB write, no embedding on the caller's thread: it only marks the
    loop to start its next pass early, no sooner than `wake_min_interval_s` after the last pass began (debounced). A no-op while the job
    is not running (flag off)."""
    try:
        if _JOB is not None:
            _WAKE.set()
    except Exception:  # noqa: BLE001
        pass


def _wake_min(jc: dict, interval: float) -> float:
    v = jc.get("wake_min_interval_s")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1 else interval  # default: today's interval_s


def _wait_or_wake(w: float, last_start: float, min_int: float, clock=time.monotonic, stop_wait=None, slice_s: float = 5.0) -> bool:
    """Wait up to w seconds; True when a stop was requested. A wake ends the wait once min_int has passed since last_start (many wakes
    inside that window give one early pass)."""
    stop_wait = stop_wait or _STOP.wait
    end = clock() + w
    while True:
        rem = end - clock()
        if rem <= 0:
            _WAKE.clear()  # the next pass starts now; a wake seen during this wait is consumed by it
            return False
        if stop_wait(min(slice_s, rem)):
            return True
        if _WAKE.is_set():
            if clock() - last_start >= min_int:
                _WAKE.clear()
                return False
            end = min(end, last_start + min_int)


def _announce(job: "Job", env) -> None:
    """Startup self-report: first-pass delay at INFO; WARNING naming (names only, never values) any missing credential variables."""
    try:
        log.info("chunk_taxo: first pass in %d s", int(_num(job.jcfg, "initial_delay_s")))
        from vera import chunk_taxo_embed as emb  # noqa: PLC0415
        name = str(job.cfg.get("embedder") or env.get("CHUNK_TAXO_EMBEDDER") or "")
        if name == "cloudflare":
            missing = [v for v in (emb.ENV_ACCOUNT, emb.ENV_TOKEN) if not (env.get(v) or "").strip()]
            if missing:
                log.warning("chunk_taxo: credentials_missing, absent env var names: %s", ", ".join(missing))
    except Exception:  # noqa: BLE001
        pass


def start(cfg=None, env=None) -> bool:
    """Start the single daemon thread when config chunk_taxo.enabled is true. Never raises; no thread, engine or connection
    otherwise. Returns True only when a thread was started."""
    global _JOB, _THREAD
    try:
        with _GUARD:
            if _THREAD is not None:
                return False
            if cfg is None:
                from vera import chunk_taxo as core  # noqa: PLC0415
                cfg = core.load_cfg()
            if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
                return False
            _JOB = Job(cfg)
            _THREAD = threading.Thread(target=loop, args=(_JOB,), name="chunk-taxo-job", daemon=True)
            _THREAD.start()
            _announce(_JOB, env if env is not None else os.environ)
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("chunk_taxo job not started: CT_START_ERROR (%s)", type(exc).__name__)
        return False
