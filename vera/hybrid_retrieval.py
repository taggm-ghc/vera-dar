"""HYBRID-1109 (OD-1109-HYBRID): hybrid stored-passage retrieval (word matching + embedding similarity). FLAG OFF by default.

Owner decision 2026-11-09, "Yes, hybrid search": stored-passage retrieval adds embedding similarity to word matching; each
question is embedded once through Cloudflare Workers AI (same account, token and model as the chunk-taxonomy job, free tier;
OD-1109-DEMB-3); if Cloudflare is unavailable or over quota, retrieval falls back to word-only search.

Pieces (config/hybrid_retrieval.json):
  * Query vector: vera.chunk_taxo_embed adapter (https pin to api.cloudflare.com, no redirects, bounded reads, token never
    logged), with ONE attempt and a short timeout (timeout_s). Cached in process memory keyed by sha256(question) (never
    the text), so all corpora and retrieve calls of one question share one call. After an error the module stays
    word-only for cooldown_s; after a quota or budget stop, until 00:00 UTC.
  * Budget: the adapter's daily neuron guard. The day's persisted job usage (tax_job_runs, read OFF the request path with
    the read-only account) is passed in with set_day_used; ask-time calls may use the whole daily_neuron_budget, while the
    job's adapter stops at daily_neuron_budget - cloudflare.reserve_neurons (the ask-time share). Ask-time neurons are
    counted in the process-wide adapter counter (shared with the job in the same process), not persisted: a question is
    about 0.04 neurons, so the reserve covers thousands of questions; Cloudflare's own 10k/day limit is a hard stop that
    never charges (v3.8a), and it only causes word-only fallback.
  * Vector index: per served FullDocCorpus, built in a background thread (never on the request path) from
    vera_vjay.tax_chunk_vectors rows of the corpus's stored partitions (evidence_spans span_id, about 4000 characters),
    model = the embedder's model id, and only where the row's text_sha256 equals sha256 of the served partition text
    (first job.text_chars characters, as the job hashes it). Held as one float32 L2-normalised matrix (5,500 x 1024 is
    22.5 MB); refused above max_index_mb. Refreshed every index_refresh_s (the old index keeps serving meanwhile).
  * Candidate retrieval: vera.dar_corpus.DarCorpus.retrieve fuses the BM25 list and a vector list by reciprocal rank
    fusion (rank only). The vector list ranks the 400-character retrieval chunks by their partition's cosine similarity
    (chunk midpoint -> partition), at most chunks_per_partition per partition (best BM25 first). Chunks without a vector
    (inline-fetched documents, or not yet embedded by the job) stay lexical-only; newly fetched chunks are NOT embedded at
    ask time (deferred to the background job).
  * Admission (vera.fulldoc_admission): unchanged except (a) the vector stem-floor rule (stem_floor_sim, null = off,
    pending calibration) and (b) the admission order is the RRF of the lexical order and the vector order when any
    candidate has a vector.

Tracing: codes and counts only (status, n vectors, embed milliseconds); the question text, its vector and the token are
never logged or traced. Flag off: nothing here runs beyond the enabled() check, so behaviour is byte-identical.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("vera.hybrid")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "hybrid_retrieval.json"
VECTOR = "vector"  # branch name in vera.dar_corpus.retrieve

DEFAULTS = {"enabled": False, "timeout_s": 3.0, "rrf_k": 60, "vector_k": None, "chunks_per_partition": 2,
            "stem_floor_sim": None, "stem_floor_min_hits": 1, "cooldown_s": 60, "query_cache_max": 32,
            "query_cache_ttl_s": 120, "index_refresh_s": 3600, "max_index_mb": 96, "page_ids": 256,
            "ask_restart_margin_neurons": 500, "lock_budget_s": 3.0}
# Stage 2 (orchestrator decision within OD-1109-HYBRID): ask-time passage re-rank inside the top partitions. Own flag.
RERANK_DEFAULTS = {"enabled": False, "top_partitions": 3, "max_passages": 48, "max_chars": 2000, "timeout_s": 3.0,
                   "cooldown_s": 60, "cache_max": 2048, "reserve_slots": 2}
_RR_BOUNDS = {"top_partitions": (1, 20), "max_passages": (1, 64), "max_chars": (50, 20000), "timeout_s": (0.2, 30),
              "cooldown_s": (0, 86400), "cache_max": (0, 100000), "reserve_slots": (0, 20)}
_BOUNDS = {"timeout_s": (0.2, 30), "rrf_k": (1, 1000), "vector_k": (1, 1000), "chunks_per_partition": (1, 100),
           "stem_floor_sim": (-1.0, 1.0), "stem_floor_min_hits": (1, 20), "cooldown_s": (0, 86400),
           "query_cache_max": (1, 4096), "query_cache_ttl_s": (0, 86400), "index_refresh_s": (10, 10**7),
           "max_index_mb": (1, 400), "page_ids": (1, 5000), "ask_restart_margin_neurons": (0, 10**6),
           "lock_budget_s": (0.2, 30)}

SQL_VECTORS = ("SELECT v.span_id, v.text_sha256, v.vec FROM vera_vjay.tax_chunk_vectors v "
               "WHERE v.model = :model AND v.span_id = ANY(:ids) ORDER BY v.span_id")
SQL_NEURONS_TODAY = ("SELECT COALESCE(SUM(r.neurons_used), 0) FROM vera_vjay.tax_job_runs r "
                     "WHERE r.classifier_name = 'chunk_taxo_auto' AND r.utc_day = timezone('utc', now())::date")

# HYBRID-HANG-1109: re-entrant, and every request-path acquisition is bounded (_locked): no ask can wait on a hybrid lock
# beyond LOCK_TIMEOUT_S. (The plain Lock self-deadlocked when _embedder called config() while holding it.)
_LOCK = threading.RLock()
LOCK_TIMEOUT_S = 2.0
LOCK_BUDGET_S = 3.0  # default per-ask total of all lock waits (config lock_budget_s overrides)
INFLIGHT_STALE_S = 30.0  # HYBRID-SINGLEFLIGHT-1109: an embedder creation older than this is presumed dead
_INFLIGHT: dict = {}  # slot -> (Event, started monotonic); guarded by _LOCK


def _begin_ask_budget(seconds: float) -> None:
    """HYBRID-HARDEN-1109: start this thread's per-ask lock-wait budget (lock acquisitions and the embedder single-flight
    wait share it, so the worst case is bounded per ask, not per acquisition). Separate from the embed deadlines."""
    _TL.lock_left = max(0.0, float(seconds))


def _wait_left() -> float | None:
    """Seconds of lock-wait budget left for this thread's ask, or None when no ask budget is active."""
    return getattr(_TL, "lock_left", None)


def _spend_wait(dt: float) -> None:
    left = getattr(_TL, "lock_left", None)
    if left is not None:
        _TL.lock_left = max(0.0, left - max(0.0, dt))


@contextmanager
def _locked(lock=None, timeout: float | None = None, budgeted: bool = True):
    """Acquire `lock` (default _LOCK) for at most `timeout` seconds (default LOCK_TIMEOUT_S), and, inside an ask, no more
    than the ask's remaining lock-wait budget (an exhausted budget still tries once without waiting); TimeoutError
    otherwise. The request path catches it and falls back to word-only search."""
    lk = _LOCK if lock is None else lock
    t = LOCK_TIMEOUT_S if timeout is None else timeout
    left = _wait_left() if budgeted else None
    if left is not None:
        t = min(t, left)
    t0 = time.monotonic()
    ok = lk.acquire(timeout=t) if t > 0 else lk.acquire(blocking=False)
    if budgeted:
        _spend_wait(time.monotonic() - t0)
    if not ok:
        raise TimeoutError("hybrid_lock")
    try:
        yield
    finally:
        lk.release()


_STATE: dict = {"cfg": None, "emb": None, "emb_reason": None, "emb_at": None, "cool_until": 0.0, "cool_code": None,
                "emb2": None, "emb2_reason": None, "emb2_at": None, "rr_cool_until": 0.0, "rr_cool_code": None,
                "ask_day": None, "ask_neurons": 0.0,
                "neurons_today": None, "loader": None, "clock": time.monotonic, "wall": time.time}
_PCACHE: "OrderedDict[tuple, object]" = OrderedDict()  # (sha256(passage text), model) -> unit float32 vector (stage 2)
_QCACHE: "OrderedDict[str, tuple]" = OrderedDict()  # sha256(question) -> (expires, vec | None, code, embed_ms)
_TL = threading.local()


# ------------------------------------------------------------------------------------------------ config
def _clean(raw) -> dict:
    c = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return c
    if raw.get("enabled") is True:
        c["enabled"] = True
    for k, (lo, hi) in _BOUNDS.items():
        v = raw.get(k, DEFAULTS[k])
        if v is None and DEFAULTS[k] is None:
            c[k] = None
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi:
            c[k] = v
    rr, rraw = dict(RERANK_DEFAULTS), raw.get("rerank")
    if isinstance(rraw, dict):
        rr["enabled"] = rraw.get("enabled") is True
        for k, (lo, hi) in _RR_BOUNDS.items():
            v = rraw.get(k, rr[k])
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi:
                rr[k] = v
    c["rerank"] = rr
    return c


def config() -> dict:
    """The validated config (cached; a bad file or value means the default for that key; a bad file means OFF). Never
    raises and never waits on the lock beyond LOCK_TIMEOUT_S: on a lock timeout it returns the cached config, else OFF."""
    cached = _STATE["cfg"]
    if cached is not None:  # HYBRID-HARDEN-1109: lock-free fast path (a single dict read; the cfg dict is replaced, never mutated)
        return cached
    try:
        with _locked():
            if _STATE["cfg"] is None:
                try:
                    _STATE["cfg"] = _clean(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001  unreadable => off
                    _STATE["cfg"] = _clean({})
            return _STATE["cfg"]
    except TimeoutError:
        cached = _STATE["cfg"]
        return cached if cached is not None else _clean({})


def enabled() -> bool:
    try:
        return config().get("enabled") is True
    except Exception:  # noqa: BLE001
        return False


def set_config_for_tests(raw: dict | None) -> None:
    with _LOCK:
        _STATE["cfg"] = None if raw is None else _clean(raw)


def set_loader_for_tests(loader) -> None:
    """loader: object with vectors(ids: list[int], model: str) -> iterable of (span_id, text_sha256, vec) and
    neurons_today() -> float. None = the default read-only DB loader."""
    with _LOCK:
        _STATE["loader"] = loader


def set_clock_for_tests(clock=None, wall=None) -> None:
    with _LOCK:
        _STATE["clock"] = clock or time.monotonic
        _STATE["wall"] = wall or time.time


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update(cfg=None, emb=None, emb_reason=None, emb_at=None, cool_until=0.0, cool_code=None,
                      emb2=None, emb2_reason=None, emb2_at=None, rr_cool_until=0.0, rr_cool_code=None,
                      ask_day=None, ask_neurons=0.0, ask_boot=False,
                      neurons_today=None, loader=None, clock=time.monotonic, wall=time.time)
        _QCACHE.clear()
        _PCACHE.clear()
        _INFLIGHT.clear()
    _TL.__dict__.clear()


# ------------------------------------------------------------------------------------------------ per-thread status
def _set_status(**kw) -> None:
    _TL.status = dict(kw)


def clear_status() -> None:
    _TL.status = None


def last_status() -> dict | None:
    """The hybrid status of this thread's last FullDocCorpus.retrieve (codes and counts only), or None."""
    s = getattr(_TL, "status", None)
    return dict(s) if isinstance(s, dict) else None


# ------------------------------------------------------------------------------------------------ embedder + query vector
def _taxo_cfg() -> dict:
    from vera import chunk_taxo
    return chunk_taxo.load_cfg()


def _text_chars() -> int:
    try:
        from vera import chunk_taxo_job
        return int(chunk_taxo_job._num(_taxo_cfg().get("job") or {}, "text_chars"))
    except Exception:  # noqa: BLE001
        return 8000


def _embedder(slot: str = "emb"):
    """(embedder | None, reason). slot "emb" = the question embedder (timeout_s), "emb2" = the stage-2 passage embedder
    (rerank.timeout_s); each is a separate adapter instance (one attempt). Created once; a failed creation is retried after
    cooldown_s. Does not raise for a failed creation; the lock waits in it can raise TimeoutError (hybrid_lock) when the
    lock is held elsewhere past the budget, which callers treat as word-only."""
    now = _STATE["clock"]()
    cooldown = float(config()["cooldown_s"])  # HYBRID-HANG-1109: outside the lock (this call under the lock deadlocked)
    with _locked():
        if _STATE[slot] is not None:
            return _STATE[slot], "ok"
        if _STATE[slot + "_at"] is not None and now - _STATE[slot + "_at"] < cooldown:
            return None, _STATE[slot + "_reason"]
        flight = _INFLIGHT.get(slot)
        if flight is not None and time.monotonic() - flight[1] > INFLIGHT_STALE_S:
            flight = None  # a creator that never finished: stop waiting on it, take over
        if flight is None:
            flight = _INFLIGHT[slot] = (threading.Event(), time.monotonic())
            creator = True
        else:
            creator = False
    if not creator:
        # HYBRID-SINGLEFLIGHT-1109: one creator per slot; waiters wait at most LOCK_TIMEOUT_S, then fall back to word-only.
        left = _wait_left()
        t0 = time.monotonic()
        flight[0].wait(LOCK_TIMEOUT_S if left is None else min(LOCK_TIMEOUT_S, left))
        _spend_wait(time.monotonic() - t0)
        try:
            with _locked():
                if _STATE[slot] is not None:
                    return _STATE[slot], "ok"
                if _STATE[slot + "_at"] is not None and flight[0].is_set():
                    return None, _STATE[slot + "_reason"]  # the creator's failure, shared (it is in the cooldown record)
        except TimeoutError:
            pass
        return None, "embedder_wait_timeout"
    emb, why = None, "embedder_error"
    try:
        from vera.chunk_taxo_embed import load_embedder
        tc = _taxo_cfg()
        to = float(config()["timeout_s"] if slot == "emb" else config()["rerank"]["timeout_s"])
        cf = dict(tc.get("cloudflare") or {}, reserve_neurons=0, max_attempts=1, timeout_s=to)
        emb, why = load_embedder(dict(tc, cloudflare=cf))  # kept outside _LOCK: it reads config and env, and a custom factory may do I/O
    except Exception as exc:  # noqa: BLE001
        emb, why = None, "embedder_error"
        log.warning("hybrid embedder unavailable (%s)", type(exc).__name__)
    finally:
        try:
            with _locked():
                owner = _INFLIGHT.get(slot) is flight
                # R-SF1: a superseded (taken-over) creator writes only a success into an empty slot; a failure never
                # replaces a good embedder, and a creator that no longer owns the slot never writes a failure.
                if (owner or (emb is not None and _STATE[slot] is None)) and not (emb is None and _STATE[slot] is not None):
                    _STATE[slot], _STATE[slot + "_reason"], _STATE[slot + "_at"] = emb, why, now
                if owner:
                    del _INFLIGHT[slot]
        except TimeoutError:
            # HYBRID-HARDEN-1109: ownership check and pop under the lock, bounded (0.5 s, outside the ask budget); if the lock
            # is still unavailable the entry is left and the stale rule frees it (a pop without the lock raced a takeover).
            try:
                with _locked(timeout=0.5, budgeted=False):
                    if _INFLIGHT.get(slot) is flight:
                        del _INFLIGHT[slot]
            except TimeoutError:
                pass
        flight[0].set()
    return emb, why


def _ask_share() -> float:
    """The ask-time share of the daily neuron budget: config/chunk_taxo.json cloudflare.reserve_neurons."""
    try:
        return float((_taxo_cfg().get("cloudflare") or {}).get("reserve_neurons", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _ask_used() -> float:
    """Neurons used by ask-time calls today. In-memory only (no DB write from the ask path). A process restart loses the
    count, so the FIRST day this process sees starts at (ask share - ask_restart_margin_neurons): after a restart only the
    margin remains for the rest of that day, so restarts cannot grant a fresh full share (each adds at most the margin).
    Later days in the same process start at 0."""
    day = time.strftime("%Y-%m-%d", time.gmtime(_STATE["wall"]()))
    floor = max(0.0, _ask_share() - float(config().get("ask_restart_margin_neurons", 500)))  # outside the lock
    with _locked():
        if _STATE["ask_day"] != day:
            boot = _STATE.get("ask_boot", True)
            _STATE["ask_boot"] = False
            _STATE["ask_day"] = day
            _STATE["ask_neurons"] = floor if boot else 0.0
        return _STATE["ask_neurons"]


def _ask_charge(neurons: float) -> None:
    _ask_used()
    with _locked():
        _STATE["ask_neurons"] += max(0.0, float(neurons))


def _neurons_of(emb) -> float:
    st = getattr(emb, "stats_run", None)
    return float(st.get("neurons", 0.0)) if isinstance(st, dict) else 0.0


def neurons_estimate(texts, emb=None) -> float:
    """Estimated neurons for embedding `texts`: tokens (chars / chars_per_token) x neurons_per_m / 1e6 (bge-m3: 1075)."""
    cfg = getattr(emb, "cfg", None) or {}
    mcfg = getattr(emb, "mcfg", None) or {}
    cpt = float(cfg.get("chars_per_token", 3.0) or 3.0)
    npm = float(mcfg.get("neurons_per_m", 1075) or 1075)
    return sum(len(t) for t in texts) / cpt / 1e6 * npm


def model_name() -> str | None:
    emb, _ = _embedder()
    n = getattr(emb, "name", None) if emb is not None else None
    return str(n) if n else None


def _cool(code: str, seconds: float) -> None:
    with _locked():
        _STATE["cool_until"] = _STATE["clock"]() + max(0.0, float(seconds))
        _STATE["cool_code"] = code


def _embed_deadline(emb, text: str, timeout_s: float):
    """emb.embed([text]) under an OVERALL deadline (timeout_s covers connect, send, read and retries together, not each
    socket operation). The call runs in a daemon thread (its own socket timeout still bounds it); on expiry the ask moves on
    (TimeoutError -> word-only) and the thread's neurons are still charged when it ends."""
    return _embed_many_deadline(emb, [text], timeout_s)


def _embed_many_deadline(emb, texts: list, timeout_s: float, est: float | None = None):
    """emb.embed(texts) under an overall deadline (see _embed_deadline). Charges the adapter's neuron delta to the ask-time
    count; an adapter without stats_run is charged `est` (stage 2) or nothing. HYBRID-HANG-1109: stage 2 uses it too, so no
    ask-path embedding call waits beyond its deadline."""
    box: dict = {}

    def run():
        try:
            n0 = _neurons_of(emb)
            box["out"] = emb.embed(list(texts))
            if hasattr(emb, "stats_run") or est is None:
                _ask_charge(max(_neurons_of(emb) - n0, 0.0))
            else:
                _ask_charge(est)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc

    t = threading.Thread(target=run, name="hybrid-embed", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError("deadline")
    if "exc" in box:
        raise box["exc"]
    return box["out"]


def query_vector(text: str):
    """(unit float32 vector | None, code, embed_ms). One embedding call per distinct question text (cached by its sha256).
    Embedding failures are returned as a code, not raised. It can raise TimeoutError (hybrid_lock) when a hybrid lock is
    held elsewhere past the ask's lock budget; vector_fn (its caller) catches that and the ask goes word-only. The text is
    never logged."""
    import numpy as np
    cfg = config()
    key = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    now = _STATE["clock"]()
    with _locked():
        hit = _QCACHE.get(key)
        if hit is not None and hit[0] >= now:
            _QCACHE.move_to_end(key)
            return hit[1], hit[2], 0
        if now < _STATE["cool_until"]:
            return None, "cooldown:" + str(_STATE["cool_code"]), 0
        day_used = _STATE["neurons_today"]
    emb, why = _embedder()
    if emb is None:
        return None, "embedder:" + str(why), 0
    from vera.chunk_taxo_embed import Deferred, EmbedError
    if day_used is not None and hasattr(emb, "set_day_used"):
        emb.set_day_used(day_used)
    t0 = time.perf_counter()
    vec, code = None, "ok"
    try:
        out = _embed_deadline(emb, str(text), float(cfg["timeout_s"]))
        v = np.asarray(out[0], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or n <= 0:
            raise EmbedError("bad_response")
        vec = v / n
    except Deferred as d:
        code = "deferred:" + str(getattr(d, "code", "deferred"))
        _cool(code, getattr(d, "retry_in_s", 3600.0))
    except EmbedError as e:
        code = "embed_error:" + str(e)
        _cool(code, cfg["cooldown_s"])
    except Exception as e:  # noqa: BLE001
        code = "embed_error:" + type(e).__name__
        _cool(code, cfg["cooldown_s"])
    ms = int(round((time.perf_counter() - t0) * 1000))
    if code != "ok":
        log.info("hybrid retrieval fell back to word-only search (%s)", code)
    with _locked():
        _QCACHE[key] = (now + float(cfg["query_cache_ttl_s"]), vec, code, ms)
        _QCACHE.move_to_end(key)
        while len(_QCACHE) > int(cfg["query_cache_max"]):
            _QCACHE.popitem(last=False)
    return vec, code, ms


# ------------------------------------------------------------------------------------------------ vector index
class _DbLoader:
    """Read-only DB access: the fulldoc read-only account (env var NAMED by config/fulldoc_answer.json db_url_env, default
    VERA_DB_URL_RO; the value is never logged). Two fixed, parameterised SELECTs; one short-lived engine per build."""

    def __init__(self, env=None):
        self.env = env if env is not None else os.environ

    def _engine(self):
        from vera import fulldoc_corpus, fulldoc_db
        cfg = fulldoc_corpus.load_config()
        url = (self.env.get(str(cfg.get("db_url_env") or fulldoc_db.DEFAULT_URL_ENV)) or "").strip()
        if not url:
            raise RuntimeError("no_ro_connection")
        from sqlalchemy import create_engine
        ms = int(cfg.get("db_statement_timeout_ms") or 15000)
        return create_engine(url, connect_args={"connect_timeout": 5, "options": f"-c statement_timeout={ms}"},
                             hide_parameters=True, pool_pre_ping=False)

    def _rows(self, sql: str, params: dict) -> list:
        from sqlalchemy import text
        eng = self._engine()
        try:
            with eng.connect() as conn:
                return list(conn.execute(text(sql), params).fetchall())
        finally:
            eng.dispose()

    def vectors(self, ids, model):
        for r in self._rows(SQL_VECTORS, {"model": model, "ids": [int(i) for i in ids]}):
            yield int(r[0]), str(r[1]), r[2]

    def neurons_today(self) -> float:
        rows = self._rows(SQL_NEURONS_TODAY, {})
        return float(rows[0][0]) if rows and rows[0][0] is not None else 0.0


def _loader():
    return _STATE["loader"] or _DbLoader()


class VectorIndex:
    """Stored-partition vectors of one served corpus: rows (document_id, start, end) and a float32 unit matrix."""

    def __init__(self, model: str, rows: list, mat, built_at: float, n_spans: int):
        self.model, self.rows, self.mat, self.built_at, self.n_spans = model, rows, mat, built_at, n_spans
        self.dim = int(mat.shape[1]) if mat is not None and mat.ndim == 2 else 0
        by_doc: dict = {}
        for i, (did, s, e) in enumerate(rows):
            by_doc.setdefault(did, []).append((s, e, i))
        self._by_doc = {d: (sorted(v), [x[0] for x in sorted(v)]) for d, v in by_doc.items()}
        self._maps: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._mlock = threading.Lock()

    def __len__(self):
        return len(self.rows)

    def nbytes(self) -> int:
        return int(self.mat.nbytes) if self.mat is not None else 0

    def part_of(self, did, start: int, end: int) -> int:
        """Row index of the partition holding the chunk's midpoint, else -1."""
        ent = self._by_doc.get(did)
        if ent is None:
            return -1
        parts, starts = ent
        mid = (int(start) + int(end)) // 2
        j = bisect.bisect_right(starts, mid) - 1
        if j >= 0 and parts[j][0] <= mid < parts[j][1]:
            return parts[j][2]
        return -1

    def chunk_map(self, dar, key_to_doc: dict):
        """(part index per chunk of dar._all as int32 array, chunk indexes per partition). Cached per chunk list."""
        import numpy as np
        allc = dar._all
        key = (id(allc), len(allc))
        with _locked(self._mlock):
            hit = self._maps.get(key)
            if hit is not None:
                return hit
        pidx = np.full(len(allc), -1, dtype=np.int32)
        per = [[] for _ in range(len(self.rows))]
        for c in allc:
            did = key_to_doc.get(c.doc["source_key"])
            p = self.part_of(did, c.start, c.end) if did is not None else -1
            if p >= 0:
                pidx[c.idx] = p
                per[p].append(c.idx)
        out = (pidx, per)
        with _locked(self._mlock):
            self._maps[key] = out
            while len(self._maps) > 4:
                self._maps.popitem(last=False)
        return out


def _spans_of(corpus) -> list:
    out = []
    for did, st in (getattr(corpus, "stamps", None) or {}).items():
        if did not in getattr(corpus, "_texts", {}):
            continue
        for x in (st or {}).get("corpus_stamps") or ():
            sid, s, e = x.get("span_id"), x.get("start"), x.get("end")
            if isinstance(sid, int) and not isinstance(sid, bool) and isinstance(s, int) and isinstance(e, int) and e > s:
                out.append((sid, did, s, e))
    out.sort()
    return out


def build_index(corpus, model: str, loader=None) -> tuple:
    """(VectorIndex | None, code). Loads vectors page by page (each page converted to float32 at once, so the transient
    Python lists stay at one page), checks text_sha256 against the served text, and refuses above max_index_mb."""
    import numpy as np
    cfg = config()
    loader = loader or _loader()
    spans = _spans_of(corpus)
    if not spans:
        return None, "no_stored_spans"
    by_id = {sid: (did, s, e) for sid, did, s, e in spans}
    n_chars = _text_chars()
    cap = int(float(cfg["max_index_mb"]) * 1024 * 1024)
    page = int(cfg["page_ids"])
    rows, mat, n, dim = [], None, 0, None
    ids = [sid for sid, _d, _s, _e in spans]
    stale = 0
    for lo in range(0, len(ids), page):
        for sid, sha, vec in loader.vectors(ids[lo:lo + page], model):
            loc = by_id.get(int(sid))
            if loc is None or vec is None:
                continue
            did, s, e = loc
            if hashlib.sha256(corpus._texts[did][s:e][:n_chars].encode("utf-8")).hexdigest() != sha:
                stale += 1  # the stored vector is of other text: not used until the job re-embeds it
                continue
            v = np.asarray(vec, dtype=np.float32)
            if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
                continue
            if dim is None:
                dim = int(v.size)
                if len(ids) * dim * 4 > cap:
                    return None, "index_too_large"
                mat = np.empty((len(ids), dim), dtype=np.float32)
            if v.size != dim:
                continue
            nv = float(np.linalg.norm(v))
            if nv <= 0:
                continue
            mat[n] = v / nv
            rows.append(loc)
            n += 1
    if n == 0:
        return None, "no_vectors"
    idx = VectorIndex(model, rows, mat[:n], _STATE["wall"](), len(spans))
    idx.stale = stale
    return idx, "ok"


def _refresh_neurons(loader) -> None:
    try:
        v = float(loader.neurons_today())
        with _locked():
            _STATE["neurons_today"] = v
    except Exception as exc:  # noqa: BLE001
        log.info("hybrid neurons_today unavailable (%s)", type(exc).__name__)


def _build_into(corpus, model: str, hs: dict) -> None:
    try:
        loader = _loader()
        _refresh_neurons(loader)
        idx, code = build_index(corpus, model, loader)
        if idx is not None and getattr(corpus, "_dar", None) is not None:
            idx.chunk_map(corpus._dar, corpus._key_to_doc)  # pre-warm the chunk map here, not on the request path
        with hs["lock"]:
            if idx is not None:
                hs["index"] = idx
            hs["code"] = code
            hs["at"] = _STATE["clock"]()
        log.info("hybrid vector index %s (%d rows)", code, len(idx) if idx is not None else 0)
    except Exception as exc:  # noqa: BLE001
        with hs["lock"]:
            hs["code"] = "index_error:" + type(exc).__name__
            hs["at"] = _STATE["clock"]()
        log.warning("hybrid vector index build failed (%s)", type(exc).__name__)
    finally:
        with hs["lock"]:
            hs["building"] = False


def _corpus_state(corpus) -> dict:
    with _locked():
        hs = corpus.__dict__.get("_hybrid_state")
        if hs is None or hs.get("owner") != id(corpus):
            hs = {"owner": id(corpus), "lock": threading.Lock(), "index": None, "code": "not_ready", "at": None,
                  "building": False}
            corpus.__dict__["_hybrid_state"] = hs
        return hs


def ensure_index(corpus, model: str, sync: bool = False):
    """(VectorIndex | None, code). Starts a background (re)build when none exists or the index is older than
    index_refresh_s; the request never waits for it (sync=True is for tests and the measurement script)."""
    hs = _corpus_state(corpus)
    now = _STATE["clock"]()
    refresh_s = float(config()["index_refresh_s"])
    start = False
    with _locked(hs["lock"]):
        idx = hs["index"]
        if idx is not None and idx.model != model:
            hs["index"], idx = None, None
        due = hs["at"] is None or now - hs["at"] >= refresh_s
        if due and not hs["building"]:
            hs["building"] = True
            start = True
    if start:
        if sync:
            _build_into(corpus, model, hs)
        else:
            threading.Thread(target=_build_into, args=(corpus, model, hs), name="vera-hybrid-index", daemon=True).start()
    with _locked(hs["lock"]):
        return hs["index"], ("ok" if hs["index"] is not None else hs["code"])


# ------------------------------------------------------------------------------------------------ retrieval hook
def vector_fn(corpus, question: str):
    """Called by FullDocCorpus.retrieve when enabled(). Returns (fn | None, sim_of | None) and sets this thread's status.
    fn(dar, scores) -> ranked chunk list for the vector branch; sim_of(chunk_idx) -> cosine or None. vector_fn itself never raises (fn may; its call site catches)."""
    try:
        _begin_ask_budget(LOCK_BUDGET_S)  # HYBRID-HARDEN-1109: the ask's lock-wait budget (re-read from config just below)
        cfg = config()
        _begin_ask_budget(float(cfg.get("lock_budget_s", LOCK_BUDGET_S)))
        if getattr(corpus, "_dar", None) is None or not getattr(corpus, "stamps", None):
            _set_status(status="no_stored_spans", vectors=0)
            return None, None
        model = model_name()
        if model is None:
            _, why = _embedder()
            _set_status(status="embedder:" + str(why), vectors=0)
            return None, None
        idx, code = ensure_index(corpus, model)
        if idx is None:
            _set_status(status=code, vectors=0)
            return None, None
        q, qcode, ms = query_vector(question)
        if q is None:
            _set_status(status=qcode, vectors=len(idx), embed_ms=ms)
            return None, None
        if q.size != idx.dim:
            _set_status(status="dim_mismatch", vectors=len(idx), embed_ms=ms)
            return None, None
        sims = idx.mat @ q
        vk = cfg["vector_k"]
        per_part = int(cfg["chunks_per_partition"])
        state = {}

        def sim_of(i):
            pidx = state.get("pidx")
            if pidx is None or i is None or i >= len(pidx) or pidx[i] < 0:
                return None
            return float(sims[pidx[i]])

        def psg_of(i):  # stage 2: the passage's own cosine, only for re-ranked passages
            return state.get("psg", {}).get(i)
        sim_of.psg = psg_of

        def fn(dar, scores):
            import numpy as np
            pidx, per = idx.chunk_map(dar, corpus._key_to_doc)
            state["pidx"] = pidx
            k = int(vk) if vk is not None else int(dar.ccfg["branch_k"])
            mask, allc = dar._mask, dar._all
            out = []
            for p in np.argsort(-sims, kind="stable"):
                cs = [i for i in per[int(p)] if i < len(allc) and (mask is None or mask[i])]
                if not cs:
                    continue
                cs.sort(key=lambda i: (-scores.get(i, 0.0), allc[i].start))
                out.extend(allc[i] for i in cs[:per_part])
                if len(out) >= k:
                    break
            out = out[:k]
            rr = cfg["rerank"]
            if rr.get("enabled") is not True:
                return out
            ranked = _rerank(q, sims, pidx, per, dar, scores, rr, int(cfg["rrf_k"]), state)
            fn.reserve = ranked[:int(rr["reserve_slots"])]
            if not ranked:
                return out  # stage-2 fallback: the stage-1 vector list
            seen = {c.idx for c in ranked}
            return (ranked + [c for c in out if c.idx not in seen])[:k]

        _set_status(status="ok", vectors=len(idx), embed_ms=ms)
        return fn, sim_of
    except Exception as exc:  # noqa: BLE001  any failure => word-only search
        _set_status(status="error:" + type(exc).__name__, vectors=0)
        log.warning("hybrid retrieval error, word-only search (%s)", type(exc).__name__)
        return None, None


def fuse_order(items: list, lex_key, vec_key, k: int) -> list:
    """Reciprocal rank fusion of the lexical order (all items, by lex_key) and one or more vector orders (vec_key: one
    callable or a list; each order holds the items whose key is not None, by descending key, ties in lexical order).
    Ties keep lexical order."""
    lex = sorted(items, key=lex_key)
    pos = {id(t): r for r, t in enumerate(lex, start=1)}
    sc = {id(t): 1.0 / (k + pos[id(t)]) for t in lex}
    for key in (vec_key if isinstance(vec_key, (list, tuple)) else [vec_key]):
        vec = sorted((t for t in lex if key(t) is not None), key=lambda t, key=key: (-key(t), pos[id(t)]))
        for r, t in enumerate(vec, start=1):
            sc[id(t)] += 1.0 / (k + r)
    return sorted(lex, key=lambda t: (-sc[id(t)], pos[id(t)]))


# ------------------------------------------------------------------------------------------------ stage 2: passage re-rank
def _rr_status(**kw) -> None:
    st = getattr(_TL, "status", None)
    if isinstance(st, dict):
        st["rerank"] = dict(kw)


def passage_vectors(texts: list, model: str):
    """(float32 unit matrix rows in `texts` order | None, code, neurons_est). Cached per (sha256(text), model) in an LRU of
    rerank.cache_max entries (process memory only; no DB writes). The uncached texts go out in ONE batched call (at most
    rerank.max_passages <= 64, the adapter batch size), checked against the ask-time share of the daily budget first.
    Embedding failures are returned as a code, not raised. It can raise TimeoutError (hybrid_lock) when a hybrid lock is held
    elsewhere past the ask's lock budget; the stage-2 caller sits under vector_fn's retrieve try and the ask goes word-only.
    Texts are never logged."""
    import numpy as np
    rr = config()["rerank"]
    keys = [(hashlib.sha256(t.encode("utf-8")).hexdigest(), model) for t in texts]
    now = _STATE["clock"]()
    with _locked():
        have = {k: _PCACHE[k] for k in keys if k in _PCACHE}
        for k in have:
            _PCACHE.move_to_end(k)
        cool = now < _STATE["rr_cool_until"]
        cool_code = _STATE["rr_cool_code"]
    miss = [i for i, k in enumerate(keys) if k not in have]
    est = 0.0
    if miss:
        if cool:
            return None, "cooldown:" + str(cool_code), 0.0
        emb, why = _embedder("emb2")
        if emb is None:
            return None, "embedder:" + str(why), 0.0
        if getattr(emb, "name", model) != model:
            return None, "model_mismatch", 0.0
        mtexts = list(dict.fromkeys(texts[i] for i in miss))
        est = neurons_estimate(mtexts, emb)
        if _ask_used() + est > _ask_share():
            code = "deferred:ask_share_reached"
            with _locked():
                _STATE["rr_cool_until"] = now + _seconds_to_midnight()
                _STATE["rr_cool_code"] = code
            return None, code, est
        from vera.chunk_taxo_embed import Deferred, EmbedError
        with _locked():
            day_used = _STATE["neurons_today"]
        if day_used is not None and hasattr(emb, "set_day_used"):
            emb.set_day_used(day_used)
        code, wait = None, float(rr["cooldown_s"])
        try:
            out = _embed_many_deadline(emb, mtexts, float(rr["timeout_s"]), est)
            if not isinstance(out, list) or len(out) != len(mtexts):
                raise EmbedError("bad_response")
            new = {}
            for t, v in zip(mtexts, out):
                a = np.asarray(v, dtype=np.float32)
                n = float(np.linalg.norm(a))
                if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or n <= 0:
                    raise EmbedError("bad_response")
                new[(hashlib.sha256(t.encode("utf-8")).hexdigest(), model)] = a / n
        except Deferred as d:
            code, wait = "deferred:" + str(getattr(d, "code", "deferred")), float(getattr(d, "retry_in_s", 3600.0))
        except EmbedError as e:
            code = "embed_error:" + str(e)
        except Exception as e:  # noqa: BLE001
            code = "embed_error:" + type(e).__name__
        if code is not None:
            with _locked():
                _STATE["rr_cool_until"] = now + max(0.0, wait)
                _STATE["rr_cool_code"] = code
            log.info("hybrid passage re-rank fell back to stage 1 (%s)", code)
            return None, code, est
        have.update(new)
        cap = int(rr["cache_max"])
        with _locked():
            for k2, v in new.items():
                if cap > 0:
                    _PCACHE[k2] = v
                    _PCACHE.move_to_end(k2)
            while len(_PCACHE) > cap:
                _PCACHE.popitem(last=False)
    rows = [have[k] for k in keys]
    if len({r.size for r in rows}) != 1:
        return None, "dim_mismatch", est
    return np.vstack(rows), "ok", est


def _seconds_to_midnight() -> float:
    from vera.chunk_taxo_embed import seconds_to_utc_midnight
    return seconds_to_utc_midnight(_STATE["wall"]())


def _rerank(q, sims, pidx, per, dar, scores, rr: dict, k_rrf: int, state: dict) -> list:
    """Stage 2: the top rerank.top_partitions partitions by RRF of (partition cosine order, partition best-BM25 order);
    their retrieval chunks (masked, in position order, at most max_passages, text cut to max_chars) are embedded and
    ranked by cosine to the question. Returns the ranked chunks, or [] on any fallback (status traced)."""
    import numpy as np
    t0 = time.perf_counter()
    mask, allc = dar._mask, dar._all
    best: dict = {}
    for i, sc in scores.items():
        if i < len(pidx) and pidx[i] >= 0 and (mask is None or mask[i]) and sc > best.get(int(pidx[i]), 0.0):
            best[int(pidx[i])] = sc
    vec_order = [int(p) for p in np.argsort(-sims, kind="stable")]
    lex_order = sorted(best, key=lambda p: (-best[p], p))
    fs: dict = {}
    for order in (vec_order, lex_order):
        for r, p in enumerate(order, start=1):
            fs[p] = fs.get(p, 0.0) + 1.0 / (k_rrf + r)
    vpos = {p: r for r, p in enumerate(vec_order)}
    top = sorted(fs, key=lambda p: (-fs[p], vpos.get(p, 0)))[:int(rr["top_partitions"])]
    cand = []
    for p in top:
        cand.extend(sorted((i for i in per[p] if i < len(allc) and (mask is None or mask[i])), key=lambda i: allc[i].start))
    cand = cand[:int(rr["max_passages"])]
    if not cand:
        _rr_status(status="no_passages", partitions=len(top), passages=0)
        return []
    texts = [allc[i].text[:int(rr["max_chars"])] for i in cand]
    model = model_name()
    mat, code, est = passage_vectors(texts, model) if model else (None, "embedder:none", 0.0)
    ms = int(round((time.perf_counter() - t0) * 1000))
    if mat is None or mat.shape[1] != q.size:
        _rr_status(status=code if mat is None else "dim_mismatch", partitions=len(top), passages=len(cand),
                   neurons_est=round(est, 3), ms=ms)
        return []
    ps = mat @ q
    state["psg"] = {i: float(v) for i, v in zip(cand, ps)}
    order = sorted(range(len(cand)), key=lambda j: (-float(ps[j]), allc[cand[j]].start))
    _rr_status(status="ok", partitions=len(top), passages=len(cand), neurons_est=round(est, 3), ms=ms)
    return [allc[cand[j]] for j in order]

