"""TAXO-ROUTE-1110 R3: taxonomy router, SHADOW mode.

Purpose (owner 2026-11-10): the taxonomy conserves effort by excluding clearly unrelated domains from retrieval, without
pigeon-holing (D2, D3: multi-membership, generous margins, recall over precision).

Stage: `mode` off (default) | shadow | on. Shadow computes the routed set for an ask and records counts and recall against
the global candidates; it NEVER changes retrieval results (mode "on" is reserved and behaves as shadow here).

Design:
* Domain centers are rebuilt from the in-memory hybrid vector matrix and the active chunk-to-domain links (no extra
  embedding call, no extra vectors). A partition's domain set is a uint32 bitmask (at most 32 domains), so the whole router
  is a few KB plus one uint32 per partition.
* Route: every domain with cosine >= best - route_rel_margin and >= route_abs_floor, plus neighbour domains (center cosine
  >= neighbour_cos to a selected one). Always-in set U: chunks with no vector (pidx < 0) and partitions with no link.
* Global fallback (route = "global:<reason>") when: router not loaded, version mismatch, no question vector, no domain fits,
  no domain clearly excluded, or fewer than min_routed_scored routed chunks have a keyword score, or over the time budget,
  or any error. The router takes no lock on the request path beyond what the caller holds (the router object is built before
  the index is published and is immutable); it never blocks an ask.
Records hold codes and counts only; never question text.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "taxo_router.json"
MODES = ("off", "shadow", "on")
DEFAULTS = {"mode": "off", "route_rel_margin": 0.10, "route_abs_floor": 0.20, "neighbour_cos": 0.70,
            "min_routed_scored": 40, "time_budget_ms": 50}
_BOUNDS = {"route_rel_margin": (0.0, 2.0), "route_abs_floor": (-1.0, 1.0), "neighbour_cos": (-1.0, 1.0),
           "min_routed_scored": (0, 100000), "time_budget_ms": (1, 10000)}
MAX_DOMAINS = 32

# One row per (span, domain): the best link cosine. Chunk links of kind 'cosine%' (strict 'cosine', generous 'cosine_r<n>', and
# routing-only 'cosine_soft') and legacy NULL-kind rows count; any other kind never does. Domains come from topic links (no
# chunk-domain table). Subtopic links reach their topic through tax_topic_subtopic. Only active entities and links.
SQL_DOMAIN_LINKS = (
    "SELECT l.span_id, dt.domain_id, max(l.score) FROM ("
    "SELECT ct.span_id, ct.topic_id, ct.score FROM vera_vjay.tax_chunk_topic ct "
    "WHERE ct.state = 'active' AND (ct.score_kind IS NULL OR ct.score_kind LIKE 'cosine%') "
    "UNION ALL SELECT cs.span_id, ts.topic_id, cs.score FROM vera_vjay.tax_chunk_subtopic cs "
    "JOIN vera_vjay.tax_topic_subtopic ts ON ts.subtopic_id = cs.subtopic_id AND ts.state = 'active' "
    "JOIN vera_vjay.tax_subtopics sb ON sb.subtopic_id = cs.subtopic_id AND sb.state = 'active' "
    "WHERE cs.state = 'active' AND (cs.score_kind IS NULL OR cs.score_kind LIKE 'cosine%')) l "
    "JOIN vera_vjay.tax_topics t ON t.topic_id = l.topic_id AND t.state = 'active' "
    "JOIN vera_vjay.tax_domain_topic dt ON dt.topic_id = l.topic_id AND dt.state = 'active' "
    "JOIN vera_vjay.tax_domains d ON d.domain_id = dt.domain_id AND d.state = 'active' "
    "WHERE l.span_id = ANY(:ids) GROUP BY l.span_id, dt.domain_id")

_STATE: dict = {"cfg": None}
_TL = threading.local()


# ------------------------------------------------------------------------------------------------ config
def _clean(raw) -> dict:
    c = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return c
    if raw.get("mode") in MODES:
        c["mode"] = raw["mode"]
    for k, (lo, hi) in _BOUNDS.items():
        v = raw.get(k, DEFAULTS[k])
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi:
            c[k] = v
    return c


def config() -> dict:
    """Validated config, cached (lock-free: the dict is replaced, never mutated). Bad file => mode off. Never raises."""
    c = _STATE["cfg"]
    if c is None:
        try:
            c = _clean(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            c = _clean({})
        _STATE["cfg"] = c
    return c


def mode() -> str:
    try:
        return config()["mode"]
    except Exception:  # noqa: BLE001
        return "off"


def set_config_for_tests(raw: dict | None) -> None:
    _STATE["cfg"] = None if raw is None else _clean(raw)


# ------------------------------------------------------------------------------------------------ usage hook
def assoc_hook(span_id: int, domain_key, similarity: float) -> tuple:
    """=== USAGE-DECAY INTEGRATION HOOK (owner rule 2026-11-10; CHUNK-USAGE branch chunk-usage-1110) ===
    Returns (weight, removed) for one (chunk partition, domain) association. `similarity` is the stored link cosine (the
    initial weight, owner rule); 1.0 when the link has no score. Frequently matched but unused associations decay in weight
    and, below a threshold, are soft-removed; the chunk itself stays globally searchable (it falls into the always-in set U and
    the lexical walk). Default: weight = similarity, nothing removed. Replace the body with a read of the chunk_usage API
    later; do not import chunk_usage here until then. Applied at router build time only."""
    return similarity, False


# ------------------------------------------------------------------------------------------------ build
class Router:
    """Immutable after build. Tied to one VectorIndex (checked by identity and size)."""

    def __init__(self, idx, domains, centers, part_mask, neighbour_mask, n_links, job_state):
        self.idx_id, self.n_rows = id(idx), len(idx)
        self.domains, self.centers, self.part_mask, self.neighbour_mask = domains, centers, part_mask, neighbour_mask
        self.n_links = n_links
        js = job_state if isinstance(job_state, dict) else {}
        self.job_version = js.get("version")
        self.display_ok = js.get("display_ok") is True  # the taxonomy display gate of the job run the links came from

    def nbytes(self) -> int:
        return int(self.centers.nbytes + self.part_mask.nbytes)


def build(idx, links, cfg: dict | None = None, job_state: dict | None = None):
    """Router | None. links: iterable of (span_id, domain_key[, link_cosine]). Needs idx.span_ids (parallel to idx.rows).
    job_state: {"version", "display_ok"} of the job run the links belong to. Returns None when there is nothing usable
    (no links, over MAX_DOMAINS). Centers are weighted by the link cosine, via a dense weight vector (no row copies)."""
    import numpy as np
    cfg = cfg or config()
    sids = getattr(idx, "span_ids", None)
    if not sids or len(sids) != len(idx):
        return None
    row_of = {int(s): i for i, s in enumerate(sids)}
    dom_ids, per_dom = {}, {}
    n_links = 0
    for item in links:
        sid, dk = item[0], item[1]
        sim = item[2] if len(item) > 2 else None
        i = row_of.get(int(sid))
        if i is None:
            continue
        sim = float(sim) if isinstance(sim, (int, float)) and not isinstance(sim, bool) and math.isfinite(sim) else 1.0
        w, removed = assoc_hook(int(sid), dk, sim)
        if removed:
            continue
        j = dom_ids.setdefault(dk, len(dom_ids))
        if j >= MAX_DOMAINS:
            return None
        per_dom.setdefault(j, {})[i] = max(float(w), 1e-3)  # a link always counts as membership; weight only shapes the center
        n_links += 1
    if not per_dom:
        return None
    nd = len(dom_ids)
    centers = np.zeros((nd, idx.dim), dtype=np.float32)
    part_mask = np.zeros(len(idx), dtype=np.uint32)
    for j, members in per_dom.items():
        rows = np.fromiter(members.keys(), dtype=np.int64, count=len(members))
        wvec = np.zeros(len(idx), dtype=np.float32)
        wvec[rows] = np.fromiter(members.values(), dtype=np.float32, count=len(members))
        c = wvec @ idx.mat
        nrm = float(np.linalg.norm(c))
        if nrm > 0:
            centers[j] = c / nrm
        part_mask[rows] |= np.uint32(1 << j)
    cc = centers @ centers.T
    nb = np.zeros(nd, dtype=np.uint32)
    for a in range(nd):
        for b in range(nd):
            if a != b and cc[a, b] >= float(cfg["neighbour_cos"]):
                nb[a] |= np.uint32(1 << b)
    keys = [None] * nd
    for k, j in dom_ids.items():
        keys[j] = k
    return Router(idx, keys, centers, part_mask, nb, n_links, job_state)


# ------------------------------------------------------------------------------------------------ route
def route(router, idx, q, pidx, scores: dict, cfg: dict | None = None) -> dict:
    """Decision dict (codes and counts only): {"route": "routed"|"global:<reason>", "R": bool array over chunks | None,
    "n_domains", "n_selected", "n_routed", "n_chunks"}. Never raises."""
    import numpy as np
    cfg = cfg or config()
    out = {"route": "global:error", "R": None, "n_domains": 0, "n_selected": 0, "n_routed": 0, "n_chunks": 0, "n_unlinked": 0}
    try:
        if router is None:
            out["route"] = "global:router_not_loaded"
            return out
        if router.idx_id != id(idx) or router.n_rows != len(idx):
            out["route"] = "global:version_mismatch"
            return out
        # TODO(pre-"on", R2-4): during a taxonomy job pass the links can mix old and new versions; the build-time
        # before/after version check only covers one build. Before "on", add a per-ask version check (or rebuild on change).
        pending = None  # a job-version / display-gate fallback still computes the counterfactual routed set below
        if router.job_version is None:
            pending = "global:job_version_unknown"
        elif not router.display_ok:
            pending = "global:display_gate_failed"
        if q is None:
            out["route"] = "global:no_vector"
            return out
        nd = len(router.domains)
        out["n_domains"] = nd
        s = router.centers @ q
        best = float(s.max())
        if best < float(cfg["route_abs_floor"]):
            out["route"] = "global:no_domain_fits"
            return out
        sel = 0
        for j in range(nd):
            if s[j] >= best - float(cfg["route_rel_margin"]) and s[j] >= float(cfg["route_abs_floor"]):
                sel |= 1 << j
                sel |= int(router.neighbour_mask[j])
        n_sel = bin(sel).count("1")
        out["n_selected"] = n_sel
        # partition in R: matches a selected domain, or has no link (always-in set U)
        pm = router.part_mask
        part_in = ((pm & np.uint32(sel)) != 0) | (pm == 0)
        n = len(pidx)
        R = np.ones(n, dtype=bool)  # chunks with no vector (pidx < 0) stay in
        has = pidx >= 0
        R[has] = part_in[pidx[has]]
        out["n_chunks"], out["n_routed"] = n, int(R.sum())
        out["n_unlinked"] = int((~has).sum() + (pm[pidx[has]] == 0).sum())  # |U|: no vector, or no link
        out["R"] = R  # kept for the counterfactual even when the route falls back to global
        if pending:
            out["route"] = pending
            return out
        if n_sel >= nd:
            out["route"] = "global:no_domain_excluded"
            return out
        scored_in = sum(1 for i in scores if i < n and R[i])
        if scored_in < int(cfg["min_routed_scored"]):
            out["route"] = "global:few_routed_scored"
            return out
        out["route"] = "routed"
        return out
    except Exception as exc:  # noqa: BLE001
        out["route"] = "global:error"
        out["R"] = None
        log.info("taxo router error, global (%s)", type(exc).__name__)
        return out


# ------------------------------------------------------------------------------------------------ shadow
# Per-thread (per-ask) record. Cleared at ask start (hybrid_retrieval.clear_status). Codes and counts only: no question text.
def last_shadow() -> dict | None:
    s = getattr(_TL, "shadow", None)
    return dict(s) if isinstance(s, dict) else None


def clear_shadow() -> None:
    _TL.shadow = None
    _TL.R = None
    _TL.cand = None


def note_candidate(span_id, chunk_idx) -> None:
    """Called by the retrieval path for each returned candidate: remembers whether the chunk is inside the routed set R
    (shadow only; no effect on results). Never raises."""
    try:
        R, cand = getattr(_TL, "R", None), getattr(_TL, "cand", None)
        if R is None or cand is None or chunk_idx is None or not (0 <= chunk_idx < len(R)):
            return
        cand[span_id] = bool(R[chunk_idx])
    except Exception:  # noqa: BLE001
        pass


def finish(admitted) -> dict | None:
    """End of ask: count admitted evidence that falls outside the routed set (plan gate Q1). `admitted` holds admitted dicts
    (or bare span ids). A merged passage is checked member by member (merged_span_ids): it counts as missed when any known
    member is outside R, and as unmapped only when no member is known. Returns the record for the ask trace (codes and
    counts only), or None when shadow did not run. Never raises."""
    try:
        rec = getattr(_TL, "shadow", None)
        if not isinstance(rec, dict):
            return None
        cand = getattr(_TL, "cand", None)
        items = list(admitted or ())
        if cand is not None and getattr(_TL, "R", None) is not None:
            missed = unmapped = 0
            for a in items:
                if isinstance(a, dict):
                    members = list(a.get("merged_span_ids") or ()) or [a.get("span_id")]
                else:
                    members = [a]
                known = [m for m in members if m in cand]
                if not known:
                    unmapped += 1
                elif any(not cand[m] for m in known):
                    missed += 1
            rec["admitted_total"], rec["admitted_unmapped"], rec["missed_admitted"] = len(items), unmapped, missed
        else:
            rec["admitted_total"], rec["admitted_unmapped"], rec["missed_admitted"] = len(items), len(items), None
        log.info("taxo router shadow: route=%s selected=%s/%s routed=%s/%s missed_admitted=%s", rec.get("route"),
                 rec.get("n_selected"), rec.get("n_domains"), rec.get("n_routed"), rec.get("n_chunks"),
                 rec.get("missed_admitted"))
        return dict(rec)
    except Exception:  # noqa: BLE001
        return None


def shadow(router, idx, q, pidx, scores: dict, global_idx, branch_k: int) -> None:
    """Compute the routed set and record counts. Never changes anything the caller returns; never raises.
    recall_candidates (top branch_k keyword chunks + the vector list found outside R) is secondary and only for asks that
    would have been routed; a fallback or over-budget ask has recall None plus the reason and, when R was computed, the
    counterfactual missed_candidates. Uses thread CPU time, not process time."""
    try:
        cfg = config()
        if cfg["mode"] == "off":
            return
        clear_shadow()
        t0 = time.thread_time()
        dec = route(router, idx, q, pidx, scores, cfg)
        R = dec.pop("R")
        dt = time.thread_time() - t0
        if R is not None:
            top = heapq.nlargest(int(branch_k), scores.items(), key=lambda kv: kv[1])
            glob = {int(i) for i, _ in top} | {int(i) for i in (global_idx or ())}
            n = len(pidx)
            missed = sum(1 for i in glob if i < n and not R[i])
        else:
            glob, missed = set(), None
        route_code = dec["route"]
        if route_code == "routed" and dt * 1000.0 > float(cfg["time_budget_ms"]):
            route_code = dec["route"] = "global:over_budget"
        dec.update(mode=cfg["mode"], effective="shadow", global_candidates=len(glob), missed_candidates=missed,
                   recall_candidates=(None if route_code != "routed" or missed is None
                                      else (1.0 if not glob else round(1.0 - missed / len(glob), 4))),
                   recall_none_reason=(None if route_code == "routed" else route_code),
                   cpu_thread_s=round(dt, 6))
        _TL.shadow, _TL.R, _TL.cand = dec, R, {}
    except Exception:  # noqa: BLE001
        _TL.shadow = {"route": "global:error", "mode": "shadow", "effective": "shadow", "recall_candidates": None,
                      "recall_none_reason": "global:error"}
        _TL.R, _TL.cand = None, None
