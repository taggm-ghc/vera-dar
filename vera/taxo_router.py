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

SQL_DOMAIN_LINKS = (
    "SELECT DISTINCT cl.span_id, dt.domain_id FROM ("
    "SELECT ct.span_id, ct.topic_id FROM vera_vjay.tax_chunk_topic ct WHERE ct.state = 'active' "
    "UNION SELECT cs.span_id, st.topic_id FROM vera_vjay.tax_chunk_subtopic cs "
    "JOIN vera_vjay.tax_subtopics st ON st.subtopic_id = cs.subtopic_id AND st.state = 'active' "
    "WHERE cs.state = 'active') cl "
    "JOIN vera_vjay.tax_domain_topic dt ON dt.topic_id = cl.topic_id AND dt.state = 'active' "
    "WHERE cl.span_id = ANY(:ids)")

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
def assoc_hook(span_id: int, domain_key) -> tuple:
    """=== USAGE-DECAY INTEGRATION HOOK (owner rule 2026-11-10; CHUNK-USAGE branch chunk-usage-1110) ===
    Returns (weight, removed) for one (chunk partition, domain) association. Frequently matched but unused associations
    decay in weight and, below a threshold, are soft-removed; the chunk itself stays globally searchable (it is still in the
    always-global fallback and the lexical walk). Default: weight 1.0, nothing removed. Replace the body with a read of the
    chunk_usage API later; do not import chunk_usage here until then. Applied at router build time only."""
    return 1.0, False


# ------------------------------------------------------------------------------------------------ build
class Router:
    """Immutable after build. Tied to one VectorIndex (checked by identity and size)."""

    def __init__(self, idx, domains, centers, part_mask, neighbour_mask, n_links):
        self.idx_id, self.n_rows = id(idx), len(idx)
        self.domains, self.centers, self.part_mask, self.neighbour_mask = domains, centers, part_mask, neighbour_mask
        self.n_links = n_links

    def nbytes(self) -> int:
        return int(self.centers.nbytes + self.part_mask.nbytes)


def build(idx, links, cfg: dict | None = None):
    """Router | None. links: iterable of (span_id, domain_key). Needs idx.span_ids (parallel to idx.rows). Returns None when
    there is nothing usable (no links, over MAX_DOMAINS)."""
    import numpy as np
    cfg = cfg or config()
    sids = getattr(idx, "span_ids", None)
    if not sids or len(sids) != len(idx):
        return None
    row_of = {int(s): i for i, s in enumerate(sids)}
    dom_ids, per_dom = {}, {}
    n_links = 0
    for sid, dk in links:
        i = row_of.get(int(sid))
        if i is None:
            continue
        w, removed = assoc_hook(int(sid), dk)
        if removed or not (w > 0):
            continue
        j = dom_ids.setdefault(dk, len(dom_ids))
        if j >= MAX_DOMAINS:
            return None
        per_dom.setdefault(j, {})[i] = float(w)
        n_links += 1
    if not per_dom:
        return None
    nd = len(dom_ids)
    centers = np.zeros((nd, idx.dim), dtype=np.float32)
    part_mask = np.zeros(len(idx), dtype=np.uint32)
    for j, members in per_dom.items():
        rows = np.fromiter(members.keys(), dtype=np.int64, count=len(members))
        w = np.fromiter(members.values(), dtype=np.float32, count=len(members))
        c = (idx.mat[rows] * w[:, None]).sum(axis=0)
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
    return Router(idx, keys, centers, part_mask, nb, n_links)


# ------------------------------------------------------------------------------------------------ route
def route(router, idx, q, pidx, scores: dict, cfg: dict | None = None) -> dict:
    """Decision dict (codes and counts only): {"route": "routed"|"global:<reason>", "R": bool array over chunks | None,
    "n_domains", "n_selected", "n_routed", "n_chunks"}. Never raises."""
    import numpy as np
    cfg = cfg or config()
    out = {"route": "global:error", "R": None, "n_domains": 0, "n_selected": 0, "n_routed": 0, "n_chunks": 0}
    try:
        if router is None:
            out["route"] = "global:router_not_loaded"
            return out
        if router.idx_id != id(idx) or router.n_rows != len(idx):
            out["route"] = "global:version_mismatch"
            return out
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
        if n_sel >= nd:
            out["route"] = "global:no_domain_excluded"
            return out
        # partition in R: matches a selected domain, or has no link (always-in set U)
        pm = router.part_mask
        part_in = ((pm & np.uint32(sel)) != 0) | (pm == 0)
        n = len(pidx)
        R = np.ones(n, dtype=bool)  # chunks with no vector (pidx < 0) stay in
        has = pidx >= 0
        R[has] = part_in[pidx[has]]
        out["n_chunks"], out["n_routed"] = n, int(R.sum())
        scored_in = sum(1 for i in scores if i < n and R[i])
        if scored_in < int(cfg["min_routed_scored"]):
            out["route"] = "global:few_routed_scored"
            return out
        out["route"], out["R"] = "routed", R
        return out
    except Exception as exc:  # noqa: BLE001
        out["route"] = "global:error"
        out["R"] = None
        log.info("taxo router error, global (%s)", type(exc).__name__)
        return out


# ------------------------------------------------------------------------------------------------ shadow
def last_shadow() -> dict | None:
    s = getattr(_TL, "shadow", None)
    return dict(s) if isinstance(s, dict) else None


def clear_shadow() -> None:
    _TL.shadow = None


def shadow(router, idx, q, pidx, scores: dict, global_idx, branch_k: int) -> None:
    """Compute the routed set and record counts and recall against the global candidates (the top branch_k keyword chunks
    plus the vector list, as chunk indexes). Never changes anything the caller returns; never raises."""
    try:
        cfg = config()
        if cfg["mode"] == "off":
            return
        t0 = time.process_time()
        dec = route(router, idx, q, pidx, scores, cfg)
        R = dec.pop("R")
        top = heapq.nlargest(int(branch_k), scores.items(), key=lambda kv: kv[1])
        glob = {int(i) for i, _ in top} | {int(i) for i in (global_idx or ())}
        n = len(pidx)
        missed = 0
        if R is not None:
            missed = sum(1 for i in glob if i < n and not R[i])
        dt = time.process_time() - t0
        if R is not None and dt * 1000.0 > float(cfg["time_budget_ms"]):
            dec["route"] = "global:over_budget"
            missed = 0
        dec.update(mode=cfg["mode"], global_candidates=len(glob), missed=missed,
                   recall=(1.0 if not glob else round(1.0 - missed / len(glob), 4)), process_time_s=round(dt, 6))
        _TL.shadow = dec
    except Exception:  # noqa: BLE001
        _TL.shadow = {"route": "global:error", "mode": "shadow"}
