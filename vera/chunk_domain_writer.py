"""WP-D-1113 (plan work/CHUNK-ASSOC-1113, D4, OD-A, Amendment 1 A4): writes one tax_chunk_domain row per chunk that has a strict taxonomy
link, in the same transaction as that chunk's topic and sub-topic links (called from vera/chunk_taxo_job.py _write_chunk_page).

Derivation: a chunk's domain is the parent domain of its deepest STRICT node. The core's association item already carries it
(`assoc[span]["domain"]` = parent of the deepest strict node through the model hierarchy, the same hierarchy the job writes to
tax_domain_topic), so no extra read is needed. Rules:
  * strict only: a chunk whose item has no `level` (abstained) gets no domain row; routing-only `multi` links never create a domain row;
  * one row per chunk (a multi-topic chunk gets the domain of its strict node only);
  * a chunk the core decided at level `domain` (which stores no other link) gets a strict `cosine` row scored from its distance;
    every other chunk gets a `cosine_lvl` row scored by its own cosine to the domain's center, defined as in plan A4 K7: the unit mean of
    the unit vectors of the chunks whose strict decision falls in that domain (NOT the core's cluster-group center, which averages all
    chunks of the group; the two differ, so this module computes its own from the build's vectors while they are in memory, one float per
    labelled chunk); a score that cannot be computed (zero vector, no usable center) is stored as NULL, which reads as strict;
  * rows of an earlier classifier version for the page's chunks are retired first (same classifier_name, `classifier_version <> :version`),
    exactly as the topic links are, so a chunk that lost its label or changed domain does not keep a stale row;
  * the table is probed once per pass (probe_chunk_domain: to_regclass plus column INSERT and UPDATE(state) privileges). Absent, or a
    privilege missing (migration 027 not applied): nothing is written, the run row records domain_table_absent 1 and
    domain_rows_skipped_no_table (rows that would have been written). Present: domain_table_absent 0;
  * kill switch: job config `chunk_domain_write` false (literal) disables the module (no probe, no write, domain_write_disabled 1).
Only fixed statements of vera/chunk_taxo_store.py are used (probe_chunk_domain, link_retire_chunk_domain, link_insert_chunk_domain). No DDL, no
DELETE. Run-row metrics are numbers only: domain_table_absent, domain_rows_skipped_no_table, domain_rows_attempted (rows submitted; ON
CONFLICT DO NOTHING hides conflicts), domain_chunks_without_domain (stored strict link but no domain id resolved), domain_rows_retired,
domain_write_disabled.
"""
from __future__ import annotations

import math

from vera import chunk_taxo_store as store

KIND = "chunk_domain"


class State:
    """Per-pass state: table_ok is None until probed, then True or False."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.table_ok = None
        self.attempted = 0
        self.skipped = 0
        self.no_domain = 0
        self.retired = 0


def new_state(enabled: bool = True) -> State:
    return State(enabled)


def probe(conn, st: State) -> bool:
    if st.table_ok is None:
        row = store.run_sql(conn, "probe_chunk_domain").first()
        st.table_ok = bool(row[0]) if row is not None else False
    return st.table_ok


def _get(o, name, default=None):
    if isinstance(o, dict):
        return o.get(name, default)
    return getattr(o, name, default)


def domain_key(item):
    """Domain key of a chunk's strict decision, or None (abstained, routing-only or no domain)."""
    if not isinstance(item, dict) or item.get("level") not in ("subtopic", "topic", "domain"):
        return None
    key = item.get("domain")
    return key if key is not None else None


def _sim(distance):
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return None
    return round(max(-1.0, min(1.0, 1.0 - d)), 6) if math.isfinite(d) else None


def domain_scores(model, vecs) -> dict:
    """{span_id: (cosine, score_kind)} for every chunk with a strict decision whose domain center is known. `vecs` is aligned with
    model.span_order (the contract of core.quality). Never raises: anything unexpected yields fewer scores (stored as NULL)."""
    out: dict = {}
    try:
        order, assoc = list(_get(model, "span_order") or ()), _get(model, "assoc") or {}
        unit, sums = {}, {}
        for i, sid in enumerate(order):  # pass 1: the plan's center (A4 K7, D4): unit mean of the vectors of the domain's STRICT members
            key = domain_key(assoc.get(sid))
            if key is None:
                continue
            v = [float(x) for x in vecs[i]]
            norm = math.sqrt(sum(x * x for x in v))
            if not norm > 0.0:
                continue
            u = [x / norm for x in v]
            unit[sid] = u
            acc = sums.setdefault(key, [0.0] * len(u))
            if len(acc) != len(u):
                continue
            for j, x in enumerate(u):
                acc[j] += x
        centers = {}
        for key, acc in sums.items():
            cn = math.sqrt(sum(x * x for x in acc))
            if cn > 0.0:
                centers[key] = [x / cn for x in acc]
        for sid in order:  # pass 2: own cosine to that center
            item = assoc.get(sid)
            key = domain_key(item)
            if key is None:
                continue
            if item.get("level") == "domain":
                s = _sim(item.get("distance"))
                if s is not None:
                    out[sid] = (s, "cosine")
                continue
            cen, u = centers.get(key), unit.get(sid)
            if cen is None or u is None or len(cen) != len(u):
                continue
            out[sid] = (round(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, cen)))), 6), "cosine_lvl")
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return {}
    return out


def _score(scores, sid):
    if not scores or sid not in scores:
        return None, None
    sim, kind = scores[sid]
    if isinstance(sim, (int, float)) and math.isfinite(sim) and isinstance(kind, str):
        return round(float(sim), 6), kind
    return None, None


def write_page(conn, st: State, spans, assoc, ids, version, scores=None) -> None:
    """Retire the page's old-version domain rows and insert the new ones. `ids` is the job's {level: {model key: entity id}}.
    Disabled: nothing at all. Table absent: only the skipped count moves."""
    if not st.enabled:
        return
    rows, no_domain = [], 0
    for sid in spans:
        key = domain_key(assoc.get(sid))
        if key is None:
            continue
        eid = ids.get("domain", {}).get(key)
        if eid is None:
            no_domain += 1
            continue
        sim, kind = _score(scores, sid)
        rows.append({"a": sid, "b": eid, "version": version, "score": sim, "score_kind": kind})
    if not probe(conn, st):
        st.skipped += len(rows)
        return
    res = store.run_sql(conn, "link_retire_chunk_domain", {"version": version, "ids": list(spans)})
    n = getattr(res, "rowcount", None)
    if isinstance(n, int) and n > 0:
        st.retired += n
    st.no_domain += no_domain
    st.attempted += len(rows)
    store.run_sql(conn, "link_insert_chunk_domain", rows)


def metrics(st: State) -> dict:
    """Numbers for the run row (merged into metrics_run by the job)."""
    if not st.enabled:
        return {"domain_write_disabled": 1}
    if st.table_ok is None:  # no chunk page was written this pass: nothing was probed, nothing is claimed
        return {}
    if not st.table_ok:
        return {"domain_table_absent": 1, "domain_rows_skipped_no_table": st.skipped}
    return {"domain_table_absent": 0, "domain_rows_attempted": st.attempted, "domain_chunks_without_domain": st.no_domain,
            "domain_rows_retired": st.retired}
