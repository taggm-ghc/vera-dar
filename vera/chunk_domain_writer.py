"""WP-D-1113 (plan work/CHUNK-ASSOC-1113, D4 and OD-A): writes one tax_chunk_domain row per chunk that has a strict taxonomy link,
in the same transaction as that chunk's topic and sub-topic links (called from vera/chunk_taxo_job.py _write_chunk_page).

Derivation: a chunk's domain is the parent domain of its deepest STRICT node. The core's association item already carries it
(`assoc[span]["domain"]` = parent of the deepest strict node through the model hierarchy, the same hierarchy the job writes to
tax_domain_topic), so no extra read is needed. Rules:
  * strict only: a chunk whose item has no `level` (abstained) gets no domain row; routing-only `multi` links never create a domain row;
  * one row per chunk (a multi-topic chunk gets the domain of its strict node only);
  * score and score_kind are NULL (a NULL score_kind reads as strict, 020 meaning). The per-association cosine to the domain center
    (`cosine_lvl`) belongs to WP-L; this module accepts an optional `scores` mapping {span_id: (cosine, kind)} so WP-L can pass it;
  * rows of an earlier classifier version for the page's chunks are retired first (same classifier_name, `classifier_version <> :version`),
    exactly as the topic links are, so a chunk that lost its label or changed domain does not keep a stale row;
  * the table is probed once per pass (to_regclass). Absent (migration 027 not applied): nothing is written, the run row records
    `domain_table_absent` 1. Present: `domain_table_absent` 0.
Only fixed statements of vera/chunk_taxo_store.py are used (probe_chunk_domain, link_retire_chunk_domain, link_insert_chunk_domain). No DDL, no
DELETE. Run-row metrics are numbers only: domain_table_absent, domain_rows_attempted (rows submitted; ON CONFLICT DO NOTHING hides
conflicts), domain_chunks_without_domain (stored strict link but no domain id resolved), domain_rows_retired.
"""
from __future__ import annotations

import math

from vera import chunk_taxo_store as store

KIND = "chunk_domain"


class State:
    """Per-pass state: table_ok is None until probed, then True or False."""

    def __init__(self):
        self.table_ok = None
        self.attempted = 0
        self.no_domain = 0
        self.retired = 0


def new_state() -> State:
    return State()


def probe(conn, st: State) -> bool:
    if st.table_ok is None:
        row = store.run_sql(conn, "probe_chunk_domain").first()
        st.table_ok = bool(row[0]) if row is not None else False
    return st.table_ok


def domain_key(item):
    """Domain key of a chunk's strict decision, or None (abstained or no domain)."""
    if not isinstance(item, dict) or item.get("level") not in ("subtopic", "topic", "domain"):
        return None
    key = item.get("domain")
    return key if key is not None else None


def _score(scores, sid):
    if not scores or sid not in scores:
        return None, None
    sim, kind = scores[sid]
    if isinstance(sim, (int, float)) and math.isfinite(sim) and isinstance(kind, str):
        return round(float(sim), 6), kind
    return None, None


def write_page(conn, st: State, spans, assoc, ids, version, scores=None) -> None:
    """Retire the page's old-version domain rows and insert the new ones. `ids` is the job's {level: {model key: entity id}}.
    A no-op (apart from the probe) while the table is absent."""
    if not probe(conn, st):
        return
    res = store.run_sql(conn, "link_retire_chunk_domain", {"version": version, "ids": list(spans)})
    n = getattr(res, "rowcount", None)
    if isinstance(n, int) and n > 0:
        st.retired += n
    rows = []
    for sid in spans:
        item = assoc.get(sid)
        key = domain_key(item)
        if key is None:
            continue
        eid = ids.get("domain", {}).get(key)
        if eid is None:
            st.no_domain += 1
            continue
        sim, kind = _score(scores, sid)
        rows.append({"a": sid, "b": eid, "version": version, "score": sim, "score_kind": kind})
    st.attempted += len(rows)
    store.run_sql(conn, "link_insert_chunk_domain", rows)


def metrics(st: State) -> dict:
    """Numbers for the run row (merged into metrics_run by the job)."""
    if st.table_ok is None:  # no chunk page was written this pass: nothing was probed, nothing is claimed
        return {}
    absent = 0 if st.table_ok else 1
    out = {"domain_table_absent": absent}
    if not absent:
        out.update(domain_rows_attempted=st.attempted, domain_chunks_without_domain=st.no_domain, domain_rows_retired=st.retired)
    return out
