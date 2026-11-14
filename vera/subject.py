"""SUBJECT-BUILD-1114 WP-S1 (plan work/SUBJECT-1114): what a cited passage is about, read from the stored chunk-taxonomy associations.

Read-only: no DDL, no write, no model call. One fixed SELECT (300 ms statement timeout) over the migration 020/022/027 tables
(vera_vjay.tax_*), one pure composer. Never raises: any failure gives reason 'unavailable'.

Output (always this shape; the labels are the stored node labels, unsanitised: the caller words and checks them):
  {"domain": str|None, "topic": str|None, "subtopic": str|None,
   "basis": "subtopic"|"topic"|"domain"|"document_vote"|"none",
   "reason": None|"no_record"|"display_gate_closed"|"unavailable"|"unmapped",
   "ids": {"domain": int|None, "topic": int|None, "subtopic": int|None},   # node ids per level, so callers can look up names
   "counts": {"multi_parent": int, "domain_mismatch": int}}

Id resolution (full-document chunks): the live corpus path (vera/fulldoc_corpus.py) gives a passage the string id "<document_id>:<start>-<end>"
(not a database id). The integer evidence_spans ids live in the stamps of the stored chunks the passage overlaps (`corpus_chunks[].span_id`,
vera/fulldoc_db.py corpus_stamps) and, for a database-served span, in `span_id`; a merged passage lists its members in `merged_span_ids`
(vera/fulldoc_admission.py). Only integers are used; strings of that shape are ignored. The document key is `canonical_source_id` only
(a bare `document_id` is a different id space and is not matched; never the absent `source_id`)."""
from __future__ import annotations

import contextlib

from sqlalchemy import bindparam, text

from vera.corpus_taxonomy import CLASSIFIER, STRICT_CS, STRICT_CT

V = "vera_vjay."
TIMEOUT_MS = 300
SET_TIMEOUT = f"SET LOCAL statement_timeout = {TIMEOUT_MS}"
PROBE_SQL = "SELECT to_regclass('vera_vjay.tax_chunk_domain') IS NOT NULL"

_N = "NULL::bigint"
_F = "NULL::double precision"
_LINKS = (
    f"SELECT 'cs' AS k, cs.span_id AS span, cs.subtopic_id AS node, {_N} AS parent, cs.score AS score, sb.label AS label FROM {V}tax_chunk_subtopic cs "
    f"JOIN {V}tax_subtopics sb ON sb.subtopic_id = cs.subtopic_id AND sb.state = 'active' "
    "WHERE cs.classifier_name = :cn AND cs.state = 'active' AND " + STRICT_CS + " AND cs.span_id IN :spans "
    f"UNION ALL SELECT 'ct', ct.span_id, ct.topic_id, {_N}, ct.score, t.label FROM {V}tax_chunk_topic ct "
    f"JOIN {V}tax_topics t ON t.topic_id = ct.topic_id AND t.state = 'active' "
    "WHERE ct.classifier_name = :cn AND ct.state = 'active' AND " + STRICT_CT + " AND ct.span_id IN :spans")
_DOMAIN_ROWS = (  # tax_chunk_domain rows are strict by construction (vera/chunk_domain_writer.py): any score kind, never strict-filtered
    f" UNION ALL SELECT 'cd', cd.span_id, cd.domain_id, {_N}, cd.score, d.label FROM {V}tax_chunk_domain cd "
    f"JOIN {V}tax_domains d ON d.domain_id = cd.domain_id AND d.state = 'active' "
    "WHERE cd.classifier_name = :cn AND cd.state = 'active' AND cd.span_id IN :spans")
_REST = (
    f" UNION ALL SELECT 'ts', {_N}, ts.subtopic_id, ts.topic_id, {_F}, NULL::text FROM {V}tax_topic_subtopic ts "  # hierarchy, filtered as Q_HIER
    f"JOIN {V}tax_topics t ON t.topic_id = ts.topic_id AND t.state = 'active' "
    f"JOIN {V}tax_subtopics sb ON sb.subtopic_id = ts.subtopic_id AND sb.state = 'active' WHERE ts.state = 'active'"
    f" UNION ALL SELECT 'dt', {_N}, dt.topic_id, dt.domain_id, {_F}, NULL::text FROM {V}tax_domain_topic dt "
    f"JOIN {V}tax_domains d ON d.domain_id = dt.domain_id AND d.state = 'active' "
    f"JOIN {V}tax_topics t ON t.topic_id = dt.topic_id AND t.state = 'active' WHERE dt.state = 'active'"
    f" UNION ALL SELECT 'tn', {_N}, t.topic_id, {_N}, {_F}, t.label FROM {V}tax_topics t WHERE t.state = 'active'"
    f" UNION ALL SELECT 'dn', {_N}, d.domain_id, {_N}, {_F}, d.label FROM {V}tax_domains d WHERE d.state = 'active'"
    # document votes (fallback only); the parent column carries the document's canonical key so one SELECT serves every citation
    f" UNION ALL SELECT 'vt', {_N}, st.topic_id, s.canonical_source_id, st.score, t.label FROM {V}tax_source_topic st "
    f"JOIN {V}tax_topics t ON t.topic_id = st.topic_id AND t.state = 'active' JOIN {V}sources s ON s.source_id = st.source_id "
    "WHERE st.classifier_name = :cn AND st.state = 'active' AND s.canonical_source_id IN :docs"
    f" UNION ALL SELECT 'vd', {_N}, sd.domain_id, s.canonical_source_id, sd.score, d.label FROM {V}tax_source_domain sd "
    f"JOIN {V}tax_domains d ON d.domain_id = sd.domain_id AND d.state = 'active' JOIN {V}sources s ON s.source_id = sd.source_id "
    "WHERE sd.classifier_name = :cn AND sd.state = 'active' AND s.canonical_source_id IN :docs")
SQL_WITH_DOMAIN = _LINKS + _DOMAIN_ROWS + _REST
SQL_NO_DOMAIN = _LINKS + _REST

_PROBE_CACHE: dict = {"version": object(), "has": False}  # per process; re-run when the job version changes
_ENGINE: dict = {}
_NEG = float("-inf")


def _int_id(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isascii() and x.isdigit():
        return int(x)
    return None


def resolve_ids(citation) -> tuple[tuple, int | None]:
    """(sorted integer span ids, integer document key or None) of one cited passage; see the module docstring."""
    c = citation if isinstance(citation, dict) else {}
    ids = set()
    for v in (c.get("span_id"), *(c.get("merged_span_ids") or ())):
        n = _int_id(v)
        if n is not None:
            ids.add(n)
    s0, e0 = c.get("start"), c.get("end")
    if isinstance(s0, int) and isinstance(e0, int):
        for ch in c.get("corpus_chunks") or ():  # only stored chunks that really overlap the passage (no all-chunks fallback)
            if isinstance(ch, dict) and isinstance(ch.get("start"), int) and isinstance(ch.get("end"), int) \
                    and ch["start"] < e0 and s0 < ch["end"]:
                n = _int_id(ch.get("span_id"))
                if n is not None:
                    ids.add(n)
    key = c.get("canonical_source_id")
    if key is None:
        key = c.get("document_id")
    return tuple(sorted(ids)), _int_id(key)


def _out(domain=None, topic=None, subtopic=None, basis="none", reason=None, ids=None, counts=None) -> dict:
    return {"domain": domain, "topic": topic, "subtopic": subtopic, "basis": basis, "reason": reason,
            "ids": {"domain": None, "topic": None, "subtopic": None, **(ids or {})},
            "counts": {"multi_parent": 0, "domain_mismatch": 0, **(counts or {})}}


def _best(cands):  # cands: (depth, score, node, span); total order, so row order never matters
    return max(cands, key=lambda c: (c[0], _NEG if c[1] is None else float(c[1]), -c[2]))


def compose(rows) -> dict:
    """Pure. rows: (kind, span_id, node_id, parent_id, score, label) as the fixed SELECT returns them. Kinds: cs/ct/cd chunk links
    (sub-topic, topic, domain), ts/dt hierarchy edges (node, parent), tn/dn node labels, vt/vd document votes."""
    lab = {"s": {}, "t": {}, "d": {}}
    parents = {"s": {}, "t": {}}  # sub-topic -> topic ids, topic -> domain ids
    links, cdom, votes_t, votes_d = [], {}, [], []
    for k, span, node, parent, score, label in rows:
        if k == "cs":
            links.append((3, score, node, span))
            lab["s"][node] = label
        elif k == "ct":
            links.append((2, score, node, span))
            lab["t"][node] = label
        elif k == "cd":
            links.append((1, score, node, span))
            cdom.setdefault(span, set()).add(node)
            lab["d"][node] = label
        elif k == "ts":
            parents["s"].setdefault(node, set()).add(parent)
        elif k == "dt":
            parents["t"].setdefault(node, set()).add(parent)
        elif k == "tn":
            lab["t"].setdefault(node, label)
        elif k == "dn":
            lab["d"].setdefault(node, label)
        elif k == "vt":
            votes_t.append((2, score, node, None))
            lab["t"].setdefault(node, label)
        elif k == "vd":
            votes_d.append((1, score, node, None))
            lab["d"].setdefault(node, label)
    counts = {"multi_parent": 0, "domain_mismatch": 0}

    def up(level, node):  # lowest active parent id; several parents counted
        ps = sorted(p for p in parents[level].get(node, ()) if p is not None)
        if len(ps) > 1:
            counts["multi_parent"] += 1
        return ps[0] if ps else None

    def named(level, node):
        return None if node is None else (lab[level].get(node) or None)

    if links:
        depth, _, node, span = _best(links)
        sub = top = dom = None
        if depth == 3:
            sub = node
            top = up("s", sub)
        elif depth == 2:
            top = node
        if depth == 1:
            dom = node
        elif top is not None:
            dom = up("t", top)
        if depth > 1:
            cd = cdom.get(span, set())
            if dom is None and cd:
                dom = min(cd)  # no hierarchy domain: the chunk's own domain row
            elif dom is not None and cd and dom not in cd:
                dom = None
                counts["domain_mismatch"] += 1
        basis = "subtopic" if sub is not None else "topic" if top is not None else "domain"
        return _out(named("d", dom), named("t", top), named("s", sub), basis, None,
                    {"domain": dom, "topic": top, "subtopic": sub}, counts)
    if votes_t:  # the document's vote (the document's, never the chunk's)
        top = _best(votes_t)[2]
        dom = up("t", top)
        if dom is None and votes_d:
            dom = _best(votes_d)[2]
        return _out(named("d", dom), named("t", top), None, "document_vote", None, {"domain": dom, "topic": top}, counts)
    if votes_d:
        dom = _best(votes_d)[2]
        return _out(named("d", dom), None, None, "document_vote", None, {"domain": dom}, counts)
    return _out(reason="no_record")


def _default_engine():
    if "e" not in _ENGINE:
        from sqlalchemy import create_engine

        from vera.db import get_role_engine

        def make(url, **kw):
            return create_engine(url, pool_size=1, max_overflow=0, pool_timeout=0.5, connect_args={"connect_timeout": 1}, **kw)
        _ENGINE["e"] = get_role_engine("rw", engine_factory=make)
    return _ENGINE["e"]


@contextlib.contextmanager
def warm_engine():
    """A private, short-lived engine for the names warm-up (rw role, one connection), disposed on exit. It never shares the
    answer path's one-connection pool, so a warm-up read cannot make an answer lose its subject line."""
    from sqlalchemy import create_engine

    from vera.db import get_role_engine

    def make(url, **kw):
        return create_engine(url, pool_size=1, max_overflow=0, pool_timeout=5, connect_args={"connect_timeout": 5}, **kw)
    eng = get_role_engine("rw", engine_factory=make)
    try:
        yield eng
    finally:
        try:
            eng.dispose()
        except Exception:  # noqa: BLE001
            pass


def _gate(cfg, status):
    """(open, classifier). Open only when stored_taxonomy.enabled is true AND the latest finished run has display_ok."""
    if cfg is None:
        from vera import corpus_description
        cfg = corpus_description.load_cfg()
    sc = (cfg or {}).get("stored_taxonomy") or {}
    if sc.get("enabled") is not True:
        return False, None, {}
    if status is None:
        from vera import corpus_taxonomy
        status = corpus_taxonomy.job_status()
    st = status if isinstance(status, dict) else {}
    return st.get("display_ok") is True, str(sc.get("classifier_name") or CLASSIFIER), st


def _canonical_key(citation):
    """Integer canonical_source_id of the passage, or None. A bare document_id is a sources.source_id, a different key space,
    so it is never matched against canonical ids (the document-vote fallback needs the canonical id)."""
    c = citation if isinstance(citation, dict) else {}
    return _int_id(c.get("canonical_source_id"))


def _rows_for(rows, spans, doc) -> list:
    """The rows that belong to one citation: chunk links on its spans, votes of its document (a vote row whose key column is
    empty is kept, for fixtures without it), and every hierarchy/label row."""
    sp = set(spans)
    out = []
    for r in rows:
        k = r[0]
        if k in ("cs", "ct", "cd"):
            if r[1] in sp:
                out.append(r)
        elif k in ("vt", "vd"):
            if r[3] is None or (doc is not None and r[3] == doc):
                out.append(r)
        else:
            out.append(r)
    return out


def subjects_for(citations, *, engine=None, cfg=None, status=None) -> list:
    """One subject dict per citation, from ONE SELECT for the whole answer (gate checked once). Never raises: on any error
    every element has reason 'unavailable'. engine, cfg and status are test seams."""
    cits = list(citations or [])
    try:
        ok, cn, st = _gate(cfg, status)
        if not ok:
            return [_out(reason="display_gate_closed") for _ in cits]
        keys = []
        for c in cits:
            spans, _ = resolve_ids(c)
            keys.append((spans, _canonical_key(c)))
        live = [k for k in keys if k[0] or k[1] is not None]
        if not live:
            return [_out(reason="unmapped") for _ in cits]
        all_spans = sorted({s for sp, _ in live for s in sp})
        all_docs = sorted({d for _, d in live if d is not None})
        eng = engine if engine is not None else _default_engine()
        ver = st.get("version")
        with eng.begin() as c:
            c.execute(text(SET_TIMEOUT))
            if _PROBE_CACHE["version"] != ver:
                _PROBE_CACHE.update(has=bool(c.execute(text(PROBE_SQL)).scalar()), version=ver)
            sql = SQL_WITH_DOMAIN if _PROBE_CACHE["has"] else SQL_NO_DOMAIN
            rows = [tuple(r) for r in c.execute(
                text(sql).bindparams(bindparam("spans", expanding=True), bindparam("docs", expanding=True)),
                {"cn": cn, "spans": all_spans or [-1], "docs": all_docs or [-1]}).all()]
        return [_out(reason="unmapped") if not (sp or d is not None) else compose(_rows_for(rows, sp, d)) for sp, d in keys]
    except Exception:  # noqa: BLE001  never raises: the caller shows nothing
        return [_out(reason="unavailable") for _ in cits]


def subject_for(citation, *, engine=None, cfg=None, status=None) -> dict:
    """The subject of one cited passage (see the module docstring). engine, cfg and status are test seams. Never raises."""
    return subjects_for([citation], engine=engine, cfg=cfg, status=status)[0]
