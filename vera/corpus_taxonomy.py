"""WP6 (CHUNK-TAXO-WP6-1109): the Corpus page's stored-taxonomy view. Reads the per-chunk and per-document labels the
in-app chunk-taxonomy job (classifier chunk_taxo_auto; categories derived mechanically from semantic centers) stored in the migration-020 tables (vera_vjay.tax_*), and aggregates them into
domain > topic > subtopic counts. Read-only SELECTs; no model call; no source title, URL or id ever leaves this module.

Two views of a document's subject areas are kept apart and labelled (DBX-1020 owner rule: a document's subject areas are the
UNION of its chunks'; the kit also stores one calibrated VOTE per document, B9):
  * vote  : documents whose stored document-level label (tax_source_topic / tax_source_domain) is the entity. A domain counts
            when a direct tax_source_domain row exists OR a tax_source_topic row joins through tax_domain_topic (F7); the
            tax_source_*_eff_v views are NOT used (their basis column marks topic-labelled domains 'derived').
  * union : documents with at least one stored chunk label under the entity (direct chunk label, or chunk-subtopic rolled up
            through tax_topic_subtopic). Chunk counts are distinct passages. Domain figures are distinct, never sums of topics.
Every count is filtered on classifier_name = :cn and state = 'active' links and entities (earlier TAXO-BACKFILL rows may coexist).
Abstentions are not stored (F8, D-054): "not labelled" is derived from missing rows and cannot separate NO_FIT from never classified.
Documents are counted as distinct canonical sources, as the 020 views do. The SQL is portable (PG and SQLite) so it is tested
against real tables."""
from sqlalchemy import text

CLASSIFIER = "chunk_taxo_auto"  # default; the in-app job's writer name (OD-1109-TAXO-AUTO). Config: corpus_description.stored_taxonomy.classifier_name
V = "vera_vjay."

# Display counts use STRICT links only: score_kind NULL or 'cosine'. Routing-only links (cosine_r<k>, cosine_soft; TAXO-ROUTE-R1-1110) are excluded.
STRICT_CT = "(ct.score_kind IS NULL OR ct.score_kind = 'cosine')"
STRICT_CS = "(cs.score_kind IS NULL OR cs.score_kind = 'cosine')"
STRICT_1 = "(score_kind IS NULL OR score_kind = 'cosine')"
_CL = (  # chunk -> topic, direct UNION subtopic rolled up; distinct (span, topic)
    f"cl AS (SELECT ct.span_id, ct.topic_id FROM {V}tax_chunk_topic ct JOIN {V}tax_topics t ON t.topic_id = ct.topic_id AND t.state = 'active' "
    "WHERE ct.classifier_name = :cn AND ct.state = 'active' AND " + STRICT_CT + " "
    f"UNION SELECT cs.span_id, ts.topic_id FROM {V}tax_chunk_subtopic cs "
    f"JOIN {V}tax_subtopics sb ON sb.subtopic_id = cs.subtopic_id AND sb.state = 'active' "
    f"JOIN {V}tax_topic_subtopic ts ON ts.subtopic_id = cs.subtopic_id AND ts.state = 'active' "
    f"JOIN {V}tax_topics t2 ON t2.topic_id = ts.topic_id AND t2.state = 'active' WHERE cs.classifier_name = :cn AND cs.state = 'active' AND " + STRICT_CS + ")")
_DOC = "COALESCE(s.canonical_source_id, s.source_id)"
_SP = f"JOIN {V}evidence_spans e ON e.span_id = {{x}}.span_id JOIN {V}sources s ON s.source_id = e.source_id"

Q_TOPIC_UNION = (f"WITH {_CL} SELECT cl.topic_id, count(DISTINCT cl.span_id), count(DISTINCT {_DOC}) FROM cl " + _SP.format(x="cl") + " GROUP BY cl.topic_id")
Q_DOMAIN_UNION = (f"WITH {_CL} SELECT dt.domain_id, count(DISTINCT cl.span_id), count(DISTINCT {_DOC}) FROM cl "
                  f"JOIN {V}tax_domain_topic dt ON dt.topic_id = cl.topic_id AND dt.state = 'active' " + _SP.format(x="cl") + " GROUP BY dt.domain_id")
Q_SUBTOPIC_UNION = (f"SELECT cs.subtopic_id, count(DISTINCT cs.span_id), count(DISTINCT {_DOC}) FROM {V}tax_chunk_subtopic cs "
                    f"JOIN {V}tax_subtopics sb ON sb.subtopic_id = cs.subtopic_id AND sb.state = 'active' " + _SP.format(x="cs")
                    + " WHERE cs.classifier_name = :cn AND cs.state = 'active' AND " + STRICT_CS + " GROUP BY cs.subtopic_id")
Q_TOPIC_VOTE = (f"SELECT st.topic_id, count(DISTINCT {_DOC}) FROM {V}tax_source_topic st JOIN {V}tax_topics t ON t.topic_id = st.topic_id AND t.state = 'active' "
                f"JOIN {V}sources s ON s.source_id = st.source_id WHERE st.classifier_name = :cn AND st.state = 'active' GROUP BY st.topic_id")
Q_DOMAIN_VOTE = (  # F7: direct source_domain OR source_topic joined through domain_topic
    f"SELECT u.domain_id, count(DISTINCT {_DOC}) FROM (SELECT sd.source_id, sd.domain_id FROM {V}tax_source_domain sd "
    f"JOIN {V}tax_domains d ON d.domain_id = sd.domain_id AND d.state = 'active' WHERE sd.classifier_name = :cn AND sd.state = 'active' "
    f"UNION SELECT st.source_id, dt.domain_id FROM {V}tax_source_topic st JOIN {V}tax_topics t ON t.topic_id = st.topic_id AND t.state = 'active' "
    f"JOIN {V}tax_domain_topic dt ON dt.topic_id = st.topic_id AND dt.state = 'active' "
    f"JOIN {V}tax_domains d2 ON d2.domain_id = dt.domain_id AND d2.state = 'active' WHERE st.classifier_name = :cn AND st.state = 'active') u "
    f"JOIN {V}sources s ON s.source_id = u.source_id GROUP BY u.domain_id")
Q_HIER = (f"SELECT d.domain_id, d.slug, d.label, t.topic_id, t.slug, t.label, sb.subtopic_id, sb.slug, sb.label FROM {V}tax_domains d "
          f"LEFT JOIN {V}tax_domain_topic dt ON dt.domain_id = d.domain_id AND dt.state = 'active' "
          f"LEFT JOIN {V}tax_topics t ON t.topic_id = dt.topic_id AND t.state = 'active' "
          f"LEFT JOIN {V}tax_topic_subtopic ts ON ts.topic_id = t.topic_id AND ts.state = 'active' "
          f"LEFT JOIN {V}tax_subtopics sb ON sb.subtopic_id = ts.subtopic_id AND sb.state = 'active' WHERE d.state = 'active' "
          "ORDER BY d.slug, t.slug, sb.slug")
Q_TOTALS = (f"SELECT (SELECT count(*) FROM {V}evidence_spans), (SELECT count(DISTINCT {_DOC}) FROM {V}sources s), "
            f"(SELECT count(DISTINCT {_DOC}) FROM {V}sources s WHERE EXISTS (SELECT 1 FROM {V}evidence_spans e WHERE e.source_id = s.source_id))")
Q_LABELLED = (f"WITH ls AS (SELECT span_id FROM {V}tax_chunk_topic WHERE classifier_name = :cn AND state = 'active' AND " + STRICT_1 + " "
              f"UNION SELECT span_id FROM {V}tax_chunk_subtopic WHERE classifier_name = :cn AND state = 'active' AND " + STRICT_1 + ") "
              f"SELECT (SELECT count(*) FROM ls), (SELECT count(DISTINCT {_DOC}) FROM ls " + _SP.format(x="ls") + ")")
Q_VOTED = (f"SELECT count(DISTINCT {_DOC}) FROM (SELECT source_id FROM {V}tax_source_topic WHERE classifier_name = :cn AND state = 'active' "
           f"UNION SELECT source_id FROM {V}tax_source_domain WHERE classifier_name = :cn AND state = 'active') u JOIN {V}sources s ON s.source_id = u.source_id")

# CORPUS-UNCAT-1113 / CORPUS-WORDING-N1-1113: documents with NO stored topic or sub-topic record. A document has one when any of its version rows has an
# active, configured-classifier document-level topic vote (tax_source_topic to an active topic) or a chunk in the display CTE _CL (strict links,
# active entities). A document-level domain vote alone (tax_source_domain) is not a topic or sub-topic record and does not count. Columns: distinct documents, distinct labelled documents, distinct labelled passages.
Q_UNLABELLED = (
    f"WITH {_CL}, lab AS (SELECT e.source_id FROM cl JOIN {V}evidence_spans e ON e.span_id = cl.span_id "
    f"UNION SELECT st.source_id FROM {V}tax_source_topic st JOIN {V}tax_topics t3 ON t3.topic_id = st.topic_id AND t3.state = 'active' "
    "WHERE st.classifier_name = :cn AND st.state = 'active') "
    f"SELECT count(DISTINCT {_DOC}), count(DISTINCT CASE WHEN lab.source_id IS NOT NULL THEN {_DOC} END), (SELECT count(DISTINCT span_id) FROM cl) "
    f"FROM {V}sources s LEFT JOIN lab ON lab.source_id = s.source_id")


def read_unlabelled(engine, classifier: str = CLASSIFIER) -> dict | None:
    """Counts of documents and stored chunks with no stored topic or sub-topic record (aggregate only; no title, id or area name), or None for an
    empty corpus. documents_without_passages: the labelling job reads only evidence_spans, so these can never be labelled.
    Raises on a database error (the caller treats that as 'not available')."""
    with engine.connect() as c:
        d, dl, pl = (int(x or 0) for x in c.execute(text(Q_UNLABELLED), {"cn": classifier}).one())
        n_pass, n_docs, n_docs_pass = (int(x or 0) for x in c.execute(text(Q_TOTALS)).one())
    if d == 0:
        return None
    return {"classifier": classifier, "documents": d, "documents_unlabelled": max(0, d - dl),
            "documents_without_passages": max(0, n_docs - n_docs_pass), "passages": n_pass, "passages_unlabelled": max(0, n_pass - pl)}


def job_status() -> dict:
    """The ONE status entry point (INT-1109): vera.chunk_taxo_status.status(), the cached reader of the latest 'finished'
    tax_job_runs row (state 'finished' only; running, deferred and failed rows are ignored). It works in every API process
    (the job's in-memory status exists only where the job thread runs, and is 'off' when the flag is off), never raises, and
    returns {state, version, metrics, display_ok, reasons, last_run}. display_ok is true only when that run's mechanical display
    gate (config/chunk_taxo.json display_gate) passed. Module absent, any error, no finished run or migration 024 absent ->
    display_ok False, so the Corpus page shows the interim view."""
    try:
        from vera import chunk_taxo_status
        st = dict(chunk_taxo_status.status())
        st["display_ok"] = st.get("display_ok") is True and st.get("state") == "ok"
        return st
    except Exception:  # noqa: BLE001  module absent or failing
        return {"display_ok": False, "version": None, "reasons": ["job status unavailable"]}


def read_stored(engine, classifier: str = CLASSIFIER) -> dict | None:
    """The stored-taxonomy payload, or None when no label of this classifier is stored (the caller then shows the interim page).
    Raises on a database error (the caller treats that as 'not available')."""
    p = {"cn": classifier}
    with engine.connect() as c:
        one = lambda q: {int(r[0]): tuple(int(x) for x in r[1:]) for r in c.execute(text(q), p).all()}  # noqa: E731
        raw = dict(tu=one(Q_TOPIC_UNION), du=one(Q_DOMAIN_UNION), su=one(Q_SUBTOPIC_UNION), tv=one(Q_TOPIC_VOTE), dv=one(Q_DOMAIN_VOTE))
        hier = c.execute(text(Q_HIER)).all()
        totals = c.execute(text(Q_TOTALS)).one()
        lab = c.execute(text(Q_LABELLED), p).one()
        voted = c.execute(text(Q_VOTED), p).scalar()
    return build(raw, hier, tuple(int(x or 0) for x in totals), tuple(int(x or 0) for x in lab), int(voted or 0), classifier)


def build(raw: dict, hier, totals, labelled, voted_docs: int, classifier: str = CLASSIFIER) -> dict | None:
    """Pure aggregation of the query results. None when nothing is stored."""
    n_chunks, n_docs, n_docs_chunks = totals
    lab_chunks, lab_docs = labelled
    if lab_chunks == 0 and voted_docs == 0:
        return None
    doms: dict = {}
    for did, dslug, dlabel, tid, tslug, tlabel, sid, sslug, slabel in hier:
        d = doms.setdefault(did, {"slug": dslug, "label": dlabel, "topics": {}})
        if tid is None:
            continue
        t = d["topics"].setdefault(tid, {"slug": tslug, "label": tlabel, "subtopics": {}})
        if sid is not None:
            t["subtopics"].setdefault(sid, {"slug": sslug, "label": slabel})
    z2, z1 = (0, 0), (0,)
    out = []
    for did, d in doms.items():
        du, dv = raw["du"].get(did, z2), raw["dv"].get(did, z1)
        tops = []
        for tid, t in d["topics"].items():
            tu, tv = raw["tu"].get(tid, z2), raw["tv"].get(tid, z1)
            subs = [{"slug": s["slug"], "label": s["label"], "chunks": raw["su"].get(sid, z2)[0], "union_docs": raw["su"].get(sid, z2)[1]}
                    for sid, s in t["subtopics"].items()]
            tops.append({"slug": t["slug"], "label": t["label"], "chunks": tu[0], "union_docs": tu[1], "vote_docs": tv[0], "subtopics": subs})
        out.append({"slug": d["slug"], "label": d["label"], "chunks": du[0], "union_docs": du[1], "vote_docs": dv[0], "topics": tops})
    return {"classifier": classifier,
            "totals": {"chunks": n_chunks, "documents": n_docs, "documents_with_chunks": n_docs_chunks,
                       "chunks_labelled": lab_chunks, "chunks_not_labelled": max(0, n_chunks - lab_chunks),
                       "documents_union_labelled": lab_docs, "documents_vote_labelled": voted_docs,
                       "documents_union_not_labelled": max(0, n_docs_chunks - lab_docs),
                       "documents_vote_not_labelled": max(0, n_docs - voted_docs)},
            "domains": out}
