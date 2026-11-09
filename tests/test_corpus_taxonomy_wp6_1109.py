"""WP6 (CHUNK-TAXO-WP6-1109): the stored-taxonomy Corpus view. Real SQL against an in-memory SQLite database with the 020 tables
(schema attached as vera_vjay), the pure renderer, the describer flag and the API key set. No network, no PostgreSQL, no model."""
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

from tests.test_corpus_desc_1107 import FALLBACK, _capture, _describer, _no_llm, _NORM, ROWS  # noqa: F401
from vera import corpus_description as cd, corpus_taxonomy as ct, sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
CN = ct.CLASSIFIER
assert CN == "chunk_taxo_auto"
LINK = "(link_id INTEGER PRIMARY KEY, {a} INTEGER, {b} INTEGER, classifier_name TEXT, state TEXT)"


def _engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(e, "connect")
    def _att(dbapi, _):
        dbapi.execute("ATTACH DATABASE ':memory:' AS vera_vjay")
    with e.begin() as c:
        for t, cols in {"sources": "(source_id INTEGER PRIMARY KEY, canonical_source_id INTEGER)",
                        "evidence_spans": "(span_id INTEGER PRIMARY KEY, source_id INTEGER)",
                        "tax_domains": "(domain_id INTEGER PRIMARY KEY, slug TEXT, label TEXT, state TEXT)",
                        "tax_topics": "(topic_id INTEGER PRIMARY KEY, slug TEXT, label TEXT, state TEXT)",
                        "tax_subtopics": "(subtopic_id INTEGER PRIMARY KEY, slug TEXT, label TEXT, state TEXT)",
                        "tax_domain_topic": "(link_id INTEGER PRIMARY KEY, domain_id INTEGER, topic_id INTEGER, state TEXT)",
                        "tax_topic_subtopic": "(link_id INTEGER PRIMARY KEY, topic_id INTEGER, subtopic_id INTEGER, state TEXT)",
                        "tax_chunk_subtopic": LINK.format(a="span_id", b="subtopic_id"), "tax_chunk_topic": LINK.format(a="span_id", b="topic_id"),
                        "tax_source_topic": LINK.format(a="source_id", b="topic_id"), "tax_source_domain": LINK.format(a="source_id", b="domain_id")}.items():
            c.execute(text(f"CREATE TABLE vera_vjay.{t} {cols}"))
    return e


def _ins(c, table, rows):
    for r in rows:
        c.execute(text(f"INSERT INTO vera_vjay.{table} VALUES ({','.join(str(x) if isinstance(x, int) else repr(x) for x in r)})"))


@pytest.fixture
def eng():
    """Taxonomy: D1 Health{T1 Flu{S1 Models}, T2 Vaccines}, D2 Ocean{T3 Reefs}, D3 Empty (no topic). 6 documents (doc 5 = a second
    version of doc 4, same canonical), 12 passages."""
    e = _engine()
    with e.begin() as c:
        _ins(c, "sources", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 6)])
        _ins(c, "evidence_spans", [(i, [1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 6, 6][i - 1]) for i in range(1, 13)])
        _ins(c, "tax_domains", [(1, "health", "Health", "active"), (2, "ocean", "Ocean", "active"), (3, "empty", "Empty", "active"), (4, "old", "Retired", "retired")])
        _ins(c, "tax_topics", [(1, "flu", "Flu", "active"), (2, "vaccines", "Vaccines", "active"), (3, "reefs", "Reefs", "active")])
        _ins(c, "tax_subtopics", [(1, "models", "Models", "active"), (2, "gone", "Gone", "retired")])
        _ins(c, "tax_domain_topic", [(1, 1, 1, "active"), (2, 1, 2, "active"), (3, 2, 3, "active")])
        _ins(c, "tax_topic_subtopic", [(1, 1, 1, "active"), (2, 1, 2, "active")])
        # chunks: 1,2 -> subtopic Models (-> Flu); 3 -> Flu direct AND Models (counts once under Flu); 4 -> Vaccines; 6 -> Reefs; 7 -> Flu (doc 4)
        # 8 labelled by ANOTHER classifier (ignored); 9 retired link (ignored); 5,10,11,12 unlabelled
        _ins(c, "tax_chunk_subtopic", [(1, 1, 1, CN, "active"), (2, 2, 1, CN, "active"), (3, 3, 1, CN, "active"), (4, 9, 1, CN, "retired"), (5, 8, 1, "taxo_chunk_assign", "active"),
                                       (6, 1, 2, CN, "active")])  # subtopic 2 is retired: ignored
        _ins(c, "tax_chunk_topic", [(1, 3, 1, CN, "active"), (2, 4, 2, CN, "active"), (3, 6, 3, CN, "active"), (4, 7, 1, CN, "active")])
        # document votes: doc1 -> Flu (topic), doc2 -> Health (domain only), doc3 -> Flu AND Reefs, doc4 -> Reefs; another classifier's row ignored
        _ins(c, "tax_source_topic", [(1, 1, 1, CN, "active"), (2, 3, 1, CN, "active"), (3, 3, 3, CN, "active"), (4, 4, 3, CN, "active"), (5, 6, 2, "taxo_chunk_assign", "active")])
        _ins(c, "tax_source_domain", [(1, 2, 1, CN, "active")])
    return e


def test_aggregation_matches_hand_counts(eng):
    p = ct.read_stored(eng)
    t = p["totals"]
    assert t == {"chunks": 12, "documents": 5, "documents_with_chunks": 5, "chunks_labelled": 6, "chunks_not_labelled": 6,
                 "documents_union_labelled": 4, "documents_vote_labelled": 4, "documents_union_not_labelled": 1,
                 "documents_vote_not_labelled": 1}
    dom = {d["slug"]: d for d in p["domains"]}
    assert set(dom) == {"health", "ocean", "empty"}  # retired domain hidden; an empty domain stays visible with zeros
    h, o, e = dom["health"], dom["ocean"], dom["empty"]
    flu = next(x for x in h["topics"] if x["slug"] == "flu")
    vac = next(x for x in h["topics"] if x["slug"] == "vaccines")
    # Flu chunks: 1,2,3 (via Models, 3 also direct: once) + 7 direct = 4 passages; docs: 1 (chunks 1-3) and 4 (chunk 7) = 2
    assert (flu["chunks"], flu["union_docs"], flu["vote_docs"]) == (4, 2, 2)
    assert flu["subtopics"] == [{"slug": "models", "label": "Models", "chunks": 3, "union_docs": 1}]  # retired subtopic hidden
    assert (vac["chunks"], vac["union_docs"], vac["vote_docs"]) == (1, 1, 0)
    # Health (domain) is the DISTINCT union of its topics: passages 1,2,3,7,4 = 5; documents 1,2,4 = 3. Never the sum of topics (4+1).
    assert (h["chunks"], h["union_docs"]) == (5, 3)
    # F7 vote rule: docs 2 (direct source_domain), 1 and 3 (source_topic Flu through domain_topic) = 3; doc 6 other classifier ignored
    assert h["vote_docs"] == 3
    assert (o["chunks"], o["union_docs"], o["vote_docs"]) == (1, 1, 2)  # Reefs: passage 6 (doc 3); votes docs 3 and 4
    assert (e["chunks"], e["union_docs"], e["vote_docs"], e["topics"]) == (0, 0, 0, [])


def test_union_and_vote_differ_and_a_second_version_counts_once(eng):
    p = ct.read_stored(eng)
    reefs = next(x for d in p["domains"] for x in d["topics"] if x["slug"] == "reefs")
    assert reefs["vote_docs"] == 2 and reefs["union_docs"] == 1  # doc 4 votes Reefs but none of its passages is labelled Reefs
    flu = next(x for d in p["domains"] for x in d["topics"] if x["slug"] == "flu")
    assert flu["union_docs"] == 2  # doc 5 (second version of doc 4) is the same canonical document
    with eng.begin() as c:
        _ins(c, "tax_chunk_topic", [(9, 8, 1, CN, "active")])  # a passage of version doc 5 (span 8) labelled Flu
    flu = next(x for d in ct.read_stored(eng)["domains"] for x in d["topics"] if x["slug"] == "flu")
    assert flu["union_docs"] == 2 and flu["chunks"] == 5


def test_empty_tables_give_none_and_no_error():
    e = _engine()
    assert ct.read_stored(e) is None
    with e.begin() as c:
        _ins(c, "sources", [(1, 1)])
        _ins(c, "evidence_spans", [(1, 1)])
        _ins(c, "tax_domains", [(1, "health", "Health", "active")])
    assert ct.read_stored(e) is None  # entities but no stored label: nothing to show, the interim page stands


def test_other_classifier_only_gives_none(eng):
    with eng.begin() as c:
        for t in ("tax_chunk_subtopic", "tax_chunk_topic", "tax_source_topic", "tax_source_domain"):
            c.execute(text(f"UPDATE vera_vjay.{t} SET classifier_name = 'taxo_backfill_vote'"))
    assert ct.read_stored(eng) is None


def test_abstentions_are_shown_as_one_combined_figure(eng):
    lines = sb.taxonomy_lines(ct.read_stored(eng))
    text_ = _NORM(" ".join(lines))
    assert "6 of 12 stored passages have no subject label" in text_
    assert "1 of 5 documents have no main label" in text_ and "1 of the 5 documents with stored passages have no labelled passage" in text_
    assert "Not labelled or outside the subject list" in text_ and "not told apart" in text_


def test_both_views_are_labelled_and_nothing_names_a_source(eng):
    lines = sb.taxonomy_lines(ct.read_stored(eng))
    text_ = _NORM(" ".join(lines))
    assert "Main label (vote)" in text_ and "Contains passages on (union)" in text_ and "automatic and approximate" in text_
    assert "Flu: main label for 2 documents; passages in 2 documents (4 passages)" in text_
    assert "Models: passages in 1 documents (3 passages)" in text_
    assert "do not add up" in text_
    blob = json.dumps(ct.read_stored(eng))
    for banned in ("http", "title", "url", "source_id", "span_id", "canonical"):
        assert banned not in blob.lower()


def test_malformed_payload_renders_nothing():
    for bad in (None, {}, {"domains": "x"}, {"domains": [], "totals": {}}, {"domains": [{"label": "x"}], "totals": {"documents": 1}}, "str", 5):
        assert sb.taxonomy_lines(bad) == []


def test_markdown_in_labels_is_escaped():
    p = {"totals": {"chunks": 2, "documents": 1, "documents_with_chunks": 1, "chunks_labelled": 1, "chunks_not_labelled": 1,
                    "documents_union_labelled": 1, "documents_vote_labelled": 1, "documents_union_not_labelled": 0, "documents_vote_not_labelled": 0},
         "domains": [{"label": "[x](http://evil)", "chunks": 1, "union_docs": 1, "vote_docs": 1, "topics": []}]}
    assert "[x](http://evil)" not in " ".join(sb.taxonomy_lines(p))


def test_flag_off_adds_nothing_to_the_interim_page():
    for data in (FALLBACK, {**FALLBACK, "stored_taxonomy": None}, {**FALLBACK, "stored_taxonomy": {}}):
        a, ca = _capture()
        sb.render_stored_taxonomy(a, dict(data))
        assert ca == []
    assert [e for e in (sb.render_stored_taxonomy(_capture()[0], None),)] == [None]


def _page(monkeypatch, data):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("offline")))
    monkeypatch.setattr(sb, "_corpus", lambda st: dict(data))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages/corpus_description.py"), default_timeout=30).run()
    assert not at.exception
    return _NORM(" ".join(e.value for kind in ("title", "markdown", "caption") for e in getattr(at.main, kind)))


def test_page_flag_off_is_identical_and_flag_on_appends(monkeypatch, eng):
    off = _page(monkeypatch, FALLBACK)
    assert _page(monkeypatch, {**FALLBACK, "stored_taxonomy": None}) == off
    on = _page(monkeypatch, {**FALLBACK, "stored_taxonomy": ct.read_stored(eng)})
    assert on.startswith(off.split(" Current coverage")[0][:200]) and "Stored subject labels" in on and "Stored subject labels" not in off
    assert "Main label (vote)" in on and "Flu: main label for 2 documents" in on


def test_shipped_flag_is_off_and_api_key_absent_when_off():
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert cfg["corpus_description"]["stored_taxonomy"]["enabled"] is False
    from main import CorpusDescription
    assert "stored_taxonomy" not in CorpusDescription(summary_source="x").model_dump()
    assert CorpusDescription(summary_source="x", stored_taxonomy={"a": 1}).model_dump()["stored_taxonomy"] == {"a": 1}
    assert "stored_taxonomy" in sb.CORPUS_FIELDS


def _on_describer(monkeypatch, engine, enabled):
    d = _describer(lambda: (True, ROWS))
    d.cfg["stored_taxonomy"] = {"enabled": enabled}
    d.taxo_status_fn = lambda: {"display_ok": True, "version": "v1", "reasons": []}
    d.engine_factory = lambda *a, **k: engine
    return d


def test_describer_flag_off_never_reads_the_tables(monkeypatch):
    calls = []
    monkeypatch.setattr(ct, "read_stored", lambda e: calls.append(1))
    out = _on_describer(monkeypatch, None, False).current(_no_llm)
    assert "stored_taxonomy" not in out and calls == []


def test_describer_flag_on_adds_payload_and_failure_degrades(monkeypatch, eng):
    d = _on_describer(monkeypatch, eng, True)
    d.cfg.pop("subject_areas", None)
    monkeypatch.setattr(cd, "stamp_and_profile", lambda *a, **k: ({"source_count": 5, "corpus_updated_at": "2026-11-09T00:00:00+00:00"},
                                                                  {"titles": [], "excerpts": [], "published_years": ["2005", "2026"], "year_counts": {}}))
    out = d._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"})
    assert out["totals"]["chunks"] == 12
    d2 = _on_describer(monkeypatch, eng, True)
    monkeypatch.setattr(ct, "read_stored", lambda e: (_ for _ in ()).throw(RuntimeError("db down")))
    assert d2._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"}) is None  # falls back to the interim page


def test_queries_are_read_only_and_filter_on_classifier():
    for q in (ct.Q_TOPIC_UNION, ct.Q_DOMAIN_UNION, ct.Q_SUBTOPIC_UNION, ct.Q_TOPIC_VOTE, ct.Q_DOMAIN_VOTE, ct.Q_HIER, ct.Q_TOTALS, ct.Q_LABELLED, ct.Q_VOTED):
        u = q.upper()
        assert u.startswith(("SELECT", "WITH")) and not any(w in u for w in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ", "TRUNCATE"))
        assert ("classifier_name = :cn" in q) or q in (ct.Q_HIER, ct.Q_TOTALS)
        assert "_eff_v" not in q  # F7: the effective views are not used


def test_display_gate_blocks_the_block_and_classifier_name_is_config(monkeypatch, eng):
    d = _on_describer(monkeypatch, eng, True)
    d.taxo_status_fn = lambda: {"display_ok": False, "version": "v1", "reasons": ["silhouette"]}
    assert d._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"}) is None  # gate failed: interim page stands
    d = _on_describer(monkeypatch, eng, True)
    assert d._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"})["version"] == "v1"
    d = _on_describer(monkeypatch, eng, True)
    d.cfg["stored_taxonomy"]["classifier_name"] = "taxo_chunk_assign"  # config value: the other writer's two rows are read instead
    other = d._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"})
    assert other["classifier"] == "taxo_chunk_assign" and other["totals"]["chunks_labelled"] == 1
    monkeypatch.setattr(ct, "job_status", lambda: {"display_ok": True, "version": "v2"})
    d = _on_describer(monkeypatch, eng, True)
    d.taxo_status_fn = None
    assert d._stored_taxonomy({"source_count": 5, "corpus_updated_at": "x"})["version"] == "v2"


def test_job_status_missing_modules_means_no_display():
    assert ct.job_status()["display_ok"] is False  # no DB reachable in tests: fail closed (interim page)
