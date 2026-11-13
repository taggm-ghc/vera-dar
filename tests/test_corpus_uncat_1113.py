"""CORPUS-UNCAT-1113: the Corpus Description page states how much of the corpus has no stored subject label.
Real SQL against the in-memory SQLite 020 tables of tests/test_corpus_taxonomy_wp6_1109.py (real column names), the describer,
the API model, the code-built line and the page renderer. No network, no PostgreSQL, no model. Fixtures are synthetic integer ids."""
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from tests.sql_contract_support import collect_required, extract_columns, pending_objects
from tests.test_corpus_desc_1107 import FALLBACK, _describer, _no_llm, _NORM  # noqa: F401
from tests.test_corpus_taxonomy_wp6_1109 import CN, _engine, _ins, eng  # noqa: F401
from vera import corpus_description as cd, corpus_taxonomy as ct, help_pages, sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
TEXT_PREFIX = "**Chunks with a topic or sub-topic record in the active topic hierarchy:** "


# --- (a) per-document labelled / unlabelled rule on the real tables ---------------------------------------------------
def _base():
    """One document (1) with one passage (1); taxonomy: domain 1 active, 2 retired; topic 1 active, 2 retired; subtopic 1 active and
    rolled up to topic 1, subtopic 2 active but with NO active tax_topic_subtopic row, subtopic 3 retired (linked to topic 1)."""
    e = _engine()
    with e.begin() as c:
        _ins(c, "sources", [(1, 1)])
        _ins(c, "evidence_spans", [(1, 1)])
        _ins(c, "tax_domains", [(1, "d1", "D1", "active"), (2, "d2", "D2", "retired")])
        _ins(c, "tax_topics", [(1, "t1", "T1", "active"), (2, "t2", "T2", "retired")])
        _ins(c, "tax_subtopics", [(1, "s1", "S1", "active"), (2, "s2", "S2", "active"), (3, "s3", "S3", "retired")])
        _ins(c, "tax_topic_subtopic", [(1, 1, 1, "active"), (2, 1, 3, "active")])
    return e


LABELLED = {
    "vote_topic": ("tax_source_topic", [(1, 1, 1, CN, "active")]),
    "chunk_topic": ("tax_chunk_topic", [(1, 1, 1, CN, "active")]),
    "chunk_subtopic": ("tax_chunk_subtopic", [(1, 1, 1, CN, "active")]),
    "explicit_cosine": ("tax_chunk_topic", [(1, 1, 1, CN, "active", "cosine")]),
}
UNLABELLED = {
    "nothing": None,
    "cosine_soft": ("tax_chunk_topic", [(1, 1, 1, CN, "active", "cosine_soft")]),
    "cosine_r3": ("tax_chunk_topic", [(1, 1, 1, CN, "active", "cosine_r3")]),
    "other_classifier": ("tax_chunk_topic", [(1, 1, 1, "taxo_chunk_assign", "active")]),
    "other_classifier_vote": ("tax_source_topic", [(1, 1, 1, "taxo_chunk_assign", "active")]),
    "inactive_chunk": ("tax_chunk_topic", [(1, 1, 1, CN, "retired")]),
    "inactive_vote": ("tax_source_topic", [(1, 1, 1, CN, "retired")]),
    "retired_subtopic": ("tax_chunk_subtopic", [(1, 1, 3, CN, "active")]),
    "retired_topic_vote": ("tax_source_topic", [(1, 1, 2, CN, "active")]),
    "retired_topic_chunk": ("tax_chunk_topic", [(1, 1, 2, CN, "active")]),
    # CORPUS-WORDING-N1-1113: a document-level domain vote alone is not a topic or sub-topic record, so the document stays uncounted
    "domain_vote_only": ("tax_source_domain", [(1, 1, 1, CN, "active")]),
    "retired_domain_vote": ("tax_source_domain", [(1, 1, 2, CN, "active")]),
    "orphan_subtopic": ("tax_chunk_subtopic", [(1, 1, 2, CN, "active")]),
}


@pytest.mark.parametrize("name", sorted(LABELLED))
def test_each_stored_label_kind_makes_a_document_labelled(name):
    e = _base()
    table, rows = LABELLED[name]
    with e.begin() as c:
        _ins(c, table, rows)
    r = ct.read_unlabelled(e, CN)
    assert r["documents"] == 1 and r["documents_unlabelled"] == 0, name
    assert (r["passages"], r["documents_without_passages"]) == (1, 0)


@pytest.mark.parametrize("name", sorted(UNLABELLED))
def test_lookalike_links_do_not_make_a_document_labelled(name):
    e = _base()
    if UNLABELLED[name]:
        table, rows = UNLABELLED[name]
        with e.begin() as c:
            _ins(c, table, rows)
    r = ct.read_unlabelled(e, CN)
    assert r["documents"] == 1 and r["documents_unlabelled"] == 1, name
    assert r["passages"] == 1 and r["passages_unlabelled"] == 1, name


def test_a_document_without_stored_passages_is_counted_and_named():
    e = _base()
    with e.begin() as c:
        c.execute(text("DELETE FROM vera_vjay.evidence_spans"))
    r = ct.read_unlabelled(e, CN)
    assert r == {"classifier": CN, "documents": 1, "documents_unlabelled": 1, "documents_without_passages": 1,
                 "passages": 0, "passages_unlabelled": 0}


# --- (b) second versions count once ----------------------------------------------------------------------------------
def _versions(label_span=None, label_vote_source=None):
    e = _engine()
    with e.begin() as c:
        _ins(c, "sources", [(1, 1), (2, 1)])
        _ins(c, "evidence_spans", [(1, 1), (2, 2)])
        _ins(c, "tax_topics", [(1, "t1", "T1", "active")])
        if label_span:
            _ins(c, "tax_chunk_topic", [(1, label_span, 1, CN, "active")])
        if label_vote_source:
            _ins(c, "tax_source_topic", [(1, label_vote_source, 1, CN, "active")])
    return ct.read_unlabelled(e, CN)


def test_two_version_rows_count_once_and_are_labelled_if_any_row_is():
    for kw in ({"label_span": 2}, {"label_span": 1}, {"label_vote_source": 2}):
        r = _versions(**kw)
        assert (r["documents"], r["documents_unlabelled"]) == (1, 0), kw
    r = _versions()
    assert (r["documents"], r["documents_unlabelled"], r["passages"], r["passages_unlabelled"]) == (1, 1, 2, 2)


# --- (c) empty corpus, zero passages ---------------------------------------------------------------------------------
def test_empty_corpus_gives_none_and_no_line():
    assert ct.read_unlabelled(_engine(), CN) is None
    assert sb.unlabelled_line(None) is None


def test_sources_but_no_passages_omit_the_passage_clause():
    e = _engine()
    with e.begin() as c:
        _ins(c, "sources", [(1, 1), (2, 2)])
    r = ct.read_unlabelled(e, CN)
    assert (r["documents"], r["documents_unlabelled"], r["documents_without_passages"], r["passages"]) == (2, 2, 2, 0)
    line = sb.unlabelled_line(r)
    assert "Chunks with" not in line and "0 of 2 (0%)" in line and line.startswith("**Documents with a topic or sub-topic record in the active topic hierarchy:**")


# --- (d) describer ---------------------------------------------------------------------------------------------------
PAYLOAD = {"classifier": CN, "documents": 10, "documents_unlabelled": 3, "documents_without_passages": 0, "passages": 100, "passages_unlabelled": 40}


def _desc(flag):
    d = _describer(lambda: (True, __import__("tests.test_corpus_desc_1107", fromlist=["ROWS"]).ROWS))
    if flag is not None:
        d.cfg["unlabelled_share"] = {"enabled": flag}
    return d


def test_key_present_only_when_the_flag_is_exactly_true(monkeypatch):
    calls = []
    monkeypatch.setattr(ct, "read_unlabelled", lambda e, c: calls.append(c) or dict(PAYLOAD))
    out = _desc(True).current(_no_llm)
    assert out["source_count"] and out["unlabelled"] == PAYLOAD and calls == [CN]
    for flag in (False, "true", 1, None, {"x": 1}):
        calls.clear()
        out = _desc(flag).current(_no_llm)
        assert "unlabelled" not in out and calls == [], flag
    d = _desc(None)
    d.cfg.pop("unlabelled_share", None)
    assert "unlabelled" not in d.current(_no_llm)  # missing block: deny by default


def test_classifier_name_follows_the_stored_taxonomy_config(monkeypatch):
    calls = []
    monkeypatch.setattr(ct, "read_unlabelled", lambda e, c: calls.append(c) or dict(PAYLOAD))
    d = _desc(True)
    d.cfg["stored_taxonomy"] = {"enabled": False, "classifier_name": "taxo_chunk_assign"}
    d.current(_no_llm)
    assert calls == ["taxo_chunk_assign"]


def test_reader_failure_or_none_drops_the_key_and_never_breaks_the_page(monkeypatch):
    monkeypatch.setattr(ct, "read_unlabelled", lambda e, c: (_ for _ in ()).throw(RuntimeError("db down")))
    out = _desc(True).current(_no_llm)
    assert "unlabelled" not in out and out["source_count"]
    monkeypatch.setattr(ct, "read_unlabelled", lambda e, c: None)
    assert "unlabelled" not in _desc(True).current(_no_llm)


def test_cached_within_ttl_and_recomputed_after_it_or_on_a_corpus_change(monkeypatch):
    calls = []
    monkeypatch.setattr(ct, "read_unlabelled", lambda e, c: calls.append(1) or dict(PAYLOAD))
    d = _desc(True)
    t = [0.0]
    d.clock = lambda: t[0]
    ov = {"source_count": 5, "corpus_updated_at": "a"}
    assert d._unlabelled(ov) == PAYLOAD and d._unlabelled(ov) == PAYLOAD and len(calls) == 1
    t[0] = float(cd.subject_cfg(d.cfg)["cache_s"]) - 1
    d._unlabelled(ov)
    assert len(calls) == 1
    d._unlabelled({"source_count": 6, "corpus_updated_at": "a"})
    assert len(calls) == 2
    t[0] = 10 * float(cd.subject_cfg(d.cfg)["cache_s"])
    d._unlabelled({"source_count": 6, "corpus_updated_at": "a"})
    assert len(calls) == 3


def test_describer_reads_the_real_tables(eng):
    d = _desc(True)
    d.engine_factory = lambda *a, **k: eng
    assert d._unlabelled({"source_count": 5, "corpus_updated_at": "x"}) == {
        "classifier": CN, "documents": 5, "documents_unlabelled": 1, "documents_without_passages": 0, "passages": 12, "passages_unlabelled": 6}


# --- (e) API model ---------------------------------------------------------------------------------------------------
def test_api_model_drops_unlabelled_when_none_and_keeps_it_when_set():
    from main import CorpusDescription
    base = CorpusDescription(summary_source="x").model_dump()
    assert "unlabelled" not in base and "stored_taxonomy" not in base
    assert CorpusDescription(summary_source="x", unlabelled=dict(PAYLOAD)).model_dump()["unlabelled"] == PAYLOAD
    both = CorpusDescription(summary_source="x", unlabelled=dict(PAYLOAD), stored_taxonomy={"a": 1}).model_dump()
    assert both["unlabelled"] == PAYLOAD and both["stored_taxonomy"] == {"a": 1}
    assert "unlabelled" in sb.CORPUS_FIELDS


# --- (f) the code-built line -----------------------------------------------------------------------------------------
def _p(u, n, x=0, uc=0, nc=0):
    return {"classifier": CN, "documents": n, "documents_unlabelled": u, "documents_without_passages": x, "passages": nc, "passages_unlabelled": uc}


def test_percent_wording_branches():
    assert "1,000 of 1,000 (100%)" in sb.unlabelled_line(_p(0, 1000))
    assert "1 of 1,000 (under 1%)" in sb.unlabelled_line(_p(999, 1000))
    assert "4 of 1,000 (under 1%)" in sb.unlabelled_line(_p(996, 1000))
    assert "5 of 1,000 (1%)" in sb.unlabelled_line(_p(995, 1000))
    assert "995 of 1,000 (over 99%)" in sb.unlabelled_line(_p(5, 1000))
    assert "994 of 1,000 (99%)" in sb.unlabelled_line(_p(6, 1000))
    assert "0 of 1,000 (0%)" in sb.unlabelled_line(_p(1000, 1000))
    assert "4 of 5 (80%)" in sb.unlabelled_line(_p(1, 5))
    assert "11,111 of 12,345 (90%)" in sb.unlabelled_line(_p(1234, 12345))


def test_malformed_payloads_give_no_line():
    ok = _p(1, 5, 0, 1, 10)
    assert sb.unlabelled_line(ok)
    bad = [None, {}, "x", 5, [], _p(0, 0), _p(6, 5), _p(-1, 5), _p(True, 5), _p(1, True), _p(1.5, 5), _p("1", 5), _p(1, 5, 0, 11, 10),
           _p(1, 5, -1), _p(1, 5, 6), _p(1, 5, 0, -1, 10), _p(1, 5, 0, 1, True)]
    bad += [{k: v for k, v in ok.items() if k != k2} for k2 in ("documents", "documents_unlabelled", "documents_without_passages",
                                                                "passages", "passages_unlabelled")]
    for b in bad:
        assert sb.unlabelled_line(b) is None, b


# --- (j) the exact display text, both branches -----------------------------------------------------------------------
TAIL = ("Documents not yet processed by the labelling program and those it could not place in a topic are not told apart. "
        "Topic and sub-topic records are assigned automatically and are approximate; "
        "a document stored in several versions is counted once.")
DOCS = "Documents with such a record: "
XC = " Of the 3 documents without one, 2 have no stored chunks, which the labelling program cannot label."


def test_display_text_with_and_without_the_no_passages_clause():
    assert sb.unlabelled_line(_p(3, 10, 2, 40, 100)) == (
        TEXT_PREFIX + "60 of 100 (60%). " + DOCS + "7 of 10 (70%)." + XC + " " + TAIL)
    assert sb.unlabelled_line(_p(3, 10, 0, 40, 100)) == (
        TEXT_PREFIX + "60 of 100 (60%). " + DOCS + "7 of 10 (70%). " + TAIL)
    assert sb.unlabelled_line(_p(3, 10, 2, 0, 0)) == (
        "**Documents with a topic or sub-topic record in the active topic hierarchy:** 7 of 10 (70%)." + XC + " " + TAIL)
    assert sb.unlabelled_line(_p(3, 10)) == (
        "**Documents with a topic or sub-topic record in the active topic hierarchy:** 7 of 10 (70%). " + TAIL)
    assert "domain record" not in sb.unlabelled_line(_p(3, 10, 2, 40, 100)).lower()
    assert "subject label" not in sb.unlabelled_line(_p(3, 10, 2, 40, 100)).lower()


# --- (g) and (i) page rendering --------------------------------------------------------------------------------------
class _St:
    def __init__(self):
        self.out = []

    def title(self, x):
        self.out.append(("title", x))

    def markdown(self, x):
        self.out.append(("markdown", x))

    def caption(self, x):
        self.out.append(("caption", x))

    def __getattr__(self, name):  # any other Streamlit call (subheader, page_link, ...) is recorded under its own name
        return lambda *a, **k: self.out.append((name, a[0] if a else ""))

    def captions(self):
        return [v for k, v in self.out if k == "caption"]


def _render(monkeypatch, data, subjects):
    monkeypatch.setattr(sb, "_corpus", lambda st: dict(data))
    monkeypatch.setattr(sb, "subjects_enabled", lambda *a, **k: subjects)
    st = _St()
    help_pages.render_corpus_page(st)
    return st.captions()


def test_line_appears_exactly_once_after_the_count_and_before_the_stored_taxonomy(monkeypatch, eng):
    stored = ct.read_stored(eng)
    line = sb.unlabelled_line(PAYLOAD)
    for subjects in (False, True):
        for with_stored in (False, True):
            data = {**FALLBACK, "unlabelled": dict(PAYLOAD)}
            if with_stored:
                data["stored_taxonomy"] = stored
            caps = _render(monkeypatch, data, subjects)
            assert caps.count(line) == 1, (subjects, with_stored)
            i = caps.index(line)
            assert "sources" in caps[0] and caps[0].startswith("**VERA-DAR's corpus:**")
            assert i >= 1 and all(not c.startswith(TEXT_PREFIX) for c in caps[:i])
            if with_stored:
                j = next(k for k, c in enumerate(caps) if "Stored topic and sub-topic records" in c)
                assert i < j, (subjects, with_stored)
            else:
                assert not any("Stored topic and sub-topic records" in c for c in caps)


def test_line_is_absent_when_the_key_is_absent_or_malformed(monkeypatch):
    for data in (dict(FALLBACK), {**FALLBACK, "unlabelled": None}, {**FALLBACK, "unlabelled": _p(9, 5)}):
        assert not any(c.startswith(TEXT_PREFIX) for c in _render(monkeypatch, data, False))


def test_line_is_not_on_about_the_notes_or_the_sidebar(monkeypatch):
    data = {**FALLBACK, "unlabelled": dict(PAYLOAD)}
    st = _St()
    sb.render_corpus(st, data)
    sb.render_corpus_notes(st, data)
    assert not any(TEXT_PREFIX in c for c in st.captions())
    import inspect
    assert "unlabelled" not in inspect.getsource(sb.render_sources_sidebar) + inspect.getsource(sb.render_corpus_notes)
    monkeypatch.setattr(sb, "_corpus", lambda st: dict(data))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages/about.py"), default_timeout=30).run()
    assert not at.exception
    shown = " ".join(e.value for kind in ("markdown", "caption") for e in getattr(at.main, kind))
    assert "topic or sub-topic record" not in shown


def test_wp6_fixture_exact_hand_counts(eng):
    with eng.connect() as c:
        assert tuple(c.execute(text(ct.Q_UNLABELLED), {"cn": CN}).one()) == (5, 4, 6)
        assert tuple(c.execute(text(ct.Q_TOTALS)).one()) == (12, 5, 5)
    assert ct.read_unlabelled(eng, CN) == {"classifier": CN, "documents": 5, "documents_unlabelled": 1, "documents_without_passages": 0,
                                          "passages": 12, "passages_unlabelled": 6}


# --- (h) the query is read-only; SQL contract subset -------------------------------------------------------------------
def test_query_is_read_only_and_filters_on_the_classifier():
    q = ct.Q_UNLABELLED
    u = q.upper()
    assert u.startswith("WITH") and "classifier_name = :cn" in q and "_eff_v" not in q
    assert not any(w in u for w in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE ", "TRUNCATE"))


def test_every_column_pair_of_the_query_is_in_the_required_contract_or_exempt_by_a_pending_draft():
    found = set(extract_columns(ct.Q_UNLABELLED))
    assert found, "extractor saw nothing: the query is not in the shape the contract extractor reads"
    pending = pending_objects()
    exempt = {(t, c) for (t, c) in found if t in pending and c in pending[t][1]}
    outside = (found - exempt) - set(collect_required())
    assert not outside, sorted(outside)


def test_shipped_config_flag_and_docs_and_help_text():
    cfg = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    blk = cfg["corpus_description"]["unlabelled_share"]
    assert blk["enabled"] is True and "CORPUS-UNCAT-1113" in blk["_note"]
    assert cfg["corpus_description"]["stored_taxonomy"]["enabled"] is False  # untouched
    assert "`unlabelled`" in (ROOT / "docs" / "api.md").read_text()
    paras = json.loads((ROOT / "config" / "ui_help_pages.json").read_text())["corpus_page"]["paragraphs"]
    assert any("no subject label" in p and "approximate" in p for p in paras)
    assert paras[-1] == "The page never names an individual source." or any(p.startswith("The page never names") for p in paras)
