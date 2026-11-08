"""CORPUS-DESC-1107 (owner 2026-11-07): the corpus description always says what the corpus covers, by domain and, where
possible, by topic and subtopic, never only a count; it is its own Corpus Description page; How to use points to it.
Mocked DB rows only (fake engine); no database, no network, no model call."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vera import corpus_description as cd
from vera import help_pages, sources_sidebar as sb

ROOT = Path(__file__).resolve().parent.parent
_NORM = lambda t: re.sub(r"\s+", " ", re.sub(r"[*\\]", "", str(t))).strip()  # noqa: E731


def _row(k, field, parent=None, state="active", anchor=None):
    return {"topic_key": k, "version": 1, "label": f"Label {k}", "field": field,
            "anchor_text": anchor or ("anchor words for " + k + " " + k), "parent_openalex_topic": "none",
            "parent_topic_key": parent, "origin": "seed", "state": state}


ROWS = [_row("flu", "epidemiology", anchor="influenza forecasting outbreak transmission surveillance epidemic"),
        _row("flu-model", "epidemiology", parent="flu", anchor="influenza forecasting model ensemble nowcast compartmental"),
        _row("coral", "ecology", anchor="coral reef bleaching marine ecology biodiversity ocean warming"),
        _row("gone", "ecology", state="retired", anchor="retired topic should never match anything here at all")]
ENTRIES = [{"id": r["topic_key"], "label": r["label"], "field": r["field"], "parent_topic_key": r["parent_topic_key"]}
           for r in ROWS if r["state"] == "active"]


# --- pure computation -----------------------------------------------------------------------------------------------
def test_subject_areas_rolls_up_subtopics_and_shares_add_up():
    primary = {1: "flu", 2: "flu-model", 3: "flu", 4: "coral", 5: None, 6: "unknown-key"}
    sa = cd.subject_areas(ENTRIES, primary)
    assert sa["basis_documents"] == 6
    epi, eco = sa["domains"]
    assert (epi["label"], epi["share"], eco["label"], eco["share"]) == ("epidemiology", 0.5, "ecology", 0.1667)
    assert epi["topics"] == [{"label": "Label flu", "share": 0.5, "subtopics": [{"label": "Label flu-model", "share": 0.1667}]}]
    assert sa["unmatched_share"] == round(2 / 6, 4)  # no key, and a key that is not an active registry topic
    assert sa["other_domains"] == {"count": 0, "share": 0.0}
    assert cd.subject_areas(ENTRIES, {}) is None


def test_display_limits_only_bound_listing_and_sum_the_rest():
    entries = [{"id": f"t{i}", "label": f"Topic {i}", "field": f"area {i}", "parent_topic_key": None} for i in range(5)]
    primary = {i: f"t{i % 5}" for i in range(10)}
    sa = cd.subject_areas(entries, primary, {"display_domains": 2})
    assert len(sa["domains"]) == 2 and sa["other_domains"] == {"count": 3, "share": 0.6}
    assert sa["unmatched_share"] == 0.0


def test_assign_topic_stored_key_first_then_anchor_match_then_none():
    from vera import micro_topic
    reg = micro_topic.Registry({"entries": [{"id": e["id"], "anchor_text": r["anchor_text"], "field": e["field"]}
                                            for e, r in zip(ENTRIES, ROWS)]})
    stored = [{"scheme": "registry", "id": "coral", "score": 1.0}]
    assert cd.assign_topic(reg, "Influenza forecasting", "", stored, 2.0, 2000) == "coral"
    assert cd.assign_topic(reg, "Influenza forecasting", "outbreak surveillance epidemic", None, 2.0, 2000) == "flu"
    assert cd.assign_topic(reg, "Unrelated words entirely", "", None, 2.0, 2000) is None
    assert cd.assign_topic(reg, "", "", "not json", 2.0, 2000) is None


# --- read path: one read-only SELECT, mocked rows ------------------------------------------------------------------
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def one(self):
        return self.rows[0]

    def all(self):
        return self.rows

    def scalar(self):
        return self.rows[0][0]

    def __iter__(self):
        return iter(self.rows)


DOCS = [(3, "Influenza forecasting with ensembles", "influenza forecasting model ensemble nowcast", None),
        (2, "Coral reef bleaching", "coral reef bleaching marine", None),
        (1, "A study of something else", "", None)]


class _Eng:
    def __init__(self, docs=DOCS):
        self.log, self.docs = [], docs

    def connect(self):
        eng = self

        class C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                q = str(sql)
                eng.log.append((q, params))
                if q == cd.SUBJECT_ROWS_SQL:
                    return _Res(eng.docs)
                if q.startswith("SELECT count(*), max(created_at)"):
                    return _Res([(3, None, 0)])
                if "min(substring" in q:
                    return _Res([("2005", "2026")])
                if "interval '30 days'" in q:
                    return _Res([(0,)])
                return _Res([])
        return C()


def test_subject_query_is_one_bounded_select_on_existing_tables():
    q = cd.SUBJECT_ROWS_SQL
    assert q.lstrip().upper().startswith("SELECT") and "LIMIT :n" in q
    assert set(re.findall(r"vera_vjay\.(\w+)", q)) == {"sources", "candidates"}
    from tests import sql_contract_support as scs
    assert scs.extract_columns(q) <= {("sources", c) for c in ("canonical_source_id", "provenance", "content", "candidate_id",
                                                                "version")} | {("candidates", c) for c in ("title", "candidate_id")}


def test_subject_areas_for_reads_registry_through_reader_and_rows_from_db():
    eng = _Eng()
    sa = cd.subject_areas_for(eng, {}, {"text_window_chars": 500}, registry_reader=lambda: (True, ROWS))
    assert [p for q, p in eng.log] == [{"ms": "5000"}, {"w": 500, "n": cd.SUBJECT_DEFAULTS["scan_max"]}]
    assert eng.log[0][0] == "SELECT set_config('statement_timeout', :ms, true)"  # transaction-local, before the SELECT
    assert sa["basis_documents"] == 3
    assert {d["label"] for d in sa["domains"]} == {"epidemiology", "ecology"}
    assert sa["unmatched_share"] == round(1 / 3, 4)
    assert "Influenza" not in repr(sa) and "Coral reef" not in repr(sa)  # no source title leaves the module
    assert cd.subject_areas_for(eng, {}, {}, registry_reader=lambda: (False, [])) is None  # table absent: none, no seed
    assert cd.subject_areas_for(eng, {"subject_areas": {"enabled": False}}, {}, registry_reader=lambda: (True, ROWS)) is None


def _describer(subject_reader):
    d = cd.CorpusDescriber({"read_db_url_env": "X", "max_titles": 5, "retry_s": 300, "prompt_version": "v"},
                           engine_factory=lambda *a, **k: _Eng(), env={"X": "postgresql://fake"}, clock=lambda: 0.0)
    d.store_cfg = {"enabled": False}
    d.subject_mt_cfg, d.subject_registry_reader = {}, subject_reader
    return d


def _no_llm(*a, **k):
    raise RuntimeError("provider down")


def test_fallback_state_carries_subject_areas_and_no_duplicate_count():
    out = _describer(lambda: (True, ROWS)).current(_no_llm)
    assert out["summary_source"] == "deterministic_fallback"
    assert out["summary"] == "Publication years: 2005 to 2026."  # the count is on the section's first line only
    assert "temporarily unavailable" not in out["summary"]
    assert out["subject_areas"]["domains"]
    from main import CorpusDescription
    assert CorpusDescription(**out).subject_areas == out["subject_areas"]


def test_subject_failure_never_breaks_the_digest():
    def boom():
        raise RuntimeError("registry unreadable")
    out = _describer(boom).current(_no_llm)
    assert out["summary_source"] == "deterministic_fallback" and "subject_areas" not in out


# --- rendering ------------------------------------------------------------------------------------------------------
def _capture():
    cap = []

    class St:
        def caption(self, x):
            cap.append(x)
    return St(), cap


SA = cd.subject_areas(ENTRIES, {1: "flu", 2: "flu-model", 3: "coral", 4: None})
FALLBACK = {"summary_source": "deterministic_fallback", "source_count": 314, "corpus_updated_at": "2026-11-07T00:00:00+00:00",
            "summary": "Publication years: 2005 to 2026.", "topics": [], "coverage": None, "topic_shares": None,
            "summary_generated_at": None, "summary_model": None, "subject_areas": SA}
DIGEST = {**FALLBACK, "summary_source": "agentic_checked", "summary": "A collection on influenza and reefs.",
          "summary_generated_at": "2026-11-07T00:00:00+00:00", "coverage": "Mostly epidemiology."}


def test_render_fallback_describes_domains_topics_subtopics():
    st, cap = _capture()
    sb.render_corpus(st, FALLBACK)
    text = _NORM(" ".join(cap))
    assert text.count("314 sources") == 1 and "temporarily unavailable" not in text
    assert "Subject areas (provisional; share of 4 documents" in text  # TAXO-MATCH-FIX-1108 wording
    assert "Epidemiology: 50%. Topics: Label flu 50% (including Label flu-model 25%)." in text
    assert "Ecology: 25%. Topics: Label coral 25%." in text and "Not yet matched to a listed subject area: 1 of 4 documents (25%)." in text


def test_render_without_subjects_says_so_and_drops_backend_labels():
    st, cap = _capture()
    sb.render_corpus(st, {**FALLBACK, "subject_areas": None})
    assert sb.NO_SUBJECTS in cap
    bad = {**SA, "domains": [{**SA["domains"][0], "label": "groq outputs"}]}
    assert sb._subject_lines(bad) == []
    assert sb._subject_lines({"basis_documents": 3, "domains": [], "other_domains": {}, "unmatched_share": 1.0})[-1] == \
        "No document matched a listed subject area yet."


def test_render_checked_digest_shows_subjects_and_model_line():
    st, cap = _capture()
    sb.render_corpus(st, DIGEST)
    text = _NORM(" ".join(cap))
    assert "Subject areas" in text and "Description written by a language model on 2026-11-07" in text


# --- pages ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def _offline(monkeypatch):
    import requests

    def _no_net(*_a, **_k):
        raise requests.exceptions.ConnectionError("offline test")
    monkeypatch.setattr(requests, "get", _no_net)
    monkeypatch.setattr(requests, "post", _no_net)


def _page_text(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    out = []
    for kind in ("title", "header", "subheader", "markdown", "caption"):
        out += [_NORM(e.value) for e in getattr(at.main, kind)]
    return " ".join(out), at


def test_corpus_page_is_in_the_nav_after_about():
    paths = [p for p, _ in sb.NAV]
    assert paths[paths.index("pages/about.py") + 1] == "pages/corpus_description.py" == help_pages.CORPUS_PAGE
    assert (ROOT / help_pages.CORPUS_PAGE).exists()


@pytest.mark.parametrize("state", ["deterministic", "digest"])
def test_corpus_page_renders_in_both_states(monkeypatch, _offline, state):
    data = FALLBACK if state == "deterministic" else DIGEST
    monkeypatch.setattr(sb, "_corpus", lambda st: dict(data))
    text, at = _page_text("pages/corpus_description.py")
    assert "Corpus Description" in [_NORM(t.value) for t in at.main.title]
    assert text.count("VERA-DAR's corpus: 314 sources, newest source 2026-11-07.") == 1
    assert "Epidemiology: 50%" in text
    assert ("Description written by a language model" in text) is (state == "digest")
    assert ("Current coverage: Mostly epidemiology." in text) is (state == "digest")


def test_about_has_one_link_line_and_no_corpus_section(monkeypatch, _offline):
    monkeypatch.setattr(sb, "_corpus", lambda st: dict(FALLBACK))
    text, at = _page_text("pages/about.py")
    assert help_pages.CORPUS_HEADING not in text and "Subject areas" not in text and "314 sources" not in text
    labels = [e.proto.label for e in at.get("page_link")]
    link = help_pages.load()["about"]["corpus_link"]
    assert link in labels or link in text


def test_how_to_use_has_the_corpus_tip_and_link(_offline):
    text, at = _page_text("pages/how_to_use.py")
    assert "For the best answers, ask about subjects already in VERA-DAR's collection." in text
    assert "slower and less likely to get an answer" in text
    labels = [e.proto.label for e in at.get("page_link")]
    link = help_pages.load()["how_to_use"]["sections"][0]["corpus_link"]
    assert link in labels or link in text
    assert "DAR-9" not in text


def test_registry_is_read_on_the_callers_engine_not_the_rw_engine():
    from vera import micro_topic, topic_registry
    used = []

    class Eng:
        def connect(self):
            used.append(1)
            raise ConnectionError("fake engine")
    with pytest.raises(topic_registry.TopicRegistryError):
        micro_topic.registry_active_data({}, Eng())
    assert used == [1]


def test_engine_gets_a_connect_timeout():
    seen = []
    d = cd.CorpusDescriber({"read_db_url_env": "X", "max_titles": 5, "retry_s": 300, "prompt_version": "v"},
                           engine_factory=lambda url, **kw: seen.append(kw) or _Eng(), env={"X": "postgresql://fake"})
    d._engine_or_none()
    assert seen[0]["connect_args"] == {"connect_timeout": cd.SUBJECT_DEFAULTS["connect_timeout_s"]}


def test_subjects_single_flight_never_waits_and_keeps_the_previous_value():
    calls = []

    def reader():
        calls.append(1)
        return (True, ROWS)
    d = _describer(reader)
    ov = {"source_count": 3, "corpus_updated_at": None}
    d._subject_lock.acquire()  # a rebuild is in flight
    try:
        assert d._subjects(ov) is None and calls == []  # nothing cached yet: no wait, no second build
    finally:
        d._subject_lock.release()
    first = d._subjects(ov)
    assert first and calls == [1]
    d.clock = lambda: 10_000.0  # cache expired
    d._subject_lock.acquire()
    try:
        assert d._subjects(ov) == first and calls == [1]  # concurrent miss: previous value, no stampede
    finally:
        d._subject_lock.release()
