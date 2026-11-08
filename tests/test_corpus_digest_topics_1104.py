"""TOPIC-TAGGING-1104: digest topics and scope from vera_vjay.toptag_document_topics. Offline: fake engine, fake llm."""
import datetime
import threading

import pytest

from vera import corpus_description as cd
from tests.digest_fakes import FakeLLM

OVERVIEW = {"source_count": 20, "corpus_updated_at": "2026-11-04T10:00:00+00:00"}
PROFILE = {"titles": [], "excerpts": [], "source_types": {"journal-article": 20}, "discovered_via": {}, "text_kinds": {},
           "published_years": ["2005", "2026"], "year_counts": {}, "added_last_30_days": 3}
# R4: table mode is an explicit owner setting (T12); these tests exercise it, so they set it
CFG = {"read_db_url_env": "X", "max_titles": 6, "prompt_version": "v2", "retry_s": 60, "echo_retry_s": 3600,
       "topics_source": "table_if_present", "table_cache_s": 0}
TREE_READER = lambda: TREE  # noqa: E731
NEWEST = datetime.datetime(2026, 11, 4, 9, 0, tzinfo=datetime.timezone.utc)
# 20 documents (some chunk-stored full documents); 12 primary epidemics, 4 primary ecology, 4 untagged
TOPICS = [{"key": "epidemics", "label": "epidemic forecasting", "primary_docs": 12, "any_docs": 14},
          {"key": "ecology", "label": "marine and soil ecology", "primary_docs": 4, "any_docs": 6}]
SAMPLES = {"epidemics": ["Influenza nowcasting with hospital admissions", "Ensemble forecasts of dengue incidence"],
           "ecology": ["Coral bleaching under marine heatwaves"]}
GOOD = "The collection covers epidemic forecasting and ecology."
# R4 (T5 guard tests): the registry comes from vera.micro_topic.registry_topic_tree (injected here); the roll-up runs in Python.
TREE = {"epidemics": {"label": "epidemic forecasting", "state": "active", "parent": None, "version": 1},
        "ecology": {"label": "marine and soil ecology", "state": "active", "parent": None, "version": 1},
        "influenza": {"label": "influenza forecasting", "state": "active", "parent": "epidemics", "version": 1},
        "dengue": {"label": "dengue forecasting", "state": "active", "parent": "epidemics", "version": 1}}
# document rows (source_id, topic_key, weight) giving TOPICS: docs 1-12 epidemics primary, 13-16 ecology primary, 13-14 also
# epidemics, 1-2 also ecology; docs 17-20 untagged
DOC_ROWS = ([(i, "epidemics", 0.9) for i in range(1, 13)] + [(i, "ecology", 0.9) for i in range(13, 17)]
            + [(13, "epidemics", 0.3), (14, "epidemics", 0.3), (1, "ecology", 0.2), (2, "ecology", 0.2)])
SAMPLE_SIDS = {1: "Influenza nowcasting with hospital admissions", 2: "Ensemble forecasts of dengue incidence",
               13: "Coral bleaching under marine heatwaves"}


class Res:
    def __init__(self, rows):
        self.rows = rows

    def scalar(self):
        return self.rows[0][0] if self.rows else None

    def one(self):
        return self.rows[0]

    def __iter__(self):
        return iter(self.rows)

    def all(self):
        return list(self.rows)


class FakeConn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.eng.sql.append(sql)
        e = self.eng
        if "to_regclass" in sql:
            return Res([(not e.exists,)])
        if "FROM vera_vjay.toptag_document_topics d" in sql:
            if not e.rows_n:
                return Res([])
            rows = list(e.doc_rows) + list(e.subs)
            return Res([(sid, k, w, e.rows_n if i == 0 else 1, NEWEST) for i, (sid, k, w) in enumerate(rows)])
        if "FILTER (WHERE st.outcome" in sql:  # R5 (R5K-3) coverage
            return Res([e.cov])
        if "SELECT DISTINCT e.source_id" in sql:
            return Res([(x,) for x in e.open_docs])
        if "count(DISTINCT canonical_source_id) FROM vera_vjay.sources" in sql:
            return Res([(e.n_docs,)])
        if "ANY(:ids)" in sql:
            e.sample_params = params
            return Res([(sid, t, sid in getattr(e, "fulldoc_sids", ())) for sid, t in sorted(e.samples.items()) if sid in params["ids"]])
        raise AssertionError("unexpected SQL: " + sql[:80])


class FakeEngine:
    def __init__(self, exists=True, rows_n=30, n_docs=20, doc_rows=DOC_ROWS, samples=SAMPLE_SIDS, subs=()):
        self.exists, self.rows_n, self.n_docs, self.doc_rows, self.samples = exists, rows_n, n_docs, doc_rows, samples
        self.subs = subs
        self.cov, self.open_docs = (100, 100), ()  # R5: full coverage unless a test says otherwise
        self.sql, self.sample_params = [], None

    def connect(self):
        return FakeConn(self)


def table_llm(sentences=(GOOD,), **kw):
    # labels are not checked in table mode, so the checker only sees the sentences (nothing to cluster)
    return FakeLLM([("epidemic forecasting", [1]), ("marine and soil ecology", [2])], list(sentences), **kw)


# ---- topics_from_table -------------------------------------------------------------------------------------------
def test_table_absent_returns_none():
    assert cd.topics_from_table(FakeEngine(exists=False), CFG, TREE_READER) is None


def test_table_without_active_rows_returns_empty_topics_not_none():
    tab = cd.topics_from_table(FakeEngine(rows_n=0), CFG, TREE_READER)  # R4 (skeptic 2): table exists -> never the model-cluster fallback
    assert tab["topics"] == [] and tab["n_docs"] == 20


def test_table_present_shape_and_signature():
    eng = FakeEngine()
    tab = cd.topics_from_table(eng, CFG, TREE_READER)
    assert tab["n_docs"] == 20 and tab["topics"] == sorted([{**t, "subtopics": []} for t in TOPICS], key=lambda t: t["key"])
    assert tab["signature"] == (20, 20, NEWEST.isoformat(), cd.registry_signature(TREE), 30 + 19, 100, 100)  # R4 set sum; R5 cov
    assert all(s.lstrip().upper().startswith("SELECT") for s in eng.sql)  # read-only
    assert not [q for q in eng.sql if "topreg" in q]  # R4: no registry SQL in this module


def test_rollup_denies_inactive_chains_and_orders_ties():
    tree = {**TREE, "old": {"label": "old", "state": "retired", "parent": None, "version": 2},
            "under-old": {"label": "u", "state": "active", "parent": "old", "version": 1},
            "lvl2": {"label": "l2", "state": "active", "parent": "influenza", "version": 1},
            "lvl3": {"label": "l3", "state": "active", "parent": "lvl2", "version": 1},
            "cyc-a": {"label": "a", "state": "active", "parent": "cyc-b", "version": 1},
            "cyc-b": {"label": "b", "state": "active", "parent": "cyc-a", "version": 1}}
    assert cd.roll_up(tree, "influenza") == "epidemics" and cd.roll_up(tree, "lvl2") == "epidemics"  # depth 2 rolls up
    for k in ("old", "under-old", "lvl3", "cyc-a", "missing"):
        assert cd.roll_up(tree, k) is None, k  # inactive chain, too deep, cycle, unknown: not counted
    rows = [(1, "epidemics", 0.5), (1, "ecology", 0.5), (2, "under-old", 0.9), (2, "ecology", 0.1)]
    tab = cd.topics_from_table(FakeEngine(doc_rows=rows), CFG, lambda: tree)
    assert tab["primary"] == {1: "ecology", 2: "ecology"}  # tie by key; a retired chain is never primary
    sql = cd.TABLE_DOC_ROWS_SQL
    assert "d.state = 'active'" in sql and "max(v.version)" in sql and "topreg" not in sql and "INSERT" not in sql.upper()


def test_subtopics_attached_to_top_level_topics():
    subs = [("epidemics", "influenza", "influenza forecasting", 9), ("epidemics", "dengue", "dengue forecasting", 5),
            ("unknown", "x", "ignored", 1)]
    rows = [(i, "influenza", 0.5) for i in range(1, 10)] + [(i, "dengue", 0.4) for i in range(1, 6)] + [(3, "x", 0.9)]
    tab = cd.topics_from_table(FakeEngine(subs=rows), CFG, TREE_READER)
    epi = next(t for t in tab["topics"] if t["key"] == "epidemics")
    assert epi["subtopics"] == [{"key": "influenza", "label": "influenza forecasting", "any_docs": 9},
                                {"key": "dengue", "label": "dengue forecasting", "any_docs": 5}]
    assert epi["primary_docs"] == 12 and next(t for t in tab["topics"] if t["key"] == "ecology")["subtopics"] == []
    assert "subtopic" not in cd.table_clusters(tab, SAMPLES)["scope"][0]["label"]  # Scope stays top-level labels
    assert "influenza forecasting" not in cd.run_table_digest(dict(OVERVIEW), dict(PROFILE), cd.table_clusters(tab, SAMPLES),
                                                              table_llm(), CFG)["coverage"]


def test_sample_sql_includes_fulldoc_sources_and_prefers_them():
    eng = FakeEngine()
    tab = cd.topics_from_table(eng, CFG, TREE_READER)
    by_topic = {}
    for sid, k in tab["primary"].items():
        if sid in eng.samples:
            by_topic.setdefault(k, []).append(sid)
    k, sids = max(by_topic.items(), key=lambda kv: len(kv[1]))
    assert len(sids) >= 2
    eng.fulldoc_sids = {sorted(sids)[-1]}  # only the last source is chunk-stored full-document
    fd_title = eng.samples[sorted(sids)[-1]]
    assert "<> :fd_kind" not in cd.TABLE_SAMPLE_SQL
    assert cd.table_title_samples(eng, 1, tab["primary"])[k] == [fd_title]
    assert fd_title in cd.table_title_samples(eng, 3, tab["primary"])[k]


def test_sample_sql_passes_kind():
    eng = FakeEngine()
    tab = cd.topics_from_table(eng, CFG, TREE_READER)
    out = cd.table_title_samples(eng, 3, tab["primary"])
    assert {k: sorted(v) for k, v in out.items()} == {k: sorted(v) for k, v in SAMPLES.items()}
    assert eng.sample_params == {"ids": sorted(tab["primary"]), "fd_kind": cd.FULLDOC_CHUNKS_KIND}
    assert cd.table_title_samples(eng, 3, {}) == {}


# ---- table mode pipeline -----------------------------------------------------------------------------------------
def tab_dict():
    return {"n_docs": 20, "topics": TOPICS, "signature": (20, 30, "x")}


def test_pipeline_table_mode_no_cluster_call_and_shares_over_all_documents():
    llm = table_llm()
    table = cd.table_clusters(tab_dict(), SAMPLES)
    out = cd.run_table_digest(dict(OVERVIEW), dict(PROFILE), table, llm, CFG)
    assert "cluster" not in llm.steps() and llm.steps() == ["draft", "check"]
    assert out["prose"] == "checked" and out["summary"] == GOOD
    sh = out["topic_shares"]
    assert sh["basis_documents"] == 20  # all documents, full documents included
    assert [(e["label"], e["documents"], e["pct"]) for e in sh["main"]] == [
        ("epidemic forecasting", 12, 60), ("marine and soil ecology", 4, 20)]
    assert sh["ungrouped"] == {"documents": 4, "pct": 20}  # untagged documents are ungrouped
    assert out["topics"] == ["epidemic forecasting", "marine and soil ecology"]
    # Scope ordered by documents carrying the topic at all
    assert out["coverage"] == "The collection currently includes research on epidemic forecasting and marine and soil ecology."


def test_labels_need_no_checker_approval_and_titles_stay_non_fulldoc():
    llm = table_llm()
    table = cd.table_clusters(tab_dict(), SAMPLES)
    cd.run_table_digest(dict(OVERVIEW), dict(PROFILE), table, llm, CFG)
    check = next(c for c in llm.calls if c["step"] == "check")
    assert "Topic label:" not in check["messages"][1]["content"]
    assert sorted(table["titles"]) == sorted(t for ts in SAMPLES.values() for t in ts)  # only the sampled titles exist


def test_scope_orders_by_any_docs_not_primary():
    topics = [{"key": "a", "label": "alpha topic", "primary_docs": 10, "any_docs": 10},
              {"key": "b", "label": "beta topic", "primary_docs": 2, "any_docs": 15}]
    t = cd.table_clusters({"n_docs": 20, "topics": topics, "signature": ()}, {})
    assert [s["label"] for s in t["scope"]] == ["beta topic", "alpha topic"]
    assert [c["label"] for c in t["clusters"]] == ["alpha topic", "beta topic"]


def test_ties_are_deterministic_by_label():
    topics = [{"key": "z", "label": "zeta studies", "primary_docs": 5, "any_docs": 5},
              {"key": "a", "label": "alpha studies", "primary_docs": 5, "any_docs": 5}]
    for order in (topics, topics[::-1]):
        t = cd.table_clusters({"n_docs": 10, "topics": order, "signature": ()}, {})
        assert [c["label"] for c in t["clusters"]] == ["alpha studies", "zeta studies"]


def test_prose_failure_still_returns_table_topics_with_fallback_summary():
    llm = table_llm(fail_step="draft")
    table = cd.table_clusters(tab_dict(), SAMPLES)
    out = cd.run_table_digest(dict(OVERVIEW), dict(PROFILE), table, llm, CFG)
    assert out["prose"] == "fallback" and out["summary"] == cd.fallback_summary(OVERVIEW, PROFILE)
    assert out["topics"] == ["epidemic forecasting", "marine and soil ecology"]
    assert out["topic_shares"]["basis_documents"] == 20 and out["coverage"]


def test_unsupported_prose_falls_back_not_invented():
    llm = table_llm(supported=set())
    out = cd.run_table_digest(dict(OVERVIEW), dict(PROFILE), cd.table_clusters(tab_dict(), SAMPLES), llm, CFG)
    assert out["prose"] == "fallback" and out["summary"] == cd.fallback_summary(OVERVIEW, PROFILE)


def test_label_echoing_a_sampled_title_is_left_out():
    topics = [{"key": "x", "label": "Influenza nowcasting with hospital admissions", "primary_docs": 5, "any_docs": 5}]
    t = cd.table_clusters({"n_docs": 5, "topics": topics, "signature": ()}, {"x": ["Influenza nowcasting with hospital admissions"]})
    assert t["clusters"] == [] and t["scope"] == []


# ---- describer ---------------------------------------------------------------------------------------------------
class DescEngine(FakeEngine):
    def connect(self):
        c = FakeConn(self)
        orig = c.execute

        def ex(stmt, params=None):
            sql = str(stmt)
            if sql.startswith("SELECT count(*), max(created_at)"):  # 2780e1c: the head query also counts full documents
                return Res([(self.n_docs, datetime.datetime(2026, 11, 4, 10, tzinfo=datetime.timezone.utc), 0)])
            if "GROUP BY 1 ORDER BY 2 DESC LIMIT 8" in sql or "GROUP BY 1 ORDER BY 1" in sql:
                return Res([])
            if "min(substring" in sql:
                return Res([("2005", "2026")])
            if "interval '30 days'" in sql:
                return Res([(3,)])
            if "FROM vera_vjay.sources s" in sql and "LIMIT :n" in sql:
                return Res([])
            return orig(stmt, params)
        c.execute = ex
        return c


def describer(engine, **cfg):
    d = cd.CorpusDescriber({**CFG, **cfg}, engine_factory=lambda url, **kw: engine, env={"X": "postgres://stand-in"})
    d.store_cfg = {"enabled": False}
    d.tree_reader = TREE_READER
    return d


class GateLLM(FakeLLM):
    """Blocks inside the model until released, to prove current() never waits for the model."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()
        self.started = 0
        self._lk = threading.Lock()

    def __call__(self, *a, **kw):
        with self._lk:
            self.started += 1
        self.gate.wait(5)
        return super().__call__(*a, **kw)


def test_current_returns_at_once_without_calling_model_then_built_digest():
    eng, llm = DescEngine(), GateLLM([("epidemic forecasting", [1]), ("marine and soil ecology", [2])], [GOOD])
    d = describer(eng)
    first = d.current(llm)  # the model is blocked: a call here would hang the test
    assert llm.calls == [] and not llm.gate.is_set()  # no model reply was needed to answer
    assert first["topics_source"] == "table" and first["summary_source"] == "deterministic_fallback"
    assert first["topics"] == ["epidemic forecasting", "marine and soil ecology"]
    assert first["topic_shares"]["basis_documents"] == 20
    llm.gate.set()
    d._thread.join(5)
    second = d.current(llm)
    assert second["summary"] == GOOD and second["summary_source"] == cd.CHECKED
    assert "cluster" not in llm.steps()


def test_single_flight_two_triggers_one_build():
    eng, llm = DescEngine(), GateLLM([("epidemic forecasting", [1]), ("marine and soil ecology", [2])], [GOOD])
    d = describer(eng)
    d.current(llm)
    d.current(llm)
    assert d.request_refresh() is False  # one build already in flight
    llm.gate.set()
    d._thread.join(5)
    assert llm.steps().count("draft") == 1
    d.current(llm)
    assert llm.steps().count("draft") == 1  # same signature: no rebuild


def test_signature_change_starts_a_new_build_and_old_digest_shows_meanwhile():
    eng, llm = DescEngine(), table_llm()
    d = describer(eng)
    d.current(llm)
    d._thread.join(5)
    eng.rows_n = 31  # document_topics changed
    shown = d.current(llm)
    assert shown["summary"] == GOOD  # last digest straight away
    d._thread.join(5)
    assert llm.steps().count("draft") == 2


def test_request_refresh_never_raises():
    d = describer(DescEngine(exists=True))
    assert d.request_refresh() is False  # no llm yet
    d._llm = table_llm()
    d._engine = object()  # engine without connect(): the helper raises inside
    assert d.request_refresh() is False
    d2 = cd.CorpusDescriber(dict(CFG), env={})
    assert d2.request_refresh(table_llm()) is False  # not connected


def test_request_refresh_starts_build_on_change():
    eng, llm = DescEngine(), table_llm()
    d = describer(eng)
    d._llm = llm
    assert d.request_refresh() is True
    d._thread.join(5)
    assert d.current(llm)["summary"] == GOOD


# ---- fall back to the old path -----------------------------------------------------------------------------------
@pytest.fixture
def old_path(monkeypatch):
    calls = []

    def fake_stamp(engine, n, ex=0):
        calls.append("stamp")
        return dict(OVERVIEW), dict(PROFILE, titles=["t"])

    def fake_pipe(overview, profile, llm, cfg, *a, **kw):
        calls.append("pipeline")
        return {"summary": "old", "topics": ["x"], "coverage": "c", "topic_shares": None, "model": "m", "trace": []}
    monkeypatch.setattr(cd, "stamp_and_profile", fake_stamp)
    monkeypatch.setattr(cd, "run_digest_pipeline", fake_pipe)
    return calls


@pytest.mark.parametrize("engine", [FakeEngine(exists=False)])
def test_absent_table_uses_old_path(old_path, engine):
    d = describer(engine)
    out = d.current(table_llm())
    assert old_path == ["stamp", "pipeline"] and out["summary"] == "old" and "topics_source" not in out


def test_model_clusters_config_forces_old_path(old_path):
    d = describer(FakeEngine(), topics_source="model_clusters")
    out = d.current(table_llm())
    assert old_path == ["stamp", "pipeline"] and out["summary"] == "old"


def test_unreadable_table_falls_back_to_old_path(old_path, monkeypatch):
    monkeypatch.setattr(cd, "topics_from_table", lambda e, c=None, t=None: (_ for _ in ()).throw(RuntimeError("db")))
    d = describer(FakeEngine())
    assert d.current(table_llm())["summary"] == "old"


# ---- R4 (convergence round 4, 2026-11-05) -------------------------------------------------------------------------------------
def test_default_is_model_clusters_even_with_active_table_rows(old_path):
    """R4 (p3m3 1 = skeptic 3): with no topics_source set, an active document_topics row never changes the public digest."""
    cfg = {k: v for k, v in CFG.items() if k != "topics_source"}
    eng = FakeEngine()
    d = cd.CorpusDescriber(cfg, engine_factory=lambda url, **kw: eng, env={"X": "postgres://stand-in"})
    d.store_cfg = {"enabled": False}
    out = d.current(table_llm())
    assert old_path == ["stamp", "pipeline"] and out["summary"] == "old" and "topics_source" not in out
    assert not [q for q in eng.sql if "toptag_document_topics" in q]  # the table is not even read
    d._llm = table_llm()
    assert d.request_refresh() is False


def test_table_mode_with_no_active_roots_shows_no_model_labels(monkeypatch):
    """R4 (skeptic 2): table present but no active registry topic -> empty topics and a fixed line, no model call."""
    eng = DescEngine(rows_n=0)
    llm = table_llm()
    d = describer(eng)
    out = d.current(llm)
    assert out["topics"] == [] and out["coverage"] == cd.NO_TABLE_TOPICS_LINE and out["topics_source"] == "table"
    assert llm.calls == [] and d._thread is None
    d._llm = llm
    assert d.request_refresh() is False and llm.calls == []


def test_request_refresh_prose_calls_go_through_the_pacer():
    """R4 (skeptic 4): the worker's pacer is applied to every digest prose call."""
    class Pace:
        n = 0

        def wait(self, extra=0.0, tokens=0):
            Pace.n += 1
    eng, llm = DescEngine(), table_llm()
    d = describer(eng)
    d._llm = llm
    assert d.request_refresh(pace=Pace()) is True
    d._thread.join(5)
    assert Pace.n == len(llm.calls) > 0


def test_signal_during_a_build_is_not_dropped():
    """R4 (skeptic 13): a change signalled while a build runs triggers one more build after it."""
    eng, llm = DescEngine(), GateLLM([("epidemic forecasting", [1]), ("marine and soil ecology", [2])], [GOOD, GOOD])
    d = describer(eng)
    d._llm = llm
    assert d.request_refresh() is True
    first = d._thread
    eng.rows_n = 31
    assert d.request_refresh() is False and d._dirty is True
    llm.gate.set()
    first.join(5)
    if d._thread is not first:
        d._thread.join(5)
    assert llm.steps().count("draft") == 2


def test_table_signature_is_cached_for_the_ttl():
    """R4 (p3m3 10): page views within table_cache_s reuse the table read."""
    eng = DescEngine()
    clock = [100.0]
    d = cd.CorpusDescriber({**CFG, "table_cache_s": 60}, engine_factory=lambda url, **kw: eng, env={"X": "postgres://stand-in"},
                           clock=lambda: clock[0])
    d.store_cfg = {"enabled": False}
    d.tree_reader = TREE_READER
    d._table_cached(eng)
    n = len(eng.sql)
    d._table_cached(eng)
    assert len(eng.sql) == n
    clock[0] += 61
    d._table_cached(eng)
    assert len(eng.sql) > n


def test_registry_reads_go_through_micro_topic_and_the_registry_module():
    """R4 (orchestrator, T5 guard tests): the digest's default tree reader is vera.micro_topic.registry_active_tree, which uses
    only the registry module's load_active_registry_data (owner-checked READ ONLY, ACTIVE rows)."""
    from vera import micro_topic

    def row(k, v, state, parent=None, label=None):
        return {"topic_key": k, "version": v, "label": label or f"Label {k}", "field": "test", "anchor_text": "anchor words for " + k * 3,
                "parent_openalex_topic": "T1", "parent_topic_key": parent, "origin": "seed", "state": state}
    rows = [row("aa", 1, "active"), row("aa", 2, "draft", label="Label new"), row("bb", 1, "retired", "aa"), row("cc", 1, "active", "aa")]
    tree = micro_topic.registry_active_tree({}, reader=lambda: (True, rows))
    assert tree == {"aa": {"label": "Label aa", "state": "active", "parent": None}, "cc": {"label": "Label cc", "state": "active", "parent": "aa"}}
    assert micro_topic.registry_active_keys({}, reader=lambda: (True, rows)) == {"aa", "cc"}
    assert micro_topic.registry_active_tree({}, reader=lambda: (False, [])) == {}
    src = open(cd.__file__, encoding="utf-8").read()
    assert "topic_registry" not in src and "vera_vjay.topreg_topics" not in src and "topic_judge" not in src


# ---- R5 (R5K-3; fixed after the convergence bound, NOT re-reviewed) ---------------------------------------------------------
def test_table_mode_needs_minimum_backfill_coverage():
    """R5K-3: below table_min_coverage no topics are published (partial backfill); a partly tagged document is not counted."""
    eng = FakeEngine()
    eng.cov = (100, 40)  # 40% of the latest-version full-document parts are terminal
    tab = cd.topics_from_table(eng, CFG, TREE_READER)
    assert tab["topics"] == [] and tab["gated"] == "coverage_below_min" and tab["coverage_share"] == 0.4
    eng.cov = (0, 0)
    assert cd.topics_from_table(eng, CFG, TREE_READER)["gated"] == "coverage_below_min"  # nothing tagged = no coverage
    eng.cov = (100, 96)
    eng.open_docs = (13, 14, 15, 16)  # the ecology documents still have open parts
    tab = cd.topics_from_table(eng, CFG, TREE_READER)
    by = {t["key"]: t for t in tab["topics"]}
    assert tab["gated"] is None and 13 not in tab["primary"] and by["ecology"]["primary_docs"] == 0
    assert by["ecology"]["any_docs"] == 2 and by["epidemics"]["primary_docs"] == 12  # only docs 1-2 still carry ecology
    assert cd.topics_from_table(eng, {**CFG, "table_min_coverage": 0.99}, TREE_READER)["topics"] == []
    assert cd.TABLE_MIN_COVERAGE_DEFAULT == 0.95 and "topreg" not in cd.TABLE_COVERAGE_SQL + cd.TABLE_OPEN_DOCS_SQL


def test_coverage_gate_shows_the_no_topics_line(monkeypatch):
    eng = DescEngine()
    eng.cov = (10, 1)
    llm = table_llm()
    out = describer(eng).current(llm)
    assert out["topics"] == [] and out["coverage"] == cd.NO_TABLE_TOPICS_LINE and llm.calls == []


@pytest.fixture(autouse=True)
def _api_subjects_on(monkeypatch):
    """CORPUS-INTERIM-1108: the API default is off (fail safe); these tests pin the old view, so they switch it on."""
    from vera import corpus_description as _cd
    monkeypatch.setitem(_cd.SUBJECT_DEFAULTS, "enabled", True)
