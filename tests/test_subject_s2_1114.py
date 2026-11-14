"""SUBJECT-BUILD-1114 WP-S2: assoc_description / assoc_basis on stored-corpus sources (T14-T17). Fakes only: no DB, no network, no model."""
import json

import main
from vera import subject, subject_names

PRE_CHANGE_KEYS = [  # AskSource JSON keys before WP-S2 (recorded shape); FRESH keys are dropped when None
    "n", "title", "url", "published", "source_type", "provider", "identifier", "retrieved_at", "licence_decision",
    "reference", "in_text", "characterisation", "span", "canonical_source_id", "source_id", "admission_passed",
    "acquisition", "span_id", "disposition", "licence"]
PRE_CHANGE_JSON = ('{"n":1,"title":"t","url":"u","published":null,"source_type":null,"provider":null,"identifier":null,'
                   '"retrieved_at":null,"licence_decision":null,"reference":null,"in_text":null,"characterisation":null,'
                   '"span":null,"canonical_source_id":null,"source_id":null,"admission_passed":null,"acquisition":null,'
                   '"span_id":null,"disposition":null,"licence":null}')

CFG_ON = {"enabled": True, "answer_subject": {"enabled": True}, "stored_taxonomy": {"enabled": True, "classifier_name": "c"}}
CFG_OFF = {"enabled": True, "answer_subject": {"enabled": False}, "stored_taxonomy": {"enabled": True}}
STATUS = {"display_ok": True, "version": "v1"}


def _cit(sid, doc=7, start=0, end=100):
    return {"span_id": f"{doc}:{start}-{end}", "start": start, "end": end, "document_id": doc, "canonical_source_id": doc,
            "corpus_chunks": [{"start": 0, "end": 200, "span_id": sid}]}


class _Res:
    def __init__(self, rows):
        self._r = rows

    def scalar(self):
        return False  # probe: tax_chunk_domain absent

    def all(self):
        return self._r


class FakeEngine:
    def __init__(self, rows=(), raises=False):
        self.rows, self.raises, self.sql = list(rows), raises, []

    def begin(self):
        eng = self

        class Cx:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def execute(s, stmt, params=None):
                q = str(stmt)
                eng.sql.append(q)
                if q.startswith("SET LOCAL") or "to_regclass" in q:
                    return _Res([])
                if eng.raises:
                    raise RuntimeError("boom")
                return _Res(eng.rows)
        return Cx()

    def selects(self):
        return [q for q in self.sql if "tax_chunk_subtopic" in q]


ROWS = [("cs", 11, 5, None, 0.9, "solar, grid"), ("ts", None, 5, 3, None, None), ("dt", None, 3, 1, None, None),
        ("tn", None, 3, None, None, "energy, power"), ("dn", None, 1, None, None, "engineering, tech")]


def _setup(monkeypatch, eng):
    from vera import corpus_taxonomy
    monkeypatch.setattr(corpus_taxonomy, "job_status", lambda: STATUS)
    monkeypatch.setattr(subject, "_default_engine", lambda: eng)
    subject._PROBE_CACHE.update(version=object(), has=False)


def test_t14_flag_off_shape_and_no_calls():
    s = main.AskSource(n=1, title="t", url="u")
    assert list(json.loads(s.model_dump_json()).keys()) == PRE_CHANGE_KEYS
    assert s.model_dump_json() == PRE_CHANGE_JSON
    rows = [{"n": 1, "span": "chars=0-9"}]
    called = []
    main._attach_subjects(rows, [_cit(11)], cfg=CFG_OFF, subjects_for=lambda *a, **k: called.append(1))
    assert called == [] and rows == [{"n": 1, "span": "chars=0-9"}]
    assert main._subject_flag_on({}) is False and main._subject_flag_on(None) is False
    cfgfile = json.loads(open("config/ask-provider-chain.json").read())["corpus_description"]
    assert cfgfile["answer_subject"]["enabled"] is False and "subject_names" in cfgfile


def test_t14_fields_dropped_when_none_and_serialised_when_set():
    d = json.loads(main.AskSource(n=1, title="t", url="u", assoc_basis="topic",
                                  assoc_description={"domain": "d", "topic": "t", "subtopic": None}).model_dump_json())
    assert d["assoc_basis"] == "topic" and d["assoc_description"]["topic"] == "t"


def test_t15_only_stored_carry_fields(monkeypatch):
    eng = FakeEngine(ROWS)
    _setup(monkeypatch, eng)
    rows = [{"n": 1, "span": "chars=0-100"}, {"n": 2, "span": None}]
    main._attach_subjects(rows, [_cit(11), None], cfg=CFG_ON)
    assert rows[0]["assoc_basis"] == "subtopic"
    assert rows[0]["assoc_description"] == {"domain": "engineering, tech", "topic": "energy, power", "subtopic": "solar, grid"}
    assert "assoc_basis" not in rows[1] and "assoc_description" not in rows[1]


def test_t15_cached_name_is_shown_with_terms(monkeypatch):
    _setup(monkeypatch, FakeEngine(ROWS))
    monkeypatch.setattr(subject_names, "get_display", lambda lv, i: "solar power (terms: solar, grid)" if lv == "subtopic" else None)
    rows = [{"n": 1}]
    main._attach_subjects(rows, [_cit(11)], cfg=CFG_ON)
    assert rows[0]["assoc_description"]["subtopic"] == "solar power (terms: solar, grid)"
    assert rows[0]["assoc_description"]["topic"] == "energy, power"


def test_t15_no_record_and_closed_gate(monkeypatch):
    _setup(monkeypatch, FakeEngine([]))
    rows = [{"n": 1}]
    main._attach_subjects(rows, [_cit(11)], cfg=CFG_ON)
    assert rows[0] == {"n": 1, "assoc_basis": "none", "assoc_description": {"domain": None, "topic": None, "subtopic": None}}
    eng = FakeEngine(ROWS)
    _setup(monkeypatch, eng)
    monkeypatch.setattr(main, "_subject_flag_on", lambda cfg=None: True)
    rows = [{"n": 1}]
    main._attach_subjects(rows, [_cit(11)], cfg={**CFG_ON, "stored_taxonomy": {"enabled": False}})
    assert rows == [{"n": 1}] and eng.sql == []  # display_gate_closed: fields absent, no DB


def test_t15_citation_selection_fulldoc_and_lane():
    class Ev:
        admitted = ({"span_id": "7:0-9", "start": 0, "end": 9, "corpus_chunks": []},
                    {"span_id": "8:0-9", "lane": "x", "answer_only": True})

    class R:
        evidence = Ev()
    out = main._subject_citations(R(), [{}, {}], True)
    assert out[0] is Ev.admitted[0] and out[1] is None
    assert main._subject_citations(R(), [{}], True) == [None]  # misaligned: nothing attached
    assert main._subject_citations(R(), [{"span": "chars=0-1"}, {"span": None}], False)[1] is None


def test_t16_raising_or_unavailable_leaves_rows_unchanged(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    rows = [{"n": 1}]
    main._attach_subjects(rows, [_cit(11)], cfg=CFG_ON, subjects_for=boom)
    assert rows == [{"n": 1}]
    eng = FakeEngine(ROWS, raises=True)  # SELECT raises: subject_for returns unavailable
    _setup(monkeypatch, eng)
    rows = [{"n": 1}]
    main._attach_subjects(rows, [_cit(11)], cfg=CFG_ON)
    assert rows == [{"n": 1}]


def test_t17_query_counts(monkeypatch):
    eng = FakeEngine(ROWS)
    _setup(monkeypatch, eng)
    # two passages of one document resolving to the same span ids share one call
    rows = [{"n": 1}, {"n": 2}]
    main._attach_subjects(rows, [_cit(11, start=0, end=50), _cit(11, start=60, end=90)], cfg=CFG_ON)
    assert len(eng.selects()) == 1
    assert sum("to_regclass" in q for q in eng.sql) == 1  # cached probe, once
    assert all("tax_chunk_domain" not in q for q in eng.selects())  # table absent: statement without the domain join
    # defect 1: distinct passages and documents still cost exactly ONE SELECT per answer
    eng2 = FakeEngine(ROWS + [("cs", 12, 5, None, 0.8, "solar, grid"), ("ct", 13, 3, None, 0.7, "energy, power")])
    _setup(monkeypatch, eng2)
    rows = [{"n": 1}, {"n": 2}, {"n": 3}]
    main._attach_subjects(rows, [_cit(11), _cit(12, doc=8), _cit(13, doc=9)], cfg=CFG_ON)
    assert len(eng2.selects()) == 1 and sum("to_regclass" in q for q in eng2.sql) == 1
    assert [r["assoc_basis"] for r in rows] == ["subtopic", "subtopic", "topic"]  # each row composed from its own spans
    # flag-on answer with no stored citation: no call at all
    eng3 = FakeEngine(ROWS)
    _setup(monkeypatch, eng3)
    main._attach_subjects([{"n": 1}], [None], cfg=CFG_ON)
    assert eng3.sql == []
