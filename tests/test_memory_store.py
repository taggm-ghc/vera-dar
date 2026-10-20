"""Item #84 storage/recall: fake reader/store, no DB. #74: the question and identity are never written."""
import json
from pathlib import Path
from datetime import datetime, timezone

from vera import memory_store as ms
from vera.memory_gate import load_gate_config
from vera.memory_store import MemoryService, memory_from_env

_SHIPPED = json.load(open(ms.__file__.replace("vera/memory_store.py", "config/ask-provider-chain.json")))["memory"]
# R-028 (2026-10-19): the shipped config has memory.store.enabled=false; these tests exercise the store/operator code itself,
# so they run it enabled (only that flag differs from the shipped block; before: BLOCK was the shipped block as-is).
BLOCK = {**_SHIPPED, "store": {**_SHIPPED["store"], "enabled": True}}
CFG = BLOCK["store"]
QUESTION = "What does quantized inference change about transformer latency?"
SRC = [{"n": 1, "url": "https://arxiv.org/abs/2401.00001", "provider": "arxiv", "identifier": "arXiv:2401.00001",
        "licence_decision": "allow", "title": "T"}]
GOOD = {"id": 0, "sentence": "Int8 weight quantization reduced memory use by roughly half in the cited study [1].",
        "citations": [1], "verdict": "supported"}


class FakeReader:
    def __init__(self, found=(7, 100), dup=False):
        self.found, self.dup, self.asked = found, dup, []

    def source(self, u): self.asked.append(u); return self.found
    def span(self, s, e): return 11
    def answer(self): return 5
    def duplicate(self, t, s): return self.dup
    def pending_count(self): return 0  # D6 pending-queue cap
    def recall(self, q): return [{"claim_text": "x", "source_title": "t", "source_url": "u", "saved_at": "d"}]


class FakeStore:
    def __init__(self): self.claims = []
    def span_id(self, s, e): return 11
    def answer_id(self): return 5
    def add_claim(self, a, t, s): self.claims.append((a, t, s)); return 1


def svc(reader=None, store=None, **over):
    return MemoryService(dict(CFG, **over), load_gate_config(), reader or FakeReader(), store or FakeStore())


def test_writes_stripped_claim_only():
    st = FakeStore(); n, sk = svc(store=st).write_claims([GOOD], SRC, QUESTION)
    assert n == 1 and sk == {}
    assert st.claims == [(5, "Int8 weight quantization reduced memory use by roughly half in the cited study.", [11])]


def test_source_not_in_corpus_skipped():
    st = FakeStore(); n, sk = svc(FakeReader(found=None), st).write_claims([GOOD], SRC, QUESTION)
    assert n == 0 and sk == {ms.SKIP_NOT_IN_CORPUS: 1} and not st.claims


def test_duplicate_and_unsupported_skipped():
    assert svc(FakeReader(dup=True)).write_claims([GOOD], SRC, QUESTION)[1] == {ms.SKIP_DUPLICATE: 1}
    bad = dict(GOOD, verdict="partial")
    assert svc().write_claims([bad], SRC, QUESTION)[0] == 0


def test_caps():
    st = FakeStore()
    n, sk = svc(store=st, max_writes_per_answer=1).write_claims([GOOD, GOOD], SRC, QUESTION)
    assert n == 1 and sk == {ms.SKIP_CAP: 1}
    s = svc(max_writes_per_day=1)
    assert s.write_claims([GOOD], SRC, QUESTION)[0] == 1 and s.write_claims([GOOD], SRC, QUESTION)[0] == 0


def test_errors_never_raise():
    class Boom(FakeReader):
        def source(self, u): raise RuntimeError("db down")
        def recall(self, q): raise RuntimeError("db down")
    s = svc(Boom())
    assert s.write_claims([GOOD], SRC, QUESTION)[1] == {"error": "RuntimeError"}
    assert s.recall(QUESTION) == []


def test_no_question_or_identity_stored():
    """#74: every value bound by the write path is a claim, fixed text, ids; never the question."""
    bound = []

    class Conn:
        def execute(self, stmt, params): bound.append(dict(params)); return self
        def first(self): return (1,)
        def begin_nested(self):
            class SP:
                commit = rollback = lambda self: None
            return SP()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Eng:
        def begin(self): return Conn()
    store = ms.WoMemoryStore(Eng(), CFG)
    store.answer_id(); store.span_id(7, 100); store.add_claim(5, "A finding.", [11])
    flat = json.dumps(bound, default=str)
    assert "quantized" not in flat and "latency" not in flat
    assert not {"ip", "client", "session", "user", "question"} & {k for p in bound for k in p}


def test_recall_query_terms_and_shape():
    r = ms.RoMemoryReader(object(), CFG)
    q = r.query_terms(QUESTION)
    assert " or " in q and len(q.split(" or ")) <= CFG["max_query_terms"]
    rows = []
    r._rows = lambda sql, p: rows.append((sql, p)) or [("c", datetime(2026, 10, 1, tzinfo=timezone.utc), "T", "u")]
    out = r.recall(QUESTION)
    assert out == [{"claim_text": "c", "source_title": "T", "source_url": "u", "saved_at": "2026-10-01T00:00:00+00:00"}]
    assert QUESTION not in json.dumps(rows[0][1], default=str)  # only the term query, never the raw question
    assert r.recall("a an") == []


def test_from_env_disabled_or_unset():
    assert memory_from_env(None) is None
    assert memory_from_env(BLOCK, env={}) is None
    assert memory_from_env(dict(BLOCK, store=dict(CFG, enabled=False)), env={"VERA_DB_URL_RO": "x"}) is None


def test_decay_is_true_half_life_and_rank_floor():
    r = ms.RoMemoryReader(object(), CFG); seen = []
    r._rows = lambda sql, p: seen.append((sql, p)) or []
    r.recall(QUESTION)
    sql, p = seen[0]
    assert "power(0.5" in sql and "exp(" not in sql and p["hl"] == CFG["half_life_days"] == 14
    assert p["floor"] == CFG["min_rank"] > 0


def test_one_span_per_cited_source_and_daily_cap_atomic():
    src2 = [dict(SRC[0]), dict(SRC[0], url="https://arxiv.org/abs/2401.00002", identifier="arXiv:2401.00002")]
    two = dict(GOOD, citations=[1, 2]); st = FakeStore()
    sp = iter([11, 12]); rd = FakeReader(); rd.span = lambda s, e: next(sp)
    svc(rd, st).write_claims([two], src2, QUESTION)
    assert st.claims[0][2] == [11, 12]
    s = svc(max_writes_per_day=1)
    assert [s._reserve(), s._reserve()] == [True, False]


def test_engines_get_timeouts_from_config():
    seen = []
    ms._SERVICE.clear()
    memory_from_env(BLOCK, env={"VERA_DB_URL_RO": "x", "VERA_DB_URL_WO": "y"},
                    engine_factory=lambda url, **kw: seen.append(kw) or object())
    ms._SERVICE.clear()
    assert len(seen) == 2 and all(k["connect_args"]["connect_timeout"] == CFG["connect_timeout_s"]
                                  and str(CFG["statement_timeout_ms"]) in k["connect_args"]["options"] for k in seen)


def test_recall_requires_min_matched_terms_from_config():
    assert CFG["min_matched_terms"] >= 2
    r = ms.RoMemoryReader(object(), CFG); seen = []
    r._rows = lambda sql, p: seen.append((sql, p)) or []
    r.recall(QUESTION)
    sql, p = seen[0]
    assert "unnest(CAST(:terms AS text[]))" in sql and "plainto_tsquery" in sql
    assert p["minterms"] == CFG["min_matched_terms"]
    assert p["terms"] == r.query_terms(QUESTION).split(" or ") and len(p["terms"]) >= p["minterms"]


# Candidate-only rewording (ON-1020-P1): the original constant reused a held-out evaluation question.
Q_LONG = ("Which metrics, such as review latency, merged commit counts or developer interviews, "
          "have been used to evaluate automated refactoring tools, and how do they change findings?")


def test_query_terms_keep_content_words():
    cfg = json.loads((Path(ms.__file__).parent.parent / "config" / "ask-provider-chain.json").read_text())["memory"]["store"]
    terms = ms.RoMemoryReader(None, cfg).query_terms(Q_LONG).split(" or ")
    for w in ("metrics", "review", "latency", "merged", "commit", "counts", "developer", "interviews", "used",
              "evaluate", "automated", "refactoring", "tools", "change", "findings"):
        assert w in terms, w
    for w in ("which", "such", "have", "been", "how", "they"):
        assert w not in terms
    assert len(terms) <= int(cfg["max_query_terms"])
