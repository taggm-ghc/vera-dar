"""SUBJECT-FOLLOWUP-1114: per-node title reader, adopted end-to-end wiring test (from the independent re-verifier's scratch test,
reviewed: fakes only), and the warm-up's private engine. No DB, no network, no model."""
import contextlib

import main
from vera import corpus_description, subject, subject_names as sn

CFG = {"enabled": True, "max_titles": 7, "answer_subject": {"enabled": True}, "stored_taxonomy": {"enabled": True},
       "subject_names": {"enabled": True, "max_calls_per_version": 5}}
NODE = {"level": "topic", "id": 7, "label": "Solar energy", "terms": ["solar", "grid", "panel"],
        "parent": "Energy", "children": [], "domain": "Energy"}


class FakeEng:
    disposed = False

    def dispose(self):
        FakeEng.disposed = True


def _run(monkeypatch, title, reply, node_map=None, node_boom=False):
    main._SUBJ_WARM["version"] = None
    seen = {"engines": 0}

    @contextlib.contextmanager
    def fake_warm():
        seen["engines"] += 1
        yield FakeEng()

    def fake_profile(engine, max_titles, excerpt_chars=0):
        seen["max_titles"] = max_titles
        return {}, {"titles": [title]}

    def fake_node_titles(engine, cn, per_node=40):
        seen["cn"] = cn
        if node_boom:
            raise RuntimeError("db")
        return node_map or {}
    monkeypatch.setattr(corpus_description, "stamp_and_profile", fake_profile)
    monkeypatch.setattr(subject, "warm_engine", fake_warm)
    monkeypatch.setattr(sn, "read_nodes", lambda e: [dict(NODE)])
    monkeypatch.setattr(sn, "read_node_titles", fake_node_titles)
    monkeypatch.setattr(main, "json_with_chain", lambda chain: (lambda m, max_tokens=40, timeout_s=None: ({"name": reply}, "f")))
    reg = sn.SubjectNames()
    out = {}

    def starter(*a, **k):
        out.update(reg.warm_up(*a, **k))
    assert main._maybe_warm_subject_names(CFG, lambda: {"display_ok": True, "version": "v1"}, starter=starter) is True
    return reg, out, seen


def test_sample_title_echo_rejected_via_wiring(monkeypatch):
    reg, out, seen = _run(monkeypatch, "Solar power in rural grids", "Solar power in rural grids")
    assert seen["max_titles"] == 7
    assert out["refused"] == 1 and reg.get_name("topic", 7) is None


def test_non_echo_accepted_via_wiring(monkeypatch):
    reg, out, _ = _run(monkeypatch, "Medieval pottery trade routes", "Solar power")
    assert out["accepted"] == 1 and reg.get_name("topic", 7) == "Solar power"


def test_per_node_title_echo_rejected_though_not_in_sample(monkeypatch):
    reg, out, seen = _run(monkeypatch, "Medieval pottery trade routes", "Solar power",
                          node_map={("topic", 7): ["Solar power"]})
    assert out["refused"] == 1 and reg.get_name("topic", 7) is None
    assert seen["cn"] == "chunk_taxo_auto"


def test_other_nodes_titles_do_not_block(monkeypatch):
    reg, out, _ = _run(monkeypatch, "Medieval pottery", "Solar power", node_map={("topic", 8): ["Solar power"]})
    assert out["accepted"] == 1


def test_node_titles_read_failure_fails_closed(monkeypatch):
    reg, out, _ = _run(monkeypatch, "Medieval pottery", "Solar power", node_boom=True)
    assert out["refused"] == 1 and out["accepted"] == 0 and reg.get_name("topic", 7) is None


def test_warmup_never_uses_the_answer_engine(monkeypatch):
    def boom():
        raise AssertionError("answer-path engine used by the warm-up")
    monkeypatch.setattr(subject, "_default_engine", boom)
    _, out, seen = _run(monkeypatch, "Medieval pottery", "Solar power")
    assert out["accepted"] == 1 and seen["engines"] == 3  # nodes, titles, node titles: each a private engine


def test_warm_engine_disposes_and_is_not_the_shared_one(monkeypatch):
    made = []

    def fake_role(role, engine_factory=None):
        made.append(role)
        return FakeEng()
    import vera.db as vdb
    monkeypatch.setattr(vdb, "get_role_engine", fake_role)
    FakeEng.disposed = False
    with subject.warm_engine() as e:
        assert isinstance(e, FakeEng) and not FakeEng.disposed
    assert FakeEng.disposed
    assert made == ["rw"]
    assert subject._ENGINE.get("e") is not e


def test_node_titles_failure_attempted_once_per_warmup(monkeypatch):
    main._SUBJ_WARM["version"] = None
    reads = []

    @contextlib.contextmanager
    def fake_warm():
        yield FakeEng()

    def boom(engine, cn, per_node=40):
        reads.append(1)
        raise RuntimeError("db")
    monkeypatch.setattr(corpus_description, "stamp_and_profile", lambda e, m, x=0: ({}, {"titles": []}))
    monkeypatch.setattr(subject, "warm_engine", fake_warm)
    monkeypatch.setattr(sn, "read_nodes", lambda e: [dict(NODE), dict(NODE, id=8), dict(NODE, id=9)])
    monkeypatch.setattr(sn, "read_node_titles", boom)
    monkeypatch.setattr(main, "json_with_chain", lambda c: (lambda m, max_tokens=40, timeout_s=None: ({"name": "Solar power"}, "f")))
    reg, out = sn.SubjectNames(), {}
    main._maybe_warm_subject_names(CFG, lambda: {"display_ok": True, "version": "v1"},
                                   starter=lambda *a, **k: out.update(reg.warm_up(*a, **k)))
    assert len(reads) == 1 and out["refused"] == 3 and out["accepted"] == 0


def test_node_titles_sql_is_bounded_strict_and_read_only():
    sql = sn.node_titles_sql()
    assert sql.lstrip().upper().startswith("WITH") and ":per" in sql and ":cn" in sql
    assert "score_kind IS NULL OR cs.score_kind = 'cosine'" in sql
    for bad in ("INSERT", "UPDATE ", "DELETE", "DROP", "ALTER"):
        assert bad not in sql.upper()
