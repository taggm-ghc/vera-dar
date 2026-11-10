"""ARCH-FIX-1110 E (ARCHETYPES-DIAG-1110): hybrid ON (fake embedder, tests/hybrid_on_support.py; no network, no DB). A role
taxonomy that says "roles / patterns" instead of "archetypes" is pulled into the selected passages by the stage-2 rerank
reserved slots (rerank.reserve_slots); with rerank off, or with the reserved slots at 0 on a tight candidate list, it is not. The answer writer's refusal
rule is untouched: this test covers passage supply only."""
import pytest  # noqa: F401

from tests.hybrid_on_support import _hybrid_still_on, hybrid_on  # noqa: F401  (module-scoped autouse: hybrid ON)
from tests import test_hybrid_retrieval_1109 as base
from vera import fulldoc_admission as fa, hybrid_retrieval as hr
from vera.fulldoc_corpus import load_config

# no "archetype", "category", "type" or "kind" anywhere: only roles / patterns vocabulary
ROLES = ["Below are common roles and patterns for agents in an orchestrated system. ",
         "The worker role executes narrow tasks such as retrieval; the supervisor role assigns work to workers. ",
         "The critic role reviews outputs and requests revisions; the router role selects which agent takes a turn. "]


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(base, "ROLE_B", ROLES)
    # in the fake embedder, "role" and "pattern" share the concept the question's "archetype" has (as a real embedder would)
    monkeypatch.setitem(base.SYN, "role", "KIND")
    monkeypatch.setitem(base.SYN, "pattern", "KIND")
    c, _ = base._corpus(spans_b=[(0, len(base._texts()[1]))])  # doc B is ONE partition: stage 1 cannot separate roles from intro
    return c


def _roles_selected(res, c):
    b0 = c._texts["DB"].index("Below are")
    return any(a["document_id"] == "DB" and a["end"] > b0 for a in res.admitted)


def _cfg(**rr):
    hr.set_config_for_tests({"enabled": True, "chunks_per_partition": 1,
                             "rerank": dict({"enabled": True, "top_partitions": 3, "max_passages": 48}, **rr)})


def test_roles_vocabulary_passage_enters_via_rerank_reserved_slots(corpus, hybrid_on):  # noqa: F811
    q = base.Q
    assert "role" not in q.lower() and "pattern" not in q.lower()
    _cfg(enabled=False)
    off = fa.admit(q, corpus, load_config())
    assert not _roles_selected(off, corpus)
    # a tight candidate list (candidate_k 2): only the reserved stage-2 slots can put the roles passage into it
    cfg = load_config()
    cfg["retrieval"] = dict(cfg.get("retrieval") or {}, candidate_k=2)
    _cfg(reserve_slots=0)
    assert not _roles_selected(fa.admit(q, corpus, cfg), corpus)
    _cfg(reserve_slots=2)
    tight = fa.admit(q, corpus, cfg)
    assert tight.trace["hybrid"]["rerank"]["status"] == "ok" and _roles_selected(tight, corpus), tight.trace
    assert hr.config()["rerank"]["reserve_slots"] == 2
