"""NONASK-1103 WP1: Report page mode boundary, Ask layer order, L-a..L-g fixes, redaction and the Enter-submits form.
Offline only: synthetic bodies, no network, no database. Cause texts from the backend are fixture values, never pinned."""
import os
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.test_item78_sources_sidebar import _tree_text
from vera import ui_wake
from vera.m7 import live_report as lr

PAGE = os.path.join(os.path.dirname(__file__), "..", "pages", "vera_demo.py")
S_URL = "https://sentinel-host-nonask.example.invalid"
S_KEY = "sk-SENTINELKEYnonask123456"
FIX_CAUSE = "Fixture cause sentence for the visitor."
# Dev-only strings (H1 list). Checked structurally: absent in public, present in dev where the fixture produces them.
DEV_ONLY = ("Documents inspected", "Search and acquisition", "Cost and latency", "Tokens used", "Latency (s)",
            "Answer-eligible", "stored:", "reused from the collection:", "Research was run because",
            "Evidence authorized", "Answer generated", "Listed as candidates", "Technical details")


def _spans(n, docs):
    return [{"span_id": f"s{i}", "document_id": f"d{i % docs}"} for i in range(n)]


def _body(kind="reading_list", **over):
    """Fixture: 6 passages from 4 documents selected, no answer (the L-a symptom) unless kind == answer."""
    b = {"result_kind": kind, "answer": "An answer [1]." if kind == "answer" else "", "tokens_used": 321, "cost_usd": 0.001,
         "model": "fixture-model-x",
         "answer_status": "complete" if kind == "answer" else "not_answered",
         "limitations": [{"kind": "k", "text": "Fixture limitation sentence."}],
         "process_notes": [{"kind": "k", "text": "Fixture process note."}],
         "claim_check": {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0, "removed": 0},
         "contract": {"layers": ["L0"], "fallback": None if kind == "answer" else "insufficient_full_document_evidence",
                      "inline_acquisition": {"status": "ok", "n_answer_eligible": 10, "n_persisted": 0, "reused": 2,
                                             "documents": [{"persisted": False, "reason": "below_persist_min"}],
                                             "ms": {"search": 5, "total": 9}},
                      "research": {"user_text": "Fixture research outcome sentence.", "triggers": ["stale"],
                                   "portals": [{"portal": "openalex", "status": "completed"},
                                               {"portal": "europepmc", "status": "completed"}],
                                   "all_portals_completed": False,
                                   "candidates": [{"doi": "10.1/x", "outcome": "rejected", "reason": "R-X"}]}},
         "sources": [], "candidates": [{"n": 1, "title": "Cand one", "url": "https://example.org/c1", "provider": "arxiv"}],
         "examined_full_documents": [{"document_id": "d1", "title": "Inspected doc", "chunks_inspected": 9,
                                      "candidate_spans": 4, "admitted_spans": 2}],
         "admitted_evidence": _spans(6, 4), "citations": []}
    if kind == "answer":
        b["sources"] = [{"n": 1, "title": "Doc One", "url": "https://example.org/one", "provider": "arxiv"}]
        b["citations"] = [{"claim_id": 1, "n": 1, "span_id": "s0", "document_id": "d0", "verdict": "supported",
                           "quote": "short quote", "span": "chars=1-9"}]
    b.update(over)
    return b


def _app(rep, public):
    from vera.m7 import live_report
    live_report.render_live_report(__import__("streamlit"), rep, None, public)


def _render(body, public):
    rep = lr.build_live_report(body, "Q?", 1.5)
    return AppTest.from_function(_app, kwargs={"rep": rep, "public": public}, default_timeout=30).run()


# --- mode boundary ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [(None, True), (True, True), (False, False), ("", True), (0, True)])
def test_is_public_fails_closed(val, expected, monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    assert lr.is_public(val) is expected


def test_none_resolves_process_mode_and_unreadable_mode_is_public(monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "0")
    assert lr.is_public(None) is False
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    assert lr.is_public(None) is True
    monkeypatch.setattr("vera.public_mode.is_public_mode", lambda env=None: (_ for _ in ()).throw(RuntimeError("x")))
    assert lr.is_public(None) is True


@pytest.mark.parametrize("kind", ["answer", "reading_list"])
def test_public_has_no_dev_strings_and_dev_has_them(kind):
    pub, dev = _tree_text(_render(_body(kind), True)), _tree_text(_render(_body(kind), False))
    for t in DEV_ONLY:
        assert t not in pub, t
    for t in ("Technical details", "Search and acquisition", "Cost and latency", "Tokens used"):
        assert t in dev, t


def test_heading_order_public_and_dev():
    at = _render(_body("answer"), True)
    assert [h.value for h in at.header] == ["Answer", "Evidence and limitations"]
    assert [e.label for e in at.expander] == ["Research and verification"]
    at = _render(_body("answer"), False)
    assert [h.value for h in at.header] == ["Answer", "Evidence and limitations"]
    assert [e.label for e in at.expander] == ["Research and verification", "Technical details"]


# --- L-a: no contradiction in the no-answer branch -------------------------------------------------------------

def test_la_public_no_answer_has_no_contradiction():
    text = _tree_text(_render(_body(), True)).replace("\\", "")  # drop markdown escapes
    assert "Fixture process note." in text and "Fixture research outcome sentence." in text
    assert "Fixture limitation sentence." in text and "Cand one" in text
    assert "selected as possible evidence" not in text and "passages selected" not in text.lower()
    assert "Evidence authorized" not in text and "Answer generated" not in text


def test_la_dev_shows_inconsistency_caption():
    text = _tree_text(_render(_body(), False))
    assert "passages were selected but no answer was generated" in text
    assert "Evidence authorized for answer generation: yes." in text and "Answer generated: no." in text


def test_cause_text_is_rendered_when_present_and_absent_degrades():
    b = _body()
    b["contract"]["no_answer_cause"] = {"code": "fixture_code", "text": FIX_CAUSE}
    pub, dev = _tree_text(_render(b, True)), _tree_text(_render(b, False))
    assert FIX_CAUSE in pub and "fixture_code" not in pub  # code is dev-only
    assert FIX_CAUSE in dev and "fixture_code" in dev
    assert lr.build_live_report(_body(), "q", 1.0)["no_answer_cause"] is None
    ans = _body("answer")
    ans["contract"]["no_answer_cause"] = {"code": "c", "text": FIX_CAUSE}
    assert lr.build_live_report(ans, "q", 1.0)["no_answer_cause"] is None
    assert lr.build_live_report(_body(contract={"no_answer_cause": "bad"}), "q", 1.0)["no_answer_cause"] is None


# --- L-b: Documents inspected heading only with rows, as a real table ------------------------------------------

def test_lb_heading_iff_rows_and_table_is_text():
    at = _render(_body(), False)
    text = _tree_text(at)
    assert "(passage counts" in text and "Inspected doc" in text and len(at.table) >= 1
    empty = _body(examined_full_documents=[{"document_id": "d1"}])
    assert "(passage counts" not in _tree_text(_render(empty, False))
    assert "(passage counts" not in _tree_text(_render(_body(examined_full_documents=[]), False))


# --- L-c: services grouped, completeness derived from statuses shown -------------------------------------------

def test_lc_grouping_and_consistent_wording():
    rep = lr.build_live_report(_body(), "q", 1.0)
    assert lr.group_services(rep["research"]["portals"]) == ["scholarly index: 2 completed"]
    text = _tree_text(_render(_body(), False))
    assert "scholarly index: 2 completed. All completed." in text
    assert "Not all completed" not in text and "listed twice" not in text
    assert "The service reported incomplete." in text  # flag false disagrees with the statuses shown


def test_lc_failed_portal_and_portal_counts_optional():
    b = _body()
    b["contract"]["research"]["portals"].append({"portal": "ddg", "status": "failed"})
    b["contract"]["research"]["portals_attempted"], b["contract"]["research"]["portals_completed"] = 3, 2
    text = _tree_text(_render(b, False))
    assert "Not all completed" in text and "Search services completed: 2 of 3." in text
    assert "Search services completed" not in _tree_text(_render(_body(), False))  # counts absent: degrade


# --- L-e / redaction: hostile fixture ---------------------------------------------------------------------------

HOSTILE = ["Gate B rejected the stage 2 passages.", "Scored by fixture-model-x on host sentinel.example.invalid.",
           "stored: 0; reused: 2", f"See {S_URL}/x", "sha " + "a" * 40 + "b" * 24, "openalex returned nothing",
           f"key {S_KEY}"]


def test_hostile_notes_are_dropped_and_secrets_redacted(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", S_KEY)
    b = _body("answer", limitations=HOSTILE + ["Plain limitation."], process_notes=HOSTILE + ["Plain note."])
    b["contract"]["research"]["user_text"] = HOSTILE[0]
    rep = lr.build_live_report(b, "Q?", 1.0)
    assert rep["limitations"] == ["Plain limitation."] and rep["process_notes"] == ["Plain note."]
    assert rep["research_user_text"] == ""
    text = _tree_text(_render(b, True)).replace("\\", "")
    for bad in HOSTILE[:-1] + [S_KEY, "fixture-model-x", "Gate B"]:
        assert bad not in text, bad
    assert "Plain limitation." in text and "Plain note." in text


def test_cause_text_with_secret_is_redacted_at_render():
    b = _body()
    b["contract"]["no_answer_cause"] = {"code": "c", "text": f"Fixture {S_KEY} sentence"}
    rep = lr.build_live_report(b, "q", 1.0)
    out = []

    class St:
        def __getattr__(self, n):
            def f(*a, **k):
                out.extend(str(x) for x in a)
                return self if n in ("expander", "container") else (self, self, self) if n == "columns" else None
            return f
        __enter__ = lambda self: self
        __exit__ = lambda self, *a: False
    lr.render_live_report(St(), rep, lr.redactor("", env={"VERA_API_KEY": S_KEY}), True)
    assert S_KEY not in "\n".join(out)


def test_absent_new_fields_back_compat():
    rep = lr.build_live_report({"result_kind": "answer", "answer": "A"}, "q", 1.0)
    assert rep["limitations"] == [] and rep["process_notes"] == [] and rep["research_user_text"] == ""
    assert rep["status_label"] == "" and rep["research"] is None
    assert not _render({"result_kind": "answer", "answer": "A"}, True).exception


def test_status_label_comes_from_ui_evidence():
    rep = lr.build_live_report(_body("answer", answer_status="partial"), "q", 1.0)
    assert rep["status_label"] == "Partial answer"
    assert "Limited evidence" in _tree_text(_render(_body(), True))


# --- C7: Enter-submits form, no double submit ------------------------------------------------------------------

@pytest.fixture
def page_env(monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_BASE_URL", S_URL)
    monkeypatch.setenv("VERA_API_KEY", S_KEY)
    monkeypatch.setattr("vera.sources_sidebar._corpus", lambda st: None)
    monkeypatch.setattr("vera.db.get_session", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {}))
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return SimpleNamespace(status_code=200, json=lambda: _body("answer"))
    monkeypatch.setattr(requests, "post", post)
    return calls


def _submit(at, q="Does X affect Y?"):
    at.text_input[0].input(q)
    at.button[0].click()
    return at.run()


def test_form_submits_once(page_env):
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    at = _submit(at)
    assert not at.exception and len(page_env) == 1 and page_env[0]["question"] == "Does X affect Y?"
    assert "Answer" in [h.value for h in at.header]


def test_second_submit_while_in_flight_or_queued_is_ignored(page_env):
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.session_state[ui_wake.PENDING_KEY] = {"question": "queued"}  # a run is queued
    at.session_state["vera_report_inflight_since"] = 0.0
    at = _submit(at)  # the queued one is taken by the wake step; the new submit is ignored
    assert len(page_env) == 1 and page_env[0]["question"] == "queued"
    assert "vera_report_inflight_since" not in at.session_state  # cleared after the run
    at = _submit(at)  # idle again: a normal submit proceeds
    assert len(page_env) == 2


def test_inflight_marker_blocks_and_stale_marker_does_not(page_env):
    import time
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.session_state["vera_report_inflight_since"] = time.monotonic()
    at = _submit(at)
    assert page_env == []
    at.session_state["vera_report_inflight_since"] = time.monotonic() - 10_000
    at = _submit(at)
    assert len(page_env) == 1
