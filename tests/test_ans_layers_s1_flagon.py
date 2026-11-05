"""ANS-LAYERS-1101 S1 flag-ON coverage: for each path, the note text is NOT in `answer` and IS in the right list
(limitations or process_notes). Offline: stubbed providers/writer/checker, no DB, no network. Paths that need
fastapi / sqlalchemy / openai-backed rigs skip via importorskip until those dependencies exist."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import answer_contract as ac
from vera import ask_service
from vera.ask_service import AskResult

ROOT = Path(ask_service.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def sep_on(monkeypatch):
    monkeypatch.setattr(ac, "notes_separate", lambda *a, **k: True)


def _lists(r):
    return ac.split_notes(r.notes)


# ---------------------------------------------------------------- grounded path (ask_service ~:891-894, :917-922)
GCFG = {"removed_citation_note": "NOTE-BAD-CITES", "no_sources_text": "NO SOURCES"}
SRC = [{"n": 1, "title": "T", "url": "https://arxiv.org/abs/1", "published": None, "snippet": "abstract"}]
DEC = SimpleNamespace(accepted=True, tokens_used=0, cost_usd=0.0)


def _grounded(monkeypatch, writer_text, cfg):
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(writer_text, 10, 0.0))
    return ask_service._answer_grounded("q", "m", None, SRC, cfg, DEC, {})


def test_grounded_bad_citation_note_stripped_into_limitations(monkeypatch):
    r = _grounded(monkeypatch, "Claim [1]. Bogus [4].", GCFG)
    assert r.answer == "Claim [1]. Bogus ." and "NOTE-BAD-CITES" not in r.answer
    assert _lists(r) == (["NOTE-BAD-CITES"], []) and r.notes[0]["kind"] == "removed_citation"


def test_grounded_claim_check_removed_notes_stripped_into_limitations(monkeypatch):
    cc = {"enabled": True, "removed_note": "NOTE-REMOVED", "responsiveness": {"removed_note": "NOTE-RESP-REMOVED"}}
    from vera import claim_check
    monkeypatch.setattr(claim_check, "check_answer",
                        lambda text, sources, cfg: ("Claim [1].\n\nNOTE-REMOVED\n\nNOTE-RESP-REMOVED", {"outcome": "ok"}))
    r = _grounded(monkeypatch, "Claim [1]. Off [1].", {**GCFG, "claim_check": cc})
    assert r.answer == "Claim [1]." and "NOTE-REMOVED" not in r.answer and "NOTE-RESP-REMOVED" not in r.answer
    lim, proc = _lists(r)
    assert set(lim) == {"NOTE-REMOVED", "NOTE-RESP-REMOVED"} and proc == []


# ---------------------------------------------------------------- exhaustion grace on an answer (:1332)
def test_exhaustion_grace_note_on_answer_is_a_process_note(monkeypatch):
    from vera import provider_exhaustion as px
    from vera.pricing.config import PricingRecord
    pr = PricingRecord(provider="groq", model="m", input=0.0, output=0.0, source_url="u",
                       retrieved_at="2026-01-01", effective_from="2026-01-01")
    px.reset_for_tests()
    px.set_grace_openai("gpt-x", pr)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **k: object())  # third-party constructor only
    ans = AskResult("Verified answer [1].", 100, 0.002, result_kind="answer", accepted=True,
                    claim_check={"outcome": "ok"}, contract={"layers": [], "fallback": None})
    monkeypatch.setattr(ask_service, "_fulldoc", lambda *a, **k: ans)
    try:
        res = ask_service.complete_after_exhaustion("q", None, {"work": True, "decision": DEC})
    finally:
        px.set_grace_openai(None, None)
        px.reset_for_tests()
    note = px.grace_cfg()["notice_openai"].strip()
    assert note and res.result_kind == "answer" and res.contract["exhaustion_grace"]["mode"] == "openai_completed"
    assert res.answer == "Verified answer [1]." and note not in res.answer
    assert _lists(res) == ([], [note]) and res.notes[-1]["kind"] == "research_outcome"


# ---------------------------------------------------------------- fulldoc_answer: removed note, partial statements
fd = pytest.importorskip("tests.test_fulldoc_d8_parts")
fr = pytest.importorskip("tests.test_fulldoc_responsive")


@pytest.fixture(autouse=True)
def _fd_key(monkeypatch):
    monkeypatch.setenv("FD_TEST_KEY", "x")


def test_responsiveness_removed_note_in_limitations_not_answer():
    out = fr.run(fd.reply([fr.Q], [fr.A1 + " " + fr.B1]), fr.Checker({"Beta": "no"}))
    r = out.result
    note = fr.RESP["removed_note"]
    assert "Alpha" in r.answer and "Beta" not in r.answer and note not in r.answer
    lim, proc = _lists(r)
    assert note in lim and note not in proc


def test_per_part_partial_statement_in_limitations_not_answer():
    out = fd.run(fd.reply(fd.TWO, [fd.A1, fd.A2]), {"Gamma": "unsupported"})
    r = out.result
    stmt = fd.PP["partial_statement"]
    assert r.claim_check["answer_completeness"] == "partial" and stmt not in r.answer
    lim, proc = _lists(r)
    assert stmt in lim and stmt not in proc and r.notes[0]["kind"] == "partial"
    assert "do not support an answer to this part" in r.answer  # the per-part unanswered line is answer content


def test_responsiveness_partial_statement_in_limitations_not_answer():
    out = fr.run(fd.reply([fr.Q], [fr.A1]), fr.Checker(default="partial"))
    r = out.result
    stmt = fr.RESP["partial_statement"]
    assert r.claim_check["answer_completeness"] == "partial" and stmt not in r.answer
    assert stmt in _lists(r)[0]


def test_unchecked_note_in_limitations_not_answer():
    from vera import fulldoc_answer as fa
    cc_cfg = {"partial_marker": " (p)", "removed_note": "R"}
    cfg = {"answer": {"unchecked_note": "UNCHECKED-NOTE", "quote_max_chars": 200}}
    rec = [{"id": 0, "sentence": "A [1].", "citations": [1], "verdict": "supported", "part": 1}]
    out = []
    text, _ = fa.checked_only("A [1]. B.", rec, [{"n": 1}], cc_cfg, cfg, {}, notes_out=out)
    assert "UNCHECKED-NOTE" not in text and out == [("unchecked", "UNCHECKED-NOTE")]
    assert ac.split_notes(tuple({"kind": k, "text": t} for k, t in out)) == (["UNCHECKED-NOTE"], [])


# ---------------------------------------------------------------- off-topic safety warning (real _inline site)
def test_no_offtopic_warning_acquisition_in_process_notes(monkeypatch):  # INTEGRATE-1104 S2: renamed (no warning)
    from vera import acq_config, acq_discovered, fulldoc_corpus, inline_acquire, provider_exhaustion
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(inline_acquire, "background_note", lambda acq, o: "BG-NOTE")
    monkeypatch.setattr(inline_acquire, "answer_note", lambda acq, o: "ACQ-NOTE")
    monkeypatch.setattr(acq_config, "override_answer_prefix", lambda acq: "WARN-OFFTOPIC may be wrong")
    monkeypatch.setattr(ask_service, "_inline_notice", lambda acq, o: "NOTICE")
    monkeypatch.setattr(acq_discovered, "use_discovered", lambda acq: False)
    monkeypatch.setattr(provider_exhaustion, "note_work", lambda **k: None)

    class _O:
        corpus, status = object(), "ok"

        def public(self):
            return {"status": "ok"}
    monkeypatch.setattr(inline_acquire, "acquire", lambda q, acq, served=None, **kw: _O())
    monkeypatch.setattr(ask_service, "_fulldoc",
                        lambda *a, **k: AskResult(answer="ANS", tokens_used=1, cost_usd=0.0, result_kind="answer"))
    r = ask_service._inline("q", "m", None, SimpleNamespace(topic_key=""), None, None, object(), in_scope=False)
    # INTEGRATE-1104 / SCOPE-GATE-1104 S2: _inline has no off-topic wording any more (a safe question is never warned
    # for its subject); the acquisition and background notes stay process notes, out of the answer text
    assert r.answer == "ANS" and "offtopic" not in (r.contract or {})
    for t in ("WARN-OFFTOPIC may be wrong", "ACQ-NOTE", "BG-NOTE"):
        assert t not in r.answer
    lim, proc = _lists(r)
    assert lim == [] and proc == ["ACQ-NOTE", "BG-NOTE"]


# ---------------------------------------------------------------- main.py wiring into AskResponse
@pytest.fixture
def contract_cfg(monkeypatch):
    state = {"layers": ["L0", "L2"], "discovery_memory": {"write_on_l0": True}, "memory": {"evidence_recall_enabled": False},
             "texts": json.loads((ROOT / "config" / "discovery_contract.json").read_text())["texts"]}
    monkeypatch.setattr(ac, "load_config", lambda path=ac.CONFIG_PATH: state)
    return state


def _post(monkeypatch, result):
    testclient = pytest.importorskip("fastapi.testclient")
    pytest.importorskip("sqlalchemy")
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (result, "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    r = testclient.TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r.json()


def test_main_wires_limitations_process_notes_and_status(monkeypatch, contract_cfg):
    res = AskResult("Body [1].", 1, 0.0, result_kind="answer", accepted=True,
                    claim_check={"answer_completeness": "partial"},
                    notes=({"kind": "partial", "text": "PARTIAL-X"}, {"kind": "removed", "text": "REMOVED-X"},
                           {"kind": "research_outcome", "text": "PROC-Y"}, {"kind": "background", "text": "BG-Y"}))
    j = _post(monkeypatch, res)
    assert j["answer"] == "Body [1]." and all(t not in j["answer"] for t in ("PARTIAL-X", "REMOVED-X", "PROC-Y", "BG-Y"))
    assert j["limitations"] == ["PARTIAL-X", "REMOVED-X"] and j["process_notes"] == ["PROC-Y", "BG-Y"]
    assert j["answer_status"] == "partial"


def test_main_offtopic_warning_lands_in_limitations_and_status_partial(monkeypatch, contract_cfg):
    res = AskResult("Body [1].", 1, 0.0, result_kind="answer", accepted=False, contract={"offtopic": True},
                    notes=({"kind": "offtopic_warning", "text": "WARN may be wrong"},))
    j = _post(monkeypatch, res)
    assert "WARN" not in j["answer"] and j["limitations"] == ["WARN may be wrong"] and j["process_notes"] == []
    assert j["answer_status"] == "partial"
