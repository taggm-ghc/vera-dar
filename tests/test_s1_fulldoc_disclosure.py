"""S1 / DISC-E2: three-state disclosure wording chosen by the runtime fulldoc flag (exact strings pinned)."""
import json
from pathlib import Path

from vera import disclosure, sources_sidebar as sb, ui_evidence as ue

ROOT = Path(__file__).resolve().parent.parent
PASSAGES_SENTENCE = ("When it answers from its own collection, your question, the selected passages with their titles and "
                     "links, and the draft answer sentences are sent to third-party AI services.")
STATE1_DISCLOSURE_START = "VERA-DAR is an AI system. For each in-scope question it searches published scholarly sources and the web and "
STATE23_DISCLOSURE_START = ("VERA-DAR is an AI system. For each in-scope question it first searches a small collection of full "
                            "documents that its operator selected and licence-checked in advance. If passages in them address the "
                            "question, it writes a short answer from up to six passages, cites them with brief quotes, and has an AI "
                            "model check each cited sentence against its passage; unsupported sentences are removed.")
HELP_LIST = ("Ask a question. VERA-DAR lists sources that may be relevant, with links. It has not read them and does not "
             "write an answer.")
HELP_FULLDOC = ("Ask a question. VERA-DAR answers only from the full documents it holds, with cited passages, or says why it "
                "cannot; otherwise it lists sources it has not read.")
HOW_FULLDOC = ("**How VERA-DAR answers:** for each in-scope question the Ask page first searches a small collection of full "
               "documents that the operator selected and licence-checked in advance. If passages in them address the question, "
               "it writes a short answer from up to six passages, cites them with brief quotes, and has an AI model check each "
               "cited sentence; unsupported sentences are removed. Otherwise it searches published scholarly sources and the "
               "web and lists documents it has not read. Open the sources and read them yourself before relying on any claim.")
NOT_READ_NEW = "VERA-DAR has not acquired or read the documents in this list. This is a discovery result, not an evidence-grounded answer."
INSUFFICIENT_NEW = ("VERA-DAR searched {n_inspected} full document(s) it holds but found no passage that answers this "
                    "question well enough, so it has not written an answer or answered from memory.")
BAD = "has not read the full documents"


def _cfg(tmp_path, enabled):
    p = tmp_path / "fd.json"
    p.write_text(json.dumps({"fulldoc": {"enabled": enabled}} if enabled is not None else {}))
    return p


def test_flag_parsing_fails_closed_for_text_selection(tmp_path):
    assert disclosure.fulldoc_enabled(_cfg(tmp_path, False)) is False
    assert disclosure.fulldoc_enabled(_cfg(tmp_path, None)) is False  # key absent = path disabled
    for v in (True, "true", "1", "on"):
        assert disclosure.fulldoc_enabled(_cfg(tmp_path, v)) is True
    assert disclosure.fulldoc_enabled(tmp_path / "missing.json") is True  # unreadable -> over-disclose
    assert disclosure.fulldoc_enabled(ROOT / "config" / "fulldoc_answer.json") is False  # shipped = off (not flipped)


def test_state1_off_keeps_todays_text(tmp_path):
    t = disclosure.disclosure_text(fulldoc_path=_cfg(tmp_path, False))
    assert t.startswith(STATE1_DISCLOSURE_START) and t == disclosure.DISCLOSURE_TEXT
    assert "has not acquired or read these full documents, so it does not use abstracts or snippets as evidence" in t
    assert PASSAGES_SENTENCE not in t and "passages" not in t
    assert disclosure._THIRD_PARTY.strip() in t  # question-only transmission is true only while off


def test_state2_and_3_disclosure_names_passage_transmission_and_never_claims_unread_collection(tmp_path):
    on = disclosure.disclosure_text(fulldoc_path=_cfg(tmp_path, True))
    assert on.startswith(STATE23_DISCLOSURE_START)
    assert PASSAGES_SENTENCE in on  # whenever fulldoc is on
    assert disclosure._THIRD_PARTY.strip() in on and on.endswith(disclosure._NO_SAVE.strip() + disclosure._DEMO_NO_STORAGE)
    assert BAD not in on and "has not acquired or read these full documents" not in on
    assert "It has not read those listed documents" in on  # scoped to the list only
    # unreadable fulldoc config over-discloses
    assert PASSAGES_SENTENCE in disclosure.disclosure_text(fulldoc_path=tmp_path / "missing.json")


def test_sidebar_text_follows_runtime_state(monkeypatch):
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: False)
    assert sb.how_answers_text() == sb._HOW_LIST and "has not read the full documents" in sb.how_answers_text()
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: True)
    t = sb.how_answers_text()
    assert t == HOW_FULLDOC and BAD not in t and "up to six passages" in t
    assert "lists documents it has not read" in t


def test_help_text_follows_runtime_state(monkeypatch):
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: False)
    assert disclosure.ask_help_text() == HELP_LIST
    monkeypatch.setattr(disclosure, "fulldoc_enabled", lambda *a, **k: True)
    h = disclosure.ask_help_text()
    assert h == HELP_FULLDOC and BAD not in h
    assert "answers from the sources it holds" not in HELP_LIST + HELP_FULLDOC


def test_state2_claim_check_caption_exact():
    cc = {"outcome": "checked", "checked": 3, "supported": 3, "partial": 0, "removed": 0}
    t = ue.claim_check_text(cc, fulldoc=True)
    assert t.startswith("Claim check: 3 cited claims checked by a model against the cited passages "
                        "(full span text; excerpts shown may be abridged); 3 supported")
    assert BAD not in t and "abstract" not in t
    assert "abstract text" in ue.claim_check_text(cc, fulldoc=False)  # legacy branch unchanged


def test_state3_fallback_and_not_read_lines_exact():
    cfg = json.loads((ROOT / "config" / "fulldoc_answer.json").read_text())
    assert cfg["answer"]["insufficient_text"] == INSUFFICIENT_NEW
    line = ue.fallback_text("insufficient_full_document_evidence", n_inspected=2)
    assert line.startswith("VERA-DAR searched 2 full document(s)") and BAD not in line
    assert "inspected" not in line and "{" not in line
    nr = json.loads((ROOT / "config" / "discovery_contract.json").read_text())["texts"]["not_read_line"]
    assert nr == NOT_READ_NEW and "these full documents" not in nr  # scoped to the listed web results


def test_docs_carry_the_passage_transmission_statement():
    api = (ROOT / "docs" / "api.md").read_text()
    assert "When the optional full-document path is enabled, up to six selected passages (text, title, link) and the draft answer sentences are transmitted too." in api
    rd = (ROOT / "README.md").read_text()
    assert "the question, the selected passages with their titles and links, and draft answer sentences are sent to third-party AI services" in rd
