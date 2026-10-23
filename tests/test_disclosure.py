"""Item #74 M2: the public disclosure exists, is on every page, has no URL/host, and matches the memory config."""
import ast
import json
import re
from pathlib import Path
from unittest import mock

from vera import disclosure
from vera.memory_gate import CONFIG_PATH

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["streamlit_app.py", "pages/vera_demo.py", "pages/trace_eval.py"]


def _store() -> dict:
    return json.loads(CONFIG_PATH.read_text())["memory"]["store"]


def _contract(tmp_path, layers=("L0", "L2"), evidence=False, note="Discovery notes are submitted, and those that pass wait for owner review."):
    p = tmp_path / "contract.json"
    p.write_text(json.dumps({"layers": list(layers), "memory": {"evidence_recall_enabled": evidence},
                             "texts": {"discovery_note_submitted": note}}))
    return p


def test_text_exists_and_states_core_claims():
    t = disclosure.DISCLOSURE_TEXT.lower()
    for phrase in ("not neutral or complete", "may be wrong", "check the sources", "no profile",
                   "does not save your question"):
        assert phrase in t


def _fd(tmp_path, enabled):
    p = tmp_path / "fd.json"
    p.write_text(json.dumps({"fulldoc": {"enabled": enabled}}))
    return p


def test_disclosure_states_no_abstract_evidence(tmp_path):
    off = disclosure.disclosure_text(fulldoc_path=_fd(tmp_path, False))  # state 1
    on = disclosure.DISCLOSURE_TEXT  # shipped = state 2 (P1.6)
    assert "does not use abstracts or snippets as evidence" in off
    assert "does not use their abstracts or snippets as evidence" in on
    for t in (off, on):
        for bad in ("grounded only on index records and abstracts", "findings that passed",
                    "checked against the abstract they cite", "1,500", "1500"):
            assert bad not in t
    src = (ROOT / "vera" / "disclosure.py").read_text()
    assert "1,500" not in src and "1500" not in src and "abstracts (up to" not in src


def test_memory_sentence_follows_layers(tmp_path):
    off = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)))
    assert "discovery note" not in off.lower()
    on = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0", "L2"), note="EXACT TEXT FROM CONFIG."))
    assert "discovery notes" in on and on.endswith("EXACT TEXT FROM CONFIG.")
    shipped = json.loads((ROOT / "config" / "discovery_contract.json").read_text())["texts"]["discovery_note_submitted"]
    # R-028: the shipped layers are ["L0"], so the shipped text has no memory sentence; with L2 it ends with the config text
    assert shipped not in disclosure.DISCLOSURE_TEXT
    l2 = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0", "L2"), note=shipped))
    assert l2.endswith(shipped)


def test_licence_sentence_a0_exact(tmp_path):
    assert ("For listed records, only the indexing services' metadata licence was checked; the articles' own "
            "licences were not verified with the publisher.") in disclosure.disclosure_text(fulldoc_path=_fd(tmp_path, False))  # state 1
    on = disclosure.DISCLOSURE_TEXT  # shipped = state 2 (P1.6): its own exact licence sentences
    assert ("Licences for documents in its own collection were checked against the licence recorded in each document "
            "and VERA-DAR's licence rules, not with the publisher. For listed records, only the indexing services' "
            "metadata licence was checked; the articles' own licences were not verified with the publisher.") in on


def test_no_l1_wording_when_l1_off(tmp_path):
    t = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0", "L2"))).lower()
    assert "synthes" not in t and "generated answer" not in t and "l1" not in t


def test_unreadable_contract_config_is_l0_text(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    l0 = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)))
    assert disclosure.disclosure_text(contract_path=bad) == l0 == disclosure.disclosure_text(contract_path=tmp_path / "missing.json")


def test_confirm_flag_does_not_change_text_while_evidence_recall_off(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"memory": {"store": {"confirm_required": False}}}))
    assert disclosure.disclosure_text(cfg) == disclosure.disclosure_text() == disclosure.DISCLOSURE_TEXT
    assert disclosure.DISCLOSURE_TEXT_UNCONFIRMED == disclosure.DISCLOSURE_TEXT


def test_evidence_findings_sentence_only_when_recall_on(tmp_path):
    assert "Earlier findings" not in disclosure.disclosure_text(contract_path=_contract(tmp_path, evidence=False))
    assert disclosure.disclosure_text(contract_path=_contract(tmp_path, evidence=True)).endswith(
        "Earlier findings were checked only against the abstract or index record they cite.")


def test_render_calls_caption_with_text():
    st = mock.MagicMock()
    disclosure.render_disclosure(st)
    st.caption.assert_called_once_with(disclosure.disclosure_text())


def test_every_page_calls_render_disclosure_once():
    for rel in PAGES:
        tree = ast.parse((ROOT / rel).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "render_disclosure"]
        imported = any(isinstance(n, ast.ImportFrom) and n.module == "vera.disclosure" for n in ast.walk(tree))
        assert imported and len(calls) == 1, rel


def test_pages_do_not_hardcode_the_text():
    for rel in PAGES:
        assert "AI-generated" not in (ROOT / rel).read_text(), rel


def test_no_url_or_host():
    for t in (disclosure.DISCLOSURE_TEXT, disclosure.DISCLOSURE_TEXT_UNCONFIRMED):
        assert not re.search(r"https?://|www\.|\.(com|org|net|io|ai|app)\b|onrender|localhost|\d+\.\d+\.\d+\.\d+", t, re.I)


# ---- Owner decision 2026-10-19 (Option C): exact text, and the storage claims follow the shipped configuration ----
EXACT_TAIL = ("Your question is sent to third-party AI and search services to generate a response. These services have "
              "their own data-handling and retention policies. VERA-DAR does not save your question or information "
              "identifying you. During this demonstration, VERA-DAR does not save newly discovered documents or research "
              "notes from your questions.")


def _chain(tmp_path, corpus_enabled):
    p = tmp_path / "chain.json"
    p.write_text(json.dumps({"corpus": {"enabled": corpus_enabled}}))
    return p


def test_shipped_config_is_demo_configuration():
    """The shipped switches the demonstration sentence rests on."""
    chain = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
    assert chain["corpus"]["enabled"] is False
    assert json.loads((ROOT / "config" / "discovery_contract.json").read_text())["layers"] == ["L0"]


EXACT_TAIL_P17 = ("Your question is sent to third-party AI and search services to generate a response. These services have "
                  "their own data-handling and retention policies. VERA-DAR does not save your question or information "
                  "identifying you. Its collection also grows automatically: VERA-DAR may add open-access full-text "
                  "documents found while searching for your question, after automated licence and content checks, "
                  "recording each document's source, licence and the time it was added; it does not store your question.")


def test_exact_owner_text_as_shipped():  # WED-ACQ2 P1.7: automatic acquisition ships switched on (owner 2026-10-21)
    assert disclosure.DISCLOSURE_TEXT.endswith(" " + EXACT_TAIL_P17)
    assert "During this demonstration" not in disclosure.DISCLOSURE_TEXT
    assert "A third-party hosted language model may be used" not in disclosure.DISCLOSURE_TEXT  # folded into the exact text


def test_demo_sentence_only_when_corpus_and_l2_off(tmp_path, monkeypatch):
    monkeypatch.setattr(disclosure, "auto_acquire_enabled", lambda: False)  # the pre-P1.7 state keeps its exact text
    off = _chain(tmp_path, False)
    assert disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)), chain_path=off).endswith(EXACT_TAIL)
    for chain, layers in ((_chain(tmp_path, True), ("L0",)), (off, ("L0", "L2")), (tmp_path / "missing.json", ("L0",))):
        t = disclosure.disclosure_text(contract_path=_contract(tmp_path, layers), chain_path=chain)
        assert "During this demonstration" not in t, (chain, layers)
        assert "does not save your question or information identifying you" in t


def test_corpus_enabled_text_does_not_claim_no_new_storage(tmp_path):
    t = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)), chain_path=_chain(tmp_path, True))
    assert "may add documents found for your question to its own collection" in t
    assert "not save newly discovered" not in t


def test_text_does_not_imply_third_parties_retain_nothing():
    t = disclosure.DISCLOSURE_TEXT
    assert "own data-handling and retention policies" in t
    for bad in ("never keeps", "nothing is stored", "no one keeps", "deleted immediately", "not retained"):
        assert bad not in t.lower()


def test_r028_no_save_sentence_still_emitted_and_backed_by_counts_only_logging(monkeypatch):  # WED-ACQ D2
    monkeypatch.setattr(disclosure, "auto_acquire_enabled", lambda: False)  # R-028 text holds while P1.7 is off
    t = disclosure.disclosure_text()
    assert ("During this demonstration, VERA-DAR does not save newly discovered documents or research notes from "
            "your questions.") in t
    assert "VERA-DAR does not save your question or information identifying you." in t
    # the only acquisition log is the counts-only line: no question/title/url/identifier is read by it
    import inspect

    from vera import ask_service
    fn = ast.parse(inspect.getsource(ask_service._log_acquisition_counts)).body[0]
    fn.body = fn.body[1:]  # drop the docstring
    code = ast.unparse(fn)
    for token in ("question", "title", "url", "identifier", "doi", "snippet"):
        assert token not in code


def test_p17_auto_acquire_sentence_replaces_r028_and_keeps_third_party_sentences(tmp_path, monkeypatch):
    """WED-ACQ2 AC-13: flag on -> the R-028 demonstration sentence is replaced; both third-party sentences and the
    no-save sentence stay verbatim; flag off -> exactly the R-028 text."""
    off = _chain(tmp_path, False)
    monkeypatch.setattr(disclosure, "auto_acquire_enabled", lambda: True)
    on = disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)), chain_path=off)
    assert on.endswith(EXACT_TAIL_P17) and "During this demonstration" not in on
    assert disclosure._THIRD_PARTY in on and disclosure._NO_SAVE in on
    monkeypatch.setattr(disclosure, "auto_acquire_enabled", lambda: False)
    assert disclosure.disclosure_text(contract_path=_contract(tmp_path, ("L0",)), chain_path=off).endswith(EXACT_TAIL)


def test_p17_flag_reader_is_the_shipped_config_flag(tmp_path, monkeypatch):
    from vera import acq_config
    assert disclosure.auto_acquire_enabled() is True  # shipped config enables it (owner 2026-10-21)
    monkeypatch.setattr(acq_config, "load_config", lambda *a, **k: None)  # unreadable config = feature off
    assert disclosure.auto_acquire_enabled() is False
